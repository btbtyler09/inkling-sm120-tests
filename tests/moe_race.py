#!/usr/bin/env python3
"""moe_race.py -- decode-size MoE race on one sm_120 GPU: Marlin vs b12x vs FlashInfer W4A4.

Inkling-2 routed experts (384 experts, top-8, H=4608, full I=2304, NVFP4
weights, SiTU gate cap 4 / up cap 25) at the rank-0 shard of an
intermediate-split TP MoE (every rank holds all 384 experts):
  I=384 (TP6), I=576 (TP4);  M = 8, 16, 32, 64 tokens.

All backends get the SAME random checkpoint-format NVFP4 codes / e4m3 block
scales / per-expert amax (the checkpoint's NVFP4 tensor format, generated
directly at the rank-0 slice shape).

  marlin      : Inkling2MarlinExperts (W4A16 Marlin).
  fi_w4a4     : Inkling2MarlinExperts with _W4A4=True, _W4A4_MIN=0 (FlashInfer
                CUTLASS fused MoE, NVFP4 activations, per-batch dynamic global
                scale + four-over-six block scales). Built as a t=1 module on the
                slice tensors (I_local as the whole intermediate): the module
                refuses W4A4 together with INKLING2_MOE_TP>1; the math is identical.
  b12x_<mode> : b12x fused_moe W4A16 (ActivationMode.A16).
  b12x_a4_<m> : b12x fused_moe W4A4 (ActivationMode.A4, ModelOpt NVFP4) with
                STATIC per-layer activation scales calibrated on this case:
                a1 = 2688 / amax|x|, a2 = 2688 / amax|SiTU(FC1)| (reciprocal
                ModelOpt input_scale; see B12X_A4_NOTE). Prepared per (I, M).
  <mode>: heur = PreparationSession(autotune=False), tuned = autotune=True.

Two timings per (backend, I, M): "experts" (MoE only, fixed routings) and
"layer" (router GEMM + bias + top-8 via moe_marlin.route_torch, then the
experts). b12x 0.7.3's route_topk only knows softmax / sqrtsoftplus, not
Inkling's sigmoid + correction-bias router, so b12x also uses route_torch.
Timing: ITERS calls captured in one CUDA graph (cycling through --routings
distinct inputs), REPS replays, median replay / ITERS = us per call.
Usage: moe_race.py [--dry-run] [--out res.json] [--I 384,576] [--M 8,16,32,64]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
import types

import torch

E, TOPK, H, I_FULL = 384, 8, 4608, 2304
GATE_CAP, UP_CAP = 4.0, 25.0
DEN = 2688.0  # 6 * 448
ROUTE_SCALE = 1.0
PEAK_BPS = 1.79e12  # RTX PRO 6000 Blackwell DRAM peak
TP_OF_I = {384: 6, 576: 4}  # rank-local intermediate slice -> TP degree (world = t, rank 0)
FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
B12X_CALLS = (
    "fused_moe.plan_weights(source=PackedSource(format=MODELOPT_NVFP4, w13_layout=W13), "
    "activation=ActivationSpec(mode=A16|A4, nonlinearity='situ', io_dtype=bf16), "
    "geometry=MoEGeometry(num_experts=384, hidden_size=4608, intermediate_size=I))",
    "fused_moe.prepare_weights(plan=, weights=PackedWeights(w13=[up;gate] u8, w2 u8, "
    "w13/w2_block_scales=b12x._lib.intrinsics.swizzle_block_scale(e4m3), "
    "w13/w2_global_scales=amax/2688 f32[E]; A4 only: input_scale=2688/amax|x|, "
    "intermediate_scale=2688/amax|SiTU|, immutable_input_scales=True))",
    "fused_moe.plan_execution(experts=, capacity=ExecutionCapacity(max_tokens=M, top_k=8))",
    "plan.request(name=, prepare_call=f, benchmark_call=f); f(state) -> "
    "PreparedCall(run=state.run(state.bind(scratch=, a=, experts=, topk_weights=, topk_ids=, output=)))",
    "b12x.preparation.PreparationSession(autotune=heur:False|tuned:True).prepare((request,))",
    "fused_moe.bind(plan, scratch=empty(plan.scratch_specs()), a=x, experts=, topk_weights=w, "
    "topk_ids=ids, output=out); fused_moe.run(binding=) under session.capture() + torch.cuda.graph",
)
B12X_A4_NOTE = (
    "b12x 0.7.3 A4: PackedWeights.input_scale/intermediate_scale become a1_gscale/a2_gscale, documented as "
    "the RECIPROCAL activation global scale (_impl.py:442; planning.py:789-858); the FC1 quantizer multiplies "
    "x by it (_shared/kernels/dynamic.py:4216) and preparation folds alpha = w_global / a_gscale "
    "(_impl.py:6630-6664, 'vLLM exposes reciprocal input scales'). So a_gscale = 1/input_scale_modelopt = "
    "448*6/amax. Scales are static per expert ([E] or scalar) and baked at prepare_weights; 0.7.3 has no "
    "per-token activation outer scale (ActivationSpec has no outer_scale field) and no four-over-six block-scale "
    "selection (no '4over6'/'fouroversix' anywhere in b12x 0.7.3). Inkling's dynamic per-token + 4o6 recipe "
    "is NOT reproducible on b12x 0.7.3; here: static per-layer scales calibrated on this case's inputs."
)


def model_cfg(inter=I_FULL):
    return types.SimpleNamespace(n_routed_experts=E, num_experts_per_tok=TOPK, hidden_size=H,
                                 intermediate_size=inter, mlp_gate_softcap=GATE_CAP,
                                 mlp_up_softcap=UP_CAP)


def expert_bytes(i: int) -> int:
    """FP4 packed + e4m3 (block 16) scale bytes of one expert's w13 [2I x H] and w2 [H x I]."""
    return (2 * i * H) // 2 + (2 * i * H) // 16 + (H * i) // 2 + (H * i) // 16


def raw_slice(i: int, seed: int) -> dict[str, torch.Tensor]:
    """Checkpoint-format NVFP4 tensors at the rank-0 slice shape (w13 rows interleave gate/up)."""
    g = torch.Generator().manual_seed(seed)
    den = torch.tensor([DEN])
    return {
        "w13_weight": torch.randint(0, 256, (E, 2 * i, H // 2), generator=g, dtype=torch.uint8),
        "w13_weight_scale": (torch.rand(E, 2 * i, H // 16, generator=g) * 1.75 + 0.25).to(torch.float8_e4m3fn),
        "w13_weight_amax": torch.rand(E, generator=g) * 20 + 10,
        "w13_weight_outer_scale_denominator": den,
        "w2_weight": torch.randint(0, 256, (E, H, i // 2), generator=g, dtype=torch.uint8),
        "w2_weight_scale": (torch.rand(E, H, i // 16, generator=g) * 1.75 + 0.25).to(torch.float8_e4m3fn),
        "w2_weight_amax": torch.rand(E, generator=g) * 20 + 10,
        "w2_weight_outer_scale_denominator": den.clone(),
    }


def make_inputs(m: int, n: int, seed: int, dev="cuda"):
    """x [M,H] bf16, n routings (ids [M,8] int32 distinct per token, softmax weights f32), and
    n distinct layer inputs xs (routed by route_torch in the whole-layer timing)."""
    g = torch.Generator().manual_seed(seed * 1000 + m)
    x = (torch.randn(m, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    routes = []
    for _ in range(n):
        ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]).to(torch.int32)
        w = torch.softmax(torch.randn(m, TOPK, generator=g), -1).float()
        routes.append((ids.to(dev), w.to(dev)))
    xs = [(torch.randn(m, H, generator=g) * 0.5).to(torch.bfloat16).to(dev) for _ in range(n)]
    return x, routes, xs


def make_router(seed: int):
    g = torch.Generator().manual_seed(seed + 77)
    w = (torch.randn(E, H, generator=g) * 0.02).to(torch.bfloat16).cuda()
    b = (torch.randn(E, generator=g) * 0.01).float().cuda()
    return w, b


# ------------------------------------------------------- Marlin / FI W4A4 --
def check_marlin_layout(i: int) -> dict:
    from vllm.models.inkling2.moe_marlin import moe_channel_slices, moe_layout

    t = TP_OF_I[i]
    lay = moe_layout(t, 0, t, E, I_FULL, 16)
    assert moe_channel_slices(I_FULL, t)[0] == (0, I_FULL // t) and lay["inter_local"] == i == I_FULL // t, lay
    assert lay["num_experts"] == E and lay["ep_size"] == 1 and lay["inter_start"] == 0, lay
    return lay


def build_marlin(i: int, raw: dict, w4a4: bool = False):
    """Marlin W4A16 (t = TP_OF_I[i], rank 0) or FlashInfer W4A4 (t = 1 on the slice tensors)."""
    import vllm.models.inkling2.moe_marlin as mm

    check_marlin_layout(i)
    saved = (mm._W4A4, mm._W4A4_MIN)
    mm._W4A4, mm._W4A4_MIN = w4a4, (0 if w4a4 else saved[1])  # = INKLING2_W4A4_PREFILL=1, _MIN_TOKENS=0
    try:
        with torch.device("cuda"):
            if w4a4:
                mod = mm.Inkling2MarlinExperts(model_cfg(i), "nvfp4", rank_world=(0, 1), moe_tp=1, defer_finalize=True)
            else:
                t = TP_OF_I[i]
                mod = mm.Inkling2MarlinExperts(model_cfg(), "nvfp4", rank_world=(0, t), moe_tp=t, defer_finalize=True)
        # weight_loader = shard_expert_tensor: rank 0 takes rows [0, 2I) of w13 and
        # columns [0, I/2) of w2 -- exactly the whole of a slice-shaped tensor.
        for name, tensor in raw.items():
            p = getattr(mod, name)
            p.weight_loader(p, tensor.cuda())
        mod.finalize_weights()
    finally:
        mm._W4A4, mm._W4A4_MIN = saved
    if w4a4:
        assert mod.w4a4 and not mod.use_marlin and mod.intermediate_size == i
    else:
        assert mod.expert_map is None and mod.prepared_intermediate() == i
    return mod


# -------------------------------------------------------------------- b12x --
def b12x_weight_plan(i: int, a4: bool = False):
    from b12x.moe import fused_moe as fm

    mode = fm.ActivationMode.A4 if a4 else fm.ActivationMode.A16
    return fm.plan_weights(
        source=fm.PackedSource(format=fm.PackedSourceFormat.MODELOPT_NVFP4, w13_layout=fm.W13Layout.W13),
        activation=fm.ActivationSpec(mode=mode, nonlinearity="situ", io_dtype=torch.bfloat16),
        geometry=fm.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=i),
    )


def b12x_tensors(raw: dict) -> dict:
    """b12x-layout GPU tensors (kept once per I; every prepare_weights gets clones)."""
    from b12x._lib.intrinsics import swizzle_block_scale

    def up_gate(t):  # checkpoint rows (gate0, up0, gate1, ...) -> W13 layout = [up; gate]
        return torch.cat([t[:, 1::2], t[:, 0::2]], 1).contiguous().cuda()

    def swz(s_u8):
        return swizzle_block_scale(s_u8).contiguous().view(torch.float8_e4m3fn)

    g13 = raw["w13_weight_amax"] / raw["w13_weight_outer_scale_denominator"]
    g2 = raw["w2_weight_amax"] / raw["w2_weight_outer_scale_denominator"]
    return dict(w13=up_gate(raw["w13_weight"]), w2=raw["w2_weight"].contiguous().cuda(),
                w13_block_scales=swz(up_gate(raw["w13_weight_scale"].view(torch.uint8))),
                w2_block_scales=swz(raw["w2_weight_scale"].view(torch.uint8).cuda()),
                w13_global_scales=g13.float().contiguous().cuda(), w2_global_scales=g2.float().contiguous().cuda())


def build_b12x(i: int, bt: dict, a4_scales: tuple[float, float] | None = None):
    """a4_scales = (a1_gscale, a2_gscale) = (2688/amax|x|, 2688/amax|SiTU|) selects A4."""
    from b12x.moe import fused_moe as fm

    plan = b12x_weight_plan(i, a4=a4_scales is not None)
    kw = {k: v.clone() for k, v in bt.items()}  # prepare may repack its inputs in place
    if a4_scales is not None:
        kw.update(input_scale=torch.full((E,), a4_scales[0], dtype=torch.float32, device="cuda"),
                  intermediate_scale=torch.full((E,), a4_scales[1], dtype=torch.float32, device="cuda"),
                  immutable_input_scales=True)
    return fm.prepare_weights(plan=plan, weights=fm.PackedWeights(**kw))


def alloc_scratch(specs):
    t = [torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in specs]
    return t[0] if len(t) == 1 else t


def b12x_prepare(experts, m, triples, out, mode, cache_dir, tag):
    """Prepare one exact-M plan; bindings for each (x, ids, w) triple (shared scratch + output)."""
    from b12x.moe import fused_moe as fm
    from b12x.preparation import PreparationSession, PreparedCall

    plan = fm.plan_execution(experts=experts, capacity=fm.ExecutionCapacity(max_tokens=m, top_k=TOPK))
    x0, ids0, w0 = triples[0]

    def factory(state):
        scratch = alloc_scratch(state.scratch.scratch_specs())
        b = state.bind(scratch=scratch, a=x0, experts=experts, topk_weights=w0, topk_ids=ids0, output=out)
        return PreparedCall(run=lambda: state.run(b), owners=(scratch, b))

    sess = PreparationSession(autotune=(mode == "tuned"), cache_dir=cache_dir)
    t0 = time.time()
    sess.prepare((plan.request(name=f"moe_{tag}_M{m}", prepare_call=factory, benchmark_call=factory),))
    print(f"  {tag} M={m} prepared in {time.time() - t0:.1f}s; selection={plan.selection}", flush=True)
    scratch = alloc_scratch(plan.scratch_specs())
    bindings = [fm.bind(plan, scratch=scratch, a=x, experts=experts, topk_weights=w, topk_ids=ids, output=out)
                for x, ids, w in triples]
    return sess, plan, bindings, scratch


# ----------------------------------------------------------------- timing ---
def time_graph(calls, iters, reps, capture_ctx=None):
    for _ in range(3):
        for c in calls:
            c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with (capture_ctx() if capture_ctx is not None else _null()):
        with torch.cuda.graph(g):
            for k in range(iters):
                calls[k % len(calls)]()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for s, e in ev:
        s.record()
        g.replay()
        e.record()
    torch.cuda.synchronize()
    ms = statistics.median(s.elapsed_time(e) for s, e in ev)
    del g
    return ms * 1000.0 / iters


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------- reference ------
def situ(g, u):
    return (0.5 * GATE_CAP) * torch.tanh(g / GATE_CAP) * (1.0 + torch.tanh(g * 0.5)) * (UP_CAP * torch.tanh(u / UP_CAP))


def ref_moe(raw, triples):
    """fp32 dequantized reference over the touched experts of every (x, ids, w) triple.
    Returns ([out per triple], amax of the SiTU output = GEMM2 input)."""
    lut = torch.tensor(FP4, device="cuda")

    def deq(codes, scales, gscale):
        c = codes.cuda()
        val = [torch.where((n & 8) > 0, -lut[(n & 7).long()], lut[(n & 7).long()]) for n in (c & 0xF, c >> 4)]
        return torch.stack(val, -1).flatten(-2) * scales.cuda().float().repeat_interleave(16, -1) * gscale

    outs = [torch.zeros(x.shape[0], H, device="cuda") for x, _, _ in triples]
    amax = 0.0
    den13 = raw["w13_weight_outer_scale_denominator"].item()
    den2 = raw["w2_weight_outer_scale_denominator"].item()
    for e in torch.cat([ids.flatten() for _, ids, _ in triples]).unique().tolist():
        w13 = deq(raw["w13_weight"][e], raw["w13_weight_scale"][e], raw["w13_weight_amax"][e].item() / den13)
        w2 = deq(raw["w2_weight"][e], raw["w2_weight_scale"][e], raw["w2_weight_amax"][e].item() / den2)
        for (x, ids, w), out in zip(triples, outs):
            rows, slots = (ids == e).nonzero(as_tuple=True)
            if rows.numel():
                h = x[rows].float() @ w13.t()
                a = situ(h[:, 0::2], h[:, 1::2])
                amax = max(amax, a.abs().max().item())
                out.index_add_(0, rows, (a @ w2.t()) * w[rows, slots].unsqueeze(1))
    return outs, amax


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


# -------------------------------------------------------------- dry run -----
def dry_run(args):
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    ok = True
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: F401
        from vllm.models.inkling2 import moe_marlin as mm
        print(f"marlin: {mm.__file__}; MoEActivation.SITU={MoEActivation.SITU}; route_torch={mm.route_torch}")
        print(f"  module W4A4 globals (patched per build): _W4A4={mm._W4A4} _W4A4_MIN={mm._W4A4_MIN} "
              f"_W4A4_4O6={mm._W4A4_4O6}")
        from flashinfer.fused_moe import ActivationType, cutlass_fused_moe, cutlass_fused_moe_workspace_size  # noqa: F401
        from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config  # noqa: F401
        print(f"  flashinfer W4A4 imports OK (ActivationType.Situ={ActivationType.Situ})")
        for i in args.I:
            lay = check_marlin_layout(i)
            with torch.device("meta"):
                mod = mm.Inkling2MarlinExperts(model_cfg(), "nvfp4", rank_world=(0, TP_OF_I[i]),
                                               moe_tp=TP_OF_I[i], defer_finalize=True)
                saved = (mm._W4A4, mm._W4A4_MIN)
                mm._W4A4, mm._W4A4_MIN = True, 0
                try:
                    fi = mm.Inkling2MarlinExperts(model_cfg(i), "nvfp4", rank_world=(0, 1), moe_tp=1,
                                                  defer_finalize=True)
                finally:
                    mm._W4A4, mm._W4A4_MIN = saved
            shapes = {n: tuple(p.shape) for n, p in mod.named_parameters()}
            fshapes = {n: tuple(p.shape) for n, p in fi.named_parameters()}
            assert shapes == fshapes, (shapes, fshapes)
            print(f"  I={i}: world=t={TP_OF_I[i]} rank0 layout {lay}\n    params {shapes}\n"
                  f"    fi_w4a4 t=1 module: w4a4={fi.w4a4} use_marlin={fi.use_marlin} same param shapes")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"marlin/FI side FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    try:
        import flashinfer.experimental.b12x as fb
        from b12x._lib.intrinsics import swizzle_block_scale  # noqa: F401
        from b12x.moe import fused_moe as fm
        from b12x.preparation import PreparationSession, PreparedCall  # noqa: F401
        print(f"b12x: {fb.__file__}")
        for name in ("plan_weights", "prepare_weights", "plan_execution", "bind", "run", "ExecutionCapacity",
                     "PackedWeights", "PackedSource", "ActivationSpec", "MoEGeometry", "W13Layout"):
            getattr(fm, name)
        for i in args.I:
            for a4 in (False, True):
                try:
                    p = b12x_weight_plan(i, a4=a4)
                    print(f"  I={i} {'A4 ' if a4 else 'A16'}: plan_weights OK -> packing={p.prepared_format.packing.value}"
                          f", quant_modes={sorted(p._impl.quant_modes)}, transforms={sorted(t.value for t in p._impl.transforms)}"
                          f", prepares_runtime_alphas={p._impl.prepares_runtime_alphas}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  I={i} {'A4' if a4 else 'A16'}: plan_weights REJECTED: {type(exc).__name__}: {exc}")
        print("b12x API calls the GPU run makes:")
        for c in B12X_CALLS:
            print(f"  - {c}")
        print(f"A4 scale note: {B12X_A4_NOTE}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"b12x side FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    print("NOT checked without a GPU: Marlin repack + kernels, FlashInfer W4A4 finalize/workspace/kernels, b12x "
          "prepare_weights (CUDA tensors), plan_execution lowering, PreparationSession, autotune, bind/run, capture.")
    backends = (["marlin", "fi_w4a4"] + [f"b12x_{x}" for x in args.b12x_modes]
                + [f"b12x_a4_{x}" for x in args.b12x_a4_modes])
    print("case plan (backend x I x M x {experts, layer}):")
    for i in args.I:
        for m in args.M:
            print(f"  I={i:4d} (TP{TP_OF_I[i]}) M={m:3d}  backends={','.join(backends)}  "
                  f"bytes/expert={expert_bytes(i)}  max experts touched={min(E, m * TOPK)}")
    print("activation: SiTU on all backends (b12x 'situ' hardwired beta=4.0 / linear_beta=25.0; "
          "FlashInfer ActivationType.Situ with beta 4 / linear_beta 25 tensors)")
    print("layer routing: moe_marlin.route_torch for every backend (b12x 0.7.3 route_topk: softmax/sqrtsoftplus only)")
    return 0 if ok else 1


# --------------------------------------------------------------------- main -
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="moe_race.json")
    ap.add_argument("--I", type=lambda s: [int(v) for v in s.split(",")], default=[384, 576])
    ap.add_argument("--M", type=lambda s: [int(v) for v in s.split(",")], default=[8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=32, help="calls captured per graph")
    ap.add_argument("--reps", type=int, default=50, help="graph replays timed")
    ap.add_argument("--routings", type=int, default=4, help="distinct routings / layer inputs cycled in the graph")
    ap.add_argument("--b12x-modes", type=lambda s: [v for v in s.split(",") if v], default=["heur", "tuned"])
    ap.add_argument("--b12x-a4-modes", type=lambda s: [v for v in s.split(",") if v], default=["heur", "tuned"])
    ap.add_argument("--no-fi", action="store_true", help="skip the FlashInfer W4A4 backend")
    ap.add_argument("--b12x-cache", default=None, help="PreparationSession cache_dir (tuning cache)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    assert all(i in TP_OF_I for i in args.I), f"--I must be within {list(TP_OF_I)}"
    if args.dry_run:
        return dry_run(args)
    assert torch.cuda.is_available(), "GPU run needs CUDA (use --dry-run otherwise)"
    from vllm.models.inkling2.moe_marlin import route_torch

    torch.manual_seed(args.seed)
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU {dev.name} sm_{dev.major}{dev.minor} {dev.total_memory / 2**30:.0f} GiB; torch {torch.__version__}")
    notes = {"activation": "SiTU gate_cap=4 up_cap=25 on all backends",
             "layer_routing": "moe_marlin.route_torch (fp32 router GEMM + sigmoid + bias top-8) for every backend; "
                              "b12x 0.7.3 route_topk supports only softmax/sqrtsoftplus",
             "fi_w4a4": "Inkling2MarlinExperts _W4A4=1 _W4A4_MIN=0, t=1 module on the slice tensors "
                        "(module refuses W4A4 with MOE_TP>1); dynamic per-batch global scale, 4o6 block scales",
             "b12x_a4": B12X_A4_NOTE, "b12x_calls": B12X_CALLS, "errors": [], "numerics": [], "a4_scales": []}
    results = []
    rw, rb = make_router(args.seed)

    def record(backend, kind, i, m, us, ids_list):
        uniq = statistics.mean(len(torch.unique(ids)) for ids in ids_list)
        nbytes = uniq * expert_bytes(i)
        gbs = nbytes / (us * 1e-6) / 1e9
        r = dict(backend=backend, kind=kind, I=i, tp=TP_OF_I[i], M=m, us=round(us, 2), unique_experts=uniq,
                 weight_bytes=int(nbytes), GBps=round(gbs, 1), pct_peak=round(100 * gbs * 1e9 / PEAK_BPS, 1))
        results.append(r)
        print(f"RESULT {backend:13s} {kind:7s} I={i:4d} M={m:3d}  {us:8.2f} us  experts={uniq:6.1f}  "
              f"{nbytes / 1e6:8.1f} MB  {gbs:7.1f} GB/s  {r['pct_peak']:5.1f}% of 1.79 TB/s", flush=True)

    def fail(what, exc):
        msg = f"{what}: {type(exc).__name__}: {exc}"
        print(f"ERROR {msg}", flush=True)
        traceback.print_exc()
        notes["errors"].append(msg)

    for i in args.I:
        print(f"=== I={i} (TP{TP_OF_I[i]} rank 0, {E} experts, {expert_bytes(i) / 1e6:.3f} MB/expert)", flush=True)
        raw = raw_slice(i, args.seed + i)
        cases = {}
        for m in args.M:
            x, routes, xs = make_inputs(m, args.routings, args.seed)
            lroutes = [route_torch(xx, rw, rb, TOPK, ROUTE_SCALE) for xx in xs]  # (w, ids) per layer input
            ref_out, mid_amax = ref_moe(raw, [(x, ids, w) for ids, w in routes]
                                        + [(xx, ids, w) for xx, (w, ids) in zip(xs, lroutes)])
            x_amax = max(t.abs().max().item() for t in [x, *xs])
            cases[m] = dict(x=x, routes=routes, xs=xs, lroutes=lroutes, ref=ref_out[0], x_amax=x_amax,
                            mid_amax=mid_amax, outs={})
        torch.cuda.empty_cache()

        def numerics(backend, m, out):
            c = cases[m]
            n = dict(backend=backend, I=i, M=m, rel_vs_ref=rel(out, c["ref"]))
            if "marlin" in c["outs"] and backend != "marlin":
                n["rel_vs_marlin"] = rel(out, c["outs"]["marlin"])
            c["outs"][backend] = out.clone()
            notes["numerics"].append(n)
            print(f"NUMERICS {n}", flush=True)

        # Marlin W4A16 and FlashInfer CUTLASS W4A4 (same module class, same call contract).
        for backend, w4a4 in (("marlin", False),) + (() if args.no_fi else (("fi_w4a4", True),)):
            try:
                mod = build_marlin(i, raw, w4a4=w4a4)
                for m in args.M:
                    c = cases[m]
                    x = c["x"]
                    calls = [lambda x=x, ids=ids, w=w: mod(x, w, ids, all_reduce=False) for ids, w in c["routes"]]
                    record(backend, "experts", i, m, time_graph(calls, args.iters, args.reps),
                           [ids for ids, _ in c["routes"]])

                    def layer(xx):
                        w, ids = route_torch(xx, rw, rb, TOPK, ROUTE_SCALE)
                        return mod(xx, w, ids, all_reduce=False)

                    calls = [lambda xx=xx: layer(xx) for xx in c["xs"]]
                    record(backend, "layer", i, m, time_graph(calls, args.iters, args.reps),
                           [ids for _, ids in c["lroutes"]])
                    numerics(backend, m, mod(x, c["routes"][0][1], c["routes"][0][0], all_reduce=False))
            except Exception as exc:  # noqa: BLE001
                fail(f"{backend} I={i}", exc)
            mod = None
            torch.cuda.empty_cache()

        # b12x W4A16 (weights prepared once per I) and W4A4 (static scales: prepared per (I, M)).
        try:
            bt = b12x_tensors(raw)
        except Exception as exc:  # noqa: BLE001
            fail(f"b12x tensor conversion I={i}", exc)
            bt = None
        arms = [("a16", mode) for mode in args.b12x_modes] + [("a4", mode) for mode in args.b12x_a4_modes]
        experts16 = None
        if bt is not None and args.b12x_modes:
            try:
                experts16 = build_b12x(i, bt)
                print(f"  b12x A16 weights prepared: {experts16.plan.prepared_format}", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail(f"b12x A16 REJECTED I={i} at plan_weights/prepare_weights", exc)
        for prec, mode in arms if bt is not None else ():
            backend = f"b12x_{mode}" if prec == "a16" else f"b12x_a4_{mode}"
            for m in args.M:
                c = cases[m]
                experts = experts16
                try:
                    from b12x.moe import fused_moe as fm

                    if prec == "a4":
                        scales = (DEN / c["x_amax"], DEN / c["mid_amax"])
                        notes["a4_scales"].append(dict(I=i, M=m, x_amax=c["x_amax"], mid_amax=c["mid_amax"],
                                                       a1_gscale=scales[0], a2_gscale=scales[1]))
                        experts = build_b12x(i, bt, a4_scales=scales)
                    if experts is None:
                        continue
                    out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
                    ids_bufs = [torch.empty(m, TOPK, dtype=torch.int32, device="cuda") for _ in c["xs"]]
                    w_bufs = [torch.empty(m, TOPK, dtype=torch.float32, device="cuda") for _ in c["xs"]]
                    triples = ([(c["x"], ids, w) for ids, w in c["routes"]]
                               + list(zip(c["xs"], ids_bufs, w_bufs)))
                    sess, plan, bindings, scratch = b12x_prepare(experts, m, triples, out, mode,
                                                                 args.b12x_cache, backend)
                    nr = len(c["routes"])
                    calls = [lambda b=b: fm.run(binding=b) for b in bindings[:nr]]
                    record(backend, "experts", i, m, time_graph(calls, args.iters, args.reps, sess.capture),
                           [ids for ids, _ in c["routes"]])

                    def layer(xx, ib, wb, b):
                        w, ids = route_torch(xx, rw, rb, TOPK, ROUTE_SCALE)
                        ib.copy_(ids)
                        wb.copy_(w)
                        fm.run(binding=b)

                    calls = [lambda a=a: layer(*a) for a in zip(c["xs"], ids_bufs, w_bufs, bindings[nr:])]
                    record(backend, "layer", i, m, time_graph(calls, args.iters, args.reps, sess.capture),
                           [ids for _, ids in c["lroutes"]])
                    fm.run(binding=bindings[0])
                    torch.cuda.synchronize()
                    numerics(backend, m, out)
                    sess.close()
                    del sess, plan, bindings, scratch
                except Exception as exc:  # noqa: BLE001
                    fail(f"{backend} I={i} M={m}", exc)
                if prec == "a4":
                    experts = None
                    torch.cuda.empty_cache()
        experts16 = bt = None
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"results": results, "notes": notes, "args": vars(args), "gpu": dev.name}, f, indent=1)
    print(f"wrote {args.out}")
    print("\nbackend        kind       I     M        us   GB/s   %peak")
    for r in results:
        print(f"{r['backend']:13s} {r['kind']:7s} {r['I']:4d} {r['M']:4d} {r['us']:9.2f} {r['GBps']:6.0f} "
              f"{r['pct_peak']:6.1f}")
    print("\nnumerics (rel L2): backend I M vs_ref vs_marlin")
    for n in notes["numerics"]:
        print(f"  {n['backend']:13s} {n['I']:4d} {n['M']:3d} {n['rel_vs_ref']:.2e} {n.get('rel_vs_marlin', float('nan')):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
