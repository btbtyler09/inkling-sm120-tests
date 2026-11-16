"""ar_bench.py [--groups "0,1,2,3;0,1,2,3,4,5"] [--sizes 1,2,4,8,16,32,64] -- decode all-reduce latency, no model load.

For each GPU group (indices as seen inside the container), spawns one process per GPU, initializes vLLM's distributed
state exactly as a TP server does, and reports:
  - which all-reduce backends vLLM enabled for that group (custom all-reduce, FlashInfer PCIe IPC, else NCCL), and why
    custom all-reduce is off when it is (vLLM disables it for >2 PCIe-only GPUs)
  - per-call latency (p50 / p99 over CUDA-graph replays of 32 calls, the way decode runs) of a bf16 [M, 4608] all-reduce
    for M tokens: raw NCCL (torch.distributed) and vLLM's tensor_model_parallel_all_reduce (the path the model uses)
  - each GPU's PCIe bus id and NUMA node (socket), so a group spanning both sockets is visible
Inkling-2 decode: 2 all-reduces per layer x 48 layers per step, so per-step AR cost ~ 96 x the per-call number.
Runs in about 1-3 minutes per group. VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 in the env also tests that backend."""
import argparse, json, os, socket, statistics, sys, time

HIDDEN = 4608


def numa_of(bus_id: str) -> str:
    b = bus_id.lower()
    if b.count(":") == 2 and len(b.split(":")[0]) == 8:
        b = b[4:]   # nvidia-smi 00000000:41:00.0 -> sysfs 0000:41:00.0
    try:
        return open(f"/sys/bus/pci/devices/{b}/numa_node").read().strip()
    except OSError:
        return "?"


def worker(rank, world, port, sizes, out_q):
    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.distributed import tensor_model_parallel_all_reduce

    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        ps.init_distributed_environment(world, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
        ps.initialize_model_parallel(tensor_model_parallel_size=world)
        tp = ps.get_tp_group()
        comm = tp.device_communicator
        ca = getattr(comm, "ca_comm", None)
        fi = getattr(comm, "fi_pcie_ipc_ar_comm", None)
        info = {"custom_ar": ca is not None and not ca.disabled,
                "custom_ar_fully_connected": getattr(ca, "fully_connected", None),
                "flashinfer_pcie_ipc": fi is not None and not getattr(fi, "disabled", True)}
        res = {}
        for m in sizes:
            x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
            for name in ("nccl", "vllm"):
                def call(name=name):
                    if name == "nccl":
                        dist.all_reduce(x, group=tp.device_group)
                        return x
                    return tensor_model_parallel_all_reduce(x)
                for _ in range(5):
                    call()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with ps.graph_capture(torch.device("cuda", rank)) as gc:
                    with torch.cuda.graph(g, stream=gc.stream):
                        for _ in range(32):
                            call()
                torch.cuda.synchronize()
                ts = []
                for _ in range(40):
                    dist.barrier(group=tp.cpu_group)
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); g.replay(); b.record(); b.synchronize()
                    ts.append(a.elapsed_time(b) * 1000 / 32)
                ts.sort()
                res[f"{name}_M{m}"] = (statistics.median(ts), ts[int(0.99 * (len(ts) - 1))])
                del g
        if rank == 0:
            out_q.put({"info": info, "res": res})
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()


def run_group(gpus, sizes):
    import torch.multiprocessing as mp

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, len(gpus), port, sizes, q)) for r in range(len(gpus))]
    for p in procs:
        p.start()
    try:
        out = q.get(timeout=600)
    except Exception as exc:  # noqa: BLE001
        out = {"error": repr(exc)}
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="0,1,2,3;0,1,2,3,4,5")
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default="ar_bench.json")
    a = ap.parse_args()
    sizes = [int(v) for v in a.sizes.split(",")]
    import subprocess

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        sys.exit("ar_bench: no nvidia-smi in this container: run with --gpus")
    gpu_info = {}
    for ln in smi:
        i, bus, name = [v.strip() for v in ln.split(",", 2)]
        gpu_info[int(i)] = (bus, numa_of(bus), name)
        print(f"GPU {i} {name} bus {bus} numa {numa_of(bus)}")
    print(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout)
    report = {"gpus": gpu_info, "groups": {}, "env_fi_pcie_ipc": os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC")}
    for grp in a.groups.split(";"):
        gpus = [int(v) for v in grp.split(",") if v != ""]
        if any(g not in gpu_info for g in gpus):
            print(f"GROUP {grp}: SKIPPED (only {len(gpu_info)} GPUs visible)"); continue
        t0 = time.time()
        out = run_group(gpus, sizes)
        report["groups"][grp] = out
        socks = sorted({gpu_info[g][1] for g in gpus})
        if "error" in out:
            print(f"GROUP {grp} (numa {socks}): FAILED {out['error']}"); continue
        inf = out["info"]
        print(f"GROUP {grp} (numa nodes {socks}, {time.time() - t0:.0f} s): custom_ar={inf['custom_ar']} "
              f"(fully_connected={inf['custom_ar_fully_connected']}) flashinfer_pcie_ipc={inf['flashinfer_pcie_ipc']}")
        for m in sizes:
            n50, n99 = out["res"][f"nccl_M{m}"]
            v50, v99 = out["res"][f"vllm_M{m}"]
            print(f"  M={m:3d}  nccl p50 {n50:7.1f} us p99 {n99:7.1f} | vllm path p50 {v50:7.1f} us p99 {v99:7.1f} "
                  f"| x96 per decode step: {96 * v50 / 1000:.2f} ms")
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"AR_BENCH_DONE wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
