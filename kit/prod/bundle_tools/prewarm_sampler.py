"""prewarm_sampler.py MODEL_DIR MAX_NUM_SEQS [NUM_SPEC_TOKENS] — compile every sampler kernel variant into TRITON_CACHE_DIR.

vLLM's JIT warmup only compiles kernels that register during runner setup; the sampler's top-k/top-p, top-p split and
gumbel kernels never register (the dummy sampler run uses none of those options), so in production they compile on the
first request whose batch shape/option mix is new. Their own warmup enumerations exist (get_warmup_keys(vllm_config)):
this script runs them against a stand-in config (vocab, max batch) so the compiled variants land in the bundle's cache.
Run it with the same image, GPU and TRITON_CACHE_DIR as the bundle build."""
import importlib, inspect, json, os, sys, time, types

model_dir, max_seqs = sys.argv[1], int(sys.argv[2])
spec = int(sys.argv[3]) if len(sys.argv) > 3 else 0
cfg = json.load(open(os.path.join(model_dir, "config.json")))
# kit copy (build-bundle.sh mounts it over the image's): + llm_config, where Nemotron VL keeps its LM settings (11-17)
vocab = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size") or cfg.get("llm_config", {}).get("vocab_size")
if not vocab:
    sys.exit("no vocab_size in config.json")

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel  # noqa: E402

model_config = types.SimpleNamespace(get_vocab_size=lambda: vocab, vocab_size=vocab)
vcfg = types.SimpleNamespace(model_config=model_config, num_speculative_tokens=spec,
                             scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs))
MODULES = ["vllm.v1.sample.ops.topk_topp_triton"] + [f"vllm.v1.worker.gpu.sample.{m}" for m in (
    "gumbel", "logprob", "min_p", "penalties", "watermark", "bad_words", "logit_bias", "prompt_logprob", "sampler", "thinking_budget")]
done, skipped = [], []
for mname in MODULES:
    try:
        mod = importlib.import_module(mname)
    except Exception as e:
        skipped.append(f"{mname}: import {type(e).__name__}"); continue
    for name, obj in vars(mod).items():
        if not isinstance(obj, VllmJitKernel):
            continue
        t = time.time()
        try:
            params = inspect.signature(obj.get_warmup_keys).parameters
            keys = obj.get_warmup_keys(vllm_config=vcfg) if "vllm_config" in params else obj.get_warmup_keys(vcfg)
            obj.compile_many(keys)
            done.append({"kernel": f"{mname.rsplit('.', 1)[-1]}.{name}", "variants": len(keys), "s": round(time.time() - t, 1)})
        except Exception as e:
            skipped.append(f"{mname}.{name}: {type(e).__name__}: {str(e)[:160]}")
print(json.dumps({"vocab": vocab, "max_batch": max_seqs * (spec + 1), "compiled": done, "skipped": skipped}, indent=1))
print("PREWARM_SAMPLER", "OK" if done else "NOTHING", sum(d["variants"] for d in done), "variants")
