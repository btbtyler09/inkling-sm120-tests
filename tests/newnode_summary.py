"""newnode_summary.py PUB_DIR -> markdown: the newnode phase in one place. All-reduce per GPU group (decode p50 us at M=1/8/64,
x96 ms per decode step at M=8, prefill algbw GB/s at M=2048) and per layout boot (KV tokens, max concurrency at 262,144,
model memory, per-GPU memory after boot). Numbers only."""
import json, os, re, sys

P = sys.argv[1]
out = ["## newnode summary", ""]
st = os.path.join(P, "newnode_settings.txt")
if os.path.exists(st):
    out += ["Serve settings used for the boots (production):", "", "```"] + open(st).read().splitlines() + ["```", ""]
arj = os.path.join(P, "ar_bench.json")
if os.path.exists(arj):
    r = json.load(open(arj))
    gpus = r.get("gpus", {})
    out += ["### All-reduce (bf16 [M, 4608], vLLM path, CUDA graph)", "",
            "| group | NUMA nodes | backend | M=1 p50 us | M=8 p50 us | M=64 p50 us | decode step x96 (M=8) | M=2048 GB/s |",
            "|---|---|---|---|---|---|---|---|"]
    for g, v in r.get("groups", {}).items():
        if "error" in v:
            out.append(f"| {g} | | FAILED: {v['error'][:60]} | | | | | |")
            continue
        ids = [int(x) for x in g.split(",")]
        numa = sorted({str(gpus.get(str(i), gpus.get(i, ["", "?"]))[1]) for i in ids})
        inf = v["info"]
        be = "custom" if inf.get("custom_ar") else ("flashinfer-pcie-ipc" if inf.get("flashinfer_pcie_ipc") else "NCCL")
        res = v["res"]
        p = lambda m: res.get(f"vllm_M{m}", [float("nan")])[0]
        bw = 2048 * 4608 * 2 / p(2048) / 1e3 if f"vllm_M2048" in res else float("nan")
        out.append(f"| {g} | {','.join(numa)} | {be} | {p(1):.1f} | {p(8):.1f} | {p(64):.1f} | {96 * p(8) / 1000:.2f} ms | {bw:.1f} |")
    out.append("")
names = {"NB0": "TP6, no draft", "NB": "TP6 + co-located DSpark (B)", "NC": "TP6 + drafter on its own GPU (C)",
         "NT4": "TP4 one socket + co-located DSpark"}
out += ["### Layout boots (production serve settings above, max-model-len 262,144)", "",
        "| layout | status | KV tokens | max concurrency @262k | model GiB/GPU | GPU memory used / total after boot (MiB) |",
        "|---|---|---|---|---|---|"]
status = {}
tsv = os.path.join(P, "arms.tsv")
if os.path.exists(tsv):
    for ln in open(tsv):
        a = ln.rstrip("\n").split("\t")
        status[a[0]] = f"{a[1]} {a[3][:60] if len(a) > 3 and a[1] != 'PASS' else ''}".strip()
for tag, name in names.items():
    d = os.path.join(P, "arms", tag)
    log = open(os.path.join(d, "server.log"), errors="replace").read() if os.path.exists(os.path.join(d, "server.log")) else ""
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log)
    ml = re.findall(r"Model loading took ([\d.]+) GiB", log)
    used = {}
    if os.path.exists(os.path.join(d, "mem_after_boot.txt")):
        for ln in open(os.path.join(d, "mem_after_boot.txt")):
            if "," in ln:
                i, m = ln.split(",", 1)
                used[i.strip()] = m.strip().replace(" MiB", "")
    tot = {}
    if os.path.exists(os.path.join(d, "mem_total.txt")):
        for ln in open(os.path.join(d, "mem_total.txt")):
            if "," in ln:
                i, m = ln.split(",", 1)
                tot[i.strip()] = m.strip().replace(" MiB", "")
    mem = "; ".join(f"{i}: {u}/{tot.get(i, '?')}" for i, u in used.items() if u not in ("0", ""))
    out.append(f"| {name} | {status.get(tag, 'not run')} | {kv[-1][0] if kv else ''} | {kv[-1][1] + 'x' if kv else ''} | "
               f"{ml[-1] if ml else ''} | {mem.strip('; ')} |")
print("\n".join(out))
