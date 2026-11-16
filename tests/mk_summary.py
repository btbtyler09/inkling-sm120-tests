"""mk_summary.py PUB_DIR -> summary.md on stdout: arm status, decode tok/s, long-context TTFT/ITL, DSpark stats, the per-step
GPU time of each profiled arm (top kernels), the MoE kernel race, and the W4A4 logprob drift."""
import glob, os, re, sys

P = sys.argv[1]
con = open(os.path.join(P, "console.log"), errors="replace").read() if os.path.exists(os.path.join(P, "console.log")) else ""
out = ["# Inkling-2 2-GPU test results", ""]
env = os.path.join(P, "env.txt")
if os.path.exists(env):
    head = [ln for ln in open(env, errors="replace").read().splitlines() if ln.startswith(("image:", "model code", "vllm "))]
    out += ["```"] + head + ["```", ""]
out += ["## Arms", "", "| arm | status | finished (UTC) | detail |", "|---|---|---|---|"]
for ln in open(os.path.join(P, "arms.tsv")).read().splitlines() if os.path.exists(os.path.join(P, "arms.tsv")) else []:
    a = ln.split("\t") + ["", "", "", ""]
    out.append(f"| {a[0]} | {a[1]} | {a[2]} | {a[3][:160].replace('|', '/')} |")
arms = sorted(d for d in os.listdir(os.path.join(P, "arms"))) if os.path.isdir(os.path.join(P, "arms")) else []


def res(tag, key):
    m = re.findall(rf"^RES {re.escape(tag)} {re.escape(key)} (.*)$", con, re.M)
    return m[-1].strip() if m else ""


out += ["", "## Decode (post_live tps, aggregate tok/s) and DSpark", "", "| arm | c=1 | c=4 | c=8 | drafted | accepted |", "|---|---|---|---|---|---|"]
for t in arms:
    sp = res(t, "spec")
    dr = re.search(r"Drafted: (\d+)", sp); ac = re.search(r"Accepted: (\d+)", sp)
    out.append(f"| {t} | {res(t, 'c=1').replace('aggregate=', '')} | {res(t, 'c=4').replace('aggregate=', '')} | "
               f"{res(t, 'c=8').replace('aggregate=', '')} | {dr.group(1) if dr else ''} | {ac.group(1) if ac else ''} |")
out += ["", "## Long context (ld_ctx reuse, c=1)", "", "| arm | ctx | cold TTFT | warm TTFT | decode tok/s | ITL p50 | ITL p99 |", "|---|---|---|---|---|---|---|"]
for t in arms:
    for x in ("ctx32", "ctx128"):
        r = res(t, x)
        if not r:
            continue
        g = lambda k: (re.search(rf"{k}=([0-9.]+)", r) or [None, ""])[1]
        out.append(f"| {t} | {x} | {g('cold_ttft_med')} s | {g('warm_ttft_med')} s | {g('decode_tok_s_med')} | {g('itl_p50')} ms | {g('itl_p99')} ms |")
out += ["", "## Per-step GPU time (profiler window, 32 steps)", ""]
for t in arms:
    for c in (1, 4):
        f = os.path.join(P, "arms", t, f"kernels_c{c}.txt")
        if not os.path.exists(f):
            continue
        lines = open(f, errors="replace").read().splitlines()
        if not lines or not lines[0].startswith("TRACE"):
            continue
        w = re.search(r"window ([0-9.]+) us/step, GPU busy ([0-9.]+) us/step \((\d+)%\)", lines[0])
        out.append(f"**{t} c={c}**: window {w.group(1)} us/step, GPU busy {w.group(2)} us/step ({w.group(3)}%)" if w else f"**{t} c={c}**")
        out += ["```"] + [ln[:150] for ln in lines[1:9]] + ["```"]
rf = os.path.join(P, "moe_race.log")
if os.path.exists(rf):
    out += ["", "## MoE kernel race (per-GPU expert shard, us per call)", "", "```"]
    out += [ln[:170] for ln in open(rf, errors="replace").read().splitlines() if ln.startswith(("RESULT", "ERROR"))][:80]
    out += ["```"]
for t in ("W4", "W4P"):
    lp = os.path.join(P, "arms", t, "lp_vs_W0.txt")
    if os.path.exists(lp):
        out += ["", f"## {t} (W4A4) vs W0 (W4A16) logprob drift", "", "```"] + open(lp, errors="replace").read().splitlines()[-8:] + ["```"]
print("\n".join(out))
