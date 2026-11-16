"""ld_ctx.py N C [MODE] -- long-context agentic load against :8000 (speed only; the slice's text is meaningless).
C sessions, each with its own ~N-token document (distinct per session). Per session: turn 1 = cold prefill + 128 out,
turns 2-3 = same document + a short new question (prefix-cache hit) + 256 out. All sessions run concurrently.
MODE=reuse (default): prints cold TTFT, warm TTFT, decode tok/s, ITL p50/p99 over the warm turns.
MODE=interfere: C-1 sessions decode warm turns while ONE extra session sends a cold N-token prefill; prints the decoding
sessions' ITL p50/p99/max inside the prefill window vs outside it (how much a new long prompt stalls the others)."""
import concurrent.futures as cf, json, sys, threading, time, urllib.request
N, C = int(sys.argv[1]), int(sys.argv[2]); MODE = sys.argv[3] if len(sys.argv) > 3 else "reuse"
LINE = "Line {i}: the inspection of pump station {s} found pressure {p} kPa and flow {f} litres per second.\n"
OFF = int(__import__("os").environ.get("DOC_OFFSET", "0"))   # distinct documents per repeated try (no cross-try cache hits)
def doc(r):
    r += OFF
    return f"Session {r}.\n" + "".join(LINE.format(i=i, s=(i * 7 + r) % 97, p=300 + (i * 13 + r) % 211, f=20 + (i * 3 + r) % 57)
                                       for i in range(N // 26))
def stream(prompt, max_tokens, seed):
    body = {"model": "inkling", "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.7,
            "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); times = []; usage = {}
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); usage = d.get("usage") or usage
        if any(c.get("text") for c in d.get("choices", [])): times.append(time.time())
    k = usage.get("completion_tokens", 0); ttft = (times[0] - t0) if times else float("nan")
    tps = (k - 1) / (times[-1] - times[0]) if len(times) > 1 and k > 1 else 0.0
    return ttft, tps, times, usage.get("prompt_tokens", 0)
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")
def itls(times): return [b - a for a, b in zip(times, times[1:])]
def session(r, turns=3):
    d = doc(r); out = []
    for t in range(turns):
        q = "\nSummarise the log above." if t == 0 else f"\nQuestion {t}: which stations reported flow above {30 + 7 * t}?"
        out.append(stream(d + q, 128 if t == 0 else 256, r * 10 + t))
    return out
if MODE == "reuse":
    with cf.ThreadPoolExecutor(C) as ex: res = list(ex.map(session, range(C)))
    cold = [s[0][0] for s in res]; warm = [t[0] for s in res for t in s[1:]]; dec = [t[1] for s in res for t in s[1:]]
    it = [x for s in res for t in s[1:] for x in itls(t[2])]
    print(f"LDCTX reuse N~{N} c={C} prompt_tokens={res[0][0][3]} cold_ttft_med={pct(cold, .5):.2f}s warm_ttft_med={pct(warm, .5):.2f}s "
          f"decode_tok_s_med={pct(dec, .5):.1f} decode_sum~{pct(dec, .5) * C:.1f} itl_p50={1e3 * pct(it, .5):.1f}ms itl_p99={1e3 * pct(it, .99):.1f}ms"
          f" | ttft per session/turn: {[[round(t[0], 2) for t in s] for s in res]}")
else:
    for r in range(C - 1): stream(doc(r) + "\nSummarise the log above.", 8, r)        # warm the decoders' prefixes
    win = {}
    def decoder(r):
        return stream(doc(r) + "\nQuestion 1: which stations reported flow above 37?", 768, r * 10 + 1)
    def intruder():
        time.sleep(4); win["t0"] = time.time(); x = stream(doc(1000 + N) + "\nSummarise the log above.", 16, 999)
        win["t1"] = win["t0"] + x[0]; win["ttft"] = x[0]
    with cf.ThreadPoolExecutor(C) as ex:
        fs = [ex.submit(decoder, r) for r in range(C - 1)]; fi = ex.submit(intruder)
        res = [f.result() for f in fs]; fi.result()
    t0, t1 = win["t0"], win["t1"]; inn, out = [], []
    for s in res:
        for a, b in zip(s[2], s[2][1:]): (inn if t0 <= b <= t1 + 0.05 else out).append(b - a)
    print(f"LDCTX interfere N~{N} decoders={C - 1} intruder_ttft={win['ttft']:.2f}s | ITL outside p50={1e3 * pct(out, .5):.1f}ms "
          f"p99={1e3 * pct(out, .99):.1f}ms | inside prefill window p50={1e3 * pct(inn, .5):.1f}ms p99={1e3 * pct(inn, .99):.1f}ms "
          f"max={1e3 * max(inn or [float('nan')]):.0f}ms n_inside={len(inn)}")
