#!/usr/bin/env python3
"""post_live.py -- checks to run AFTER Inkling is live, through any OpenAI-compatible endpoint (stdlib only).
Env: BASE (default http://127.0.0.1:8000), MODEL (default inkling), API_KEY (optional bearer token), REASONING=0 (tools: reasoning off).

  ttft                      one streamed chat: time to first reasoning / content delta, number of content deltas
  tps [C=1]                 C concurrent chats of 512 tokens: aggregate and per-request tokens/s
  tools [N=20]              N requests each with tool_choice auto / named / required: well-formed calls to the right function
  text                      4 short questions with known answers (capital, 17x23, a reading question, a list): is the text correct?
  json [N=20]               N requests with response_format (json_schema and json_object alternate): valid JSON with the right content
  long [TOKENS=150000]      one prompt of about TOKENS tokens with a code planted in the middle: is the code found?
  soak [N=32]               N short chats, 8 at a time: every answer on topic, total tokens/s
  (text, json, tools: run once as is = reasoning ON, once with REASONING=0)
  docqa FILE.jsonl [C=1] [N]  real document questions (JSONL), C at a time: TTFT, decode tok/s, aggregate, draft acceptance
  kl-collect OUT FILE...    next-token top-20 log-probabilities over the text in FILE... (teacher forced), saved to OUT
  kl-diff A.json B.json     compare two kl-collect outputs (e.g. original weights vs NVFP4 layers 40-47)
Each prints one result line starting with the check name.
"""
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "inkling")
# REASONING=0: ask for no reasoning (reasoning_effort 0 through the chat template); default leaves the model's reasoning on
NOREASON = {"chat_template_kwargs": {"reasoning_effort": 0}} if os.environ.get("REASONING") == "0" else {}
HDR = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDR["Authorization"] = "Bearer " + os.environ["API_KEY"]
CALC = {"type": "function", "function": {"name": "run_calc_check", "description": "Recompute a total from line items and compare it with the stated total.",
        "parameters": {"type": "object", "properties": {"line_items": {"type": "array", "items": {"type": "number"}}, "stated_total": {"type": "number"}},
                       "required": ["line_items", "stated_total"]}}}
NOTE = {"type": "function", "function": {"name": "save_note", "description": "Save a short note.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
ASK = "The invoice lists 120.50, 80.25 and 19.25 and states a total of 220.00. Check the arithmetic."


def post(path, body, stream=False):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(), HDR), timeout=1800)
    return r if stream else json.load(r)


def ttft():
    t0 = time.time()
    first = {}
    n = 0
    r = post("/v1/chat/completions", {"model": MODEL, "stream": True, "max_tokens": 400,
                                      "messages": [{"role": "user", "content": "Explain in about 200 words how a heat pump works."}]}, True)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for c in json.loads(line[6:]).get("choices", []):
            d = c.get("delta", {})
            if d.get("reasoning") or d.get("reasoning_content"):
                first.setdefault("reasoning", time.time() - t0)
            if d.get("content"):
                first.setdefault("content", time.time() - t0)
                n += 1
    total = time.time() - t0
    ok = n >= 5 and first.get("content", total) < 0.8 * total
    print(f"ttft {'PASS' if ok else 'FAIL'} first_reasoning={first.get('reasoning', float('nan')):.2f}s first_content={first.get('content', float('nan')):.2f}s "
          f"total={total:.2f}s content_deltas={n} (FAIL = the answer arrived in one burst at the end)")


def tps(c):
    def one(i):
        t = time.time()
        r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 512, "seed": i,
                                          "messages": [{"role": "user", "content": f"Write a detailed 400-word essay about topic number {i}: rivers, bridges, clocks or maps."}]})
        return r["usage"]["completion_tokens"], time.time() - t
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(one, range(c)))
    wall = time.time() - t0
    print(f"tps c={c} aggregate={sum(n for n, _ in res) / wall:.1f} tok/s per_request={[round(n / s, 1) for n, s in res]} wall={wall:.1f}s (includes prefill)")


def _spec_counters():
    """vLLM's spec-decode counters from METRICS_BASE/metrics (default BASE; the gateway may not forward /metrics:
    then point METRICS_BASE at the container, e.g. http://vllm-inkling:8000). {} when unreachable or spec decode is off."""
    try:
        txt = urllib.request.urlopen(os.environ.get("METRICS_BASE", BASE).rstrip("/") + "/metrics", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_total" in line.split(" ")[0]:
            k, v = line.rsplit(" ", 1)
            k = k.split("{")[0]
            out[k] = out.get(k, 0.0) + float(v)
    return out


def docqa(path, c, n):
    """N requests of the JSONL file's prompts (cycled), C at a time, streamed. Each line: {"messages": [...]} or
    {"document": "...", "question": "..."}. Sampling = the server's defaults (no temperature sent). Reports TTFT and
    decode tokens/s per request (median), aggregate tokens/s, and the draft acceptance over the run if /metrics answers."""
    prompts = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            prompts.append(p["messages"] if "messages" in p else
                           [{"role": "user", "content": f"Answer the question from the document.\n\nDocument:\n{p['document']}\n\nQuestion: {p['question']}"}])
    if not prompts:
        sys.exit(f"docqa: no prompts in {path}")
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))

    def one(i):
        t0 = time.time()
        first = None
        usage = {}
        r = post("/v1/chat/completions", {"model": MODEL, "stream": True, "max_tokens": max_tokens, "stream_options": {"include_usage": True},
                                          "messages": prompts[i % len(prompts)], **NOREASON}, True)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if first is None and (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")):
                    first = time.time()
        t1 = time.time()
        k = usage.get("completion_tokens", 0)
        dec = (k - 1) / (t1 - first) if first and k > 1 and t1 > first else 0.0
        return (first or t1) - t0, dec, k, usage.get("prompt_tokens", 0)

    before = _spec_counters()
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    after = _spec_counters()
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get("vllm:spec_decode_num_drafts_total", 0)
    acc = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    spec = (f"acceptance_length={1 + acc / drafts:.2f} (drafts={int(drafts)} accepted={int(acc)})" if drafts else
            "acceptance_length=n/a (spec decode off, or /metrics not reachable: set METRICS_BASE, or read 'Mean acceptance length' in docker logs)")
    print(f"docqa c={c} n={n} {RSN} prompt_tokens_median={med([r[3] for r in res])} ttft_median={med([r[0] for r in res]):.2f}s "
          f"decode_tok_s_median={med([r[1] for r in res]):.1f} aggregate_tok_s={sum(r[2] for r in res) / wall:.1f} wall={wall:.1f}s {spec}")


def tools(n):
    out = {}
    for name, choice in (("auto", "auto"), ("named", {"type": "function", "function": {"name": "run_calc_check"}}), ("required", "required")):
        ok = 0
        bad = []
        for i in range(n):
            try:
                m = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 1024, "seed": 100 + i, "tools": [CALC, NOTE], "tool_choice": choice, **NOREASON,
                                                  "messages": [{"role": "user", "content": ASK}]})["choices"][0]["message"]
                calls = m.get("tool_calls") or []
                a = json.loads(calls[0]["function"]["arguments"]) if calls else None
                good = bool(calls) and calls[0]["function"]["name"] == "run_calc_check" and isinstance(a, dict) and "stated_total" in a and isinstance(a.get("line_items"), list)
                if name == "required":
                    good = bool(calls) and isinstance(a, dict)
                ok += good
                good or bad.append((m.get("content") or "")[:60] if not calls else calls[0]["function"]["name"])
            except Exception as e:  # noqa: BLE001
                bad.append(f"error {str(e)[:60]}")
        out[name] = ok
        print(f"tools {name}: {ok}/{n} well-formed" + (f"; first miss: {bad[0]!r}" if bad else ""))
    print(f"tools reasoning={'off' if NOREASON else 'on'} {'PASS' if out['named'] == n and out['required'] == n else 'FAIL'} auto={out['auto']}/{n} named={out['named']}/{n} required={out['required']}/{n} "
          "(named/required below N means tool_choice is not enforced by the server: a forced calc-check can be skipped)")


RSN = f"reasoning={'off' if NOREASON else 'on'}"
PROBES = [  # (name, prompt, words that must all be in the answer, lower case)
    ("capital", "What is the capital of France? Answer in one short sentence.", ["paris"]),
    ("arithmetic", "What is 17 times 23? Give the number.", ["391"]),
    ("reading", "Read this note and answer the question.\n\nNote: The maintenance window for pump station 4 was moved from Monday to Thursday "
                "because the replacement valve arrives on Wednesday. Karen Ostrowski approved the change.\n\nQuestion: On which day is the "
                "maintenance window now, and who approved it?", ["thursday", "ostrowski"]),
    ("list", "List the days of the week that start with the letter T.", ["tuesday", "thursday"]),
]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}, "population_millions": {"type": "number"}},
          "required": ["city", "country", "population_millions"], "additionalProperties": False}


def answer(messages, **kw):
    """-> (content, finish_reason, usage)"""
    r = post("/v1/chat/completions", {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 2048), **NOREASON, **kw})
    return (r["choices"][0]["message"].get("content") or ""), r["choices"][0].get("finish_reason"), r.get("usage", {})


def text():
    good = 0
    for name, prompt, want in PROBES:
        try:
            c, fin, _ = answer([{"role": "user", "content": prompt}], temperature=0)
            miss = [w for w in want if w not in c.lower()]
            ok = not miss and fin == "stop"
            print(f"text {name}: {'ok' if ok else 'MISS'} finish={fin} expected {want} answer={c.strip()[:160]!r}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"text {name}: ERROR {str(e)[:200]}")
        good += ok
    print(f"text {RSN} {'PASS' if good == len(PROBES) else 'FAIL'} {good}/{len(PROBES)} correct answers that ended normally")


def json_mode(n):
    def one(i):
        rf = {"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA}} if i % 2 == 0 else {"type": "json_object"}
        try:
            c, fin, _ = answer([{"role": "user", "content": "Give the largest city of Japan with its country and population in millions as JSON "
                                 "with the keys city, country, population_millions."}], response_format=rf, seed=700 + i)
            d = json.loads(c)
            return "ok" if isinstance(d, dict) and "tokyo" in json.dumps(d).lower() else f"wrong content: {c[:80]}"
        except Exception as e:  # noqa: BLE001
            return f"error: {str(e)[:140]}"
    with cf.ThreadPoolExecutor(4) as ex:
        out = list(ex.map(one, range(n)))
    bad = [o for o in out if o != "ok"]
    print(f"json {RSN} {'PASS' if not bad else 'FAIL'} {n - len(bad)}/{n} valid JSON naming Tokyo" + (f"; first miss: {bad[0]!r}" if bad else ""))


def long_prompt(tokens):
    lines = max(tokens // 26, 50)          # one log line is about 26 tokens; the reply reports the real prompt_tokens
    code = "7391-KESTREL"
    body = []
    for i in range(lines):
        body.append(f"Log entry {i:06d}: sensor {i % 97} at station {i % 13} reported value {(i * 37) % 1000} with status nominal.")
        if i == lines // 2:
            body.append(f"Log entry special: the access code for the archive room is {code}.")
    prompt = "Below is a long log. One entry states an access code.\n\n" + "\n".join(body) + "\n\nWhat is the access code for the archive room? Answer with the code only."
    t = time.time()
    c, fin, u = answer([{"role": "user", "content": prompt}], temperature=0, max_tokens=1024)
    print(f"long {RSN} {'PASS' if code.lower() in c.lower() else 'FAIL'} prompt_tokens={u.get('prompt_tokens')} finish={fin} wall={time.time() - t:.1f}s "
          f"expected {code!r} answer={c.strip()[:80]!r}")


def soak(n):
    topics = ["rivers", "bridges", "clocks", "maps", "railways", "lighthouses", "glaciers", "printing"]

    def one(i):
        w = topics[i % 8]
        try:
            c, fin, u = answer([{"role": "user", "content": f"Write three sentences about {w}. Use the word {w} at least once."}], max_tokens=400, seed=i)
            return (w[:-1] in c.lower()), u.get("completion_tokens", 0), f"finish={fin} answer={c.strip()[:60]!r}"
        except Exception as e:  # noqa: BLE001
            return False, 0, f"error: {str(e)[:120]}"
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(one, range(n)))
    wall = time.time() - t0
    bad = [o[2] for o in out if not o[0]]
    print(f"soak {RSN} {'PASS' if not bad else 'FAIL'} {n - len(bad)}/{n} on-topic answers, 8 at a time, {sum(o[1] for o in out) / wall:.0f} tokens/s total, "
          f"wall={wall:.1f}s" + (f"; first miss: {bad[0]}" if bad else ""))


def kl_collect(out, files):
    text = "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    chunks = [text[i:i + 3000] for i in range(0, min(len(text), 3000 * 64), 3000)]
    rows = []
    for ch in chunks:
        r = post("/v1/completions", {"model": MODEL, "prompt": ch, "max_tokens": 1, "prompt_logprobs": 20, "temperature": 0})
        pl = r["choices"][0].get("prompt_logprobs") or []
        rows.append([{k: [v["logprob"], v.get("decoded_token")] for k, v in p.items()} if p else None for p in pl])
    json.dump({"model": MODEL, "base": BASE, "rows": rows}, open(out, "w"))
    print(f"kl-collect chunks={len(rows)} positions={sum(len(r) for r in rows)} -> {out}")


def kl_diff(a, b):
    A, B = json.load(open(a))["rows"], json.load(open(b))["rows"]
    n = agree = 0
    kl = dl = 0.0
    worst = 0.0
    for ra, rb in zip(A, B):
        for pa, pb in zip(ra, rb):
            if not pa or not pb:
                continue
            ta = sorted(pa.items(), key=lambda kv: -kv[1][0])[:20]
            tb = sorted(pb.items(), key=lambda kv: -kv[1][0])[:20]
            floor = min(v[0] for _, v in tb)
            za = math.log(sum(math.exp(v[0]) for _, v in ta))
            # both sides renormalised over A's top-20 tokens (a token missing from B's list gets B's smallest listed logprob)
            lb = [pb[t][0] if t in pb else floor for t, _ in ta]
            zb = math.log(sum(math.exp(x) for x in lb))
            k = sum(math.exp(v[0] - za) * ((v[0] - za) - (x - zb)) for (t, v), x in zip(ta, lb))
            n += 1
            agree += ta[0][0] == tb[0][0]
            kl += max(k, 0.0)
            worst = max(worst, k)
            dl += abs(ta[0][1][0] - (pb[ta[0][0]][0] if ta[0][0] in pb else floor))
    print(f"kl-diff positions={n} top1_agreement={agree / max(n, 1):.4f} mean_kl_top20~={kl / max(n, 1):.5f} nats max_kl~={worst:.3f} "
          f"mean_abs_dlogprob_top1={dl / max(n, 1):.4f} (approximate: top-20 only; same model twice should give 1.0000 / 0)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ttft":
        ttft()
    elif cmd == "tps":
        tps(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif cmd == "tools":
        tools(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "text":
        text()
    elif cmd == "json":
        json_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "long":
        long_prompt(int(sys.argv[2]) if len(sys.argv) > 2 else 150000)
    elif cmd == "soak":
        soak(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    elif cmd == "docqa":
        c = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        docqa(sys.argv[2], c, int(sys.argv[4]) if len(sys.argv) > 4 else max(8, 2 * c))
    elif cmd == "kl-collect":
        kl_collect(sys.argv[2], sys.argv[3:])
    elif cmd == "kl-diff":
        kl_diff(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
