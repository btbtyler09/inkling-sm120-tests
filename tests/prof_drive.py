"""prof_drive.py BASE MODEL C N_TOKENS -- profile window against a live vLLM server started with --profiler-config:
warm 2 requests, POST /start_profile, C concurrent requests of exactly N_TOKENS generated tokens (ignore_eos, temperature
0.7), POST /stop_profile. Prints the generated token count and the window wall time. stdlib only."""
import json, sys, threading, time, urllib.request

BASE, MODEL, C, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PROMPT = "Write a long, detailed essay about the history of bridges, covering materials, engineering and famous examples."


def post(path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def gen(n, out, i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT + f" (part {i})"}], "max_tokens": n,
            "temperature": 0.7, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    out[i] = json.loads(post("/v1/chat/completions", body))["usage"]["completion_tokens"]


for i in range(2):
    gen(64, {}, i)
post("/start_profile")
out = {}
t = time.time()
th = [threading.Thread(target=gen, args=(N, out, i)) for i in range(C)]
[x.start() for x in th]
[x.join() for x in th]
wall = time.time() - t
post("/stop_profile", timeout=1200)
print(f"PROF c={C} tokens={sum(out.values())} per_req={list(out.values())} wall={wall:.2f}s")
