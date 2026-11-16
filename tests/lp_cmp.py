"""lp_cmp.py collect OUT.json | compare A.json B.json [C.json ...]
collect: fixed prompts against :8000 /v1/completions, greedy (a NUMERICS test only, never a serving setting), with the chosen
tokens' logprobs and the prompt's teacher-forced logprobs (prompt_logprobs=1).
compare: identical generated tokens, |dlogprob| on the shared prefix and on every prompt position, B.. vs A."""
import json, sys, urllib.request
B = "http://127.0.0.1:8000"
DOC = ("The bridge was completed in 1932 after six years of work. It carries eight lanes of traffic, two railway lines and a footpath. "
       "The main span is 503 metres long and the arch rises 134 metres above the water. ")
PROMPTS = [
    "Read this note and answer the question.\n\nNote: The maintenance window for pump station 4 was moved from Monday to Thursday "
    "because the replacement valve arrives on Wednesday. Karen Ostrowski approved the change.\n\nQuestion: On which day is it now?",
    "Write a short paragraph about how rivers shape valleys.",
    "Summarise the following document in three sentences.\n\n" + DOC * 20,                 # ~1.2k tokens
    "Here is a long log.\n" + (DOC * 150) + "\nWhat year was the bridge completed?",        # ~9k tokens: split-KV range for decode
    "List the prime numbers below 60.",
    "def parse(line):\n    # parse 'key=value' pairs separated by semicolons\n",
]
def post(body):
    r = urllib.request.Request(B + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())
def collect(out):
    res = []
    for p in PROMPTS:
        d = post({"model": "inkling", "prompt": p, "max_tokens": 128, "temperature": 0, "seed": 0, "logprobs": 1,
                  "prompt_logprobs": 1, "return_token_ids": True, "ignore_eos": True})
        c = d["choices"][0]
        lp = c["logprobs"]["token_logprobs"]
        # per prompt position: {token_id: logprob} for the tokens vLLM returns (the prompt token + top-1)
        plp = [None if x is None else {k: v["logprob"] for k, v in x.items()}
               for x in (c.get("prompt_logprobs") or d.get("prompt_logprobs") or [])]
        res.append({"ids": c.get("token_ids"), "lp": lp, "plp": plp, "text": c["text"][:80]})
    json.dump(res, open(out, "w")); print("collected", len(res), "prompts ->", out)
def compare(a, others):
    A = json.load(open(a))
    for o in others:
        X = json.load(open(o)); same = 0; gmax = 0.0; pmax = 0.0; pmean = []; firsts = []
        for x, y in zip(A, X):
            n = next((i for i, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), len(x["ids"]))
            same += x["ids"] == y["ids"]
            if x["ids"] != y["ids"]: firsts.append(n)
            gmax = max([gmax] + [abs(u - v) for u, v in zip(x["lp"][:n], y["lp"][:n]) if u is not None and v is not None])
            d = [abs(u[k] - v[k]) for u, v in zip(x["plp"], y["plp"]) if u and v for k in u if k in v]
            if d: pmax = max(pmax, max(d)); pmean += d
        print(f"{o} vs {a}: identical generations {same}/{len(A)} first-divergence {firsts} | max|dlogprob| generated(shared prefix) {gmax:.4f} "
              f"| prompt teacher-forced: max {pmax:.4f} mean {sum(pmean) / max(1, len(pmean)):.5f} over {len(pmean)} positions")
if __name__ == "__main__":
    collect(sys.argv[2]) if sys.argv[1] == "collect" else compare(sys.argv[2], sys.argv[3:])
