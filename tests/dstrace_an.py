"""dstrace_an.py TRACE.jsonl [...] -- draft-ahead hit rates from an INKLING2_DSPARK_TRACE log.

Each line (one step, TP rank 0): ids, qlen/rej of the verify that just finished, anchor (= that verify's sampled bonus
token), drafts (the K tokens proposed now) and top3 (drafter top-3 per position). Joined per request id: the drafts of
step n against step n+1's accept count a = qlen - 1 - rej (decode steps only, qlen == K + 1) and bonus b.

Prints:
  accept-length distribution, mean accepted tokens per step (a + 1)
  A  accept-all: hit = a == K (for a K-1 run, the K-th draft doubles as the bonus guess: the measurable form)
  B  per-likely-length, L lengths x G bonus guesses: hit = a in the L most frequent lengths and b in the drafter's top-G at
     position a+1 other than the proposed token (a < K; a == K needs a bonus guess past the block: counted as a miss)
  C  anchor depth m: P(a >= m) (upper bound) and a proxy: a >= m and (a == K or b in top-3 at a+1 minus the draft token)
     (the real C needs the drafter re-run from d_m: the stale-context replay)
  batch: per step, the fraction of steps where EVERY request hit (what a weight-bound redraft needs), by batch size."""
import collections, json, sys

recs = []
for f in sys.argv[1:]:
    with open(f) as fh:
        recs += [json.loads(ln) for ln in fh if ln.strip()]
recs.sort(key=lambda r: r["t"])
K = recs[0]["k"] if recs else 0
last = {}            # req id -> (step, drafts, top3) of its previous logged step
steps = []           # per record: list of (a, b, drafts, top3) for requests with a decode verify of the previous drafts
LIGHT = any(r.get("top3") is None for r in recs)   # production light trace: no top-3 (B and the C proxy need it)
for r in recs:
    row = []
    st = r.get("step")
    for i, rid in enumerate(r["ids"]):
        prev = last.get(rid)
        if prev and r["qlen"][i] == K + 1 and (st is None or prev[0] is None or st == prev[0] + 1):
            _, d, t = prev
            row.append((K - r["rej"][i], r["anchor"][i], d, t or [[None] * 3] * K))
        last[rid] = (st, r["drafts"][i], None if r.get("top3") is None else r["top3"][i])
    if row:
        steps.append(row)
obs = [x for s in steps for x in s]
if not obs:
    sys.exit("no joined decode steps")
n = len(obs)
hist = collections.Counter(a for a, *_ in obs)
print(f"K={K}: {n} request-steps in {len(steps)} steps; mean accepted per step (a+1) {sum(a + 1 for a, *_ in obs) / n:.2f}")
print("P(a):  " + "  ".join(f"{a}:{hist[a] / n:.3f}" for a in range(K + 1)))
print("P(a>=m): " + "  ".join(f"m={m}:{sum(v for a, v in hist.items() if a >= m) / n:.3f}" for m in range(1, K + 1)))


def bonus_in_top(a, b, d, t, g):
    if a >= K:
        return False
    alts = [x for x in t[a] if x != d[a]][:g]
    return b in alts


def hit_A(a, b, d, t):
    return a == K


order = [a for a, _ in hist.most_common()]


def hit_B(L, G):
    keep = set(order[:L])
    return lambda a, b, d, t: a in keep and bonus_in_top(a, b, d, t, G)


def hit_C(m, proxy):
    if not proxy:
        return lambda a, b, d, t: a >= m
    return lambda a, b, d, t: a >= m and (a == K or bonus_in_top(a, b, d, t, 3))


def report(name, fn):
    p = sum(fn(*x) for x in obs) / n
    by = collections.defaultdict(lambda: [0, 0])
    for s in steps:
        by[len(s)][0] += all(fn(*x) for x in s)
        by[len(s)][1] += 1
    bs = "  ".join(f"c={c}:{h / t:.3f}(n={t})" for c, (h, t) in sorted(by.items()) if t >= 5)
    # independence model for batch sizes not observed
    ind = "  ".join(f"c={c}:{p ** c:.3f}" for c in (1, 2, 4, 8))
    print(f"{name:34s} per-request {p:.3f} | all-hit observed {bs} | p^c {ind}")


print("\n-- draft-ahead hit rates" + (" (light trace: no top-3, B and C-proxy rows are not meaningful)" if LIGHT else ""))
report("A accept-all (K-1 run)", hit_A)
for L in ((1, 2, 3) if not LIGHT else ()):
    for G in (1, 2):
        report(f"B lengths={L} bonus-guesses={G} (C={L * G})", hit_B(L, G))
for m in (1, 2, 3, 4):
    report(f"C anchor m={m} upper bound", hit_C(m, False))
    report(f"C anchor m={m} proxy", hit_C(m, True))
