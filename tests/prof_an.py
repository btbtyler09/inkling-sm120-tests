"""prof_an.py summarize DIR STEPS [TOP] | compare DIR_A STEPS_A DIR_B STEPS_B [TOP]

summarize: GPU kernels of the torch-profiler trace in DIR (newest *.json / *.json.gz), grouped by name: per-step us (total /
STEPS), launches per step, share. Also GPU busy per step and the profile window per step (busy vs idle = host overhead).
compare: per-step us by kernel name B minus A (e.g. A = no draft, B = DSpark): the kernels the drafter adds, sorted.
STEPS = decode steps in the window (on the 7-layer slice DSpark acceptance is 0, so steps = generated tokens / C)."""
import glob, gzip, json, os, re, sys
from collections import defaultdict


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True), key=os.path.getmtime)
    fs = [f for f in fs if f.endswith((".json", ".json.gz"))]
    if not fs:
        sys.exit(f"no trace in {d}")
    f = fs[-1]
    with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
        tr = json.load(fh)
    ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    return f, ev


def short(n):
    n = re.sub(r"^void ", "", n)
    if not re.match(r"(std::enable_if|cutlass|cute|flashinfer|marlin)", n):
        n = re.sub(r"<.*", "", n)   # plain kernels: the name before the template arguments is enough
    n = re.sub(r"\((int|unsigned|long|float|bool|char|c10|at::)[^)]*\)$", "", n)
    return re.sub(r"\s+", " ", n)[:150]


def table(ev):
    agg = defaultdict(lambda: [0.0, 0])
    for e in ev:
        a = agg[short(e["name"])]
        a[0] += e["dur"]
        a[1] += 1
    t0 = min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ev)
    # GPU busy = union of kernel intervals
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev)
    busy, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, en in iv[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, en
        else:
            ce = max(ce, en)
    busy += ce - cs
    return agg, t1 - t0, busy


def summarize(d, steps, top):
    f, ev = load(d)
    agg, span, busy = table(ev)
    tot = sum(v[0] for v in agg.values())
    print(f"TRACE {f}: {len(ev)} GPU events; window {span / steps:.1f} us/step, GPU busy {busy / steps:.1f} us/step "
          f"({100 * busy / span:.0f}%), kernel sum {tot / steps:.1f} us/step, launches {len(ev) / steps:.1f}/step")
    for n, (us, k) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
        print(f"  {us / steps:9.1f} us/step  {k / steps:6.2f}x  {100 * us / tot:5.1f}%  {n}")
    # one process can drive several GPUs (remote drafter: TP rank 0 + the drafter GPU): split by device
    by_dev = defaultdict(list)
    for e in ev:
        by_dev[(e.get("args") or {}).get("device", e.get("pid"))].append(e)
    if len(by_dev) > 1:
        for dv, de in sorted(by_dev.items(), key=lambda x: str(x[0])):
            da, _, db = table(de)
            dt = sum(v[0] for v in da.values())
            print(f"DEVICE {dv}: GPU busy {db / steps:.1f} us/step, kernel sum {dt / steps:.1f} us/step, launches {len(de) / steps:.1f}/step")
            for n, (us, k) in sorted(da.items(), key=lambda x: -x[1][0])[:8]:
                print(f"    {us / steps:9.1f} us/step  {k / steps:6.2f}x  {n}")
    return agg


def compare(da, sa, db, sb, top):
    fa, ea = load(da)
    fb, eb = load(db)
    aa, spa, ba = table(ea)
    ab, spb, bb = table(eb)
    print(f"A {fa}: window {spa / sa:.1f} us/step busy {ba / sa:.1f}; B {fb}: window {spb / sb:.1f} us/step busy {bb / sb:.1f}")
    print(f"B-A: window {spb / sb - spa / sa:+.1f} us/step, busy {bb / sb - ba / sa:+.1f} us/step, "
          f"launches {len(eb) / sb - len(ea) / sa:+.1f}/step")
    names = set(aa) | set(ab)
    d = {n: (ab.get(n, [0, 0])[0] / sb - aa.get(n, [0, 0])[0] / sa, ab.get(n, [0, 0])[1] / sb - aa.get(n, [0, 0])[1] / sa) for n in names}
    for n, (us, k) in sorted(d.items(), key=lambda x: -x[1][0])[:top]:
        print(f"  {us:+9.1f} us/step  {k:+6.2f}x  {n}")


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 30)
    else:
        compare(sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 30)
