"""slice_build.py OUT_DIR [--local DIR] [--manifest-out M | --manifest-in M] [--threads 6] [--max-mbps 0]

Builds the two test slices of Inkling-2-Small: only the tensors of decoder layers 0-6 plus the non-layer tensors (embedding,
LM head, norms, vision tower) go into new, smaller safetensors files, plus the DSpark draft (mtp.safetensors) and the small
config/tokenizer files.

  OUT_DIR/ckpt   6 layers (num_hidden_layers 6)
  OUT_DIR/ckpt7  7 layers + the DSpark draft with target_layer_ids 0..7 (a truncated target: drafts are never accepted;
                 this slice measures the drafter's COST, not its acceptance)

Source: --local DIR = a full Inkling-2-Small checkpoint already on disk (the original thinkingmachines/Inkling-2-Small or
the NVFP4 requant btbtyler09/Inkling-2-Small-NVFP4: layers 0-6, the embeddings, head, vision tower and draft are the same
bytes in both). Anything missing there is fetched from the requant on Hugging Face (pinned revision) with HTTP range
requests, never whole shards (about 57 GB if nothing is local).

Integrity: every tensor written (sliced layers and draft) is checked against a per-tensor sha256 manifest
(slice_manifest.json next to this script), whichever source it came from. --manifest-out writes that manifest instead.
Small config/tokenizer files are checked against the requant's SHA256SUMS when downloaded; local ones are reported.
Re-running skips finished files."""
import argparse, concurrent.futures as cf, hashlib, json, os, re, shutil, struct, sys, threading, time, urllib.request

REPO = "btbtyler09/Inkling-2-Small-NVFP4"
REV = "0a7810367526879b86c724f0c22587b7d11fec26"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
UA = {"User-Agent": "inkling-slice/1.0"}
LAYERS = 7
SMALL = ["config.json", "hf_quant_config.json", "chat_template.jinja", "processor_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "tiktoken/tokenizer.model"]
CHUNK = 1 << 24
_rate_lock = threading.Lock()
_rate = {"t0": time.time(), "bytes": 0, "max": 0.0}


def throttle(n):
    if not _rate["max"]:
        return
    with _rate_lock:
        _rate["bytes"] += n
        ahead = _rate["bytes"] / (_rate["max"] * 1e6) - (time.time() - _rate["t0"])
    if ahead > 0:
        time.sleep(ahead)


def http(url, start=None, end=None, tries=6):
    h = dict(UA)
    if start is not None:
        h["Range"] = f"bytes={start}-{end}"
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)
        except Exception as e:  # noqa: BLE001
            print(f"retry {url.rsplit('/', 1)[-1]} {start}: {e}", file=sys.stderr)
            time.sleep(2 + 3 * i)
    sys.exit(f"FAILED {url}")


class Source:
    """A checkpoint: a local directory, or the pinned Hugging Face revision."""

    def __init__(self, local=None):
        self.local = local
        self._hdr = {}

    def name(self):
        return f"local {self.local}" if self.local else f"hf {REPO}@{REV[:8]}"

    def has(self, f):
        return os.path.isfile(os.path.join(self.local, f)) if self.local else True

    def open_range(self, f, start, end):  # inclusive end; returns a file-like object
        if self.local:
            fh = open(os.path.join(self.local, f), "rb")
            fh.seek(start)
            return _Limited(fh, end - start + 1)
        return http(BASE + f, start, end)

    def read_all(self, f):
        if self.local:
            with open(os.path.join(self.local, f), "rb") as fh:
                return fh.read()
        with http(BASE + f) as r:
            return r.read()

    def header(self, f):
        if f not in self._hdr:
            with self.open_range(f, 0, 7) as r:
                n = struct.unpack("<Q", r.read())[0]
            with self.open_range(f, 8, 7 + n) as r:
                self._hdr[f] = (json.loads(r.read()), 8 + n)
        return self._hdr[f]

    def weight_map(self):
        return json.loads(self.read_all("model.safetensors.index.json"))["weight_map"]


class _Limited:
    def __init__(self, fh, n):
        self.fh, self.left = fh, n

    def read(self, n=-1):
        n = self.left if n < 0 else min(n, self.left)
        b = self.fh.read(n)
        self.left -= len(b)
        return b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.fh.close()


def llm_layer(key):
    """Decoder layer index of a key, None for non-decoder tensors (embeddings, head, norms, the vision tower's own layers)."""
    m = re.match(r"model\.llm\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def wanted(key, n=LAYERS):
    i = llm_layer(key)
    return i is None or i < n


HASH_ONLY = False   # --hash-only: read and hash every tensor, write nothing (manifest runs, source checks)


def write_slice(out, items, manifest, record):
    """items: [(key, source, file)] -> one safetensors file at out, every tensor hashed (and checked unless record)."""
    if os.path.exists(out) and not record and not HASH_ONLY:
        return out  # the .part is renamed only after every tensor matched
    hdrs = [(k, s, f, *s.header(f)) for k, s, f in items]
    new, off = {}, 0
    for k, s, f, h, _ in hdrs:
        a, b = h[k]["data_offsets"]
        new[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(os.devnull if HASH_ONLY else out + ".part", "wb") as o:
        o.write(struct.pack("<Q", len(hb))); o.write(hb)
        for k, s, f, h, d0 in hdrs:
            a, b = h[k]["data_offsets"]
            dig, got = hashlib.sha256(), 0
            with s.open_range(f, d0 + a, d0 + b - 1) as r:
                while True:
                    x = r.read(CHUNK)
                    if not x:
                        break
                    if not s.local:
                        throttle(len(x))
                    dig.update(x); o.write(x); got += len(x)
            if got != b - a:
                sys.exit(f"SHORT_READ {k} from {s.name()} {f}: {got} of {b - a}")
            if record:
                manifest[k] = dig.hexdigest()
            elif manifest.get(k) != dig.hexdigest():
                sys.exit(f"SHA_MISMATCH {k} from {s.name()} {f}: this checkpoint's tensor differs from the test manifest")
    if not HASH_ONLY:
        os.rename(out + ".part", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--local", help="a full Inkling-2-Small checkpoint dir already on disk (HF is the fallback)")
    ap.add_argument("--manifest-out")
    ap.add_argument("--manifest-in", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "slice_manifest.json"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-mbps", type=float, default=0.0, help="cap the download rate in MB/s (0 = no cap)")
    ap.add_argument("--hash-only", action="store_true", help="hash every tensor (record or check) without writing slices")
    a = ap.parse_args()
    global HASH_ONLY
    HASH_ONLY = a.hash_only
    _rate["max"] = a.max_mbps
    record = a.manifest_out is not None
    man = {} if record else json.load(open(a.manifest_in))
    manifest = {} if record else man["tensors"]
    hf = Source()
    loc = Source(a.local) if a.local else None
    if loc and not loc.has("model.safetensors.index.json"):
        sys.exit(f"--local {a.local}: no model.safetensors.index.json there")
    src = os.path.join(a.out, "src")
    os.makedirs(os.path.join(src, "tiktoken"), exist_ok=True)
    t0 = time.time()
    sums = None
    for f in SMALL:   # small files: local copy if present (reported), else downloaded and checked
        p = os.path.join(src, f)
        if os.path.exists(p):
            continue
        if loc and loc.has(f):
            shutil.copyfile(os.path.join(a.local, f), p)
            print(f"file {f}: local copy, sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]}")
        else:
            if sums is None:
                with http(BASE + "SHA256SUMS") as r:
                    sums = {ln.split()[1]: ln.split()[0] for ln in r.read().decode().splitlines() if ln.strip()}
            b = hf.read_all(f)
            if hashlib.sha256(b).hexdigest() != sums[f]:
                sys.exit(f"SHA_MISMATCH {f}")
            open(p, "wb").write(b)
    # tensors: from the local checkpoint where it has them, else from HF; grouped by the HF shard layout
    hf_wm = hf.weight_map() if record or not loc else None
    loc_wm = loc.weight_map() if loc else {}
    keys = sorted(k for k in (loc_wm if loc else hf_wm) if wanted(k))
    if loc and any(k not in loc_wm for k in manifest if not k.startswith("mtp:")):
        hf_wm = hf.weight_map()
        keys = sorted(set(keys) | {k for k in manifest if not k.startswith("mtp:")})
    groups = {}
    for k in keys:
        if loc and k in loc_wm:
            groups.setdefault("slice-" + loc_wm[k], []).append((k, loc, loc_wm[k]))
        else:
            groups.setdefault("slice-hf-" + hf_wm[k], []).append((k, hf, hf_wm[k]))
    msrc = loc if loc and loc.has("mtp.safetensors") else hf
    mh, _ = msrc.header("mtp.safetensors")
    mtp_items = [(k, msrc, "mtp.safetensors") for k in sorted(x for x in mh if x != "__metadata__")]
    mman = {k[4:]: v for k, v in manifest.items() if k.startswith("mtp:")} if not record else {}

    def do_mtp():
        out = os.path.join(src, "mtp.safetensors")
        res = write_slice(out, mtp_items, mman, record)
        if record:
            manifest.update({"mtp:" + k: v for k, v in mman.items()})
        return res

    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(write_slice, os.path.join(src, g), items, manifest, record) for g, items in sorted(groups.items())]
        futs.append(ex.submit(do_mtp))
        outs = [f.result() for f in futs]
    if not record:
        missing = [k for k in manifest if not k.startswith("mtp:") and k not in keys]
        if missing:
            sys.exit(f"MISSING {len(missing)} manifest tensors, e.g. {missing[:3]}")
    new_wm = {k: g for g, items in groups.items() for k, _, _ in items}
    full = json.load(open(os.path.join(src, "config.json")))
    for name, n, dspark in (() if HASH_ONLY else (("ckpt", 6, False), ("ckpt7", 7, True))):
        d = os.path.join(a.out, name)
        os.makedirs(os.path.join(d, "tiktoken"), exist_ok=True)
        for f in [x for x in SMALL if x != "config.json"] + [os.path.basename(o) for o in outs]:
            if not os.path.lexists(os.path.join(d, f)):
                os.symlink(os.path.relpath(os.path.join(src, f), os.path.dirname(os.path.join(d, f))), os.path.join(d, f))
        c = json.loads(json.dumps(full)); t = c["text_config"]
        t["num_hidden_layers"] = n; t["local_layer_ids"] = [i for i in t["local_layer_ids"] if i < n]
        if dspark:
            c["mtp_config"]["target_layer_ids"] = list(range(8))
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=1)
        json.dump({"metadata": {}, "weight_map": {k: v for k, v in new_wm.items() if wanted(k, n)}},
                  open(os.path.join(d, "model.safetensors.index.json"), "w"))
    if record:
        json.dump({"repo": REPO, "revision": REV, "layers": LAYERS, "tensors": manifest}, open(a.manifest_out, "w"), indent=0)
    gb = 0.0 if HASH_ONLY else sum(os.path.getsize(o) for o in outs) / 1e9
    nloc = sum(1 for items in groups.values() for _, s, _ in items if s is loc)
    print(f"SLICE_OK {a.out}: {len(new_wm)} tensors ({nloc} from {loc.name() if loc else 'no local dir'}, {len(new_wm) - nloc} "
          f"from {hf.name()}) + draft from {msrc.name()}, {gb:.1f} GB, {time.time() - t0:.0f} s"
          + (f"; manifest -> {a.manifest_out}" if record else "; every tensor matched the manifest"))


if __name__ == "__main__":
    main()
