## One image + Nemotron (ink11nemo) — 2026-11-17

**Image.** `ink11nemo-minimal` = `ink11kva-minimal` + 3 changed files (file-level diff of both images, everything else identical):
`vllm/transformers_utils/processors/nano_nemotron_vl.py` (0071: image preprocessing without a compiler; used only by Nemotron 3.5
Super VL), `vllm/entrypoints/openai/chat_completion/stream_processor.py` (0089: a forced tool call streams no text; only with
`VLLM_CUDA_FORCED_TOOL_STREAM_NO_CONTENT=1`, which only `create-nemotron.sh` sets) and `OVERLAY.txt`. Same vllm / torch / triton /
flashinfer / CUDA versions as ink11kva, so **the aux bundles shipped with this kit still match** (checked against their
manifests). `site.env`: one `IMAGE=` + `EXPECT_DIGEST=` for every service (`site.env.example`). `start-all.sh` starts
`vllm-nemotron` right after Inkling when `NEMOTRON_GPUS` is set. Old `NEMOTRON_IMAGE` / `NEMOTRON_DIGEST` lines: delete them.
**r2 (the image to deploy, `sha256:22d7a0cc…`).** One layer on top of ink11nemo-minimal removes three wheels whose bundled
ffmpeg carried HIGH CVEs (exactly the files of their own pip RECORDs; listed in its `OVERLAY.txt`): opencv-python-headless
5.0.0.93 (ffmpeg 8.1.1: 22 Grype HIGH, fixed upstream but no newer wheel), PyAV 19.0.2 (ffmpeg 9.0.2: CVE-2026-107696, published
2026-10-08, no fix released) and PyNvVideoCodec 2.0.4 (unused). Images, text and pooling are unaffected (images go through PIL).
Lost: **video** (a video request gets an HTTP 500 error response, `No module named 'cv2'`; the server stays up) and **audio in
mp4/m4a/aac/webm containers** (HTTP 500 `Please install vllm[audio]`; wav/flac/ogg/mp3 still decode via soundfile). Nemotron keeps
`video:1` in `--limit-mm-per-prompt` (part of the bundle key). Restore the three wheels when an ffmpeg release fixes the 8.1.x /
9.0.2 CVEs and the wheels ship it (check weekly). Scanners read the merged filesystem; the removed bytes are still in the lower
layers of the image (a layer-by-layer review sees them; a squashed rebuild would remove them at the cost of a full 11.9 GB pull).
Scans 2026-11-17: Trivy 0.69.1 0 CRITICAL/HIGH, 14 MEDIUM; Grype 0.121.1 0 CRITICAL/HIGH, 135 MEDIUM, 22 LOW; 0 secrets.
**NOT yet run on a GPU with this image:** Nemotron end to end (bundle, boot, checks), the one-image Inkling/aux boots.

**Build kernel bundles** (dev box, Docker + idle RTX PRO 6000s; nothing goes to the network). Nemotron needs a bundle (this image
cannot compile its kernels); the aux bundles only need rebuilding if a manifest stops matching (refused at boot, loud).
1. `docker pull btbtyler09/vllm-cuda@sha256:feef2c370dca9724e378db3ed8e688ada69dafbcfadcae74f681aad5a67c23d5` (-full, 14.8 GB) and
   `docker pull btbtyler09/vllm-cuda@sha256:22d7a0cc2f77f5ce16a04b84bd33931fc7a7f3986a5b77f679969f8feb043ce8` (-r2-minimal, 11.9 GB).
2. Nemotron (2 idle cards, the runtime dir production uses, config.json sha256 `35189a80…`), from this `prod/` directory:
   ```
   BUILD_IMAGE=btbtyler09/vllm-cuda@sha256:feef2c370dca9724e378db3ed8e688ada69dafbcfadcae74f681aad5a67c23d5 \
   SHIP_IMAGE=btbtyler09/vllm-cuda@sha256:22d7a0cc2f77f5ce16a04b84bd33931fc7a7f3986a5b77f679969f8feb043ce8 \
   FIPS_ENABLED_OVERRIDE_FILE=<as in site.env, if the box is FIPS> \
   ./build-nemotron-bundle.sh GPU-aaaa,GPU-bbbb /path/to/nemotron-3.5-super-vl-runtime /data/bb-nemotron
   ```
   Aux: `./build-bundle.sh shrew-ocr|embed|rerank GPU-cccc "$AUX_ROOT/<its dir>" /data/bb-<svc>` (same env; 1 card).
   **Time:** Nemotron ~1–1.5 h (capture 30–60 min, verify 25–45 min), aux ~20–35 min each; it prints each step, a heartbeat
   per minute, and ends with `BUILD-BUNDLE PASS nemotron: …/bundle-nemotron.tar.gz sha256 …` or `ERROR: …` and exit 1 (logs in
   `OUT_DIR/logs/` and `OUT_DIR/capture|verify/bundle-*/serve_*.log`; nothing retries or falls back).
   **Disk:** OUT_DIR >= 20 GB free (the script checks); the images ~27 GB; the model where it already is. `/tmp` of each step is
   a 64 GB RAM-backed tmpfs (`TMPFS_SIZE=` to change). The capture runs on -full (with compilers); the verify boots the -minimal
   image from a fresh extraction of the tarball: PASS = nothing compiled at runtime.
   OUT_DIR's `capture/` and `verify/` are owned by uid 2000 (the containers' user): remove them with
   `docker run --rm -v /data/bb-nemotron:/o --entrypoint rm <minimal image> -rf /o/capture /o/verify`.
3. Copy `bundle-nemotron.tar.gz` and `.sha256` to the production box, then there: `sha256sum -c bundle-nemotron.tar.gz.sha256`,
   `mkdir -p "$WORKDIR/nemotron-bundles" && tar xzf bundle-nemotron.tar.gz -C "$WORKDIR/nemotron-bundles/"`
   (gives `bundle-nemotron/manifest.json`, the default `NEMOTRON_BUNDLE`). Aux: as in "Kernel cache" below.
4. `site.env`: `IMAGE` + `EXPECT_DIGEST` = ink11nemo-r2-minimal (`sha256:22d7a0cc…`), `NEMOTRON_GPUS=` (2 cards not in `GPUS`), `NEMOTRON_DIR=`.
   Then `./start-all.sh` (or `./create-nemotron.sh` alone).

**First boot: what to check.**
- Inkling: `create-inkling.sh` prints `image: ink11 (KVA) overlay present`; `docker inspect -f '{{index .Config.Labels "vllm-cuda.image-digest"}}' vllm-inkling` = `sha256:22d7a0cc…`; then POST_LIVE_TESTS 1, 3, 4 as before.
- Nemotron: `create-nemotron.sh` ends with `NEMOTRON OK` (bundle applied, no runtime compile, CPU KV offload on; any of those
  missing = `ERROR` + exit 1, container left running for `docker logs vllm-nemotron`). Then, with
  `BASE=http://<nemotron host:port> MODEL=nemotron-3.5-super-vl`: `python3 post_live.py text`, `tools 20`, `json 20`, each also with
  `REASONING=0 REASONING_OFF_KWARGS='{"enable_thinking": false}'`; one image request; `docker logs vllm-nemotron 2>&1 | grep -c 'RUNTIME CUDA COMPILE'` = 0.
- shrew-ocr / embed / rerank: `create-aux.sh` prints the `kernel bundle OK` line (a `WARNING: no 'kernel bundle OK'` = stop and
  look); their usual one-request health check passes.

