# Inkling-2 2-GPU tests (RTX PRO 6000 / sm_120)

Speed tests for the Inkling-2-Small vLLM port. Run them on two GPUs; you get back one tarball of results.

The test models are truncated (6 or 7 decoder layers), so the DSpark drafter's drafts are never accepted. Drafter arms
measure what the drafter **costs** per step, not its acceptance.

## Run

Needs Docker with the NVIDIA runtime and 2 GPUs. Disk: the test models go to `/work/slices` (~60 GB), and logs, traces
and the compile cache go to `/work` (~20 GB). On one disk that is 80 GB. The run checks free space first and stops with
a message naming the mount that is short.

- **Small home disk:** keep `/work` in your home dir and add `-v /big/disk/inkling-slices:/work/slices` to the command
  below. That dir needs 60 GB free, and home needs 20 GB.

```bash
WORKDIR=~/inkling-tests          # needs 20 GB free (80 GB if you don't add the /work/slices mount)
mkdir -p "$WORKDIR" && cd "$WORKDIR"
echo 0 > fips_enabled          # FIPS hosts: the image's OpenSSL dies at import unless this file is mounted (harmless elsewhere)
docker run --rm --gpus '"device=0,1"' --ipc host --shm-size 32g \
  --user "$(id -u):0" -e HOME=/work \
  -v "$PWD/fips_enabled:/proc/sys/crypto/fips_enabled:ro" \
  -v /path/to/Inkling-2-Small:/model:ro \
  -e IMAGE_REF=sha256:f7c4a4fc540618c66757a532cd49b14aa93f93cbbeac2bc9376ba07d64c98f9f \
  -v "$PWD:/work" \
  btbtyler09/vllm-cuda@sha256:f7c4a4fc540618c66757a532cd49b14aa93f93cbbeac2bc9376ba07d64c98f9f
```

Note: `tests/` here is ahead of that image: /metrics fetched via Python (the image has no curl, so its
`metrics_*.txt` are empty) and a startup check for every external command. The next image picks them up.

Image: `btbtyler09/vllm-cuda:inkling2-tests-20261116`, built from this repo's `tests/` with inkling2 model code
`inkling2-ink10-rc 4e6fc4b71b`.

- **Weights:** `-v /path/to/Inkling-2-Small:/model:ro` is the full checkpoint already on disk: the original
  `thinkingmachines` one or the NVFP4 requant. The test models are cut from it; nothing is downloaded. Every tensor is
  checked against `tests/slice_manifest.json` (sha256), and the run stops on a mismatch.
  - Without `/model`, about 57 GB is downloaded from huggingface.co instead: only layers 0-6, embeddings, head, vision
    and draft. `-e MAX_MBPS=50` caps the rate.
  - Both paths were verified on a cloud RTX PRO 6000 on 2026-11-16.
- **Results:** `$WORKDIR/results.tgz`, rewritten after every arm, so a partial run still returns data.
  `summary.md` is at the top of the tarball and also next to it. Send `results.tgz` back privately.
- **Subset:** `-e PHASES="race b12x"`. The `slice` phase must have run once; it is skipped when the slices exist.
- **Other GPUs:** change `device=0,1`. Inside the container they become GPU 0 and GPU 1.

## Phases, in run order (most valuable first), with estimated wall time

| Phase | Arms | Measures | Est. time |
|---|---|---|---|
| `env` | | GPUs, topology, driver/CUDA, versions, image digest | <1 min |
| `slice` | | Cuts the 6- and 7-layer test models from `/model`, sha-checked | 5-10 min |
| `race` | | MoE kernels at the TP6/TP4 per-GPU expert shapes, M=8-64: Marlin W4A16, b12x W4A16 (heuristic/tuned), b12x W4A4 (static scale), FlashInfer CUTLASS W4A4 (dynamic). Experts-only and whole layer (router + top-8 + experts), numerics vs Marlin and vs fp32 | 15-25 min |
| `b12x` | MT, BT, MD, BD | End-to-end slice decode: Marlin (M*) vs b12x MoE backend (B*), no draft (*T) and DSpark K=7 (*D) | 40 min |
| `remote` | R1, T7, D1, D1E | Drafter on its own GPU (R1) vs co-located with CUDA graph (D1) or eager (D1E); T7 = no drafter | 40 min |
| `fp8` | D1F8, R1F8 | FP8 (weight-only) drafter weights, co-located and remote | 20 min |
| `kdyn` | D1K4, D1DYN | DSpark K=4, and K by batch size (7 / 5 / 3) | 20 min |
| `w4a4` | W0, W4, W4P | W4A16 vs W4A4 at every batch size (W4) vs W4A4 for prefill only (W4P): TTFT 32K/128K, decode, logprob drift | 50 min |

**Total about 3-3.5 hours.** Each server arm boots once, takes about 1-2 min including graph capture, then runs its tests.

## What every arm saves (`pub/arms/<ARM>/`)

- `cmd.txt`: the exact env and command line. `server.log`: the full boot and serve log, with backend selection,
  "GPU KV cache size", graph capture and spec-decode metrics.
- `metrics_before.txt` / `metrics_after.txt`: vLLM `/metrics` snapshots. These carry spec-decode drafted/accepted
  counts per position.
- `prof_c1.log`, `prof_c4.log`, `trace_c*/`, `kernels_c*.txt`: a torch profiler window of 32 decode steps at c=1 and
  c=4, and the per-kernel table (µs/step).
- `tps_c*.log`, `ctx32.log`, `ctx128.log`, `lp.json`: every client's full output (tok/s, TTFT, ITL p50/p99, logprobs).
- `mem_after_boot.txt`: per-GPU memory once the server is up.

Run-wide files:
- `pub/arms.tsv`: every arm PASS / FAIL / SKIPPED with its reason. A failed arm keeps its logs, and the run continues.
- `pub/gpu.csv`: per-GPU utilization, SM/memory clock, power, temperature and memory, every second.
- `pub/env.txt`, `pub/console.log`.
- `pub/moe_race.{log,json}`, `pub/slice.log`.

Expected tarball size: about 0.3-1 GB, mostly the profiler traces.

## Files

- `tests/run_all`: the runner (the image's entrypoint). `tests/mk_summary.py`: writes summary.md.
- `tests/slice_build.py` + `tests/slice_manifest.json`: build the test models; per-tensor sha256.
- `tests/moe_race.py`: the MoE kernel race. `tests/prof_drive.py`, `tests/prof_an.py`: profile window and kernel table.
- `tests/post_live.py`, `tests/ld_ctx.py`, `tests/lp_cmp.py`: load and logprob clients.
- `Dockerfile`: the test image = the inkling vLLM image + the current inkling2 model code + `tests/`.
- `rehearse/`: a stub server for dry-running `run_all` without GPUs.

Model: Inkling-2-Small by Thinking Machines (Apache-2.0). NVFP4 requant: `btbtyler09/Inkling-2-Small-NVFP4`.
