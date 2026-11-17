# prod_env.sh -- sourced by create-*.sh and prod_preflight.sh. Site values come from ./site.env (never committed).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# For these the command line beats site.env (runbook: `ALLREDUCE_IPC=0 ./create-inkling.sh`; start-all.sh's `nobundle`):
# remember what the environment set, source site.env, put it back. Everything else: site.env wins.
_OV="ALLREDUCE_IPC ENFORCE_TOOL_CHOICE SPLIT_KV_STATIC SHREW_VARIANT AUX_NO_BUNDLE KV_MIB KV_GIB OFFLOAD_GIB HEALTH_TIMEOUT_S RESTART_POLICY EXTRA_SERVE_ARGS FIPS_ENABLED_OVERRIDE_FILE AUX_GPU_MEMORY_UTILIZATION MAXSEQ DSPARK DSPARK_K REUSE_REL_LOGITS LOCAL_SPLIT_KV GLOBAL_SPLIT_TUNED CONV_TAIL_CACHE MPS AUX_MPS_PCT P2P_DISABLE NEMOTRON_SPEC NEMOTRON_NO_BUNDLE NEMOTRON_OFFLOAD_GIB"
_ovs=(); for _v in $_OV; do [ -n "${!_v+x}" ] && _ovs+=("$_v=${!_v}"); done
[ -r "$HERE/site.env" ] && . "$HERE/site.env"
for _p in ${_ovs[@]+"${_ovs[@]}"}; do _v=${_p%%=*}; _c=${_p#*=}   # ${a[@]+...}: an empty array is "unbound" under set -u before bash 4.4
  [ "${!_v-}" = "$_c" ] || { echo "note: $_v='$_c' from the command line, not '${!_v-}' from site.env" >&2; printf -v "$_v" '%s' "$_c"; }; done
unset _OV _ovs _v _p _c
# create-inkling.sh turns on ENFORCE_TOOL_CHOICE and SPLIT_KV_STATIC, which images before ink7 ignore; the aux bundles match ink8 only
# (same vllm version string in ink9.1 / ink10 / ink11).
# Image identity (2026-11-17): ONE image for all four services (Inkling, Nemotron, shrew-ocr, embed/rerank) = ink11nemo-minimal
# (ink11kva-minimal + 3 files: 0071 nano_nemotron_vl.py, 0089 stream_processor.py (env opt-in, set by create-nemotron.sh only),
# OVERLAY.txt; same vllm/torch/triton/flashinfer versions, so the aux bundles still match). Pin from site.env (or the environment):
#   EXPECT_DIGEST=sha256:<digest>  IMAGE must end with @<digest>: ink11nemo-r2-minimal sha256:22d7a0cc2f77f5ce16a04b84bd33931fc7a7f3986a5b77f679969f8feb043ce8
#                                  (= ink11nemo-minimal 1e567856 minus PyNvVideoCodec, PyAV, OpenCV: their bundled ffmpeg had HIGH CVEs;
#                                  no video, no mp4/aac audio), or a previous image for a rollback (ink11nemo-minimal 1e567856…,
#                                  ink11kva-minimal sha256:1fc652f4…, ink9.1 sha256:f5740ac4…).
#   EXPECT_DIGEST unset / empty    IMAGE must be the local tag INK11_TAG (any registry prefix) and its OVERLAY.txt must carry
#                                  the ink11 section (checked in ink9_check: a tag can be re-pointed, the file cannot lie about it).
# create-nemotron.sh also requires the ink11nemo section in OVERLAY.txt (whatever the pin).
# Bump this, site.env.example and RUNBOOK_PROD.md together when another image ships.
INK11_TAG=btbtyler09/vllm-cuda:0.34.0rc1.dev307-cu130-ink11nemo-r2-minimal
EXPECT_DIGEST=${EXPECT_DIGEST:-}
: "${IMAGE:?set IMAGE in site.env: the ink11 tag $INK11_TAG, or <registry>/vllm-cuda@<digest> with EXPECT_DIGEST=<digest>}" "${MODEL_DIR:?}" "${WORKDIR:?}" "${NETWORK:?}"
PORT=${PORT:-8000}; SERVED_NAME=${SERVED_NAME:-inkling}
die(){ echo "ERROR: $*" >&2; exit 1; }
# TP (site.env): tensor-parallel size of vllm-inkling = the number of cards in GPUS. Default 8 (TP8 on 8 cards, unchanged).
# NGPU is the older name of the same value; set either (both = they must agree). TP=4 on one socket: RUNBOOK "ink11".
TP=${TP:-${NGPU:-8}}
[[ "$TP" =~ ^[1-8]$ ]] || die "TP must be 1..8 (got '$TP')"
[ -z "${NGPU:-}" ] || [ "$NGPU" = "$TP" ] || die "TP=$TP and NGPU=$NGPU disagree in site.env / the environment (set one of them)"
NGPU=$TP
# gpus_resolve: GPUS (site.env) = the TP cards, comma separated, as UUIDs (GPU-...) or nvidia-smi indices (0,1,2,3). Indices are
# turned into UUIDs here, so docker --gpus, the MPS daemon and every check see UUIDs. Unknown or repeated cards, or a count that is
# not TP, return 1 with an ERROR line. TP8 with 8 UUIDs passes through unchanged.
gpus_resolve(){ local map u v out= seen=,
  [ -n "${GPUS:-}" ] || { echo "ERROR: set GPUS in site.env: $NGPU GPU UUIDs or indices, comma separated (TP=$NGPU)" >&2; return 1; }
  [ "$(awk -F, '{print NF}' <<< "$GPUS")" = "$NGPU" ] || { echo "ERROR: GPUS must list $NGPU GPUs (TP=$NGPU; got '$GPUS')" >&2; return 1; }
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)   # list first, then match (grep -q under pipefail: SIGPIPE)
  for u in ${GPUS//,/ }; do
    if [[ "$u" =~ ^[0-9]+$ ]]; then v=$(awk -F', *' -v i="$u" '$1 == i {print $2}' <<< "$map")
    else v=$(awk -F', *' -v x="$u" '$2 == x {print $2}' <<< "$map"); fi
    [ -n "$v" ] || { echo "ERROR: GPU $u not on this host" >&2; return 1; }
    [[ "$seen" != *",$v,"* ]] || { echo "ERROR: GPUS names $v twice ('$GPUS')" >&2; return 1; }
    seen="$seen$v,"; out=${out:+$out,}$v; done
  GPUS=$out; }
# KV_GIB is per card. At TP8 with 48 layers one 262,144-token request is charged about 13.4 GiB per card at admission with
# --max-num-batched-tokens 4096 (sat_flags.sh; 25.4 GiB with 8192). 12 (24) GiB of it is the short-conv state of in-flight
# tokens, which does not shrink with TP or with an FP8 KV cache; below one request vLLM refuses to boot ("To serve at least one
# request with the model's max seq len"). COMPUTED from the formula that matches four real boots (1 card 24 GiB bf16, fp8, and
# 4096; 2 cards 16 GiB) to the token; never booted at TP8.
# ink11 (CONV_TAIL_CACHE=1, the default): the short-conv state is tail-only, so that per-request charge is mostly gone. Measured
# TP4 FP8 KV 10 GiB per card: 1,852,556 KV tokens = 7.07 requests of 262,144 (was 368,476). The figures above hold again with
# CONV_TAIL_CACHE=0 or a pre-ink11 image.
KV_GIB=${KV_GIB:-32}; OFFLOAD_GIB=${OFFLOAD_GIB:-256}
# shrew-ocr checkpoint that create-aux.sh serves and prod_preflight.sh checks:
#   fp8-v0.5  (default) shrew-ocr-preview v0.5, calibrated FP8 checkpoint; vLLM reads the quantization from its config.json (no flag)
#   gptq-v0.4 the previous line (v0.4 GPTQ 8-bit, --dtype half): the ROLLBACK for shrew-ocr. Its bundle lives one level down.
# SHREW_DIR = directory name under AUX_ROOT; SHREW_BUNDLE = path under AUX_BUNDLES. Each checkpoint has its own kernel bundle.
SHREW_VARIANT=${SHREW_VARIANT:-fp8-v0.5}
case $SHREW_VARIANT in
  fp8-v0.5)  SHREW_DIR=${SHREW_DIR:-shrew-ocr-preview-v0.5-FP8}; SHREW_FLAGS=(); SHREW_BUNDLE=bundle-shrew-ocr ;;
  gptq-v0.4) SHREW_DIR=${SHREW_DIR:-shrew-ocr-runtime}; SHREW_FLAGS=(--dtype half); SHREW_BUNDLE=gptq-v0.4/bundle-shrew-ocr ;;
  *) die "SHREW_VARIANT must be fp8-v0.5 or gptq-v0.4 (got '$SHREW_VARIANT')" ;;
esac
# Optional site-specific container opt-out; it does not change host FIPS mode (RUNBOOK "FIPS-mode hosts").
FIPS_MOUNTS=()
if [ -n "${FIPS_ENABLED_OVERRIDE_FILE:-}" ]; then
  [ -f "$FIPS_ENABLED_OVERRIDE_FILE" ] && [ "$(cat "$FIPS_ENABLED_OVERRIDE_FILE")" = 0 ] || die "FIPS_ENABLED_OVERRIDE_FILE must be a readable file containing 0"
  FIPS_MOUNTS=(-v "$FIPS_ENABLED_OVERRIDE_FILE:/proc/sys/crypto/fips_enabled:ro")
fi
if [ -n "$EXPECT_DIGEST" ]; then
  [[ "$IMAGE" == *@"$EXPECT_DIGEST" ]] || die "IMAGE ($IMAGE) is not pinned to EXPECT_DIGEST=$EXPECT_DIGEST"
  IMAGE_PIN=$EXPECT_DIGEST
else
  [[ "$IMAGE" == "$INK11_TAG" || "$IMAGE" == */"$INK11_TAG" ]] \
    || die "IMAGE ($IMAGE) is neither the ink11 tag $INK11_TAG nor pinned by digest (set EXPECT_DIGEST=sha256:... with IMAGE=<registry>/vllm-cuda@sha256:...)"
  IMAGE_PIN=tag:${INK11_TAG##*:}
fi
# Inkling settings (RUNBOOK "Opt-in: DSpark draft, bias reuse, more sequences" and "ink11"). Since the ink11 kit the three decode
# paths that passed GPU validation (sm_120 numerics 11-15, ink11 pod 11-17) are ON by default; each stays overridable to 0:
#   MAXSEQ=32 (default) | 64 | 128       --max-num-seqs: concurrent requests decoded together; more queue as before
#   DSPARK=0 (default) | 1               the checkpoint's DSpark draft (mtp.safetensors), 7 tokens per step. Needs ink9 or later
#   REUSE_REL_LOGITS=1 (default) | 0     attention reuses the prologue's relative logits (INKLING2_REUSE_REL_LOGITS). ink9+
#   LOCAL_SPLIT_KV=1 (default) | 0       split-KV decode for the sliding-window layers (INKLING2_LOCAL_SPLIT_KV). ink9.1+
#   GLOBAL_SPLIT_TUNED=1 (default) | 0   global layers' split count from the SM count (INKLING2_GLOBAL_SPLIT_TUNED). ink9.1+
#                                        An ink8 image refuses these defaults at create: set all three to 0 for ink8.
#   CONV_TAIL_CACHE=1 (default) | 0      INKLING2_CONV_TAIL_CACHE: tail-only short-conv state cache (ink11; the image's default
#                                        is also on). 0 = the dense conv cache of ink10 and before (rollback; ~5x fewer KV tokens).
#                                        Ignored by images before ink11.
#   DSPARK_K=7 (default) | 1..7          draft tokens per step with DSPARK=1. Below 7 needs the ink10 image or later
#                                        (ink9/ink9.1 refuse any count but 7 at boot)
MAXSEQ=${MAXSEQ:-32}; DSPARK=${DSPARK:-0}; DSPARK_K=${DSPARK_K:-7}; REUSE_REL_LOGITS=${REUSE_REL_LOGITS:-1}
LOCAL_SPLIT_KV=${LOCAL_SPLIT_KV:-1}; GLOBAL_SPLIT_TUNED=${GLOBAL_SPLIT_TUNED:-1}; CONV_TAIL_CACHE=${CONV_TAIL_CACHE:-1}
[[ "$MAXSEQ" =~ ^[1-9][0-9]*$ ]] && [ "$MAXSEQ" -le 256 ] || die "MAXSEQ must be a whole number 1..256 (got '$MAXSEQ')"
case $DSPARK in 0|1) ;; *) die "DSPARK must be 0 or 1 (got '$DSPARK')" ;; esac
case $DSPARK_K in [1-7]) ;; *) die "DSPARK_K must be 1..7 (got '$DSPARK_K')" ;; esac
case $REUSE_REL_LOGITS in 0|1) ;; *) die "REUSE_REL_LOGITS must be 0 or 1 (got '$REUSE_REL_LOGITS')" ;; esac
case $LOCAL_SPLIT_KV in 0|1) ;; *) die "LOCAL_SPLIT_KV must be 0 or 1 (got '$LOCAL_SPLIT_KV')" ;; esac
case $GLOBAL_SPLIT_TUNED in 0|1) ;; *) die "GLOBAL_SPLIT_TUNED must be 0 or 1 (got '$GLOBAL_SPLIT_TUNED')" ;; esac
case $CONV_TAIL_CACHE in 0|1) ;; *) die "CONV_TAIL_CACHE must be 0 or 1 (got '$CONV_TAIL_CACHE')" ;; esac
# MPS=0 (default) | 1: the GPU services run as clients of ONE host MPS control daemon (./mps.sh start, RUNBOOK "MPS"), so the
# aux services and the Inkling rank on a card run kernels side by side instead of taking turns, and each aux service is capped
# at AUX_MPS_PCT % of the card's SMs (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE). Inkling is a client too, uncapped. AUX_MPS_PCT has
# NO default on purpose: it is picked from the interference measurement; MPS=1 without it is refused.
# MPS=1 container path UNTESTED on real docker; proven only via the pod-side shim (RUNBOOK "MPS").
MPS=${MPS:-0}; AUX_MPS_PCT=${AUX_MPS_PCT:-}
case $MPS in 0|1) ;; *) die "MPS must be 0 or 1 (got '$MPS')" ;; esac
if [ "$MPS" = 1 ]; then
  [[ "$AUX_MPS_PCT" =~ ^[1-9][0-9]*$ ]] && [ "$AUX_MPS_PCT" -le 100 ] || die "MPS=1 needs AUX_MPS_PCT=1..100 in site.env (got '$AUX_MPS_PCT'; no default: pick it from the measurement, RUNBOOK 'MPS')"
fi
# One daemon for all cards: Inkling's TP ranks share one container and one environment, so one CUDA_MPS_PIPE_DIRECTORY.
MPS_DIR=${MPS_DIR:-$WORKDIR/mps}; MPS_PIPE=$MPS_DIR/pipe; MPS_LOG=$MPS_DIR/log
mps_up(){ [ -S "$MPS_PIPE/control" ] && pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; }
# mps_args aux|inkling -> MPS_ARGS (docker run flags; empty at MPS=0). Clients need the daemon's pipe directory at the same
# path, the host IPC namespace, and the same uid as every other client (the kit runs all services as uid 2000).
mps_args(){ MPS_ARGS=(); [ "$MPS" = 1 ] || return 0
  mps_up || die "MPS=1 but no MPS control daemon on $MPS_PIPE: run ./mps.sh start (as root) first"
  MPS_ARGS=(-v "$MPS_PIPE:$MPS_PIPE" -e "CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE")
  [ "$1" = aux ] && MPS_ARGS+=(-e "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$AUX_MPS_PCT")
  return 0; }
# mps_verify NAME: after health, the container's GPU processes must show as MPS clients (type M+C in nvidia-smi). A client
# whose MPS connection fails can run as a plain context instead, which brings back the time slicing MPS=1 is meant to remove.
mps_verify(){ [ "$MPS" = 1 ] || return 0; local pids mc n=0 p
  pids=$(docker top "$1" -eo pid 2>/dev/null | tail -n +2 | tr -s ' \n' ' ') || true
  mc=$(nvidia-smi 2>/dev/null | awk '/M\+C/{print $5}' | tr '\n' ' ')
  for p in $pids; do [[ " $mc " == *" $p "* ]] && n=$((n + 1)); done
  if [ "$n" -gt 0 ]; then echo "mps: $1 has $n GPU process(es) attached as MPS clients (M+C)"
  else echo "FAIL: MPS=1 but no GPU process of $1 shows as an MPS client (M+C) in nvidia-smi: it runs WITHOUT MPS. Check ./mps.sh status and the server log $MPS_LOG/server.log"; return 1; fi; }
# ink9_check: DSPARK=1 / REUSE_REL_LOGITS=1 need an image that carries the ink9 files (ink8 would crash in the draft's sm100
# kernel, or ignore the flag); LOCAL_SPLIT_KV=1 / GLOBAL_SPLIT_TUNED=1 need ink9.1 (ink9 ignores them silently).
# Checked from the image itself (OVERLAY.txt), so it holds for any registry digest of the image. The ink11 OVERLAY.txt keeps the
# ink9.1 text first, so it passes both. ink11: with IMAGE given by tag (no EXPECT_DIGEST) the file must carry the ink11 section;
# with a digest pin, a pre-ink11 image only gets a note (CONV_TAIL_CACHE has no effect there: the rollback-to-previous-image path).
ink11_marker='--- ink11 layer'
ink9_check(){ local o=/usr/local/share/vllm-cuda/OVERLAY.txt
  if [ "$DSPARK" = 1 ] || [ "$REUSE_REL_LOGITS" = 1 ]; then
    docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -q dspark_portable $o \
      || die "DSPARK=1 / REUSE_REL_LOGITS=1 need the ink9 image (or later); $IMAGE has no $o naming the ink9 files"; fi
  if [ "$LOCAL_SPLIT_KV" = 1 ] || [ "$GLOBAL_SPLIT_TUNED" = 1 ]; then
    docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -q INKLING2_LOCAL_SPLIT_KV $o \
      || die "LOCAL_SPLIT_KV=1 / GLOBAL_SPLIT_TUNED=1 need the ink9.1 image (or later); $IMAGE has no $o naming them (set both to 0 for an older image)"; fi
  if docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -qF -- "$ink11_marker" $o; then
    echo "image: ink11 (KVA) overlay present; CONV_TAIL_CACHE=$CONV_TAIL_CACHE"
  elif [ -z "$EXPECT_DIGEST" ]; then
    die "IMAGE $IMAGE is the ink11 tag but its $o has no '$ink11_marker' section: not the ink11 build (load the ink11 image, or pin another image with EXPECT_DIGEST)"
  else echo "note: $IMAGE has no ink11 section in $o (pre-ink11 image): CONV_TAIL_CACHE=$CONV_TAIL_CACHE has no effect"; fi; }
refuse_existing(){ ! docker container inspect "$1" >/dev/null 2>&1 || die "container '$1' already exists ($(docker container inspect -f '{{.State.Status}}' "$1")). Remove it yourself first: docker rm -f $1"; }
# docker_run_failed NAME CREATED_BY ERRFILE: the create script's `docker run -d` of NAME failed; ERRFILE holds docker's stderr.
# `docker run -d` creates the container, then starts it: a failed start (e.g. the FIPS mount on a kernel without
# /proc/sys/crypto: runc cannot create the mount point) leaves it "Created", which refuse_existing would block next time.
# docker's own exit code is not passed on: the daemon reports 127 whenever the runtime's error says "no such file or directory",
# which reads like a command missing from this script. refuse_existing ran first, so a NAME carrying this script's created-by
# label was made by this run: remove it (nothing else), print one ERROR line with docker's message, exit 1.
docker_run_failed(){ local n=$1 by=$2 err=$3
  echo "ERROR: docker run failed: $(cat "$err")" >&2
  if [ "$(docker container inspect -f '{{index .Config.Labels "vllm-cuda.created-by"}}' "$n" 2>/dev/null)" = "$by" ]; then
    if docker rm -f "$n" >/dev/null 2>&1; then echo "removed the half-created container $n" >&2
    else echo "WARNING: could not remove the half-created container $n: docker rm -f $n" >&2; fi; fi
  exit 1; }
# docker_run_ok ERRFILE: success path; show anything docker printed on stderr (warnings) as before.
docker_run_ok(){ [ -s "$1" ] && cat "$1" >&2; rm -f "$1"; return 0; }
# health NAME PORT MODEL TIMEOUT_S [chat|embed|rerank] : wait for /health inside the container, then send one request.
health(){ local n=$1 p=$2 m=$3 t=$4 k=${5:-chat} s=$SECONDS
  until docker exec "$n" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$p/health', timeout=3)" >/dev/null 2>&1; do
    # "running 0": a container under a restart policy that crash-loops is "running" most of the time, so count restarts too
    [ "$(docker container inspect -f '{{.State.Status}} {{.RestartCount}}' "$n" 2>/dev/null)" = "running 0" ] || { docker logs --tail 30 "$n" 2>&1 | cut -c1-300; die "$n stopped (or restarted) during boot (log tail above; full log: docker logs $n)"; }
    [ $((SECONDS - s)) -lt "$t" ] || { docker logs --tail 30 "$n" 2>&1 | cut -c1-300; die "$n not healthy after ${t}s"; }; sleep 10; done
  echo "ok: $n healthy after $((SECONDS - s))s"
  docker exec -i "$n" python3 - "$p" "$m" "$k" <<'PY'
import json, sys, urllib.error, urllib.request
p, m, k = sys.argv[1:4]
path, body = {"chat": ("/v1/chat/completions", {"model": m, "max_tokens": 64, "chat_template_kwargs": {"reasoning_effort": 0}, "messages": [{"role": "user", "content": "Reply with one short sentence: what is the capital of France?"}]}),
              "embed": ("/v1/embeddings", {"model": m, "input": ["hello"]}),
              "rerank": ("/rerank", {"model": m, "query": "capital of France", "documents": ["Paris is the capital of France.", "Bananas are yellow."]})}[k]
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{p}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
except urllib.error.HTTPError as e:   # show the server's answer, not a Python traceback
    sys.exit(f"HEALTH REQUEST FAILED: POST {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:600]}")
out = r["choices"][0]["message"] if k == "chat" else (len(r["data"][0]["embedding"]) if k == "embed" else [round(x["relevance_score"], 3) for x in r["results"]])
print("HEALTH REQUEST OK:", json.dumps(out)[:400])
PY
}
