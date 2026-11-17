#!/bin/bash
# build-bundle.sh SERVICE GPUS MODEL_DIR OUT_DIR   -- build + strictly verify one service's kernel bundle on a GPU box with Docker.
#   SERVICE    nemotron (2 GPUs, TP2) | shrew-ocr | embed | rerank (1 GPU each)
#   GPUS       GPU UUIDs or nvidia-smi indices, comma separated (idle cards: the run refuses a card with >1 GiB in use)
#   MODEL_DIR  the service's runtime model dir, exactly the folder production mounts (nemotron: NEMOTRON_DIR; aux: AUX_ROOT/<dir>)
#   OUT_DIR    a new or empty directory with >= 20 GB free; gets bundle-<svc>.tar.gz (+ .sha256), logs/, BUILD_BUNDLE_<svc>.txt
# Steps (each in its own throwaway container, no network, uid 2000, read-only root, all capabilities dropped except SYS_PTRACE,
# no-new-privileges, the FIPS mount when FIPS_ENABLED_OVERRIDE_FILE is set; writable: OUT_DIR and a /tmp tmpfs only):
#   1 preflight   images, GPUs, model, disk                                                      ~1 min
#   2 capture     BUILD image (-full, has the JIT compilers) boots the service with its PRODUCTION serve flags (the same
#                 definition create-nemotron.sh / create-aux.sh use), drives the image's traffic sweep (text ladder up to
#                 max-num-seqs, long prompts, images for VL (no video: r2 has no decoder), pooling for embed/rerank) + a 2-min soak, compiles the
#                 sampler variants, writes the bundle                                    nemotron ~30-60 min; aux ~10-20 min
#   3 verify      SHIP image (-minimal, no compiler) boots from a FRESH extraction of the tarball with the same flags and
#                 the same sweep + a VERIFY_SOAK_MIN soak; PASS only if nothing was compiled at runtime   nemotron ~25-45 min
#   4 publish     OUT_DIR/bundle-<svc>.tar.gz + .sha256 (the tarball that passed verify, unmodified)  -> "BUILD-BUNDLE PASS"
# Any failure: "ERROR: ..." and exit 1; logs stay in OUT_DIR/logs and the bundle dirs. Nothing is retried, nothing falls back.
# Settings (environment):
#   BUILD_IMAGE   default the ink11nemo-full tag (or <registry>/vllm-cuda@sha256:feef2c37...)   BUILD-ONLY image, never serve it
#   SHIP_IMAGE    default the ink11nemo-r3-minimal tag (or <registry>/vllm-cuda@sha256:ab4a1359...)  = site.env IMAGE
#                 (a bundle built with r2 as SHIP_IMAGE is valid for r3: same libraries and flags)
#   FIPS_ENABLED_OVERRIDE_FILE  as in site.env (a file containing 0, mounted over /proc/sys/crypto/fips_enabled)
#   SHREW_VARIANT fp8-v0.5 (default) | gptq-v0.4   (shrew-ocr only; as in site.env)
#   VERIFY_SOAK_MIN 5 (default)   BUILD_SOAK_MIN 2 (default)   TMPFS_SIZE 64g (default; /tmp of each container, RAM-backed)
#   VERIFY_ONLY   path to a capture tarball (OUT_DIR/capture/bundle-<svc>.tar.gz of an earlier run): skip step 2, verify
#                 and publish that tarball (same preflight; OUT_DIR must be a NEW directory)
# Then on the production box: tar xzf bundle-<svc>.tar.gz -C <bundle parent> (RUNBOOK "Build kernel bundles").
set -euo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
S=${1:-}; G=${2:-}; MD=${3:-}; O=${4:-}
[ -n "$S" ] && [ -n "$G" ] && [ -n "$MD" ] && [ -n "$O" ] || die "usage: build-bundle.sh nemotron|shrew-ocr|embed|rerank GPUS MODEL_DIR OUT_DIR"
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/nemotron_flags.sh"; . "$HERE/aux_flags.sh"
BUILD_IMAGE=${BUILD_IMAGE:-btbtyler09/vllm-cuda:0.34.0rc1.dev307-cu130-ink11nemo-full}
SHIP_IMAGE=${SHIP_IMAGE:-btbtyler09/vllm-cuda:0.34.0rc1.dev307-cu130-ink11nemo-r3-minimal}
VERIFY_SOAK_MIN=${VERIFY_SOAK_MIN:-5}; BUILD_SOAK_MIN=${BUILD_SOAK_MIN:-2}; TMPFS_SIZE=${TMPFS_SIZE:-64g}
[[ "$VERIFY_SOAK_MIN" =~ ^[0-9]+$ && "$BUILD_SOAK_MIN" =~ ^[0-9]+$ ]] || die "VERIFY_SOAK_MIN / BUILD_SOAK_MIN must be whole minutes"
TOOLS=/usr/local/share/vllm-cuda/tools   # the image's own kernel_bundle.sh / traffic_sweep.py (same files as the kit's packaging/bundle)
. "$HERE/fi_load_only.sh"   # FI_LOAD_ONLY=(mounts + env)
if [ -n "${VERIFY_ONLY:-}" ]; then VERIFY_ONLY=$(cd "$(dirname "$VERIFY_ONLY")" 2>/dev/null && pwd)/$(basename "$VERIFY_ONLY")
  [ -s "$VERIFY_ONLY" ] || die "VERIFY_ONLY $VERIFY_ONLY: no such tarball"; fi
T0=$(date +%s); el(){ local s=$(( $(date +%s) - $1 )); printf '%dm%02ds' $((s / 60)) $((s % 60)); }
step(){ echo; echo "=== [$1] $2   (expected $3; started $(date -u +%H:%M:%SZ), total so far $(el $T0))"; }

# ---- per-service definition: the PRODUCTION flags (shared files), the in-container model path, sweep settings
NGPU=1; IPC=(--ipc private); EXTRA_ENV=()
case $S in
  nemotron)
    NEMOTRON_SPEC=mtp3; nemotron_flags || exit 1   # the bundle is mtp3's (create-nemotron.sh's default), NEMOTRON_OFFLOAD_GIB as site
    RT=$NEMOTRON_RT; NAME=$NEMOTRON_SERVED; NGPU=2; FL=("${NEM_FLAGS[@]}"); SHM=; IPC=(--ipc host)
    # no --video: the ship image (r2) has no video decoder (OpenCV/PyAV removed for their bundled ffmpeg's CVEs; no service
    # takes video). --limit-mm-per-prompt keeps video:1 (it is part of the bundle key); a video request gets an error response.
    SWEEP=(-e MM_IMAGES=8)
    BENV=""; for ((i = 1; i < ${#NEMOTRON_ENV[@]}; i += 2)); do BENV+="${NEMOTRON_ENV[$i]} "; done; BENV+="VLLM_SKIP_P2P_CHECK=0" ;;
  shrew-ocr|embed|rerank)
    # SHREW_VARIANT -> SHREW_DIR / SHREW_FLAGS exactly as prod_env.sh does (keep the two in step)
    case ${SHREW_VARIANT:-fp8-v0.5} in
      fp8-v0.5)  SHREW_DIR=shrew-ocr-preview-v0.5-FP8; SHREW_FLAGS=() ;;
      gptq-v0.4) SHREW_DIR=shrew-ocr-runtime; SHREW_FLAGS=(--dtype half) ;;
      *) die "SHREW_VARIANT must be fp8-v0.5 or gptq-v0.4" ;; esac
    aux_service "$S"; RT=$M; NAME=$S; FL=(--gpu-memory-utilization 0.15 --kv-cache-memory-bytes $(( KV * 1048576 )) "${F[@]}"); IPC+=(--shm-size "$SHM")
    case $S in shrew-ocr) SWEEP=(-e MM_IMAGES=1 -e MM_PROFILE=shrew) ;;
               embed)     SWEEP=(-e MM_IMAGES=1 -e POOLING=embed -e MM_PROFILE=embed -e SWEEP_MAXSEQ=1024) ;;
               rerank)    SWEEP=(-e POOLING=score -e MM_PROFILE=rerank -e SWEEP_MAXSEQ=1024) ;; esac
    BENV="" ;;
  *) die "unknown service '$S' (nemotron | shrew-ocr | embed | rerank)" ;;
esac
B=bundle-$S

# ---- 1 preflight
step 1/4 "preflight: images, GPUs, model, disk" "~1 min"
command -v docker >/dev/null && command -v nvidia-smi >/dev/null || die "needs docker and nvidia-smi on this host"
for i in "$BUILD_IMAGE" "$SHIP_IMAGE"; do docker image inspect "$i" >/dev/null 2>&1 || die "image $i is not on this host (docker pull it first)"; done
[ "$(docker image inspect -f '{{index .Config.Labels "vllm-cuda.deploy"}}' "$BUILD_IMAGE")" = forbidden-build-only ] \
  || die "BUILD_IMAGE $BUILD_IMAGE is not the build-only -full capture image (label vllm-cuda.deploy=forbidden-build-only)"
for i in "$BUILD_IMAGE" "$SHIP_IMAGE"; do
  docker run --rm --pull never --network none --entrypoint grep "$i" -qF -- '--- ink11nemo layer' /usr/local/share/vllm-cuda/OVERLAY.txt \
    || die "$i has no ink11nemo section in OVERLAY.txt (not the ink11nemo build)"; done
docker run --rm --pull never --network none --entrypoint sh "$SHIP_IMAGE" -c 'command -v nvcc || command -v gcc || command -v c++' >/dev/null 2>&1 \
  && die "SHIP_IMAGE $SHIP_IMAGE has a compiler: verify must run on the compiler-free -minimal image that serves"
docker run --rm --pull never --network none --entrypoint sh "$BUILD_IMAGE" -c 'for t in cc c++ gcc g++ nvcc ninja ld ar as; do command -v $t >/dev/null || { echo "missing $t"; exit 1; }; done' \
  || die "BUILD_IMAGE $BUILD_IMAGE lacks a JIT tool (above): use ink11nemo-full sha256:feef2c37... or later"
libs(){ docker run --rm --pull never --network none --entrypoint /opt/rt/bin/python "$1" -I -c 'import sys,json,importlib.metadata as md
sys.path.insert(0,"/usr/local/lib/vllm-cuda"); import bundle_entry as be; v={}
for n in be.LIBS:
    try: v[n]=md.version(n)
    except md.PackageNotFoundError: pass
print(json.dumps(v, sort_keys=True))' 2>/dev/null; }
LB=$(libs "$BUILD_IMAGE"); LS=$(libs "$SHIP_IMAGE"); [ -n "$LB" ] && [ "$LB" = "$LS" ] || die "library versions differ between BUILD_IMAGE and SHIP_IMAGE (a bundle is keyed to them): build=$LB ship=$LS"
echo "images: build=$BUILD_IMAGE ship=$SHIP_IMAGE, same libraries $LS"
ALL=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits) || die "nvidia-smi failed"
IFS=, read -r -a GL <<< "$G"; [ ${#GL[@]} = "$NGPU" ] || die "$S needs exactly $NGPU GPU(s), got ${#GL[@]} ($G)"
U=()
for g in "${GL[@]}"; do g=${g// /}
  l=$(awk -F', *' -v g="$g" '$1 == g || $2 == g' <<< "$ALL"); [ -n "$l" ] || die "GPU $g not on this host (nvidia-smi --query-gpu=index,uuid --format=csv)"
  u=$(awk -F', *' '{print $2}' <<< "$l"); mem=$(awk -F', *' '{print $3}' <<< "$l")
  [ "$mem" -lt 1024 ] || die "GPU $g ($u) has $mem MiB in use: the build needs idle cards (stop what runs there first)"
  U+=("$u"); done
[ "$(printf '%s\n' "${U[@]}" | sort -u | wc -l)" = "$NGPU" ] || die "the same GPU is named twice ($G)"
GPUS=$(IFS=,; echo "${U[*]}"); echo "GPUs: $GPUS ($(nvidia-smi --query-gpu=name --format=csv,noheader -i "${U[0]}"))"
MD=$(cd "$MD" 2>/dev/null && pwd) || die "MODEL_DIR not found"
[ -f "$MD/config.json" ] || die "$MD/config.json missing"
CS=$(sha256sum < "$MD/config.json" | cut -c1-64)
if [ "$S" = nemotron ]; then [ "$CS" = "$NEMOTRON_CONFIG_SHA256" ] || die "$MD/config.json sha256 $CS is not Nemotron 3.5 Super VL NVFP4 @59b87216 ($NEMOTRON_CONFIG_SHA256)"
  [ -f "$MD/model.safetensors.index.json" ] || die "$MD/model.safetensors.index.json missing"; fi
echo "model: $MD -> /models/$RT (config.json sha256 $CS; the bundle is keyed to it)"
FIPS=(); if [ -n "${FIPS_ENABLED_OVERRIDE_FILE:-}" ]; then
  [ -f "$FIPS_ENABLED_OVERRIDE_FILE" ] && [ "$(cat "$FIPS_ENABLED_OVERRIDE_FILE")" = 0 ] || die "FIPS_ENABLED_OVERRIDE_FILE must be a readable file containing 0"
  FIPS=(-v "$FIPS_ENABLED_OVERRIDE_FILE:/proc/sys/crypto/fips_enabled:ro"); fi
mkdir -p "$O" && O=$(cd "$O" && pwd); [ -z "$(ls -A "$O")" ] || die "OUT_DIR $O is not empty (use a new directory: nothing is overwritten)"
free=$(df -Pk "$O" | awk 'NR==2{print int($4/1048576)}'); [ "$free" -ge 20 ] || die "OUT_DIR has $free GB free, needs >= 20 GB"
mkdir -p "$O/logs" "$O/capture" "$O/verify"; chmod 0777 "$O/capture" "$O/verify" "$O/logs"   # the containers run as uid 2000
echo "out: $O ($free GB free); serve flags ($S): ${FL[*]}"
printf '%s\n' "${FL[@]}" > "$O/logs/serve_flags.txt"

# ---- kb PHASE IMAGE BUNDLE_DIR(in /out) LOG : one hardened, offline container running the image's kernel_bundle.sh
kb(){ local ph=$1 img=$2 bd=$3 log=$4 n=build-bundle-$S-$1
  # the kit's prewarm_sampler.py (image copy + Nemotron VL's llm_config vocab) replaces the image's, read-only
  local pw=(-v "$HERE/bundle_tools/prewarm_sampler.py:$TOOLS/bundle/prewarm_sampler.py:ro")
  # verify (and production, create-nemotron.sh): FlashInfer load-only for the bundled JIT modules, see bundle_tools/vllm_cuda_fi_load_only.py
  [ "$ph" = verify ] && pw+=("${FI_LOAD_ONLY[@]}")
  docker rm -f "$n" >/dev/null 2>&1 || true
  docker run -d --pull never --name "$n" --label vllm-cuda.created-by=build-bundle.sh \
    --user 2000:0 --read-only --tmpfs "/tmp:rw,exec,nosuid,size=$TMPFS_SIZE" \
    --cap-drop ALL --cap-add SYS_PTRACE --security-opt no-new-privileges \
    "${IPC[@]}" --ulimit memlock=-1:-1 --network none --gpus "\"device=$GPUS\"" \
    -v "$MD:/models/$RT:ro" -v "$O:/out" "${pw[@]}" ${FIPS[@]+"${FIPS[@]}"} \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf-home -e XDG_CACHE_HOME=/tmp/xdg-cache \
    -e VLLM_HOST_IP=127.0.0.1 -e RUNNER=process -e WORK=/tmp/work -e SRVDIR=/tmp/vllm-cuda-srv \
    -e MODEL_DIR="/models/$RT" -e BUNDLE_DIR="$bd" -e NAME="$NAME" -e PORT=8000 "${SWEEP[@]}" \
    -e SOAK_MIN="$VERIFY_SOAK_MIN" -e BUILD_SOAK_MIN="$BUILD_SOAK_MIN" -e BUNDLE_ENV="$BENV" -e VERIFY_STRICT=0 \
    --entrypoint bash "$img" -c 'mkdir -p /tmp/work/pub && exec bash '"$TOOLS"'/bundle/kernel_bundle.sh "$0" -- "$@"' "$ph" "${FL[@]}" > /dev/null \
    || die "docker run of the $ph container failed (above)"
  trap 'docker rm -f '"$n"' >/dev/null 2>&1; echo "ERROR: interrupted; container '"$n"' removed" >&2; exit 1' INT TERM
  local t=$(date +%s) last=
  while docker ps -q -f "name=^$n$" | grep -q .; do sleep 60
    last=$(docker logs "$n" 2>&1 | grep -E '^(boot |SWEEP|STATS|BOOT FAILED|BUNDLE|soak|verify with)' | tail -1 | cut -c1-150 || true)
    echo "  ... $ph $(el $t)${last:+ | $last}"; done
  docker logs "$n" > "$log" 2>&1; local rc; rc=$(docker inspect -f '{{.State.ExitCode}}' "$n" 2>/dev/null || echo 1)
  docker rm -f "$n" > /dev/null 2>&1; trap - INT TERM
  echo "  $ph container exited rc=$rc after $(el $t); log $log"; tail -4 "$log" | sed 's/^/  | /'; }

# ---- 2 capture on the BUILD image
TB=$O/capture/$B.tar.gz
if [ -n "${VERIFY_ONLY:-}" ]; then
  step 2/4 "capture SKIPPED (VERIFY_ONLY): using $VERIFY_ONLY" "<1 min"
  tar -tzf "$VERIFY_ONLY" "$B/manifest.json" >/dev/null 2>&1 || die "VERIFY_ONLY tarball has no $B/manifest.json (not a $S capture tarball)"
  cp "$VERIFY_ONLY" "$TB"; echo "capture skipped; tarball from $VERIFY_ONLY" > "$O/logs/capture.log"
else
[ "$S" = nemotron ] && E2="30-60 min" || E2="10-20 min"
step 2/4 "capture: $BUILD_IMAGE boots $S with the production flags + traffic sweep" "$E2"
kb build "$BUILD_IMAGE" "/out/capture/$B" "$O/logs/capture.log"
grep -q "BUNDLE BUILD PASS" "$O/logs/capture.log" || die "capture FAILED: $(grep -E 'BOOT FAILED|BUNDLE BUILD' "$O/logs/capture.log" | tail -2 | cut -c1-300) -- logs: $O/logs/capture.log, $O/capture/$B/serve_build.log"
fi
[ -s "$TB" ] || die "capture wrote no tarball ($TB)"
SHA=$(sha256sum < "$TB" | cut -c1-64); echo "bundle tarball $(du -h "$TB" | cut -f1) sha256 $SHA"
grep -m1 -E '^manifest:' "$O/logs/capture.log" | cut -c1-300 || true

# ---- 3 strict verify on the SHIP image, from a fresh extraction of that tarball
[ "$S" = nemotron ] && E3="25-45 min" || E3="10-15 min"
step 3/4 "verify: $SHIP_IMAGE (no compiler) boots from a fresh extraction of the tarball, same flags + sweep + ${VERIFY_SOAK_MIN}-min soak" "$E3"
tar -xzf "$TB" -C "$O/verify" && chmod -R a+rwX "$O/verify" || die "could not extract $TB"
kb verify "$SHIP_IMAGE" "/out/verify/$B" "$O/logs/verify.log"
V=$(grep -E "^BUNDLE VERIFY" "$O/logs/verify.log" | tail -1)
[[ "$V" == "BUNDLE VERIFY PASS"* ]] || die "verify FAILED: ${V:-no verdict line} -- logs: $O/logs/verify.log, $O/verify/$B/serve_verify.log"
[ "$(sha256sum < "$TB" | cut -c1-64)" = "$SHA" ] || die "the tarball changed during verify"

# ---- 4 publish
step 4/4 "publish" "<1 min"
cp "$TB" "$O/$B.tar.gz"; ( cd "$O" && sha256sum "$B.tar.gz" > "$B.tar.gz.sha256" )
{ echo "service: $S"; echo "built: $(date -u +%FT%TZ) in $(el $T0)"; echo "build image: $BUILD_IMAGE"; echo "ship image (verified): $SHIP_IMAGE"
  echo "libraries: $LS"; echo "gpus: $GPUS"; echo "model: /models/$RT config.json sha256 $CS"; echo "serve flags: ${FL[*]}"
  echo "bundle: $B.tar.gz sha256 $SHA"; echo "verify: $V"; } > "$O/BUILD_BUNDLE_$S.txt"
echo "$V" | cut -c1-400
echo "BUILD-BUNDLE PASS $S: $O/$B.tar.gz sha256 $SHA (total $(el $T0))"
