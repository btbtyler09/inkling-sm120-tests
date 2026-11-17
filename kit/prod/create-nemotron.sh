#!/bin/bash
# create-nemotron.sh -- create the `vllm-nemotron` container: NVIDIA Nemotron 3.5 Super VL NVFP4 (text + image + video), TP2 on the
# two cards of NEMOTRON_GPUS (UUIDs or nvidia-smi indices), on the box's one image (IMAGE / EXPECT_DIGEST in site.env, checked by
# prod_env.sh; it must be ink11nemo-minimal or later: its OVERLAY.txt must carry the ink11nemo section).
# Serve flags (mtp3 default, 262144 context, 64 sequences, FP8 KV, CPU KV offload) on vLLM 0.34, defined once in
# nemotron_flags.sh (build-bundle.sh builds the kernel bundle with the same definition). NEMOTRON_GPUS must not overlap GPUS.
# Same hardening as the kit's other create scripts (uid 2000, read-only root, all capabilities dropped except SYS_PTRACE for the
# tensor-parallel IPC, host IPC, unlimited memlock for the pinned CPU KV pool, FIPS mount from site.env). Refuses to replace a
# container. Every failure prints an ERROR line and exits 1.
# Image: ink11nemo = ink11kva + 0071 (eager image preprocessing, no compiler) + 0089 (forced tool choice streams carry no content,
# opt-in: VLLM_CUDA_FORCED_TOOL_STREAM_NO_CONTENT=1, set here only). No --trust-remote-code: transformers 5.22 ships the config.
# Kernel bundle (bundle-nemotron: built with build-bundle.sh nemotron on this image, verified on -minimal) is mounted read-only;
# the entrypoint refuses to start if it does not match (KERNEL_BUNDLE_REQUIRED=1). NEMOTRON_NO_BUNDLE=1: no bundle, FlashInfer
# kernels outside the AOT set then cannot compile on this compiler-free image (do not use in production).
# Settings (site.env or the command line):
#   NEMOTRON_GPUS        2 GPUs, comma separated (required)
#   NEMOTRON_DIR         the runtime model dir (config.json sha256 must be 35189a80..., the checkpoint the bundle was built with)
#   NEMOTRON_BUNDLE      the extracted bundle dir (default $WORKDIR/nemotron-bundles/bundle-nemotron)
#   NEMOTRON_SPEC        mtp3 (default; native MTP, 3 draft tokens) | none (needs NEMOTRON_NO_BUNDLE=1: the bundle is mtp3's)
#   NEMOTRON_OFFLOAD_GIB 64 (default) total GiB of pinned host RAM for the CPU KV pool over both ranks (32 GiB/rank); 0 = off
#   NEMOTRON_PORT        container port (default 8000); NEMOTRON_PUBLISH host port to publish (default none)
set -euo pipefail
. "$(dirname "$0")/prod_env.sh"
. "$(dirname "$0")/nemotron_flags.sh"   # the serve flags + shared checks (one definition with build-bundle.sh)
N=vllm-nemotron
: "${NEMOTRON_GPUS:?set NEMOTRON_GPUS in site.env: the 2 cards Inkling does not use}" "${NEMOTRON_DIR:?set NEMOTRON_DIR in site.env: the Nemotron runtime model dir}"
[ -z "${NEMOTRON_IMAGE:-}${NEMOTRON_DIGEST:-}" ] || die "NEMOTRON_IMAGE / NEMOTRON_DIGEST are gone: Nemotron runs on the box's one IMAGE (site.env IMAGE + EXPECT_DIGEST). Remove them from site.env"
NEMOTRON_SPEC=${NEMOTRON_SPEC:-mtp3}; NEMOTRON_OFFLOAD_GIB=${NEMOTRON_OFFLOAD_GIB:-64}; NEMOTRON_NO_BUNDLE=${NEMOTRON_NO_BUNDLE:-0}
NEMOTRON_PORT=${NEMOTRON_PORT:-8000}; NEMOTRON_BUNDLE=${NEMOTRON_BUNDLE:-$WORKDIR/nemotron-bundles/bundle-nemotron}
case $NEMOTRON_NO_BUNDLE in 0|1) ;; *) die "NEMOTRON_NO_BUNDLE must be 0 or 1 (got '$NEMOTRON_NO_BUNDLE')" ;; esac
[[ "$NEMOTRON_PORT" =~ ^[0-9]+$ ]] || die "NEMOTRON_PORT must be a port number (got '$NEMOTRON_PORT')"
[ "$NEMOTRON_SPEC" != none ] || [ "$NEMOTRON_NO_BUNDLE" = 1 ] || die "NEMOTRON_SPEC=none has no kernel bundle (bundle-nemotron is built for mtp3): set NEMOTRON_NO_BUNDLE=1 too"
nemotron_flags || exit 1   # prints its own ERROR line
nemotron_image "$IMAGE"
nemotron_gpus
echo "tensor parallel 2 on NEMOTRON_GPUS=$NEMOTRON_GPUS"
refuse_existing "$N"
nemotron_model
RT=$NEMOTRON_RT
B=(); if [ "$NEMOTRON_NO_BUNDLE" != 1 ]; then
  [ -f "$NEMOTRON_BUNDLE/manifest.json" ] || die "$NEMOTRON_BUNDLE/manifest.json missing (extract bundle-nemotron.tar.gz under $(dirname "$NEMOTRON_BUNDLE"), RUNBOOK 'Nemotron'; or NEMOTRON_NO_BUNDLE=1)"
  B=(-v "$NEMOTRON_BUNDLE:/opt/kernel-bundles/bundle-nemotron:ro" -e KERNEL_BUNDLE=/opt/kernel-bundles -e KERNEL_BUNDLE_REQUIRED=1 -e KERNEL_BUNDLE_STRICT=0)
else echo "WARNING: NEMOTRON_NO_BUNDLE=1: no kernel bundle (kernels outside FlashInfer's AOT set cannot compile on this image)"; fi
DR_ERR=$(mktemp); trap 'rm -f "$DR_ERR"' EXIT
docker run -d --pull never --name "$N" \
  --label vllm-cuda.service=nemotron --label "vllm-cuda.image-digest=$IMAGE_PIN" --label vllm-cuda.created-by=create-nemotron.sh \
  --restart "${RESTART_POLICY:-no}" \
  --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid,size=64g \
  --cap-drop ALL --cap-add SYS_PTRACE --security-opt no-new-privileges \
  --ipc host --ulimit memlock=-1:-1 \
  --gpus "\"device=$NEMOTRON_GPUS\"" \
  --network "$NETWORK" ${NEMOTRON_PUBLISH:+-p "$NEMOTRON_PUBLISH:$NEMOTRON_PORT"} \
  -v "$NEMOTRON_DIR:/models/$RT:ro" "${B[@]}" \
  ${FIPS_MOUNTS[@]+"${FIPS_MOUNTS[@]}"} \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf-home -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e VLLM_SKIP_P2P_CHECK=0 "${NEMOTRON_ENV[@]}" \
  "$IMAGE" \
  serve "/models/$RT" --served-model-name "$NEMOTRON_SERVED" --port "$NEMOTRON_PORT" "${NEM_FLAGS[@]}" \
  2>"$DR_ERR" || docker_run_failed "$N" create-nemotron.sh "$DR_ERR"
docker_run_ok "$DR_ERR"
echo "created $N: NEMOTRON_SPEC=$NEMOTRON_SPEC NEMOTRON_OFFLOAD_GIB=$NEMOTRON_OFFLOAD_GIB bundle=$([ "$NEMOTRON_NO_BUNDLE" = 1 ] && echo none || echo "$NEMOTRON_BUNDLE") image=$IMAGE ($IMAGE_PIN)"
health "$N" "$NEMOTRON_PORT" "$NEMOTRON_SERVED" "${HEALTH_TIMEOUT_S:-1800}" chat
# Boot verdict from the log (counts only; an ERROR line names what to do). The container is left running on an ERROR.
LOG=$(docker logs "$N" 2>&1); c() { grep -c -- "$1" <<< "$LOG" || true; }
grep -E "GPU KV cache size|Maximum concurrency|kernel bundle OK|SimpleCPUOffloadConnector" <<< "$LOG" | cut -c1-200 | sort -u | head -6 || true
BAD=0
if [ "$NEMOTRON_NO_BUNDLE" != 1 ] && [ "$(c "kernel bundle OK")" = 0 ]; then echo "ERROR: no 'kernel bundle OK' line in $N: the bundle was not applied (docker logs $N). The container is left running."; BAD=1; fi
if [ "$(c "RUNTIME CUDA COMPILE")" != 0 ]; then echo "ERROR: $N attempted a runtime CUDA compile (grep 'RUNTIME CUDA COMPILE' in docker logs $N): the bundle does not cover this config. The container is left running."; BAD=1; fi
if [ "$NEMOTRON_OFFLOAD_GIB" != 0 ] && [ "$(c "SimpleCPUOffloadConnector")" = 0 ]; then echo "ERROR: NEMOTRON_OFFLOAD_GIB=$NEMOTRON_OFFLOAD_GIB but no SimpleCPUOffloadConnector line in the log: CPU KV offload is NOT on. The container is left running."; BAD=1; fi
[ "$BAD" = 0 ] || exit 1
echo "NEMOTRON OK"
