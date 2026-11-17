#!/bin/bash
# create-aux.sh SERVICE GPU_UUID[,GPU_UUID...] [INSTANCE]   SERVICE = shrew-ocr | embed | rerank   -- one aux service, the image pinned in prod_env.sh.
# Several comma-separated UUIDs = one endpoint with that many data-parallel replicas (--data-parallel-size = count; RUNBOOK
# "Shared GPUs and data parallelism"; a DP>1 service needs `nobundle`, the bundles are keyed to one replica).
# Serve flags are the kit's create-<service>.sh flags (shape flags identical: check with aux_flag_diff.py). Memory: the KV cache
# is an explicit --kv-cache-memory-bytes, because each card is shared with an Inkling rank; --gpu-memory-utilization
# (AUX_GPU_MEMORY_UTILIZATION, default 0.15) is then only vLLM's startup free-memory check. Private IPC, all capabilities dropped.
# MPS=1 (prod_env.sh): MPS client capped at AUX_MPS_PCT % of the SMs, host IPC namespace (RUNBOOK "MPS").
# shrew-ocr serves the checkpoint chosen by SHREW_VARIANT (prod_env.sh): fp8-v0.5 by default, gptq-v0.4 = the previous line.
# The kernel bundle (built and verified on this image on an RTX PRO 6000) is mounted read-only; the entrypoint refuses to
# start if it does not match (KERNEL_BUNDLE_REQUIRED=1). To run without a bundle (kernels compile at first use): AUX_NO_BUNDLE=1.
set -euo pipefail
. "$(dirname "$0")/prod_env.sh"
S=${1:?shrew-ocr|embed|rerank}; G=${2:?GPU UUID}; I=${3:-}; : "${AUX_ROOT:?dir holding the aux -runtime model dirs}"
# A kernel bundle is tied to the vllm version string of the image it was built on: bundles built on another image are refused
# at boot (loud, by design). The bundles shipped with this kit were built on the image prod_env.sh pins.
. "$(dirname "$0")/aux_flags.sh"   # per-service model dir + serve flags (one definition with build-bundle.sh)
aux_service "$S" || die "unknown service $S"
N=vllm-$S${I:+-$I}; refuse_existing "$N"
# list first, then grep: `nvidia-smi | grep -q` under pipefail can fail falsely (grep exits at the match, nvidia-smi gets SIGPIPE)
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader); IFS=, read -r -a GPU_LIST <<< "$G"
[ ${#GPU_LIST[@]} -gt 0 ] || die "no GPUs specified"
for gpu in "${GPU_LIST[@]}"; do grep -qx "$gpu" <<< "$UUIDS" || die "GPU $gpu not on this host"; done
[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq "${#GPU_LIST[@]}" ] || die "duplicate GPU UUID"
DP=${#GPU_LIST[@]}
DP_FLAGS=(); [ "$DP" -eq 1 ] || DP_FLAGS=(--data-parallel-size "$DP")
[ -f "$AUX_ROOT/$M/config.json" ] || die "$AUX_ROOT/$M/config.json missing"
B=(); if [ "${AUX_NO_BUNDLE:-0}" != 1 ]; then
  BD=${AUX_BUNDLES:-$WORKDIR/aux-bundles}/bundle-$S; [ "$S" = shrew-ocr ] && BD=${AUX_BUNDLES:-$WORKDIR/aux-bundles}/$SHREW_BUNDLE; [ -f "$BD/manifest.json" ] || die "$BD/manifest.json missing (extract the bundle tarball of this service there, see RUNBOOK 'Aux kernel bundles', or AUX_NO_BUNDLE=1)"
  B=(-v "$BD:/opt/kernel-bundles/bundle-$S:ro" -e KERNEL_BUNDLE=/opt/kernel-bundles -e KERNEL_BUNDLE_REQUIRED=1 -e KERNEL_BUNDLE_STRICT=0); fi
# MPS=1: an MPS client capped at AUX_MPS_PCT % of the SMs. MPS clients must share the daemon's IPC namespace, so the private
# IPC namespace of this container becomes the host's (the only hardening that loosens; caps stay dropped). UNTESTED on real docker.
mps_args aux; IPC=private; [ "$MPS" = 1 ] && IPC=host
# docker's stderr goes to a file: on failure it is the ERROR line (docker_run_failed in prod_env.sh), on success it is shown as is
DR_ERR=$(mktemp); trap 'rm -f "$DR_ERR"' EXIT
docker run -d --pull never --name "$N" \
  --label "vllm-cuda.service=$S" --label "vllm-cuda.image-digest=$IMAGE_PIN" --label vllm-cuda.created-by=create-aux.sh \
  --restart "${RESTART_POLICY:-no}" \
  --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid,size=16g \
  --cap-drop ALL --security-opt no-new-privileges \
  --ipc $IPC --shm-size $SHM \
  --gpus "\"device=$G\"" \
  --network "$NETWORK" \
  -v "$AUX_ROOT/$M:/models/$M:ro" "${B[@]}" \
  ${FIPS_MOUNTS[@]+"${FIPS_MOUNTS[@]}"} ${MPS_ARGS[@]+"${MPS_ARGS[@]}"} \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf-home -e XDG_CACHE_HOME=/tmp/xdg-cache \
  "$IMAGE" \
  serve "/models/$M" --served-model-name "$S" --port 8000 ${DP_FLAGS[@]+"${DP_FLAGS[@]}"} --gpu-memory-utilization "${AUX_GPU_MEMORY_UTILIZATION:-0.15}" --kv-cache-memory-bytes $(( KV * 1048576 )) "${F[@]}" \
  2>"$DR_ERR" || docker_run_failed "$N" create-aux.sh "$DR_ERR"
docker_run_ok "$DR_ERR"
health "$N" 8000 "$S" "${HEALTH_TIMEOUT_S:-1200}" $K
mps_verify "$N"
# grep reads the whole log: `grep -m1` closes the pipe early, docker logs dies of SIGPIPE and pipefail printed a false WARNING
if [ "${AUX_NO_BUNDLE:-0}" != 1 ]; then l=$(docker logs "$N" 2>&1 | grep "kernel bundle OK" || true)
  if [ -n "$l" ]; then head -1 <<< "$l" | cut -c1-200; else echo "WARNING: no 'kernel bundle OK' line in $N"; fi; fi
