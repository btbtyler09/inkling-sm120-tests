#!/bin/bash
# create-inkling.sh -- create the `vllm-inkling` container: Inkling-2-Small, tensor parallel TP (site.env, default 8) on the TP
# cards of GPUS (UUIDs or nvidia-smi indices; prod_env.sh gpus_resolve), the image pinned in prod_env.sh.
#   TP4: TP=4 GPUS=<4 cards> KV_GIB=10 EXTRA_SERVE_ARGS="--kv-cache-dtype fp8" (site.env.example).
#   KV_GIB stays per card, OFFLOAD_GIB the total over the ranks; --max-model-len 262144 and --max-num-batched-tokens 4096 as at TP8.
# Same hardening as the kit's multi-GPU create scripts (uid 2000, read-only root, caps dropped except SYS_PTRACE for the
# tensor-parallel IPC, host IPC, unlimited memlock for the pinned CPU KV pool). Refuses to replace a container.
# The serve flags are ../sat/sat_flags.sh (one definition).
#   Forced tool calls (tool_choice required / a named function) are enforced by a grammar: ENFORCE_TOOL_CHOICE=1 (default here).
#   ENFORCE_TOOL_CHOICE=0 turns it off.
#   SPLIT_KV_STATIC=1 (default here): long-context decode uses the split attention kernel inside the CUDA graph
#   (ledger 7/40: without it decode slows with context length). SPLIT_KV_STATIC=0 turns it off.
#   EXTRA_SERVE_ARGS (site.env, default empty): stock `vllm serve` flags appended to the flags of sat_flags.sh, e.g.
#   EXTRA_SERVE_ARGS="--kv-cache-dtype fp8" for the FP8 KV cache. Remove the line and recreate to go back.
#   ALLREDUCE_IPC (site.env): tensor-parallel all-reduce.
#     0 (default) NCCL. No tuning at startup.
#     tune        ONE-TIME boot: FlashInfer PCIe IPC all-reduce, tuned once on this box. The tune file is written to
#                 $WORKDIR/inkling-cache (persistent). The boot loads pcie-tune-hook/ (PYTHONPATH): on this compiler-free
#                 image FlashInfer's timing delay kernel cannot be built, which disqualified every candidate and left the
#                 file empty, so every boot tuned again (RUNBOOK "All-reduce"). When it is healthy, set ALLREDUCE_IPC=tuned.
#     tuned       Every boot after `tune`: PCIe IPC all-reduce reading that file, no profiling. Refuses to start without
#                 the file; after boot it checks the log for cache hits and for any profiling (= the file did not match).
#     1           PCIe IPC all-reduce as the image does it: on this image the tuning fails and repeats at every start
#                 (hours at TP8). Kept only for comparison; do not use.
#   tune and tuned keep the GPU order of GPUS: the tune file is keyed on the ring order of the cards.
#   Settings (site.env or command line; prod_env.sh checks the values). Always passed explicitly, never baked into the image:
#     MAXSEQ=N            --max-num-seqs N (default 32)
#     DSPARK=1            --speculative-config dspark, 7 draft tokens; needs $MODEL_DIR/mtp.safetensors and the ink9 image (default 0)
#     REUSE_REL_LOGITS    INKLING2_REUSE_REL_LOGITS, default 1 since the ink11 kit (ink9 image); 0 turns it off
#     LOCAL_SPLIT_KV      INKLING2_LOCAL_SPLIT_KV, default 1 since the ink11 kit (ink9.1 image); 0 turns it off
#     GLOBAL_SPLIT_TUNED  INKLING2_GLOBAL_SPLIT_TUNED, default 1 since the ink11 kit (ink9.1 image); 0 turns it off
#     CONV_TAIL_CACHE     INKLING2_CONV_TAIL_CACHE, default 1 (ink11 tail-only conv state cache); 0 = dense cache (rollback)
#     MPS=1 (+ AUX_MPS_PCT)  the ranks are uncapped clients of the host MPS daemon (./mps.sh start; RUNBOOK "MPS")
#   $WORKDIR/inkling-cache is mounted as the kernel cache: kernels compiled at first use (no compiler needed, proven on one
#   card) persist there, so a restart does not recompile them.
set -euo pipefail
. "$(dirname "$0")/prod_env.sh"
gpus_resolve || exit 1   # GPUS -> TP UUIDs on this host (indices accepted), else an ERROR line
echo "tensor parallel $NGPU on GPUS=$GPUS"
refuse_existing vllm-inkling
[ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ] || die "MODEL_DIR=$MODEL_DIR has no config.json / index"
# DSPARK=1: the draft is the checkpoint's mtp.safetensors (not in the index; part of the HF repo and its SHA256SUMS)
[ "$DSPARK" = 0 ] || [ -f "$MODEL_DIR/mtp.safetensors" ] || die "DSPARK=1 but $MODEL_DIR/mtp.safetensors is missing (copy it from the checkpoint repo; check it against SHA256SUMS)"
ink9_check
mkdir -p "$WORKDIR/inkling-cache"
# chown through a root container: works for an operator who is not root on the host (a plain chown fails silently there)
chown 2000:0 "$WORKDIR/inkling-cache" 2>/dev/null \
  || docker run --rm --pull never --network none --user 0:0 -v "$WORKDIR/inkling-cache:/cache" --entrypoint chown "$IMAGE" 2000:0 /cache || true
docker run --rm --pull never --network none --user 2000:0 -v "$WORKDIR/inkling-cache:/cache" --entrypoint sh "$IMAGE" -c 'touch /cache/.w && rm /cache/.w' \
  || die "$WORKDIR/inkling-cache is not writable by uid 2000 (chown 2000:0 it)"
ALLREDUCE_IPC=${ALLREDUCE_IPC:-0}; AR_ENV=()
# vLLM tunes the PCIe all-reduce in its own setup step and writes pcie_ipc_allreduce_tp_<ranks>.json under the FlashInfer
# autotune cache (/cache/vllm/flashinfer_autotune_cache/<flashinfer>/<arch>/<config hash>/). The later whole-model autotune
# pass must skip the op: it re-profiles with other buckets at every all-reduce call. Setting the variable REPLACES vLLM's
# default skip list, which skips fp4_gemm for the CuTe-DSL NVFP4 kernel: keep it listed. tune and tuned use the same value,
# so both boots resolve the same config hash directory.
SKIP_PCIE=(-e "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=flashinfer::pcie_ipc_all_reduce,fp4_gemm")
TUNE_GLOB="$WORKDIR/inkling-cache/vllm/flashinfer_autotune_cache/*/*/*/pcie_ipc_allreduce_tp_$(seq -s- 0 $((NGPU - 1))).json"
tune_files() { local f; for f in $TUNE_GLOB; do [ -f "$f" ] && echo "$f"; done; return 0; }
tune_entries() { grep -c "flashinfer::pcie_ipc_all_reduce" "$1" || true; }
case $ALLREDUCE_IPC in
  0) IPC=0 ;;
  1) IPC=1; echo "WARNING: ALLREDUCE_IPC=1 tunes the PCIe all-reduce at every start and on this image the tuning never succeeds (hours at TP8); use tune once, then tuned" >&2 ;;
  tune) IPC=1; AR_ENV=("${SKIP_PCIE[@]}" -v "$HERE/pcie-tune-hook:/opt/pcie-tune-hook:ro" -e PYTHONPATH=/opt/pcie-tune-hook)
        [ -f "$HERE/pcie-tune-hook/sitecustomize.py" ] || die "$HERE/pcie-tune-hook/sitecustomize.py missing (kit incomplete)"
        old=$(tune_files); [ -z "$old" ] || echo "note: an earlier tune file exists and is extended/overwritten for the same keys: $old" ;;
  tuned) IPC=1; AR_ENV=("${SKIP_PCIE[@]}")
         files=$(tune_files); [ -n "$files" ] || die "ALLREDUCE_IPC=tuned but no tune file matches $TUNE_GLOB: run ALLREDUCE_IPC=tune once first"
         for f in $files; do n=$(tune_entries "$f"); [ "$n" -gt 0 ] || die "$f holds no pcie_ipc_all_reduce entry (an empty tune: run ALLREDUCE_IPC=tune again)"
           echo "tune file: $f ($n entries)"; done ;;
  *) die "ALLREDUCE_IPC must be 0 (NCCL, default), tune (one-time), tuned or 1 (got '$ALLREDUCE_IPC')" ;;
esac
AR_MODE=$ALLREDUCE_IPC
# P2P_DISABLE=1: NCCL moves tensor-parallel traffic through host shared memory (--ipc host) instead of GPU peer-to-peer DMA
# across the PCIe switches. At TP8 over PCIe vLLM's own custom all-reduce is already off (not fully connected), so with
# ALLREDUCE_IPC=0 this removes all GPU peer-to-peer traffic. NCCL_DEBUG=INFO with SUBSYS INIT,P2P,SHM logs the transport chosen
# per channel at boot only ("via SHM" expected, no "via P2P"). Stability A/B; costs decode/prefill speed (RUNBOOK "P2P").
P2P_DISABLE=${P2P_DISABLE:-0}; P2P_ENV=()
case $P2P_DISABLE in
  0) ;;
  1) [ "$IPC" = 0 ] || die "P2P_DISABLE=1 needs ALLREDUCE_IPC=0 (the PCIe IPC all-reduce is peer-to-peer by design)"
     P2P_ENV=(-e NCCL_P2P_DISABLE=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,P2P,SHM)
     echo "P2P_DISABLE=1: NCCL over host shared memory, no GPU peer-to-peer" ;;
  *) die "P2P_DISABLE must be 0 (default) or 1 (got '$P2P_DISABLE')" ;;
esac
export NGPU KV_GIB OFFLOAD_GIB MAXSEQ ALLREDUCE_IPC=$IPC   # sat_flags.sh reads ALLREDUCE_IPC as 0/1 and MAXSEQ
_PORT=$PORT; . "$HERE/../sat/sat_flags.sh"; PORT=$_PORT   # sat_flags.sh sets PORT=8000 for the pod scripts; keep the site value
# DSPARK=1: DSPARK_K draft tokens per step (default 7 = the checkpoint's block size 8 - 1; ink9/ink9.1 refuse any other
# count at boot, ink10+ takes 1..7). No spaces in the JSON: it stays one argument.
[ "$DSPARK" = 1 ] && FLAGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$DSPARK_K}")
# MPS=1: every TP rank is an MPS client, uncapped (no CUDA_MPS_ACTIVE_THREAD_PERCENTAGE). It already runs in the host IPC
# namespace. A rank that is NOT a client would time-slice against the MPS server instead of running beside the aux clients.
mps_args inkling
# docker's stderr goes to a file: on failure it is the ERROR line (docker_run_failed in prod_env.sh), on success it is shown as is
DR_ERR=$(mktemp); trap 'rm -f "$DR_ERR"' EXIT
docker run -d --pull never --name vllm-inkling \
  --label vllm-cuda.service=inkling --label "vllm-cuda.image-digest=$IMAGE_PIN" --label vllm-cuda.created-by=create-inkling.sh \
  --restart "${RESTART_POLICY:-no}" \
  --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid,size=64g \
  --cap-drop ALL --cap-add SYS_PTRACE --security-opt no-new-privileges \
  --ipc host --ulimit memlock=-1:-1 \
  --gpus "\"device=$GPUS\"" \
  --network "$NETWORK" ${PUBLISH:+-p "$PUBLISH:$PORT"} \
  -v "$MODEL_DIR:/models/inkling-2-small:ro" -v "$WORKDIR/inkling-cache:/cache" \
  ${FIPS_MOUNTS[@]+"${FIPS_MOUNTS[@]}"} ${MPS_ARGS[@]+"${MPS_ARGS[@]}"} \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf-home -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e VLLM_CUDA_CACHE_ROOT=/cache -e VLLM_SKIP_P2P_CHECK=0 ${P2P_ENV[@]+"${P2P_ENV[@]}"} \
  -e VLLM_USE_SIMPLE_KV_OFFLOAD=1 -e "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=$IPC" ${AR_ENV[@]+"${AR_ENV[@]}"} ${RENDER_ENV:+-e "$RENDER_ENV"} \
  -e "VLLM_INKLING2_ENFORCE_TOOL_CHOICE=${ENFORCE_TOOL_CHOICE:-1}" \
  -e "INKLING2_SPLIT_KV_STATIC=${SPLIT_KV_STATIC:-1}" \
  -e "INKLING2_REUSE_REL_LOGITS=$REUSE_REL_LOGITS" \
  -e "INKLING2_LOCAL_SPLIT_KV=$LOCAL_SPLIT_KV" -e "INKLING2_GLOBAL_SPLIT_TUNED=$GLOBAL_SPLIT_TUNED" \
  -e "INKLING2_CONV_TAIL_CACHE=$CONV_TAIL_CACHE" \
  "$IMAGE" \
  serve /models/inkling-2-small --served-model-name "$SERVED_NAME" --port "$PORT" "${FLAGS[@]}" \
  2>"$DR_ERR" || docker_run_failed vllm-inkling create-inkling.sh "$DR_ERR"
docker_run_ok "$DR_ERR"
[ -n "${EXTRA_SERVE_ARGS:-}" ] && echo "extra serve flags: $EXTRA_SERVE_ARGS"
echo "opt-ins: MAXSEQ=$MAXSEQ DSPARK=$DSPARK DSPARK_K=$DSPARK_K REUSE_REL_LOGITS=$REUSE_REL_LOGITS LOCAL_SPLIT_KV=$LOCAL_SPLIT_KV GLOBAL_SPLIT_TUNED=$GLOBAL_SPLIT_TUNED CONV_TAIL_CACHE=$CONV_TAIL_CACHE MPS=$MPS${AUX_MPS_PCT:+ AUX_MPS_PCT=$AUX_MPS_PCT}"
echo "shape: TP=$NGPU KV_GIB=$KV_GIB/card OFFLOAD_GIB=$OFFLOAD_GIB image=$IMAGE ($IMAGE_PIN)"
case $AR_MODE in tune) AR_NOTE=" + one-time PCIe all-reduce tuning (11 sizes)";; *) AR_NOTE="";; esac
echo "created vllm-inkling. First boot: weight load$AR_NOTE + first-use kernel compiles."
health vllm-inkling "$PORT" "$SERVED_NAME" "${HEALTH_TIMEOUT_S:-3600}" chat
mps_verify vllm-inkling
# no match must not turn a healthy boot into exit 1 (set -e + pipefail)
docker logs vllm-inkling 2>&1 | { grep -E "all-reduce backends|GPU KV cache size|SimpleCPUOffloadConnector: role" || echo "WARNING: no all-reduce / KV cache size line in the log (docker logs vllm-inkling)"; } | cut -c1-200 | sort -u | head -6
# PCIe tune verdict, from this boot's log and the file on disk (counts only: a FAIL line names what to do)
if [ "$AR_MODE" = tune ] || [ "$AR_MODE" = tuned ]; then
  LOG=$(docker logs vllm-inkling 2>&1)
  c() { grep -c -- "$1" <<< "$LOG" || true; }
  hook=$(c "pcie-tune-hook: FlashInfer autotuner delay kernel"); tuning=$(c "Tuning flashinfer::pcie_ipc_all_reduce")
  skipped=$(c "unsupported tactic(s) for flashinfer::pcie_ipc_all_reduce"); hits=$(c "Config cache hit for flashinfer::pcie_ipc_all_reduce")
  seed=$(c "running seed launch configurations"); nomatch=$(c "but it holds no entry for this workspace")
  init=$(c "Initialized FlashInfer PCIe IPC all-reduce")
  echo "pcie all-reduce: initialized=$init hook=$hook tuning_lines=$tuning failed_candidate_lines=$skipped cache_hits=$hits seed_warnings=$seed no_match_warnings=$nomatch"
  files=$(tune_files); for f in $files; do echo "tune file: $f ($(tune_entries "$f") entries)"; done
  # A precondition miss (world size, dtype, multi-node, no graph sizes) only logs a vLLM WARNING and falls back to NCCL; a
  # failed IPC workspace raises and the boot dies (health fails above). The fallback must not look like a tuned boot: exit 1.
  [ "$init" -gt 0 ] || { echo "ERROR: ALLREDUCE_IPC=$AR_MODE but the PCIe all-reduce did not initialize: the server runs on NCCL, NOT the tuned all-reduce."
    grep -E "FlashInfer PCIe IPC all-reduce" <<< "$LOG" | grep -vi "Initialized" | head -3 | cut -c1-220 | sed 's/^/  vLLM: /'
    echo "  Read docker logs vllm-inkling, or set ALLREDUCE_IPC=0 in site.env and recreate. The container is left running (on NCCL)."; AR_BAD=1; }
  if [ "$AR_MODE" = tune ]; then
    [ "$hook" -gt 0 ] || echo "FAIL: the tuning hook did not load (no 'pcie-tune-hook' line): the tune ran without it and is empty"
    [ "$skipped" = 0 ] || echo "FAIL: $skipped lines of failed candidates: the tuning still fails on this box (debug: FLASHINFER_LOGGING_LEVEL=DEBUG)"
    [ -n "$files" ] && [ "$(tune_entries "$(head -1 <<< "$files")")" -gt 0 ] || echo "FAIL: no tune entries written"
    echo "if no FAIL line above: set ALLREDUCE_IPC=tuned in site.env, docker rm -f vllm-inkling, ./create-inkling.sh"
  else
    # tuned = the tuned speed was asked for: a boot that re-profiled or runs on seed configs is not that, so exit 1 (below)
    [ "$tuning" = 0 ] || { echo "ERROR: ALLREDUCE_IPC=tuned but this boot profiled the all-reduce ($tuning lines): the tune file does not match this config (flags, GPU order or image changed), so this server is NOT running the tuned all-reduce. Re-tune: ALLREDUCE_IPC=tune, docker rm -f vllm-inkling, ./create-inkling.sh (or ALLREDUCE_IPC=0). The container is left running."; AR_BAD=1; }
    [ "$seed" = 0 ] && [ "$nomatch" = 0 ] || { echo "ERROR: ALLREDUCE_IPC=tuned but the server runs on FlashInfer's seed launch configurations (seed_warnings=$seed no_match_warnings=$nomatch): the tune file does not match this workspace, NOT tuned (seed is slower than NCCL here). Re-tune: ALLREDUCE_IPC=tune, docker rm -f vllm-inkling, ./create-inkling.sh (or ALLREDUCE_IPC=0). The container is left running."; AR_BAD=1; }
    [ "$hits" -gt 0 ] || echo "WARNING: no 'Config cache hit for flashinfer::pcie_ipc_all_reduce' line in the log"
  fi
  [ "${AR_BAD:-0}" = 0 ] || exit 1
fi
