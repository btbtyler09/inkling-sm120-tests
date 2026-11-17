# sat_flags.sh — sourced. THE production serve flags for Inkling TP8. The kernel bundle is tied to the shape flags here
# (TP, max-model-len, max-num-seqs, max-num-batched-tokens): change them and the bundle must be rebuilt.
# max-num-batched-tokens 4096 (2026-11-14; was 8192): the KV admission charge per request includes the short-conv state of
# 2 x this many in-flight tokens (32,768 B per position and layer, per card, not divided by TP). 4096 halves it (TP8 x 48 layers:
# 25.4 -> 13.4 GiB per card per 262,144-token request, computed); measured cost on 1 card / 6 layers: one long prompt prefills
# 2-7 % slower. EXTRA_SERVE_ARGS="--max-num-batched-tokens 8192" goes back (last flag wins).
KV_GIB=${KV_GIB:-24}                # per card; lower it if an aux tenant shares the card (set explicitly, never a fraction)
OFFLOAD_GIB=${OFFLOAD_GIB:-256}      # total across ranks, host RAM, simple CPU offload
FLAGS=(--tensor-parallel-size ${NGPU:-8} --max-model-len ${MAXLEN:-262144} --max-num-seqs ${MAXSEQ:-32} --max-num-batched-tokens 4096
       --kv-cache-memory-bytes $((KV_GIB * 1024 * 1024 * 1024)) --kv-offloading-size $OFFLOAD_GIB
       --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling)   # parsers: required by the application; they do not change kernels
# EXTRA_SERVE_ARGS: stock `vllm serve` flags appended as they are (space separated, no quoting), default none.
#   EXTRA_SERVE_ARGS="--kv-cache-dtype fp8"  -> FP8 (e4m3) KV cache: half the attention KV bytes per token (ledger 40). It is a
#   kernel shape flag: a kernel cache / bundle built without it compiles the FP8 attention kernels at first use. Remove it to go back.
read -r -a _EXTRA_SERVE_ARGS <<< "${EXTRA_SERVE_ARGS:-}"; FLAGS+=(${_EXTRA_SERVE_ARGS[@]+"${_EXTRA_SERVE_ARGS[@]}"})
# ALLREDUCE_IPC=1 (default): FlashInfer PCIe IPC all-reduce for TP ranks (vLLM default is 0 = NCCL on PCIe cards). UNMEASURED on
# 8 cards as of 11-13: the smoke step A/Bs it (ALLREDUCE_IPC=0) and the faster one ships; its tune file
# pcie_ipc_allreduce_tp_<gpu ids>.json is written at first boot. Changing this changes the bundle (comm path): rebuild.
export BUNDLE_ENV="VLLM_USE_SIMPLE_KV_OFFLOAD=1 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=${ALLREDUCE_IPC:-1}"
export MODEL_DIR=${MODEL_DIR:-/models/Inkling-2-Small} NAME=inkling PORT=8000 WORK=/tmp/work
