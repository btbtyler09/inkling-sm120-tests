# sat_flags.sh -- sourced. The Inkling-2 serve flags the tests use (same shape flags as the production config).
# Inputs: NGPU, MAXLEN, MAXSEQ, KV_GIB (per GPU), OFFLOAD_GIB, EXTRA_SERVE_ARGS. Output: FLAGS, NAME, PORT.
KV_GIB=${KV_GIB:-24}
OFFLOAD_GIB=${OFFLOAD_GIB:-16}
FLAGS=(--tensor-parallel-size ${NGPU:-1} --max-model-len ${MAXLEN:-262144} --max-num-seqs ${MAXSEQ:-32} --max-num-batched-tokens 4096
       --kv-cache-memory-bytes $((KV_GIB * 1024 * 1024 * 1024)) --kv-offloading-size $OFFLOAD_GIB
       --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling)
read -r -a _EXTRA_SERVE_ARGS <<< "${EXTRA_SERVE_ARGS:-}"; FLAGS+=(${_EXTRA_SERVE_ARGS[@]+"${_EXTRA_SERVE_ARGS[@]}"})
export NAME=inkling PORT=8000
