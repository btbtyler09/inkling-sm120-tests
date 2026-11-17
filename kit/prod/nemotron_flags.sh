# nemotron_flags.sh -- sourced by create-nemotron.sh and build-nemotron-bundle.sh: ONE definition of Nemotron 3.5 Super VL's serve
# flags, so the kernel bundle is always built with exactly the flags the service runs (the bundle is refused otherwise).
# nemotron_flags: reads NEMOTRON_SPEC (mtp3|none) and NEMOTRON_OFFLOAD_GIB (whole GiB, 0 = off); sets NEM_FLAGS = every flag after
# `--port`. Shape flags (bundle key): TP, max-model-len, max-num-seqs, max-num-batched-tokens, kv-cache-dtype, chunked prefill,
# limit-mm-per-prompt, kernel-config, compilation-config, speculative-config. The rest (parsers, offload size, prefix caching,
# mamba options, media) are not part of the key but are kept identical anyway.
NEMOTRON_RT=nemotron-3.5-super-vl-runtime    # path inside the container: /models/$NEMOTRON_RT (a compile-hash factor: keep it)
NEMOTRON_SERVED=nemotron-3.5-super-vl
NEMOTRON_CONFIG_SHA256=35189a8038d1b1f934e58cba50eb7e97b42a4da569424e35b9fd1048058bc7ea   # nvidia/...-NVFP4 @59b87216 (upstream bytes)
nemotron_flags(){
  local spec=() off=()
  case ${NEMOTRON_SPEC:-mtp3} in
    mtp3) spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
    none) spec=() ;;
    *) echo "ERROR: NEMOTRON_SPEC must be mtp3 or none (got '${NEMOTRON_SPEC}')" >&2; return 1 ;;
  esac
  [[ "${NEMOTRON_OFFLOAD_GIB:-64}" =~ ^[0-9]+$ ]] || { echo "ERROR: NEMOTRON_OFFLOAD_GIB must be a whole number of GiB, 0 = off (got '${NEMOTRON_OFFLOAD_GIB}')" >&2; return 1; }
  [ "${NEMOTRON_OFFLOAD_GIB:-64}" = 0 ] || off=(--kv-offloading-size "${NEMOTRON_OFFLOAD_GIB:-64}")
  # NEMOTRON_KV_CACHE_BYTES (opt-in, per GPU): a FIXED KV pool instead of the 0.90-of-free-memory sizing, so every boot (capture,
  # verify, prod) gets the same block count and Triton specializes its kernels identically. Unset = today's behaviour. Capture,
  # verify and prod must all use the same value (the bundle's Triton kernels are only reused then).
  if [ -n "${NEMOTRON_KV_CACHE_BYTES:-}" ]; then
    [[ "$NEMOTRON_KV_CACHE_BYTES" =~ ^[0-9]+$ ]] || { echo "ERROR: NEMOTRON_KV_CACHE_BYTES must be a whole number of bytes (got '$NEMOTRON_KV_CACHE_BYTES')" >&2; return 1; }
    off+=(--kv-cache-memory-bytes "$NEMOTRON_KV_CACHE_BYTES"); fi
  NEM_FLAGS=(--tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 8192
    --gpu-memory-utilization 0.90 --kv-cache-dtype fp8
    --enable-prefix-caching --mamba-cache-mode align --enable-chunked-prefill
    --mamba-backend triton --mamba-ssm-cache-dtype float32
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
    --limit-mm-per-prompt '{"image":8,"video":1}'
    --video-pruning-rate 0.5 --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
    --kernel-config '{"moe_backend":"flashinfer_cutlass","enable_flashinfer_autotune":false}'
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    ${off[@]+"${off[@]}"} ${spec[@]+"${spec[@]}"}); }
# Container env both scripts pass to the server (offload backend; 0089 forced-tool streaming contract).
NEMOTRON_ENV=(-e VLLM_USE_SIMPLE_KV_OFFLOAD=1 -e VLLM_CUDA_FORCED_TOOL_STREAM_NO_CONTENT=1)
# The shared checks below need prod_env.sh sourced first (die, gpus_resolve, GPUS/NGPU).
# nemotron_image IMAGE: dies unless IMAGE is on this host and its OVERLAY.txt carries the ink11nemo section (0071/0089 files).
nemotron_image(){ local img=$1
  docker image inspect "$img" >/dev/null 2>&1 || die "image $img is not on this host (docker pull it first)"
  docker run --rm --pull never --network none --entrypoint grep "$img" -qF -- '--- ink11nemo layer' /usr/local/share/vllm-cuda/OVERLAY.txt \
    || die "$img has no ink11nemo section in OVERLAY.txt: not an ink11nemo build (Nemotron needs its 0071/0089 files; IMAGE in site.env must be ink11nemo-minimal)"; }
# nemotron_gpus: NEMOTRON_GPUS -> 2 UUIDs on this host, none of them in GPUS (Inkling's cards); dies otherwise.
nemotron_gpus(){ local gs=${GPUS:-} ns=$NGPU r g
  GPUS=$NEMOTRON_GPUS; NGPU=2; gpus_resolve || exit 1
  NEMOTRON_GPUS=$GPUS; GPUS=$gs; NGPU=$ns
  if [ -n "$gs" ]; then r=$(GPUS=$gs; NGPU=$ns; gpus_resolve >/dev/null 2>&1 && echo "$GPUS" || echo "$gs")
    for g in ${NEMOTRON_GPUS//,/ }; do [[ ",$r," != *",$g,"* ]] || die "NEMOTRON_GPUS card $g is also in GPUS (Inkling's cards)"; done; fi; }
# nemotron_model: NEMOTRON_DIR holds the checkpoint the bundle is keyed to (config.json sha256), with its index.
nemotron_model(){ local s
  [ -f "$NEMOTRON_DIR/config.json" ] && [ -f "$NEMOTRON_DIR/model.safetensors.index.json" ] || die "NEMOTRON_DIR=$NEMOTRON_DIR has no config.json / model.safetensors.index.json"
  s=$(sha256sum < "$NEMOTRON_DIR/config.json" | cut -c1-64)
  [ "$s" = "$NEMOTRON_CONFIG_SHA256" ] || die "NEMOTRON_DIR/config.json sha256 $s is not the checkpoint the bundle was built with ($NEMOTRON_CONFIG_SHA256)"; }
