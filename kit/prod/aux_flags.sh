# aux_flags.sh -- sourced by create-aux.sh and build-bundle.sh: ONE definition of each aux service's model dir and serve flags,
# so a kernel bundle is built with exactly the flags the service runs. Needs prod_env.sh-style SHREW_DIR / SHREW_FLAGS set first
# (prod_env.sh sets them from SHREW_VARIANT; build-bundle.sh sets them the same way).
# aux_service SERVICE: sets M (dir name under AUX_ROOT = path in the container /models/$M), KV (MiB default), SHM, K (health kind),
# F (serve flags after the memory flags). Returns 1 for an unknown service.
aux_service(){
  case $1 in
    shrew-ocr) M=$SHREW_DIR; KV=${KV_MIB:-6144}; SHM=8g; K=chat   # checkpoint and its flags: SHREW_VARIANT in prod_env.sh
               F=(${SHREW_FLAGS[@]+"${SHREW_FLAGS[@]}"} --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 2048 --limit-mm-per-prompt '{"image":1}' --no-enable-prefix-caching
                  --chat-template-content-format openai) ;;   # text is always handed to the chat template as a list of parts, as in training
                  # (a template whose string branch indents the text would otherwise prepend 8 spaces); not a kernel-bundle key
    embed)     M=nemotron-embed-vl-1b-runtime; KV=${KV_MIB:-2048}; SHM=4g; K=embed
               F=(--runner pooling --max-model-len 10240 --limit-mm-per-prompt '{"image":1}' --trust-remote-code) ;;
    rerank)    M=nemotron-rerank-vl-1b-runtime; KV=${KV_MIB:-2048}; SHM=4g; K=rerank
               F=(--runner pooling --max-model-len 10240 --limit-mm-per-prompt '{"image":1}' --trust-remote-code) ;;
    *) return 1 ;;
  esac; }
