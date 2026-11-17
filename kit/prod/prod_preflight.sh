#!/bin/bash
# prod_preflight.sh -- changes nothing and stops no service; run it while the box is still serving. Last line: PREFLIGHT PASS|FAIL.
# Its only writes: it creates $WORKDIR (empty, so the free-space check sees the right disk) and $WORKDIR/logs.
. "$(dirname "$0")/prod_env.sh"; f=0; chk(){ if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; f=1; fi; }
# GPUS -> UUIDs first (indices accepted), so the MPS and aux-overlap checks below compare UUIDs; the verdict is a chk further down
GPUS_RAW=${GPUS:-}; gpus_ok=0   # first call in a subshell for the message, second in this shell to keep the resolved GPUS
if gpus_err=$(gpus_resolve 2>&1); then gpus_ok=1; gpus_resolve; fi
[ -n "$EXPECT_DIGEST" ] && echo "info image pin: EXPECT_DIGEST=$EXPECT_DIGEST" || echo "info image pin: ink11 tag $IMAGE (no EXPECT_DIGEST)"
chk "image present by digest" 'docker image inspect "$IMAGE" >/dev/null 2>&1'
chk "image has no compiler" '! docker run --rm --pull never --network none --entrypoint sh "$IMAGE" -c "command -v gcc || command -v nvcc" >/dev/null 2>&1'
# FIPS mounts here too: on a FIPS host vLLM dies at import without the opt-out, so this check would fail although the create scripts work
# FIPS_ENABLED_OVERRIDE_FILE belongs on the FIPS host only. A kernel without /proc/sys/crypto (not FIPS-capable) has no mount point
# for it: every container start fails there. Then the two checks run without the mount and a WARN says why (HOST_CRYPTO_DIR: test hook).
if [ ${#FIPS_MOUNTS[@]} -gt 0 ] && [ ! -d "${HOST_CRYPTO_DIR:-/proc/sys/crypto}" ]; then
  echo "WARN FIPS checks: FIPS_ENABLED_OVERRIDE_FILE is set but this host has no ${HOST_CRYPTO_DIR:-/proc/sys/crypto} (kernel not FIPS-capable): Docker cannot mount the file over /proc/sys/crypto/fips_enabled, so create-inkling.sh / create-aux.sh would fail at docker run. Set it only on the FIPS host (RUNBOOK 'FIPS-mode hosts'); the two checks below run without the mount"
  FIPS_MOUNTS=()
fi
chk "image starts under the hardening flags (uid 2000, read-only, no caps, no-new-privileges)" 'docker run --rm --pull never --network none --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid --cap-drop ALL --security-opt no-new-privileges ${FIPS_MOUNTS[@]+"${FIPS_MOUNTS[@]}"} --entrypoint python3 "$IMAGE" -c "import vllm" >/dev/null 2>&1'
chk "TLS context initializes under container FIPS settings" 'docker run --rm --pull never --network none --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid --cap-drop ALL --security-opt no-new-privileges ${FIPS_MOUNTS[@]+"${FIPS_MOUNTS[@]}"} --entrypoint python3 "$IMAGE" -c "import ssl; ssl.create_default_context()" >/dev/null 2>&1'
n=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "RTX PRO 6000"); chk "GPUs: $n RTX PRO 6000 (need $NGPU)" '[ "$n" -ge "$NGPU" ]'
echo "info driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | tr '\n' ' ')(pods used 595.91.07)"
chk "weights: every shard in the index exists" 'python3 -c "import json,os,sys; d=sys.argv[1]; sys.exit(any(not os.path.exists(os.path.join(d,f)) for f in set(json.load(open(d+\"/model.safetensors.index.json\"))[\"weight_map\"].values())))" "$MODEL_DIR"'
if [ -f "$MODEL_DIR/SHA256SUMS" ] && [ "${SKIP_SHA:-0}" != 1 ]; then chk "weights sha256 (SHA256SUMS; about 20 min, SKIP_SHA=1 skips)" '(cd "$MODEL_DIR" && sha256sum -c --quiet SHA256SUMS)'; else echo "info weights sha256 not checked"; fi
chk "weights readable by uid 2000 / gid 0" 'docker run --rm --pull never --network none --user 2000:0 -v "$MODEL_DIR":/m:ro --entrypoint sh "$IMAGE" -c "head -c 1 /m/model-00001-of-00024.safetensors >/dev/null && head -c 1 /m/config.json >/dev/null"'
[ "$DSPARK" = 1 ] && chk "DSPARK=1: draft weights $MODEL_DIR/mtp.safetensors present" '[ -f "$MODEL_DIR/mtp.safetensors" ]'
{ [ "$DSPARK" = 1 ] || [ "$REUSE_REL_LOGITS" = 1 ]; } && chk "DSPARK/REUSE_REL_LOGITS: image carries the ink9 files (OVERLAY.txt)" 'docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -q dspark_portable /usr/local/share/vllm-cuda/OVERLAY.txt'
{ [ "$LOCAL_SPLIT_KV" = 1 ] || [ "$GLOBAL_SPLIT_TUNED" = 1 ]; } && chk "LOCAL_SPLIT_KV/GLOBAL_SPLIT_TUNED: image carries the ink9.1 files (OVERLAY.txt)" 'docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -q INKLING2_LOCAL_SPLIT_KV /usr/local/share/vllm-cuda/OVERLAY.txt'
# ink11: required when IMAGE is the ink11 tag (no EXPECT_DIGEST); with a digest pin, info only (a rollback image has none)
if [ -z "$EXPECT_DIGEST" ]; then chk "ink11 tag: image carries the ink11 (KVA) section (OVERLAY.txt)" 'docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -qF -- "$ink11_marker" /usr/local/share/vllm-cuda/OVERLAY.txt'
elif docker run --rm --pull never --network none --entrypoint grep "$IMAGE" -qF -- "$ink11_marker" /usr/local/share/vllm-cuda/OVERLAY.txt >/dev/null 2>&1; then echo "info image is ink11 (KVA); CONV_TAIL_CACHE=$CONV_TAIL_CACHE"
else echo "info image has no ink11 section (pre-ink11): CONV_TAIL_CACHE has no effect"; fi
if [ "$MPS" = 1 ]; then
  chk "MPS=1: nvidia-cuda-mps-control on the host" 'command -v nvidia-cuda-mps-control >/dev/null'
  chk "MPS=1: MPS control daemon up on $MPS_PIPE (./mps.sh start, as root)" 'mps_up'
  chk "MPS=1: cards in compute mode Default or Exclusive_Process" '! nvidia-smi --query-gpu=uuid,compute_mode --format=csv,noheader | awk -F", " -v g=",${GPUS:-}," "index(g, \",\"\$1\",\") && \$2 != \"Default\" && \$2 != \"Exclusive_Process\"" | grep -q .'
  echo "info MPS=1 AUX_MPS_PCT=$AUX_MPS_PCT: MPS=1 container path UNTESTED on real docker; proven only via the pod-side shim"
fi
echo "info opt-ins: MAXSEQ=$MAXSEQ DSPARK=$DSPARK REUSE_REL_LOGITS=$REUSE_REL_LOGITS LOCAL_SPLIT_KV=$LOCAL_SPLIT_KV GLOBAL_SPLIT_TUNED=$GLOBAL_SPLIT_TUNED CONV_TAIL_CACHE=$CONV_TAIL_CACHE MPS=$MPS TP=$NGPU"
chk "docker network $NETWORK exists" 'docker network inspect "$NETWORK" >/dev/null 2>&1'
mkdir -p "$WORKDIR"; free=$(df -BG --output=avail "$WORKDIR" 2>/dev/null | tail -1 | tr -dc 0-9); chk "WORKDIR free ${free:-0}G (need 20G)" '[ "${free:-0}" -ge 20 ]'
# Host memory budget (GiB). Pinned = never reclaimable; the rest are estimates from measured RSS (pods + MI100 workers).
tot=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
aux_n=0; aux_shm=0; IFS=';' read -ra _ae <<< "${AUX_SERVICES:-}"
for _e in "${_ae[@]}"; do read -ra _w <<< "$_e"; [ ${#_w[@]} -gt 1 ] || continue
  _r=$(awk -F, '{print NF}' <<< "${_w[1]}"); aux_n=$(( aux_n + _r ))
  case ${_w[0]} in shrew-ocr) aux_shm=$(( aux_shm + 8 )) ;; *) aux_shm=$(( aux_shm + 4 )) ;; esac; done
b_pin=$OFFLOAD_GIB; b_ink=$(( 20 * NGPU + 4 )); b_aux=$(( 10 * aux_n + aux_shm )); b_os=32; b_sum=$(( b_pin + b_ink + b_aux + b_os ))
echo "info host memory budget (GiB): pinned KV pool $b_pin | Inkling workers+engine ~$b_ink (20/rank + 4) | aux ~$b_aux ($aux_n replicas x 10 + shm $aux_shm) | OS/page cache headroom $b_os | total $b_sum of $tot"
chk "host memory budget ${b_sum}G <= 80% of ${tot}G RAM; lower OFFLOAD_GIB or the aux replicas in site.env if not" '[ $(( b_sum * 100 )) -le $(( tot * 80 )) ]'
# GPU topology and peer-to-peer matrices: the first thing to read on a new board (PCIe switches, sockets, ACS)
mkdir -p "$WORKDIR/logs"
{ nvidia-smi topo -m; echo; echo "== p2p read"; nvidia-smi topo -p2p r; echo; echo "== p2p write"; nvidia-smi topo -p2p w; } > "$WORKDIR/logs/topo.txt" 2>&1
sed -n '1,12p' "$WORKDIR/logs/topo.txt" | sed 's/^/info topo  /'
np=$(sed -n '/== p2p read/,/== p2p write/p' "$WORKDIR/logs/topo.txt" | grep -oE '\b(CNS|GNS|TNS|NS|NC|CS)\b' | wc -l)
echo "info p2p read: $np GPU pairs not OK (full matrix: $WORKDIR/logs/topo.txt). Not OK is fine with ALLREDUCE_IPC=0 (NCCL picks its path); ALLREDUCE_IPC tune/tuned needs every pair OK"
ALLREDUCE_IPC=${ALLREDUCE_IPC:-0}   # unset = 0 (as create-inkling.sh); under set -u an unset value aborted every check below
[ "$ALLREDUCE_IPC" = 0 ] || chk "ALLREDUCE_IPC=$ALLREDUCE_IPC: every GPU pair has P2P read" '[ "$np" = 0 ]'
# aux services on Inkling's cards without MPS time-slice the TP ranks: one slow rank slows all of them
_sh=; for _e in "${_ae[@]}"; do read -ra _w <<< "$_e"; for _u in ${_w[1]//,/ }; do [[ ",${GPUS:-}," == *",$_u,"* ]] && _sh="$_sh ${_w[0]}@${_u:0:12}"; done; done
[ -n "$_sh" ] && [ "$MPS" != 1 ] && echo "WARN aux on Inkling's GPUs without MPS:$_sh -- Inkling decode stalls while they compute. Put aux on other cards, or MPS=1 (RUNBOOK 'Shared GPUs')"
[ -d /var/log/journal ] && echo "info journald: persistent (/var/log/journal)" || echo "WARN journald is volatile: a host hang leaves no log. sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald (RUNBOOK first boot)"
chk "memlock unlimited inside a container" '[ "$(docker run --rm --pull never --network none --ulimit memlock=-1:-1 --entrypoint sh "$IMAGE" -c "ulimit -l")" = unlimited ]'
# GPUS (create-inkling.sh) and the aux prerequisites (create-aux.sh), so that step 1 finds what steps 3 and 4 would trip on
chk "GPUS: $NGPU distinct cards on this host for TP=$NGPU (GPUS=$GPUS_RAW${gpus_err:+; $gpus_err})" '[ "$gpus_ok" = 1 ]'
[ "$gpus_ok" = 1 ] || GPUS=
# a GPU inside a container (NVIDIA container toolkit) with create-inkling.sh's hardening, so a broken toolkit shows up before
# step 2 stops anything. Safe while serving: nvidia-smi -L only lists the card and allocates no GPU memory.
[ -n "$GPUS" ] && chk "GPU ${GPUS%%,*} visible inside a container under the hardening flags (nvidia-smi -L names an RTX PRO 6000)" 'docker run --rm --pull never --network none --user 2000:0 --read-only --tmpfs /tmp:rw,exec,nosuid --cap-drop ALL --security-opt no-new-privileges --gpus "\"device=${GPUS%%,*}\"" --entrypoint nvidia-smi "$IMAGE" -L 2>&1 | grep -q "RTX PRO 6000"'
if [ -n "${AUX_ROOT:-}" ]; then AI=$IMAGE
  iv=$(docker run --rm --pull never --network none --entrypoint python3 "$AI" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null)
  echo "info shrew-ocr checkpoint: SHREW_VARIANT=$SHREW_VARIANT -> $AUX_ROOT/$SHREW_DIR, bundle ${AUX_BUNDLES:-$WORKDIR/aux-bundles}/$SHREW_BUNDLE"
  for s in "shrew-ocr:$SHREW_DIR" embed:nemotron-embed-vl-1b-runtime rerank:nemotron-rerank-vl-1b-runtime; do m=${s#*:}; s=${s%%:*}
    chk "aux $s: $AUX_ROOT/$m/config.json readable by uid 2000 / gid 0" 'docker run --rm --pull never --network none --user 2000:0 -v "$AUX_ROOT/$m":/m:ro --entrypoint sh "$AI" -c "head -c 1 /m/config.json >/dev/null"'
    [ "${AUX_NO_BUNDLE:-0}" = 1 ] && continue; BD=${AUX_BUNDLES:-$WORKDIR/aux-bundles}/bundle-$s; [ "$s" = shrew-ocr ] && BD=${AUX_BUNDLES:-$WORKDIR/aux-bundles}/$SHREW_BUNDLE
    [ "$s" = shrew-ocr ] && [ "$SHREW_VARIANT" = fp8-v0.5 ] && chk "aux shrew-ocr: $SHREW_DIR/config.json is a compressed-tensors FP8 checkpoint" 'grep -q "\"quant_method\": \"compressed-tensors\"" "$AUX_ROOT/$m/config.json"'
    [ "$s" = shrew-ocr ] && [ -f "$AUX_ROOT/$m/SHA256SUMS" ] && [ "${SKIP_SHA:-0}" != 1 ] && chk "aux shrew-ocr: weights sha256 (SHA256SUMS)" '(cd "$AUX_ROOT/$m" && sha256sum -c --quiet SHA256SUMS)'
    bv=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["libs"]["vllm"])' "$BD/manifest.json" 2>/dev/null)
    chk "aux $s: bundle built for this image (bundle vllm '${bv:-no manifest}', image vllm '$iv'); else use the bundles shipped with this kit, or AUX_NO_BUNDLE=1" '[ -n "$bv" ] && [ "$bv" = "$iv" ]'
    # a bundle is also tied to the model's config.json (sha256): another checkpoint of the same model needs its own bundle
    bc=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["serve"]["model_config_sha256"])' "$BD/manifest.json" 2>/dev/null); mc=$(sha256sum "$AUX_ROOT/$m/config.json" 2>/dev/null | cut -d' ' -f1)
    chk "aux $s: bundle built for this checkpoint (config.json sha256 ${mc:0:12}, bundle ${bc:0:12}); else AUX_NO_BUNDLE=1 for this service until a bundle is built for it" '[ -n "$bc" ] && [ "$bc" = "$mc" ]'
  done; fi
# containers_before.txt is the rollback list: write it once. A second preflight after step 2 would replace it with a list
# that no longer has the stopped GPU services.
mkdir -p "$WORKDIR/logs"; [ -s "$WORKDIR/logs/containers_before.txt" ] || docker ps --format '{{.Names}} {{.Image}}' > "$WORKDIR/logs/containers_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv > "$WORKDIR/logs/gpus_before.csv"
echo "info running containers (rollback = these, untouched): $(cut -d' ' -f1 "$WORKDIR/logs/containers_before.txt" | tr '\n' ' ')"
[ $f = 0 ] && echo "PREFLIGHT PASS" || { echo "PREFLIGHT FAIL"; exit 1; }
