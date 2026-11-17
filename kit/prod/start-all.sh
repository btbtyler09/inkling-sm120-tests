#!/bin/bash
# start-all.sh [status]  -- bring up every service of this box IN ORDER: Inkling first, then Nemotron (only when NEMOTRON_GPUS
# is set in site.env), then each line of AUX_SERVICES (site.env).
# Use it for the first start (runbook steps 3 and 4 in one command) AND after a reboot or a stop. For each service:
#   container missing -> create it (create-inkling.sh / create-aux.sh);  stopped -> docker start;  running -> left alone.
# Then it waits until the service is healthy and sends one request, before the next service starts. It stops at the first
# service that fails, and starts nothing after it. It never removes or replaces a container.
#   AUX_SERVICES in site.env: entries separated by ';', each `SERVICE GPU_UUID[,GPU_UUID...] [INSTANCE] [nobundle]`, e.g.
#   AUX_SERVICES="shrew-ocr GPU-aaaa 1; shrew-ocr GPU-bbbb 2; rerank GPU-cccc; embed GPU-cccc"
#   or data parallel: AUX_SERVICES="shrew-ocr GPU-a,GPU-b,GPU-c,GPU-d,GPU-e nobundle; rerank GPU-f,GPU-g nobundle; embed GPU-h"
#   (`nobundle` = start that one without a kernel bundle, e.g. a shrew-ocr checkpoint that has no bundle yet)
# `./start-all.sh status` only lists the services and their state; it starts nothing.
set -euo pipefail
. "$(dirname "$0")/prod_env.sh"
MODE=${1:-start}; [ "$MODE" = start ] || [ "$MODE" = status ] || die "usage: start-all.sh [status]"
# a failed `docker container inspect` still prints an empty line: take the output only when the command succeeded
state(){ local s; if s=$(docker container inspect -f '{{.State.Status}}' "$1" 2>/dev/null) && [ -n "$s" ]; then echo "$s"; else echo missing; fi; }

# one line per service: NAME|KIND|create command
LIST=("vllm-inkling|chat|./create-inkling.sh")
# Nemotron 3.5 Super VL (create-nemotron.sh, TP2 on NEMOTRON_GPUS): right after Inkling, when NEMOTRON_GPUS is set in site.env
[ -z "${NEMOTRON_GPUS:-}" ] || LIST+=("vllm-nemotron|chat|./create-nemotron.sh")
IFS=';' read -ra ENTRIES <<< "${AUX_SERVICES:-}"
for e in "${ENTRIES[@]}"; do
  read -ra w <<< "$e"; [ ${#w[@]} -gt 0 ] || continue
  s=${w[0]}; g=${w[1]:-}; inst=; nb=
  [ -n "$g" ] || die "AUX_SERVICES entry '$e' has no GPU UUID"
  for x in "${w[@]:2}"; do if [ "$x" = nobundle ]; then nb=1; elif [[ "$x" =~ ^[0-9]+$ ]]; then inst=$x; else die "AUX_SERVICES entry '$e': unknown word '$x'"; fi; done
  case $s in shrew-ocr) k=chat ;; embed) k=embed ;; rerank) k=rerank ;; *) die "AUX_SERVICES entry '$e': unknown service '$s'" ;; esac
  LIST+=("vllm-$s${inst:+-$inst}|$k|${nb:+AUX_NO_BUNDLE=1 }./create-aux.sh $s $g${inst:+ $inst}")
done
[ ${#LIST[@]} -gt 1 ] || echo "note: AUX_SERVICES is empty in site.env: only Inkling is handled"
dup=$(printf '%s\n' "${LIST[@]}" | cut -d'|' -f1 | sort | uniq -d | head -1); [ -z "$dup" ] || die "AUX_SERVICES names $dup twice (give copies an INSTANCE number)"

if [ "$MODE" = status ]; then
  for l in "${LIST[@]}"; do IFS='|' read -r n k c <<< "$l"; printf '%-22s %-10s %s\n' "$n" "$(state "$n")" "$c"; done; exit 0
fi

cd "$HERE"; done_=()
# MPS=1: the host MPS daemon comes first (after a reboot it is gone; containers started without it would run unattached)
if [ "$MPS" = 1 ]; then mps_up || ./mps.sh start || { echo "STOP: MPS=1 but the MPS daemon could not be started (./mps.sh start, as root). Nothing started."; exit 1; }; fi
for l in "${LIST[@]}"; do
  IFS='|' read -r n k c <<< "$l"; st=$(state "$n")
  echo "=== $n ($st)"
  case $st in
    missing) env $c || { echo "STOP: $n failed (above). Started before it: ${done_[*]:-none}. Not started: everything after $n."; exit 1; } ;;   # create-*.sh waits for health and sends the request
    running|exited|created)
      [ "$st" = running ] || docker start "$n" >/dev/null
      if [ "$n" = vllm-inkling ]; then p=$PORT; m=$SERVED_NAME; t=${HEALTH_TIMEOUT_S:-3600}
      elif [ "$n" = vllm-nemotron ]; then p=${NEMOTRON_PORT:-8000}; m=nemotron-3.5-super-vl; t=${HEALTH_TIMEOUT_S:-1800}; else p=8000; m=${n#vllm-}; m=${m%-[0-9]*}; t=${HEALTH_TIMEOUT_S:-1200}; fi
      ( health "$n" "$p" "$m" "$t" "$k" ) || { echo "STOP: $n failed (above). Started before it: ${done_[*]:-none}. Not started: everything after $n."; exit 1; } ;;
    *) echo "STOP: $n is in state '$st'. Look at it (docker logs $n) and remove it yourself before running this again."; exit 1 ;;
  esac
  done_+=("$n")
done
echo "ALL UP in order: ${done_[*]}"
