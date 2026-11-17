#!/bin/bash
# mps.sh start|stop|status  -- the ONE host MPS control daemon for all cards of GPUS (RUNBOOK "MPS"). Used only with MPS=1.
#   start   as root. Idempotent: a running daemon is left alone. Refuses a card whose compute mode is not Default or
#           Exclusive_Process (MPS cannot serve Prohibited cards; Exclusive_Thread is not a mode MPS supports). It does not
#           change compute modes. Pipe and log directories: $WORKDIR/mps/{pipe,log} (MPS_DIR overrides), owned by uid 2000,
#           the user every kit container runs as: the daemon starts one MPS server per user and all clients share it.
#   stop    as root. Refuses while a kit container is running (they are MPS clients: stop them first). `quit` makes the
#           daemon stop accepting clients and exit once its servers and clients have finished.
#   status  daemon up?, servers, per-card MPS clients (nvidia-smi type M+C), last server log lines.
# The daemon is a host process: it is not tied to any container and survives container restarts. It does NOT survive a host
# reboot: start-all.sh runs `mps.sh start` first when MPS=1.
# MPS=1 container path UNTESTED on real docker; proven only via the pod-side shim.
set -euo pipefail
. "$(dirname "$0")/prod_env.sh"
MODE=${1:-status}; gpus_resolve || exit 1   # GPUS (UUIDs or indices) -> the TP UUIDs
ctl(){ CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG nvidia-cuda-mps-control "$@"; }
need_root(){ [ "$(id -u)" = 0 ] || die "mps.sh $MODE must run as root (the daemon spawns the MPS server as the clients' user, uid 2000)"; }
kit_running(){ docker ps --filter label=vllm-cuda.created-by --format '{{.Names}}' 2>/dev/null | grep -E '^vllm-' || true; }
case $MODE in
  start)
    need_root
    if mps_up; then echo "mps: daemon already running on $MPS_PIPE"; exit 0; fi
    command -v nvidia-cuda-mps-control >/dev/null || die "nvidia-cuda-mps-control not found (it ships with the NVIDIA driver)"
    bad=$(nvidia-smi --query-gpu=uuid,compute_mode --format=csv,noheader | awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") && $2 != "Default" && $2 != "Exclusive_Process" {print $1" "$2}')
    [ -z "$bad" ] || die "compute mode not usable with MPS on: $bad (set Default or Exclusive_Process with nvidia-smi -c, yourself)"
    [ -e "$MPS_PIPE/control" ] && { echo "note: stale control socket from a dead daemon: removing $MPS_PIPE/control"; rm -f "$MPS_PIPE/control"; }
    mkdir -p "$MPS_PIPE" "$MPS_LOG"; chown 2000:0 "$MPS_PIPE" "$MPS_LOG"; chmod 0770 "$MPS_PIPE" "$MPS_LOG"
    CUDA_VISIBLE_DEVICES=$GPUS ctl -d || die "nvidia-cuda-mps-control -d failed (log: $MPS_LOG/control.log)"
    for _ in 1 2 3 4 5 6 7 8 9 10; do mps_up && break; sleep 1; done
    mps_up || die "the MPS daemon did not come up (log: $MPS_LOG/control.log)"
    echo "mps: daemon up on $MPS_PIPE for $GPUS (log $MPS_LOG)" ;;
  stop)
    need_root
    mps_up || { echo "mps: no daemon running"; exit 0; }
    r=$(kit_running); [ -z "$r" ] || die "kit containers still running (MPS clients): $(echo $r). Stop them first (docker stop ...), then mps.sh stop"
    echo quit | ctl; for _ in $(seq 1 30); do mps_up || break; sleep 1; done
    mps_up && die "daemon still running 30 s after quit (clients left? see $MPS_LOG/control.log)"
    echo "mps: daemon stopped" ;;
  status)
    if mps_up; then echo "mps: daemon UP on $MPS_PIPE"; else echo "mps: daemon DOWN (pipe dir $MPS_PIPE)"; fi
    [ "$(id -u)" = 0 ] && mps_up && echo "mps: servers: $(echo get_server_list | ctl 2>/dev/null | tr '\n' ' ')"
    echo "mps: GPU processes attached as MPS clients (M+C): $(nvidia-smi 2>/dev/null | grep -c 'M+C' || true)"
    [ -f "$MPS_LOG/server.log" ] && { echo "mps: last server log lines:"; tail -5 "$MPS_LOG/server.log"; } || true ;;
  *) die "usage: mps.sh start|stop|status" ;;
esac
