# stub_lib.sh -- sourced by run_all when INKLING_TESTS_STUB is set (rehearsal without GPUs): servers become stub_server.py,
# nvidia-smi / moe_race become stubs. Records every server's start command in $W/pub/stub_cmds.log.
STUBDIR=$(dirname "${BASH_SOURCE[0]}")
srv_start(){ local n=$1 m=$2 b=$3 envs=$4; shift 5; srv_stop $n
  echo "srv_start $n model=$m env=[$envs] args=[$*]" >> $W/pub/stub_cmds.log
  ( python3 $STUBDIR/stub_server.py $PORT > $SRVDIR/$n.log 2>&1 < /dev/null & echo $! > $SRVDIR/$n.pid ); sleep 1; }
srv_stop(){ local p=$(cat $SRVDIR/$1.pid 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; rm -f $SRVDIR/$1.pid; sleep 0.5; }
