#!/bin/bash
# build-nemotron-bundle.sh GPUS NEMOTRON_DIR OUT_DIR  = build-bundle.sh nemotron GPUS NEMOTRON_DIR OUT_DIR (see there; RUNBOOK
# "Build kernel bundles"). GPUS: the 2 idle cards (UUIDs or indices). Output: OUT_DIR/bundle-nemotron.tar.gz (+ .sha256).
exec "$(dirname "$0")/build-bundle.sh" nemotron "$@"
