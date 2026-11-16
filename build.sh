#!/bin/bash
# build.sh TAG -- build the test image from inkling2-ink10-rc HEAD (model code) + tests/. Prints the image id.
set -eu
D=$(cd "$(dirname "$0")" && pwd); B=${VLLM_SRC:-$HOME/vllm-ink10-rc}; TAG=${1:?tag}   # VLLM_SRC: the vLLM tree (inkling2-ink10-rc)
rm -rf "$D/ovl"; mkdir -p "$D/ovl/inkling2"
git -C $B archive HEAD vllm/models/inkling2 | tar -x --strip-components=3 -C "$D/ovl/inkling2"
cp $B/packaging/inkling/ink10/ctx/inkling2/*.so "$D/ovl/inkling2/"
git -C $B show HEAD:vllm/models/inkling/nvidia/sconv_swa_attn.py > "$D/ovl/sconv_swa_attn.py"
echo "model code: inkling2-ink10-rc $(git -C $B rev-parse --short HEAD)" > "$D/tests/VERSION"
docker build -q -t "$TAG" "$D"
