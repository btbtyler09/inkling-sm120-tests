#!/bin/bash
# harvest-bundle.sh CAPTURE_TARBALL VERIFY_OUT_DIR NEW_DIR  -- make a bundle whose Triton kernels are the ones the SHIP image
# compiled itself during a verify run, for when capture (-full) and verify (-minimal) disagree on Triton cache keys.
#   CAPTURE_TARBALL  OUT_DIR/capture/bundle-<svc>.tar.gz of the capture (FlashInfer, vLLM/torch.compile, DeepGEMM, ... caches)
#   VERIFY_OUT_DIR   the OUT_DIR of a build-bundle.sh run whose verify left OUT_DIR/verify/triton-runtime/ (this kit's
#                    build-bundle.sh writes it: every Triton key dir compiled at runtime during that verify)
#   NEW_DIR          a new or empty directory: gets bundle-<svc>.tar.gz + .sha256 + HARVEST_<svc>.txt
# The capture's files are kept as they are; the runtime Triton key dirs are ADDED to cache/triton (a key dir that already
# exists is left alone). Absolute paths inside the copied Triton group files are rewritten at boot by the image's
# bundle_entry.py (relocate_triton_groups), as for any bundle. Then verify the new tarball (twice: proves it is boot-stable):
#   VERIFY_ONLY=NEW_DIR/bundle-<svc>.tar.gz ./build-bundle.sh <svc> GPUS MODEL_DIR <new OUT_DIR>
set -euo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
TB=${1:-}; VO=${2:-}; ND=${3:-}
[ -n "$TB" ] && [ -n "$VO" ] && [ -n "$ND" ] || die "usage: harvest-bundle.sh CAPTURE_TARBALL VERIFY_OUT_DIR NEW_DIR"
[ -s "$TB" ] || die "capture tarball $TB not found"
RT=$VO/verify/triton-runtime
[ -d "$RT" ] || die "$RT not found: run the verify with this kit's build-bundle.sh (it writes verify/triton-runtime)"
B=$(tar -tzf "$TB" | head -1 | cut -d/ -f1); [[ "$B" == bundle-* ]] || die "$TB: top directory '$B' is not bundle-<svc>"
tar -tzf "$TB" "$B/manifest.json" >/dev/null 2>&1 || die "$TB has no $B/manifest.json"
mkdir -p "$ND" && ND=$(cd "$ND" && pwd); [ -z "$(ls -A "$ND")" ] || die "NEW_DIR $ND is not empty (nothing is overwritten)"
W=$ND/.work; mkdir -p "$W"; trap 'rm -rf "$W"' EXIT
tar -xzf "$TB" -C "$W" || die "could not extract $TB"
mkdir -p "$W/$B/cache/triton"
before=$(find "$W/$B/cache/triton" -type f | wc -l)
added=0 kept=0
for k in "$RT"/*/; do k=${k%/}; [ -d "$k" ] || continue
  if [ -e "$W/$B/cache/triton/$(basename "$k")" ]; then kept=$((kept + 1)); else cp -r "$k" "$W/$B/cache/triton/" && added=$((added + 1)); fi; done
[ "$added" -gt 0 ] || die "no new Triton key dirs in $RT (nothing to harvest)"
after=$(find "$W/$B/cache/triton" -type f | wc -l)
chmod -R a+rX "$W/$B"
{ echo "harvested: $(date -u +%FT%TZ)"; echo "capture tarball: $TB sha256 $(sha256sum < "$TB" | cut -c1-64)"
  echo "runtime triton from: $RT"; echo "triton key dirs added: $added (already present: $kept); triton files $before -> $after"; } > "$W/$B/HARVEST.txt"
tar -C "$W" -czf "$ND/$B.tar.gz" "$B"
( cd "$ND" && sha256sum "$B.tar.gz" > "$B.tar.gz.sha256" )
cp "$W/$B/HARVEST.txt" "$ND/HARVEST_${B#bundle-}.txt"
echo "HARVEST OK $B: added $added Triton key dirs ($kept already present), triton files $before -> $after"
echo "  $ND/$B.tar.gz  sha256 $(cut -c1-64 "$ND/$B.tar.gz.sha256")"
echo "next: VERIFY_ONLY=$ND/$B.tar.gz ./build-bundle.sh ${B#bundle-} GPUS MODEL_DIR <new OUT_DIR>   (twice)"
