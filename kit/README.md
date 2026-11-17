# vllm-cuda serving kit (ink11nemo, 2026-11-17)

Create scripts and the kernel-bundle builder for the one-image layout: Inkling, Nemotron 3.5 Super VL, shrew-ocr and
embed/rerank all run on `btbtyler09/vllm-cuda@sha256:ab4a135936f387010f8f408304ff8736a27fce6aeba236287be05437e781f58b`
(ink11nemo-r3-minimal: Trivy + Grype 0 CRITICAL/HIGH; no video and no mp4/aac audio; host driver >= 580; see the RUNBOOK section). Kernel bundles are built with the build-only image
`btbtyler09/vllm-cuda@sha256:feef2c370dca9724e378db3ed8e688ada69dafbcfadcae74f681aad5a67c23d5` (ink11nemo-full, never serve it).

- `prod/RUNBOOK_BUNDLES.md`: pull, build a bundle (`prod/build-nemotron-bundle.sh` / `prod/build-bundle.sh`), copy, start, first-boot checks.
- `prod/create-*.sh`, `prod/start-all.sh`: the services (copy `prod/site.env.example` to `prod/site.env` first).
- `prod/post_live.py`: post-boot checks against a running endpoint.

Keep the `prod/` and `sat/` layout: the scripts source each other by relative path. Nothing here is GPU-validated with this
image yet; see the RUNBOOK section.
