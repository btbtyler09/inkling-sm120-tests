# fi_load_only.sh (sourced): FI_LOAD_ONLY = the docker run arguments that make FlashInfer load a kernel bundle's JIT modules
# (their .so) without running ninja. Needed on the compiler-free image: bundle_entry.py copies the bundle into a fresh
# /tmp/kcache-XXXX per boot, FlashInfer 0.7.3 rewrites build.ninja with the new paths and ninja then tries to recompile with
# nvcc, which the image does not have (exit 127). See bundle_tools/vllm_cuda_fi_load_only.py. Used by create-nemotron.sh and
# build-bundle.sh's verify step.
_fi_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bundle_tools
_fi_sp=/opt/rt/lib/python3.12/site-packages
FI_LOAD_ONLY=(-v "$_fi_dir/vllm_cuda_fi_load_only.py:$_fi_sp/vllm_cuda_fi_load_only.py:ro"
              -v "$_fi_dir/vllm_cuda_fi_load_only.pth:$_fi_sp/vllm_cuda_fi_load_only.pth:ro"
              -e VLLM_CUDA_FI_LOAD_ONLY=1)
