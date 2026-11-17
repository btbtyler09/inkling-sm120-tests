"""FlashInfer load-only for kernel bundles on the compiler-free image (active only with VLLM_CUDA_FI_LOAD_ONLY=1).

Why: FlashInfer 0.7.3 trusts only AOT artifacts without building (JitSpec.try_load). For a JIT module it always goes through
build(): it rewrites build.ninja with the CURRENT absolute paths and runs ninja. bundle_entry.py copies the bundle into a fresh
/tmp/kcache-XXXX on every boot, so the paths differ from the capture, ninja decides the bundled objects are stale and starts
nvcc, which the minimal image does not have (exit 127; the bundle VERIFY failure of 2026-11-17).
What: when a JIT module's .so is already present (it came from the bundle, whose manifest pins this exact FlashInfer version,
GPU and serve flags; the module name encodes its parameters), load that .so directly and never run ninja. A module that is NOT
in the bundle still goes to build() and fails loudly, as before.
How it gets in: the kit bind-mounts this file and vllm_cuda_fi_load_only.pth ("import vllm_cuda_fi_load_only") read-only into
the image's site-packages, so every Python process (API server, engine core, TP workers) runs it at startup. It only installs
an import hook; flashinfer.jit.core is patched when something imports it."""
import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "flashinfer.jit.core"


def _patch(core):
    # build_and_load is defined once, on JitSpec (subclasses such as JitSpecNvcc override try_load, not this)
    if getattr(core.JitSpec, "_vllm_cuda_load_only", False):
        return
    orig = core.JitSpec.build_and_load

    def build_and_load(self):
        mod = self.try_load()                      # AOT artifacts first, unchanged
        if mod is not None:
            return mod
        so = getattr(self, "jit_library_path", None)
        if so is not None and so.exists():         # bundled JIT module: load it, never run ninja
            sys.stderr.write(f"vllm-cuda: flashinfer load-only {self.name}\n")
            return self.load()
        return orig(self)                          # not in the bundle: FlashInfer's own path (build -> fails loudly)

    core.JitSpec.build_and_load = build_and_load
    core.JitSpec._vllm_cuda_load_only = True


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("VLLM_CUDA_FI_LOAD_ONLY") == "1":
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
