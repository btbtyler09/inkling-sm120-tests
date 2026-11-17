# Loaded only by the ONE-TIME PCIe all-reduce tuning boot (create-inkling.sh, ALLREDUCE_IPC=tune), through PYTHONPATH.
#
# Why: FlashInfer's autotuner puts a GPU delay kernel in front of every timed launch. That kernel lives in the JIT module
# `trtllm_utils`, which the -minimal image does not ship and cannot build (no nvcc). Building it raises inside the timed
# window, so EVERY candidate of flashinfer::pcie_ipc_all_reduce is disqualified ("Skipped 49 unsupported tactic(s)"),
# no winner is stored, the tune file is written empty ("Saved 0 configs") and each later call profiles again.
#
# What: replace that delay with torch.cuda._sleep (a spin kernel compiled into torch itself), the same role and length
# (FlashInfer asks for 100 us before a graph replay, 5000 us before eager launches). Nothing else is changed. The
# measurement, the candidates, the cache keys and the file vLLM writes are FlashInfer's own.
import importlib.abc
import importlib.machinery
import sys

_TARGET = "flashinfer.autotuner.autotuner"


def _patch(mod):
    import torch

    cycles_per_us = {}

    def delay_kernel(stream_delay_micro_secs):
        dev = torch.cuda.current_device()
        if dev not in cycles_per_us:
            khz = getattr(torch.cuda.get_device_properties(dev), "clock_rate", 0) or 2_500_000
            cycles_per_us[dev] = max(1, khz // 1000)   # kHz -> cycles per microsecond (boost clock: a delay at least this long)
        torch.cuda._sleep(int(stream_delay_micro_secs) * cycles_per_us[dev])

    mod.delay_kernel = delay_kernel
    mod.get_trtllm_utils_module = lambda: None
    sys.stderr.write("pcie-tune-hook: FlashInfer autotuner delay kernel -> torch.cuda._sleep (one-time tuning boot)\n")


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
