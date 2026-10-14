# SPDX-License-Identifier: Apache-2.0
"""Debug-only host DRAM instrumentation for the host-DRAM fixes (b/565207901).

Off unless its environment variable is "1". Flags are read from
``os.environ`` directly (not ``vllm_torchtpu.envs``) so they stay out of the
TPU compile-cache key.

  VLLM_TPU_MEMDBG_TRACE=1       log RssAnon, tcmalloc in-use / page-heap free
                                bytes and live compiled executables at startup
                                stages, and around the C1 and G1 releases.
  VLLM_TPU_MEMDBG_PERIODIC_S=N  also log RssAnon/tcmalloc every N s at serving.

Ablations (turn one fix off to measure it against a run with all fixes):
  VLLM_TPU_MEMDBG_NO_A1D=1  copies of compiled artifacts reload the program.
  VLLM_TPU_MEMDBG_NO_C1=1   keep the expert staging pool after loading.
  VLLM_TPU_MEMDBG_NO_G1=1   do not return freed startup heap after warmup.
"""

import ctypes
import gc
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


TRACE = _flag("VLLM_TPU_MEMDBG_TRACE")
NO_A1D = _flag("VLLM_TPU_MEMDBG_NO_A1D")
NO_C1 = _flag("VLLM_TPU_MEMDBG_NO_C1")
NO_G1 = _flag("VLLM_TPU_MEMDBG_NO_G1")

# tcmalloc is LD_PRELOADed in our images; resolve from the process.
_process_symbols = ctypes.CDLL(None)
_tag = "unknown"


def configure(role: str, dp_rank: int, rank: int) -> None:
    """Name this worker's records; called once from TPUWorker.__init__."""
    global _tag
    _tag = f"{role}_dp{dp_rank}_r{rank}"
    if TRACE:
        logger.info(
            "[MEMDBG] tag=%s ablations={A1d off: %s, C1 off: %s, G1 off: %s}",
            _tag,
            NO_A1D,
            NO_C1,
            NO_G1,
        )


def _tcmalloc_property_mb(name: str) -> float:
    get = getattr(_process_symbols, "MallocExtension_GetNumericProperty", None)
    if get is None:
        return -1.0
    value = ctypes.c_size_t()
    get.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    if not get(name.encode(), ctypes.byref(value)):
        return -1.0
    return value.value / 2**20


def _rss_anon_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024
    return 0.0


def _live_compiled_executables() -> int:
    """Compiled TPU artifacts reachable by the GC (startup only; not frozen)."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    return sum(isinstance(o, TorchTpuCompiledExecutable) for o in gc.get_objects())


def stage(name: str, count_executables: bool = True) -> None:
    """Record one stage: RssAnon, tcmalloc in-use and free, executables."""
    if not TRACE:
        return
    executables = -1
    if count_executables:
        gc.collect()
        executables = _live_compiled_executables()
    logger.info(
        "[MEMDBG] stage=%s tag=%s RssAnon=%.1f MB tcmalloc_allocated=%.1f MB "
        "tcmalloc_pageheap_free=%.1f MB live_compiled_executables=%d",
        name,
        _tag,
        _rss_anon_mb(),
        _tcmalloc_property_mb("generic.current_allocated_bytes"),
        _tcmalloc_property_mb("tcmalloc.pageheap_free_bytes"),
        executables,
    )


def start_periodic_sampler() -> None:
    """Log RssAnon and tcmalloc state every VLLM_TPU_MEMDBG_PERIODIC_S s."""
    period = float(os.environ.get("VLLM_TPU_MEMDBG_PERIODIC_S", "0"))
    if not TRACE or period <= 0:
        return
    import threading
    import time

    def _loop() -> None:
        while True:
            time.sleep(period)
            stage("periodic", count_executables=False)

    threading.Thread(target=_loop, name="memdbg-sampler", daemon=True).start()

