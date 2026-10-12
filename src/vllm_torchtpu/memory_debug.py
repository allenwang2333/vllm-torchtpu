# SPDX-License-Identifier: Apache-2.0
"""Debug-only host DRAM instrumentation and experiments (b/565207901).

Everything here is off unless its environment variable is set to "1", so a
build of this branch behaves like its base by default. The flags are read
from ``os.environ`` directly rather than ``vllm_torchtpu.envs`` so they stay
out of the TPU compile-cache key.

Instrumentation:
  VLLM_TPU_MEMDBG_TRACE=1      log RSS, torch_tpu compile-cache entries and
                               live compiled executables at startup stages,
                               and trim the allocator once startup is done.
  VLLM_TPU_MEMDBG_HEAP_DUMP=1  also write a tcmalloc heap profile per stage
                               (needs LD_PRELOAD=libtcmalloc_and_profiler).
                               Only rank 0 of DP rank 0 of each role dumps;
                               file names carry role and rank, so prefill and
                               decode sharing one output dir no longer clobber
                               each other.

Experiments:
  VLLM_TPU_MEMDBG_DROP_SERIALIZE_ENTRY=1  (A1) after the compiler adaptor has
      taken TpuBackend's bundled entry, drop the ``serialize`` closures that
      keep it -- and the second executable it carries -- alive.
  VLLM_TPU_MEMDBG_CLEAR_COMPILE_CACHE=1   (A2) evict torch_tpu's in-memory
      CompilationCache once capture_model is done, releasing executables no
      Python object holds any more (profile_run's throwaway). Not done right
      after profile_run: that would free its HBM program region before the
      KV budget probe and grow the KV cache past what capture_model expects.
  VLLM_TPU_MEMDBG_RELEASE_STAGER_POOL=1   (C1) free the expert staging pool's
      host buffers once weights are on the device.
"""

import contextlib
import ctypes
import gc
import os
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


TRACE = _flag("VLLM_TPU_MEMDBG_TRACE")
HEAP_DUMP = _flag("VLLM_TPU_MEMDBG_HEAP_DUMP")
DROP_SERIALIZE_ENTRY = _flag("VLLM_TPU_MEMDBG_DROP_SERIALIZE_ENTRY")
CLEAR_COMPILE_CACHE = _flag("VLLM_TPU_MEMDBG_CLEAR_COMPILE_CACHE")
RELEASE_STAGER_POOL = _flag("VLLM_TPU_MEMDBG_RELEASE_STAGER_POOL")

_TCMALLOC_PATHS = (
    "/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4",
    "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4",
)

_tag = "unknown"
_dumps_heap = False
_heap_profiler: Any = None


def configure(role: str, dp_rank: int, rank: int) -> None:
    """Name this worker's records; called once from TPUWorker.__init__."""
    global _tag, _dumps_heap
    _tag = f"{role}_dp{dp_rank}_r{rank}"
    _dumps_heap = HEAP_DUMP and dp_rank == 0 and rank == 0
    if TRACE or HEAP_DUMP:
        logger.info(
            "[MEMDBG] tag=%s heap_dump=%s drop_serialize_entry=%s "
            "clear_compile_cache=%s release_stager_pool=%s",
            _tag,
            _dumps_heap,
            DROP_SERIALIZE_ENTRY,
            CLEAR_COMPILE_CACHE,
            RELEASE_STAGER_POOL,
        )


def _proc_mem_mb() -> tuple[float, float]:
    rss = anon = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("RssAnon:"):
                anon = int(line.split()[1]) / 1024
    return rss, anon


def _compile_cache_entries() -> int:
    import torch

    return len(torch.tpu._get_cache_stats().per_entry_stats)


def _live_compiled_executables() -> int:
    """TorchTpuCompiledExecutable objects still reachable by the GC.

    A gc scan rather than torch_tpu's ``_COMPILED_EXECUTABLES`` registry,
    which the pinned torch_tpu predates. Startup-only, so the cost is fine.
    """
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    return sum(isinstance(o, TorchTpuCompiledExecutable) for o in gc.get_objects())


def _heap_dump(stage: str) -> None:
    global _heap_profiler
    if _heap_profiler is None:
        prefix = os.path.join(
            os.environ.get("VLLM_TPU_MEMDBG_HEAP_DIR", "/cdk-outputs"),
            f"heap_{_tag}",
        )
        lib = ctypes.CDLL(_TCMALLOC_PATHS[0])
        lib.HeapProfilerStart.argtypes = [ctypes.c_char_p]
        lib.HeapProfilerDump.argtypes = [ctypes.c_char_p]
        lib.HeapProfilerStart(prefix.encode())
        _heap_profiler = lib
        logger.info("[MEMDBG] heap profiler started, prefix=%s", prefix)
    _heap_profiler.HeapProfilerDump(stage.encode())


def stage(name: str) -> None:
    """Record one startup stage: RSS, compile-cache and executable counts."""
    if not (TRACE or _dumps_heap):
        return
    gc.collect()
    rss, anon = _proc_mem_mb()
    logger.info(
        "[MEMDBG] stage=%s tag=%s VmRSS=%.1f MB RssAnon=%.1f MB "
        "compile_cache_entries=%d live_compiled_executables=%d",
        name,
        _tag,
        rss,
        anon,
        _compile_cache_entries(),
        _live_compiled_executables(),
    )
    if _dumps_heap:
        _heap_dump(name)


def trim_allocator(name: str) -> None:
    """Return freed heap to the OS so RSS reflects what is still live."""
    if not TRACE:
        return
    gc.collect()
    before, _ = _proc_mem_mb()
    for path in _TCMALLOC_PATHS:
        try:
            ctypes.CDLL(path).MallocExtension_ReleaseFreeMemory()
            break
        except (OSError, AttributeError):
            continue
    with contextlib.suppress(AttributeError):  # malloc_trim is glibc-only
        ctypes.CDLL(None).malloc_trim(0)
    after, _ = _proc_mem_mb()
    logger.info(
        "[MEMDBG] trim=%s tag=%s VmRSS %.1f -> %.1f MB", name, _tag, before, after
    )


def maybe_clear_compile_cache(reason: str) -> None:
    """(A2) Evict torch_tpu's CompilationCache.

    The cache shares ownership of every executable with its Python holder, so
    eviction frees only programs nothing else references. A live one evicted
    here is reloaded from the tier-2 disk cache on its next miss.
    """
    if not CLEAR_COMPILE_CACHE:
        return
    import torch

    before = _compile_cache_entries()
    torch.tpu._clear_cache()
    gc.collect()
    logger.info(
        "[MEMDBG] cleared compile cache after %s: %d -> %d entries",
        reason,
        before,
        _compile_cache_entries(),
    )


def maybe_drop_serialize_refs(compiled_fn: Any) -> None:
    """(A1) Drop TpuBackend's ``serialize`` closures once the entry is taken.

    TpuBackend attaches ``serialize = lambda: entry`` both to the runnable it
    returns and to the aot_autograd wrapper that runnable closes over. The
    entry holds a pickle-reconstructed copy of the executable, so either
    closure keeps a second program alive for the life of the process.
    """
    if not DROP_SERIALIZE_ENTRY:
        return
    holders = [compiled_fn]
    for cell in getattr(compiled_fn, "__closure__", None) or ():
        try:
            holders.append(cell.cell_contents)
        except ValueError:  # empty cell
            continue
    dropped = 0
    for obj in holders:
        if "serialize" in getattr(obj, "__dict__", {}):
            del obj.serialize
            dropped += 1
    logger.debug("[MEMDBG] dropped %d serialize closure(s)", dropped)
