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
                               (needs LD_PRELOAD=libtcmalloc.so.4, the image
                               default; do not preload a second gperftools lib).
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
  VLLM_TPU_MEMDBG_CLEAR_PRECOMPILE_CONTEXT=1  (A1b) clear torch's
      PrecompileContext once capture_model is done. AOTAutogradCache.save
      records a deepcopy of every bundled entry there -- for TPU entries a
      pickle round trip, i.e. one more loaded executable per graph -- for a
      mega-AOT serialization we disable (VLLM_USE_MEGA_AOT_ARTIFACT=0).

With VLLM_TPU_MEMDBG_TRACE=1, rank 0 also logs which objects hold the live
compiled executables after capture_model (``holders`` lines).
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
CLEAR_PRECOMPILE_CONTEXT = _flag("VLLM_TPU_MEMDBG_CLEAR_PRECOMPILE_CONTEXT")
RELEASE_ENTRY_COPIES = _flag("VLLM_TPU_MEMDBG_RELEASE_ENTRY_COPIES")
SHARE_EXECUTABLE_COPIES = _flag("VLLM_TPU_MEMDBG_SHARE_EXECUTABLE_COPIES")

# Resolve tcmalloc symbols from the allocator already in the process
# (LD_PRELOAD; libtcmalloc.so.4 in our images, which carries the heap
# profiler). dlopen-ing a second gperftools library -- even
# libtcmalloc_and_profiler -- runs its heap-checker constructor again and
# aborts the worker ("Heap-check constructor called twice").
_process_symbols = ctypes.CDLL(None)


def _tcmalloc_fn(name: str) -> Any:
    fn = getattr(_process_symbols, name, None)
    if fn is None:
        logger.warning("[MEMDBG] %s not found; is tcmalloc LD_PRELOADed?", name)
    return fn


_tag = "unknown"
_dumps_heap = False
_is_rank0 = False
_heap_profiler: Any = None


def configure(role: str, dp_rank: int, rank: int) -> None:
    """Name this worker's records; called once from TPUWorker.__init__."""
    global _tag, _dumps_heap, _is_rank0
    _tag = f"{role}_dp{dp_rank}_r{rank}"
    _is_rank0 = dp_rank == 0 and rank == 0
    _dumps_heap = HEAP_DUMP and _is_rank0
    if SHARE_EXECUTABLE_COPIES:
        _share_executable_copies()
    if TRACE or HEAP_DUMP:
        flags = {
            "heap_dump": _dumps_heap,
            "A1": DROP_SERIALIZE_ENTRY,
            "A1b": CLEAR_PRECOMPILE_CONTEXT,
            "A1c": RELEASE_ENTRY_COPIES,
            "A1d": SHARE_EXECUTABLE_COPIES,
            "A2": CLEAR_COMPILE_CACHE,
            "C1": RELEASE_STAGER_POOL,
        }
        logger.info("[MEMDBG] tag=%s flags=%s", _tag, flags)


def _share_executable_copies() -> None:
    """(A1d) Make copy/deepcopy of a TPU executable share the loaded program.

    torch copies OutputCode objects in BundledOutputCodeLoadable.pre_save (cold
    start) and deep-copies bundled entries in deserialize_bundled_cache_entry
    (warm start) so a later serialization starts from a pristine object. For
    TorchTpuCompiledExecutable the copy goes through __reduce__ -- serialize
    and load a second program -- although prepare_for_serialization is a no-op
    and a loaded executable is never mutated. Sharing removes the copy at its
    source instead of releasing it afterwards.
    """
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    TorchTpuCompiledExecutable.__copy__ = lambda self: self
    TorchTpuCompiledExecutable.__deepcopy__ = lambda self, memo: self


def _tcmalloc_allocated_mb() -> float:
    """Bytes the application holds in tcmalloc (excludes allocator caches)."""
    get = getattr(_process_symbols, "MallocExtension_GetNumericProperty", None)
    if get is None:
        return -1.0
    value = ctypes.c_size_t()
    get.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    if not get(b"generic.current_allocated_bytes", ctypes.byref(value)):
        return -1.0
    return value.value / 2**20


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


def _precompile_artifacts() -> int:
    from torch._dynamo.precompile_context import PrecompileContext

    return len(PrecompileContext._backend_artifacts_by_key)


def maybe_clear_precompile_context(reason: str) -> None:
    """(A1b) Drop the entry copies AOTAutogradCache.save parked in torch."""
    if not CLEAR_PRECOMPILE_CONTEXT:
        return
    from torch._dynamo.precompile_context import PrecompileContext

    before = _precompile_artifacts()
    PrecompileContext.clear()
    gc.collect()
    logger.info(
        "[MEMDBG] cleared PrecompileContext after %s: %d artifacts", reason, before
    )


def maybe_release_entry_copies(reason: str) -> None:
    """(A1c) Drop the executable copies that bundled AOTAutograd entries keep.

    AOTAutogradCache.save calls BundledOutputCodeLoadable.pre_save, which
    replaces the entry's ``result`` with ``copy(result)`` -- for TPU a
    serialize/load round trip, i.e. a second loaded executable -- so the
    entry can be pickled. The entry then stays reachable from
    SerializableCompiledFunction.serialize_fn closures (and TpuBackend's
    ``serialize`` lambdas) for the life of the process. Runtime wrappers hold
    the executable they run directly, so dropping ``result`` from every live
    entry once startup has saved its artifacts releases only the copies.
    """
    if not RELEASE_ENTRY_COPIES:
        return
    from torch._functorch._aot_autograd.aot_autograd_result import (
        BundledOutputCodeLoadable,
    )

    gc.collect()
    released = 0
    for obj in gc.get_objects():
        if isinstance(obj, BundledOutputCodeLoadable) and obj.result is not None:
            obj.result = None
            released += 1
    gc.collect()
    logger.info(
        "[MEMDBG] released %d bundled-entry executable copies after %s",
        released,
        reason,
    )


def _label(obj: Any) -> str:
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"{type(obj).__name__}:{obj.__qualname__}"
    return type(obj).__qualname__


def log_executable_holders(stage: str, depth: int = 6, top: int = 15) -> None:
    """Rank 0 only: histogram of referrer chains above each live executable.

    Answers "who keeps this program alive" for copies a fix was expected to
    release. Each direct referrer (up to three per executable) is followed
    ``depth`` levels up, taking the first non-frame referrer per level. One
    gc pass per level (get_referents over every tracked object) keeps this to
    a few seconds per level even on a multi-million-object heap.
    """
    if not (TRACE and _is_rank0):
        return
    import collections
    import inspect

    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    gc.collect()
    objects = [o for o in gc.get_objects() if not inspect.isframe(o)]
    mine = {id(objects)}

    def parents_of(targets: dict[int, Any]) -> dict[int, list[Any]]:
        found: dict[int, list[Any]] = collections.defaultdict(list)
        for o in objects:
            if id(o) in mine:
                continue
            for r in gc.get_referents(o):
                if id(r) in targets:
                    found[id(r)].append(o)
        return found

    exes = [o for o in objects if isinstance(o, TorchTpuCompiledExecutable)]
    mine.add(id(exes))
    direct = parents_of({id(e): e for e in exes})
    chains = [([_label(p)], p) for e in exes for p in direct.get(id(e), [])[:3]]
    for _ in range(depth - 1):
        ups = parents_of({id(cur): cur for _, cur in chains})
        extended = []
        for labels, cur in chains:
            above = ups.get(id(cur))
            if above:
                labels.append(_label(above[0]))
                cur = above[0]
            extended.append((labels, cur))
        chains = extended
    counts = collections.Counter(" <- ".join(labels) for labels, _ in chains)
    logger.info(
        "[MEMDBG] holders stage=%s tag=%s executables=%d", stage, _tag, len(exes)
    )
    for chain, count in counts.most_common(top):
        logger.info("[MEMDBG] holders %3d x %s", count, chain)


def _heap_dump(stage: str) -> None:
    global _heap_profiler
    if _heap_profiler is None:
        start = _tcmalloc_fn("HeapProfilerStart")
        dump = _tcmalloc_fn("HeapProfilerDump")
        if start is None or dump is None:
            _heap_profiler = False
            return
        prefix = os.path.join(
            os.environ.get("VLLM_TPU_MEMDBG_HEAP_DIR", "/cdk-outputs"),
            f"heap_{_tag}",
        )
        start.argtypes = [ctypes.c_char_p]
        dump.argtypes = [ctypes.c_char_p]
        start(prefix.encode())
        _heap_profiler = dump
        logger.info("[MEMDBG] heap profiler started, prefix=%s", prefix)
    if _heap_profiler:
        _heap_profiler(stage.encode())


def stage(name: str) -> None:
    """Record one startup stage: RSS, compile-cache and executable counts."""
    if not (TRACE or _dumps_heap):
        return
    gc.collect()
    rss, anon = _proc_mem_mb()
    logger.info(
        "[MEMDBG] stage=%s tag=%s VmRSS=%.1f MB RssAnon=%.1f MB "
        "tcmalloc_allocated=%.1f MB compile_cache_entries=%d "
        "live_compiled_executables=%d precompile_artifacts=%d",
        name,
        _tag,
        rss,
        anon,
        _tcmalloc_allocated_mb(),
        _compile_cache_entries(),
        _live_compiled_executables(),
        _precompile_artifacts(),
    )
    if _dumps_heap:
        _heap_dump(name)


def trim_allocator(name: str) -> None:
    """Return freed heap to the OS so RSS reflects what is still live."""
    if not TRACE:
        return
    gc.collect()
    before, _ = _proc_mem_mb()
    release = _tcmalloc_fn("MallocExtension_ReleaseFreeMemory")
    if release is not None:
        release()
    with contextlib.suppress(AttributeError):  # malloc_trim is glibc-only
        _process_symbols.malloc_trim(0)
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
