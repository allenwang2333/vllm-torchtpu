# SPDX-License-Identifier: Apache-2.0
"""Debug-only host DRAM instrumentation for the host-DRAM fixes (b/565207901).

Off unless its environment variable is "1". Flags are read from
``os.environ`` directly (not ``vllm_torchtpu.envs``) so they stay out of the
TPU compile-cache key.

  VLLM_TPU_MEMDBG_TRACE=1       log RssAnon, tcmalloc in-use / page-heap free
                                bytes and live compiled executables at startup
                                stages, and around the C1 and G1 releases.
  VLLM_TPU_MEMDBG_PERIODIC_S=N  also log RssAnon/tcmalloc every N s at serving.
  VLLM_TPU_MEMDBG_SMAPS=1       also break anonymous RSS down by mapping kind
                                (heap, thread stacks, file-backed, large anon
                                regions) from /proc/self/smaps at each stage.
  VLLM_TPU_MEMDBG_PYCENSUS=1    at post_capture (before the GC heap is frozen)
                                log the Python object types holding the most
                                memory and the CPU tensors by size.
  VLLM_TPU_MEMDBG_HEAP_DUMP=1   rank 0 of each role writes a tcmalloc heap
                                profile at each stage (gperftools tcmalloc is
                                LD_PRELOADed) to VLLM_TPU_MEMDBG_HEAP_DIR.

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
SMAPS = _flag("VLLM_TPU_MEMDBG_SMAPS")
HEAP_DUMP = _flag("VLLM_TPU_MEMDBG_HEAP_DUMP")
PYCENSUS = _flag("VLLM_TPU_MEMDBG_PYCENSUS")
NO_A1D = _flag("VLLM_TPU_MEMDBG_NO_A1D")
NO_C1 = _flag("VLLM_TPU_MEMDBG_NO_C1")
NO_G1 = _flag("VLLM_TPU_MEMDBG_NO_G1")

# tcmalloc is LD_PRELOADed in our images; resolve from the process.
_process_symbols = ctypes.CDLL(None)
_tag = "unknown"
_is_rank0 = False
_heap_profiler: object = None


def configure(role: str, dp_rank: int, rank: int) -> None:
    """Name this worker's records; called once from TPUWorker.__init__."""
    global _tag, _is_rank0
    _tag = f"{role}_dp{dp_rank}_r{rank}"
    _is_rank0 = dp_rank == 0 and rank == 0
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


def _smaps_breakdown() -> str:
    """Anonymous RSS by mapping kind, plus the largest anonymous regions."""
    kinds: dict[str, float] = {}
    big: list[tuple[float, float, str]] = []
    header = ""
    size_kb = 0
    with open("/proc/self/smaps") as f:
        for line in f:
            first = line.split(None, 1)[0]
            if "-" in first and not first.endswith(":"):
                header = line
                continue
            if first == "Size:":
                size_kb = int(line.split()[1])
            elif first == "Anonymous:":
                anon_kb = int(line.split()[1])
                if not anon_kb:
                    continue
                parts = header.split()
                path = parts[5] if len(parts) > 5 else ""
                if path == "[heap]":
                    kind = "brk_heap"
                elif path.startswith("[stack"):
                    kind = "main_stack"
                elif path.startswith("/dev/shm") or "memfd" in path:
                    kind = "shm"
                elif path:
                    kind = "file_private"
                elif size_kb in (8192, 8196, 10240, 16384):
                    kind = "thread_stack"
                elif size_kb >= 65536:
                    kind = "anon_ge64M"
                elif size_kb >= 1024:
                    kind = "anon_1M_64M"
                else:
                    kind = "anon_lt1M"
                kinds[kind] = kinds.get(kind, 0.0) + anon_kb / 1024
                if not path and size_kb >= 262144:
                    big.append((anon_kb / 1024, size_kb / 1024, parts[0]))
    threads = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                threads = int(line.split()[1])
    big.sort(reverse=True)
    parts_out = " ".join(f"{k}={v:.0f}" for k, v in sorted(kinds.items()))
    top = ",".join(f"{rss:.0f}/{size:.0f}" for rss, size, _ in big[:6])
    return f"threads={threads} {parts_out} big_anon_rss/size_MB=[{top}]"


def _heap_dump(name: str) -> None:
    global _heap_profiler
    if _heap_profiler is None:
        start = getattr(_process_symbols, "HeapProfilerStart", None)
        dump = getattr(_process_symbols, "HeapProfilerDump", None)
        if start is None or dump is None:
            logger.warning("[MEMDBG] gperftools heap profiler not available")
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
        _heap_profiler(name.encode())  # type: ignore[operator]


def _py_census(name: str, top: int = 20) -> None:
    """Python objects by type (shallow sizeof) and CPU tensors by storage."""
    import collections
    import sys

    import torch

    sizes: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    storages: dict[int, tuple[int, str]] = {}
    for obj in gc.get_objects():
        t = type(obj)
        try:
            sizes[t.__qualname__] += sys.getsizeof(obj)
        except TypeError:
            continue
        counts[t.__qualname__] += 1
        if t is torch.Tensor or isinstance(obj, torch.Tensor):
            try:
                if obj.device.type != "cpu":
                    continue
                st = obj.untyped_storage()
                storages[st.data_ptr()] = (
                    st.nbytes(),
                    f"{tuple(obj.shape)}:{obj.dtype}",
                )
            except Exception:  # noqa: BLE001 - fake/meta tensors etc.
                continue
    for qual, size in sizes.most_common(top):
        logger.info(
            "[MEMDBG] pycensus stage=%s tag=%s %s n=%d shallow=%.1f MB",
            name,
            _tag,
            qual,
            counts[qual],
            size / 2**20,
        )
    total = sum(n for n, _ in storages.values())
    logger.info(
        "[MEMDBG] pycensus stage=%s tag=%s cpu_tensor_storages=%d total=%.1f MB",
        name,
        _tag,
        len(storages),
        total / 2**20,
    )
    for nbytes, desc in sorted(storages.values(), reverse=True)[:top]:
        logger.info(
            "[MEMDBG] pycensus stage=%s tag=%s cpu_tensor %s %.1f MB",
            name,
            _tag,
            desc,
            nbytes / 2**20,
        )


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
    if SMAPS:
        logger.info("[MEMDBG] smaps stage=%s tag=%s %s", name, _tag, _smaps_breakdown())
    if PYCENSUS and _is_rank0 and name == "post_capture":
        _py_census(name)
    if HEAP_DUMP and _is_rank0 and name != "periodic":
        _heap_dump(name)


def start_periodic_sampler() -> None:
    """Log RssAnon and tcmalloc state every VLLM_TPU_MEMDBG_PERIODIC_S s."""
    period = float(os.environ.get("VLLM_TPU_MEMDBG_PERIODIC_S", "0"))
    if not TRACE or period <= 0:
        return
    import threading
    import time

    # One labelled stage (smaps + heap dump) after N periods of serving.
    serving_at = int(os.environ.get("VLLM_TPU_MEMDBG_SERVING_STAGE_TICKS", "0"))

    def _loop() -> None:
        tick = 0
        while True:
            time.sleep(period)
            tick += 1
            name = "serving" if tick == serving_at else "periodic"
            stage(name, count_executables=False)

    threading.Thread(target=_loop, name="memdbg-sampler", daemon=True).start()

