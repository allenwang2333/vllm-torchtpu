# SPDX-License-Identifier: Apache-2.0
"""Huge-page-backed host buffers for packed Engram uploads.

Adapted from vllm-project/vllm-torchtpu commit 2ed6462e8, common/host_pages.py.
Keep the aligned, populated mapping; use Python's madvise, derive the PMD size,
and fall back when advice fails or the mapping is not fully huge-page-backed.
NUMA inspection and per-upload logging from the original are omitted.
"""

import ctypes
import mmap
from pathlib import Path

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_THP_SIZE = Path("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")
_SMAPS = Path("/proc/self/smaps")


def _hugepage_backed(address: int, size: int) -> bool:
    """Require full THP backing in every VMA overlapping the requested range."""
    covered = 0
    start = end = 0
    with _SMAPS.open() as stream:
        for line in stream:
            field = line.split()
            if "-" in field[0]:
                start, end = (int(value, 16) for value in field[0].split("-"))
            elif field[0] == "AnonHugePages:":
                overlap = min(end, address + size) - max(start, address)
                if overlap > 0:
                    # A merged VMA may extend beyond our buffer. A partial THP
                    # count cannot tell us which of its pages are huge.
                    if int(field[1]) * 1024 != end - start:
                        return False
                    covered += overlap
    return covered == size


def alloc_packed_staging(nbytes: int, *, huge: bool = True) -> torch.Tensor:
    """Allocate a zeroed CPU uint8 buffer, preferring transparent huge pages.

    The returned tensor owns its mapping, including when only a view survives.
    Unavailable THP support or incomplete backing uses the ordinary torch
    allocator with a warning. ``huge=False`` is the diagnostic control path.
    """
    if nbytes < 0:
        raise ValueError("Engram staging size must be nonnegative")
    if not huge or nbytes == 0:
        return torch.zeros(nbytes, dtype=torch.uint8, device="cpu")

    mapping = None
    try:
        page_size = int(_THP_SIZE.read_text().strip())
        if page_size < mmap.PAGESIZE or page_size % mmap.PAGESIZE:
            raise ValueError(f"Invalid transparent huge-page size: {page_size}")
        size = -(-nbytes // page_size) * page_size
        mapping = mmap.mmap(
            -1, size + page_size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        offset = -base % page_size
        address = base + offset
        mapping.madvise(mmap.MADV_HUGEPAGE, offset, size)
        ctypes.memset(address, 0, size)
        if _hugepage_backed(address, size):
            tensor = torch.frombuffer(
                mapping, dtype=torch.uint8, count=nbytes, offset=offset
            )
            mapping = None  # torch.frombuffer retains the mapping's owner.
            return tensor
        reason = f"the aligned {size}-byte region is not fully backed by huge pages"
    except OSError as error:
        # Ordinary pageable memory has identical contents and copy semantics.
        reason = str(error)
    finally:
        if mapping is not None:
            mapping.close()

    logger.warning_once(
        "[Engram] Huge-page staging unavailable for %d bytes: %s; "
        "using torch.zeros, which may increase H2D latency. "
        "Check the host transparent_hugepage policy. "
        "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING=0 disables this optimization.",
        nbytes,
        reason,
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
