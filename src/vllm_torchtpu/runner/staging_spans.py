# SPDX-License-Identifier: Apache-2.0
"""Span bookkeeping for reusable host staging buffers that must read as zero
outside the rows a step writes."""

from __future__ import annotations

import torch

Span = tuple[int, int]


def zero_stale_spans(buf: torch.Tensor, stale: list[Span], fresh: list[Span]) -> None:
    """Zero each `stale` [start, stop) span of `buf`, skipping what the `fresh`
    spans cover: the caller overwrites those in full right after."""
    fresh = sorted(fresh)
    for start, stop in stale:
        pos = start
        for f_start, f_stop in fresh:
            if f_stop <= pos or f_start >= stop:
                continue
            if f_start > pos:
                buf[pos:f_start].zero_()
            pos = max(pos, f_stop)
            if pos >= stop:
                break
        if pos < stop:
            buf[pos:stop].zero_()
