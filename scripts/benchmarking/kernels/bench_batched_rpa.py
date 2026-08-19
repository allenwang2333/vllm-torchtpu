#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Microbenchmark: `batched_rpa` vs `batched_rpa_longctx` at the 397B shape.

Answers "is the seq-along-lane / stacked-scheduling kernel actually faster
than mainline batched RPA for Qwen3.5-397B-A17B-FP8 on v7x-8?" in minutes
instead of a full server run.

Both kernel wrappers are imported directly by module path; the
`USE_BATCHED_RPA_LONGCTX` / `USE_BATCHED_RPA_SEQ_ON_LANE` env switches are
*not* consulted, so one process benches every impl/layout combination. The
longctx KV layout is selected through its `kv_layout=` argument.

Shapes model one DP8 rank of the 397B deployment at ISL=8192 / OSL=1024, so
the numbers are directly per-rank (DP attention is not sharded over heads:
each rank owns all 32 Q / 2 KV heads).

Usage:
    # full sweep, both impls, both longctx layouts
    python3 scripts/benchmarking/kernels/bench_batched_rpa.py

    # the single question that matters, on the dominant serving shape
    python3 scripts/benchmarking/kernels/bench_batched_rpa.py \\
        --shape decode_c256_dp8 --shape mixed_8k1k

    # longctx only, comparing its two KV layouts against each other
    python3 scripts/benchmarking/kernels/bench_batched_rpa.py \\
        --impl longctx --kv-layout both

    python3 scripts/benchmarking/kernels/bench_batched_rpa.py --list-shapes
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    wrapper as mainline_wrapper
from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
    configs as longctx_configs
from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
    wrapper as longctx_wrapper

# `device_time.py` is a sibling script, not an installed module, so the
# script's own directory has to be on the path before importing it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `device_time` is the single-measurement primitive `bench` is built on; it is
# re-exported here so `python -i bench_batched_rpa.py` can reach it.
from device_time import bench, device_time  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Model shape: Qwen/Qwen3.5-397B-A17B-FP8, one DP8 rank.
# ---------------------------------------------------------------------------

HEAD_DIM = 256
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
PAGE_SIZE = 256
Q_DTYPE = jnp.bfloat16
KV_DTYPE = jnp.float8_e4m3fn

HEAD_SUBLANE = "head_sublane"
SEQ_LANE = "seq_lane"

_KV_LAYOUTS = {
    HEAD_SUBLANE: longctx_configs.KVLayout.HEAD_ALONG_SUBLANE,
    SEQ_LANE: longctx_configs.KVLayout.SEQ_ALONG_LANE,
}

MAINLINE = "mainline"
LONGCTX = "longctx"


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Shape:
    """One request batch handed to a single `ragged_paged_attention` call.

    `q_lens` and `kv_lens` are per request and must be ordered decode-first
    (see `build_distribution`). `kv_lens[i]` is the sequence length *after*
    this step's tokens are appended, which is what the kernel expects: the
    writeback lands the `q_lens[i]` new tokens at positions
    `kv_lens[i] - q_lens[i] .. kv_lens[i] - 1`.
    """

    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    why: str

    def __post_init__(self):
        assert len(self.q_lens) == len(self.kv_lens), self.name
        assert all(q <= kv for q, kv in zip(self.q_lens, self.kv_lens)), (
            f"{self.name}: q_len must fit inside kv_len")

    @property
    def num_seqs(self) -> int:
        return len(self.q_lens)

    @property
    def total_q_tokens(self) -> int:
        return sum(self.q_lens)


def _spread(num_seqs: int, lo: int, hi: int) -> tuple[int, ...]:
    """`num_seqs` KV lengths evenly covering [lo, hi].

    Real decode batches never share one KV length: requests arrive at
    different times, so they sit at different points of their OSL window.
    A single length would let the kernel's per-step bounds clipping look
    better than it is.
    """
    if num_seqs == 1:
        return (hi, )
    step = (hi - lo) / (num_seqs - 1)
    return tuple(int(round(lo + i * step)) for i in range(num_seqs))


def _decode_shape(name: str, num_seqs: int, lo: int, hi: int,
                  why: str) -> Shape:
    return Shape(name, (1, ) * num_seqs, _spread(num_seqs, lo, hi), why)


# ISL=8192 / OSL=1024 => a request's KV length walks 8192 -> 9216 over its
# decode phase, so a steady-state batch is spread across that band.
_ISL, _OSL = 8192, 1024
_KV_LO, _KV_HI = _ISL, _ISL + _OSL

SHAPES: dict[str, Shape] = {
    "decode_c256_dp8":
    _decode_shape(
        "decode_c256_dp8", 32, _KV_LO, _KV_HI,
        "Steady-state decode at global concurrency 256 over DP8 => 32 "
        "single-token queries per rank. This is where the 397B server "
        "spends most of its time, so it is the shape that decides the "
        "longctx-vs-mainline question."),
    "decode_c256_dp8_max":
    _decode_shape(
        "decode_c256_dp8_max", 48, _KV_LO, _KV_HI,
        "Top of the 32-48 band: DP ranks are not balanced, and the runner "
        "pads num_reqs up to a bucket, so the busiest rank sees ~48 decode "
        "rows. Guards against a kernel that only wins at the smaller "
        "batch_size the block-size heuristic picks for 32."),
    "decode_c64_dp8":
    _decode_shape(
        "decode_c64_dp8", 8, _KV_LO, _KV_HI,
        "Low-concurrency decode (global 64 over DP8 => 8 rows/rank). "
        "Latency-oriented operating point; there is almost no work to hide "
        "DMA behind, so this is where per-call fixed overhead (schedule "
        "kernel, pipeline prologue) dominates."),
    "decode_c64_dp8_max":
    _decode_shape(
        "decode_c64_dp8_max", 16, _KV_LO, _KV_HI,
        "Top of the 8-16 band for the low-concurrency point; same purpose "
        "as decode_c256_dp8_max."),
    "mixed_8k1k":
    Shape(
        "mixed_8k1k",
        (1, ) * 32 + (2048, ),
        _spread(32, _KV_LO, _KV_HI) + (_ISL, ),
        "One 2048-token chunked-prefill chunk (the last chunk of an 8192 "
        "ISL prompt, so kv_len=8192) admitted alongside 32 in-flight "
        "decodes at kv_len~8700. This is what continuous batching actually "
        "issues, and it is the only shape that exercises both the DECODE "
        "and the MIXED kernel launches in one call.",
    ),
    "prefill_2048":
    Shape(
        "prefill_2048",
        (2048, ),
        (2048, ),
        "A single 2048-token prefill chunk with no history: the first chunk "
        "of an 8192 ISL prompt, and the pure-MIXED-pass cost that TTFT is "
        "built out of.",
    ),
    "prefill_8192":
    Shape(
        "prefill_8192",
        (8192, ),
        (8192, ),
        "The whole 8192-token prompt in one call (chunked prefill "
        "disabled). Longest single-request Q tile in the deployment; "
        "isolates the prefill block-size heuristic from any decode work.",
    ),
}


# ---------------------------------------------------------------------------
# Input construction (mirrors tests/kernels/experimental/batched_rpa_longctx)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Everything that is not per-shape."""

    head_dim: int = HEAD_DIM
    num_q_heads: int = NUM_Q_HEADS
    num_kv_heads: int = NUM_KV_HEADS
    page_size: int = PAGE_SIZE
    q_dtype: Any = Q_DTYPE
    kv_dtype: Any = KV_DTYPE
    sm_scale: float | None = None
    k_scale: float | None = None
    v_scale: float | None = None
    shuffle_pages: bool = False
    seed: int = 0

    @property
    def scale(self) -> float:
        return (self.head_dim**-0.5
                if self.sm_scale is None else self.sm_scale)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def build_distribution(q_lens: Sequence[int]) -> np.ndarray:
    """Build the `distribution` array and assert the batch layout is legal.

    `distribution` is *not* a count triple; it is the exclusive end boundary
    of each request class in a batch that is sorted by class:

        decode requests: [0, distribution[0])
        prefill requests: [distribution[0], distribution[1])
        mixed requests:  [distribution[1], distribution[2])

    (see `configs.RpaCase.get_range`). `distribution[2]` is therefore the
    total request count. Both wrappers only launch the DECODE and MIXED
    passes, so the prefill bucket must stay empty -- which is exactly what
    `TPUModelRunner` emits: `[num_decode, num_decode, num_reqs]`.
    """
    num_seqs = len(q_lens)
    num_decodes = 0
    for q_len in q_lens:
        if q_len != 1:
            break
        num_decodes += 1

    assert all(q_len == 1 for q_len in q_lens[:num_decodes]), (
        "requests in the decode range must have q_len == 1")
    assert all(q_len > 1 for q_len in q_lens[num_decodes:]), (
        "batch must be sorted decode-first; a q_len==1 request after the "
        "first multi-token request would be silently routed to the MIXED "
        "pass")

    distribution = np.array([num_decodes, num_decodes, num_seqs], np.int32)
    assert distribution[0] == distribution[1], (
        "the prefill bucket must be empty: neither wrapper launches "
        "RpaCase.PREFILL")
    assert distribution[2] == num_seqs
    return distribution


def _random_logical_cache(rng, num_pages: int, cfg: RunConfig) -> np.ndarray:
    """Layout-independent KV cache: (num_pages, page_size, 2*H_kv, head_dim).

    Filled in page chunks so the float32 staging buffer stays small; the
    cache itself is hundreds of MB at the 397B decode shapes.
    """
    shape = (num_pages, cfg.page_size, 2 * cfg.num_kv_heads, cfg.head_dim)
    out = np.empty(shape, dtype=cfg.kv_dtype)
    per_page = cfg.page_size * 2 * cfg.num_kv_heads * cfg.head_dim
    chunk = max(1, (1 << 22) // per_page)
    for start in range(0, num_pages, chunk):
        stop = min(start + chunk, num_pages)
        block = rng.standard_normal(
            (stop - start, ) + shape[1:], dtype=np.float32) * 0.3
        out[start:stop] = block.astype(cfg.kv_dtype)
    return out


def _pack_head_along_sublane(logical: np.ndarray,
                             cache_shape: tuple[int, ...]) -> np.ndarray:
    """(P, T, 2H, D) -> (P, T, 2H_aligned // packing, packing, D_aligned).

    Matches `wrapper.prepare_inputs`: KV head `h` lands at
    `[:, :, h // packing, h % packing, :]`.
    """
    num_pages, page_size, num_kv_x2, head_dim = logical.shape
    _, cache_page_size, groups, packing, aligned_head_dim = cache_shape
    assert cache_page_size == page_size
    assert groups * packing >= num_kv_x2
    assert aligned_head_dim >= head_dim

    padded = np.zeros(
        (num_pages, page_size, groups * packing, aligned_head_dim),
        dtype=logical.dtype)
    padded[:, :, :num_kv_x2, :head_dim] = logical
    return np.ascontiguousarray(
        padded.reshape(num_pages, page_size, groups, packing,
                       aligned_head_dim))


def _pack_seq_along_lane(logical: np.ndarray,
                         cache_shape: tuple[int, ...]) -> np.ndarray:
    """(P, T, 2H, D) -> (P, 2H, D_aligned // packing, packing, page_size).

    Matches the `KVLayout.SEQ_ALONG_LANE` branch of
    `wrapper.prepare_inputs`: head_dim moves onto sublanes and the token
    index becomes the lane index.
    """
    num_pages, page_size, num_kv_x2, head_dim = logical.shape
    _, heads, dim_groups, packing, cache_page_size = cache_shape
    assert heads == num_kv_x2
    assert cache_page_size == page_size
    assert dim_groups * packing >= head_dim

    padded = np.zeros((num_pages, page_size, num_kv_x2, dim_groups * packing),
                      dtype=logical.dtype)
    padded[:, :, :, :head_dim] = logical
    return np.ascontiguousarray(
        padded.reshape(num_pages, page_size, num_kv_x2, dim_groups,
                       packing).transpose(0, 2, 3, 4, 1))


def _kv_cache_shapes(cfg: RunConfig,
                     total_pages: int) -> dict[str, tuple[int, ...]]:
    """Per-layout cache shapes, straight from the wrappers."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name, layout in _KV_LAYOUTS.items():
        shapes[name] = tuple(
            longctx_wrapper.get_kv_cache_shape(
                total_num_pages=total_pages,
                page_size=cfg.page_size,
                actual_num_kv_heads=cfg.num_kv_heads,
                actual_head_dim=cfg.head_dim,
                kv_dtype=cfg.kv_dtype,
                kv_layout=layout,
            ))
    mainline_shape = tuple(
        mainline_wrapper.get_kv_cache_shape(
            total_num_pages=total_pages,
            page_size=cfg.page_size,
            actual_num_kv_heads=cfg.num_kv_heads,
            actual_head_dim=cfg.head_dim,
            kv_dtype=cfg.kv_dtype,
        ))
    assert mainline_shape == shapes[HEAD_SUBLANE], (
        f"mainline cache shape {mainline_shape} no longer matches longctx "
        f"HEAD_ALONG_SUBLANE {shapes[HEAD_SUBLANE]}; the two impls can no "
        "longer share packed cache contents")
    return shapes


@dataclasses.dataclass
class Case:
    """Host-side inputs for one shape, shared by every impl/layout."""

    shape: Shape
    cfg: RunConfig
    q_host: np.ndarray
    k_host: np.ndarray
    v_host: np.ndarray
    kv_lens: np.ndarray
    page_indices: np.ndarray
    cu_q_lens: np.ndarray
    distribution: np.ndarray
    caches: dict[str, np.ndarray]
    pages_per_seq: int
    total_pages: int

    @property
    def cache_mib(self) -> float:
        cache = next(iter(self.caches.values()))
        return cache.nbytes / (1 << 20)


def build_case(shape: Shape, cfg: RunConfig, layouts: Sequence[str]) -> Case:
    """Build every host tensor for `shape`, once, for all layouts."""
    rng = np.random.default_rng(cfg.seed)

    total_q = shape.total_q_tokens
    q_host = (rng.standard_normal(
        (total_q, cfg.num_q_heads, cfg.head_dim), dtype=np.float32) *
              0.5).astype(cfg.q_dtype)
    k_host = (rng.standard_normal(
        (total_q, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32) *
              0.5).astype(cfg.kv_dtype)
    v_host = (rng.standard_normal(
        (total_q, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32) *
              0.5).astype(cfg.kv_dtype)

    pages_per_seq = _cdiv(max(shape.kv_lens), cfg.page_size)
    total_pages = pages_per_seq * shape.num_seqs

    page_indices = np.arange(total_pages, dtype=np.int32)
    if cfg.shuffle_pages:
        # vLLM hands out pages from a free pool, so a long-lived server's page
        # tables are fragmented; contiguous indices are the best-case DMA
        # pattern.
        page_indices = rng.permutation(page_indices).astype(np.int32)

    cu_q_lens = np.zeros(shape.num_seqs + 1, dtype=np.int32)
    np.cumsum(shape.q_lens, out=cu_q_lens[1:])

    cache_shapes = _kv_cache_shapes(cfg, total_pages)
    logical = _random_logical_cache(rng, total_pages, cfg)
    caches = {}
    for layout in layouts:
        if layout == SEQ_LANE:
            caches[layout] = _pack_seq_along_lane(logical,
                                                  cache_shapes[layout])
        else:
            caches[layout] = _pack_head_along_sublane(logical,
                                                      cache_shapes[layout])
    del logical

    return Case(
        shape=shape,
        cfg=cfg,
        q_host=q_host,
        k_host=k_host,
        v_host=v_host,
        kv_lens=np.asarray(shape.kv_lens, dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=build_distribution(shape.q_lens),
        caches=caches,
        pages_per_seq=pages_per_seq,
        total_pages=total_pages,
    )


# ---------------------------------------------------------------------------
# Kernel invocation
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Variant:
    impl: str
    layout: str

    @property
    def label(self) -> str:
        return f"{self.impl}/{self.layout}"


def select_variants(impl: str, kv_layout: str) -> list[Variant]:
    layouts = ([HEAD_SUBLANE, SEQ_LANE]
               if kv_layout == "both" else [kv_layout])
    variants: list[Variant] = []
    if impl in (MAINLINE, "both"):
        # Mainline has exactly one KV layout; --kv-layout only gates longctx.
        variants.append(Variant(MAINLINE, HEAD_SUBLANE))
    if impl in (LONGCTX, "both"):
        variants.extend(Variant(LONGCTX, layout) for layout in layouts)
    return variants


def make_call(variant: Variant, case: Case):
    """Return a zero-arg callable running one kernel invocation.

    All four of `queries`/`keys`/`values`/`kv_cache` are donated by the
    wrapper's `jax.jit`, so every call needs fresh buffers: Q/K/V are
    re-uploaded from the host and the KV cache is threaded through from the
    previous call's output, exactly as the serving loop does.
    """
    cfg = case.cfg
    kv_lens = jnp.asarray(case.kv_lens)
    page_indices = jnp.asarray(case.page_indices)
    cu_q_lens = jnp.asarray(case.cu_q_lens)
    distribution = jnp.asarray(case.distribution)
    state = {"kv": jnp.asarray(case.caches[variant.layout])}

    kwargs: dict[str, Any] = {
        "sm_scale": cfg.scale,
        "k_scale": cfg.k_scale,
        "v_scale": cfg.v_scale,
    }
    if variant.impl == MAINLINE:
        rpa = mainline_wrapper.ragged_paged_attention
    else:
        rpa = longctx_wrapper.ragged_paged_attention
        kwargs["kv_layout"] = _KV_LAYOUTS[variant.layout]

    def call():
        out, new_kv = rpa(
            jnp.asarray(case.q_host),
            jnp.asarray(case.k_host),
            jnp.asarray(case.v_host),
            state["kv"],
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            **kwargs,
        )
        state["kv"] = new_kv
        return out, new_kv

    return call


def run_once(variant: Variant, case: Case) -> np.ndarray:
    """One call from a pristine cache; returns the output as float32."""
    call = make_call(variant, case)
    out, _ = jax.block_until_ready(call())
    return np.asarray(out.astype(jnp.float32))


# ---------------------------------------------------------------------------
# Timing (device_time.py, written by Task A)
# ---------------------------------------------------------------------------

# `bench`'s return type is owned by device_time.py; accept a bare number, a
# dataclass, or a mapping keyed by any of these.
_DEVICE_US_KEYS = ("device_us", "kernel_us", "main_pallas_us", "module_us",
                   "median_us", "p50_us", "us")


def _device_us(result: Any) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        result = dataclasses.asdict(result)
    if isinstance(result, dict):
        for key in _DEVICE_US_KEYS:
            value = result.get(key)
            if value:
                return float(value)
    raise TypeError(
        f"cannot read a device time out of {type(result).__name__}: "
        f"{result!r}; expected a number or one of {_DEVICE_US_KEYS}")


def _bench(call, warmup: int, iters: int) -> float:
    """Call `device_time.bench` with whichever knobs it exposes."""
    try:
        params = inspect.signature(bench).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {}
    if "warmup" in params:
        kwargs["warmup"] = warmup
    if "iters" in params:
        kwargs["iters"] = iters
    elif "n" in params:
        kwargs["n"] = iters
    return _device_us(bench(call, **kwargs))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Result:
    shape: str
    impl: str
    layout: str
    device_us: float | None = None
    speedup: float | None = None
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    error: str | None = None


def _errors(out: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Max abs error, and that error relative to the reference's magnitude.

    Per-element relative error is meaningless here: attention outputs pass
    through zero, so a handful of near-zero entries would dominate. Scaling
    by max|ref| answers the question that matters -- is this a rounding
    difference or a different answer.
    """
    max_abs = float(np.max(np.abs(out - ref)))
    denom = float(np.max(np.abs(ref)))
    return max_abs, (max_abs / denom if denom > 0 else float("nan"))


def run_shape(shape: Shape, cfg: RunConfig, variants: Sequence[Variant],
              warmup: int, iters: int, check_parity: bool) -> list[Result]:
    layouts = sorted({v.layout for v in variants})
    case = build_case(shape, cfg, layouts)
    print(
        f"[bench] {shape.name}: {shape.num_seqs} reqs, "
        f"{shape.total_q_tokens} q tokens, kv_lens "
        f"{min(shape.kv_lens)}..{max(shape.kv_lens)}, "
        f"{case.pages_per_seq} pages/req, {case.total_pages} pages, "
        f"{case.cache_mib:.0f} MiB cache/layout, "
        f"distribution={case.distribution.tolist()}",
        flush=True)

    results: list[Result] = []
    reference: np.ndarray | None = None
    baseline_us: float | None = None

    for variant in variants:
        result = Result(shape.name, variant.impl, variant.layout)
        results.append(result)
        try:
            if check_parity:
                out = run_once(variant, case)
                if reference is None:
                    reference = out
                    result.max_abs_err = 0.0
                    result.max_rel_err = 0.0
                else:
                    result.max_abs_err, result.max_rel_err = _errors(
                        out, reference)
                del out
            result.device_us = _bench(make_call(variant, case), warmup, iters)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not
            # abort the sweep; the table shows the failure in place.
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"[bench]   {variant.label}: FAILED {result.error}",
                  flush=True)
            continue

        if variant.impl == MAINLINE and baseline_us is None:
            baseline_us = result.device_us
        parity = ("" if result.max_abs_err is None else
                  f" max_abs_err={result.max_abs_err:.3e}")
        print(f"[bench]   {variant.label}: {result.device_us:.1f} us{parity}",
              flush=True)

    if baseline_us:
        for result in results:
            if result.device_us:
                result.speedup = baseline_us / result.device_us
    return results


def print_table(results: Sequence[Result], reference_label: str) -> None:
    header = (f"{'shape':<22s} {'impl':<9s} {'layout':<13s} "
              f"{'device_us':>10s} {'speedup':>8s} {'max_abs_err':>12s} "
              f"{'max_rel_err':>12s}")
    print()
    print(f"errors are vs {reference_label} (the first variant that ran for "
          "that shape); speedup is vs mainline device_us")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"{r.shape:<22s} {r.impl:<9s} {r.layout:<13s} "
                  f"  ERROR: {r.error}")
            continue
        speedup = f"{r.speedup:.2f}x" if r.speedup else "-"
        abs_err = ("-" if r.max_abs_err is None else f"{r.max_abs_err:.3e}")
        rel_err = ("-" if r.max_rel_err is None else f"{r.max_rel_err:.3e}")
        print(f"{r.shape:<22s} {r.impl:<9s} {r.layout:<13s} "
              f"{r.device_us:>10.1f} {speedup:>8s} {abs_err:>12s} "
              f"{rel_err:>12s}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl",
                        choices=(MAINLINE, LONGCTX, "both"),
                        default="both")
    parser.add_argument(
        "--kv-layout",
        choices=(HEAD_SUBLANE, SEQ_LANE, "both"),
        default="both",
        help="longctx KV cache layout; mainline always uses its single "
        "head-along-sublane layout and is unaffected")
    parser.add_argument("--shape",
                        action="append",
                        choices=sorted(SHAPES),
                        metavar="NAME",
                        help="restrict to one shape; repeatable")
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-parity",
                        action="store_true",
                        help="skip the numeric cross-check (saves one call "
                        "and one output copy per variant)")
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--num-q-heads", type=int, default=NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--kv-dtype",
                        choices=("float8_e4m3fn", "bfloat16"),
                        default="float8_e4m3fn")
    parser.add_argument("--sm-scale",
                        type=float,
                        default=None,
                        help="default: head_dim ** -0.5")
    parser.add_argument("--kv-scale",
                        type=float,
                        default=1.0,
                        help="k_scale/v_scale for an fp8 cache, matching the "
                        "dequant multiply the served model emits; ignored "
                        "for a bf16 cache")
    parser.add_argument("--shuffle-pages",
                        action="store_true",
                        help="fragment the page table like a long-lived "
                        "server instead of allocating contiguously")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_shapes:
        for name, shape in SHAPES.items():
            print(f"{name}\n  {shape.num_seqs} reqs, "
                  f"q_lens={_summarize(shape.q_lens)}, "
                  f"kv_lens={_summarize(shape.kv_lens)}\n  {shape.why}\n")
        return 0

    kv_dtype = getattr(jnp, args.kv_dtype)
    is_fp8 = kv_dtype != Q_DTYPE
    cfg = RunConfig(
        head_dim=args.head_dim,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        page_size=args.page_size,
        q_dtype=Q_DTYPE,
        kv_dtype=kv_dtype,
        sm_scale=args.sm_scale,
        k_scale=args.kv_scale if is_fp8 else None,
        v_scale=args.kv_scale if is_fp8 else None,
        shuffle_pages=args.shuffle_pages,
        seed=args.seed,
    )

    variants = select_variants(args.impl, args.kv_layout)
    shape_names = args.shape or list(SHAPES)

    print(f"[bench] jax {jax.__version__} on {jax.devices()[0]}")
    print(f"[bench] head_dim={cfg.head_dim} q_heads={cfg.num_q_heads} "
          f"kv_heads={cfg.num_kv_heads} page_size={cfg.page_size} "
          f"q={jnp.dtype(cfg.q_dtype).name} kv={jnp.dtype(cfg.kv_dtype).name} "
          f"sm_scale={cfg.scale:.6f} k_scale={cfg.k_scale}")
    print(f"[bench] variants={[v.label for v in variants]} "
          f"warmup={args.warmup} iters={args.iters}")

    results: list[Result] = []
    for name in shape_names:
        results.extend(
            run_shape(SHAPES[name], cfg, variants, args.warmup, args.iters,
                      not args.skip_parity))

    print_table(results, variants[0].label if variants else "n/a")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "config": {
                        "head_dim": cfg.head_dim,
                        "num_q_heads": cfg.num_q_heads,
                        "num_kv_heads": cfg.num_kv_heads,
                        "page_size": cfg.page_size,
                        "q_dtype": jnp.dtype(cfg.q_dtype).name,
                        "kv_dtype": jnp.dtype(cfg.kv_dtype).name,
                        "sm_scale": cfg.scale,
                        "k_scale": cfg.k_scale,
                        "shuffle_pages": cfg.shuffle_pages,
                        "seed": cfg.seed,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    },
                    "results": [dataclasses.asdict(r) for r in results],
                },
                handle,
                indent=2)
        print(f"\n[bench] wrote {args.json}")

    return 0


def _summarize(values: Sequence[int]) -> str:
    lo, hi = min(values), max(values)
    span = str(lo) if lo == hi else f"{lo}..{hi}"
    return f"[{span}] x{len(values)}"


if __name__ == "__main__":
    sys.exit(main())
