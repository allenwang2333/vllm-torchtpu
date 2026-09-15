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
"""A/B the two batched-RPA KV-cache layouts at Qwen3-Coder-30B's serving shape.

## What is being measured

With an fp8 KV cache, ``PallasAttentionBackend.get_kv_cache_shape``
(``src/vllm_torchtpu/layers/adapter/attention.py``) rounds the packed KV-head
axis up to the fp8 packing factor::

    num_kv_heads_x2 = num_kv_heads * 2                      # 1 * 2 = 2
    num_kv_heads_x2 = cdiv(num_kv_heads_x2, 4) * 4          # -> 4

At TP=8 this model has one KV head per rank, so half of that axis is padding
and an fp8 page costs 512 B/token -- exactly what bf16 costs. The fp8 KV cache
buys no bandwidth at all.

``KVLayout.SEQ_ALONG_LANE`` (selected in production by
``VLLM_KV_CACHE_LAYOUT=HND``) puts the sequence on the minor axis and head_dim
on the sublanes, so the 32-bit word is filled by head_dim rather than by KV
heads and the page is a genuine 256 B/token. This script measures whether that
halving turns into time on the batched-RPA kernel, at the production shape,
for both a decode-heavy and a prefill-heavy step.

It runs the kernel directly on one TPU chip: no vLLM server, no model weights,
synthetic KV cache and query tensors. Launch it through the
``run-command-in-cdk`` recipe.

## Page size: the two arms are NOT page-size matched, and that is production

``RpaConfigs.validate_inputs`` in
``src/vllm_torchtpu/kernels/experimental/batched_rpa/configs.py`` rejects any
page size other than 128 under SEQ_ALONG_LANE (non-PCP), and
``PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes`` mirrors
that: ``[128]`` under ``HND``, ``[256]`` otherwise. So the shipping
configuration of the two arms is

    head_along_sublane -> page_size 256   (the production default today)
    seq_along_lane     -> page_size 128   (the only page size the kernel takes)

and this script defaults each arm to its own production page size. Pass
``--page-size`` to force a common one; forcing 256 on ``seq_along_lane`` will
raise out of ``validate_inputs`` rather than silently measure something else.
``--check`` always matches the page size across arms (``--check-page-size``,
default 128) so that layout is the only variable in the numerical comparison.

## How the arm is selected -- read this before trusting a delta

The layout is passed to the kernel as an **explicit**
``configs.KVLayout`` argument, the same way
``tests/kernels/experimental/batched_rpa_longctx/test_seq_along_lane_parity.py``
does. Nothing on this path reads ``VLLM_KV_CACHE_LAYOUT``: that env var is
consumed by ``vllm.v1.attention.backends.utils.get_kv_cache_layout``, which is
``functools.lru_cache``-d and therefore latches the first value it ever sees in
a process. This script still exports ``VLLM_KV_CACHE_LAYOUT`` before importing
anything, and cross-checks it against the selected enum, so that a future
reader who wires in a vLLM code path does not get two identical arms.

The load-bearing guard is not the env var, though -- it is the per-arm banner
plus the assertion on bytes/token. Every run
prints the KV cache shape the wrapper actually resolved and asserts
512 B/token for ``head_along_sublane`` and 256 B/token for ``seq_along_lane``.
If both arms print the same number, the A/B is void.

## Invocation

    uv_venv/bin/python scripts/microbench_batched_rpa_kv_layout.py \
        --layout head_along_sublane
    uv_venv/bin/python scripts/microbench_batched_rpa_kv_layout.py \
        --layout seq_along_lane
    uv_venv/bin/python scripts/microbench_batched_rpa_kv_layout.py --check

Each timed arm emits one ``RESULT_JSON: {...}`` line per phase.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Model geometry.
#
# VERIFIED on 2026-09-15 against
# https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/raw/main/config.json
# (HF revision dcaee4d4dfc5ee71ad501f01f530e5652438fde0):
#   num_attention_heads=32, num_key_value_heads=4, head_dim=128,
#   num_hidden_layers=48.
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
HF_REVISION = "dcaee4d4dfc5ee71ad501f01f530e5652438fde0"

NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = 128
NUM_HIDDEN_LAYERS = 48  # reported only; this script times one layer.

# Tensor parallelism of the nightly bench config
# (scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh).
TENSOR_PARALLEL_SIZE = 8

# Per-rank head counts. 4 KV heads do not divide 8 ranks; the caller's
# stipulation for this measurement is the floor-with-a-minimum-of-one form
# below, i.e. one KV head per rank. NOTE: this is an assumption about what the
# runner does with an indivisible KV-head count (replicate vs. pad was not
# established) -- but it is also the case that makes the padding waste worst,
# which is precisely the case this benchmark exists to measure.
NUM_Q_HEADS_PER_RANK = NUM_ATTENTION_HEADS // TENSOR_PARALLEL_SIZE  # 4
NUM_KV_HEADS_PER_RANK = max(1, NUM_KEY_VALUE_HEADS // TENSOR_PARALLEL_SIZE)  # 1

# Workload: ISL 8192 / OSL 1024 / concurrency 64.
INPUT_SEQ_LEN = 8192
OUTPUT_SEQ_LEN = 1024
CONCURRENCY = 64
MAX_MODEL_LEN = 9216
MAX_NUM_SEQS = 64

# Decode-step context length. The cached length walks 8192 -> 9216 over the
# 1024 output tokens; 8704 is the midpoint and is a whole number of pages at
# both 128 and 256.
DECODE_KV_LEN = 8704

# Prefill-step chunk. One 8192-token chunked prefill.
PREFILL_Q_TOKENS = 8192

# Page sizes each layout actually ships with -- see the module docstring.
PAGE_SIZE_HEAD_ALONG_SUBLANE = 256
PAGE_SIZE_SEQ_ALONG_LANE = 128

# What the padding costs. Asserted against the wrapper's resolved shape.
EXPECTED_BYTES_PER_TOKEN = {
    "head_along_sublane": 512,
    "seq_along_lane": 256,
}

_VLLM_LAYOUT_BY_ARM = {
    "head_along_sublane": "NHD",
    "seq_along_lane": "HND",
}

_LAYOUT_CHOICES = tuple(_VLLM_LAYOUT_BY_ARM)


# ---------------------------------------------------------------------------
# Imports that must happen after VLLM_KV_CACHE_LAYOUT is exported.
# ---------------------------------------------------------------------------
def _export_vllm_layout_env(arm: str) -> None:
    """Export ``VLLM_KV_CACHE_LAYOUT`` for ``arm`` before any import.

    Nothing on this script's kernel path reads it (see the module docstring),
    but ``get_kv_cache_layout`` is ``lru_cache``-d, so setting it late is the
    classic way to end up benchmarking one layout twice. Set it first, always.
    """
    os.environ["VLLM_KV_CACHE_LAYOUT"] = _VLLM_LAYOUT_BY_ARM[arm]


def _import_kernel():
    """Import the batched-RPA wrapper and configs. Requires a TPU client."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import (  # noqa: E402
        configs, wrapper)
    return wrapper, configs


def _cross_check_env(arm: str, kv_layout) -> str | None:
    """Best effort: confirm vLLM would resolve the same layout as ``arm``.

    Off by default (``--cross-check-env``): reaching the resolver means
    importing vLLM and ``vllm_torchtpu.layers.adapter.attention``, which pulls
    in torch/torch_tpu and can take the TPU client somewhere this benchmark
    does not want it. The kernel path here never reads the env var, so this is
    a belt-and-braces check for a future reader, not a correctness gate.

    Returns a human-readable note, or a string saying why it could not run.
    """
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        from vllm_torchtpu.layers.adapter.attention import \
            KV_LAYOUT_BY_VLLM_LAYOUT
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"vllm layout resolver unavailable ({type(exc).__name__})"
    resolved = get_kv_cache_layout()
    mapped = KV_LAYOUT_BY_VLLM_LAYOUT[resolved]
    if mapped is not kv_layout:
        raise SystemExit(
            f"FATAL: VLLM_KV_CACHE_LAYOUT resolves to {resolved!r} -> {mapped}"
            f" but this arm is {kv_layout}. get_kv_cache_layout() is"
            " lru_cache-d; something read it before the env was set. Refusing"
            " to run: the two arms would not differ.")
    return f"vllm get_kv_cache_layout()={resolved!r} -> {mapped.name}"


# ---------------------------------------------------------------------------
# Shape / bytes accounting.
# ---------------------------------------------------------------------------
def _bytes_per_token(kv_cache_shape: tuple[int, ...], page_size: int,
                     itemsize: int) -> float:
    """Bytes of KV cache one token occupies, padding included.

    Derived from the wrapper's own resolved shape rather than recomputed, so
    the number printed is the number allocated.
    """
    elements_per_page = math.prod(kv_cache_shape[1:])
    return elements_per_page * itemsize / page_size


def _resolve_page_size(arm: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    if arm == "seq_along_lane":
        return PAGE_SIZE_SEQ_ALONG_LANE
    return PAGE_SIZE_HEAD_ALONG_SUBLANE


# ---------------------------------------------------------------------------
# Workload construction.
# ---------------------------------------------------------------------------
class Phase:
    DECODE = "decode"
    PREFILL = "prefill"
    EMPTY = "empty"


def _build_phase_inputs(jnp, jax, *, phase: str, page_size: int,
                        decode_kv_len: int, prefill_q_tokens: int,
                        prefill_context: int, max_num_seqs: int,
                        pages_per_seq: int, kv_dtype, q_dtype, cache_shape,
                        cache_init: str, seed: int):
    """Build one step's kernel arguments.

    Mirrors what the runner hands the kernel: request metadata is padded to
    ``max_num_seqs`` even when a single sequence is live, ``page_indices`` is
    the flattened ``[num_seqs * pages_per_seq]`` block table, and ``kv_lens``
    counts the new tokens as well as the cached ones.
    """
    import numpy as np

    key = jax.random.key(seed)
    k_q, k_kv, k_cache = jax.random.split(key, 3)

    if phase == Phase.DECODE:
        num_seqs = min(CONCURRENCY, max_num_seqs)
        q_lens = [1] * num_seqs
        kv_lens_list = [decode_kv_len] * num_seqs
        # distribution: [decode_end, prefill_end, total]
        distribution = [num_seqs, num_seqs, num_seqs]
    elif phase == Phase.PREFILL:
        num_seqs = 1
        q_lens = [prefill_q_tokens]
        kv_lens_list = [prefill_context + prefill_q_tokens]
        # All mixed: the kernel's MIXED pass handles chunked prefill.
        distribution = [0, 0, 1]
    elif phase == Phase.EMPTY:
        # No sequence falls in any range, so both the DECODE and the MIXED
        # kernel plan zero steps and their pipeline bodies never execute. What
        # remains is exactly the per-launch fixed cost of the pair -- which is
        # the quantity the `safe_steps > 0` memset guard targets, and which is
        # otherwise buried under a 1.4 ms floor of padded KV streaming.
        #
        # One sequence is still described so the query tensor is non-empty;
        # `distribution = [0, 0, 0]` is what excludes it from every range.
        num_seqs = 1
        q_lens = [1]
        kv_lens_list = [page_size]
        distribution = [0, 0, 0]
    else:
        raise ValueError(f"unknown phase {phase!r}")

    total_q_tokens = sum(q_lens)
    for kv_len in kv_lens_list:
        if kv_len > pages_per_seq * page_size:
            raise ValueError(
                f"{phase}: kv_len={kv_len} exceeds pages_per_seq*page_size="
                f"{pages_per_seq * page_size}; raise --max-model-len")

    kv_lens = np.zeros((max_num_seqs, ), dtype=np.int32)
    kv_lens[:num_seqs] = kv_lens_list

    cu_q_lens = np.full((max_num_seqs + 1, ), total_q_tokens, dtype=np.int32)
    cu_q_lens[0] = 0
    cu_q_lens[1:num_seqs + 1] = np.cumsum(q_lens, dtype=np.int32)

    page_indices = np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)

    query = (jax.random.normal(
        k_q, (total_q_tokens, NUM_Q_HEADS_PER_RANK, HEAD_DIM),
        dtype=jnp.float32) * 0.5).astype(q_dtype)
    new_kv = (jax.random.normal(
        k_kv, (2, total_q_tokens, NUM_KV_HEADS_PER_RANK, HEAD_DIM),
        dtype=jnp.float32) * 0.5).astype(kv_dtype)
    new_k, new_v = new_kv[0], new_kv[1]

    kv_cache = _make_cache(jnp, jax, k_cache, cache_shape, kv_dtype,
                           cache_init)

    args = dict(
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(np.asarray(distribution, dtype=np.int32)),
    )
    meta = dict(
        num_live_seqs=num_seqs,
        q_lens=q_lens,
        kv_lens=kv_lens_list,
        total_q_tokens=total_q_tokens,
    )
    return query, new_k, new_v, kv_cache, args, meta


def _make_cache(jnp, jax, key, cache_shape, kv_dtype, cache_init: str):
    """Allocate the KV cache.

    ``random`` builds one random page in fp32 and tiles it, so the fp32
    intermediate stays page-sized instead of cache-sized. Random rather than
    zeros because an all-zero cache is the one input a compiler or a DMA engine
    could plausibly shortcut, and this benchmark's whole claim is about bytes
    moved.
    """
    if cache_init == "zeros":
        return jnp.zeros(cache_shape, dtype=kv_dtype)
    if cache_init != "random":
        raise ValueError(f"unknown --cache-init {cache_init!r}")
    page_shape = (1, ) + tuple(cache_shape[1:])
    page = (jax.random.normal(key, page_shape, dtype=jnp.float32) *
            0.5).astype(kv_dtype)
    return jnp.tile(page, (cache_shape[0], ) + (1, ) * (len(cache_shape) - 1))


# ---------------------------------------------------------------------------
# The measured call.
# ---------------------------------------------------------------------------
def _make_step(jax, wrapper, *, kv_layout, sm_scale, q_scale, k_scale,
               v_scale):
    """A jitted one-layer attention step.

    ``wrapper.ragged_paged_attention`` donates queries/keys/values as well as
    the cache. Wrapping it in an outer ``jit`` that donates only the cache
    keeps q/k/v alive across iterations, so the timing loop does not pay a
    host->device transfer per sample. The cache is donated and fed back, the
    way the runner does it.
    """

    def _call(queries, keys, values, kv_cache, kv_lens, page_indices,
              cu_q_lens, distribution):
        return wrapper.ragged_paged_attention(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=kv_layout,
        )

    return jax.jit(_call, donate_argnums=(3, ))


def _block_sizes_for(wrapper, configs, *, phase, page_size, pages_per_seq,
                     max_num_seqs, kv_layout, q_dtype, kv_dtype):
    """(schedule capacity, bq, bkv) the kernel will plan this phase with."""
    mode = (configs.RpaCase.DECODE
            if phase == Phase.DECODE else configs.RpaCase.MIXED)
    return wrapper.schedule_capacity(
        mode=mode,
        num_q_heads=NUM_Q_HEADS_PER_RANK,
        num_kv_heads=NUM_KV_HEADS_PER_RANK,
        head_dim=HEAD_DIM,
        num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
        kv_layout=kv_layout,
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _gbps(num_bytes: float | None, seconds: float) -> float | None:
    if num_bytes is None or seconds <= 0:
        return None
    return num_bytes / seconds / 1e9


def _fmt(value, spec="%.3f"):
    return "n/a" if value is None else spec % value


def _emit_result_json(payload: dict) -> None:
    print("RESULT_JSON: " + json.dumps(payload, sort_keys=True))


# ---------------------------------------------------------------------------
# Benchmark.
# ---------------------------------------------------------------------------
def _run_phase(jax, jnp, wrapper, configs, *, arm, kv_layout, phase,
               args) -> dict:
    page_size = _resolve_page_size(arm, args.page_size)
    pages_per_seq = math.ceil(args.max_model_len / page_size)
    total_pages = args.max_num_seqs * pages_per_seq

    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float8_e4m3fn

    cache_shape = wrapper.get_kv_cache_shape(
        total_pages,
        page_size,
        NUM_KV_HEADS_PER_RANK,
        HEAD_DIM,
        kv_dtype,
        kv_layout=kv_layout,
    )
    itemsize = jnp.dtype(kv_dtype).itemsize
    bytes_per_token = _bytes_per_token(cache_shape, page_size, itemsize)

    expected = EXPECTED_BYTES_PER_TOKEN[arm]
    if bytes_per_token != expected:
        raise SystemExit(
            f"FATAL: arm={arm} resolved {bytes_per_token} B/token from "
            f"kv_cache_shape={cache_shape} (page_size={page_size}), expected "
            f"{expected}. Either the layout did not take effect or "
            f"get_kv_cache_shape changed. Refusing to report a number.")

    query, new_k, new_v, kv_cache, kernel_args, meta = _build_phase_inputs(
        jnp,
        jax,
        phase=phase,
        page_size=page_size,
        decode_kv_len=args.decode_kv_len,
        prefill_q_tokens=args.prefill_q_tokens,
        prefill_context=args.prefill_context,
        max_num_seqs=args.max_num_seqs,
        pages_per_seq=pages_per_seq,
        kv_dtype=kv_dtype,
        q_dtype=q_dtype,
        cache_shape=cache_shape,
        cache_init=args.cache_init,
        seed=args.seed,
    )

    capacity = bq = bkv = None
    try:
        capacity, bq, bkv = _block_sizes_for(
            wrapper,
            configs,
            phase=phase,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            max_num_seqs=args.max_num_seqs,
            kv_layout=kv_layout,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
    except Exception as exc:  # noqa: BLE001 - accounting only, never fatal
        print(f"  warning: could not resolve block sizes ({exc!r}); "
              "bandwidth will be reported from the unique-bytes model only")

    pairs = None
    if bq and bkv:
        pairs = wrapper.schedule_pairs(meta["q_lens"], meta["kv_lens"], bq,
                                       bkv)
        if capacity and pairs > capacity:
            print(f"  warning: schedule pairs={pairs} exceeds the kernel's "
                  f"SMEM schedule capacity={capacity}; the kernel may fault "
                  "or silently truncate its schedule")

    # Two KV-traffic models, both reported:
    #   unique   -- every cached token once, rounded up to whole pages. The
    #               floor: what the step moves with perfect reuse.
    #   streamed -- what the kernel's (query block, KV block) schedule actually
    #               fetches, counting a KV block once per query block that
    #               attends to it. Equal to unique for decode (one query block
    #               per sequence); larger for a long prefill.
    unique_tokens = sum(
        math.ceil(kv_len / page_size) * page_size
        for kv_len in meta["kv_lens"])
    kv_bytes_unique = unique_tokens * bytes_per_token
    kv_bytes_streamed = (pairs * bkv * bytes_per_token) if pairs else None

    if phase == Phase.EMPTY:
        # Both kernels plan zero steps, so no KV is read at all. Reporting the
        # schedule's nominal volume here would manufacture a bandwidth figure
        # for traffic that never happens.
        kv_bytes_unique = 0.0
        kv_bytes_streamed = 0.0
        pairs = 0

    step = _make_step(
        jax,
        wrapper,
        kv_layout=kv_layout,
        sm_scale=args.sm_scale if args.sm_scale is not None else HEAD_DIM**
        -0.5,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
        v_scale=args.v_scale,
    )

    streamed_mb = ("n/a" if kv_bytes_streamed is None else
                   "%.1f MB" % (kv_bytes_streamed / 1e6))
    print(f"\n[{arm} / {phase}]")
    print(f"  kv_cache_shape      = {tuple(cache_shape)}  float8_e4m3fn")
    print(f"  page_size           = {page_size}  "
          f"pages_per_seq={pages_per_seq}  total_pages={total_pages}")
    print(f"  BYTES PER TOKEN     = {bytes_per_token:.1f}   "
          f"(expected {expected} for {arm})")
    print(f"  live seqs           = {meta['num_live_seqs']}  "
          f"q_tokens={meta['total_q_tokens']}  "
          f"kv_len[0]={meta['kv_lens'][0]}")
    print(f"  block sizes         = bq={bq} bkv={bkv} "
          f"schedule_pairs={pairs} capacity={capacity}")
    print(f"  kv bytes / step     = unique {kv_bytes_unique / 1e6:.1f} MB, "
          f"streamed {streamed_mb}")

    # Build identity. Without this an A/B across a trace-time env var is
    # circular: the timing difference is the only evidence the two arms are
    # different programs, and "the arms differ because they timed differently"
    # is not evidence. The hash covers the compiled HLO, which carries the
    # Mosaic payload in the custom call's backend_config, so a kernel-body
    # change moves it. Two arms printing the same hash is a void A/B.
    guard_env = os.environ.get("VLLM_RPA_GUARD_EMPTY_MEMSET", "<unset>")
    try:
        compiled_text = step.lower(query, new_k, new_v, kv_cache,
                                   **kernel_args).compile().as_text()
        build_hash = hashlib.sha256(
            compiled_text.encode("utf-8")).hexdigest()[:16]
        build_bytes = len(compiled_text)
    except Exception as exc:  # noqa: BLE001 - identity only, never fatal
        build_hash = f"<unavailable: {exc!r}>"
        build_bytes = -1
    print(f"  VLLM_RPA_GUARD_EMPTY_MEMSET = {guard_env}")
    print(f"  BUILD HASH          = {build_hash}  "
          f"(sha256/16 of compiled HLO, {build_bytes} B)")

    compile_start = time.perf_counter()
    out, kv_cache = step(query, new_k, new_v, kv_cache, **kernel_args)
    jax.block_until_ready((out, kv_cache))
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(max(0, args.warmup - 1)):
        out, kv_cache = step(query, new_k, new_v, kv_cache, **kernel_args)
        jax.block_until_ready((out, kv_cache))

    profile_ctx = (jax.profiler.trace(args.profile_out)
                   if args.profile_out else contextlib.nullcontext())
    samples_ms: list[float] = []
    with profile_ctx:
        for _ in range(args.iters):
            start_ns = time.perf_counter_ns()
            out, kv_cache = step(query, new_k, new_v, kv_cache, **kernel_args)
            jax.block_until_ready((out, kv_cache))
            samples_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

    median_ms = statistics.median(samples_ms)
    min_ms = min(samples_ms)
    ordered = sorted(samples_ms)
    p90_ms = ordered[math.ceil(0.9 * len(ordered)) - 1]

    gbps_unique_median = _gbps(kv_bytes_unique, median_ms / 1e3)
    gbps_unique_min = _gbps(kv_bytes_unique, min_ms / 1e3)
    gbps_streamed_median = _gbps(kv_bytes_streamed, median_ms / 1e3)
    gbps_streamed_min = _gbps(kv_bytes_streamed, min_ms / 1e3)

    print(f"  compile+first call  = {compile_seconds:.2f} s")
    print(f"  median              = {median_ms:.4f} ms   "
          f"min = {min_ms:.4f} ms   p90 = {p90_ms:.4f} ms")
    print(f"  achieved GB/s       = unique {_fmt(gbps_unique_median)} "
          f"(median) / {_fmt(gbps_unique_min)} (min);  "
          f"streamed {_fmt(gbps_streamed_median)} / "
          f"{_fmt(gbps_streamed_min)}")

    payload = {
        "kind": "batched_rpa_kv_layout",
        "model": MODEL_ID,
        "hf_revision": HF_REVISION,
        "layout": arm,
        "kv_layout_enum": kv_layout.name,
        "vllm_kv_cache_layout": os.environ.get("VLLM_KV_CACHE_LAYOUT"),
        "phase": phase,
        "build_hash": build_hash,
        "guard_env": guard_env,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "num_q_heads": NUM_Q_HEADS_PER_RANK,
        "num_kv_heads": NUM_KV_HEADS_PER_RANK,
        "head_dim": HEAD_DIM,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "kv_cache_dtype": "float8_e4m3fn",
        "q_dtype": "bfloat16",
        "page_size": page_size,
        "pages_per_seq": pages_per_seq,
        "total_pages": total_pages,
        "kv_cache_shape": list(cache_shape),
        "bytes_per_token": bytes_per_token,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "input_seq_len": INPUT_SEQ_LEN,
        "output_seq_len": OUTPUT_SEQ_LEN,
        "concurrency": CONCURRENCY,
        "num_live_seqs": meta["num_live_seqs"],
        "total_q_tokens": meta["total_q_tokens"],
        "kv_len": meta["kv_lens"][0],
        "bq": bq,
        "bkv": bkv,
        "schedule_pairs": pairs,
        "schedule_capacity": capacity,
        "kv_bytes_unique": kv_bytes_unique,
        "kv_bytes_streamed": kv_bytes_streamed,
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_seconds": compile_seconds,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "p90_ms": p90_ms,
        "achieved_gbps": gbps_unique_median,
        "achieved_gbps_unique_median": gbps_unique_median,
        "achieved_gbps_unique_min": gbps_unique_min,
        "achieved_gbps_streamed_median": gbps_streamed_median,
        "achieved_gbps_streamed_min": gbps_streamed_min,
        "samples_ms": samples_ms,
        "device_kind": jax.local_devices()[0].device_kind,
    }
    _emit_result_json(payload)
    return payload


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
def _check_layouts(jax, jnp, wrapper, configs, args) -> int:
    """Run both layouts on identical inputs and compare outputs.

    The comparison approach is
    ``tests/kernels/experimental/batched_rpa_longctx/test_seq_along_lane_parity.py``:
    fill each layout's cache with the same K/V by running the kernel once as a
    pure prefill, then run the measured step and compare. Page size is matched
    across the arms here (``--check-page-size``, default 128 -- the only page
    size SEQ_ALONG_LANE accepts) so that layout is the only variable.

    A layout that is fast because it reads the wrong bytes fails this.
    """
    import numpy as np

    page_size = args.check_page_size
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float8_e4m3fn
    sm_scale = args.sm_scale if args.sm_scale is not None else HEAD_DIM**-0.5

    rng = np.random.default_rng(args.seed)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    cases = [
        # (name, kv_len, q_len, distribution)
        ("decode", args.check_kv_len, 1, (1, 1, 1)),
        ("prefill", args.check_q_len, args.check_q_len, (0, 0, 1)),
    ]

    failures = 0
    for name, kv_len, q_len, distribution in cases:
        pages_per_seq = math.ceil(kv_len / page_size) + 1
        total_pages = pages_per_seq

        target_k = r(kv_len, NUM_KV_HEADS_PER_RANK, HEAD_DIM)
        target_v = r(kv_len, NUM_KV_HEADS_PER_RANK, HEAD_DIM)
        query = r(q_len, NUM_Q_HEADS_PER_RANK, HEAD_DIM)
        new_k = target_k[kv_len - q_len:kv_len]
        new_v = target_v[kv_len - q_len:kv_len]

        page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)
        kv_lens = jnp.array([kv_len], jnp.int32)
        cu_q_lens = jnp.array([0, q_len], jnp.int32)
        dist = jnp.asarray(np.asarray(distribution, np.int32))

        outputs = {}
        shapes = {}
        for arm in _LAYOUT_CHOICES:
            kv_layout = configs.KVLayout[arm.upper()]
            cache_shape = wrapper.get_kv_cache_shape(
                total_pages,
                page_size,
                NUM_KV_HEADS_PER_RANK,
                HEAD_DIM,
                kv_dtype,
                kv_layout=kv_layout,
            )
            shapes[arm] = tuple(cache_shape)
            bpt = _bytes_per_token(cache_shape, page_size,
                                   jnp.dtype(kv_dtype).itemsize)
            if bpt != EXPECTED_BYTES_PER_TOKEN[arm]:
                raise SystemExit(
                    f"FATAL: arm={arm} resolved {bpt} B/token from "
                    f"{cache_shape}, expected {EXPECTED_BYTES_PER_TOKEN[arm]}")

            # Pass 1: populate the cache with the whole K/V history.
            _, cache = wrapper.ragged_paged_attention(
                jnp.zeros((kv_len, NUM_Q_HEADS_PER_RANK, HEAD_DIM), q_dtype),
                jnp.asarray(target_k, kv_dtype),
                jnp.asarray(target_v, kv_dtype),
                jnp.zeros(cache_shape, kv_dtype),
                jnp.array([kv_len], jnp.int32),
                page_indices,
                jnp.array([0, kv_len], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
                sm_scale=sm_scale,
                kv_layout=kv_layout,
            )
            # Pass 2: the measured step.
            out, _ = wrapper.ragged_paged_attention(
                jnp.asarray(query, q_dtype),
                jnp.asarray(new_k, kv_dtype),
                jnp.asarray(new_v, kv_dtype),
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                dist,
                sm_scale=sm_scale,
                kv_layout=kv_layout,
            )
            outputs[arm] = np.asarray(out.astype(jnp.float32))

        if shapes["head_along_sublane"] == shapes["seq_along_lane"]:
            raise SystemExit(
                "FATAL: both arms resolved the same kv_cache_shape "
                f"{shapes['head_along_sublane']}; the comparison is vacuous.")

        a = outputs["head_along_sublane"]
        b = outputs["seq_along_lane"]
        diff = np.abs(a - b)
        max_abs = float(diff.max())
        denom = np.maximum(np.abs(a), 1e-6)
        max_rel = float((diff / denom).max())
        finite = bool(np.isfinite(b).all() and np.isfinite(a).all())
        ok = finite and max_abs <= args.atol

        print(f"\n[check / {name}] kv_len={kv_len} q_len={q_len} "
              f"page_size={page_size}")
        print(f"  head_along_sublane shape = {shapes['head_along_sublane']}")
        print(f"  seq_along_lane     shape = {shapes['seq_along_lane']}")
        print(f"  max_abs_err = {max_abs:.6g}   max_rel_err = {max_rel:.6g}   "
              f"finite = {finite}   atol = {args.atol}")
        print(f"  {'PASS' if ok else 'FAIL'}")

        _emit_result_json({
            "kind": "batched_rpa_kv_layout_check",
            "model": MODEL_ID,
            "case": name,
            "page_size": page_size,
            "kv_len": kv_len,
            "q_len": q_len,
            "num_q_heads": NUM_Q_HEADS_PER_RANK,
            "num_kv_heads": NUM_KV_HEADS_PER_RANK,
            "head_dim": HEAD_DIM,
            "kv_cache_dtype": "float8_e4m3fn",
            "shape_head_along_sublane": list(shapes["head_along_sublane"]),
            "shape_seq_along_lane": list(shapes["seq_along_lane"]),
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "finite": finite,
            "atol": args.atol,
            "passed": ok,
        })
        if not ok:
            failures += 1

    if failures:
        print(f"\nCHECK FAILED: {failures} case(s) exceeded atol={args.atol}")
        return 1
    print("\nCHECK PASSED: both layouts agree within tolerance")
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("A/B the batched-RPA HEAD_ALONG_SUBLANE and "
                     "SEQ_ALONG_LANE fp8 KV-cache layouts at "
                     f"{MODEL_ID}'s TP={TENSOR_PARALLEL_SIZE} shape."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--layout",
        choices=_LAYOUT_CHOICES,
        default="head_along_sublane",
        help="which arm to time; one arm per process",
    )
    parser.add_argument(
        "--phase",
        choices=(Phase.DECODE, Phase.PREFILL, Phase.EMPTY, "both"),
        default="both",
        help="which step shape to time",
    )
    parser.add_argument("--warmup",
                        type=int,
                        default=3,
                        help="warmup iterations, compile excluded")
    parser.add_argument("--iters",
                        type=int,
                        default=20,
                        help="timed iterations")
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help=("force a page size on both arms; default is each layout's own "
              f"production value ({PAGE_SIZE_HEAD_ALONG_SUBLANE} for "
              f"head_along_sublane, {PAGE_SIZE_SEQ_ALONG_LANE} for "
              "seq_along_lane, which is the only one the kernel accepts)"),
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--decode-kv-len",
                        type=int,
                        default=DECODE_KV_LEN,
                        help="cached tokens per sequence in the decode step")
    parser.add_argument("--prefill-q-tokens",
                        type=int,
                        default=PREFILL_Q_TOKENS,
                        help="query tokens in the chunked-prefill step")
    parser.add_argument("--prefill-context",
                        type=int,
                        default=0,
                        help="cached tokens preceding the prefill chunk")
    parser.add_argument("--cache-init",
                        choices=("random", "zeros"),
                        default="random")
    parser.add_argument("--sm-scale",
                        type=float,
                        default=None,
                        help="default: head_dim ** -0.5")
    parser.add_argument("--q-scale", type=float, default=None)
    parser.add_argument("--k-scale", type=float, default=None)
    parser.add_argument("--v-scale", type=float, default=None)
    parser.add_argument(
        "--profile-out",
        default=None,
        help=("Capture an xprof trace of the timing loop into this directory. "
              "Device time is the only way to see an effect smaller than this "
              "harness's ~0.25 ms of host dispatch per call."))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index",
                        type=int,
                        default=0,
                        help="index into jax.local_devices()")
    parser.add_argument(
        "--cross-check-env",
        action="store_true",
        help=("additionally resolve VLLM_KV_CACHE_LAYOUT through vLLM's "
              "get_kv_cache_layout() and abort on a mismatch; imports vLLM, "
              "so off by default"),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=("numerical parity mode: run both layouts on identical inputs "
              "and compare. Ignores --layout/--phase."),
    )
    parser.add_argument("--check-page-size",
                        type=int,
                        default=PAGE_SIZE_SEQ_ALONG_LANE,
                        help="page size used by BOTH arms in --check")
    parser.add_argument("--check-kv-len",
                        type=int,
                        default=4096,
                        help="cached length for the --check decode case")
    parser.add_argument("--check-q-len",
                        type=int,
                        default=1024,
                        help="chunk length for the --check prefill case")
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help=("absolute tolerance; 2e-2 is what "
              "test_seq_along_lane_parity.py uses"),
    )

    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)

    # The env must be exported before anything is imported. In --check both
    # layouts run in one process, so HND is exported as the more restrictive
    # of the two and the explicit enum argument is what actually selects.
    _export_vllm_layout_env("seq_along_lane" if args.check else args.layout)

    import jax
    import jax.numpy as jnp

    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        print("FATAL: no local TPU device; run this on a TPU host "
              "(e.g. via the run-command-in-cdk recipe).",
              file=sys.stderr)
        return 2
    if args.device_index:
        jax.config.update("jax_default_device", devices[args.device_index])
    device = jax.local_devices()[args.device_index]

    wrapper, configs = _import_kernel()

    print("=" * 78)
    print(f"batched-RPA KV-cache layout A/B -- {MODEL_ID}")
    print(f"  hf revision        = {HF_REVISION} (config.json verified "
          "2026-09-15)")
    print(f"  device             = {device.device_kind} "
          f"(index {args.device_index} of {len(devices)})")
    print(f"  TP={TENSOR_PARALLEL_SIZE} -> per rank: "
          f"num_q_heads={NUM_Q_HEADS_PER_RANK} "
          f"num_kv_heads={NUM_KV_HEADS_PER_RANK} head_dim={HEAD_DIM}")
    print(f"  workload           = ISL {INPUT_SEQ_LEN} / OSL {OUTPUT_SEQ_LEN} "
          f"/ concurrency {CONCURRENCY} / max_model_len {args.max_model_len}")
    print(f"  VLLM_KV_CACHE_LAYOUT = "
          f"{os.environ.get('VLLM_KV_CACHE_LAYOUT')!r} (exported; the kernel "
          "takes kv_layout as an explicit argument)")
    print("=" * 78)

    if args.check:
        return _check_layouts(jax, jnp, wrapper, configs, args)

    arm = args.layout
    kv_layout = configs.KVLayout[arm.upper()]
    if args.cross_check_env:
        note = _cross_check_env(arm, kv_layout)
        if note:
            print(f"  layout cross-check: {note}")
    else:
        print("  layout cross-check: skipped (--cross-check-env to enable); "
              "the arm is selected by the explicit kv_layout argument and "
              "proved by the bytes/token assertion below")
    print(f"  ARM = {arm}  ({kv_layout.name})")

    phases = ([Phase.DECODE, Phase.PREFILL]
              if args.phase == "both" else [args.phase])
    for phase in phases:
        _run_phase(jax,
                   jnp,
                   wrapper,
                   configs,
                   arm=arm,
                   kv_layout=kv_layout,
                   phase=phase,
                   args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
