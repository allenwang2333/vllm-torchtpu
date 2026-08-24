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
"""Single-device microbenchmark for the FP8 MoE path.

Times the ops a Qwen3-Coder-30B-A3B-Instruct-FP8 decoder layer runs per
forward under TP=8 + expert parallel: the routing permute, the two grouped
matmuls, and the weighted combine. Inputs are built by calling the real
``prepare_routed_gmm_inputs`` so ``group_sizes`` has the production property
that ``sum(group_sizes) ~= num_tokens`` while ``lhs`` has ``num_tokens * topk``
rows -- most rows are routing padding the kernel never walks.

Runs standalone on one TPU core; no vLLM engine, no torch. Every measurement
prints a single ``RESULT {json}`` line so a baseline job and a post-change job
can be diffed mechanically.

Example:
    python scripts/moe_gmm_microbench.py --case both --op all --label BASELINE
    python scripts/moe_gmm_microbench.py --case decode --op gmm1 --sweep tiles
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from statistics import median

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (TileSizes, calculate_tiling,
                                                   get_scope_name, gmm_v2,
                                                   make_gmm_configs)
from vllm_torchtpu.layers.common.fused_moe_gmm import (
    moe_gmm, prepare_routed_gmm_inputs, resolve_onehot_permute_threshold)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, from its config.json.
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
NUM_GLOBAL_EXPERTS = 128
TOPK = 8
NUM_LAYERS = 48
ACTIVATION = "silu"

# Serving config under benchmark: --tensor-parallel-size=8 --enable-expert-parallel.
TP_SIZE = 8

# max_num_batched_tokens=8192 with one 8192-token request per prefill, and
# max_num_seqs=64 one-token requests per decode step.
DECODE_TOKENS = 64
PREFILL_TOKENS = 8192


@dataclass(frozen=True)
class Weights:
    """Per-rank MoE weights in gmm_v2 layout, already fp8-quantized."""

    w1: jax.Array  # [E, hidden, 2 * inter] fp8, gate and up fused
    w1_scale: jax.Array  # [E, 1, 1, 2 * inter] f32
    w2: jax.Array  # [E, inter, hidden] fp8
    w2_scale: jax.Array  # [E, 1, 1, hidden] f32


@dataclass(frozen=True)
class Routed:
    """Output of the real routing prologue for one token batch."""

    hidden: jax.Array  # [T, hidden] bf16
    topk_indices_local: jax.Array  # [T, topk] int32, -1 marks a non-local expert
    topk_weights_local: jax.Array  # [T, topk] f32
    x: jax.Array  # [T * topk, hidden] bf16, gmm1 lhs
    group_sizes: jax.Array  # [E] int32, sums to the locally-owned row count
    argsort_revert: jax.Array  # [T * topk] int32
    topk_weights_flat: jax.Array  # [T * topk] f32
    valid_mask: jax.Array  # [T * topk] bool
    num_tokens: int
    valid_rows: int


def _quantize_fp8(x: jax.Array, axis: int, block_size: int):
    """Blockwise fp8 quantization, mirroring the JAX helper in
    tests/kernels/test_gmm_v2.py. Copied rather than imported because tests/
    is not on PYTHONPATH inside the CDK pod image."""
    dtype = jnp.float8_e4m3fn
    info = jnp.finfo(dtype)
    max_val, min_val = float(info.max), float(info.min)

    orig_shape = x.shape
    blocked_shape = orig_shape[:axis] + (-1,
                                         block_size) + orig_shape[axis + 1:]
    x_blocked = x.reshape(blocked_shape)

    abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
    scale = abs_max / max_val
    x_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)
    x_q = jnp.nan_to_num(x_q.reshape(orig_shape))
    scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
    return x_q, scale


def _make_weights(key: jax.Array, local_experts: int, hidden: int,
                  inter: int) -> Weights:
    """Build fp8 MoE weights in the layout the runtime actually produces.

    The checkpoint ships [128, 128] block scales, but _process_fp8_moe_weights
    dequantizes and requantizes with MOE_REQUANTIZE_BLOCK_SIZE (default unset),
    which collapses to one scale per output channel over the whole contracting
    dim. That is what makes gmm_v2 take its quantize-lhs / dequantize-after
    path, so the microbenchmark must reproduce it exactly.
    """
    k1, k2 = jax.random.split(key)

    # Checkpoint layout is [E, N_out, K]; quantize over K, then transpose into
    # the [E, K, N_out] rhs layout gmm_v2 wants.
    w1_ref = jax.random.uniform(k1, (local_experts, 2 * inter, hidden),
                                jnp.bfloat16, -1.0, 1.0)
    w1_q, w1_s = _quantize_fp8(w1_ref, axis=2, block_size=hidden)
    w1 = jnp.transpose(w1_q, (0, 2, 1))
    w1_scale = jnp.expand_dims(jnp.transpose(w1_s, (0, 2, 1)), axis=1)

    w2_ref = jax.random.uniform(k2, (local_experts, hidden, inter),
                                jnp.bfloat16, -1.0, 1.0)
    w2_q, w2_s = _quantize_fp8(w2_ref, axis=2, block_size=inter)
    w2 = jnp.transpose(w2_q, (0, 2, 1))
    w2_scale = jnp.expand_dims(jnp.transpose(w2_s, (0, 2, 1)), axis=1)

    return Weights(w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale)


def _make_routed(key: jax.Array, num_tokens: int, hidden: int,
                 global_experts: int, local_experts: int, topk: int,
                 ep_rank: int, threshold: int) -> Routed:
    """Route `num_tokens` tokens over the global expert set, keep this rank's
    shard, and run the real permute prologue."""
    k_hidden, k_route = jax.random.split(key)

    hidden_states = jax.random.uniform(k_hidden, (num_tokens, hidden),
                                       jnp.bfloat16, -1.0, 1.0)

    # Top-k over random router logits gives distinct experts per token and a
    # roughly uniform expert load, which is the optimistic case for balance.
    logits = jax.random.uniform(k_route, (num_tokens, global_experts),
                                jnp.float32)
    topk_weights, topk_indices = jax.lax.top_k(logits, topk)
    topk_weights = jax.nn.softmax(topk_weights, axis=-1)

    # Expert-parallel shard: rank r owns global experts [r*E, (r+1)*E).
    lo = ep_rank * local_experts
    local_ids = topk_indices - lo
    topk_indices_local = jnp.where(
        (topk_indices >= lo) & (topk_indices < lo + local_experts), local_ids,
        -1).astype(jnp.int32)

    x, group_sizes, argsort_revert, weights_flat, valid_mask = (
        prepare_routed_gmm_inputs(
            hidden_states,
            topk_indices_local,
            topk_weights,
            local_num_experts=local_experts,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=threshold,
        ))

    return Routed(
        hidden=hidden_states,
        topk_indices_local=topk_indices_local,
        topk_weights_local=topk_weights,
        x=x,
        group_sizes=group_sizes,
        argsort_revert=argsort_revert,
        topk_weights_flat=weights_flat,
        valid_mask=valid_mask,
        num_tokens=num_tokens,
        valid_rows=int(jnp.sum(group_sizes)),
    )


def _time(fn, iterations: int, async_repeats: int) -> tuple[float, float, float]:
    """Return (compile_and_warmup_s, blocked_median_ms, async_mean_ms).

    Two timers on purpose. The blocked one carries per-call Python and dispatch
    overhead, which for a 64-token decode GMM is larger than the device time
    being measured; the async one keeps the queue full and is what lines up
    with the xprof device-time column.
    """
    start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - start

    samples_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        out = fn()
        jax.block_until_ready(out)
        samples_ns.append(time.perf_counter_ns() - t0)
    blocked_ms = median(samples_ns) / 1e6

    for _ in range(3):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    pending = [fn() for _ in range(async_repeats)]
    jax.block_until_ready(pending[-1])
    async_ms = (time.perf_counter() - t0) * 1e3 / async_repeats
    del pending

    return compile_s, blocked_ms, async_ms


def _emit(record: dict) -> None:
    print("RESULT " + json.dumps(record, sort_keys=True), flush=True)


def _gmm_roofline(valid_rows: int, num_groups: int, size_k: int, size_n: int,
                  latency_ms: float) -> tuple[float, float]:
    """TFLOP/s and weight GB/s for one gmm_v2 call.

    FLOPs count only rows the kernel actually walks (sum(group_sizes)), not
    size_m -- with topk=8 over 1/8 of the experts, 7/8 of size_m is routing
    padding. gmm_v2.get_cost_estimate uses size_m and would overstate this 8x.

    Bytes count the full expert weight set: every local expert holds at least
    one routed row in practice, so all num_groups blocks stream from HBM
    regardless of how few rows each serves. At decode this is the binding
    constraint, not the arithmetic.
    """
    seconds = latency_ms / 1e3
    flops = 2.0 * valid_rows * size_k * size_n
    weight_bytes = num_groups * size_k * size_n  # fp8: 1 byte per element
    return flops / seconds / 1e12, weight_bytes / seconds / 1e9


def _scope(lhs, rhs, rhs_scale, group_sizes, tile_info, fuse_act,
           maybe_quantize_lhs, out_dtype) -> tuple[str, TileSizes]:
    """Reproduce the trace scope name gmm_v2 emits, so a microbenchmark row can
    be matched against an xprof op by name."""
    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        None,
        group_sizes,
        jnp.array([0], dtype=jnp.int32),
        tile_info=tile_info,
        vmem_limit_bytes=None,
        out_dtype=out_dtype,
        acc_dtype=None,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=False,
        fuse_act=fuse_act,
    )
    return get_scope_name(cfgs), cfgs.tiles


def _tile_candidates(mode: str, base: TileSizes, size_k: int,
                     size_n: int) -> tuple[TileSizes, ...]:
    """Candidate tilings, staged around the heuristic's own choice `base`.

    Staged rather than a full cross product: the full grid is ~375 points and
    each costs a Pallas compile, which is 25+ minutes of compile alone before
    a single measurement. `kn` and `bucket` are ~25 and ~15 points and between
    them cover the two independent axes.

    tile_k must divide size_k: rhs_scale carries one block over the whole
    contracting dim, so quant_block_size == size_k and gmm_v2 requires
    tile_k % quant_block_size == 0 or quant_block_size % tile_k == 0.
    """
    tile_ks = [k for k in (128, 256, 512, 1024, 2048) if size_k % k == 0]
    tile_ns = [
        n for n in (128, 256, 384, 512, 768, 1024, 1536, 2048) if size_n % n == 0
    ]
    out: list[TileSizes] = []
    if mode in ("kn", "tiles"):
        for tile_k in tile_ks:
            for tile_n in tile_ns:
                out.append(
                    TileSizes(base.tile_m, tile_k, tile_n, base.bucket_base))
    if mode in ("bucket", "tiles"):
        for bucket_base in (8, 16, 32, 64, 128, 256):
            for mult in (1, 2, 4):
                out.append(
                    TileSizes(bucket_base * mult, base.tile_k, base.tile_n,
                              bucket_base))
    # Always measure the heuristic's own choice so every sweep carries its own
    # in-job reference point rather than relying on a separate job.
    deduped = [base] + [t for t in out if t != base]
    seen, unique = set(), []
    for t in deduped:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return tuple(unique)


def _bench_gmm(name: str, routed: Routed, lhs: jax.Array, rhs: jax.Array,
               rhs_scale: jax.Array, *, fuse_act: str | None,
               out_dtype, maybe_quantize_lhs: bool, vmem_limit: int | None,
               tile_infos, args, case: str) -> None:
    size_m, size_k = lhs.shape
    size_n = rhs.shape[2]
    group_offset = jnp.array([0], dtype=jnp.int32)

    for tile_info in tile_infos:
        try:
            scope, tiles = _scope(lhs, rhs, rhs_scale, routed.group_sizes,
                                  tile_info, fuse_act, maybe_quantize_lhs,
                                  out_dtype)

            def call():
                return gmm_v2(
                    lhs=lhs,
                    rhs=rhs,
                    rhs_scale=rhs_scale,
                    group_sizes=routed.group_sizes,
                    group_offset=group_offset[0],
                    tile_info=tile_info,
                    vmem_limit_bytes=vmem_limit,
                    preferred_element_type=out_dtype,
                    maybe_quantize_lhs=maybe_quantize_lhs,
                    zero_initialize=False,
                    fuse_act=fuse_act,
                )

            compile_s, blocked_ms, async_ms = _time(call, args.iterations,
                                                    args.async_repeats)
        except Exception as exc:  # noqa: BLE001 - candidate triage
            text = str(exc).lower()
            kind = ("VMEM/OOM" if "vmem" in text or "out of memory" in text
                    else "error")
            logger.warning("%s skipped %s after %s: %s", name, tile_info, kind,
                           str(exc).partition("\n")[0])
            continue

        tflops, gbps = _gmm_roofline(routed.valid_rows, rhs.shape[0], size_k,
                                     size_n, async_ms)
        _emit({
            "label": args.label,
            "op": name,
            "case": case,
            "tokens": routed.num_tokens,
            "m": size_m,
            "k": size_k,
            "n": size_n,
            "sum_group": routed.valid_rows,
            "tm": tiles.tile_m,
            "tk": tiles.tile_k,
            "tn": tiles.tile_n,
            "bucket": tiles.bucket_base,
            "quant_lhs": maybe_quantize_lhs,
            "vmem_limit": vmem_limit,
            "scope": scope,
            "compile_s": round(compile_s, 3),
            "t_block_ms": round(blocked_ms, 4),
            "t_async_ms": round(async_ms, 4),
            "tflops": round(tflops, 2),
            "weight_gbps": round(gbps, 1),
        })

    gmm_v2.clear_cache()
    gc.collect()


def _bench_case(case: str, num_tokens: int, weights: Weights, args,
                threshold: int) -> None:
    key = jax.random.key(args.seed)
    routed = _make_routed(key, num_tokens, args.hidden, args.global_experts,
                          args.local_experts, args.topk, args.ep_rank,
                          threshold)
    logger.info(
        "case=%s tokens=%d routed_rows=%d valid_rows=%d group_sizes=%s",
        case, num_tokens, routed.x.shape[0], routed.valid_rows,
        list(map(int, routed.group_sizes)))

    explicit = None
    if args.tile_m and args.tile_k and args.tile_n and args.bucket_base:
        explicit = TileSizes(args.tile_m, args.tile_k, args.tile_n,
                             args.bucket_base)

    if args.sweep == "none":
        tile_infos_1 = (explicit or calculate_tiling, )
        tile_infos_2 = (explicit or calculate_tiling, )
    else:
        # Anchor the sweep on whatever the heuristic picks for this shape (or
        # on --tile-* if given), then vary one axis at a time around it.
        base_1 = explicit or _scope(routed.x, weights.w1, weights.w1_scale,
                                    routed.group_sizes, calculate_tiling,
                                    ACTIVATION, not args.no_quantize_lhs,
                                    None)[1]
        base_2 = explicit or _scope(
            jax.ShapeDtypeStruct((routed.x.shape[0], args.inter),
                                 jnp.bfloat16), weights.w2, weights.w2_scale,
            routed.group_sizes, calculate_tiling, None,
            not args.no_quantize_lhs, jnp.bfloat16)[1]
        tile_infos_1 = _tile_candidates(args.sweep, base_1, args.hidden,
                                        2 * args.inter)
        tile_infos_2 = _tile_candidates(args.sweep, base_2, args.inter,
                                        args.hidden)
        if args.max_candidates:
            dropped_1 = max(0, len(tile_infos_1) - args.max_candidates)
            dropped_2 = max(0, len(tile_infos_2) - args.max_candidates)
            if dropped_1 or dropped_2:
                logger.warning(
                    "--max-candidates=%d drops %d gmm1 and %d gmm2 candidates",
                    args.max_candidates, dropped_1, dropped_2)
            tile_infos_1 = tile_infos_1[:args.max_candidates]
            tile_infos_2 = tile_infos_2[:args.max_candidates]
        logger.info("sweep=%s gmm1_candidates=%d gmm2_candidates=%d",
                    args.sweep, len(tile_infos_1), len(tile_infos_2))

    want = args.op
    quant_lhs = not args.no_quantize_lhs

    if want in ("gmm1", "all"):
        _bench_gmm("gmm1",
                   routed,
                   routed.x,
                   weights.w1,
                   weights.w1_scale,
                   fuse_act=ACTIVATION,
                   out_dtype=None,
                   maybe_quantize_lhs=quant_lhs,
                   vmem_limit=args.vmem_limit,
                   tile_infos=tile_infos_1,
                   args=args,
                   case=case)

    if want in ("gmm2", "all"):
        gmm1_out = gmm_v2(lhs=routed.x,
                          rhs=weights.w1,
                          rhs_scale=weights.w1_scale,
                          group_sizes=routed.group_sizes,
                          group_offset=jnp.array([0], dtype=jnp.int32)[0],
                          zero_initialize=False,
                          fuse_act=ACTIVATION)
        gmm1_out = jax.block_until_ready(gmm1_out[:, :args.inter])
        gmm_v2.clear_cache()
        _bench_gmm("gmm2",
                   routed,
                   gmm1_out,
                   weights.w2,
                   weights.w2_scale,
                   fuse_act=None,
                   out_dtype=jnp.bfloat16,
                   maybe_quantize_lhs=quant_lhs,
                   vmem_limit=args.vmem_limit,
                   tile_infos=tile_infos_2,
                   args=args,
                   case=case)
        del gmm1_out

    if want in ("permute", "all"):
        # The routing prologue: argsort + SparseCore ragged_gather (or the
        # one-hot matmul below the threshold).
        def permute():
            return prepare_routed_gmm_inputs(
                routed.hidden,
                routed.topk_indices_local,
                routed.topk_weights_local,
                local_num_experts=args.local_experts,
                topk=args.topk,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=threshold,
            )[0]

        compile_s, blocked_ms, async_ms = _time(jax.jit(permute),
                                                args.iterations,
                                                args.async_repeats)
        _emit({
            "label": args.label,
            "op": "permute",
            "case": case,
            "tokens": num_tokens,
            "m": routed.x.shape[0],
            "sum_group": routed.valid_rows,
            "onehot": routed.x.shape[0] <= threshold,
            "threshold": threshold,
            "compile_s": round(compile_s, 3),
            "t_block_ms": round(blocked_ms, 4),
            "t_async_ms": round(async_ms, 4),
        })

    if want in ("moe", "all"):
        # The whole per-layer MoE block: gmm1 + slice + gmm2 + combine.
        def whole():
            return moe_gmm(
                routed.x,
                weights.w1,
                weights.w1_scale,
                None,
                weights.w2,
                weights.w2_scale,
                None,
                routed.group_sizes,
                routed.argsort_revert,
                routed.topk_weights_flat,
                routed.valid_mask,
                activation=ACTIVATION,
                num_tokens=num_tokens,
                topk=args.topk,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=threshold,
            )

        compile_s, blocked_ms, async_ms = _time(jax.jit(whole),
                                                args.iterations,
                                                args.async_repeats)
        _emit({
            "label": args.label,
            "op": "moe",
            "case": case,
            "tokens": num_tokens,
            "m": routed.x.shape[0],
            "sum_group": routed.valid_rows,
            "compile_s": round(compile_s, 3),
            "t_block_ms": round(blocked_ms, 4),
            "t_async_ms": round(async_ms, 4),
            "per_model_ms": round(async_ms * NUM_LAYERS, 3),
        })

    del routed
    gmm_v2.clear_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark the Qwen3-Coder-30B-A3B FP8 MoE ops.")
    parser.add_argument("--op",
                        default="all",
                        choices=("all", "gmm1", "gmm2", "permute", "moe"))
    parser.add_argument("--case",
                        default="both",
                        choices=("both", "decode", "prefill"))
    parser.add_argument("--tokens",
                        type=int,
                        default=None,
                        help="override the token count for a single case")
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--inter", type=int, default=MOE_INTERMEDIATE_SIZE)
    parser.add_argument("--global-experts",
                        type=int,
                        default=NUM_GLOBAL_EXPERTS)
    parser.add_argument("--local-experts",
                        type=int,
                        default=NUM_GLOBAL_EXPERTS // TP_SIZE)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--sweep",
                        default="none",
                        choices=("none", "kn", "bucket", "tiles"),
                        help="tile axis to sweep around the heuristic's own "
                        "choice; tiles is the full cross product and is slow")
    parser.add_argument("--max-candidates",
                        type=int,
                        default=0,
                        help="hard cap on swept candidates per gmm; 0 = no cap")
    parser.add_argument("--tile-m", type=int, default=0)
    parser.add_argument("--tile-k", type=int, default=0)
    parser.add_argument("--tile-n", type=int, default=0)
    parser.add_argument("--bucket-base", type=int, default=0)
    parser.add_argument("--vmem-limit", type=int, default=None)
    parser.add_argument("--no-quantize-lhs", action="store_true")
    parser.add_argument("--onehot-threshold",
                        type=int,
                        default=None,
                        help="override resolve_onehot_permute_threshold()")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--async-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="BASELINE")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    if args.iterations < 1 or args.async_repeats < 1:
        parser.error("--iterations and --async-repeats must be at least 1")

    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        parser.error("run this script on a TPU node")
    if args.device_index < 0 or args.device_index >= len(devices):
        parser.error(f"--device-index must be 0..{len(devices) - 1}")

    info = pltpu.get_tpu_info()
    print(
        json.dumps({
            "device_kind": devices[0].device_kind,
            "num_devices": len(devices),
            "num_lanes": info.num_lanes,
            "num_sublanes": info.num_sublanes,
            "mxu_column_size": info.mxu_column_size,
            "vmem_capacity_bytes": info.vmem_capacity_bytes,
            "fp8_ops_per_second": info.fp8_ops_per_second,
            "int8_ops_per_second": info.int8_ops_per_second,
            "sparse_core": str(info.sparse_core),
        },
                   sort_keys=True),
        flush=True,
    )

    threshold = (args.onehot_threshold if args.onehot_threshold is not None
                 else resolve_onehot_permute_threshold())
    logger.info("onehot_moe_permute_threshold=%d", threshold)

    if args.tokens is not None:
        cases = [("custom", args.tokens)]
    elif args.case == "decode":
        cases = [("decode", DECODE_TOKENS)]
    elif args.case == "prefill":
        cases = [("prefill", PREFILL_TOKENS)]
    else:
        cases = [("decode", DECODE_TOKENS), ("prefill", PREFILL_TOKENS)]

    device = devices[args.device_index]
    with jax.default_device(device):
        weights = _make_weights(jax.random.key(args.seed + 1),
                                args.local_experts, args.hidden, args.inter)
        jax.block_until_ready((weights.w1, weights.w2))
        for case, num_tokens in cases:
            _bench_case(case, num_tokens, weights, args, threshold)


if __name__ == "__main__":
    main()
