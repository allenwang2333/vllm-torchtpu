# SPDX-License-Identifier: Apache-2.0
"""Kernel-first validation for Kimi-K3 flat vs hierarchical routed MoE.

The production-shape benchmark intentionally measures one local rank at a
time.  A route-count sweep supplies the latency curve needed to reconstruct
the critical rank for an observed global routing histogram without loading the
full model.  The small-shape parity check validates the TP=2 mathematical
contract before any latency result is accepted.

The initial A/B fixes SparseCore routing, RGR v2, one-hot disabled, and padded
token skipping.  Run this on a TPU v7x worker in the same image that will be
used for the serving experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Freeze the kernel-selection knobs before importing fused_moe_gmm.
os.environ.setdefault("USE_MOE_SPARSE_CORE", "1")
os.environ.setdefault("ONEHOT_MOE_PERMUTE_THRESHOLD", "0")
os.environ.setdefault("RAGGED_GATHER_REDUCE_VERSION", "v2")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


@dataclass(frozen=True)
class Layout:
    name: str
    ep_size: int
    tp_size: int
    local_experts: int
    intermediate_size: int


FLAT = Layout(
    name="flat_ep32_tp1",
    ep_size=32,
    tp_size=1,
    local_experts=28,
    intermediate_size=3072,
)
HIERARCHICAL = Layout(
    name="hierarchical_ep16_tp2",
    ep_size=16,
    tp_size=2,
    local_experts=56,
    intermediate_size=1536,
)


def _block(value) -> None:
    jax.tree.map(lambda x: x.block_until_ready(), value)


def _stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(samples_ms),
    }


def _write_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _tp_parity() -> dict[str, float | bool]:
    """Check that two MXFP4 width shards reconstruct a flat MoE output."""
    num_tokens = 4
    hidden_size = 512
    intermediate_size = 256
    num_experts = 4
    topk = 2
    block_size = 32
    key = jax.random.key(20260826)
    hidden_key, w1_key, w2_key, w1_scale_key, w2_scale_key = jax.random.split(
        key, 5)
    hidden = (jax.random.normal(
        hidden_key, (num_tokens, hidden_size), dtype=jnp.float32) /
              10).astype(jnp.bfloat16)
    w1 = _native_fp4(
        w1_key, (num_experts, hidden_size, intermediate_size * 2))
    w2 = _native_fp4(w2_key,
                     (num_experts, intermediate_size, hidden_size))
    w1_scale = jax.random.uniform(
        w1_scale_key,
        (num_experts, hidden_size // block_size, 1, intermediate_size * 2),
        minval=0.01,
        maxval=0.1,
        dtype=jnp.float32,
    )
    w2_scale = jax.random.uniform(
        w2_scale_key,
        (num_experts, intermediate_size // block_size, 1, hidden_size),
        minval=0.01,
        maxval=0.1,
        dtype=jnp.float32,
    )
    topk_ids = jnp.asarray([[0, 1], [2, 3], [1, 3], [0, 2]],
                           dtype=jnp.int32)
    topk_weights = jnp.asarray([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45],
                                [0.8, 0.2]],
                               dtype=jnp.bfloat16)

    def run(local_w1, local_w2, local_w1_scale, local_w2_scale):
        return fused_moe_func(
            hidden_states=hidden,
            w1=local_w1,
            w2=local_w2,
            w1_scale=local_w1_scale,
            w2_scale=local_w2_scale,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk=topk,
            activation="silu",
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=0,
        )

    flat = run(w1, w2, w1_scale, w2_scale)
    gate, up = jnp.split(w1, 2, axis=-1)
    gate_scale, up_scale = jnp.split(w1_scale, 2, axis=-1)
    partials = []
    for tp_rank in range(2):
        start = tp_rank * (intermediate_size // 2)
        stop = start + intermediate_size // 2
        shard_w1 = jnp.concatenate((gate[..., start:stop], up[..., start:stop]),
                                   axis=-1)
        shard_w1_scale = jnp.concatenate(
            (gate_scale[..., start:stop], up_scale[..., start:stop]), axis=-1)
        scale_start = start // block_size
        scale_stop = stop // block_size
        partials.append(
            run(shard_w1, w2[:, start:stop, :], shard_w1_scale,
                w2_scale[:, scale_start:scale_stop, :, :]))
    hierarchical = partials[0] + partials[1]
    _block((flat, hierarchical))

    flat_np = np.asarray(flat, dtype=np.float32)
    hierarchical_np = np.asarray(hierarchical, dtype=np.float32)
    delta = np.abs(flat_np - hierarchical_np)
    finite = bool(np.isfinite(flat_np).all()
                  and np.isfinite(hierarchical_np).all())
    close = bool(
        np.allclose(hierarchical_np, flat_np, atol=2e-1, rtol=2e-1))
    return {
        "finite": finite,
        "allclose": close,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
    }


def _native_fp4(key: jax.Array, shape: tuple[int, int,
                                              int]) -> jax.Array:
    """Materialize random native fp4 in K-major [E, K, N] layout."""
    experts, size_k, size_n = shape
    if size_n % 2:
        raise ValueError(f"native fp4 output dimension must be even: {shape}")
    packed = jax.random.bits(key, (experts, size_k, size_n // 2),
                             dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn)


def _production_weights(layout: Layout, *, hidden_size: int,
                        block_size: int, seed: int):
    keys = jax.random.split(jax.random.key(seed), 4)
    w1 = _native_fp4(
        keys[0],
        (layout.local_experts, hidden_size, layout.intermediate_size * 2),
    )
    w2 = _native_fp4(
        keys[1],
        (layout.local_experts, layout.intermediate_size, hidden_size),
    )
    w1_scale = jax.random.uniform(
        keys[2],
        (layout.local_experts, hidden_size // block_size, 1,
         layout.intermediate_size * 2),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    w2_scale = jax.random.uniform(
        keys[3],
        (layout.local_experts, layout.intermediate_size // block_size, 1,
         hidden_size),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    _block((w1, w2, w1_scale, w2_scale))
    return w1, w2, w1_scale, w2_scale


def _routing_inputs(*, layout: Layout, route_count: int, num_tokens: int,
                    topk: int, hidden_size: int, seed: int):
    if not 0 <= route_count <= topk:
        raise ValueError(f"route_count must be in [0, {topk}]")
    if route_count > layout.local_experts:
        raise ValueError("route_count exceeds the local expert count")

    hidden = (jax.random.normal(jax.random.key(seed),
                                (num_tokens, hidden_size),
                                dtype=jnp.float32) /
              10).astype(jnp.bfloat16)
    nonlocal_start = layout.local_experts
    first_row = np.concatenate((np.arange(route_count, dtype=np.int32),
                                np.arange(nonlocal_start,
                                          nonlocal_start + topk - route_count,
                                          dtype=np.int32)))
    topk_ids = np.tile(first_row, (num_tokens, 1))
    topk_weights = np.zeros((num_tokens, topk), dtype=np.float32)
    topk_weights[0, :] = 1.0 / topk
    return (hidden, jnp.asarray(topk_ids),
            jnp.asarray(topk_weights, dtype=jnp.bfloat16))


def _benchmark_layout(layout: Layout, *, route_counts: list[int],
                      hidden_size: int, block_size: int, topk: int,
                      num_tokens: int, warmups: int, iterations: int,
                      seed: int) -> list[dict]:
    w1, w2, w1_scale, w2_scale = _production_weights(
        layout, hidden_size=hidden_size, block_size=block_size, seed=seed)
    records = []
    for route_count in route_counts:
        hidden, topk_ids, topk_weights = _routing_inputs(
            layout=layout,
            route_count=route_count,
            num_tokens=num_tokens,
            topk=topk,
            hidden_size=hidden_size,
            seed=seed + route_count,
        )

        def call():
            return fused_moe_func(
                hidden_states=hidden,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=None,
                w2_bias=None,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                experts_start=jnp.asarray(0, dtype=jnp.int32),
                topk=topk,
                activation="silu",
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=0,
                skip_padded_tokens=True,
            )

        started = time.perf_counter()
        result = call()
        _block(result)
        compile_and_first_ms = (time.perf_counter() - started) * 1e3
        for _ in range(max(0, warmups - 1)):
            _block(call())
        samples_ms = []
        for _ in range(iterations):
            started = time.perf_counter()
            result = call()
            _block(result)
            samples_ms.append((time.perf_counter() - started) * 1e3)

        result_np = np.asarray(result, dtype=np.float32)
        records.append({
            "event": "kernel_result",
            "layout": asdict(layout),
            "route_count": route_count,
            "compile_and_first_ms": compile_and_first_ms,
            "finite": bool(np.isfinite(result_np).all()),
            "padded_rows_zero": bool(np.count_nonzero(result_np[1:]) == 0),
            **_stats(samples_ms),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-counts", default="1,2,4,8,16")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--mxfp4-block-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    route_counts = [int(value) for value in args.route_counts.split(",")]
    if any(value > args.topk for value in route_counts):
        raise ValueError("every route count must be <= --topk")
    if args.num_tokens < 1:
        raise ValueError("--num-tokens must be positive")
    if args.hidden_size % args.mxfp4_block_size:
        raise ValueError("hidden size must be divisible by MXFP4 block size")
    for layout in (FLAT, HIERARCHICAL):
        if layout.intermediate_size % args.mxfp4_block_size:
            raise ValueError(
                f"{layout.name} intermediate size is not MXFP4 aligned")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    _write_jsonl(
        args.output,
        {
            "event": "environment",
            "benchmark": "kimi_k3_flat_vs_hierarchical_moe_kernel",
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "topk": args.topk,
            "num_tokens": args.num_tokens,
            "valid_tokens": 1,
            "hidden_size": args.hidden_size,
            "mxfp4_block_size": args.mxfp4_block_size,
            "onehot_threshold": 0,
            "ragged_gather_reduce_version": "v2",
        },
    )

    parity = _tp_parity()
    _write_jsonl(args.output, {"event": "tp_parity", **parity})
    if not parity["finite"] or not parity["allclose"]:
        raise RuntimeError(f"TP=2 parity failed: {parity}")

    for layout_index, layout in enumerate((FLAT, HIERARCHICAL)):
        records = _benchmark_layout(
            layout,
            route_counts=route_counts,
            hidden_size=args.hidden_size,
            block_size=args.mxfp4_block_size,
            topk=args.topk,
            num_tokens=args.num_tokens,
            warmups=args.warmups,
            iterations=args.iterations,
            seed=args.seed + layout_index * 1000,
        )
        for record in records:
            _write_jsonl(args.output, record)
            if not record["finite"] or not record["padded_rows_zero"]:
                raise RuntimeError(f"kernel correctness gate failed: {record}")


if __name__ == "__main__":
    main()
