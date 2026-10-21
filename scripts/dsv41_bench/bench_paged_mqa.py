# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare resident-Q paged MQA with the existing grouped decode scorer."""

import argparse
import functools
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode_scores import (
    grouped_decode_scores,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.paged_mqa_logits import (
    paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    PagedMQAConfig,
    PersistentScoreConfig,
)


@jax.jit
def quantize_rows(values):
    values = values.astype(jnp.float32)
    scale = jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(abs(values).max(-1), 1e-4) / 448)))
    return (values / scale[..., None]).astype(jnp.float8_e4m3fn), scale


@functools.partial(jax.jit, static_argnames=("page_records", "round_relu"))
def reference_scores(
    q, weights, pool, pages, lengths, page_records, q_scale, k_scale, round_relu
):
    position = jnp.arange(pages.shape[0] * page_records)
    physical = pages[position // page_records]
    records = jnp.maximum(physical, 0) * page_records + position % page_records
    if q_scale is None:
        keys = index_key_layout.gather_records(pool, records)
        total_keys = sum(bank.shape[0] for bank in pool) * 2
    else:
        keys = jnp.zeros((position.shape[0], 128), q.dtype)
        scales = jnp.zeros(position.shape, jnp.float32)
        base = 0
        for bank, scale in zip(pool, k_scale, strict=True):
            local = records - base
            valid = (local >= 0) & (local < bank.shape[0])
            local = jnp.clip(local, 0, bank.shape[0] - 1)
            keys = jnp.where(valid[:, None], bank[local], keys)
            scales = jnp.where(valid, scale[local], scales)
            base += bank.shape[0]
        total_keys = base
    logits = jnp.einsum(
        "thd,kd->thk",
        q,
        keys,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    if q_scale is not None:
        logits = logits * q_scale[..., None] * scales
    logits = jnp.maximum(logits, 0)
    if round_relu:
        logits = jax.lax.reduce_precision(logits, 8, 7)
    scores = (logits * weights[..., None]).sum(1)
    valid = (physical >= 0) & (physical < total_keys // page_records)
    valid = valid[None, :] & (position[None, :] < lengths[:, None])
    return jnp.where(valid, scores, -jnp.inf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--width", type=int, default=98304)
    parser.add_argument("--page-records", type=int, default=128)
    parser.add_argument("--key-tile", type=int, default=PagedMQAConfig().key_tile)
    parser.add_argument("--issue-depth", type=int, default=4)
    parser.add_argument("--query-group", type=int, default=PagedMQAConfig().query_group)
    parser.add_argument("--key-buffers", type=int, default=PagedMQAConfig().key_buffers)
    parser.add_argument("--baseline-key-tile", type=int, default=99328)
    parser.add_argument("--baseline-compute-tile", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    args = parser.parse_args()
    if not args.fp8:
        parser.error("This packed-cache checkpoint requires --fp8")
    args.out.mkdir(parents=True, exist_ok=True)
    for package in ("jax", "jaxlib", "libtpu"):
        print(f"{package}={importlib.metadata.version(package)}", flush=True)
    print(jax.devices(), flush=True)
    rng = np.random.default_rng(args.seed)
    r, qn, width, page = args.requests, args.queries, args.width, args.page_records
    # Native finite BF16 halfplanes, with a distinct physical page per request.
    words = rng.integers(0, 2**32, (r * width // 2, 128), dtype=np.uint32)
    words = (words & np.uint32(0x807F807F)) | np.uint32(0x3E003E00)
    pool = tuple(jnp.asarray(x, jnp.int32) for x in np.split(words, 2))
    del words
    pages = rng.permutation(r * width // page).reshape(r, -1).astype(np.int32)
    pages[0, 3] = -1
    pages = jnp.asarray(pages)
    q = jnp.asarray(rng.normal(size=(r * qn, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(r * qn, 32)), jnp.float32)
    lengths = jnp.full((r * qn,), width - 17, jnp.int32)
    offsets = jnp.arange(r + 1, dtype=jnp.int32) * qn
    original_q, original_pool = q, pool
    q_scale, k_scale = None, ()
    if args.fp8:
        q, q_scale = quantize_rows(q)
        quantized = [quantize_rows(index_key_layout.unpack_rows(bank)) for bank in pool]
        pool = tuple(pair[0] for pair in quantized)
        k_scale = tuple(pair[1] for pair in quantized)
    config = PagedMQAConfig(
        qn, args.key_tile, args.issue_depth, args.query_group, args.key_buffers
    )
    baseline_config = PersistentScoreConfig(
        args.baseline_key_tile, args.baseline_compute_tile, 32
    )
    from scripts.dsv41_bench.paged_mqa_cache import pack_pages

    packed_pool = jax.block_until_ready(
        tuple(pack_pages(k, scale) for k, scale in zip(pool, k_scale, strict=True))
    )
    candidate = functools.partial(
        paged_mqa_logits,
        q,
        weights,
        packed_pool,
        pages,
        lengths,
        offsets,
        page_records=page,
        q_scale=q_scale,
        k_scale=None,
        config=config,
    )
    baseline = functools.partial(
        grouped_decode_scores,
        original_q.reshape(r, qn, 32, 128),
        weights.reshape(r, qn, 32),
        original_pool,
        pages * page,
        lengths.reshape(r, qn),
        page_records=page,
        score_config=baseline_config,
        block_size=0,
        disable_bounds_checks=True,
    )
    started = time.perf_counter()
    actual = jax.block_until_ready(candidate())
    compile_s = time.perf_counter() - started
    print(f"candidate compiled: {compile_s:.3f}s", flush=True)
    squared_error, squared_reference, max_error = 0.0, 0.0, 0.0
    topk_overlap = []
    for group in range(r):
        rows = slice(group * qn, (group + 1) * qn)
        expected = reference_scores(
            q[rows],
            weights[rows],
            pool,
            pages[group],
            lengths[rows],
            page,
            None if q_scale is None else q_scale[rows],
            k_scale,
            True,
        )
        measured = np.asarray(actual[rows])
        np.testing.assert_allclose(measured, expected, rtol=3e-5, atol=1e-4)
        original = np.asarray(
            reference_scores(
                original_q[rows],
                weights[rows],
                original_pool,
                pages[group],
                lengths[rows],
                page,
                None,
                (),
                False,
            )
        )
        valid = np.isfinite(original)
        error = measured[valid].astype(np.float64) - original[valid]
        squared_error += np.square(error).sum()
        squared_reference += np.square(original[valid].astype(np.float64)).sum()
        max_error = max(max_error, float(abs(error).max()))
        for left, right in zip(measured, original, strict=True):
            k = min(512, int(np.isfinite(right).sum()))
            if k:
                a = np.argpartition(left, -k)[-k:]
                b = np.argpartition(right, -k)[-k:]
                topk_overlap.append(len(np.intersect1d(a, b)) / k)
    precision = {
        "max_abs_error_vs_bf16_fp32_relu": max_error,
        "relative_l2_error": float(np.sqrt(squared_error / squared_reference)),
        "top512_overlap": float(np.mean(topk_overlap)),
    }
    (args.out / "precision.json").write_text(json.dumps(precision, indent=2) + "\n")
    del actual
    print("Correctness against rounded-ReLU reference: PASS", precision, flush=True)
    info = pltpu.get_tpu_info()
    for name, run in (("baseline", baseline), ("paged_mqa", candidate)):
        for _ in range(3):
            jax.block_until_ready(run())
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            jax.block_until_ready(run())
            samples.append((time.perf_counter_ns() - started) / 1e6)
        record = {
            "variant": name,
            "requests": r,
            "queries": qn,
            "width": width,
            "page_records": page,
            "seed": args.seed,
            "config": asdict(config if name == "paged_mqa" else baseline_config),
            "latency_ms": float(np.median(samples)),
            "samples_ms": samples,
            "timing": "host_block_until_ready",
            "bounds_checks": False,
            "dtype": str(q.dtype) if name == "paged_mqa" else "bfloat16",
            "relu_dtype": "bfloat16" if name == "paged_mqa" else "float32",
            "precision": precision if name == "paged_mqa" else None,
            "minimum_hbm_bytes": (
                r * width * (132 if args.fp8 and name == "paged_mqa" else 256)
                + r * qn * width * 4
                + sum(
                    x.nbytes
                    for x in (
                        q if name == "paged_mqa" else original_q,
                        weights,
                        pages,
                        lengths,
                        offsets,
                    )
                )
                + (q_scale.nbytes if args.fp8 and name == "paged_mqa" else 0)
            ),
            "device_hbm_bytes_per_second": info.mem_bw_bytes_per_second,
        }
        print(json.dumps(record), flush=True)
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    if args.profile:
        with jax.profiler.trace(str(args.out / "xprof")):
            for name, run in (("baseline", baseline), ("paged_mqa", candidate)):
                for step in range(5):
                    with jax.profiler.TraceAnnotation(name, step_num=step):
                        jax.block_until_ready(run())


if __name__ == "__main__":
    main()
