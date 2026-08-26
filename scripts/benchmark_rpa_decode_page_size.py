# SPDX-License-Identifier: Apache-2.0
"""Operator latency sweep of RPA v3 decode over the KV-cache page size.

Motivation. An xprof capture of Qwen3-Coder-30B-A3B-Instruct-FP8 at ISL 8192 /
OSL 1024 (TP8, EP8, fp8 KV cache) attributes 47.6% of all device time to the
decode ragged-paged-attention kernel, which the trace names
``RPAd-p_16-bq_1_1-bkv_9216_9216``. The ``p_16`` is the KV page size, and it is
16 because ``PallasAttentionBackend.get_page_size`` hard-returns 16 whenever
``max_model_len > 8192``. At that page size the kernel issues one 8 KiB DMA per
page, i.e. ``kv_len / 16`` descriptors per sequence per layer, and the measured
kernel reaches only ~1.0 TB/s against a 3.69 TB/s HBM peak.

This script isolates that kernel at the exact serving shape and sweeps the page
size, so the DMA-granularity hypothesis can be tested for the cost of one TPU
job instead of a 40-minute end-to-end benchmark.

Only the JAX kernel is imported; the vLLM serving stack is not, so this runs
standalone on a bare TPU pod.

Correctness is checked, not assumed: every page size is fed the *same logical*
K/V content, merely paged differently, so all page sizes must produce the same
attention output. Page size 16 is the reference.

Calibration. The harness is only useful if it lands near the 259 us/call the
serving profile reports. The first version did not — it read 2826 us/call,
because the outer ``jax.jit`` did not donate ``kv_cache`` and XLA therefore
copied the whole 1.93 GiB cache on every call, adding ~1 ms of page-size
independent HBM traffic. Compare ``us_per_call`` against 259 us before trusting
any ratio this script prints; ``--profile-dir`` captures an xprof trace per arm
so the remaining time can be attributed op by op.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Serving shape recovered from the xprof capture of job
# j-3eadcb82-0d4e-4c08-99db (rank 0, decode_only phase).
DEFAULT_NUM_SEQS = 64  # --max-num-seqs
DEFAULT_KV_LEN = 8192  # steady-state context at ISL 8192
DEFAULT_MAX_MODEL_LEN = 9216  # --max-model-len, sets pages_per_seq
DEFAULT_NUM_Q_HEADS = 4  # 32 q heads / TP8
# 4 kv heads at TP8: vLLM's get_num_kv_heads is max(1, 4 // 8) == 1, so each
# rank holds one kv head and the head is replicated across ranks. Note this
# does NOT halve the bytes the kernel moves: the fp8 cache layout pads to
# align_to(num_kv_heads * 2, kv_packing) == align_to(2, 4) == 4 merged head
# slots, so half of every DMA'd byte is padding at this config.
DEFAULT_NUM_KV_HEADS = 1
DEFAULT_HEAD_DIM = 128
# 235066 pages x 16 tokens in the profiled run. Held constant across page sizes
# so the HBM footprint and the page-scatter pattern stay comparable.
DEFAULT_TOTAL_SLOTS = 235066 * 16


def _load_kernel(repo_root: Path):
    """Import the v3 RPA kernel without pulling in the vLLM serving stack."""
    src = repo_root / "src"
    rpa_dir = src / "vllm_torchtpu" / "kernels" / "ragged_paged_attention" / "v3"
    package_paths = {
        "vllm_torchtpu": src / "vllm_torchtpu",
        "vllm_torchtpu.kernels": src / "vllm_torchtpu" / "kernels",
        "vllm_torchtpu.kernels.ragged_paged_attention":
        src / "vllm_torchtpu" / "kernels" / "ragged_paged_attention",
        "vllm_torchtpu.kernels.ragged_paged_attention.v3": rpa_dir,
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("vllm_torchtpu.kernels.ragged_paged_attention.v3.util",
          rpa_dir / "util.py")
    _load(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes",
        rpa_dir / "tuned_block_sizes.py",
    )
    return _load("vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel",
                 rpa_dir / "kernel.py")


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _logical_kv(
    *,
    num_seqs: int,
    kv_len: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The K/V content of every sequence, page-size independent.

    Returned as float32 so each page size quantizes from identical source
    values; otherwise the fp8 rounding itself would differ between arms and the
    cross-page-size comparison would be meaningless.
    """
    rng = np.random.default_rng(seed)
    shape = (num_seqs, kv_len, num_kv_heads, head_dim)
    keys = rng.standard_normal(shape, dtype=np.float32)
    values = rng.standard_normal(shape, dtype=np.float32)
    return keys, values


def _build_inputs(
    kernel_mod,
    *,
    page_size: int,
    num_seqs: int,
    kv_len: int,
    max_model_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    total_slots: int,
    logical_k: np.ndarray,
    logical_v: np.ndarray,
    seed: int,
) -> dict[str, jax.Array]:
    """Lay the same logical K/V out into a cache paged at ``page_size``."""
    kv_dtype = jnp.float8_e4m3fn
    rng = np.random.default_rng(seed)

    pages_per_seq = cdiv(max_model_len, page_size)
    total_num_pages = max(total_slots // page_size,
                          num_seqs * pages_per_seq + 1)

    cache_shape = kernel_mod.get_kv_cache_shape(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
    )
    # (total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing,
    #  head_dim); K and V for head h live at merged index 2h and 2h+1.
    _, _, x2_per_packing, kv_packing, padded_head_dim = cache_shape
    merged_heads = x2_per_packing * kv_packing
    assert merged_heads >= 2 * num_kv_heads
    assert padded_head_dim >= head_dim

    # Scatter every sequence's pages randomly across the pool. A compact
    # allocation would give the DMA engine near-sequential descriptors and
    # flatter the large page sizes for the wrong reason.
    page_indices = rng.choice(total_num_pages,
                              size=num_seqs * pages_per_seq,
                              replace=False).astype(np.int32)

    # Built on device, one sequence at a time. The full cache is ~2 GiB in fp8
    # and would be ~8 GiB as a host float32 staging array, so it is never
    # materialized at that width.
    cache = jnp.zeros(cache_shape, dtype=kv_dtype)
    padded_len = pages_per_seq * page_size
    for seq in range(num_seqs):
        seq_pages = page_indices[seq * pages_per_seq:(seq + 1) * pages_per_seq]
        merged = np.zeros((padded_len, merged_heads, padded_head_dim),
                          dtype=np.float32)
        # The final token of each sequence is supplied as new_k/new_v below and
        # written by the kernel itself, mirroring a real decode step.
        span = kv_len - 1
        for head in range(num_kv_heads):
            merged[:span, 2 * head, :head_dim] = logical_k[seq, :span, head]
            merged[:span, 2 * head + 1, :head_dim] = logical_v[seq, :span,
                                                               head]
        paged = merged.reshape(pages_per_seq, page_size, x2_per_packing,
                               kv_packing, padded_head_dim)
        cache = cache.at[jnp.asarray(seq_pages)].set(
            jnp.asarray(paged, dtype=kv_dtype))

    q_rng = np.random.default_rng(seed + 1)
    queries = q_rng.standard_normal((num_seqs, num_q_heads, head_dim),
                                    dtype=np.float32)

    # One new token per sequence: the classic decode step.
    new_k = logical_k[:, kv_len - 1, :, :]
    new_v = logical_v[:, kv_len - 1, :, :]

    return {
        "queries": jnp.asarray(queries, dtype=jnp.bfloat16),
        "keys": jnp.asarray(new_k, dtype=kv_dtype),
        "values": jnp.asarray(new_v, dtype=kv_dtype),
        "kv_cache": cache,
        "kv_lens": jnp.full((num_seqs, ), kv_len, dtype=jnp.int32),
        "page_indices": jnp.asarray(page_indices, dtype=jnp.int32),
        "cu_q_lens": jnp.arange(num_seqs + 1, dtype=jnp.int32),
        "distribution": jnp.asarray([num_seqs, num_seqs, num_seqs],
                                    dtype=jnp.int32),
    }


def _make_runner(kernel_mod, head_dim: int):
    sm_scale = head_dim**-0.5

    # kv_cache (argument 3) MUST be donated. The kernel's `pl.pallas_call`
    # declares `input_output_aliases={7: 0, 9: 1}`, so it updates the cache in
    # place; without donation at this outer boundary XLA has to materialize a
    # copy of the whole ~1.93 GiB cache on every call. That copy is ~1 ms of
    # pure HBM traffic, it is identical in every arm, and it flattens the
    # page-size comparison this script exists to make.
    @functools.partial(jax.jit, donate_argnums=(3, ))
    def run(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens,
            distribution):
        return kernel_mod.ragged_paged_attention(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
        )

    return run


def _latencies(
    run,
    inputs: dict[str, jax.Array],
    *,
    warmups: int,
    iterations: int,
    batch: int,
    profile_dir: Path | None = None,
) -> tuple[float, list[float], jax.Array]:
    """Time the kernel with every operand already resident on device.

    Two things keep the number honest at a target of a few hundred microseconds
    per call:

    * Nothing is transferred host->device inside the timed region. A single
      100 KiB transfer would be a visible fraction of the result.
    * Each sample dispatches ``batch`` calls back to back and blocks once at
      the end, so per-call host dispatch overhead is amortized instead of being
      serialized with the device work. JAX dispatch is asynchronous, so the
      calls queue up and the device stays fed.

    The cache is threaded from call to call because ``run`` donates it.
    """
    kv_cache = inputs["kv_cache"]
    q = inputs["queries"]
    k = inputs["keys"]
    v = inputs["values"]
    static = (inputs["kv_lens"], inputs["page_indices"], inputs["cu_q_lens"],
              inputs["distribution"])

    def _chain(cache, n: int):
        out = None
        for _ in range(n):
            out, cache = run(q, k, v, cache, *static)
        out.block_until_ready()
        cache.block_until_ready()
        return out, cache

    start = time.perf_counter()
    out, kv_cache = _chain(kv_cache, 1)
    first_ms = (time.perf_counter() - start) * 1e3
    # Writing new_k/new_v into the cache is idempotent across repeats, so any
    # call's output would do; taking the first avoids holding a second copy.
    first_out = jax.device_get(out)

    if warmups > 1:
        _, kv_cache = _chain(kv_cache, warmups - 1)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _, kv_cache = _chain(kv_cache, batch)
        samples.append((time.perf_counter() - start) * 1e3 / batch)

    if profile_dir is not None:
        with jax.profiler.trace(str(profile_dir)):
            _, kv_cache = _chain(kv_cache, batch)

    return first_ms, samples, first_out


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": p90,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stddev_ms": statistics.pstdev(samples),
    }


def _write_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-sizes",
                        type=str,
                        default="16,32,64,128,256",
                        help="comma-separated KV page sizes to sweep")
    parser.add_argument("--num-seqs", type=int, default=DEFAULT_NUM_SEQS)
    parser.add_argument("--kv-len", type=int, default=DEFAULT_KV_LEN)
    parser.add_argument("--max-model-len",
                        type=int,
                        default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads",
                        type=int,
                        default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--total-slots", type=int, default=DEFAULT_TOTAL_SLOTS)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch",
                        type=int,
                        default=8,
                        help="calls dispatched per timed sample; blocks once "
                        "at the end so host dispatch is amortized")
    parser.add_argument("--profile-dir",
                        type=Path,
                        default=None,
                        help="if set, capture one xprof trace per page size "
                        "into <dir>/p<page_size>/ after the timed loop")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    kernel_mod = _load_kernel(repo_root)

    page_sizes = [int(x) for x in args.page_sizes.split(",") if x.strip()]

    _write_jsonl(
        args.output,
        {
            "event": "environment",
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "benchmark": "rpa_v3_decode_page_size",
            "num_seqs": args.num_seqs,
            "kv_len": args.kv_len,
            "max_model_len": args.max_model_len,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "page_sizes": page_sizes,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "batch": args.batch,
        },
    )

    logical_k, logical_v = _logical_kv(
        num_seqs=args.num_seqs,
        kv_len=args.kv_len,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        seed=args.seed,
    )

    run = _make_runner(kernel_mod, args.head_dim)

    # Two byte counts, because at this config they differ by 2x and quoting
    # only one is misleading.
    #   payload: the K/V values the kernel actually needs.
    #   physical: what it has to DMA, including the head-slot padding the fp8
    #             cache layout forces (align_to(num_kv_heads*2, kv_packing)).
    probe_shape = kernel_mod.get_kv_cache_shape(1, 1, args.num_kv_heads,
                                                args.head_dim,
                                                jnp.float8_e4m3fn)
    _, _, x2_per_packing, kv_packing, padded_head_dim = probe_shape
    physical_bytes_per_token = x2_per_packing * kv_packing * padded_head_dim
    payload_bytes_per_call = (args.num_seqs * args.kv_len * 2 *
                              args.num_kv_heads * args.head_dim)
    physical_bytes_per_call = (args.num_seqs * args.kv_len *
                               physical_bytes_per_token)

    reference_out = None
    results: dict[int, dict] = {}
    for page_size in page_sizes:
        pages_per_seq = cdiv(args.max_model_len, page_size)
        inputs = _build_inputs(
            kernel_mod,
            page_size=page_size,
            num_seqs=args.num_seqs,
            kv_len=args.kv_len,
            max_model_len=args.max_model_len,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            total_slots=args.total_slots,
            logical_k=logical_k,
            logical_v=logical_v,
            seed=args.seed,
        )
        profile_dir = (None if args.profile_dir is None else args.profile_dir /
                       f"p{page_size}")
        first_ms, samples, out = _latencies(run,
                                            inputs,
                                            warmups=args.warmups,
                                            iterations=args.iterations,
                                            batch=args.batch,
                                            profile_dir=profile_dir)
        stats = _stats(samples)
        seconds = stats["median_ms"] * 1e-3
        record = {
            "event": "result",
            "page_size": page_size,
            "pages_per_seq": pages_per_seq,
            "dmas_per_call": args.num_seqs * cdiv(args.kv_len, page_size),
            "page_bytes": page_size * physical_bytes_per_token,
            "compile_and_first_call_ms": first_ms,
            "us_per_call": stats["median_ms"] * 1e3,
            "achieved_gb_s": physical_bytes_per_call / seconds / 1e9,
            "payload_gb_s": payload_bytes_per_call / seconds / 1e9,
            "physical_bytes_per_call": physical_bytes_per_call,
            "payload_bytes_per_call": payload_bytes_per_call,
            **stats,
        }

        if not args.skip_correctness:
            out_np = np.asarray(jax.device_get(out), dtype=np.float32)
            if reference_out is None:
                reference_out = out_np
                record["correctness"] = "reference"
            else:
                delta = np.abs(out_np - reference_out)
                denom = np.abs(reference_out).mean() + 1e-6
                record["correctness"] = {
                    "vs_page_size": page_sizes[0],
                    "finite": bool(np.isfinite(out_np).all()),
                    "max_abs_diff": float(delta.max(initial=0.0)),
                    "mean_abs_diff": float(delta.mean()),
                    "rel_mean_abs_diff": float(delta.mean() / denom),
                    "allclose_2pct": bool(
                        np.allclose(out_np, reference_out, rtol=2e-2,
                                    atol=2e-2)),
                }

        results[page_size] = record
        _write_jsonl(args.output, record)

    baseline = results.get(page_sizes[0])
    if baseline is not None:
        for page_size, record in results.items():
            if page_size == page_sizes[0]:
                continue
            speedup = baseline["us_per_call"] / record["us_per_call"]
            _write_jsonl(
                args.output,
                {
                    "event": "speedup",
                    "page_size": page_size,
                    "vs_page_size": page_sizes[0],
                    "speedup_x": speedup,
                    "delta_pct": (speedup - 1.0) * 100.0,
                },
            )

    _write_jsonl(args.output, {"event": "done"})


if __name__ == "__main__":
    main()
