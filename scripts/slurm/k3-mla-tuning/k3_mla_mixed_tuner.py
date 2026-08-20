#!/usr/bin/env python3
"""Tune Kimi-K3 MLA mixed-extend tiles at the serving kernel boundary.

This is a stage-1 kernel tuner.  Its geometry must be copied from an observed
serving compile; candidates that fail compilation are rejected before latency
ranking.  Finalists still require the stage-2 full-model compile gate described
in this directory's README.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "src"


def _install_lightweight_package_stubs() -> None:
    """Avoid importing vllm_torchtpu.__init__, which requires vLLM."""
    package = types.ModuleType("vllm_torchtpu")
    package.__path__ = [str(SOURCE_ROOT / "vllm_torchtpu")]
    sys.modules["vllm_torchtpu"] = package

    logger_module = types.ModuleType("vllm_torchtpu.logger")
    logger_module.init_logger = logging.getLogger
    sys.modules["vllm_torchtpu.logger"] = logger_module


_install_lightweight_package_stubs()

# Match the serving process' libtpu flags and JAX lowering patches before the
# first JAX import.  Importing the raw kernel without this module is not a
# serving-equivalent compiler environment.
import vllm_torchtpu.env_override  # noqa: E402,F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import shard_map  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.mla.v2.kernel import (  # noqa: E402
    mla_ragged_paged_attention, )


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        physical, separator, valid = item.partition(":")
        if not separator:
            valid = physical
        shape = (int(physical), int(valid))
        if shape[0] <= 0 or not 0 < shape[1] <= shape[0]:
            raise argparse.ArgumentTypeError(f"invalid shape: {item}")
        shapes.append(shape)
    return shapes


def parse_candidates(value: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for item in value.split(","):
        bkv, separator, bq = item.partition("x")
        if not separator or int(bkv) <= 0 or int(bq) <= 0:
            raise argparse.ArgumentTypeError(f"invalid candidate: {item}")
        candidates.append((int(bkv), int(bq)))
    return candidates


def parse_positive_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"invalid positive integer list: {value}")
    return values


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def emit(record: dict, output_file) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    output_file.write(line + "\n")
    output_file.flush()
    os.fsync(output_file.fileno())


def make_inputs(
    physical_tokens: int,
    valid_tokens: int,
    *,
    total_pages: int,
    max_num_seqs: int,
    pages_per_seq: int,
    page_size_per_kv_packing: int,
    kv_packing: int,
    local_num_heads: int,
    lkv_dim: int,
    r_dim: int,
):
    model_axis_size = len(jax.devices())
    num_heads = local_num_heads * model_axis_size
    logical_page_size = page_size_per_kv_packing * kv_packing
    required_pages = math.ceil(valid_tokens / logical_page_size)
    if required_pages > pages_per_seq:
        raise ValueError(
            f"valid_tokens={valid_tokens} needs {required_pages} pages, "
            f"but pages_per_seq={pages_per_seq}")
    if required_pages > total_pages:
        raise ValueError(
            f"valid_tokens={valid_tokens} needs {required_pages} pages, "
            f"but total_pages={total_pages}")

    # Match mla_attention's serving boundary: queries are token-major with the
    # global head count and are sharded over the eight-device model axis.
    ql_nope = jnp.ones((physical_tokens, num_heads, lkv_dim),
                       dtype=jnp.bfloat16)
    q_pe = jnp.ones((physical_tokens, num_heads, r_dim), dtype=jnp.bfloat16)
    new_kv_c = jnp.ones((physical_tokens, lkv_dim), dtype=jnp.bfloat16)
    new_k_pe = jnp.ones((physical_tokens, r_dim), dtype=jnp.bfloat16)
    padded_r_dim = math.ceil(r_dim / 128) * 128
    cache_kv = jnp.zeros(
        (total_pages, page_size_per_kv_packing, kv_packing,
         lkv_dim + padded_r_dim),
        dtype=jnp.bfloat16,
    )

    kv_lens = jnp.zeros((max_num_seqs, ),
                        dtype=jnp.int32).at[0].set(valid_tokens)
    cu_q_lens = jnp.full((max_num_seqs + 1, ), valid_tokens, dtype=jnp.int32)
    cu_q_lens = cu_q_lens.at[0].set(0)
    page_indices = jnp.full((max_num_seqs * pages_per_seq, ),
                            -1,
                            dtype=jnp.int32)
    page_indices = page_indices.at[:required_pages].set(
        jnp.arange(required_pages, dtype=jnp.int32))
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    mesh = Mesh(np.asarray(jax.devices()), ("model", ))
    specs = (
        P(None, "model", None),
        P(None, "model", None),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
    )
    inputs = tuple(
        jax.device_put(value, NamedSharding(mesh, spec))
        for value, spec in zip(
            (
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            ),
            specs,
            strict=True,
        ))
    jax.block_until_ready(inputs)
    return mesh, inputs


def make_invoker(mesh: Mesh, *, bkv: int, bq: int, q_split: int,
                 vmem_limit_bytes: int):
    """Build the same model-axis shard_map used by the serving wrapper."""

    def local_invoke(
        q,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        output, updated_cache = mla_ragged_paged_attention(
            q.transpose((1, 0, 2)),
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            num_kv_pages_per_block=(3, 1, bkv),
            num_queries_per_block=(1, 16, bq),
            vmem_limit_bytes=vmem_limit_bytes,
            decode_batch_size=4,
            mixed_q_split=q_split,
            s_dtype=jnp.bfloat16,
        )
        return output.transpose((1, 0, 2)), updated_cache

    return jax.jit(
        shard_map.shard_map(
            local_invoke,
            mesh=mesh,
            in_specs=(
                P(None, "model", None),
                P(None, "model", None),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=(P(None, "model", None), P()),
            check_rep=False,
        ))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def tune_candidate(
    physical_tokens: int,
    valid_tokens: int,
    *,
    bkv: int,
    bq: int,
    q_split: int,
    warmups: int,
    iterations: int,
    total_pages: int,
    max_num_seqs: int,
    pages_per_seq: int,
    page_size_per_kv_packing: int,
    kv_packing: int,
    vmem_limit_bytes: int,
    local_num_heads: int,
    lkv_dim: int,
    r_dim: int,
) -> dict:
    mesh, inputs = make_inputs(
        physical_tokens,
        valid_tokens,
        total_pages=total_pages,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=kv_packing,
        local_num_heads=local_num_heads,
        lkv_dim=lkv_dim,
        r_dim=r_dim,
    )

    invoke = make_invoker(
        mesh,
        bkv=bkv,
        bq=bq,
        q_split=q_split,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    compile_start = time.perf_counter()
    output, updated_cache = invoke(*inputs)
    jax.block_until_ready((output, updated_cache))
    expected_output_shape = (
        physical_tokens,
        local_num_heads * len(jax.devices()),
        lkv_dim,
    )
    if output.shape != expected_output_shape:
        raise ValueError(
            f"output shape {output.shape} != serving shape {expected_output_shape}"
        )
    compile_and_first_run_ms = (time.perf_counter() - compile_start) * 1000

    mutable_inputs = list(inputs)
    mutable_inputs[4] = updated_cache
    for _ in range(warmups):
        output, updated_cache = invoke(*mutable_inputs)
        jax.block_until_ready((output, updated_cache))
        mutable_inputs[4] = updated_cache

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        output, updated_cache = invoke(*mutable_inputs)
        jax.block_until_ready((output, updated_cache))
        latencies_ms.append((time.perf_counter() - start) * 1000)
        mutable_inputs[4] = updated_cache

    result = {
        "event": "result",
        "physical_tokens": physical_tokens,
        "valid_tokens": valid_tokens,
        "logical_page_size": page_size_per_kv_packing * kv_packing,
        "page_size_per_kv_packing": page_size_per_kv_packing,
        "kv_packing": kv_packing,
        "kv_tile_tokens": (page_size_per_kv_packing * kv_packing * bkv),
        "bkv": bkv,
        "bq": bq,
        "q_split": q_split,
        "local_num_heads": local_num_heads,
        "lkv_dim": lkv_dim,
        "r_dim": r_dim,
        "compile_and_first_run_ms": compile_and_first_run_ms,
        "latencies_ms": latencies_ms,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "output_shape": list(output.shape),
    }
    del inputs, mutable_inputs, output, updated_cache, invoke, mesh
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("4096:4000"),
    )
    parser.add_argument(
        "--candidates",
        type=parse_candidates,
        default=parse_candidates(
            "1x64,1x128,1x256,2x64,2x128,2x256,4x64,4x128,4x256"),
    )
    parser.add_argument("--q-splits",
                        type=parse_positive_ints,
                        default=parse_positive_ints("1,2,4"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--total-pages", type=int, default=493)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--pages-per-seq", type=int, default=128)
    parser.add_argument("--page-size-per-kv-packing", type=int, default=128)
    parser.add_argument("--kv-packing", type=int, default=2)
    parser.add_argument("--local-num-heads", type=int, default=12)
    parser.add_argument("--lkv-dim", type=int, default=512)
    parser.add_argument("--r-dim", type=int, default=64)
    parser.add_argument("--vmem-mib", type=int, default=60)
    parser.add_argument("--expected-device-count", type=int, default=8)
    parser.add_argument("--expected-device-kind", default="TPU7x")
    parser.add_argument("--expected-jax", default="0.10.2")
    parser.add_argument("--expected-jaxlib", default="0.10.2")
    parser.add_argument("--expected-libtpu", default="0.0.44.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations <= 0:
        parser.error("iterations must be positive; warmups must be nonnegative")
    invalid_splits = [
        split for split in args.q_splits
        if args.local_num_heads % split != 0
    ]
    if invalid_splits:
        parser.error("q-splits must divide local-num-heads; invalid: "
                     f"{invalid_splits}")
    versions = {
        "jax": package_version("jax"),
        "jaxlib": package_version("jaxlib"),
        "libtpu": package_version("libtpu"),
    }
    expected_versions = {
        "jax": args.expected_jax,
        "jaxlib": args.expected_jaxlib,
        "libtpu": args.expected_libtpu,
    }
    if versions != expected_versions:
        parser.error(
            f"runtime mismatch: observed={versions}, expected={expected_versions}"
        )
    devices = jax.devices()
    if len(devices) != args.expected_device_count:
        parser.error("device-count mismatch: "
                     f"observed={len(devices)}, expected={args.expected_device_count}")
    mismatched_devices = [
        device.device_kind for device in devices
        if args.expected_device_kind.lower() not in device.device_kind.lower()
    ]
    if mismatched_devices:
        parser.error("device-kind mismatch: "
                     f"observed={mismatched_devices}, expected={args.expected_device_kind}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    with args.output.open("a", encoding="utf-8") as output_file:
        emit(
            {
                "event": "manifest",
                "repo_root": str(REPO_ROOT),
                "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
                "kernel_sha256": os.environ.get("KERNEL_SHA256", "unknown"),
                "jax": versions["jax"],
                "jaxlib": versions["jaxlib"],
                "libtpu": versions["libtpu"],
                "devices": [str(device) for device in devices],
                "device_kinds":
                [device.device_kind for device in devices],
                "shapes": args.shapes,
                "candidates": args.candidates,
                "q_splits": args.q_splits,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "total_pages": args.total_pages,
                "max_num_seqs": args.max_num_seqs,
                "pages_per_seq": args.pages_per_seq,
                "page_size_per_kv_packing": args.page_size_per_kv_packing,
                "kv_packing": args.kv_packing,
                "local_num_heads": args.local_num_heads,
                "lkv_dim": args.lkv_dim,
                "r_dim": args.r_dim,
                "cache_row_dim": args.lkv_dim + math.ceil(args.r_dim / 128) * 128,
                "vmem_mib": args.vmem_mib,
                "geometry_source": os.environ.get("SERVING_GEOMETRY_SOURCE",
                                                  "unknown"),
            },
            output_file,
        )

        for physical_tokens, valid_tokens in args.shapes:
            for q_split in args.q_splits:
                for bkv, bq in args.candidates:
                    started = time.time()
                    try:
                        result = tune_candidate(
                            physical_tokens,
                            valid_tokens,
                            bkv=bkv,
                            bq=bq,
                            q_split=q_split,
                            warmups=args.warmups,
                            iterations=args.iterations,
                            total_pages=args.total_pages,
                            max_num_seqs=args.max_num_seqs,
                            pages_per_seq=args.pages_per_seq,
                            page_size_per_kv_packing=args.page_size_per_kv_packing,
                            kv_packing=args.kv_packing,
                            vmem_limit_bytes=args.vmem_mib * 1024 * 1024,
                            local_num_heads=args.local_num_heads,
                            lkv_dim=args.lkv_dim,
                            r_dim=args.r_dim,
                        )
                    except Exception as error:  # Compile/OOM candidates are rejected.
                        result = {
                            "event": "error",
                            "physical_tokens": physical_tokens,
                            "valid_tokens": valid_tokens,
                            "bkv": bkv,
                            "bq": bq,
                            "q_split": q_split,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    result["wall_seconds"] = time.time() - started
                    emit(result, output_file)

        emit({"event": "complete"}, output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
