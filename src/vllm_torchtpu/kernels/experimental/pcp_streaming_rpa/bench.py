#!/usr/bin/env python3
"""Benchmark main-branch PCP streaming attention at the attention entry."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (  # noqa: E402
    PCP_AXIS_NAME, make_pcp_streaming_rpa_kernel)


CUSTOM_CALL_REGION_TRACE_FLAG = "--xla_enable_custom_call_region_trace=true"
REGISTER_LLO_DEBUG_INFO_FLAG = "--xla_xprof_register_llo_debug_info=true"
JF_LSRA_ANNOTATE_FLAG = "--xla_jf_lsra_v2_annotate"
DEFAULT_TC_COUNTER_INDICES = (
    1, 3, 4, 10, 11, 31, 32, 33, 34, 35, 37, 38, 56, 57, 58, 73, 74, 75,
    105)


def _text_config(config: dict) -> dict:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _load_attention_config(model_path: Path) -> tuple[int, int, int]:
    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        config = _text_config(json.load(f))
    return (int(config["num_attention_heads"]),
            int(config["num_key_value_heads"]), int(config["head_dim"]))


def _append_libtpu_init_args(*new_flags: str,
                             replace_prefixes: tuple[str, ...] = ()) -> None:
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    flags = [
        flag for flag in existing.split()
        if not any(flag.startswith(prefix) for prefix in replace_prefixes)
    ]
    for flag in new_flags:
        if flag not in flags:
            flags.append(flag)
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)


def _enable_profile_libtpu_init_args(*, custom_call_region_trace: bool,
                                     register_llo_debug_info: bool) -> None:
    flags = []
    if custom_call_region_trace:
        flags.append(CUSTOM_CALL_REGION_TRACE_FLAG)
    if register_llo_debug_info:
        flags.append(REGISTER_LLO_DEBUG_INFO_FLAG)
    _append_libtpu_init_args(*flags)


def _enable_jf_dump_libtpu_init_args(dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    _append_libtpu_init_args(
        f"--xla_jf_dump_to={dump_dir}",
        JF_LSRA_ANNOTATE_FLAG,
        replace_prefixes=("--xla_jf_dump_to=", ),
    )


def _make_llo_profile_options(*, interval_us: int,
                              tensor_cores_to_trace: int,
                              duration_ms: int,
                              counter_indices: tuple[int, ...],
                              ) -> jax.profiler.ProfileOptions:
    options = jax.profiler.ProfileOptions()
    if duration_ms > 0:
        options.duration_ms = duration_ms
    counter_options = (
        f"interval_us:{interval_us} scaling:0 counter_size_bits:1 " +
        " ".join(f"indices:{index}" for index in counter_indices))
    options.advanced_configuration = {
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": counter_options,
        "num_tensor_cores_to_trace_per_device": tensor_cores_to_trace,
    }
    return options


def _parse_counter_indices(indices: str) -> tuple[int, ...]:
    parsed = tuple(int(index.strip()) for index in indices.split(",")
                   if index.strip())
    if not parsed:
        raise ValueError("--profile-counter-indices must not be empty.")
    if any(index < 0 for index in parsed):
        raise ValueError("--profile-counter-indices must be non-negative.")
    return parsed


def _dtype_label(dtype: jnp.dtype) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.dtype(jnp.bfloat16):
        return "bf16"
    if dtype == jnp.dtype(jnp.float8_e4m3fn):
        return "fp8_e4m3fn"
    if dtype == jnp.dtype(jnp.float8_e5m2):
        return "fp8_e5m2"
    return dtype.name


def _resolve_kv_dtype(dtype_name: str) -> jnp.dtype:
    normalized = dtype_name.lower().strip()
    if normalized in ("bf16", "bfloat16"):
        return jnp.dtype(jnp.bfloat16)
    if normalized in ("fp8", "fp8_e4m3", "fp8_e4m3fn",
                      "float8_e4m3fn"):
        return jnp.dtype(jnp.float8_e4m3fn)
    if normalized in ("fp8_e5m2", "float8_e5m2"):
        return jnp.dtype(jnp.float8_e5m2)
    raise ValueError(f"Unsupported KV cache dtype: {dtype_name}")


def _is_fp8_dtype(dtype: jnp.dtype) -> bool:
    dtype = jnp.dtype(dtype)
    return dtype in (
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    )


def _dtype_packing(dtype: jnp.dtype, packing_bits: int = 32) -> int:
    bits = jnp.dtype(dtype).itemsize * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"The bit width must divide {packing_bits}, got {bits}.")
    return packing_bits // bits


def _kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                    head_dim: int,
                    kv_dtype: jnp.dtype) -> tuple[int, int, int, int, int]:
    padded_head_dim = ((head_dim + 127) // 128) * 128
    kv_packing = _dtype_packing(kv_dtype)
    packed_heads_x2 = ((num_kv_heads * 2 + kv_packing - 1) //
                       kv_packing * kv_packing)
    return (num_blocks, block_size, packed_heads_x2 // kv_packing,
            kv_packing, padded_head_dim)


def _make_kv_input(shape: tuple[int, int, int], *, value: float,
                   q_dtype: jnp.dtype, kv_dtype: jnp.dtype,
                   scale: float | None) -> jax.Array:
    tensor = jnp.ones(shape, dtype=q_dtype) * value
    if _is_fp8_dtype(kv_dtype):
        if scale is None or scale == 0.0:
            raise ValueError("fp8 K/V inputs require non-zero scales.")
        tensor = tensor / scale
    return tensor.astype(kv_dtype)


def _make_inputs(*, mesh: Mesh, num_heads: int, num_kv_heads: int,
                 head_dim: int, pcp_size: int, chunk_size: int, kv_len: int,
                 page_size: int, q_dtype: jnp.dtype, kv_dtype: jnp.dtype,
                 k_scale: float | None,
                 v_scale: float | None) -> tuple[jax.Array, ...]:
    if chunk_size % pcp_size != 0:
        raise ValueError("chunk_size must be divisible by pcp_size.")
    if kv_len % (page_size * pcp_size) != 0:
        raise ValueError("kv_len must be divisible by page_size * pcp_size.")

    local_kv_cache_blocks = kv_len // (page_size * pcp_size)
    global_kv_cache_blocks = local_kv_cache_blocks * pcp_size
    kv_cache_shape = _kv_cache_shape(global_kv_cache_blocks, page_size,
                                     num_kv_heads, head_dim, kv_dtype)

    q_sharding = NamedSharding(mesh, P(PCP_AXIS_NAME, None, None))
    kv_cache_sharding = NamedSharding(
        mesh, P(PCP_AXIS_NAME, None, None, None, None))
    replicated = NamedSharding(mesh, P())

    q_host = jnp.ones((chunk_size, num_heads, head_dim), dtype=q_dtype) * 0.01
    kv_shape = (chunk_size, num_kv_heads, head_dim)
    k_host = _make_kv_input(kv_shape,
                            value=0.02,
                            q_dtype=q_dtype,
                            kv_dtype=kv_dtype,
                            scale=k_scale)
    v_host = _make_kv_input(kv_shape,
                            value=0.03,
                            q_dtype=q_dtype,
                            kv_dtype=kv_dtype,
                            scale=v_scale)
    kv_cache_host = jnp.zeros(kv_cache_shape, dtype=kv_dtype)

    q = jax.device_put(q_host, q_sharding)
    k = jax.device_put(k_host, q_sharding)
    v = jax.device_put(v_host, q_sharding)
    kv_cache = jax.device_put(kv_cache_host, kv_cache_sharding)

    block_tables = np.arange(local_kv_cache_blocks,
                             dtype=np.int32).reshape(1, local_kv_cache_blocks)
    seq_lens = jax.device_put(jnp.asarray([kv_len], dtype=jnp.int32),
                              replicated)
    page_indices = jax.device_put(jnp.asarray(block_tables.reshape(-1)),
                                  replicated)
    query_start_loc = jax.device_put(
        jnp.asarray([0, chunk_size], dtype=jnp.int32), replicated)
    request_distribution = jax.device_put(
        jnp.asarray([0, 0, 1], dtype=jnp.int32), replicated)

    return (kv_cache, q, k, v, seq_lens, page_indices, query_start_loc,
            request_distribution)


def _build_round_fn(*, mesh: Mesh, scale: float, page_size: int, kv_len: int,
                    iters: int, k_scale: float | None,
                    v_scale: float | None, q_block_size: int,
                    q_compute_size: int | None,
                    split_history_current: bool):
    entry_kwargs = dict(
        q_scale=None,
        k_scale=k_scale,
        v_scale=v_scale,
        mesh=mesh,
        sliding_window=None,
        sm_scale=scale,
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=page_size,
        max_model_len=kv_len,
    )
    if "q_block_size" in inspect.signature(
            make_pcp_streaming_rpa_kernel).parameters:
        entry_kwargs["q_block_size"] = q_block_size
    if "q_compute_size" in inspect.signature(
            make_pcp_streaming_rpa_kernel).parameters:
        entry_kwargs["q_compute_size"] = q_compute_size
    if "split_history_current" in inspect.signature(
            make_pcp_streaming_rpa_kernel).parameters:
        entry_kwargs["split_history_current"] = split_history_current
    pcp_entry = make_pcp_streaming_rpa_kernel(**entry_kwargs)

    def run_round(kv_cache, q, k, v, seq_lens, page_indices, query_start_loc,
                  request_distribution):

        def body(_i, carry):
            cache, checksum = carry
            cache, out = pcp_entry(cache, q, k, v, seq_lens, page_indices,
                                   query_start_loc, request_distribution)
            return cache, checksum + out[0, 0, 0].astype(jnp.float32)

        return lax.fori_loop(
            0,
            iters,
            body,
            (kv_cache, jnp.asarray(0.0, dtype=jnp.float32)),
        )

    return jax.jit(run_round, donate_argnums=(0, ))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark main-branch PCP streaming attention entry.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pcp-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--context-start-k", type=int, default=124)
    parser.add_argument("--context-end-k", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--q-block-size", type=int, default=512)
    parser.add_argument(
        "--q-compute-size",
        type=int,
        default=None,
        help=("Q rows processed by each softmax compute step. Defaults to "
              "--q-block-size."),
    )
    parser.add_argument("--split-history-current",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=("Use the experimental split PCP path: history "
                              "no-causal Pallas call plus current causal call."))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="bf16",
        help=("Dtype for K/V inputs and the KV cache. Q stays bf16. "
              "Supported values: bf16, fp8/fp8_e4m3fn, fp8_e5m2."),
    )
    parser.add_argument(
        "--k-scale",
        type=float,
        default=1.0,
        help=("Static dequantization scale passed to the PCP kernel for fp8 "
              "K. Ignored for bf16 KV cache."),
    )
    parser.add_argument(
        "--v-scale",
        type=float,
        default=1.0,
        help=("Static dequantization scale passed to the PCP kernel for fp8 "
              "V. Ignored for bf16 KV cache."),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help=("Optional directory for a JAX profiler trace. When set, only "
              "round 2, after warmup/compile, is profiled. The trace enables "
              "LLO debug info and TPU TC counters including MXU busy."),
    )
    parser.add_argument(
        "--profile-counter-interval-us",
        type=int,
        default=1000,
        help=("TPU periodic counter sampling interval in microseconds. The "
              "LLO profiling PDF uses 1us; the default is 1000us because "
              "1us/10us can make long-context traces very slow to stop/flush."),
    )
    parser.add_argument(
        "--profile-tensor-cores-to-trace",
        type=int,
        default=1,
        help="Number of tensor cores to trace per device for LLO profiling.",
    )
    parser.add_argument(
        "--profile-counter-indices",
        type=str,
        default=",".join(str(index) for index in DEFAULT_TC_COUNTER_INDICES),
        help=("Comma-separated TC counter indices to sample. For TPU v7x, "
              "56,57,58 are count_mxu_busy_{0,1,2} and 105 is "
              "count_sync_wait."),
    )
    parser.add_argument(
        "--profile-custom-call-region-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Enable --xla_enable_custom_call_region_trace=true when "
              "--profile-dir is set."),
    )
    parser.add_argument(
        "--profile-register-llo-debug-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Enable --xla_xprof_register_llo_debug_info=true when "
              "--profile-dir is set. Disable this if LLO debug metadata makes "
              "the kernel exceed VMEM during compilation."),
    )
    parser.add_argument(
        "--profile-duration-ms",
        type=int,
        default=50,
        help=("Profiler collection duration in milliseconds. 0 means collect "
              "until round 2 finishes; full 10-iteration LLO counter traces "
              "can be very slow to stop/flush."),
    )
    parser.add_argument(
        "--jf-dump-dir",
        type=Path,
        default=None,
        help=("Optional XLA JF dump directory. When set, LIBTPU_INIT_ARGS gets "
              "--xla_jf_dump_to=<dir> and --xla_jf_lsra_v2_annotate."),
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()
    if args.profile_counter_interval_us <= 0:
        raise ValueError("--profile-counter-interval-us must be positive.")
    if args.profile_tensor_cores_to_trace <= 0:
        raise ValueError("--profile-tensor-cores-to-trace must be positive.")
    if args.profile_duration_ms < 0:
        raise ValueError("--profile-duration-ms must be non-negative.")
    profile_counter_indices = _parse_counter_indices(
        args.profile_counter_indices)
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")
    if args.q_block_size <= 0:
        raise ValueError("--q-block-size must be positive.")
    if args.q_compute_size is not None:
        if args.q_compute_size <= 0:
            raise ValueError("--q-compute-size must be positive.")
        if args.q_compute_size > args.q_block_size:
            raise ValueError("--q-compute-size must be <= --q-block-size.")
        if args.q_block_size % args.q_compute_size != 0:
            raise ValueError("--q-block-size must be divisible by "
                             "--q-compute-size.")
    if args.profile_dir is not None:
        _enable_profile_libtpu_init_args(
            custom_call_region_trace=args.profile_custom_call_region_trace,
            register_llo_debug_info=args.profile_register_llo_debug_info,
        )
    if args.jf_dump_dir is not None:
        _enable_jf_dump_libtpu_init_args(args.jf_dump_dir)

    devices = jax.local_devices()
    if len(devices) < args.pcp_size:
        raise RuntimeError(
            f"Need at least {args.pcp_size} JAX devices, got {len(devices)}.")

    num_heads, num_kv_heads, head_dim = _load_attention_config(args.model)
    kv_len = args.context_end_k * 1024
    q_start = args.context_start_k * 1024
    if kv_len - q_start != args.chunk_size:
        raise ValueError("context range must match chunk_size.")

    mesh = Mesh(np.asarray(devices[:args.pcp_size]), (PCP_AXIS_NAME, ))
    q_dtype = jnp.dtype(jnp.bfloat16)
    kv_dtype = _resolve_kv_dtype(args.kv_cache_dtype)
    if _is_fp8_dtype(kv_dtype):
        if args.k_scale == 0.0 or args.v_scale == 0.0:
            raise ValueError("fp8 K/V scales must be non-zero.")
        k_scale = args.k_scale
        v_scale = args.v_scale
    else:
        k_scale = None
        v_scale = None
    scale = 1.0 / math.sqrt(head_dim)
    local_tokens = args.chunk_size // args.pcp_size
    local_kv_blocks = kv_len // (args.page_size * args.pcp_size)

    print("main PCP attention entry benchmark")
    print(f"  commit: {REPO_ROOT}")
    print(f"  model: {args.model}")
    print(f"  devices: {args.pcp_size} / local available {len(devices)}")
    print(f"  heads: q={num_heads}, kv={num_kv_heads}, "
          f"q_per_kv={num_heads // num_kv_heads}, head_dim={head_dim}")
    print(f"  dtype: q={_dtype_label(q_dtype)}, "
          f"k/v={_dtype_label(kv_dtype)}, kv_cache={_dtype_label(kv_dtype)}")
    if _is_fp8_dtype(kv_dtype):
        print(f"  kv scales: k_scale={k_scale}, v_scale={v_scale}")
    print(f"  context: [{q_start}, {kv_len}) tokens, chunk={args.chunk_size}")
    print(f"  page_size/interleave: {args.page_size}")
    print(f"  q_block_size: {args.q_block_size}")
    print(f"  q_compute_size: "
          f"{args.q_compute_size or args.q_block_size}")
    print(f"  split_history_current: {args.split_history_current}")
    print(f"  local_tokens/rank: {local_tokens}")
    print(f"  local_kv_blocks/rank: {local_kv_blocks}")
    print(f"  loop: lax.fori_loop iters={args.iters}")
    if args.jf_dump_dir is not None:
        print(f"  jf_dump_dir: {args.jf_dump_dir}")
        print(f"  LIBTPU_INIT_ARGS: {os.environ['LIBTPU_INIT_ARGS']}")
    if args.profile_dir is not None:
        print(f"  profile_dir: {args.profile_dir}")
        print(f"  LIBTPU_INIT_ARGS: {os.environ.get('LIBTPU_INIT_ARGS', '')}")
        print(f"  profile custom call region trace: "
              f"{args.profile_custom_call_region_trace}")
        print(f"  profile register LLO debug info: "
              f"{args.profile_register_llo_debug_info}")
        print(f"  TC counters: {profile_counter_indices}")
        print(f"  profile counter interval: "
              f"{args.profile_counter_interval_us} us")
        print(f"  tensor cores traced/device: "
              f"{args.profile_tensor_cores_to_trace}")
        print(f"  profile duration: {args.profile_duration_ms} ms "
              "(0 means full round 2)")
        if args.iters > 1:
            print("  warning: multi-iteration LLO counter traces may spend "
                  "minutes in xprof conversion; use --iters 1 for a smaller "
                  "MXU utilization capture.")

    inputs = _make_inputs(
        mesh=mesh,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pcp_size=args.pcp_size,
        chunk_size=args.chunk_size,
        kv_len=kv_len,
        page_size=args.page_size,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    run_round = _build_round_fn(
        mesh=mesh,
        scale=scale,
        page_size=args.page_size,
        kv_len=kv_len,
        iters=args.iters,
        k_scale=k_scale,
        v_scale=v_scale,
        q_block_size=args.q_block_size,
        q_compute_size=args.q_compute_size,
        split_history_current=args.split_history_current,
    )

    print("  round 1: warmup/compile")
    start = time.perf_counter()
    kv_cache, checksum = run_round(*inputs)
    checksum.block_until_ready()
    warmup_s = time.perf_counter() - start
    print(f"    warmup elapsed: {warmup_s * 1000:.3f} ms, "
          f"checksum={float(checksum):.6f}")

    print("  round 2: measured")
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        profile_context = jax.profiler.trace(
            str(args.profile_dir),
            profiler_options=_make_llo_profile_options(
                interval_us=args.profile_counter_interval_us,
                tensor_cores_to_trace=args.profile_tensor_cores_to_trace,
                duration_ms=args.profile_duration_ms,
                counter_indices=profile_counter_indices,
            ),
        )
        print(f"    LLO profiler start: {args.profile_dir}")
    else:
        profile_context = nullcontext()
    elapsed_s = 0.0
    try:
        with profile_context:
            start = time.perf_counter()
            with jax.profiler.TraceAnnotation("pcp_attention_entry_round2"):
                kv_cache, checksum = run_round(kv_cache, *inputs[1:])
                checksum.block_until_ready()
            elapsed_s = time.perf_counter() - start
    finally:
        if args.profile_dir is not None:
            print(f"    LLO profiler stop: {args.profile_dir}")
    print(f"    measured elapsed: {elapsed_s * 1000:.3f} ms")
    print(f"    per iter: {elapsed_s * 1000.0 / args.iters:.3f} ms")
    print(f"    checksum={float(checksum):.6f}")
    kv_cache.block_until_ready()


if __name__ == "__main__":
    main()