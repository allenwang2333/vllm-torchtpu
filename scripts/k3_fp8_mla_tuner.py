#!/usr/bin/env python3
"""Isolated MLA decode/verify tuning; no serving defaults are modified.

Uses the same MLA v2 entrypoint as mla_mixed_prefill_tuner.py, but generates
multi-request verify inputs, padded metadata, and nontrivial cached prefixes.
"""
import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path


@contextlib.contextmanager
def trace_kernel(directory, backend):
    if backend == "libtpu":
        from k3_libtpu_profiler import trace

        with trace(directory):
            yield
    elif backend == "torch_tpu":
        from torch_tpu._internal.profiler import _impl

        _impl.start_trace(directory, worker_rank="rank0")
        try:
            yield
        finally:
            _impl.stop_trace()
    elif backend == "jax":
        import jax

        with jax.profiler.trace(directory):
            yield
    else:
        raise ValueError(f"Unsupported profiler backend: {backend}")


def child(case, tile):
    from dataclasses import asdict

    # Isolate the real JAX kernel from the serving plugin's Torch dependencies.
    # Neither kernel code nor numerical operations are mocked.
    if os.getenv("K3_PURE_JAX_TUNER") == "1":
        import logging
        import sys
        import types

        package = types.ModuleType("vllm_torchtpu")
        package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "src/vllm_torchtpu")
        ]
        sys.modules[package.__name__] = package
        log = types.ModuleType("vllm_torchtpu.logger")
        log.init_logger = logging.getLogger
        sys.modules[log.__name__] = log
        import vllm_torchtpu.env_override  # noqa: F401

    import jax
    import jax.numpy as jnp
    import numpy as np

    if int(os.getenv("FALCON_WORLD_SIZE", "1")) > 1:
        jax.distributed.initialize(
            coordinator_address=os.environ["FALCON_JAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["FALCON_WORLD_SIZE"]),
            process_id=int(
                os.getenv("FALCON_JAX_PROCESS_ID",
                          os.getenv("FALCON_RANK", "0"))))
    from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention
    from vllm_torchtpu.kernels.mla.v2.tuned_params import (
        TunableParams, TuningKey, get_tuned_params, tuned_params_mapping)
    assert jax.device_count() == int(os.getenv("FALCON_WORLD_SIZE",
                                               "1")) * 8, jax.devices()
    assert jax.local_device_count() == 8
    print("SINGLE_HOST_TUNE_READY", jax.device_count(), flush=True)

    def host_output(value):
        return np.asarray(value, np.float32)

    c, window, tokens = case
    bq, pages, split = tile
    dtype_name = os.getenv("TUNER_KV_DTYPE", "bfloat16")
    kv_dtype = {"bfloat16": jnp.bfloat16, "fp8": jnp.float8_e4m3fn}[dtype_name]
    packing = 2 if dtype_name == "bfloat16" else 4
    page_size = int(os.getenv("TUNER_PAGE_SIZE", "256"))
    max_seqs = int(os.getenv("TUNER_MAX_SEQS", "32"))
    pages_per_seq = 32768 // page_size
    cast_probs = os.getenv("TUNER_CAST_PROBS", "1") == "1"
    assert dtype_name == "fp8" and page_size == 16 and max_seqs == 256
    assert c <= max_seqs and page_size % packing == 0
    rows = []
    for context in (128, 512, 1000, 4096):
        rng = np.random.default_rng(980406)

        def random(shape, dtype=jnp.bfloat16):
            return jnp.asarray(rng.normal(0, 0.2, shape), dtype)

        q = random((3, tokens, 512))
        qr = random((tokens, 3, 64))
        new = random((tokens, 512), kv_dtype)
        new_r = random((tokens, 64), kv_dtype)
        # Disjoint, low, non-null pages per request; KV lengths differ so
        # verification covers page boundaries and ragged cached prefixes.
        lengths = [context - i % 7 for i in range(c)]
        stride = (context + window + page_size - 1) // page_size
        cache = random((1 + c * stride, page_size // packing, packing, 640),
                       kv_dtype)
        table = np.full((max_seqs, pages_per_seq), -1, np.int32)
        for i in range(c):
            table[i, :stride] = np.arange(1 + i * stride, 1 + (i + 1) * stride)
        cu = jnp.asarray([i * window for i in range(c + 1)] + [c * window] *
                         (max_seqs - c), jnp.int32)
        lens = jnp.asarray(lengths + [0] * (max_seqs - c), jnp.int32)
        ids = jnp.asarray(table.flatten())
        distribution = jnp.asarray([c, c, c] if window == 1 else [0, 0, c],
                                   jnp.int32)

        # FP32 dense causal reference from the exact quantized inputs. Updating
        # only the logical tail also gives an exact reference for KV writes.
        expected_cache = np.asarray(cache, np.float32).copy()
        flat_cache = expected_cache.reshape(-1, page_size, 640)
        qh, qrh = np.asarray(q, np.float32), np.asarray(qr, np.float32)
        nh, nrh = np.asarray(new, np.float32), np.asarray(new_r, np.float32)
        expected = np.zeros((3, c * window, 512), np.float32)
        for i, length in enumerate(lengths):
            start = i * window
            for j in range(window):
                pos = length - window + j
                page = table[i, pos // page_size]
                flat_cache[page, pos % page_size, :512] = nh[start + j]
                flat_cache[page, pos % page_size, 512:576] = nrh[start + j]
            kv = flat_cache[table[i, :stride]].reshape(-1, 640)[:length]
            for j in range(window):
                end = length - window + j + 1
                scores = (qh[:, start + j] @ kv[:end, :512].T +
                          qrh[start + j] @ kv[:end, 512:576].T)
                prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
                prob /= prob.sum(axis=-1, keepdims=True)
                expected[:, start + j] = prob @ kv[:end, :512]

        key = TuningKey(case="batched_decode" if window == 1 else "mixed",
                        max_num_tokens=tokens,
                        actual_num_q_heads=3,
                        actual_lkv_dim=512,
                        actual_r_dim=64,
                        q_dtype="bfloat16",
                        kv_dtype="float8_e4m3fn",
                        page_size_per_kv_packing=4,
                        kv_packing=4,
                        max_num_seqs=max_seqs,
                        pages_per_seq=pages_per_seq)
        params = TunableParams(num_queries_per_block=bq,
                               num_kv_pages_per_block=pages,
                               decode_batch_size=split if window == 1 else 1,
                               q_split=split if window == 6 else 1,
                               vmem_limit_bytes=60 * 1024 * 1024)
        tuned_params_mapping[key] = params
        selected = get_tuned_params(key)
        assert selected == params
        bq, pages = selected.num_queries_per_block, selected.num_kv_pages_per_block
        split = selected.decode_batch_size if window == 1 else selected.q_split
        print("TUNING_KEY_HIT",
              json.dumps(dict(key=asdict(key), params=asdict(selected))),
              flush=True)

        def local_kernel(q, qr, new, new_r, kv, lens, ids, cu, distribution):
            return mla_ragged_paged_attention(
                ql_nope=q,
                q_pe=qr,
                new_kv_c=new,
                new_k_pe=new_r,
                cache_kv=kv,
                kv_lens=lens,
                page_indices=ids,
                cu_q_lens=cu,
                distribution=distribution,
                num_kv_pages_per_block=(pages if window == 1 else 3, 1, pages),
                num_queries_per_block=(1, 16, bq),
                mixed_q_split=split if window != 1 else 1,
                decode_batch_size=split if window == 1 else 1,
                vmem_limit_bytes=60 * 1024 * 1024,
                s_dtype=jnp.bfloat16,
                p_same_dtype_as_v=cast_probs)

        def invoke(kv):
            return local_kernel(q, qr, new, new_r, kv, lens, ids, cu,
                                distribution)

        print("KERNEL_FIRST_CALL", context, case, tile, flush=True)
        started = time.perf_counter()
        output, cache = invoke(cache)
        jax.block_until_ready((output, cache))
        compile_s = time.perf_counter() - started
        actual = host_output(output)[:, :c * window]
        error = float(np.max(np.abs(actual - expected)))
        # Ignore unused aligned RoPE lanes, but check all valid KV channels,
        # including prefixes, other requests and the untouched null page.
        cache_ok = np.array_equal(
            np.asarray(cache, np.float32)[..., :576],
            expected_cache[..., :576])
        # Only unused lanes in the final KV packed word may differ.
        # Null pages, valid KV, and every other unused position stay exact.
        observed = np.asarray(cache, np.float32).reshape(-1, page_size, 640)
        strict_positions = np.ones(observed.shape[:2], dtype=bool)
        for i, length in enumerate(lengths):
            for pos in range(length,
                             ((length + packing - 1) // packing) * packing):
                strict_positions[table[i, pos // page_size],
                                 pos % page_size] = False
        cache_valid = np.array_equal(observed[..., :576][strict_positions],
                                     flat_cache[..., :576][strict_positions])
        if not cache_ok:
            observed = np.asarray(cache,
                                  np.float32).reshape(-1, page_size, 640)
            diff = np.argwhere(observed[..., :576] != flat_cache[..., :576])
            counts = {}
            examples = []
            for page, offset, channel in diff:
                seq = (int(page) - 1) // stride
                pos = ((int(page) - 1) % stride) * page_size + int(offset)
                if page == 0:
                    kind = "null_page"
                elif pos < lengths[seq] - window:
                    kind = "valid_prefix"
                elif pos < lengths[seq]:
                    kind = "new_valid_kv"
                else:
                    kind = "unused_tail"
                counts[kind] = counts.get(kind, 0) + 1
                if len(examples) < 24:
                    examples.append(
                        dict(page=int(page),
                             offset=int(offset),
                             seq=seq,
                             pos=pos,
                             channel=int(channel),
                             kind=kind,
                             expected=float(flat_cache[page, offset, channel]),
                             actual=float(observed[page, offset, channel])))
            print("KV_DIAGNOSTIC",
                  json.dumps(
                      dict(context=context,
                           case=case,
                           tile=tile,
                           counts=counts,
                           differing_positions=np.unique(diff[:, :2],
                                                         axis=0).tolist(),
                           examples=examples)),
                  flush=True)
        # Read the written window as PREFIX in the next verify, instead
        # of overwriting it again. Compare against the canonical cache.
        def readback(kv, next_lens):
            return local_kernel(q, qr, new, new_r, kv, next_lens, ids, cu,
                                distribution)

        readback_errors = {}
        for advance in range(1, window + 1):
            # Keep accepted prefix entries while replacing rejected suffixes.
            next_lens = jnp.asarray([n + advance
                                     for n in lengths] + [0] * (max_seqs - c),
                                    jnp.int32)
            golden_cache = jnp.asarray(expected_cache, kv_dtype)
            # The kernel donates the probe, never the later timing buffer.
            probe_cache = cache.copy()
            got_next, _ = readback(probe_cache, next_lens)
            want_next, _ = readback(golden_cache, next_lens)
            jax.block_until_ready((got_next, want_next))
            readback_errors[advance] = float(
                np.max(
                    np.abs(
                        host_output(got_next)[:, :c * window] -
                        host_output(want_next)[:, :c * window])))
        next_error = float(np.max(list(readback_errors.values())))
        print("READBACK_DIAGNOSTIC",
              json.dumps(
                  dict(context=context,
                       case=case,
                       tile=tile,
                       original_output_error=error,
                       cache_exact=cache_ok,
                       next_output_max_abs_error=next_error)),
              flush=True)
        correct = bool(
            np.isfinite(actual).all() and cache_valid and next_error == 0.0
            and np.allclose(actual, expected, atol=0.005, rtol=0.03))
        if not correct:
            raise RuntimeError(
                f"reference mismatch: error={error} KV={cache_valid} next_error={next_error}"
            )
        if cache.is_deleted():
            raise RuntimeError("Readback probe invalidated the timing cache")
        for _ in range(3):
            output, cache = invoke(cache)
            jax.block_until_ready((output, cache))
        samples = []
        for _ in range(20):
            started = time.perf_counter_ns()
            output, cache = invoke(cache)
            jax.block_until_ready((output, cache))
            samples.append((time.perf_counter_ns() - started) / 1e6)
        trace_dir = os.getenv("TUNER_TRACE_DIR")
        if trace_dir and context == 1000:
            trace_backend = os.getenv("TUNER_PROFILE_BACKEND", "libtpu")
            # A run-local marker lets an ongoing sweep preserve correctness
            # results when the independently versioned profiler ABI is broken.
            # Deferred captures are NOT evidence that a candidate is ready.
            defer_file = Path("/tmp/tpu_logs/k3-defer-jax-profile")
            if trace_backend == "jax" and defer_file.exists():
                destination = Path(trace_dir)
                destination.mkdir(parents=True, exist_ok=True)
                deferred = dict(status="deferred",
                                reason=defer_file.read_text(),
                                case=case,
                                tile=tile)
                (destination / "profile-deferred.json").write_text(
                    json.dumps(deferred, indent=2))
                print("PROFILE_DEFERRED", json.dumps(deferred), flush=True)
            else:
                for repetition in range(3):
                    with trace_kernel(
                            str(Path(trace_dir) / f"rep-{repetition}"),
                            trace_backend):
                        for _ in range(15):
                            output, cache = invoke(cache)
                            jax.block_until_ready((output, cache))
                print("PROFILE_CAPTURED",
                      json.dumps(
                          dict(backend=trace_backend,
                               directory=trace_dir,
                               calls=15,
                               device_events_verified=False)),
                      flush=True)
        rows.append(
            dict(devices_total=int(os.getenv("FALCON_WORLD_SIZE", "1")) * 8,
                 kernel_devices_used=1,
                 emulated_tensor_parallel_size=32,
                 tuning_key=asdict(key),
                 concurrency=c,
                 verify_tokens=window,
                 tokens=tokens,
                 kv_len=context,
                 bq=bq,
                 bkvp=pages,
                 q_split=split,
                 latency_ms=statistics.median(samples),
                 samples_ms=samples,
                 compile_s=compile_s,
                 max_abs_error=error,
                 cache_exact=cache_ok,
                 cache_valid=cache_valid,
                 next_output_max_abs_error=next_error,
                 readback_errors_by_advance=readback_errors,
                 status="success",
                 variant=str(tile),
                 timing="host_dispatch_plus_synchronized_kernel",
                 kv_dtype=dtype_name,
                 p_same_dtype_as_v=cast_probs,
                 page_size=page_size,
                 heads=3,
                 max_num_seqs=max_seqs,
                 pages_per_seq=pages_per_seq))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=json.loads, required=True)
    parser.add_argument("--tile", type=json.loads, required=True)
    args = parser.parse_args()
    rows = child(args.case, args.tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
