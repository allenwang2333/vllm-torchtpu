#!/usr/bin/env python3
"""Isolated MLA decode/verify tuning; no serving defaults are modified.

Uses the same MLA v2 entrypoint as mla_mixed_prefill_tuner.py, but generates
multi-request verify inputs, padded metadata, and nontrivial cached prefixes.
"""
import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def child(case, tile):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention

    c, window, tokens = case
    bq, pages, split = tile
    dtype_name = os.getenv("TUNER_KV_DTYPE", "bfloat16")
    kv_dtype = {"bfloat16": jnp.bfloat16, "fp8": jnp.float8_e4m3fn}[dtype_name]
    packing = 2 if dtype_name == "bfloat16" else 4
    page_size = int(os.getenv("TUNER_PAGE_SIZE", "256"))
    max_seqs = int(os.getenv("TUNER_MAX_SEQS", "32"))
    pages_per_seq = 32768 // page_size
    cast_probs = os.getenv("TUNER_CAST_PROBS", "1") == "1"
    assert c <= max_seqs and page_size % packing == 0
    rows = []
    for context in (128, 512, 1000):
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

        def invoke(kv):
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

        started = time.perf_counter()
        output, cache = invoke(cache)
        jax.block_until_ready((output, cache))
        compile_s = time.perf_counter() - started
        actual = np.asarray(output, np.float32)[:, :c * window]
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
        next_lens = jnp.asarray([n + window
                                 for n in lengths] + [0] * (max_seqs - c),
                                jnp.int32)

        def readback(kv):
            return mla_ragged_paged_attention(
                ql_nope=q,
                q_pe=qr,
                new_kv_c=new,
                new_k_pe=new_r,
                cache_kv=kv,
                kv_lens=next_lens,
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

        golden_cache = jnp.asarray(expected_cache, kv_dtype)
        # The MLA entrypoint donates cache_kv. Keep the original state alive
        # for timing; the readback probe owns a separate device allocation.
        probe_cache = cache.copy()
        got_next, _ = readback(probe_cache)
        want_next, _ = readback(golden_cache)
        jax.block_until_ready((got_next, want_next))
        next_error = float(
            np.max(
                np.abs(
                    np.asarray(got_next, np.float32)[:, :c * window] -
                    np.asarray(want_next, np.float32)[:, :c * window])))
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
            with jax.profiler.trace(trace_dir):
                for _ in range(15):
                    output, cache = invoke(cache)
                    jax.block_until_ready((output, cache))
        rows.append(
            dict(concurrency=c,
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


def candidate_tiles(window):
    if window == 1:
        # Match serving fallback, then vary page tile and decode batch.
        baseline = (1, 3, 4)
        return [baseline] + [(1, pages, batch) for pages in (1, 2, 3, 4, 8, 16)
                             for batch in (1, 2, 4, 8)
                             if (1, pages, batch) != baseline]
    return [(16, 1, 1)] + [
        t for t in itertools.product((8, 16, 32), (1, 2, 4), (1, ))
        if t != (16, 1, 1)
    ] + [(16, 2, 2), (32, 4, 2), (48, 4, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=json.loads)
    parser.add_argument("--tile", type=json.loads)
    args = parser.parse_args()
    if args.case is not None:
        rows = child(args.case, args.tile)
        args.output.write_text(json.dumps(rows))
        return
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    buckets = json.loads(
        os.getenv("TUNER_BUCKETS", "[8,16,32,48,64,96,128,192,256,512]"))
    windows = json.loads(os.getenv("TUNER_WINDOWS", "[6,8]"))
    concurrencies = json.loads(
        os.getenv("TUNER_CONCURRENCIES", "[1,2,4,8,16,32]"))
    all_rows = []
    for window in windows:
        tiles = candidate_tiles(window)
        baseline_tile = tiles[0]
        for c in concurrencies:
            tokens = next(t for t in buckets if t >= c * window)
            for tile in tiles:
                path = root / f"w{window}-c{c}-tile{'-'.join(map(str, tile))}.json"
                cmd = [
                    sys.executable, __file__, "--case",
                    json.dumps([c, window, tokens]), "--tile",
                    json.dumps(tile), "--output",
                    str(path)
                ]
                print("TUNE_START", window, c, tile, flush=True)
                try:
                    result = subprocess.run(cmd, check=False, timeout=600)
                    if result.returncode or not path.exists():
                        raise RuntimeError(f"child exit={result.returncode}")
                    rows = json.loads(path.read_text())
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    rows = [
                        dict(concurrency=c,
                             verify_tokens=window,
                             tokens=tokens,
                             tile=tile,
                             status="error",
                             error=str(error))
                    ]
                with (root / "metrics.jsonl").open("a") as out:
                    for row in rows:
                        out.write(json.dumps(row) + "\n")
                        print("TUNE_RESULT", json.dumps(row), flush=True)
                all_rows.extend(rows)
                if tile == baseline_tile and any(r["status"] != "success"
                                                 for r in rows):
                    raise RuntimeError("Baseline failed; stop before tuning")
    winners = []
    for window in windows:
        tiles = candidate_tiles(window)
        baseline_tile = tiles[0]
        for c in concurrencies:
            valid = [
                r for r in all_rows if r["status"] == "success"
                and r["verify_tokens"] == window and r["concurrency"] == c
            ]
            ranked = []
            for tile in tiles:
                subset = [
                    r for r in valid
                    if (r["bq"], r["bkvp"], r["q_split"]) == tile
                ]
                if len(subset) == 3:
                    ranked.append((statistics.fmean(r["latency_ms"]
                                                    for r in subset), tile))
            ranked.sort()
            if not ranked or not any(t == baseline_tile for _, t in ranked):
                raise RuntimeError(
                    f"missing valid baseline/candidate: {window=} {c=}")
            baseline = next(ms for ms, t in ranked if t == baseline_tile)
            winners.append(
                dict(verify_tokens=window,
                     concurrency=c,
                     baseline_ms=baseline,
                     best_ms=ranked[0][0],
                     best_tile=ranked[0][1],
                     speedup=baseline / ranked[0][0]))
    (root / "summary.json").write_text(json.dumps(winners, indent=2))
    print("TUNE_COMPLETE", json.dumps(winners), flush=True)
    for winner in winners:
        c, window = winner["concurrency"], winner["verify_tokens"]
        tokens = next(t for t in buckets if t >= c * window)
        for variant, tile in (("baseline", candidate_tiles(window)[0]),
                              ("candidate", winner["best_tile"])):
            trace_dir = root.parent / "profiling" / "xprof" / (
                f"w{window}-c{c}-{variant}")
            env = dict(os.environ, TUNER_TRACE_DIR=str(trace_dir))
            subprocess.run([
                sys.executable, __file__, "--case",
                json.dumps([c, window, tokens]), "--tile",
                json.dumps(tile), "--output",
                str(root / f"w{window}-c{c}-{variant}-recheck.json")
            ],
                           env=env,
                           check=True,
                           timeout=600)


if __name__ == "__main__":
    main()
