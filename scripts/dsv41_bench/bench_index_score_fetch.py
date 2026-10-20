# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated A/B of the decode index scorer's page fetch (one serving-shape call).

Arms: the incumbent 2D halfplane scorer and ``proto_index_scores`` with bounds
checks on and off. Every arm is first checked bitwise against the incumbent and
against an f32 reference on a few rows; then timed as the slope between 1 and
``--reps`` data-dependent calls inside one jit, so dispatch cancels.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dsv41_bench"))
import proto_index_scores  # noqa: E402
from bench_decode_overlap import fixture, load_kernels  # noqa: E402

PAGE, TILE, GROUP, Q = 128, 4096, 4, 6


FORMATS = ("fp8_widen", "fp8_mxu", "fp4_widen")


def key_format(name):
    return next((f for f in FORMATS if name.endswith(f)), "bf16")


def arm(grouped, name):
    """``incumbent``, ``integrated``, or ``proto`` plus options: ``_joint``,
    ``_d<depth>``, ``_bc`` (bounds checks on), a key format suffix (``FORMATS``)."""
    if name in ("incumbent", "integrated"):
        # "integrated": the served path with TPU_DEEPSEEK_V41_INDEX_JOINT_FETCH.
        def served(q, w, pool, pages, lengths):
            return grouped(
                q,
                w,
                pool,
                pages,
                lengths,
                page_records=PAGE,
                key_tile=TILE,
                score_window=131072,
                fuse_mask=True,
                halfplane=True,
                request_group=GROUP,
                joint_fetch=name == "integrated",
            )

        return served
    fmt = key_format(name)
    opts = name.removesuffix(fmt).removesuffix("_").split("_")
    if opts[0] != "proto":
        raise ValueError(name)
    depth = next((int(o[1:]) for o in opts if o.startswith("d") and o[1:].isdigit()), 2)

    def run(q, w, pool, pages, lengths):
        return proto_index_scores.overlapped_decode_scores(
            q,
            w,
            pool,
            pages,
            lengths,
            page_records=PAGE,
            key_tile=TILE,
            request_group=GROUP,
            disable_bounds_checks="bc" not in opts,
            joint="joint" in opts,
            depth=depth,
            key_format=fmt,
        )

    return run


def narrow_pool(pool, key_format, seed):
    """Random finite keys in the narrow format, same records as ``pool``, two banks."""
    per_row = proto_index_scores.RECORDS_PER_ROW[key_format]
    rows = sum(b.shape[0] for b in pool) * 2 // per_row
    bits = np.random.default_rng(seed).integers(0, 2**31, (rows, 128), dtype=np.int64)
    if key_format.startswith("fp8"):
        bits &= 0x37373737  # e4m3 exponent <= 6 per byte: finite, no NaN
    words = jnp.asarray(bits.astype(np.uint32).view(np.int32))
    page_rows = PAGE // per_row
    split = (rows // page_rows // 2) * page_rows
    return (words[:split], words[split:])


def reference_rows(q, w, pool, pages, lengths, rows):
    """f32 scores of selected flat rows from halfplane words (record order)."""
    banks = pool if isinstance(pool, tuple) else (pool,)
    words = np.concatenate([np.asarray(b) for b in banks]).view(np.uint32)
    pages, lengths = np.asarray(pages), np.asarray(lengths).reshape(-1)
    q = np.asarray(q.astype(jnp.float32)).reshape(-1, 32, 128)
    w = np.asarray(w).reshape(-1, 32, 1)
    width = pages.shape[1] * PAGE
    out = {}
    for t in rows:
        r = t // Q
        records = (np.maximum(pages[r], 0)[:, None] + np.arange(PAGE)).reshape(-1)
        bits = (words[records // 2] >> (16 * (records % 2))[:, None]) & 0xFFFF
        keys = (bits.astype(np.uint32) << 16).view(np.float32)
        s = (np.maximum(q[t] @ keys.T, 0) * w[t]).sum(0)
        valid = (np.arange(width) < lengths[t]) & np.repeat(pages[r] >= 0, PAGE)
        out[t] = np.where(valid, s, -np.inf)
    return out


def timed(fn, inputs, reps, samples):
    q, rest = inputs[0], inputs[1:]

    def chain(n):
        def run(q):
            out = fn(q, *rest)
            for _ in range(n - 1):
                # Data-dependent and never true: forces serial calls.
                link = (out[0, 0] == -987654.0).astype(q.dtype)
                out = fn(q + link, *rest)
            return out

        return jax.jit(run)

    result = {}
    for n in (1, reps):
        f = chain(n)
        jax.block_until_ready(f(q))
        s = []
        for _ in range(samples):
            t0 = time.perf_counter()
            jax.block_until_ready(f(q))
            s.append((time.perf_counter() - t0) * 1e6)
        result[n] = s
    per_call = (np.median(result[reps]) - np.median(result[1])) / (reps - 1)
    return float(per_call), result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=65536)
    p.add_argument("--length", type=int, default=49408)
    p.add_argument("--requests", type=int, default=12)
    p.add_argument("--contiguous", action="store_true", help="unshuffled page tables")
    p.add_argument("--arms", nargs="+", default=["incumbent", "proto_joint_d3"])
    p.add_argument("--reps", type=int, default=11)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--seed", type=int, default=517)
    a = p.parse_args()
    _, _, _, grouped = load_kernels(ROOT)
    inputs, _, _ = fixture(
        a.requests, Q, a.width, a.length, PAGE, not a.contiguous, "random", 2, a.seed
    )
    q, w, pool, records, lengths = inputs
    pages = records[:, ::PAGE]
    inputs = (q, w, pool, pages, lengths)
    rows = [0, 7, a.requests * Q - 2, a.requests * Q - 1]
    ref = reference_rows(*inputs, rows)
    base = np.asarray(jax.jit(arm(grouped, "incumbent"))(*inputs))
    report = dict(
        width=a.width,
        length=a.length,
        requests=a.requests,
        contiguous=a.contiguous,
        libtpu_args=__import__("os").environ.get("LIBTPU_INIT_ARGS", ""),
    )
    for name in a.arms:
        fn = arm(grouped, name)
        if key_format(name) != "bf16":
            # Narrow formats score different keys: time only.
            narrow_inputs = (
                q,
                w,
                narrow_pool(pool, key_format(name), a.seed),
                pages,
                lengths,
            )
            jax.block_until_ready(jax.jit(fn)(*narrow_inputs))
            per_call, raw = timed(fn, narrow_inputs, a.reps, a.samples)
            report[name] = dict(us_per_call=per_call, t1_us=raw[1], tN_us=raw[a.reps])
            print(f"{name:10s} {per_call:8.1f} us/call  (timing only)", flush=True)
            continue
        got = np.asarray(jax.jit(fn)(*inputs))
        bitwise = bool(np.array_equal(got, base))
        err = max(
            float(np.max(np.abs(np.where(np.isfinite(ref[t]), got[t] - ref[t], 0))))
            for t in rows
        )
        masks = all(
            np.array_equal(np.isfinite(got[t]), np.isfinite(ref[t])) for t in rows
        )
        per_call, raw = timed(fn, inputs, a.reps, a.samples)
        report[name] = dict(
            us_per_call=per_call,
            bitwise_vs_incumbent=bitwise,
            max_abs_err_vs_f32=err,
            masks_match=masks,
            t1_us=raw[1],
            tN_us=raw[a.reps],
        )
        print(
            f"{name:10s} {per_call:8.1f} us/call  bitwise={bitwise} "
            f"err={err:.2e} masks={masks}",
            flush=True,
        )
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
