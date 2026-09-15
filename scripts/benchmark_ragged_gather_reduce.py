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
"""Microbenchmark the MoE output-combine (SparseCore ragged gather-reduce).

Times the v1/v2/v3 combine kernels at a single shape and checks each against
the XLA reference, so a kernel that is fast because it skips work cannot pass.

Default shape is the Qwen3-Coder-30B-A3B-FP8 prefill combine as measured in
xprof job j-1d51c638: 8192 destination tokens, top-8, hidden 2048, bf16 ->
65536 routed rows in, (8192, 2048) out.

Two properties this harness is deliberate about:

* Both v2 and v3 silently fall back to an XLA implementation when their
  preconditions do not hold (v3 falls back when ``x`` is *small* enough to sit
  in VMEM, which is exactly what a shrunken debug shape would do). Timing the
  fallback while believing you timed the kernel is the failure mode that makes
  a microbenchmark lie, so ``--assert-kernel`` greps the compiled HLO for the
  expected ``sc_ragged_gather_reduce_v*`` custom call and exits non-zero if it
  is absent.
* Inputs are placed on device and blocked on before the timing loop, so the
  measurement is kernel time and not host transfer.

Usage:
  python3 scripts/benchmark_ragged_gather_reduce.py --versions v2,v3
  python3 scripts/benchmark_ragged_gather_reduce.py --num-tokens 8192 --topk 8 \
      --hidden-size 2048 --iters 50
"""

import argparse
import functools
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


def _reference(x, indices, topk_weights, valid_rows_mask, reduce_group_size):
    """XLA reference: gather, weight, mask, group-sum. Accumulates in f32."""
    out = x[indices].astype(jnp.float32) * topk_weights[:, None].astype(
        jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(jnp.bfloat16)


def _load(version):
    if version == "v1":
        from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
            ragged_gather_reduce
        return ragged_gather_reduce, "sc_ragged_gather_reduce"
    if version == "v2":
        from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper \
            import ragged_gather_reduce_v2
        return ragged_gather_reduce_v2, "sc_ragged_gather_reduce_v2"
    if version == "v3":
        from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
            ragged_gather_reduce
        return ragged_gather_reduce, "sc_ragged_gather_reduce_v3"
    raise ValueError(f"unknown version {version!r}")


def _make_inputs(num_tokens, topk, hidden_size, invalid_frac, seed):
    """Build one MoE combine input set.

    ``indices`` is a permutation because the combine reverses the routing
    argsort; ``valid_rows_mask`` drops a fraction of routes the way dropped /
    padded expert slots do in the real layer.
    """
    rng = np.random.default_rng(seed)
    input_size = num_tokens * topk

    x = rng.normal(0.0, 1.0, size=(input_size, hidden_size)).astype(np.float32)
    indices = rng.permutation(input_size).astype(np.int32)

    # Per-token top-k weights that sum to 1, as a softmax router produces.
    w = rng.random((num_tokens, topk)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    topk_weights = w.reshape(-1)

    valid = np.ones(input_size, dtype=bool)
    if invalid_frac > 0.0:
        n_invalid = int(input_size * invalid_frac)
        valid[rng.choice(input_size, size=n_invalid, replace=False)] = False

    return (
        jnp.asarray(x, dtype=jnp.bfloat16),
        jnp.asarray(indices),
        jnp.asarray(topk_weights),
        jnp.asarray(valid),
    )


def _check_kernel_used(fn, args, topk, expected_call):
    """Return (used_kernel, matched_names) from the compiled HLO."""
    try:
        text = jax.jit(functools.partial(
            fn, reduce_group_size=topk)).lower(*args).compile().as_text()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"<could not compile for inspection: {exc}>"
    names = sorted({
        n
        for n in ("sc_ragged_gather_reduce", "sc_ragged_gather_reduce_v2",
                  "sc_ragged_gather_reduce_v3") if n in text
    })
    return (expected_call in text), names


def _time(fn, args, topk, iters, warmup):
    call = jax.jit(functools.partial(fn, reduce_group_size=topk))
    out = jax.block_until_ready(call(*args))
    for _ in range(warmup - 1):
        out = call(*args)
    jax.block_until_ready(out)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(call(*args))
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples, out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-tokens", type=int, default=8192)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--invalid-frac", type=float, default=0.0,
                   help="fraction of routes marked invalid")
    p.add_argument("--versions", default="v2,v3")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--assert-kernel", action="store_true",
                   help="exit non-zero if a version silently used its XLA "
                        "fallback instead of the Pallas kernel")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    args = p.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    input_size = args.num_tokens * args.topk

    print(f"device            : {jax.devices()[0].device_kind} "
          f"x{jax.device_count()}")
    print(f"shape             : num_tokens={args.num_tokens} topk={args.topk} "
          f"hidden={args.hidden_size}")
    print(f"                    x=({input_size}, {args.hidden_size}) bf16  ->  "
          f"out=({args.num_tokens}, {args.hidden_size}) bf16")
    print(f"invalid_frac      : {args.invalid_frac}")
    print(f"iters/warmup      : {args.iters}/{args.warmup}")

    inputs = _make_inputs(args.num_tokens, args.topk, args.hidden_size,
                          args.invalid_frac, args.seed)
    inputs = jax.block_until_ready([jax.device_put(a) for a in inputs])

    x_bytes = input_size * args.hidden_size * 2
    out_bytes = args.num_tokens * args.hidden_size * 2
    ideal_bytes = x_bytes + out_bytes
    print(f"ideal traffic     : read {x_bytes/2**20:.1f} MiB + write "
          f"{out_bytes/2**20:.1f} MiB = {ideal_bytes/2**20:.1f} MiB\n")

    ref = jax.block_until_ready(
        jax.jit(functools.partial(_reference,
                                  reduce_group_size=args.topk))(*inputs))
    ref_np = np.asarray(ref, dtype=np.float32)

    results = {}
    failures = []
    for v in versions:
        try:
            fn, expected_call = _load(v)
        except Exception as exc:
            print(f"[{v}] SKIP: import failed: {exc}")
            continue

        used, names = _check_kernel_used(fn, inputs, args.topk, expected_call)
        samples, out = _time(fn, inputs, args.topk, args.iters, args.warmup)

        out_np = np.asarray(out, dtype=np.float32)
        if out_np.shape != ref_np.shape:
            ok = False
            detail = f"shape {out_np.shape} != ref {ref_np.shape}"
        else:
            denom = np.maximum(np.abs(ref_np), 1e-6)
            rel = np.abs(out_np - ref_np) / denom
            absd = np.abs(out_np - ref_np)
            close = (absd <= args.atol) | (rel <= args.rtol)
            ok = bool(close.all())
            detail = (f"max_abs={absd.max():.4g} "
                      f"max_rel={rel.max():.4g} "
                      f"mismatch={100.0*(~close).mean():.4f}%")

        med = statistics.median(samples)
        results[v] = med
        gib_s = ideal_bytes / (med * 1e-6) / 2**30

        status = "PASS" if ok else "FAIL"
        kernel_note = ("kernel" if used else
                       "XLA FALLBACK (not the kernel!)" if used is False
                       else "unknown")
        print(f"[{v}] median {med:10.1f} us   min {min(samples):10.1f} us   "
              f"p90 {np.percentile(samples, 90):10.1f} us")
        print(f"[{v}] effective {gib_s:7.1f} GiB/s (ideal traffic / median)")
        print(f"[{v}] path      : {kernel_note}  custom_calls={names}")
        print(f"[{v}] vs ref    : {status}  {detail}")
        print()

        if not ok:
            failures.append(f"{v}: correctness {detail}")
        if args.assert_kernel and used is False:
            failures.append(f"{v}: used XLA fallback, not the Pallas kernel")

    if len(results) > 1:
        print("--- relative ---")
        base = versions[0] if versions[0] in results else next(iter(results))
        for v, med in results.items():
            if v == base:
                print(f"{v:>4}: {med:10.1f} us  (baseline)")
            else:
                sp = results[base] / med
                print(f"{v:>4}: {med:10.1f} us  {sp:5.2f}x vs {base}  "
                      f"({100.0*(results[base]-med)/results[base]:+.1f}% time)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
