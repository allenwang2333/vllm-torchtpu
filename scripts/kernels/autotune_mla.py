# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Search the MLA kernel's block schedule and emit a tuned entry.

`kernels/mla/v2/tuned_params.py` maps a shape to a block schedule. It is
populated entirely by hand, and a shape that misses falls back to
`num_kv_pages_per_block=1` -- the minimum legal value, which walks the paged
KV cache a page per block and leaves the kernel waiting on DMA. On
Kimi-Linear-48B that fallback path measured **80.3% of device time at 2.76%
FLOPS and 5.82% HBM utilization**, averaging 27.54 ms per call.

There is no tool in this repo to produce those entries; there are six such
tables and every row was typed by a person. This is that tool for the MLA
kernel: it drives `mla_ragged_paged_attention` directly, times each candidate
schedule on device, and prints a paste-ready `TunableParams`.

Driving the kernel directly rather than a whole server matters for turnaround.
A serving benchmark costs a cold compile of the full model -- about eleven
minutes -- for one point. Here a point costs one kernel compile, so the search
covers the space in the time a single server takes to boot.

Usage, on a TPU host:

    python scripts/kernels/autotune_mla.py                     # Kimi-Linear
    python scripts/kernels/autotune_mla.py --num-q-heads 12    # another shape
    python scripts/kernels/autotune_mla.py --kv-pages 1,2,4,8,16,32,64

Shape defaults are Kimi-Linear-48B-A3B at TP=4, read off the XLA expression
for `MLA-m-bq_16-bkvp_1-p_16-bsz_1` in an XProf HLO op profile:

    output      bf16[2048, 8, 512]        tokens, local heads, latent dim
    q_pe pad    bf16[2048, 8, 128]        padded RoPE dim
    kv cache    bf16[455811, 8, 2, 640]   pages, page/packing, packing, 512+128
"""

from __future__ import annotations

import argparse
import statistics
import time
import traceback

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-q-heads", type=int, default=8,
                   help="Local query heads (total heads / TP).")
    p.add_argument("--tokens-per-seq", type=int, default=256,
                   help="Query tokens per sequence in the chunk.")
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--lkv-dim", type=int, default=512)
    p.add_argument("--r-dim", type=int, default=64)
    p.add_argument("--padded-r-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--packing", type=int, default=2)
    p.add_argument("--context-pages", type=int, default=512,
                   help="Cache pages per sequence. The kernel walks these, so "
                        "it is what makes the block size matter.")
    p.add_argument("--kv-pages", type=_int_list,
                   default=[1, 2, 4, 8, 16, 32, 64],
                   help="num_kv_pages_per_block candidates.")
    p.add_argument("--queries", type=_int_list, default=[16],
                   help="num_queries_per_block candidates.")
    p.add_argument("--vmem-mb", type=_int_list, default=[60],
                   help="vmem_limit_bytes candidates, in MiB. The tuned "
                        "entries use 60; the field's own guidance is 16 to 64 "
                        "in steps of 8, lowest value giving best performance.")
    p.add_argument("--iters", type=int, default=20,
                   help="Timed iterations per candidate, after warmup.")
    return p.parse_args()


def build_inputs(a: argparse.Namespace):
    """Inputs in the layout `tests/kernels/test_mla_v2.py` establishes."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    total_tokens = a.tokens_per_seq * a.num_seqs
    total_pages = a.num_seqs * a.context_pages

    ql_nope = jax.random.normal(
        k1, (a.num_q_heads, total_tokens, a.lkv_dim), jnp.bfloat16)
    q_pe = jax.random.normal(
        k2, (total_tokens, a.num_q_heads, a.r_dim), jnp.bfloat16)
    new_kv_c = jax.random.normal(k3, (total_tokens, a.lkv_dim), jnp.bfloat16)
    new_k_pe = jax.random.normal(k4, (total_tokens, a.r_dim), jnp.bfloat16)
    cache_kv = jnp.zeros(
        (total_pages, a.page_size // a.packing, a.packing,
         a.lkv_dim + a.padded_r_dim), jnp.bfloat16)

    # Every sequence sits in the mixed segment: distribution is
    # [decode_end, prefill_end, mixed_end], and the mixed kernel is the one
    # that owns the device time.
    return dict(
        ql_nope=ql_nope,
        q_pe=q_pe,
        new_kv_c=new_kv_c,
        new_k_pe=new_k_pe,
        cache_kv=cache_kv,
        kv_lens=jnp.full((a.num_seqs, ),
                         a.context_pages * a.page_size, jnp.int32),
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(a.num_seqs + 1, dtype=jnp.int32) *
        a.tokens_per_seq,
        distribution=jnp.asarray([0, 0, a.num_seqs], jnp.int32),
    )


def time_candidate(inputs: dict, kv_pages: int, queries: int, vmem_mb: int,
                   iters: int) -> tuple[float | None, str]:
    """Median milliseconds for one schedule, or (None, reason) if it fails."""

    def run():
        out, cache = mla_ragged_paged_attention(
            inputs["ql_nope"],
            inputs["q_pe"],
            inputs["new_kv_c"],
            inputs["new_k_pe"],
            inputs["cache_kv"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["distribution"],
            # (decode, prefill, mixed) -- tune the mixed stage.
            num_kv_pages_per_block=(1, kv_pages, kv_pages),
            num_queries_per_block=(1, queries, queries),
            vmem_limit_bytes=vmem_mb * 1024 * 1024,
        )
        out.block_until_ready()
        cache.block_until_ready()

    try:
        run()  # warmup: pays the compile
    except Exception as exc:  # noqa: BLE001 - OOM and unsupported both land here
        reason = str(exc).strip().splitlines()[0][:100]
        return None, reason

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), ""


def main() -> int:
    a = parse_args()
    print(f"devices: {jax.devices()}")
    print(f"shape: heads={a.num_q_heads} lkv={a.lkv_dim} r={a.r_dim} "
          f"page_size={a.page_size} packing={a.packing}")
    print(f"work:  {a.num_seqs} seqs x {a.tokens_per_seq} tokens, "
          f"{a.context_pages} cache pages/seq\n")

    inputs = build_inputs(a)
    results: dict[tuple[int, int, int], float] = {}

    header = (f"{'kv_pages':>9} {'queries':>8} {'vmem_mb':>8} "
              f"{'median_ms':>11} {'vs bkvp=1':>10}")
    print(header)
    print("-" * len(header))

    for vmem_mb in a.vmem_mb:
        for queries in a.queries:
            for kv_pages in a.kv_pages:
                ms, reason = time_candidate(inputs, kv_pages, queries, vmem_mb,
                                            a.iters)
                if ms is None:
                    print(f"{kv_pages:>9} {queries:>8} {vmem_mb:>8} "
                          f"{'FAILED':>11}   {reason}")
                    continue
                results[(kv_pages, queries, vmem_mb)] = ms
                base = results.get((1, queries, vmem_mb))
                rel = f"{base / ms:>9.2f}x" if base else "        --"
                print(f"{kv_pages:>9} {queries:>8} {vmem_mb:>8} "
                      f"{ms:>11.3f} {rel}")

    if not results:
        print("\nno candidate ran; nothing to emit")
        return 1

    (best_kv, best_q, best_vmem), best_ms = min(results.items(),
                                                key=lambda kv: kv[1])
    base = results.get((1, best_q, best_vmem))
    print(f"\nbest: num_kv_pages_per_block={best_kv} "
          f"num_queries_per_block={best_q} vmem={best_vmem} MiB "
          f"at {best_ms:.3f} ms")
    if base:
        print(f"speedup over the untuned fallback (kv_pages=1): "
              f"{base / best_ms:.2f}x")

    print("\npaste into tuned_params_mapping in "
          "kernels/mla/v2/tuned_params.py:\n")
    print("    TuningKey(")
    print("        case=\"mixed\",")
    print(f"        actual_num_q_heads={a.num_q_heads},")
    print(f"        actual_lkv_dim={a.lkv_dim},")
    print(f"        actual_r_dim={a.r_dim},")
    print(f"        page_size_per_kv_packing={a.page_size},")
    print(f"        kv_packing={a.packing},")
    print("    ):")
    print("    TunableParams(")
    print(f"        num_kv_pages_per_block={best_kv},")
    print(f"        num_queries_per_block={best_q},")
    print(f"        vmem_limit_bytes={best_vmem * 1024 * 1024},")
    print("    ),")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - surface the whole trace to the operator
        traceback.print_exc()
        raise SystemExit(1)
