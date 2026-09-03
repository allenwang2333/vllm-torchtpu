# SPDX-License-Identifier: Apache-2.0
"""PCP scaling benchmark for the Deepseek V4 StreamIndex Top-K kernel.

Three measurements, because they answer different questions:

  ``sweep``  per-rank kernel time at each ``pcp_size``, on one device. Ranks are
             shape-symmetric, so running rank 0 with a correctly-sized shard is
             the per-rank latency; this needs no mesh and so can go past the
             device count of the host.
  ``merge``  the cross-rank stage-2 top-k on its own. Note the shape is
             ``[T // pcp, pcp * k]`` -- after the all-to-all a rank merges only
             its *own* tokens. Sizing it ``[T, pcp * k]`` is the easy mistake and
             makes the merge look like it dominates at high ``pcp_size``.
  ``mesh``   end-to-end ``streamindex_topk_pcp`` on a real mesh, which is the
             only one of the three that pays for the all-to-all.

Example, reproducing the numbers in the commit that added PCP support::

  python scripts/benchmark_dsv4_streamindex_pcp.py --mode sweep --pcp 1,2,4,8,16
  python scripts/benchmark_dsv4_streamindex_pcp.py --mode merge --pcp 2,4,8,16
  python scripts/benchmark_dsv4_streamindex_pcp.py --mode mesh  --pcp 8

``sweep`` runs one ``pcp_size`` per process when driven from a shell loop, which
is worth doing: the dense score buffer is 1 GiB at ``pcp_size=1`` and an OOM
there should not take the rest of the sweep with it.
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    PCP_AXIS_NAME, streamindex_topk, streamindex_topk_pcp)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order

P = jax.sharding.PartitionSpec


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def build_cache(num_pages: int, page_size: int, head_dim: int, seed: int):
    """Packed fp8 index cache: ``[pages, page_size // 4, 4, width]`` uint8.

    Slot ``s`` of a page lives at ``[s // 4, s % 4]``, which is exactly what the
    final reshape produces. Values are arbitrary but valid: real fp8 bytes and
    an e8m0 exponent of 127, i.e. a scale of 1.0, so nothing here is NaN.
    """
    width = align_to(head_dim + head_dim // 128, 128)
    n = num_pages * page_size
    vals = jax.random.normal(jax.random.key(seed),
                             (n, head_dim), jnp.float32) * 0.5
    fp8 = jax.lax.bitcast_convert_type(vals.astype(jnp.float8_e4m3fn),
                                       jnp.uint8).reshape(n, head_dim)
    rec = jnp.concatenate([fp8, jnp.full((n, 1), 127, jnp.uint8)], axis=-1)
    rec = jnp.pad(rec, ((0, 0), (0, width - rec.shape[-1])))
    return rec.reshape(num_pages, page_size // 4, 4, width)


def timeit(fn, *, warmup: int, iters: int) -> float:
    """Mean wall-clock milliseconds per call."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / iters * 1e3


def make_queries(args):
    kq, kw = jax.random.split(jax.random.key(0))
    q = (jax.random.normal(kq, (args.q_len, args.num_q_heads, args.head_dim),
                           jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (args.q_len, args.num_q_heads),
                                jnp.float32).astype(jnp.bfloat16)
    return q, weights


def run_sweep(args) -> None:
    """Per-rank kernel time, one rank on one device, at each pcp_size."""
    q, weights = make_queries(args)
    seq_lens = jnp.array([args.kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, args.q_len], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)  # one prefill sequence

    for pcp in args.pcp:
        # A virtual page spans page_size * pcp compressed positions, of which
        # each rank holds page_size, so the per-rank page count divides by pcp.
        vpages = cdiv(args.kv_len, args.page_size * pcp)
        local_n = vpages * args.page_size
        sublanes = align_to(vpages, args.bkv_p) * args.page_size // 128
        score_mib = args.q_len * sublanes * 128 * 4 / 2**20

        cache = build_cache(vpages, args.page_size, args.head_dim, seed=pcp)
        page_indices = jnp.arange(vpages, dtype=jnp.int32)
        fn = functools.partial(streamindex_topk,
                               q,
                               weights,
                               cache,
                               seq_lens,
                               page_indices,
                               cu_q_lens,
                               distribution,
                               k=args.k,
                               compression_ratio=args.comp_ratio,
                               num_kv_pages_per_block=args.bkv_p,
                               num_queries_per_block=args.bq_sz,
                               pcp_size=pcp,
                               interleave_size=args.interleave_size,
                               pcp_rank=jnp.int32(0),
                               return_scores=pcp > 1)
        ms = timeit(fn, warmup=args.warmup, iters=args.iters)
        print(f"pcp={pcp:3d}  local_N={local_n:8d}  "
              f"scores={score_mib:8.1f} MiB  kernel={ms:9.3f} ms")


def run_merge(args) -> None:
    """Cross-rank stage 2, at its true post-all-to-all size and the naive one."""
    for pcp in args.pcp:
        rows = args.q_len // pcp
        width = pcp * args.k
        for label, b in (("per-rank", rows), ("all-T", args.q_len)):
            cand = jax.random.normal(jax.random.key(1), (b, width),
                                     jnp.float32)
            lens = jnp.full((b, ), width, jnp.int32)
            ms = timeit(functools.partial(sparsecore_topk,
                                          cand,
                                          args.k,
                                          row_lengths=lens,
                                          write_empty_rows=True),
                        warmup=args.warmup,
                        iters=args.iters)
            print(
                f"pcp={pcp:3d}  {label:8s} [{b:5d},{width:6d}]  {ms:7.3f} ms")


def run_mesh(args) -> None:
    """End-to-end on a real mesh, the only mode that pays for the all-to-all."""
    devices = jax.devices()
    for pcp in args.pcp:
        if pcp < 2:
            # There is nothing to shard or exchange; run --mode sweep for the
            # unsharded baseline.
            print(f"pcp={pcp:3d}  skipped: mesh mode needs pcp_size > 1")
            continue
        if len(devices) < pcp:
            print(f"pcp={pcp:3d}  skipped: needs {pcp} devices, "
                  f"have {len(devices)}")
            continue
        mesh = jax.sharding.Mesh(np.array(devices[:pcp]), (PCP_AXIS_NAME, ))
        vpages = cdiv(args.kv_len, args.page_size * pcp)

        q, weights = make_queries(args)
        # Concatenate the shards along axis 0 so P(pcp) hands rank r its own.
        cache = jnp.concatenate([
            build_cache(vpages, args.page_size, args.head_dim, seed=100 + r)
            for r in range(pcp)
        ],
                                axis=0)
        cache = jax.device_put(
            cache, jax.sharding.NamedSharding(mesh, P(PCP_AXIS_NAME)))
        order, _ = build_pcp_rank_major_token_order([args.q_len], pcp,
                                                    args.interleave_size,
                                                    args.q_len)
        call = jax.jit(
            functools.partial(streamindex_topk_pcp,
                              mesh=mesh,
                              k=args.k,
                              compression_ratio=args.comp_ratio,
                              pcp_size=pcp,
                              interleave_size=args.interleave_size,
                              num_kv_pages_per_block=args.bkv_p,
                              num_queries_per_block=args.bq_sz))
        call_args = (q, weights, cache, jnp.array([args.kv_len], jnp.int32),
                     jnp.arange(vpages, dtype=jnp.int32),
                     jnp.array([0, args.q_len],
                               jnp.int32), jnp.array([0, 0, 1], jnp.int32),
                     jnp.asarray(order, jnp.int32))
        out = np.asarray(jax.block_until_ready(call(*call_args)))
        ms = timeit(functools.partial(call, *call_args),
                    warmup=args.warmup,
                    iters=args.iters)
        # Exchange volume is 8 bytes per candidate: i32 index + f32 score. It
        # depends on neither pcp_size nor kv_len, which is why PCP scales.
        a2a_mib = args.q_len * args.k * 8 / 2**20
        print(
            f"pcp={pcp:3d}  end-to-end={ms:9.3f} ms  a2a={a2a_mib:5.1f} MiB  "
            f"filled={(out >= 0).mean() * 100:5.1f}%  max={out.max()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   choices=("sweep", "merge", "mesh"),
                   default="sweep")
    p.add_argument("--pcp",
                   type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8],
                   help="comma-separated pcp_size values")
    p.add_argument("--q-len", type=int, default=1024)
    p.add_argument("--kv-len", type=int, default=262144)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=1024)
    p.add_argument("--comp-ratio", type=int, default=1)
    p.add_argument("--interleave-size", type=int, default=1)
    p.add_argument("--bkv-p",
                   type=int,
                   default=2,
                   help="num_kv_pages_per_block")
    p.add_argument("--bq-sz",
                   type=int,
                   default=128,
                   help="num_queries_per_block")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    print(f"devices: {len(jax.devices())} x {jax.devices()[0].device_kind}")
    print(f"config: q_len={args.q_len} kv_len={args.kv_len} k={args.k} "
          f"H_idx={args.num_q_heads} D_idx={args.head_dim} "
          f"c={args.comp_ratio} page_size={args.page_size} "
          f"bkv_p={args.bkv_p} bq={args.bq_sz}\n")
    {"sweep": run_sweep, "merge": run_merge, "mesh": run_mesh}[args.mode](args)


if __name__ == "__main__":
    main()
