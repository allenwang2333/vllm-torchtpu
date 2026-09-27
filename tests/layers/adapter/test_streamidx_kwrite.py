# SPDX-License-Identifier: Apache-2.0
"""TPU_STREAMIDX_KWRITE_DUS: the page-wise indexer K write equals the scatter.

`_write_indexer_k_pages` (and the page path `_page_write_indexer_k` it picks
for prefill-sized steps) must leave the cache bit-identical to
`_scatter_indexer_k` for every batch shape the StreamIndex op sees. The cache
starts as random bytes, so a stray write to a slot the step does not own, or
a missed write, shows up as a mismatch.

Run as a script for the microbenchmark (µs per call, both paths, and the
count of whole-cache copies in each compiled HLO):

    python tests/layers/adapter/test_streamidx_kwrite.py --num-blocks 384
"""

import argparse
import re
import time
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    _page_write_indexer_k,
    _scatter_indexer_k,
    _write_indexer_k_pages,
)

# GLM-5.2 indexer cache: block_size 1024, kv_packing 4, lkv_dim 256 (D=128 +
# scale, padded to the 128-lane multiple).
BLOCK_SIZE = 1024
KV_PACKING = 4
LKV_DIM = 256
PAGES_PER_SEQ = 8


def _make_batch(rng, num_blocks, num_tokens, seqs, num_seqs_padded):
    """seqs: list of (start_pos, q_len). Returns scatter/page-write inputs."""
    assert sum(q for _, q in seqs) <= num_tokens
    cache = rng.integers(
        0, 256, (num_blocks, BLOCK_SIZE // KV_PACKING, KV_PACKING, LKV_DIM), np.uint8
    )
    k = rng.integers(0, 256, (num_tokens, LKV_DIM), np.uint8)
    perm = rng.permutation(num_blocks)
    page_indices = np.zeros((num_seqs_padded, PAGES_PER_SEQ), np.int32)
    seq_lens = np.zeros((num_seqs_padded,), np.int32)
    cu = np.zeros((num_seqs_padded + 1,), np.int32)
    for i, (start, q) in enumerate(seqs):
        page_indices[i] = perm[i * PAGES_PER_SEQ : (i + 1) * PAGES_PER_SEQ]
        seq_lens[i] = start + q
        cu[i + 1] = cu[i] + q
    cu[len(seqs) + 1 :] = cu[len(seqs)]
    return (
        jnp.asarray(cache),
        jnp.asarray(k),
        jnp.asarray(seq_lens),
        jnp.asarray(page_indices.reshape(-1)),
        jnp.asarray(cu),
    )


CASES = {
    # (num_tokens bucket, [(start_pos, q_len)...], num_seqs padded)
    "aligned_full_chunk": (1024, [(4 * 1024, 1024)], 1),
    "partial_last_chunk": (1024, [(4 * 1024, 700)], 1),
    "unaligned_chunk": (1024, [(4 * 1024 + 300, 1024)], 1),
    "unaligned_partial_3pages": (2048, [(1024 + 1000, 1100)], 2),
    "multi_seq": (2048, [(3000, 1), (1024 + 17, 900), (2048, 700), (0, 300)], 8),
    "decode_bucket_16": (16, [(p, 1) for p in (5, 1023, 1024, 4095, 77, 2048, 3333,
                                             6000, 100, 8191, 0)], 16),
    "empty_step": (16, [], 4),
    "empty_prefill_bucket": (1024, [], 4),
    # 1 chunk + 16 decode rows: 17+ pages > 1024 // 1024 + 2 -> scatter branch.
    "mixed_fallback": (1024, [(2048, 700)] + [(400 * i + 5, 1) for i in range(16)],
                       20),
}


def _run_case(case, fn):
    num_tokens, seqs, num_seqs_padded = CASES[case]
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    args = _make_batch(rng, 160, num_tokens, seqs, num_seqs_padded)
    ref = np.asarray(jax.jit(_scatter_indexer_k)(*args))
    got = np.asarray(jax.jit(fn)(*args))
    n_diff = int((ref != got).sum())
    print(f"{case}: bytes_differ={n_diff} changed_vs_input="
          f"{int((ref != np.asarray(args[0])).sum())}")
    return n_diff


@pytest.mark.parametrize("case", sorted(CASES))
def test_dispatch_matches_scatter(case):
    assert _run_case(case, _write_indexer_k_pages) == 0


@pytest.mark.parametrize(
    "case", sorted(c for c in CASES if c != "mixed_fallback")
)
def test_page_path_matches_scatter(case):
    """The page path alone, given enough pages, on every shape incl. decode."""
    assert _run_case(
        case, lambda *a: _page_write_indexer_k(*a, max_pages=24)
    ) == 0


def _whole_cache_copies(hlo_text, num_blocks):
    """Count ops that produce the cache flattened to [rows, lkv_dim].

    The row scatter's relayout shows up as a `reshape` of the tiled 4-D cache
    into u8[num_blocks * 1024, 256] (a retiling, not a bitcast) and back.
    Returns (flattening ops, 4-D whole-cache copy/transpose ops).
    """
    flat = re.compile(rf"= u8\[{num_blocks * BLOCK_SIZE},256\]\{{[^}}]*\}} \w")
    copy4d = re.compile(
        rf"= u8\[{num_blocks},256,4,256\]\{{[^}}]*\}} (copy|transpose)\("
    )
    return len(flat.findall(hlo_text)), len(copy4d.findall(hlo_text))


def _bench(fn, args, iters):
    f = jax.jit(fn, donate_argnums=(0,))
    compiled = f.lower(*args).compile()
    cache = args[0]
    rest = args[1:]
    for _ in range(3):
        cache = f(cache, *rest)
    cache.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        cache = f(cache, *rest)
    cache.block_until_ready()
    us = (time.perf_counter() - t0) / iters * 1e6
    try:
        hlo = compiled.as_text()
    except Exception as e:  # noqa: BLE001
        # Some XLA builds cannot round-trip an async SparseCore-offloaded op
        # through the text printer; dump with XLA_FLAGS=--xla_dump_to instead.
        print(f"as_text failed ({type(e).__name__}); use --xla_dump_to")
        hlo = ""
    return us, hlo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-blocks", type=int, default=384)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--hlo-dir", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    shapes = {
        "prefill_aligned_1024": (1024, [(64 * 1024, 1024)], 1),
        "prefill_unaligned_1024": (1024, [(64 * 1024 + 300, 1024)], 1),
        "decode_16": (16, [(1000 * i + 5, 1) for i in range(16)], 16),
        "mixed_fallback_1024": (
            1024, [(2048, 700)] + [(1000 * i + 5, 1) for i in range(16)], 17
        ),
    }
    for name, (T, seqs, ns) in shapes.items():
        global PAGES_PER_SEQ
        PAGES_PER_SEQ = 80 if ns == 1 else 16
        nb = max(a.num_blocks, PAGES_PER_SEQ * ns)
        args = _make_batch(rng, nb, T, seqs, ns)
        cache_np = np.asarray(args[0])
        for label, fn in (("scatter", _scatter_indexer_k),
                          ("kwrite_dus", _write_indexer_k_pages)):
            args = (jnp.asarray(cache_np),) + args[1:]
            us, hlo = _bench(fn, args, a.iters)
            flat, copies = _whole_cache_copies(hlo, nb)
            print(f"BENCH {name} num_blocks={nb} {label}: {us:9.1f} us/call  "
                  f"flattened_cache_ops={flat} whole_cache_copy_ops={copies}")
            if a.hlo_dir:
                with open(f"{a.hlo_dir}/{name}_{label}.hlo.txt", "w") as fh:
                    fh.write(hlo)


if __name__ == "__main__":
    main()
