"""MoE routing-sort microbenchmark, at the Qwen3.5-397B-A17B-FP8 prefill shape.

The profile of job j-80ee1b01-98d7-498e-a834 attributes 22.072 ms/step/core
(7.51% of a 294.09 ms prefill step) to ``custom_kernel/sort``, 180 instances per
step over 60 MoE layers = 3 sorts per layer, 122.6 us each.

The three sorts per layer, from the source:
  S1  fused_moe_gmm.py:210  jnp.argsort(sort_keys)      keys int32 in [0, 64]
  S2  fused_moe_gmm.py:211  jnp.argsort(sorted_indices) keys = a full permutation
  S3  ragged_gather_reduce_v2/wrapper.py:51
                            jnp.argsort(~valid_2d, stable=True, axis=-1)  bool key

Only S1 has the small key space the counting-sort proposal targets. S2 sorts a
permutation of [0, N) -- there is no small key space there, only an inverse.

Usage:  python sort_bench.py [baseline|alt|all]
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _bench_util import banner, bench, dispatch_floor, env_report  # noqa: E402

# SORT_BENCH_TOKENS lets a CPU smoke test run the same code at a small shape.
NUM_TOKENS = int(os.environ.get("SORT_BENCH_TOKENS", 16384))
TOPK = 10
N = NUM_TOKENS * TOPK  # 163840 routed rows
E = 64  # local experts per core (512 global / EP8)
E1 = E + 1  # + sentinel for "not routed to this rank"
# EP8 with topk 10 over 512 global experts -> ~10/8 = 1.25 of a token's 10
# choices land on any one rank, so ~12.5% of rows are valid.
VALID_FRAC = 0.125


def make_keys(seed=0):
    rng = np.random.default_rng(seed)
    valid = rng.random(N) < VALID_FRAC
    k = np.where(valid, rng.integers(0, E, size=N), E).astype(np.int32)
    return jnp.asarray(k), jnp.asarray(valid)


# ---------------------------------------------------------------- baselines

def s1_argsort(keys):
    return jnp.argsort(keys)


def s2_invert_argsort(perm):
    return jnp.argsort(perm)


def s1s2_both(keys):
    si = jnp.argsort(keys)
    return si, jnp.argsort(si)


def s3_bool_argsort(valid_2d):
    return jnp.argsort(~valid_2d, descending=False, stable=True, axis=-1)


def prepare_routed_current(keys, valid):
    """The whole permutation block of prepare_routed_gmm_inputs (S1+S2+counts)."""
    si = jnp.argsort(keys)
    revert = jnp.argsort(si)
    gs = (jax.nn.one_hot(jnp.where(valid, keys, 0), E, dtype=jnp.int32) *
          valid[:, None].astype(jnp.int32)).sum(axis=0)
    return si, revert, gs


# ---------------------------------------------------- cheap XLA alternatives

_IDX_BITS = 18  # 2**18 = 262144 > N = 163840
_IDX_MASK = (1 << _IDX_BITS) - 1


def packed_sort(keys):
    """Single-operand sort of (key << 18 | index) instead of a 2-operand argsort.

    Stable by construction; halves the payload each compare-exchange pass moves.
    """
    iota = jnp.arange(N, dtype=jnp.int32)
    packed = (keys << _IDX_BITS) | iota
    return jnp.sort(packed) & _IDX_MASK


def packed_sort_plus_scatter(keys):
    """packed_sort for S1, then a scatter for S2 instead of a second sort."""
    iota = jnp.arange(N, dtype=jnp.int32)
    si = jnp.sort((keys << _IDX_BITS) | iota) & _IDX_MASK
    revert = jnp.zeros(N, jnp.int32).at[si].set(iota)
    return si, revert


def scatter_invert(perm):
    """S2 replacement: revert[perm[i]] = i."""
    return jnp.zeros(N, jnp.int32).at[perm].set(jnp.arange(N, dtype=jnp.int32))


def argsort_u8_key(keys):
    """S1 with an 8-bit key and an int32 payload (keys fit in uint8)."""
    k8 = keys.astype(jnp.uint8)
    iota = jnp.arange(N, dtype=jnp.int32)
    _, si = jax.lax.sort((k8, iota), dimension=0, num_keys=1, is_stable=True)
    return si


def argsort_u16_key(keys):
    k16 = keys.astype(jnp.uint16)
    iota = jnp.arange(N, dtype=jnp.int32)
    _, si = jax.lax.sort((k16, iota), dimension=0, num_keys=1, is_stable=True)
    return si


# ------------------------------------------------------------ counting sorts

def counting_sort_2050f979(keys):
    """Verbatim _counting_sort_permutation from the reverted commit 2050f979.

    Materializes an [N, 65] int32 one-hot (42.6 MB at this shape) and runs a
    full-length cumsum down the N axis over it.
    """
    onehot = jax.nn.one_hot(keys, E1, dtype=jnp.int32)
    counts = onehot.sum(axis=0)
    rank = ((jnp.cumsum(onehot, axis=0) - onehot) * onehot).sum(axis=1)
    base = jnp.cumsum(counts) - counts
    destination = (onehot * base).sum(axis=1) + rank
    sorted_indices = jnp.zeros(N, dtype=jnp.int32).at[destination].set(
        jnp.arange(N, dtype=jnp.int32))
    return sorted_indices, destination, counts[:E]


def _blocked_destination(keys, block):
    """Two-level counting rank: cumsum only over ``block`` rows, narrow one-hot."""
    nb = N // block
    ohb = (keys.reshape(nb, block, 1) == jnp.arange(
        E1, dtype=jnp.int32).reshape(1, 1, E1))            # bool [nb, block, E1]
    ohi = ohb.astype(jnp.int16)
    within = jnp.cumsum(ohi, axis=1) - ohi                 # int16 rank in block
    block_counts = jnp.sum(ohi, axis=1).astype(jnp.int32)  # [nb, E1]
    counts = block_counts.sum(axis=0)                      # [E1]
    block_base = jnp.cumsum(block_counts, axis=0) - block_counts
    key_base = jnp.cumsum(counts) - counts
    total_base = block_base + key_base[None, :]            # [nb, E1]
    dest = jnp.where(ohb, within.astype(jnp.int32) + total_base[:, None, :],
                     0).sum(axis=2).reshape(N)
    return dest, counts


def counting_sort_blocked(keys, block=256):
    dest, counts = _blocked_destination(keys, block)
    sorted_indices = jnp.zeros(N, dtype=jnp.int32).at[dest].set(
        jnp.arange(N, dtype=jnp.int32))
    return sorted_indices, dest, counts[:E]


def counting_dest_only_blocked(keys, block=256):
    """Destination (== S2's output) only, to price the scatter separately."""
    return _blocked_destination(keys, block)[0]


def prepare_routed_count_s2(keys, valid, block=256):
    """The permutation block with S2 replaced by the blocked counting rank.

    S1 stays a real argsort (its output is used as a gather index and the
    scatter that would reproduce it costs more than the sort). group_sizes
    falls out of the same one-hot, so the separate one-hot goes away too.
    """
    si = jnp.argsort(keys)
    dest, counts = _blocked_destination(keys, block)
    _ = valid
    return si, dest, counts[:E]


# ------------------------------------------------- S3: boolean stable partition

R3 = 8            # cfg.num_row_partitions
C3 = N // R3      # cfg.row_partition_size


def _excl_cumsum_blocked(v, block):
    """Exclusive cumsum along the last axis, computed two-level."""
    r, c = v.shape
    nb = c // block
    vb = v.reshape(r, nb, block)
    within = jnp.cumsum(vb, axis=-1) - vb
    bc = vb.sum(axis=-1)                            # [r, nb]
    bbase = jnp.cumsum(bc, axis=-1) - bc            # [r, nb]
    return (within + bbase[..., None]).reshape(r, c)


def s3_dest_cumsum(valid_2d):
    """Where each source row lands under the stable partition (flat cumsum)."""
    v = valid_2d.astype(jnp.int32)
    iv = 1 - v
    rank_v = jnp.cumsum(v, axis=-1) - v
    rank_i = jnp.cumsum(iv, axis=-1) - iv
    nvalid = v.sum(axis=-1, keepdims=True)
    return jnp.where(valid_2d, rank_v, nvalid + rank_i)


def s3_dest_blocked(valid_2d, block=512):
    v = valid_2d.astype(jnp.int32)
    iv = 1 - v
    rank_v = _excl_cumsum_blocked(v, block)
    rank_i = _excl_cumsum_blocked(iv, block)
    nvalid = v.sum(axis=-1, keepdims=True)
    return jnp.where(valid_2d, rank_v, nvalid + rank_i)


def _s3_scatter_from_dest(dest):
    off = jnp.arange(R3, dtype=jnp.int32)[:, None] * C3
    flat_dest = (dest + off).reshape(-1)
    iota = jnp.broadcast_to(jnp.arange(C3, dtype=jnp.int32),
                            (R3, C3)).reshape(-1)
    return jnp.zeros(R3 * C3, jnp.int32).at[flat_dest].set(iota).reshape(
        R3, C3)


def s3_partition_cumsum(valid_2d):
    return _s3_scatter_from_dest(s3_dest_cumsum(valid_2d))


def s3_partition_blocked(valid_2d, block=512):
    return _s3_scatter_from_dest(s3_dest_blocked(valid_2d, block))


# ------------------------------------------------------------------- Pallas

def _pallas_counting(keys, block, rank_mode):
    """Pallas counting sort: keep the [block, 128] one-hot in VMEM.

    Pass 1 writes a per-block histogram; XLA does the tiny cross-block prefix;
    pass 2 recomputes the one-hot in VMEM and emits each row's destination
    (which is exactly argsort(argsort(keys))).

    ``rank_mode`` picks how the within-block rank is computed:
      "cumsum" -- jnp.cumsum down the sublane axis inside the kernel
      "trimm"  -- strictly-lower-triangular matmul on the MXU (no scan needed)
    """
    from jax.experimental import pallas as pl

    interpret = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))
    nb = N // block
    e1p = 128  # pad the key axis to one lane group
    k2 = keys.reshape(N, 1)

    def _lanes(shape):
        # Built inside the kernel: pallas_call refuses captured constants.
        return jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # The per-block histogram is [nb, e1p], but a (1, e1p) block of it is
    # rejected: Pallas TPU wants the last two block dims divisible by (8, 128)
    # *or* equal to the array dims. Carrying it as [nb, 1, e1p] with a
    # (1, 1, e1p) block satisfies the second form (1 == 1, 128 % 128 == 0).
    def hist_kernel(k_ref, o_ref):
        oh = (k_ref[...] == _lanes((block, e1p))).astype(jnp.int32)
        o_ref[...] = jnp.sum(oh, axis=0, keepdims=True)[None]

    hist3 = pl.pallas_call(
        hist_kernel,
        grid=(nb, ),
        in_specs=[pl.BlockSpec((block, 1), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 1, e1p), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((nb, 1, e1p), jnp.int32),
        interpret=interpret,
    )(k2)
    hist = hist3[:, 0, :]                           # [nb, e1p]

    counts = hist.sum(axis=0)                       # [e1p]
    block_base = jnp.cumsum(hist, axis=0) - hist    # [nb, e1p]
    key_base = jnp.cumsum(counts) - counts          # [e1p]
    base = (block_base + key_base[None, :])[:, None, :]   # [nb, 1, e1p]

    tri = None
    if rank_mode == "trimm":
        r = jnp.arange(block, dtype=jnp.int32)
        tri = (r[:, None] > r[None, :]).astype(jnp.bfloat16)  # strictly lower

    def _finish(ohb, rank, b_ref, o_ref):
        mine = jnp.sum(jnp.where(ohb, b_ref[0], 0), axis=1, keepdims=True)
        o_ref[...] = rank + mine

    def dest_kernel_cumsum(k_ref, b_ref, o_ref):
        ohb = (k_ref[...] == _lanes((block, e1p)))
        ohi = ohb.astype(jnp.int32)
        rank = jnp.sum((jnp.cumsum(ohi, axis=0) - ohi) * ohi,
                       axis=1,
                       keepdims=True)
        _finish(ohb, rank, b_ref, o_ref)

    def dest_kernel_trimm(k_ref, b_ref, t_ref, o_ref):
        ohb = (k_ref[...] == _lanes((block, e1p)))
        prefix = jax.lax.dot_general(t_ref[...],
                                     ohb.astype(jnp.bfloat16),
                                     (((1, ), (0, )), ((), ())),
                                     preferred_element_type=jnp.float32)
        rank = jnp.sum(jnp.where(ohb, prefix, 0.0), axis=1,
                       keepdims=True).astype(jnp.int32)
        _finish(ohb, rank, b_ref, o_ref)

    dest_kernel = (dest_kernel_cumsum
                   if rank_mode == "cumsum" else dest_kernel_trimm)

    in_specs = [
        pl.BlockSpec((block, 1), lambda i: (i, 0)),
        pl.BlockSpec((1, 1, e1p), lambda i: (i, 0, 0)),
    ]
    operands = [k2, base]
    if rank_mode == "trimm":
        in_specs.append(pl.BlockSpec((block, block), lambda i: (0, 0)))
        operands.append(tri)

    out = pl.pallas_call(
        dest_kernel,
        grid=(nb, ),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((block, 1), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((N, 1), jnp.int32),
        interpret=interpret,
    )(*operands)
    return out.reshape(N)


def pallas_counting_cumsum(keys, block=1024):
    return _pallas_counting(keys, block, "cumsum")


def pallas_counting_trimm(keys, block=512):
    return _pallas_counting(keys, block, "trimm")


# --------------------------------------------------------------- correctness

def check(name, got, want):
    g, w = np.asarray(got), np.asarray(want)
    ok = g.shape == w.shape and np.array_equal(g, w)
    print(f"CHECK  {name:<40s} {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        if g.shape != w.shape:
            print(f"       shape {g.shape} vs {w.shape}", flush=True)
        else:
            bad = np.flatnonzero(g != w)
            print(f"       {bad.size} of {g.size} differ; first at {bad[:5]} "
                  f"got {g[bad[:5]]} want {w[bad[:5]]}", flush=True)
    return ok


def guarded(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        print(f"CHECK  {name:<40s} FAILED: {type(e).__name__}: "
              f"{str(e)[:500]}", flush=True)
        return None


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "all"
    dev = env_report()
    print(f"shape: N={N} (num_tokens={NUM_TOKENS} x topk={TOPK}), "
          f"local_experts={E}, valid_frac={VALID_FRAC}", flush=True)
    print(f"dispatch floor: {dispatch_floor(dev):.2f} us/call", flush=True)

    keys, valid = make_keys()
    keys = jax.device_put(keys, dev)
    valid = jax.device_put(valid, dev)
    perm = jax.device_put(jnp.argsort(keys), dev)
    valid_2d = jax.device_put(valid.reshape(8, N // 8), dev)

    if group in ("baseline", "all"):
        banner("BASELINE - the three sorts the profile counts")
        bench("S1 argsort(keys) [N] int32 k<=64", s1_argsort, (keys, ), dev)
        bench("S2 argsort(perm) [N] full permutation", s2_invert_argsort,
              (perm, ), dev)
        bench("S1+S2 chained", s1s2_both, (keys, ), dev)
        bench("S3 argsort(~valid) [8,20480] bool", s3_bool_argsort,
              (valid_2d, ), dev)
        bench("prepare_routed permutation block", prepare_routed_current,
              (keys, valid), dev)

        banner("SCALING - is the 163840-row padding what costs?")
        for n in (16384, 40960, 81920, 163840):
            sub = jax.device_put(keys[:n], dev)
            bench(f"argsort(keys[:{n}])", jnp.argsort, (sub, ), dev)

    if group in ("alt", "all"):
        banner("CORRECTNESS of the alternatives vs jnp.argsort")
        want_si = np.asarray(jnp.argsort(keys))
        want_rev = np.asarray(jnp.argsort(jnp.argsort(keys)))
        want_gs = np.asarray(
            (jax.nn.one_hot(jnp.where(valid, keys, 0), E, dtype=jnp.int32) *
             valid[:, None].astype(jnp.int32)).sum(axis=0))

        r = guarded("packed_sort", packed_sort, keys)
        if r is not None:
            check("packed_sort", r, want_si)
        r = guarded("argsort_u8_key", argsort_u8_key, keys)
        if r is not None:
            check("argsort_u8_key", r, want_si)
        r = guarded("scatter_invert", scatter_invert, perm)
        if r is not None:
            check("scatter_invert", r, want_rev)
        r = guarded("counting_sort_2050f979", counting_sort_2050f979, keys)
        if r is not None:
            check("counting_sort_2050f979 (si)", r[0], want_si)
            check("counting_sort_2050f979 (rev)", r[1], want_rev)
            check("counting_sort_2050f979 (gs)", r[2], want_gs)
        r = guarded("counting_sort_blocked", counting_sort_blocked, keys)
        if r is not None:
            check("counting_sort_blocked (si)", r[0], want_si)
            check("counting_sort_blocked (rev)", r[1], want_rev)
            check("counting_sort_blocked (gs)", r[2], want_gs)
        r = guarded("pallas_counting_cumsum", pallas_counting_cumsum, keys)
        if r is not None:
            check("pallas_counting_cumsum (rev)", r, want_rev)
        r = guarded("pallas_counting_trimm", pallas_counting_trimm, keys)
        if r is not None:
            check("pallas_counting_trimm (rev)", r, want_rev)

        banner("ALTERNATIVES - timing")
        bench("packed_sort  (S1 replacement)", packed_sort, (keys, ), dev)
        bench("argsort_u8_key (S1 replacement)", argsort_u8_key, (keys, ), dev)
        bench("argsort_u16_key (S1 replacement)", argsort_u16_key, (keys, ),
              dev)
        bench("scatter_invert (S2 replacement)", scatter_invert, (perm, ), dev)
        bench("packed_sort+scatter (S1+S2)", packed_sort_plus_scatter,
              (keys, ), dev)
        bench("counting_sort_2050f979 (S1+S2)", counting_sort_2050f979,
              (keys, ), dev, iters=20)
        for blk in (128, 256, 512):
            bench(f"counting_sort_blocked b={blk} (S1+S2)",
                  lambda k, b=blk: counting_sort_blocked(k, b), (keys, ), dev,
                  iters=50)
        for blk in (128, 256, 512, 1024):
            bench(f"counting_dest_only b={blk} (S2 only)",
                  lambda k, b=blk: counting_dest_only_blocked(k, b), (keys, ),
                  dev, iters=50)
        bench("prepare_routed w/ counting S2 (block)", prepare_routed_count_s2,
              (keys, valid), dev, iters=50)
        bench("prepare_routed CURRENT (block, A/B ref)",
              prepare_routed_current, (keys, valid), dev)
        for blk in (512, 1024, 2048, 8192):
            bench(f"pallas_counting cumsum b={blk} (S2 only)",
                  lambda k, b=blk: _pallas_counting(k, b, "cumsum"), (keys, ),
                  dev)
        for blk in (256, 512):
            bench(f"pallas_counting trimm b={blk} (S2 only)",
                  lambda k, b=blk: _pallas_counting(k, b, "trimm"), (keys, ),
                  dev)

    if group in ("s3", "all"):
        banner("S3 - boolean stable partition, [8, 20480] bool")
        want_s3 = np.asarray(s3_bool_argsort(valid_2d))
        r = guarded("s3_partition_cumsum", s3_partition_cumsum, valid_2d)
        if r is not None:
            check("s3_partition_cumsum", r, want_s3)
        for blk in (128, 256, 512, 1024, 2048):
            r = guarded(f"s3_partition_blocked b={blk}", s3_partition_blocked,
                        valid_2d, blk)
            if r is not None:
                check(f"s3_partition_blocked b={blk}", r, want_s3)

        bench("S3 CURRENT argsort(~valid) stable", s3_bool_argsort,
              (valid_2d, ), dev)
        bench("s3_partition_cumsum (full argsort out)", s3_partition_cumsum,
              (valid_2d, ), dev, iters=50)
        for blk in (128, 256, 512, 1024, 2048):
            bench(f"s3_partition_blocked b={blk}",
                  lambda v, b=blk: s3_partition_blocked(v, b), (valid_2d, ),
                  dev, iters=50)
        bench("s3_dest_blocked b=512 (no scatter)",
              lambda v: s3_dest_blocked(v, 512), (valid_2d, ), dev, iters=50)


if __name__ == "__main__":
    main()
