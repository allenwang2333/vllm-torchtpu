"""O2 tile arithmetic for streamindex_topk: extents, padded score width, VMEM.

Ints only -- no jax, no refs, no device. Everything here is testable on a
laptop, which is the point: the VMEM figures below are what decide whether a
tile is legal, and finding that out from a compile error on a four-chip pod is
the expensive way to learn it.

# Dimensions.
#   T        padded token count handed to the op (rows of the score matrix)
#   H        index_n_heads, 32
#   D        index_head_dim, 128
#   PSZ      page_size in compressed KV positions (1024 at (A), 512 at (B))
#   PPS      pages_per_seq (9 at (A), 2048 at (B))
#   N_pad    padded score extent, align_to(PPS, bkv_p) * PSZ -- NOT PPS*PSZ
#   sbs      seq_batch_size, the number of decode sequences fused per grid step
#   tq       tile extent of the query axis  (num_queries_per_block, "bq_sz")
#   tkv      tile extent of the KV axis in PAGES (num_kv_pages_per_block,
#            "bkv_p"); in positions it is tkv * PSZ, spelled bkv_sz
#   nkv      grid extent over KV = cdiv(kv_len, tkv * PSZ) -- the in-body
#            iteration count P-ITER charged 0.500 us each
"""

# Native lane width. Literal because it is a shape fact that appears in shape
# comments, which code-style.md 13 exempts; it is not a capacity constant.
LANES = 128


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def padded_n(page_size, pages_per_seq, bkv_ps):
    """The score-row width the launcher allocates, over ALL THREE tile slots.

    streamindex_topk.py:603-605 takes the max over the whole
    `num_kv_pages_per_block` tuple, including the PREFILL slot the launcher
    never uses. So a slot raised for a case that never runs still widens the
    score matrix -- which is exactly the (A) trap O2 exists to avoid: at
    PPS=9, align_to(9, 16) = 16 pages against 9 live, and N_pad goes
    10240 -> 16384, a 60% wider extent into a SparseCore whose cost is driven
    by that extent.
    """
    return max(align_to(pages_per_seq, bkv_p) * page_size
               for bkv_p in bkv_ps)


def kv_iterations(kv_len, page_size, bkv_p, q_len, bq_sz):
    """In-body iterations for one sequence: the quantity P-ITER priced.

    `measured (j-69ee326d)`: 0.500 us per iteration at B-D1-1M, linear to
    +/-0.8% over an 8x range, with copies, bytes, bytes/copy and the score
    extent all held constant.
    """
    return max(1, cdiv(q_len, bq_sz)) * max(1, cdiv(kv_len, bkv_p * page_size))


def scratch_bytes(*, sbs, bq_sz, bkv_p, page_size, kv_packing, lkv_dim,
                  num_q_heads, head_dim, q_itemsize, w_itemsize,
                  score_itemsize):
    """The four DECLARED double-buffered scratch arrays, in bytes.

    These are the terms design/measurement_and_design.md 7 counts. Reproduced
    here rather than cited so the numbers can be re-derived: the incumbent (B)
    MIXED tile (bq_sz=128, bkv_p=2, sbs=1) comes out at 2.53 MiB, which is the
    figure 7 prints.
    """
    page_size_per_kv_packing = page_size // kv_packing
    return dict(
        # bkv_x2_ref [2, sbs, bkv_p*PSZ/kv_packing, kv_packing, lkv_dim] u8
        bkv=2 * sbs * bkv_p * page_size_per_kv_packing * kv_packing * lkv_dim,
        # bq_x2_ref [2, sbs, tq, H, D] fp8
        bq=2 * sbs * bq_sz * num_q_heads * head_dim * q_itemsize,
        # bq_weights_x2_ref [2, sbs, tq, H] f32
        bqw=2 * sbs * bq_sz * num_q_heads * w_itemsize,
        # scores_block_x2_ref [2, sbs*tq, bkv_sz/128, 128] f32
        bo=2 * sbs * bq_sz * bkv_p * page_size * score_itemsize,
    )


def einsum_temp_bytes(*, sbs, bq_sz, bkv_p, page_size, num_q_heads):
    """The two UNDECLARED f32 intermediates inside `compute_scores`.

    design/measurement_and_design.md 7 counts only the declared scratch above.
    It omits these, and they are not a rounding term: `s` is the raw einsum
    output BEFORE the head reduction, so it carries an extra factor of H = 32
    over everything else in the kernel.

        s       [sbs, tq*H, bkv_sz] f32   the einsum output
        s_summed[sbs, tq,   bkv_sz] f32   after sum over H

    At the incumbent (B) MIXED tile that is 16.0 MiB against 2.53 MiB of
    declared scratch -- i.e. the omitted term is 6x the counted one.

    THIS IS AN UPPER BOUND, NOT AN OCCUPANCY. It was written on the reasoning
    that Mosaic is an array IR with no cross-op loop fusion, so `s` must be
    materialised whole. `measured (j-5136b90a-9f9e-4144-b066)` says otherwise:
    on a chip with 64.0 MiB of VMEM
    (`pltpu.get_tpu_info().vmem_capacity_bytes`, itself
    `measured (j-fb58ac5a-b339-43bb-810f)` and twice the ~31 MiB
    design/measurement_and_design.md 7 assumes), tiles priced here at 289.25
    and 74.62 MiB both compiled and produced correct output. Mosaic tiles the
    einsum. So `pallas_mosaic_tpu.supported_on` screens on `scratch_bytes`
    alone and this function is reported, not charged. It is kept because the
    ratio is the interesting part: it says which tiles will be the ones to
    fail first if the einsum ever stops being tiled.
    """
    bkv_sz = bkv_p * page_size
    return dict(
        s=sbs * bq_sz * num_q_heads * bkv_sz * 4,
        s_summed=sbs * bq_sz * bkv_sz * 4,
    )


def vmem_bytes(**kw):
    """Total VMEM a scoring launch needs: declared scratch + einsum temps."""
    parts = dict(scratch_bytes(**kw))
    parts.update(einsum_temp_bytes(
        sbs=kw["sbs"], bq_sz=kw["bq_sz"], bkv_p=kw["bkv_p"],
        page_size=kw["page_size"], num_q_heads=kw["num_q_heads"]))
    parts["total"] = sum(parts.values())
    return parts


def is_legal_tile(page_size, bkv_p):
    """streamindex_topk.py:596-601: bkv_sz must be a multiple of 128."""
    return (page_size * bkv_p) % LANES == 0
