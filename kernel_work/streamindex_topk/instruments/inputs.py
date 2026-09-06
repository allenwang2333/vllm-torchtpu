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
"""Input construction for the `streamindex_topk` instruments.

The packing here is a *mirror* of the production compressor. Mirrors drift, so
`assert_packing_matches_production()` re-derives the record width from the
shipped call site's own arithmetic and refuses to run on disagreement -- see
step 9 of "Derive the shapes" in the write-tpu-kernels skill.

The quantizer is copied from `tests/kernels/test_dsv4_streamindex_topk.py`
(`quantize_fp8_ue8m0`, L40-50 / `_ee_pack_cache`, L541-554), which is itself
copied from the compressor. Keeping the same source keeps the instrument and
the repo's own test packing bit-identical cache records.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import shapes as S

FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def _to_byte_lane(x):
    """Reinterpret the trailing axis of an 8-bit array as raw uint8 lanes."""
    return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(*x.shape[:-1], -1)


def quantize_fp8_ue8m0(x, block_size):
    """Per-block power-of-two (UE8M0) fp8 quantization.

    Mirror of tests/kernels/test_dsv4_streamindex_topk.py:40-50.
    Returns (fp8_values, ue8m0_scale_bytes, dequantized_float32).
    """
    orig_shape = x.shape
    blocks = x.reshape(*orig_shape[:-1], -1, block_size)
    amax = jnp.maximum(jnp.max(jnp.abs(blocks), axis=-1, keepdims=True), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / FP8_MAX))
    scale = jnp.exp2(exponent)
    q = jnp.clip(blocks / scale, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    # UE8M0 stores the exponent biased by 127 in a single byte.
    scale_byte = (exponent + 127.0).astype(jnp.uint8)
    dequant = q.astype(jnp.float32) * scale
    return (q.reshape(orig_shape), scale_byte.reshape(*orig_shape[:-1], -1),
            dequant.reshape(orig_shape))


def record_width(head_dim: int) -> int:
    """Bytes per compressed-KV record in `cache_kv`, including lane padding.

    Mirror of the width formula that appears three times in the repo test
    (L181-183, L316-318) and of mla_attention_op.py:230-231.
    """
    q_lkv_dim = S.align_to(head_dim, 128)
    rec = q_lkv_dim + q_lkv_dim // 128
    return S.align_to(rec, 128)


def assert_packing_matches_production():
    """Refuse to run if this mirror has drifted from the shipped constants."""
    w = record_width(S.INDEX_HEAD_DIM)
    if w != S.LKV_DIM:
        raise AssertionError(
            f"record_width({S.INDEX_HEAD_DIM}) = {w} but shapes.LKV_DIM = "
            f"{S.LKV_DIM}; the instrument's packing has drifted from "
            "mla_attention_op.py:230-231")
    if S.PAGE_SIZE_PER_KV_PACKING * S.KV_PACKING != S.PAGE_SIZE:
        raise AssertionError("page_size decomposition disagrees with "
                             "streamindex_topk.py:593")
    used = S.CACHE_RECORD_USED_BYTES
    if used != S.INDEX_HEAD_DIM + 1:
        raise AssertionError("used-byte count disagrees with the unpacker at "
                             "streamindex_topk.py:265-268")


def pack_cache(keys_f32: np.ndarray) -> tuple[jnp.ndarray, np.ndarray]:
    """Pack float keys into the shipped DSv4 fp8 cache layout.

    Args:
      keys_f32: float32[num_pages, page_size, head_dim]

    Returns:
      cache_kv: uint8[num_pages, page_size // kv_packing, kv_packing, width]
      dequantized: float32[num_pages, page_size, head_dim] -- the exact values
        the kernel will see after it unpacks, for the reference to score
        against. This is ground truth by construction, not a re-unpack.

    Vectorized; the repo test's non-early-exit paths do this in a triple Python
    loop (L324-339) which is unusable at production page counts.
    """
    num_pages, page_size, head_dim = keys_f32.shape
    width = record_width(head_dim)
    q, scale_byte, dequant = quantize_fp8_ue8m0(jnp.asarray(keys_f32),
                                                head_dim)
    val_bytes = _to_byte_lane(q)  # uint8[num_pages, page_size, head_dim]
    rec = jnp.concatenate([val_bytes, scale_byte], axis=-1)
    pad = width - rec.shape[-1]
    rec = jnp.pad(rec, ((0, 0), (0, 0), (0, pad)))
    cache = rec.reshape(num_pages, page_size // S.KV_PACKING, S.KV_PACKING,
                        width)
    return cache, np.asarray(dequant)


def build(regime: S.Regime, seed: int = 0, *, with_dequant: bool = False):
    """Construct one full set of kernel inputs for a regime.

    All sequences are given the same `kv_len`, which is the worst case for the
    decode kernel: streamindex_topk.py:275-276 sizes the KV loop by
    `max(kv_lens)` across the batch, so a ragged batch does strictly less work.
    That makes this a conservative timing input and it is stated in report.md.

    Returns a dict of the seven kernel arguments plus bookkeeping.
    """
    assert_packing_matches_production()
    S.check_legal(regime.max_model_len)
    rng = np.random.default_rng(seed)

    pps = S.pages_per_seq(regime.max_model_len)
    num_seqs = regime.max_num_seqs
    num_pages = num_seqs * pps
    num_tokens = regime.num_tokens

    if regime.phase == "decode":
        q_lens = np.ones(num_seqs, dtype=np.int32)
        num_decodes = num_seqs
        active_seqs = num_seqs
    else:
        # One prefill sequence carrying the whole chunk. This is the shape the
        # profile shows (StreamIdxTC-m-bq_128-bkvp_2 at f8e4m3fn[1024,32,128]),
        # and it maximises num_bq, which is the loop that re-reads the KV.
        q_lens = np.zeros(num_seqs, dtype=np.int32)
        q_lens[0] = num_tokens
        num_decodes = 0
        active_seqs = 1

    seq_lens = np.full(num_seqs, regime.kv_len, dtype=np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    # Disjoint, shuffled pages: paged KV is not contiguous in HBM and a
    # contiguous block table would measure a DMA pattern that never occurs.
    block_table = rng.permutation(num_pages).reshape(num_seqs, pps)
    page_indices = block_table.reshape(-1).astype(np.int32)
    # `distribution` is [num_decodes, num_decodes, num_seqs_in_batch], and the
    # third entry is the REAL sequence count, not the padded lane count -- the
    # bands at streamindex_topk.py:755-808 are
    #   DECODE batched   [0, floor(d[0]/4)*4)
    #   DECODE remainder [floor(d[0]/4)*4, d[1])
    #   MIXED            [d[1], d[2])
    # so passing `num_seqs` here for a mixed batch would make MIXED loop over
    # `num_seqs - 1` sequences with q_len == 0. Those are NOT free: `num_bq =
    # max(1, cdiv(0, bq_sz))` is 1, so each empty sequence still runs a full
    # sweep of `num_bkv` KV blocks. The first build of this file did exactly
    # that and inflated mixed KV traffic 64-fold.
    # mla_attention_op.py:183 -- the prefill-only band is empty by construction,
    # entry [1] equals entry [0].
    distribution = np.array([num_decodes, num_decodes, active_seqs],
                            dtype=np.int32)

    keys = rng.standard_normal(
        (num_pages, S.PAGE_SIZE, S.INDEX_HEAD_DIM), dtype=np.float32)
    cache_kv, dequant = pack_cache(keys)

    q_f32 = rng.standard_normal(
        (num_tokens, S.INDEX_N_HEADS, S.INDEX_HEAD_DIM), dtype=np.float32)
    q = jnp.asarray(q_f32).astype(jnp.float8_e4m3fn)
    weights = jnp.asarray(
        rng.uniform(-1.5, 1.5,
                    (num_tokens, S.INDEX_N_HEADS)).astype(np.float32))

    out = dict(
        q=q,
        indexer_weights=weights,
        cache_kv=jnp.asarray(cache_kv),
        seq_lens=jnp.asarray(seq_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
        _block_table=block_table,
        _q_lens=q_lens,
        _pages_per_seq=pps,
    )
    if with_dequant:
        out["_dequant"] = dequant
        out["_q_f32"] = np.asarray(jnp.asarray(q).astype(jnp.float32))
        out["_weights"] = np.asarray(weights)
    return out
