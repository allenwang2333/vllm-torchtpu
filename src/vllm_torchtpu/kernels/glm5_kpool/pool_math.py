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
"""GLM-5.3-Flash pooling math and the fp8 + inline-e8m0 record layout.

Pure ``jnp``; see ``__init__.py`` for why there is no Pallas kernel here.

Two numerics facts, documented so nobody "fixes" them into a difference:

* **The reference is less accurate than this, on purpose-by-accident.**
  ``Glm5NextTextIndexer.get_pooled_states`` computes the ``gate + ape`` add and
  the softmax in **float32** (hard-coded ``.float()`` at
  ``modeling_glm5_next.py:962``, regardless of the module dtype) and then casts
  the probabilities **down to the key dtype** -- bf16 in the shipped checkpoint
  -- before the ``index_kpool``-term sum (``:964-966``). :func:`pool_window`
  accumulates in the input dtype throughout. An exact elementwise diff against
  the reference will therefore never close; the CPU oracle
  (``tests/models/vllm/glm5_next/kpool_oracle.py``) records the measured gap
  instead.
* **The ``-inf`` / ``nan_to_num`` handling at ``:962-966`` is dead on TPU.**
  Only *complete* pools are ever written (``kpool_write``'s boundary mask is
  ``(pos + 1) % index_kpool == 0``), so no slot of a pooled row is ever
  invalid and the per-channel softmax never sees an all-``-inf`` column.
  :func:`pool_window` takes no validity mask for exactly that reason -- adding
  one would be dead code that hides the invariant.

``k`` arrives **already normalized**: ``LayerNorm(index_head_dim, eps=1e-6,
with bias)`` runs in torch before the op, faithful to
``modeling_glm5_next.py:763``. That keeps the "derive head_dim from the norm
weight" trap (``kernels/deepseek_v4/compress_and_store/kernel.py:331``) out of
this module entirely.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.glm5_kpool.expand import align_to

# ``layers/common/quantization.py:51-52``. e8m0 is a bare biased exponent:
# byte ``b`` decodes to ``2 ** (b - 127)``, and 255 is NaN, which the clamp to
# ``E8M0_MAX_EXP`` keeps out.
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)  # 448.0


def pool_window(k_rows: jnp.ndarray, gate_rows: jnp.ndarray,
                ape: jnp.ndarray) -> jnp.ndarray:
    """``[N, kpool, D], [N, kpool, D], [kpool, D] -> [N, D]``.

    ``pool_key[n, c] = sum_j softmax_j(gate[n, j, c] + ape[j, c]) * k[n, j, c]``

    The softmax is over the **slot** axis, **per channel**. Getting the axis
    wrong still produces a finite pooled key of the right shape and a model that
    generates fluent text, so this is one of the design's silent traps; the
    channel-independence of the result is asserted directly in
    ``tests/kernels/glm5_kpool/test_pool_math.py``.

    ``ape`` is ``index_kpool_compress_ape [index_kpool, index_head_dim]``,
    indexed by the token's slot *within its pool* -- ``pos % index_kpool`` under
    vLLM's unpadded ragged layout, which equals the reference's
    ``(idx - first_key) % index_kpool`` only because ``first_key == 0`` there.
    ``pool_write.kpool_write`` asserts that rather than assuming it.
    """
    assert k_rows.ndim == 3, k_rows.shape
    num_pools, index_kpool, head_dim = k_rows.shape
    assert gate_rows.shape == k_rows.shape, gate_rows.shape
    assert ape.shape == (index_kpool, head_dim), ape.shape

    # At least f32, never less -- the reference is explicit: ``logits =
    # grouped_gate_scores.float() + index_kpool_compress_ape.float()[None,
    # None]``, softmax in f32, and only the *probabilities* cast back down
    # (``modeling_glm5_next.py:962-966``). A bf16 softmax over 4 slots per
    # channel is finite, correctly shaped and fluent -- it just reshuffles the
    # k/k+1 near-ties the accuracy gate measures, and the blame lands on fp8
    # quantization instead of here.
    #
    # ``promote_types`` and not a bare ``.astype(float32)``: the f64 oracle
    # tests would otherwise be silently rounded to f32 by the thing they are
    # meant to be checking, which converts a precision test into a tautology.
    acc = jnp.promote_types(jnp.promote_types(gate_rows.dtype, ape.dtype),
                            jnp.float32)
    logits = gate_rows.astype(acc) + ape[None].astype(acc)
    probs = jax.nn.softmax(logits, axis=1).astype(k_rows.dtype)
    return jnp.sum(probs * k_rows, axis=1)


def quantize_fp8_ue8m0_rows(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Row-wise fp8_e4m3 values + one e8m0 scale byte per row.

    A ``jnp`` transcription of ``quantize_tensor(x, torch.float8_e4m3fn,
    axis=-1, use_ue8m0=True)`` (``layers/common/quantization.py:217-303``),
    which is what the GLM-5.2 producer already feeds ``streamindex_topk``
    (``layers/vllm/custom_ops/mla_attention_op.py:156-166``). It is transcribed
    rather than called because the whole write path is jnp and must
    run under ``jax.jit``; ``test_pool_math.py`` asserts the two are
    **bit-identical** so the transcription cannot drift.

    Returns:
        ``(values fp8_e4m3fn[..., D], scale_bits uint8[..., 1])``.
    """
    abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32)
    scale = abs_max * jnp.float32(1.0 / FP8_E4M3_MAX)
    # e8m0 carries only an exponent, so snap UP to a power of two: a smaller
    # scale pushes the row's largest magnitude past 448 where the clamp below
    # would silently clip it. log2 is not exactly rounded, so repair the octave
    # rather than trusting it. An all-zero row gives log2(0) = -inf; exp2(-inf)
    # == 0 is not < 0, so it falls through to the clamp.
    scale_exp = jnp.ceil(jnp.log2(scale))
    scale_exp = jnp.clip(
        jnp.where(jnp.exp2(scale_exp) < scale, scale_exp + 1, scale_exp),
        E8M0_MIN_EXP, E8M0_MAX_EXP)
    scale = jnp.exp2(scale_exp)
    scale_bits = (scale_exp - E8M0_MIN_EXP).astype(jnp.uint8)

    scale_inv = jnp.where(scale == 0, jnp.float32(0.0), 1.0 / scale)
    values = jnp.clip(
        x.astype(jnp.float32) * scale_inv, -FP8_E4M3_MAX,
        FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
    return values, scale_bits


def pack_fp8_ue8m0(pooled: jnp.ndarray) -> jnp.ndarray:
    """``f32[N, D] -> uint8[N, D + 1]``, the record ``streamindex_topk`` reads.

    Exactly the layout the kernel bitcasts at
    ``kernels/deepseek_v4/streamindex_topk.py:264-271``: ``D`` fp8_e4m3 value
    bytes, then **one** e8m0 scale byte at offset ``D``.

    The row is deliberately *not* padded to the 128-multiple the cache page
    uses; ``update_sparse_mla_kv_cache``-style scatter code owns that padding
    (``kernels/mla/kv_cache_utils.py:82-98``), and duplicating it here is how a
    geometry derivation gets to disagree with itself.
    """
    assert pooled.ndim == 2, pooled.shape
    values, scale_bits = quantize_fp8_ue8m0_rows(pooled)
    return jnp.concatenate(
        [jax.lax.bitcast_convert_type(values, jnp.uint8), scale_bits], axis=-1)


def pooled_record_bytes(index_head_dim: int) -> int:
    """Bytes per pooled record: fp8 values + one e8m0 scale, 128-aligned.

    ``align_to(128 + 1, 128) == 256`` at GLM-5.3-Flash's ``index_head_dim``.
    The +1 is the scale byte, not padding: dropping it would overlap the next
    record's first value, which decodes as a finite fp8 number and merely
    ranks pools wrong.
    """
    return align_to(index_head_dim + 1, 128)


#: Records per minor-axis group in the packed pooled cache. The allocation is
#: ``(num_blocks, page_size // kv_packing, kv_packing, record_bytes)``, which
#: is what ``streamindex_topk`` unpacks as
#: ``page_size = page_size_per_kv_packing * kv_packing``. It must match the
#: runner's ``_DS_V4_KV_PACKING``; the indexer asserts that tie so the two
#: cannot drift.
POOLED_CACHE_KV_PACKING = 4


def unpack_fp8_ue8m0(record: jnp.ndarray, head_dim: int) -> jnp.ndarray:
    """``uint8[N, >= D + 1] -> f32[N, D]``. The inverse of :func:`pack_fp8_ue8m0`.

    Mirrors ``load_bkv`` at ``streamindex_topk.py:262-271``: bitcast the first
    ``head_dim`` bytes as fp8_e4m3, bitcast byte ``head_dim`` as e8m0, multiply.
    Exists so a test can prove the round trip without a TPU, and so the
    write path's
    cache-scatter test has something to read records back with.
    """
    assert record.ndim == 2, record.shape
    assert record.shape[1] >= head_dim + 1, record.shape
    values = jax.lax.bitcast_convert_type(record[:, :head_dim],
                                          jnp.float8_e4m3fn)
    scale_bits = record[:, head_dim:head_dim + 1].astype(jnp.int32)
    scale = jnp.exp2((scale_bits + E8M0_MIN_EXP).astype(jnp.float32))
    return values.astype(jnp.float32) * scale
