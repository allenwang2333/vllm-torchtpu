# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV-cache quantization in pure JAX (reference path).

TurboQuant (TQ) stores each (token, KV head) as one packed ``uint8`` slot:

    [key_packed | value_data | value_scale f16 | value_zero f16 | pad]

* Keys, MSE presets: ``k / ||k||`` is rotated by an orthonormal Hadamard
  matrix ``H`` and every coordinate is replaced by the index of its nearest
  Lloyd-Max centroid (3 or 4 bits), followed by ``||k||`` as fp16.
* Keys, ``k8v4``: plain ``float8_e4m3fn`` bytes, no rotation.
* Values: per-vector uniform (asymmetric) quantization to 3 or 4 bits.

The byte layout matches upstream vLLM's Triton store kernel
(``vllm/v1/attention/ops/triton_turboquant_store.py``) so that a slot written
here dequantizes to the same values there, and vice versa.

Because ``H`` is orthonormal, ``q . k == (q H) . (k H)``: attention runs in
the rotated space by rotating the query once instead of inverse-rotating
every cached key.

This module is the correctness reference for the TPU backend. Attention is
computed by dequantizing the pages referenced by the batch into a bf16 page
pool and calling an existing ragged-paged-attention kernel, so it saves HBM
*capacity* but not HBM *bandwidth*; a fused Pallas kernel is the follow-up.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

# Matches the epsilons in vLLM's Triton store/decode kernels.
_KEY_NORM_EPS = 1e-8
_CENTROID_NORM_EPS = 1e-16
_VALUE_SCALE_MIN = 1e-8


@dataclasses.dataclass(frozen=True)
class TQLayout:
    """Byte geometry of one TurboQuant slot (one token, one KV head).

    Mirrors ``vllm.model_executor.layers.quantization.turboquant.config.
    TurboQuantConfig`` without depending on torch, so the JAX path can be
    traced and unit-tested on CPU.
    """

    head_dim: int
    key_bits: int  # 8 = FP8 keys; 3 or 4 = Lloyd-Max MSE keys.
    value_bits: int  # 3 or 4.
    norm_correction: bool

    def __post_init__(self):
        if self.key_bits not in (3, 4, 8):
            raise ValueError(f"key_bits must be 3, 4 or 8, got {self.key_bits}")
        if self.value_bits not in (3, 4):
            raise ValueError(f"value_bits must be 3 or 4, got {self.value_bits}")
        if self.head_dim % 8 != 0:
            raise ValueError(
                f"head_dim must be a multiple of 8 for bit packing, got {self.head_dim}"
            )
        if not self.key_fp8 and self.head_dim & (self.head_dim - 1):
            raise ValueError(
                "MSE TurboQuant keys need a power-of-two head_dim for the "
                f"Hadamard rotation, got {self.head_dim}"
            )

    @classmethod
    def from_tq_config(cls, tq_config) -> TQLayout:
        return cls(
            head_dim=tq_config.head_dim,
            key_bits=tq_config.key_quant_bits,
            value_bits=tq_config.value_quant_bits,
            norm_correction=tq_config.norm_correction,
        )

    @property
    def key_fp8(self) -> bool:
        return self.key_bits == 8

    @property
    def key_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.key_bits / 8)

    @property
    def key_packed_size(self) -> int:
        if self.key_fp8:
            return self.head_dim
        return self.key_mse_bytes + 2  # + fp16 norm

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_bits / 8)

    @property
    def value_packed_size(self) -> int:
        return self.value_data_bytes + 4  # + fp16 scale + fp16 zero

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        return self.slot_size + (self.slot_size % 2)

    def page_words_128(self, page_size: int) -> int:
        """Number of 128-byte lanes needed to hold `page_size` slots per head."""
        return math.ceil(page_size * self.slot_size_aligned / 128)

    def page_size_from_words_128(self, page_words: int) -> int:
        """Recovers `page_size` from `page_words_128(page_size)`."""
        return (page_words * 128) // self.slot_size_aligned


@dataclasses.dataclass(frozen=True, eq=False)
class TQTables:
    """Constant tables used by quantize/dequantize (host numpy arrays).

    Hashed by identity so it can be a static `jax.jit` argument.
    """

    hadamard: np.ndarray  # f32[D, D], orthonormal and symmetric.
    centroids: np.ndarray  # f32[2**bits], sorted.
    midpoints: np.ndarray  # f32[2**bits - 1], sorted.


def hadamard_matrix(d: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard matrix, as in vLLM's `_build_hadamard`."""
    if d <= 0 or d & (d - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {d}")
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return (h / math.sqrt(d)).astype(np.float32)


def make_tables(layout: TQLayout, centroids: np.ndarray) -> TQTables:
    """Builds the constant tables from the Lloyd-Max centroids."""
    c = np.sort(np.asarray(centroids, dtype=np.float32))
    return TQTables(
        hadamard=(
            hadamard_matrix(layout.head_dim)
            if not layout.key_fp8
            else np.zeros((0, 0), np.float32)
        ),
        centroids=c,
        midpoints=((c[:-1] + c[1:]) / 2).astype(np.float32),
    )


# ======================================================================================
# Bit packing helpers (little-endian bit stream, 8 elements per `bits` bytes)
# ======================================================================================


def pack_bits(idx: jax.Array, bits: int) -> jax.Array:
    """Packs `bits`-bit unsigned ints `[..., D]` into bytes `[..., D*bits/8]`.

    Element i occupies stream bits [i*bits, (i+1)*bits); bytes are
    little-endian. For 4 bits this is "low nibble first", and for 3 bits it is
    "8 values per 3 bytes", both matching vLLM's Triton store kernel.
    """
    *lead, d = idx.shape
    assert d % 8 == 0
    groups = idx.astype(jnp.uint32).reshape(*lead, d // 8, 8)
    shifts = jnp.arange(8, dtype=jnp.uint32) * bits
    words = jnp.sum(groups << shifts, axis=-1, dtype=jnp.uint32)
    byte_shifts = jnp.arange(bits, dtype=jnp.uint32) * 8
    out = (words[..., None] >> byte_shifts) & 0xFF
    return out.astype(jnp.uint8).reshape(*lead, d // 8 * bits)


def unpack_bits(packed: jax.Array, bits: int, d: int) -> jax.Array:
    """Inverse of `pack_bits`: bytes `[..., D*bits/8]` -> int32 `[..., D]`."""
    *lead, nbytes = packed.shape
    assert nbytes == d // 8 * bits
    groups = packed.astype(jnp.uint32).reshape(*lead, d // 8, bits)
    byte_shifts = jnp.arange(bits, dtype=jnp.uint32) * 8
    words = jnp.sum(groups << byte_shifts, axis=-1, dtype=jnp.uint32)
    shifts = jnp.arange(8, dtype=jnp.uint32) * bits
    idx = (words[..., None] >> shifts) & ((1 << bits) - 1)
    return idx.astype(jnp.int32).reshape(*lead, d)


def _f16_to_bytes(x: jax.Array) -> jax.Array:
    """f32 `[...]` -> fp16 little-endian bytes `[..., 2]`."""
    u16 = jax.lax.bitcast_convert_type(x.astype(jnp.float16), jnp.uint16)
    return jnp.stack([u16 & 0xFF, u16 >> 8], axis=-1).astype(jnp.uint8)


def _bytes_to_f32(b: jax.Array) -> jax.Array:
    """fp16 little-endian bytes `[..., 2]` -> f32 `[...]`."""
    u16 = b[..., 0].astype(jnp.uint16) | (b[..., 1].astype(jnp.uint16) << 8)
    return jax.lax.bitcast_convert_type(u16, jnp.float16).astype(jnp.float32)


# ======================================================================================
# Quantize / dequantize
# ======================================================================================


def rotate(x: jax.Array, hadamard: jax.Array) -> jax.Array:
    """`x @ H` in f32 at full precision (the TPU default would use bf16 passes)."""
    return jnp.einsum(
        "...d,de->...e",
        x.astype(jnp.float32),
        hadamard,
        precision=jax.lax.Precision.HIGHEST,
    )


def quantize_kv(
    key: jax.Array,  # [T, H, D]
    value: jax.Array,  # [T, H, D]
    layout: TQLayout,
    tables: TQTables,
) -> jax.Array:
    """Quantizes K/V into packed TurboQuant slots `u8[T, H, slot_size_aligned]`."""
    assert key.shape == value.shape and key.shape[-1] == layout.head_dim

    if layout.key_fp8:
        # Saturate: an out-of-range cast to e4m3fn (no inf) would give NaN.
        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
        k_fp8 = jnp.clip(key.astype(jnp.float32), -fp8_max, fp8_max).astype(
            jnp.float8_e4m3fn
        )
        key_bytes = jax.lax.bitcast_convert_type(k_fp8, jnp.uint8)
    else:
        k = key.astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True))
        y = rotate(k / (norm + _KEY_NORM_EPS), jnp.asarray(tables.hadamard))
        # Same result as the Triton binary search: count of midpoints <= y.
        midpoints = jnp.asarray(tables.midpoints)
        idx = jnp.sum(y[..., None] >= midpoints, axis=-1)
        key_bytes = jnp.concatenate(
            [pack_bits(idx, layout.key_bits), _f16_to_bytes(norm)[..., 0, :]],
            axis=-1,
        )

    v = value.astype(jnp.float32)
    levels = (1 << layout.value_bits) - 1
    v_min = jnp.min(v, axis=-1, keepdims=True)
    v_max = jnp.max(v, axis=-1, keepdims=True)
    scale = (v_max - v_min) / levels
    scale = jnp.where(scale > _VALUE_SCALE_MIN, scale, _VALUE_SCALE_MIN)
    # (v - min) >= 0, so floor matches Triton's truncating int cast.
    v_idx = jnp.clip(jnp.floor((v - v_min) / scale + 0.5), 0, levels)
    value_bytes = jnp.concatenate(
        [
            pack_bits(v_idx.astype(jnp.int32), layout.value_bits),
            _f16_to_bytes(scale)[..., 0, :],
            _f16_to_bytes(v_min)[..., 0, :],
        ],
        axis=-1,
    )

    slots = jnp.concatenate([key_bytes, value_bytes], axis=-1)
    pad = layout.slot_size_aligned - layout.slot_size
    if pad:
        slots = jnp.pad(slots, [(0, 0)] * (slots.ndim - 1) + [(0, pad)])
    return slots


def dequantize_kv(
    slots: jax.Array,  # u8[..., slot_size_aligned]
    layout: TQLayout,
    tables: TQTables,
) -> tuple[jax.Array, jax.Array]:
    """Dequantizes slots into f32 `(key, value)`, each `[..., D]`.

    For MSE presets the key is returned in the *rotated* space (``k H``).
    Rotate queries with the same `H` before attending to it.
    """
    d = layout.head_dim
    if layout.key_fp8:
        k_fp8 = jax.lax.bitcast_convert_type(slots[..., :d], jnp.float8_e4m3fn)
        key = k_fp8.astype(jnp.float32)
    else:
        mse = layout.key_mse_bytes
        idx = unpack_bits(slots[..., :mse], layout.key_bits, d)
        c = jnp.asarray(tables.centroids)[idx]
        if layout.norm_correction:
            c = c * jax.lax.rsqrt(
                jnp.sum(c * c, axis=-1, keepdims=True) + _CENTROID_NORM_EPS
            )
        key = _bytes_to_f32(slots[..., mse : mse + 2])[..., None] * c

    off = layout.key_packed_size
    vdb = layout.value_data_bytes
    v_idx = unpack_bits(slots[..., off : off + vdb], layout.value_bits, d)
    scale = _bytes_to_f32(slots[..., off + vdb : off + vdb + 2])
    zero = _bytes_to_f32(slots[..., off + vdb + 2 : off + vdb + 4])
    value = v_idx.astype(jnp.float32) * scale[..., None] + zero[..., None]
    return key, value


# ======================================================================================
# Paged cache update + attention
# ======================================================================================


def _compute_local_slot_mapping(
    num_tokens: int,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Returns `(local_slot, local_page)` in the `0..N-1` gathered page table."""
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_seqs = distribution[2]
    total = cu_q_lens[num_seqs]
    cu = jnp.where(jnp.arange(max_num_seqs + 1) <= num_seqs, cu_q_lens, total)

    t = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu[1:], t, side="right").astype(jnp.int32)
    seq = jnp.minimum(seq, max_num_seqs - 1)
    q_start = cu[seq]
    q_len = cu[seq + 1] - q_start
    pos = kv_lens[seq] - q_len + (t - q_start)
    local_page = seq * pages_per_seq + jnp.clip(pos // page_size, 0, pages_per_seq - 1)
    local_slot = local_page * page_size + pos % page_size
    return jnp.where(t < total, local_slot, -1), local_page


def compute_slot_mapping(
    num_tokens: int,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; distribution[2] = num_seqs
    page_size: int,
) -> jax.Array:
    """Flat cache slot (`page * page_size + offset`) of every query token.

    Uses the same metadata the RPA kernels consume: token `t` of sequence `s`
    sits at position `kv_lens[s] - q_len[s] + (t - cu_q_lens[s])`. Padding
    tokens past the last sequence get `-1`.
    """
    local_slot, local_page = _compute_local_slot_mapping(
        num_tokens, kv_lens, page_indices, cu_q_lens, distribution, page_size
    )
    page = page_indices[local_page]
    slot = page * page_size + (local_slot % page_size)
    return jnp.where(local_slot >= 0, slot, -1)


def write_kv_cache(
    kv_cache: jax.Array,  # u8[num_pages, page_size, H, S]
    slots: jax.Array,  # u8[T, H, S]
    slot_mapping: jax.Array,  # i32[T], -1 = skip
) -> jax.Array:
    num_pages, page_size, num_heads, slot_bytes = kv_cache.shape
    flat = kv_cache.reshape(num_pages * page_size, num_heads, slot_bytes)
    idx = jnp.where(slot_mapping >= 0, slot_mapping, num_pages * page_size)
    flat = flat.at[idx].set(slots, mode="drop")
    return flat.reshape(kv_cache.shape)


def pack_pages_128(pages: jax.Array, layout: TQLayout) -> jax.Array:
    """Packs `u8[N, page_size, H, S]` into 128-byte lanes `u8[N, W, H, 128]`.

    Putting 128 bytes on the minor axis keeps `num_pages` (dim 0) on the major
    axis in TPU HBM so XLA does not insert full-pool reformat copies around
    page gathers and scatters.
    """
    n, page_size, h, slot_bytes = pages.shape
    assert slot_bytes == layout.slot_size_aligned, (slot_bytes, layout)
    page_words = layout.page_words_128(page_size)
    pad = page_words * 128 - page_size * slot_bytes
    flat = pages.transpose(0, 2, 1, 3).reshape(n, h, page_size * slot_bytes)
    if pad:
        flat = jnp.pad(flat, [(0, 0), (0, 0), (0, pad)])
    return flat.reshape(n, h, page_words, 128).transpose(0, 2, 1, 3)


def unpack_pages_128(raw_pages: jax.Array, layout: TQLayout) -> jax.Array:
    """Inverse of `pack_pages_128`: `u8[N, W, H, 128]` -> `u8[N, page_size, H, S]`."""
    n, page_words, h, lane = raw_pages.shape
    assert lane == 128, lane
    slot_bytes = layout.slot_size_aligned
    page_size = layout.page_size_from_words_128(page_words)
    flat = raw_pages.transpose(0, 2, 1, 3).reshape(n, h, page_words * 128)
    flat = flat[:, :, : page_size * slot_bytes]
    return flat.reshape(n, h, page_size, slot_bytes).transpose(0, 2, 1, 3)


def merge_kv_pages(k: jax.Array, v: jax.Array) -> jax.Array:
    """Packs `[N, page_size, H, D]` K/V into the RPA v3 page layout.

    Returns `[N, page_size, 2H/packing, packing, align(D, 128)]` with K and V
    interleaved per head, the same layout `PallasAttentionBackend` allocates.
    """
    from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import merge_kv

    n, page_size, h, d = k.shape
    merged = merge_kv(k.reshape(n * page_size, h, d), v.reshape(n * page_size, h, d))
    return merged.reshape(n, page_size, *merged.shape[1:])


def tq_ragged_paged_attention(
    queries: jax.Array,  # [T, Hq, D]
    keys: jax.Array,  # [T, Hk, D]
    values: jax.Array,  # [T, Hk, D]
    kv_cache: jax.Array,  # u8[num_pages, page_size, Hk, S] or u8[num_pages, W, Hk, 128]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    layout: TQLayout,
    tables: TQTables,
    rpa_fn: Callable,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Ragged paged attention over a TurboQuant KV cache.

    1. Quantizes this step's K/V and scatters it into `kv_cache`.
    2. Dequantizes every page referenced by `page_indices` into a bf16 page
       pool laid out for the RPA kernel (pages remapped to `0..N-1`).
    3. Calls `rpa_fn` (an RPA-v3-compatible kernel) on that pool. The kernel
       also writes this step's *unquantized* K/V into the temporary pool, so
       current tokens attend at full precision and history at TQ precision,
       like upstream's prefill path.

    Peak extra memory is the bf16 pool for `max_num_seqs * pages_per_seq`
    pages, i.e. this is a correctness reference, not a memory-optimal path.

    Returns `(new_kv_cache, outputs)`.
    """
    num_pages, dim1, num_kv_heads, minor_bytes = kv_cache.shape
    slot_bytes = layout.slot_size_aligned
    num_tokens = queries.shape[0]
    compute_dtype = queries.dtype

    if minor_bytes == slot_bytes:
        page_size = dim1
        if not skip_kv_update:
            slot_mapping = compute_slot_mapping(
                num_tokens, kv_lens, page_indices, cu_q_lens, distribution, page_size
            )
            kv_cache = write_kv_cache(
                kv_cache, quantize_kv(keys, values, layout, tables), slot_mapping
            )
        gathered = kv_cache[page_indices]  # [N, page_size, Hk, S]
    else:
        assert minor_bytes == 128, (minor_bytes, layout)
        page_size = layout.page_size_from_words_128(dim1)
        gathered = unpack_pages_128(kv_cache[page_indices], layout)
        if not skip_kv_update:
            slots = quantize_kv(keys, values, layout, tables)
            local_slot, local_page = _compute_local_slot_mapping(
                num_tokens, kv_lens, page_indices, cu_q_lens, distribution, page_size
            )
            n_pages = page_indices.shape[0]
            g_flat = gathered.reshape(n_pages * page_size, num_kv_heads, slot_bytes)
            idx = jnp.where(local_slot >= 0, local_slot, n_pages * page_size)
            g_flat = g_flat.at[idx].set(slots, mode="drop")
            gathered = g_flat.reshape(n_pages, page_size, num_kv_heads, slot_bytes)
            g_raw = pack_pages_128(gathered, layout)
            is_first_in_page = jnp.concatenate(
                [jnp.array([True]), local_page[1:] != local_page[:-1]]
            )
            write_page = (local_slot >= 0) & is_first_in_page
            valid_p = jnp.where(write_page, page_indices[local_page], num_pages)
            kv_cache = kv_cache.at[valid_p].set(g_raw[local_page], mode="drop")

    hist_k, hist_v = dequantize_kv(gathered, layout, tables)
    pool = merge_kv_pages(hist_k.astype(compute_dtype), hist_v.astype(compute_dtype))
    pool_page_indices = jnp.arange(page_indices.shape[0], dtype=page_indices.dtype)

    if layout.key_fp8:
        q_in, k_in = queries, keys
    else:
        h = jnp.asarray(tables.hadamard)
        q_in = rotate(queries, h).astype(compute_dtype)
        k_in = rotate(keys, h).astype(compute_dtype)

    outputs, _ = rpa_fn(
        q_in,
        k_in,
        values.astype(compute_dtype),
        pool,
        kv_lens,
        pool_page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        skip_kv_update=skip_kv_update,
    )
    return kv_cache, outputs
