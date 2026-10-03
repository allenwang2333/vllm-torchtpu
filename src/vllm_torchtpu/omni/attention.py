# SPDX-License-Identifier: Apache-2.0
"""Diffusion attention backend for TPU execution.

Dense, unmasked, non-causal self-attention -- what every DiT block of models
like Z-Image, Flux or Wan runs -- goes to a Pallas splash-attention kernel;
everything else (masks, causal, GQA, odd sequence lengths, tiny sequences)
falls back to PyTorch SDPA.

Why not plain SDPA: torch_tpu lowers ``scaled_dot_product_attention`` to a
Pallas custom call that pads the sequence up to a coarse tile (4128 -> 5120,
+54% FLOPs) and whose tiling degrades when the length is not a power of two.
For Z-Image-Turbo at 1024x1024 (4096 image + N caption tokens -> S=4128 with
a 32-token caption, 30 heads x 128, bf16) one call costs ~1.92 ms on tpu7x.
Splash attention with ``block_q=1408`` and the full K/V resident runs S=4224
in ~1.14 ms (vs 2.24 ms SDPA at that length, -49%); at S=4096 it is 1.23 ->
1.00 ms (-19%). That is 30+ calls per denoise step.

Sequence lengths that are not lane aligned (the caption length varies per
prompt) are zero-padded to the next multiple of 128 that admits a large q
block, and the padded keys are masked out inside the kernel, so one compiled
kernel serves every caption length in a 128-token bucket.
"""

import functools
from typing import Any

import torch
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Below this many tokens the per-call overhead dominates and SDPA is as fast.
SPLASH_MIN_SEQ_LEN = 1024
# Keep the whole K/V for one head resident in VMEM up to this length
# (8192 x 128 x bf16 x {K,V} = 4 MiB); longer sequences stream K/V in chunks.
SPLASH_MAX_RESIDENT_KV = 8192
_Q_BLOCKS = (2048, 1536, 1408, 1024, 768, 512, 384, 256, 128)
_KV_BLOCKS = (4096, 3072, 2048, 1536, 1408, 1024, 512, 384, 256, 128)
_KV_COMPUTE_BLOCKS = (1536, 1408, 1024, 512, 384, 256, 128)
# Cap on block_q * block_kv_compute (fp32 score tile): 2048 x 1024 x 4 B = 8 MiB
# fits tpu7x's 64 MiB VMEM with the kernel's double buffering; block_q = S
# (4096/4224) with 1024-1408 compute blocks ran out of VMEM.
_MAX_SCORE_TILE_ELEMS = 2048 * 1024


def _first_divisor(candidates: tuple[int, ...], n: int) -> int | None:
    for c in candidates:
        if c <= n and n % c == 0:
            return c
    return None


# With a padding mask the kernel materialises a (block_q x block_kv_compute)
# mask tile per chunk; narrower chunks were ~7% faster there (384 vs 1408 at
# S=4224 on tpu7x), while the unmasked kernel prefers the widest chunk.
_MASKED_KV_COMPUTE_BLOCKS = (512, 384)


def splash_block_sizes(
    seq_len: int, masked: bool = False
) -> tuple[int, int, int] | None:
    """Pick ``(block_q, block_kv, block_kv_compute)`` for a sequence length.

    Measured on tpu7x for S=4224 (Z-Image 1024^2 + caption): the largest
    q block that divides S (1408) with the full K/V resident and 1408-wide
    compute chunks beat SDPA by ~49%; block_q=384 got 40%, 128 only 5%, and
    streaming K/V in <=1536 chunks 12-17%. Returns ``None`` when no
    lane-aligned block divides ``seq_len`` (the caller then uses SDPA).
    ``masked`` selects the chunk preference for kernels with a padding mask.
    """
    block_q = _first_divisor(_Q_BLOCKS, seq_len)
    if block_q is None:
        return None
    if seq_len <= SPLASH_MAX_RESIDENT_KV:
        block_kv = seq_len
    else:
        block_kv = _first_divisor(_KV_BLOCKS, seq_len)
        if block_kv is None:
            return None
    max_compute = _MAX_SCORE_TILE_ELEMS // block_q
    compute_candidates = tuple(c for c in _KV_COMPUTE_BLOCKS if c <= max_compute)
    if masked:
        compute_candidates = _MASKED_KV_COMPUTE_BLOCKS + compute_candidates
    block_kv_compute = _first_divisor(compute_candidates, block_kv)
    if block_kv_compute is None:
        return None
    return block_q, block_kv, block_kv_compute


# Pad the sequence to a multiple of the lane width; try a few buckets so a
# length like 8320 (=65*128, only block_q=128) can use 8448 (=6*1408) instead.
_SPLASH_ALIGN = 128
_SPLASH_PAD_BUCKETS = 4
# Small q blocks waste the MXU; below this the kernel does not beat SDPA.
_SPLASH_MIN_BLOCK_Q = 1024


def splash_plan(seq_len: int) -> tuple[int, tuple[int, int, int]] | None:
    """Return ``(padded_len, block_sizes)`` for ``seq_len`` or ``None``.

    ``padded_len`` is the smallest multiple of 128 >= ``seq_len`` (within a
    few buckets) that admits a q block of at least ``_SPLASH_MIN_BLOCK_Q``;
    padding is bounded by the bucket search (at most 4 x 128 extra tokens).
    """
    first = -(-seq_len // _SPLASH_ALIGN) * _SPLASH_ALIGN
    for i in range(_SPLASH_PAD_BUCKETS):
        padded = first + i * _SPLASH_ALIGN
        blocks = splash_block_sizes(padded, masked=padded != seq_len)
        if blocks is not None and blocks[0] >= _SPLASH_MIN_BLOCK_Q:
            return padded, blocks
    return None


def can_use_splash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any | None,
    causal: bool,
) -> bool:
    """True for dense bf16 self-attention on TPU that the splash kernel serves.

    Tensors are ``[batch, seq, heads, head_dim]`` (vLLM-Omni layout).
    """
    if causal or getattr(attn_metadata, "attn_mask", None) is not None:
        return False
    if query.device.type != "tpu" or query.dtype != torch.bfloat16:
        return False
    if query.dim() != 4 or query.shape != key.shape or key.shape != value.shape:
        return False
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        return False
    seq_len, head_dim = query.shape[1], query.shape[3]
    if seq_len < SPLASH_MIN_SEQ_LEN or head_dim % 128 != 0:
        return False
    return splash_plan(seq_len) is not None


def _padding_segment_ids(q: Any, k: Any, valid_len: int) -> Any:
    """Splash ``SegmentIds`` that hide padded keys (index >= ``valid_len``).

    ``q``/``k`` are one batch element, ``[heads, padded_len, head_dim]``.
    Queries are all segment 0, keys are segment 0 if valid and 1 if padding,
    so every (valid or padded) query attends exactly the valid keys: no fully
    masked row, hence no NaNs; padded query rows are sliced off by the caller.

    The ids are derived from the activations (``x * 0``) instead of being
    constants on purpose. Splash's computable-mask path (and a constant
    segment-id vector alike) gives the kernel a ``[padded_len, 128]`` int32
    index operand that XLA constant-folds, CSEs across all 30 layers and pins
    in VMEM for the whole denoise-step program. On tpu7x that single 2 MiB
    resident buffer made every FFN-down matmul fusion pick a tiling with half
    the VMEM budget (0.59 -> 1.30 ms per layer, +21 ms per step), which
    erased the attention gain. A value computed from ``q``/``k`` is rebuilt
    per layer and is dead right after its kernel call.
    """
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash_kernel,
    )

    padded_len = q.shape[1]
    zero_q = (q[0, :, 0] * 0).astype(jnp.int32)
    zero_k = (k[0, :, 0] * 0).astype(jnp.int32)
    is_pad = jnp.arange(padded_len, dtype=jnp.int32) >= valid_len
    return splash_kernel.SegmentIds(q=zero_q, kv=zero_k + is_pad.astype(jnp.int32))


@functools.cache
def _splash_attention_op(
    num_heads: int,
    seq_len: int,
    valid_len: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
) -> Any:
    """Build (once per shape) a torch custom op wrapping JAX splash attention.

    ``torch_tpu._internal.pallas.jax_op`` traces the JAX function into the
    lazy torch_tpu graph, so the kernel is compiled together with the
    surrounding ops. The op takes ``[batch, heads, seq, head_dim]`` with the
    softmax scale already folded into ``q``; keys at index >= ``valid_len``
    are masked out.
    """
    import jax
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash_kernel,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as splash_mask,
    )
    from torch_tpu._internal import pallas

    block_sizes = splash_kernel.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=None,
        block_kv_dkv=None,
        block_kv_dkv_compute=None,
        block_q_dq=None,
        block_kv_dq=None,
        use_fused_bwd_kernel=False,
    )
    head_mask = splash_mask.FullMask((seq_len, seq_len))
    mask = splash_mask.MultiHeadMask([head_mask for _ in range(num_heads)])
    kernel = splash_kernel.make_splash_mha(
        mask=mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
    )
    padded = valid_len < seq_len

    def _one_batch(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        if not padded:
            return kernel(q, k, v)
        return kernel(q, k, v, segment_ids=_padding_segment_ids(q, k, valid_len))

    def _splash_core(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return jax.vmap(_one_batch)(q, k, v)

    name = (
        f"pallas::omni_splash_attention_h{num_heads}_s{seq_len}_v{valid_len}"
        f"_q{block_q}_kv{block_kv}_c{block_kv_compute}"
    )
    op = pallas.jax_op(name, _splash_core)
    op.register_fake(lambda q, k, v: torch.empty_like(q))
    logger.info(
        "Built splash attention op for heads=%d seq=%d (valid %d) blocks=(%d, %d, %d).",
        num_heads,
        seq_len,
        valid_len,
        block_q,
        block_kv,
        block_kv_compute,
    )
    return op


class TpuSDPAImpl(SDPAImpl):
    """TPU attention for diffusion models: splash kernel with SDPA fallback."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        if can_use_splash_attention(query, key, value, attn_metadata, self.causal):
            return self._forward_splash(query, key, value)
        return self._forward_impl(
            query, key, value, attn_metadata, mask_mode="broadcast_k"
        )

    def _forward_splash(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        _, seq_len, num_heads, _ = query.shape
        plan = splash_plan(seq_len)
        assert plan is not None  # guaranteed by can_use_splash_attention
        padded_len, (block_q, block_kv, block_kv_compute) = plan
        op = _splash_attention_op(
            num_heads, padded_len, seq_len, block_q, block_kv, block_kv_compute
        )
        # [b, s, h, d] -> [b, h, s, d]; fold the softmax scale into q so one
        # op serves every layer regardless of scale.
        q = (query * self.softmax_scale).permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        pad = padded_len - seq_len
        if pad:
            q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        out = op(q.contiguous(), k.contiguous(), v.contiguous())
        if pad:
            out = out[:, :, :seq_len]
        return out.permute(0, 2, 1, 3).to(query.dtype)


class TpuSDPABackend(SDPABackend):
    """TPU attention backend for diffusion models (splash kernel + SDPA)."""

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl
