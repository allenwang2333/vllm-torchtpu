# SPDX-License-Identifier: Apache-2.0
"""Unpack KDA's conv/ssm state out of a unified-block-pool page.

Two cache layouts reach a KDA layer, and the layer must accept both.

*Per-layer caches* (today's default): the runner hands the layer a
2-tuple ``(sconv_cache, recurrent_cache)`` already shaped by
``get_state_shape()``. Nothing to do.

*Unified block pool* (what prefix caching requires): the runner hands
the layer ONE raw byte tensor per layer and expects the layer to carve
its own views out of it. Upstream states the contract in
``gpu_model_runner.py``::

    kv_caches[layer_name] = raw[:num_blocks * page_size_bytes].view(
        num_blocks, 1, 1, page_size_bytes)
    # the layer's bind_kv_cache unpacks each block's bytes into its
    # conv/ssm state views

``attention.py`` does ``sconv_cache, recurrent_cache = self.kv_cache``,
which raises against that single tensor -- the blocker this module
removes.

Region order matches the pool sizing in ``gdn_pool_layout.py``: the ssm
region occupies whole token rows from offset 0, the conv region follows
it. Keeping the two in agreement matters -- the sizing code decides how
many tokens a page must hold, and this code decides where to read them
from. If they disagree the state silently aliases.

The layout math is GDN's, unchanged, because it is dimension-agnostic --
it consumes byte counts, not shapes. KDA's regions map onto it directly:

    ssm_bytes  = num_heads * head_dim * head_dim   * itemsize
    conv_bytes = (conv_size - 1) * 3 * num_heads * head_dim * itemsize

against GDN's ``n_v * d_k * d_v`` and ``(kernel_size - 1) * conv_dim``.
The factor of 3 is KDA's fused q/k/v short convolution.
"""
from __future__ import annotations

import math

import torch

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def is_pooled(kv_cache) -> bool:
    """True when the runner handed this layer one pool page, not a pair."""
    if kv_cache is None:
        return False
    if isinstance(kv_cache, torch.Tensor):
        return True
    return len(kv_cache) == 1


def unpack(kv_cache, conv_shape, ssm_shape, conv_dtype, ssm_dtype):
    """Return ``(sconv_cache, recurrent_cache)`` for either layout.

    The per-layer pair passes through untouched, so enabling the pool
    cannot change behaviour on the path that does not use it.
    """
    if not is_pooled(kv_cache):
        return kv_cache[0], kv_cache[1]

    pool = kv_cache if isinstance(kv_cache, torch.Tensor) else kv_cache[0]

    ssm_bytes = _nbytes(ssm_shape, ssm_dtype)
    conv_bytes = _nbytes(conv_shape, conv_dtype)

    # A pool page is (num_blocks, block_tokens, *payload). The regions are
    # placed on whole TOKEN ROWS, not on the page as a whole -- GDN derives
    # the row size the same way at gdn_attention.py:733. Conflating the two
    # makes ssm_bytes %% token_bytes non-zero for every real shape and the
    # layout raises "must occupy complete FA token rows", which is what a
    # first draft of this module did.
    if pool.dim() < 3:
        raise ValueError(
            f"pool page must be (blocks, tokens, *payload); got {tuple(pool.shape)}")
    block_tokens = pool.shape[1]
    token_bytes = math.prod(pool.shape[2:]) * pool.element_size()
    page_bytes = block_tokens * token_bytes

    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
    )
    if layout.required_tokens > block_tokens:
        raise ValueError(
            f"KDA state does not fit one pool page: needs "
            f"{layout.required_tokens} token rows ({ssm_bytes} ssm + "
            f"{conv_bytes} conv bytes over {token_bytes}-byte rows) but the "
            f"page holds {block_tokens}.")

    nblocks = pool.shape[0]
    flat = pool.reshape(nblocks, -1).view(torch.uint8)

    ssm_off = 0
    conv_off = layout.ssm_tokens * token_bytes

    ssm = (flat[:, ssm_off:ssm_off + ssm_bytes]
           .view(ssm_dtype).reshape(nblocks, *ssm_shape))
    conv = (flat[:, conv_off:conv_off + conv_bytes]
            .view(conv_dtype).reshape(nblocks, *conv_shape))
    return conv, ssm
