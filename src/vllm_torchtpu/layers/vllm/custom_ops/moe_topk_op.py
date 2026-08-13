# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridge for Kimi K3's exact 896-way router top-k."""

import jax
import torch
from torch_tpu._internal import pallas


_K3_NUM_EXPERTS = 896
_K3_TOPK = 16


def _k3_exact_topk_jax(scores: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Select K3's top-16 with TPU's exact approx_max_k implementation.

    Despite the primitive name, recall_target=1.0 requests exact selection.
    This is the same TPU-specific path used by StreamIndex after its authors
    measured it faster than lax.top_k. Keeping scoring and renormalization in
    the enclosing compiled graph avoids changing model numerics around the
    selection itself.
    """
    if scores.shape[-1] != _K3_NUM_EXPERTS:
        raise ValueError(f"expected {_K3_NUM_EXPERTS} scores, got {scores.shape}")
    return jax.lax.approx_max_k(
        scores,
        _K3_TOPK,
        reduction_dimension=-1,
        recall_target=1.0,
    )


k3_exact_topk = pallas.jax_op("pallas::k3_exact_topk",
                              _k3_exact_topk_jax)


@k3_exact_topk.register_fake
def _fake_k3_exact_topk(scores: torch.Tensor):
    shape = (*scores.shape[:-1], _K3_TOPK)
    return (torch.empty(shape, dtype=scores.dtype, device=scores.device),
            torch.empty(shape, dtype=torch.int32, device=scores.device))
