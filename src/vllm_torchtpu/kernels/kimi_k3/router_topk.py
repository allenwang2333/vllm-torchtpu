# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batch-1 Kimi-K3 router selection in one Mosaic/Pallas kernel."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_NUM_EXPERTS = 896
_TOPK = 16
_TPU_TILE_ROWS = 8
_TPU_TILE_COLS = 128


def _router_topk_kernel(logits_ref, bias_ref, weights_ref, ids_ref):
    """Fuse sigmoid, correction bias, exact selection, and normalization.

    The 896 scores remain in VMEM while repeated argmax removes one winner at
    a time. This is more work than a sorting network, but avoids materializing
    a full sort and folds the surrounding router operations into the same
    kernel. Random finite inputs match ``lax.top_k`` exactly; ties follow
    argmax's first-index rule, while ``top_k`` does not promise a tie order.
    """
    logits = logits_ref[...].astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias_ref[...].astype(jnp.float32)
    expert_iota = jax.lax.broadcasted_iota(jnp.int32, choice.shape, 1)

    selected_weights = []
    selected_ids = []
    for _ in range(_TOPK):
        expert_id = jnp.argmax(choice, axis=1)
        is_selected = expert_iota == expert_id[:, None]
        # Mosaic cannot dynamically gather across the seven 128-wide source
        # vregs. The one-hot reduction is equivalent because exactly one
        # expert is selected per row, and keeps the access statically tiled.
        weight = jnp.sum(jnp.where(is_selected, scores, 0.0), axis=1)
        selected_weights.append(weight)
        selected_ids.append(expert_id)
        choice = jnp.where(is_selected, -jnp.inf, choice)

    weights = jnp.stack(selected_weights, axis=1)
    weights /= jnp.maximum(weights.sum(axis=1, keepdims=True), 1e-20)
    ids = jnp.stack(selected_ids, axis=1).astype(jnp.int32)

    output_padding = ((0, 0), (0, _TPU_TILE_COLS - _TOPK))
    weights_ref[...] = jnp.pad(weights, output_padding)
    ids_ref[...] = jnp.pad(ids, output_padding)


def kimi_k3_router_topk(
    logits: jax.Array,
    correction_bias: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return normalized sigmoid weights and exact top-16 expert IDs.

    This kernel intentionally supports only the measured Kimi-K3 decode shape.
    Padding the token dimension gives Mosaic a native 8x128 tile while
    keeping the seven 128-wide expert sublanes contiguous.
    """
    if logits.ndim != 2 or logits.shape != (1, _NUM_EXPERTS):
        raise ValueError(f"expected logits shape (1, {_NUM_EXPERTS}), got "
                         f"{logits.shape}")
    if correction_bias.shape != (_NUM_EXPERTS, ):
        raise ValueError("expected correction bias shape "
                         f"({_NUM_EXPERTS},), got {correction_bias.shape}")

    padded_logits = jnp.pad(logits, ((0, _TPU_TILE_ROWS - 1), (0, 0)),
                            constant_values=-jnp.inf)
    padded_bias = jnp.broadcast_to(correction_bias[None, :],
                                   padded_logits.shape)
    out_shape = (
        jax.ShapeDtypeStruct((_TPU_TILE_ROWS, _TPU_TILE_COLS), jnp.float32),
        jax.ShapeDtypeStruct((_TPU_TILE_ROWS, _TPU_TILE_COLS), jnp.int32),
    )
    weights, ids = pl.pallas_call(
        _router_topk_kernel,
        out_shape=out_shape,
        name="kimi_k3_router_topk",
        interpret=interpret,
    )(padded_logits, padded_bias)
    return weights[:1, :_TOPK], ids[:1, :_TOPK]
