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

import functools

import jax
import jax.numpy as jnp


def ref_wkv_proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> jax.Array:
    """Golden JAX reference for Kernel 1 (proj + save state)."""
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width

    # 1. Project & APE
    # 2. Geometry setup
    _, page_size, d1, d2 = cache.shape
    slot_bytes = d1 * d2
    slots_per_token = state_dim * 4 // slot_bytes
    f32_per_slot = state_dim // slots_per_token
    bytes_per_slot = f32_per_slot * 4
    assert bytes_per_slot == slot_bytes
    tokens_per_page = page_size // slots_per_token
    assert state_block_size <= tokens_per_page, (
        f"state_block_size {state_block_size} exceeds the {tokens_per_page} "
        f"token states a {page_size}-row page holds"
    )
    return _proj_and_save_state_jit(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        slot_mapping,
        cache,
        state_width,
        state_dim,
        compress_ratio,
        slots_per_token,
        tokens_per_page,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "state_width",
        "state_dim",
        "compress_ratio",
        "slots_per_token",
        "tokens_per_page",
    ),
)
def _proj_and_save_state_jit(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    state_width: int,
    state_dim: int,
    compress_ratio: int,
    slots_per_token: int,
    tokens_per_page: int,
) -> jax.Array:
    num_tokens = hidden_states.shape[0]
    num_pages, page_size, d1, d2 = cache.shape

    kv_score = jnp.matmul(hidden_states, wkv_wgate)
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:]

    ape_rows = jnp.mod(positions, compress_ratio)
    score_state = score + ape[ape_rows]
    packed = jnp.concatenate([kv, score_state], axis=-1).astype(jnp.float32)

    # 3. Unpack inline
    packed_bytes = jax.lax.bitcast_convert_type(packed, jnp.uint8)
    packed_t = packed_bytes.reshape(num_tokens, slots_per_token, d2, d1).transpose(
        0, 1, 3, 2
    )

    # 4. Scatter
    num_tokens_flat = num_pages * tokens_per_page
    valid = slot_mapping >= 0
    slots = jnp.where(valid, slot_mapping // slots_per_token, num_tokens_flat)
    flat_cache = cache.reshape(num_tokens_flat, slots_per_token, d1, d2)

    # 5. Pack inline back
    updated_cache = flat_cache.at[slots].set(packed_t, mode="drop")
    return updated_cache.reshape(num_pages, page_size, d1, d2)
