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

"""Full statement and branch coverage tests for project_and_save_state_ref.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    project_and_save_state_ref,
)


def _baseline_ref_wkv_proj_and_save_state(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> jax.Array:
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width

    kv_score = jnp.matmul(hidden_states, wkv_wgate)
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:]

    ape_rows = jnp.mod(positions, compress_ratio)
    score_state = score + ape[ape_rows]
    packed = jnp.concatenate([kv, score_state], axis=-1)

    num_pages, page_size, d1, d2 = cache.shape
    slot_bytes = d1 * d2
    slots_per_token = state_dim * 4 // slot_bytes
    f32_per_slot = state_dim // slots_per_token
    bytes_per_slot = f32_per_slot * 4
    assert bytes_per_slot == slot_bytes
    tokens_per_page = page_size // slots_per_token
    assert state_block_size <= tokens_per_page

    cache_t = cache.reshape(
        num_pages, tokens_per_page, slots_per_token, d1, d2
    ).transpose(0, 1, 2, 4, 3)
    cache_bitcast_shape = cache_t.reshape(num_pages * tokens_per_page, state_dim, 4)
    flat = jax.lax.bitcast_convert_type(cache_bitcast_shape, jnp.float32)

    num_tokens_flat = num_pages * tokens_per_page
    valid = slot_mapping >= 0
    slots = jnp.where(valid, slot_mapping // slots_per_token, num_tokens_flat)
    flat = flat.at[slots].set(packed, mode="drop")

    chunk_bytes = jax.lax.bitcast_convert_type(flat, jnp.uint8)
    chunk_bytes_t = chunk_bytes.reshape(
        num_pages, tokens_per_page, slots_per_token, d2, d1
    ).transpose(0, 1, 2, 4, 3)
    return chunk_bytes_t.reshape(num_pages, page_size, d1, d2)


class TestProjectAndSaveStateRefFull:
    """100% statement and branch coverage tests for project_and_save_state_ref.py."""

    @pytest.mark.parametrize("overlap", [False, True])
    def test_ref_wkv_proj_and_save_state_bit_exact_parity(self, overlap: bool):
        rng = np.random.default_rng(42 + int(overlap))
        num_tokens = 8
        hidden_size = 64
        head_dim = 128
        compress_ratio = 16
        coff = 1 + int(overlap)
        state_width = coff * head_dim
        num_pages = 4
        page_size = 16
        state_block_size = 2

        hidden_states = jnp.asarray(
            rng.standard_normal((num_tokens, hidden_size), dtype=np.float32)
        )
        wkv_wgate = jnp.asarray(
            rng.standard_normal((hidden_size, 2 * state_width), dtype=np.float32)
        )
        ape = jnp.asarray(
            rng.standard_normal((compress_ratio, state_width), dtype=np.float32)
        )
        positions = jnp.array([0, 3, 15, 16, 17, 31, 32, 5], dtype=jnp.int32)
        slots_per_token = (2 * state_width * 4) // (4 * 128)
        slot_mapping = jnp.array(
            [
                0 * slots_per_token,
                2 * slots_per_token,
                -1,
                5 * slots_per_token,
                7 * slots_per_token,
                -1,
                3 * slots_per_token,
                1 * slots_per_token,
            ],
            dtype=jnp.int32,
        )
        cache = jnp.asarray(
            rng.integers(0, 256, size=(num_pages, page_size, 4, 128), dtype=np.uint8),
            dtype=jnp.uint8,
        )

        expected = _baseline_ref_wkv_proj_and_save_state(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            state_block_size=state_block_size,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        actual = project_and_save_state_ref.ref_wkv_proj_and_save_state(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            state_block_size=state_block_size,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_ref_wkv_proj_and_save_state_assertions(self):
        hidden_states = jnp.zeros((2, 32), dtype=jnp.float32)
        wkv_wgate = jnp.zeros((32, 256), dtype=jnp.float32)
        ape = jnp.zeros((16, 128), dtype=jnp.float32)
        positions = jnp.zeros((2,), dtype=jnp.int32)
        slot_mapping = jnp.zeros((2,), dtype=jnp.int32)
        cache = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)

        with pytest.raises(AssertionError, match="exceeds the"):
            project_and_save_state_ref.ref_wkv_proj_and_save_state(
                hidden_states,
                wkv_wgate,
                ape,
                positions,
                slot_mapping,
                cache,
                state_block_size=64,
                head_dim=128,
                compress_ratio=16,
                overlap=False,
            )
