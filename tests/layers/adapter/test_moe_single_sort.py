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
"""The single-sort MoE permutation matches the three-sort reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import fused_moe_gmm

pytestmark = pytest.mark.cpu_test

LOCAL_EXPERTS, TOTAL_EXPERTS, TOPK = 64, 512, 10


def _expert_parallel_ids(num_tokens, seed):
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.choice(TOTAL_EXPERTS, TOPK, replace=False) for _ in range(num_tokens)]
    )


def _three_sort_reference(sort_keys):
    token_bits = max(1, (sort_keys.shape[0] // TOPK - 1).bit_length())
    token_ids = jnp.arange(sort_keys.shape[0], dtype=jnp.int32) // TOPK
    sorted_indices = jnp.argsort(sort_keys)
    packed = jax.lax.sort(
        jnp.left_shift(sort_keys, token_bits) | token_ids, is_stable=False
    )
    token_indices = jnp.bitwise_and(packed, (1 << token_bits) - 1)
    return sorted_indices, jnp.argsort(sorted_indices), token_indices


def _assert_single_sort_matches(num_tokens):
    ids = _expert_parallel_ids(num_tokens, seed=num_tokens)
    keys = jnp.asarray(
        np.where(ids < LOCAL_EXPERTS, ids, LOCAL_EXPERTS).astype(np.int32).reshape(-1)
    )
    flat_bits = max(1, (keys.shape[0] - 1).bit_length())
    actual = fused_moe_gmm._single_sort_permutation(keys, TOPK, flat_bits)
    for got, want in zip(actual, _three_sort_reference(keys)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_permutation_matches_three_sorts_at_one_token():
    _assert_single_sort_matches(1)


def test_single_sort_permutation_matches_three_sorts_at_decode_batch():
    _assert_single_sort_matches(4)


def test_single_sort_permutation_matches_three_sorts_below_ragged_threshold():
    # 204 tokens * top-10 = 2,040 routed rows, just under the 2,048 threshold.
    _assert_single_sort_matches(204)


def test_prepare_routed_gmm_inputs_single_sort_matches_three_sorts_with_padded_tokens(
    monkeypatch,
):
    num_tokens, hidden = 8, 16
    rng = np.random.default_rng(0)
    ids = _expert_parallel_ids(num_tokens, seed=1)
    # EP marks experts owned by other ranks with a negative id.
    local_ids = jnp.asarray(np.where(ids < LOCAL_EXPERTS, ids, -1).astype(np.int32))
    weights = rng.random((num_tokens, TOPK)).astype(np.float32)
    weights[-2:] = 0.0  # Two padded tokens carry zero routing weight.
    hidden_states = jnp.asarray(
        rng.standard_normal((num_tokens, hidden)).astype(np.float32)
    )
    outputs = {}
    for single_sort in (False, True):
        monkeypatch.setattr(fused_moe_gmm, "_SINGLE_SORT", single_sort)
        outputs[single_sort] = fused_moe_gmm.prepare_routed_gmm_inputs(
            hidden_states,
            local_ids,
            jnp.asarray(weights),
            local_num_experts=LOCAL_EXPERTS,
            topk=TOPK,
            use_ep=True,
            use_sparse_core=False,
            skip_padded_tokens=True,
        )
    for three_sorts, one_sort in zip(outputs[False], outputs[True]):
        np.testing.assert_array_equal(np.asarray(three_sorts), np.asarray(one_sort))
