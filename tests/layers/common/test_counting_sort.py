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
"""The counting-sort dispatch permutation must equal the argsort it replaces.

Every test here is two-sided: the equivalence tests are paired with
``test_negative_control_*``, which run a deliberately wrong tie order through
the same comparison and assert it is *caught*. An equivalence suite with no
failing case is a suite that cannot fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.common.counting_sort import (MAX_ROUTES, applicable,
                                                       invert_permutation,
                                                       key_onehot,
                                                       sort_positions)

# (routes, symbols): K3 EP32 decode at pad 1/2/4/8, then shapes that stress the
# bucket-vs-route ratio in both directions.
SHAPES = [(16, 29), (32, 29), (64, 29), (128, 29), (128, 897), (16, 17),
          (7, 29), (1, 29), (512, 29)]


def _cases(rng, n, s):
    yield "random", rng.integers(0, s, n)
    yield "all-same", np.full(n, min(7, s - 1))
    yield "all-sentinel", np.full(n, s - 1)
    yield "sorted", np.sort(rng.integers(0, s, n))
    yield "reverse", np.sort(rng.integers(0, s, n))[::-1].copy()
    yield "two-values", rng.integers(0, 2, n) * (s - 1)
    yield "round-robin", np.arange(n) % s


def _counting(keys, s):
    oh = key_onehot(keys, s)
    pos = sort_positions(oh)
    return invert_permutation(pos), pos


def _argsort(keys):
    si = jnp.argsort(keys)
    return si.astype(jnp.int32), jnp.argsort(si).astype(jnp.int32)


def _descending_tie_positions(onehot):
    """Negative control: same algorithm, ties reversed. Must disagree."""
    counts = onehot.sum(axis=0)
    start = jnp.cumsum(counts) - counts
    rank = jnp.cumsum(onehot[::-1], axis=0)[::-1] - onehot
    return (onehot * (start[None, :] + rank)).sum(axis=1).astype(jnp.int32)


@pytest.mark.parametrize("n,s", SHAPES)
def test_matches_argsort(n, s):
    rng = np.random.default_rng(0xC0FFEE + n * 1000 + s)
    for name, k in _cases(rng, n, s):
        keys = jnp.asarray(k, dtype=jnp.int32)
        got_si, got_pos = _counting(keys, s)
        want_si, want_pos = _argsort(keys)
        np.testing.assert_array_equal(np.asarray(got_pos), np.asarray(want_pos),
                                      err_msg=f"positions {name} n={n} s={s}")
        np.testing.assert_array_equal(np.asarray(got_si), np.asarray(want_si),
                                      err_msg=f"sorted_indices {name} n={n} s={s}")


def test_matches_argsort_randomised():
    rng = np.random.default_rng(7)
    for _ in range(200):
        keys = jnp.asarray(rng.integers(0, 29, 128), dtype=jnp.int32)
        got_si, got_pos = _counting(keys, 29)
        want_si, want_pos = _argsort(keys)
        np.testing.assert_array_equal(np.asarray(got_pos), np.asarray(want_pos))
        np.testing.assert_array_equal(np.asarray(got_si), np.asarray(want_si))


def test_negative_control_tie_order_is_detected():
    """The comparison must fail when tie order is wrong, or it proves nothing."""
    rng = np.random.default_rng(11)
    caught = 0
    trials = 0
    for n, s in [(128, 29), (64, 29), (32, 29)]:
        for _ in range(20):
            k = rng.integers(0, s, n)
            if len(np.unique(k)) == n:      # no ties: nothing to detect
                continue
            trials += 1
            keys = jnp.asarray(k, dtype=jnp.int32)
            wrong = _descending_tie_positions(key_onehot(keys, s))
            _, want_pos = _argsort(keys)
            caught += not np.array_equal(np.asarray(wrong), np.asarray(want_pos))
    assert trials > 0
    assert caught == trials, f"tie-order control fired on only {caught}/{trials}"


def test_positions_are_a_permutation():
    rng = np.random.default_rng(3)
    for n, s in SHAPES:
        keys = jnp.asarray(rng.integers(0, s, n), dtype=jnp.int32)
        pos = np.asarray(sort_positions(key_onehot(keys, s)))
        np.testing.assert_array_equal(np.sort(pos), np.arange(n))


def test_group_sizes_from_the_same_onehot():
    """The column sums must equal the `jax.nn.one_hot` count they replace."""
    rng = np.random.default_rng(5)
    n, experts = 128, 28
    ids = rng.integers(-4, experts, n)          # negatives = non-local under EP
    valid = ids >= 0
    keys = jnp.asarray(np.where(valid, ids, experts), dtype=jnp.int32)
    oh = key_onehot(keys, experts + 1)
    got = np.asarray(oh[:, :experts].sum(axis=0))
    want = np.asarray(
        (jax.nn.one_hot(jnp.asarray(np.where(valid, ids, 0)), experts,
                        dtype=jnp.int32)
         * jnp.asarray(valid)[:, None].astype(jnp.int32)).sum(axis=0))
    np.testing.assert_array_equal(got, want)


def test_applicable_excludes_prefill():
    assert applicable(16) and applicable(128) and applicable(MAX_ROUTES)
    assert not applicable(MAX_ROUTES + 1)
    assert not applicable(8192 * 16)
    assert not applicable(0)


# --------------------------------------------------------------------------
# Integration: the shipped `prepare_routed_gmm_inputs`, flag off vs on.
# The one-hot permute branch is pure jnp, so it executes on CPU -- which is the
# only end-to-end coverage this change can have without a TPU, since every
# test in `test_fused_moe_gmm.py` that would reach it calls `_require_tpu()`.
# --------------------------------------------------------------------------

def _prep(tokens, topk, local_experts, ids, monkeypatch, counting, thr=511,
          sparse_core=True):
    monkeypatch.setenv("TPU_MOE_COUNTING_SORT", "1" if counting else "0")
    from vllm_torchtpu.layers.common import fused_moe_gmm as F
    hidden = jnp.asarray(
        np.random.default_rng(1).standard_normal((tokens, 64)), jnp.bfloat16)
    weights = jnp.asarray(
        np.random.default_rng(2).random((tokens, topk)), jnp.bfloat16)
    return F.prepare_routed_gmm_inputs(
        hidden, jnp.asarray(ids, jnp.int32), weights,
        local_num_experts=local_experts, topk=topk, use_ep=True,
        use_sparse_core=sparse_core, onehot_moe_permute_threshold=thr,
        skip_padded_tokens=False)


ROUTING = [
    # (tokens, topk, local_experts) -- K3 EP32 decode at pad 1 and pad 8
    (1, 16, 28), (8, 16, 28), (4, 16, 28), (8, 8, 28), (2, 16, 56),
]


@pytest.mark.parametrize("tokens,topk,local_experts", ROUTING)
def test_prepare_routed_gmm_inputs_flag_parity(tokens, topk, local_experts,
                                               monkeypatch):
    rng = np.random.default_rng(tokens * 31 + topk)
    # Negative ids are the EP marker for "this expert is not on my shard".
    ids = rng.integers(-local_experts, local_experts, (tokens, topk))
    off = _prep(tokens, topk, local_experts, ids, monkeypatch, False)
    on = _prep(tokens, topk, local_experts, ids, monkeypatch, True)
    names = ("x", "group_sizes", "revert", "topk_weights_flat", "valid_mask",
             "sorted_indices")
    for name, a, b in zip(names, off, on):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=f"{name} differs")


def test_flag_on_really_takes_the_counting_path(monkeypatch):
    """Fire control: with the flag on, `jnp.argsort` must never be reached."""
    import jax.numpy as _jnp
    monkeypatch.setattr(
        _jnp, "argsort",
        lambda *a, **k: pytest.fail("argsort called with the flag ON"))
    ids = np.random.default_rng(9).integers(-28, 28, (8, 16))
    _prep(8, 16, 28, ids, monkeypatch, True)


def test_flag_off_really_takes_the_argsort_path(monkeypatch):
    """Inertness control: with the flag off, the new code must not run."""
    from vllm_torchtpu.layers.common import counting_sort as cs
    monkeypatch.setattr(
        cs, "sort_positions",
        lambda *a, **k: pytest.fail("counting sort ran with the flag OFF"))
    ids = np.random.default_rng(9).integers(-28, 28, (8, 16))
    _prep(8, 16, 28, ids, monkeypatch, False)


def test_prefill_route_count_stays_on_argsort(monkeypatch):
    """`applicable` must keep the O(n^2) inverse away from prefill shapes."""
    from vllm_torchtpu.layers.common import counting_sort as cs
    monkeypatch.setattr(
        cs, "sort_positions",
        lambda *a, **k: pytest.fail("counting sort ran above MAX_ROUTES"))
    # `use_sparse_core=False` takes the plain-gather permute, which is the
    # only permute branch that runs without a TPU at this route count.
    ids = np.random.default_rng(9).integers(-28, 28, (64, 16))   # 1024 routes
    _prep(64, 16, 28, ids, monkeypatch, True, sparse_core=False)
