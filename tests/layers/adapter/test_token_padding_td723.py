# SPDX-License-Identifier: Apache-2.0
"""TD7.23: cached padding range in ``TokenPaddingState`` and masked_fill
routing-weight zeroing match the pre-TD7.23 behaviour."""

import itertools
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState


def _reference_mask(size: int, valid: int, padded: int) -> torch.Tensor:
    host = torch.zeros(size, dtype=torch.bool)
    host[valid:padded] = True
    return host


class _CountingMask:
    """Wraps a CPU bool mask and counts ``copy_`` calls (H2D writes)."""

    def __init__(self, size: int):
        self.tensor = torch.zeros(size, dtype=torch.bool)
        self.copies = 0
        self.shape = self.tensor.shape

    def copy_(self, src):
        self.copies += 1
        return self.tensor.copy_(src)

    def __getitem__(self, item):
        return self.tensor[item]


def test_create_allocates_zero_mask():
    state = TokenPaddingState.create(5, "cpu")
    assert state.local_padding_mask.dtype == torch.bool
    assert state.local_padding_mask.tolist() == [False] * 5
    assert state.get_local_padding_mask(3).shape == (3,)


@pytest.mark.parametrize("size", [1, 6])
def test_update_sequence_matches_fresh_mask(size):
    """Every (valid, padded) transition, including negative and out-of-range
    bounds, leaves the device mask equal to a freshly built host mask."""
    state = TokenPaddingState.create(size, "cpu")
    values = [-size - 2, -2, 0, 1, size // 2, size, size + 3]
    for valid, padded in itertools.product(values, repeat=2):
        state.update(valid, padded)
        expected = _reference_mask(size, valid, padded)
        assert torch.equal(state.local_padding_mask, expected), (valid, padded)
    # Replay in reverse so every cached range is also overwritten by others.
    for valid, padded in reversed(list(itertools.product(values, repeat=2))):
        state.update(valid, padded)
        assert torch.equal(
            state.local_padding_mask, _reference_mask(size, valid, padded)
        )


def test_update_skips_copy_when_range_repeats():
    mask = _CountingMask(8)
    state = TokenPaddingState(local_padding_mask=mask)  # type: ignore[arg-type]
    state.update(8, 8)  # first update always writes
    assert mask.copies == 1
    state.update(8, 8)
    state.update(3, 3)  # empty range normalizes to (0, 0): no copy
    state.update(9, 20)  # clamps to an empty range: no copy
    assert mask.copies == 1
    state.update(5, 8)
    assert mask.copies == 2
    assert mask.tensor.tolist() == [False] * 5 + [True] * 3
    state.update(5, 100)  # clamps to the same (5, 8): no copy
    state.update(-3, 8)  # also (5, 8)
    assert mask.copies == 2
    state.update(0, 8)
    assert mask.copies == 3
    assert mask.tensor.tolist() == [True] * 8
    state.update(2, 2)
    assert mask.copies == 4
    assert mask.tensor.tolist() == [False] * 8


def test_first_update_overwrites_external_mask():
    """A mask built by the caller is written by the first update even when it
    requests an empty padding range."""
    state = TokenPaddingState(local_padding_mask=torch.ones(4, dtype=torch.bool))
    state.update(4, 4)
    assert state.local_padding_mask.tolist() == [False] * 4


def test_cache_fields_excluded_from_eq_and_repr():
    a = TokenPaddingState(local_padding_mask=torch.zeros(2, dtype=torch.bool))
    a.update(1, 2)
    assert "_host_cache" not in repr(a)
    b = TokenPaddingState(local_padding_mask=a.local_padding_mask)
    assert a == b


def _old_zero(weights, is_padding):
    return torch.where(is_padding[:, None], torch.zeros_like(weights), weights)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_fill_matches_where_zeros_like(monkeypatch, dtype):
    state = TokenPaddingState.create(6, "cpu")
    state.update(3, 6)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    ids = torch.arange(12).reshape(6, 2)
    weights = torch.randn(6, 2).to(dtype)
    weights[4, 0] = float("nan")
    weights[1, 1] = -0.0
    with mock.patch.object(token_padding, "is_forward_context_available") as avail:
        avail.return_value = False
        got_ids, got = token_padding.zero_routing_weights_for_padding(
            ids, weights, is_local_tensor=True
        )
    want = _old_zero(weights, state.get_local_padding_mask(6))
    assert got_ids is ids
    assert got.dtype == want.dtype
    view = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.view(view), want.view(view))


def test_zero_routing_without_state_and_with_stage_mask(monkeypatch):
    ids = torch.zeros(2, 1, dtype=torch.int64)
    weights = torch.ones(2, 1)
    monkeypatch.setattr(token_padding, "_padding_state", None)
    with (
        mock.patch.object(
            token_padding, "is_forward_context_available", return_value=True
        ),
        mock.patch.object(
            token_padding,
            "get_forward_context",
            return_value=SimpleNamespace(additional_kwargs={}),
        ),
    ):
        assert token_padding.zero_routing_weights_for_padding(ids, weights) == (
            ids,
            weights,
        )
    stage = torch.tensor([True, False])
    with (
        mock.patch.object(
            token_padding, "is_forward_context_available", return_value=True
        ),
        mock.patch.object(
            token_padding,
            "get_forward_context",
            return_value=SimpleNamespace(
                additional_kwargs={"token_padding_mask": stage}
            ),
        ),
    ):
        _, out = token_padding.zero_routing_weights_for_padding(ids, weights)
    assert out.tolist() == [[0.0], [1.0]]
