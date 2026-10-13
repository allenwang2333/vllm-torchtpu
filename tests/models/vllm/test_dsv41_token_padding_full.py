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

import importlib
import itertools
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _ensure_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if "vllm" not in sys.modules:
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    if "vllm.distributed" not in sys.modules:
        dist_mod = types.ModuleType("vllm.distributed")
        dist_mod.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm.distributed", dist_mod)
    if "vllm.distributed.parallel_state" not in sys.modules:
        ps_mod = types.ModuleType("vllm.distributed.parallel_state")
        ps_mod.get_dp_group = lambda: SimpleNamespace(world_size=1)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", ps_mod)
    if "vllm.forward_context" not in sys.modules:
        fc_mod = types.ModuleType("vllm.forward_context")
        fc_mod.is_forward_context_available = lambda: False  # type: ignore[attr-defined]
        fc_mod.get_forward_context = lambda: SimpleNamespace(additional_kwargs={})  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm.forward_context", fc_mod)
    layers_mod = types.ModuleType("vllm_torchtpu.layers")
    layers_mod.__path__ = [str(repo_root / "src/vllm_torchtpu/layers")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers", layers_mod)
    adapter_mod = types.ModuleType("vllm_torchtpu.layers.adapter")
    adapter_mod.__path__ = [str(repo_root / "src/vllm_torchtpu/layers/adapter")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers.adapter", adapter_mod)


class _CountingMask:
    def __init__(self, size: int) -> None:
        self.tensor = torch.zeros(size, dtype=torch.bool)
        self.copies = 0
        self.shape = self.tensor.shape

    def copy_(self, src: torch.Tensor) -> torch.Tensor:
        self.copies += 1
        return self.tensor.copy_(src)

    def __getitem__(self, item: object) -> torch.Tensor:
        return self.tensor[item]  # type: ignore[index]


def test_token_padding_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_vllm_stubs(monkeypatch)
    tp = importlib.import_module("vllm_torchtpu.layers.adapter.token_padding")
    TokenPaddingState = tp.TokenPaddingState

    state = TokenPaddingState.create(6, "cpu")
    assert state.local_padding_mask.dtype == torch.bool
    assert state.local_padding_mask.tolist() == [False] * 6
    assert state.get_local_padding_mask(4).shape == (4,)
    assert state.get_local_padding_mask(6) is state.local_padding_mask
    assert state.get_local_padding_mask(8) is state.local_padding_mask

    # Exhaustive transition parity against fresh reference masks
    values = [-8, -2, 0, 1, 3, 6, 9]
    for valid, padded in itertools.product(values, repeat=2):
        state.update(valid, padded)
        ref = torch.zeros(6, dtype=torch.bool)
        ref[valid:padded] = True
        assert torch.equal(state.local_padding_mask, ref)

    # Counting H2D copies and empty->non-empty / non-empty->empty transitions
    mask = _CountingMask(8)
    c_state = TokenPaddingState(local_padding_mask=mask)  # type: ignore[arg-type]
    c_state.update(8, 8)  # first update with empty range (0, 0)
    assert mask.copies == 1
    c_state.update(3, 3)  # cache hit on (0, 0)
    c_state.update(10, 20)  # out-of-range empty slice -> cache hit on (0, 0)
    assert mask.copies == 1
    c_state.update(5, 8)  # transition from (0, 0) -> (5, 8) (start == stop branch)
    assert mask.copies == 2
    assert mask.tensor.tolist() == [False] * 5 + [True] * 3
    c_state.update(-3, 8)  # out-of-range non-empty slice -> cache hit on (5, 8)
    assert mask.copies == 2
    c_state.update(2, 6)  # transition from (5, 8) -> (2, 6) (start < stop branch)
    assert mask.copies == 3
    assert mask.tensor.tolist() == [False, False, True, True, True, True, False, False]
    c_state.update(4, 4)  # transition from (2, 6) -> (0, 0) (p_start == p_stop branch)
    assert mask.copies == 4
    assert mask.tensor.tolist() == [False] * 8

    # Forward context branches
    tp.set_padding_state(None)
    ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    weights = torch.ones((4, 2), dtype=torch.float32)

    monkeypatch.setattr(tp, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        tp, "get_forward_context", lambda: SimpleNamespace(additional_kwargs={})
    )
    out_ids, out_w = tp.zero_routing_weights_for_padding(ids, weights)
    assert out_ids is ids and out_w is weights

    monkeypatch.setattr(
        tp,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"other_key": 1}),
    )
    out_ids, out_w = tp.zero_routing_weights_for_padding(ids, weights)
    assert out_ids is ids and out_w is weights

    stage_mask = torch.tensor([True, False, True, False])
    monkeypatch.setattr(
        tp,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"token_padding_mask": stage_mask}),
    )
    out_ids, out_w = tp.zero_routing_weights_for_padding(ids, weights)
    assert out_ids is ids
    assert out_w.tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

    # Persistent padding state branches (is_local_tensor, dp_size=1, dp_size=2)
    monkeypatch.setattr(tp, "is_forward_context_available", lambda: False)
    p_state = TokenPaddingState.create(4, "cpu")
    p_state.update(2, 4)
    tp.set_padding_state(p_state)

    _, local_w = tp.zero_routing_weights_for_padding(ids, weights, is_local_tensor=True)
    assert local_w.tolist() == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

    monkeypatch.setattr(tp, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    _, dp1_w = tp.zero_routing_weights_for_padding(ids, weights, is_local_tensor=False)
    assert dp1_w.tolist() == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

    dp2_state = TokenPaddingState.create(2, "cpu")
    dp2_state.update(1, 2)
    tp.set_padding_state(dp2_state)
    monkeypatch.setattr(
        tp,
        "get_dp_group",
        lambda: SimpleNamespace(
            world_size=2,
            all_gather=lambda t, dim=0: torch.cat([t, t], dim=dim),
        ),
    )
    _, dp2_w = tp.zero_routing_weights_for_padding(ids, weights, is_local_tensor=False)
    assert dp2_w.tolist() == [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]

    tp.set_padding_state(None)
    assert tp._padding_state is None
