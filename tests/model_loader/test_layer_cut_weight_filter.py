# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for skipping decoder layers past a num_hidden_layers override.

A model served from its first N layers has no module for a later layer, so
the loader must never fetch that layer's tensors. No TPU, no network.
"""

from types import SimpleNamespace

import pytest
import vllm.config

from vllm_torchtpu import model_loader_patches as mlp


@pytest.fixture
def overrides(monkeypatch):
    box: dict = {}
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_overrides=box)),
    )
    return box


def test_a_layer_past_the_cut_is_skipped_and_one_inside_it_kept(overrides):
    overrides["num_hidden_layers"] = 7
    assert mlp._should_skip_weight_tpu("model.layers.7.self_attn.q_a_proj.weight", None)
    # GLM-5.2's MTP layer is the one after the last decoder layer.
    assert mlp._should_skip_weight_tpu("model.layers.78.eh_proj.weight", None)
    assert not mlp._should_skip_weight_tpu(
        "model.layers.6.mlp.experts.3.down_proj.weight", None
    )
    assert not mlp._should_skip_weight_tpu("model.embed_tokens.weight", None)
    assert not mlp._should_skip_weight_tpu("lm_head.weight", None)


def test_with_no_override_every_layer_is_kept(overrides):
    assert not mlp._should_skip_weight_tpu(
        "model.layers.70.self_attn.q_a_proj.weight", None
    )


def test_the_streaming_plan_skips_a_cut_layer_even_for_a_local_expert(overrides):
    overrides["num_hidden_layers"] = 7
    assert mlp._should_skip("model.layers.9.mlp.experts.0.down_proj.weight", {0})
    assert not mlp._should_skip("model.layers.6.mlp.experts.0.down_proj.weight", {0})
