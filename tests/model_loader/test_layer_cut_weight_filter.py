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


def test_a_router_streams_cut_to_the_experts_kept(overrides):
    import torch

    overrides["n_routed_experts"] = 4
    gate = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    got = dict(
        mlp._with_experts_cut(
            [
                ("model.layers.3.mlp.gate.weight", gate),
                ("model.layers.3.mlp.gate.e_score_correction_bias", torch.arange(8.0)),
                ("layers.4.ffn.gate.bias", torch.arange(8.0)),
                # DeepSeek-V4's hash-routed layers name experts by token id.
                ("layers.0.ffn.gate.tid2eid", torch.tensor([[0, 5], [7, 2]])),
                # Not routers: a compressor's gate and a shared expert's.
                ("layers.0.attn.compressor.wgate.weight", torch.ones(8, 3)),
                ("model.layers.3.mlp.shared_expert_gate.weight", torch.ones(1, 3)),
            ]
        )
    )
    assert torch.equal(got["model.layers.3.mlp.gate.weight"], gate[:4])
    assert got["model.layers.3.mlp.gate.e_score_correction_bias"].tolist() == [0, 1, 2, 3]
    assert got["layers.4.ffn.gate.bias"].tolist() == [0, 1, 2, 3]
    assert got["layers.0.ffn.gate.tid2eid"].tolist() == [[0, 1], [3, 2]]
    assert got["layers.0.attn.compressor.wgate.weight"].shape == (8, 3)
    assert got["model.layers.3.mlp.shared_expert_gate.weight"].shape == (1, 3)


def test_with_no_expert_override_a_router_is_untouched(overrides):
    import torch

    gate = torch.ones(8, 3)
    assert list(mlp._with_experts_cut([("model.layers.3.mlp.gate.weight", gate)]))[0][1] is gate


def test_an_expert_past_the_cut_is_skipped_whatever_its_suffix(overrides):
    overrides["n_routed_experts"] = 128
    # GLM-5.2's fp8 block scales are named weight_scale_inv, which the EP
    # filter's suffix allowlist leaves out.
    name = "model.layers.3.mlp.experts.131.down_proj.weight_scale_inv"
    assert mlp._should_skip_weight_tpu(name, None)
    assert mlp._should_skip(name, set(range(16)))
    assert not mlp._should_skip_weight_tpu(
        "model.layers.3.mlp.experts.127.down_proj.weight_scale_inv", None
    )
    assert not mlp._should_skip_weight_tpu(
        "model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv", None
    )
