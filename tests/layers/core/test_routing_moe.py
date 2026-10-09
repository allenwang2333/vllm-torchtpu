# SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func


def test_fused_moe_func_selects_routing_gmm(monkeypatch):
    calls = []

    def mock_routing_gmm(*args, **kwargs):
        calls.append("routing_gmm")
        # Return mock gmm_out with compact shape:
        # [num_tokens * topk, hidden_size // 256, 2, 128]
        return jnp.ones((256 * 8, 1, 2, 128), dtype=jnp.bfloat16)

    def mock_reduce(
        rows,
        weights,
        topk_indices=None,
        group_offset=None,
        num_local_experts=None,
        **kwargs,
    ):
        calls.append("reduce")
        return jnp.ones((weights.shape[0], 256), dtype=jnp.bfloat16)

    monkeypatch.setattr(
        fused_moe_gmm.routing_gmm_kernel, "routing_gmm", mock_routing_gmm
    )
    monkeypatch.setattr(
        fused_moe_gmm.output_reduce, "output_reduce_unpack", mock_reduce
    )
    monkeypatch.setattr(envs, "USE_ROUTING_MOE", True)

    num_tokens = 256
    hidden_size = 256
    num_experts = 8
    topk = 8

    hidden_states = jnp.ones((num_tokens, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((num_experts, hidden_size, 512), dtype=jnp.bfloat16)
    w2 = jnp.ones((num_experts, 512, hidden_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((num_tokens, topk), dtype=jnp.float32) / topk
    topk_ids = jnp.zeros((num_tokens, topk), dtype=jnp.int32)

    out = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
        use_ep=False,
    )

    assert calls == ["routing_gmm", "reduce"]
    assert out.shape == (num_tokens, hidden_size)


def test_fused_moe_func_skips_routing_gmm_when_small_tokens(monkeypatch):
    calls = []

    def mock_routing_gmm(*args, **kwargs):
        calls.append("routing_gmm")

    monkeypatch.setattr(
        fused_moe_gmm.routing_gmm_kernel, "routing_gmm", mock_routing_gmm
    )
    monkeypatch.setattr(envs, "USE_ROUTING_MOE", True)
    # Mock standard moe_gmm to catch fallback
    monkeypatch.setattr(
        fused_moe_gmm,
        "moe_gmm",
        lambda *args, **kwargs: jnp.zeros((128, 256), dtype=jnp.bfloat16),
    )

    num_tokens = 128
    hidden_size = 256
    num_experts = 8
    topk = 8

    hidden_states = jnp.ones((num_tokens, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((num_experts, hidden_size, 512), dtype=jnp.bfloat16)
    w2 = jnp.ones((num_experts, 512, hidden_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((num_tokens, topk), dtype=jnp.float32) / topk
    topk_ids = jnp.zeros((num_tokens, topk), dtype=jnp.int32)

    out = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
        use_ep=False,
    )

    assert "routing_gmm" not in calls
    assert out.shape == (num_tokens, hidden_size)
