"""Replica slots must not extend the grouped fallback's primary interval."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.common.fused_moe_gmm import _local_expert_routing
from vllm_torchtpu.layers.vllm import fused_moe


@pytest.mark.parametrize("rows", [32, 64, 128, 512, 1024])
def test_replica_slots_do_not_duplicate_fallback_results(rows):
    ranks, owned, copies = 8, 64, 16
    ids = jnp.arange(rows * 10, dtype=jnp.int32).reshape(rows,
                                                         10) % (ranks * owned)
    weights = jnp.full(ids.shape, 0.1, jnp.float32)
    # Include padding and both interval edges. Copy slots have deliberately
    # different values, so evaluating one as a primary corrupts the output.
    ids = ids.at[0, :5].set(jnp.array([-1, 0, 63, 64, 511]))
    weights = weights.at[-1].set(0)
    total = jnp.zeros((rows, ), jnp.float32)
    owners = jnp.zeros(ids.shape, jnp.int32)
    for rank in range(ranks):
        local, routed = _local_expert_routing(ids, weights,
                                              jnp.array(rank * owned), owned)
        values = jnp.concatenate([
            jnp.arange(rank * owned, (rank + 1) * owned),
            jnp.full((copies, ), -10000)
        ])
        total += (values[jnp.maximum(local, 0)] * routed).sum(-1)
        owners += (local >= 0).astype(jnp.int32)
    expected = jnp.where(ids >= 0, ids * weights, 0).sum(-1)
    np.testing.assert_allclose(total, expected, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(owners, ids >= 0)


def test_custom_op_cache_separates_primary_counts(monkeypatch):
    built = []

    def register(name, function):
        op = SimpleNamespace(register_fake=lambda fn: None, function=function)
        built.append(op)
        return op

    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    monkeypatch.setattr(fused_moe.pallas, "jax_op", register)
    opts = dict(topk=10,
                activation="silu",
                use_ep=True,
                skip_padded_tokens=True)
    plain = fused_moe._get_fused_moe_custom_op(**opts)
    copies = fused_moe._get_fused_moe_custom_op(**opts, num_primary_experts=64)
    assert plain is not copies
    assert copies is fused_moe._get_fused_moe_custom_op(**opts,
                                                        num_primary_experts=64)
    assert len(built) == 2
    assert copies.function.keywords["num_primary_experts"] == 64
