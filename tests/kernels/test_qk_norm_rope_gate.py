"""Parity of the fused split + QK-RMSNorm + RoPE + gate kernel with vLLM's
eager path written in plain JAX, at the served geometry (4 q heads and 1 kv
head of 256 per core, 64 rotary dims) and a small one, for 1 to 32 rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate, reference


@pytest.fixture(autouse=True)
def _needs_a_tpu():
    import jax

    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


def _operands(seed, tokens, heads, kv_heads, dim, rot):
    rng = np.random.default_rng(seed)
    q_gate = jnp.asarray(
        rng.standard_normal((tokens, heads * 2 * dim)).astype(np.float32), jnp.bfloat16
    )
    k = jnp.asarray(
        rng.standard_normal((tokens, kv_heads * dim)).astype(np.float32), jnp.bfloat16
    )
    q_w = jnp.asarray(1.0 + 0.1 * rng.standard_normal(dim).astype(np.float32))
    k_w = jnp.asarray(1.0 + 0.1 * rng.standard_normal(dim).astype(np.float32))
    ang = rng.random((4096, rot // 2)).astype(np.float32) * 6.283
    cache = jnp.asarray(
        np.concatenate([np.cos(ang), np.sin(ang)], axis=1), jnp.bfloat16
    )
    positions = jnp.asarray(rng.integers(0, 4096, size=(tokens,)).astype(np.int32))
    return q_gate, k, q_w, k_w, cache, positions


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32])
@pytest.mark.parametrize("geometry", [(4, 1, 256, 64), (2, 2, 128, 128)])
def test_parity(tokens, geometry):
    heads, kv_heads, dim, rot = geometry
    ops = _operands(tokens * 3 + heads, tokens, heads, kv_heads, dim, rot)
    kw = dict(heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=1e-6)
    got = jax.block_until_ready(qk_norm_rope_gate(*ops, **kw))
    want = reference(*ops, **kw)
    for a, b in zip(got, want):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=2e-2, rtol=2e-2)
