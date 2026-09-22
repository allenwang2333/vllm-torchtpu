# SPDX-License-Identifier: Apache-2.0
"""Small CPU/JAX smoke tests; no model weights or TorchTPU runtime needed."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[2] /
         "src/vllm_torchtpu/kernels/mla" / "decode_dcp.py")
_SPEC = importlib.util.spec_from_file_location("k3_decode_kernel", _PATH)
kernel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernel)


@pytest.mark.parametrize("length", [1, 4, 5, 8, 9, 17])
def test_decode_page_ownership_and_empty_rank(length):
    # Deliberately non-contiguous physical pages, with NaNs in unused rows.
    page_size, latent, rope = 4, 8, 4
    q = jnp.ones((2, 2, latent), dtype=jnp.float32)
    pe = jnp.ones((2, 2, rope), dtype=jnp.float32)
    tables = jnp.array([[3, 1, 4], [0, 0, 0]], dtype=jnp.int32)
    starts = jnp.array([0, 1, 1], dtype=jnp.int32)
    lengths = jnp.array([length, 0], dtype=jnp.int32)
    kv = jnp.ones((2, latent), dtype=jnp.float32)
    new_pe = jnp.ones((2, rope), dtype=jnp.float32)
    partials, lses = [], []
    for rank in range(2):
        local = int(kernel.local_context_lengths(length, rank, 2, page_size))
        cache = jnp.full((5, 2, 2, latent + rope), jnp.nan)
        for pos in range(local):
            cache = cache.at[int(tables[0, pos // 4]), pos % 4 // 2,
                             pos % 2].set(1)
        # New token must be inserted only by its owner.
        owner = (length - 1) // 4 % 2
        if owner == rank:
            pos = local - 1
            cache = cache.at[int(tables[0, pos // 4]), pos % 4 // 2,
                             pos % 2].set(jnp.nan)
        run = jax.jit(
            lambda c, rank=rank: kernel.decode_attention(c,
                                                         q,
                                                         pe,
                                                         kv,
                                                         new_pe,
                                                         lengths,
                                                         tables,
                                                         starts,
                                                         rank=rank,
                                                         size=2,
                                                         scale=0.25,
                                                         tile_size=4))
        updated, out, lse = run(cache)
        if local:
            np.testing.assert_allclose(out[0], 1, atol=1e-5)
            np.testing.assert_allclose(lse[0], 3 + np.log(local), atol=1e-5)
        else:
            assert np.isneginf(lse[0]).all()
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out[1], 0)
        np.testing.assert_array_equal(np.isnan(updated[0]), True)
        partials.append(out)
        lses.append(lse)
    # Every logical token belongs to exactly one of the two cache shards.
    total = sum(
        int(kernel.local_context_lengths(length, r, 2, 4)) for r in range(2))
    assert total == length


def test_invalid_cp_rank():
    with pytest.raises(ValueError, match="rank/size"):
        kernel.decode_attention(None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                rank=2,
                                size=2,
                                scale=1)


def test_native_tp8_dcp2_head_and_page_layout():
    # Each of four pairs in TP8 gathers two 12-head slices. Distinct query
    # heads and KV rows make accidental replication/permutation visible.
    rng = np.random.default_rng(7)
    latent, rope, length, page = 8, 4, 35, 16
    queries = rng.normal(size=(96, latent + rope)).astype(np.float32)
    kv = rng.normal(size=(length, latent + rope)).astype(np.float32)
    tables = jnp.array([[3, 1]], dtype=jnp.int32)
    outputs = []
    for pair in range(4):
        q = jnp.asarray(queries[pair * 24:(pair + 1) * 24][None])
        partials, lses = [], []
        for rank in range(2):
            cache = jnp.full((4, 8, 2, latent + rope), jnp.nan)
            for pos in range(length - 1):
                if pos // page % 2 == rank:
                    local = pos // (2 * page) * page + pos % page
                    cache = cache.at[int(tables[0, local // page]),
                                     local % page // 2, local % 2].set(kv[pos])
            updated, out, lse = kernel.decode_attention(
                cache,
                q[..., :latent],
                q[..., latent:],
                jnp.asarray(kv[-1:, :latent]),
                jnp.asarray(kv[-1:, latent:]),
                jnp.array([length]),
                tables,
                jnp.array([0, 1]),
                rank=rank,
                size=2,
                scale=0.25,
                tile_size=16)
            assert out.shape == (1, 24, latent)
            assert np.isfinite(out).all()
            np.testing.assert_array_equal(np.isnan(updated[0]), True)
            partials.append(np.asarray(out))
            lses.append(np.asarray(lse))
        lses = np.stack(lses)
        weights = np.exp(lses - lses.max(axis=0))
        weights /= weights.sum(axis=0)
        merged = (np.stack(partials) * weights[..., None]).sum(axis=0)
        # The head reduce-scatter restores each original 12-head TP slice.
        outputs.extend(np.split(merged, 2, axis=1))
    actual = np.concatenate(outputs, axis=1)[0]
    scores = queries @ kv.T * 0.25
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual,
                               probabilities @ kv[:, :latent],
                               rtol=2e-5,
                               atol=2e-5)
