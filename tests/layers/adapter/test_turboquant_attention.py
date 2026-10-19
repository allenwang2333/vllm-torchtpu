# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU tests for the TurboQuant kernel entry: real Pallas RPA + shard_map.

`tests/layers/core/test_turboquant.py` checks the math on CPU against the
pure-JAX reference RPA. These run the production entry point
(`_tq_rpa_kernel_entry`) on TPU, sharded over KV heads, and compare it with
the same computation through the reference RPA.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter.turboquant_attention import (
    TURBOQUANT_KV_CACHE_DTYPES,
    PallasTurboQuantAttentionBackend,
    _tq_rpa_kernel_entry,
    get_tq_layout,
    get_tq_tables,
)
from vllm_torchtpu.layers.core import turboquant as tq

pytestmark = pytest.mark.tpu_test

HEAD_DIM = 128
PAGE_SIZE = 16
PAGES_PER_SEQ = 8
MAX_NUM_SEQS = 4
NUM_PAGES = 64
NUM_TOKENS = 64


def _case(preset: str, num_kv_heads: int, num_q_heads: int, seed: int = 0):
    layout = get_tq_layout(preset, HEAD_DIM)
    tables = get_tq_tables(preset, HEAD_DIM)
    rng = np.random.default_rng(seed)

    kv_lens = np.array([70, 33, 20, 9], np.int32)
    q_lens = np.array([1, 1, 12, 9], np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    distribution = np.array([2, 2, 4], np.int32)
    page_indices = rng.permutation(NUM_PAGES)[: MAX_NUM_SEQS * PAGES_PER_SEQ].astype(
        np.int32
    )

    # Populate history through the quantizer under test.
    cache = jnp.zeros(
        (NUM_PAGES, PAGE_SIZE, num_kv_heads, layout.slot_size_aligned), jnp.uint8
    )
    hist_cu = np.concatenate([[0], np.cumsum(kv_lens - q_lens)]).astype(np.int32)
    n_hist = int(hist_cu[-1])
    hist_k = jnp.asarray(rng.standard_normal((n_hist, num_kv_heads, HEAD_DIM)))
    hist_v = jnp.asarray(rng.standard_normal((n_hist, num_kv_heads, HEAD_DIM)))
    slot_mapping = tq.compute_slot_mapping(
        n_hist,
        jnp.asarray(kv_lens - q_lens),
        jnp.asarray(page_indices),
        jnp.asarray(hist_cu),
        jnp.asarray(distribution),
        PAGE_SIZE,
    )
    cache = tq.write_kv_cache(
        cache, tq.quantize_kv(hist_k, hist_v, layout, tables), slot_mapping
    )
    cache = tq.pack_pages_128(cache, layout)

    def rnd(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

    return dict(
        layout=layout,
        tables=tables,
        cache=cache,
        q=rnd(NUM_TOKENS, num_q_heads, HEAD_DIM),
        k=rnd(NUM_TOKENS, num_kv_heads, HEAD_DIM),
        v=rnd(NUM_TOKENS, num_kv_heads, HEAD_DIM),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu=jnp.asarray(cu),
        distribution=jnp.asarray(distribution),
        total=int(cu[-1]),
    )


@pytest.mark.parametrize("preset", TURBOQUANT_KV_CACHE_DTYPES)
@pytest.mark.parametrize("tp", [1, 2])
def test_kernel_entry_matches_reference(preset, tp):
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    num_kv_heads, num_q_heads = 2 * tp, 4 * tp
    c = _case(preset, num_kv_heads, num_q_heads)
    mesh = Mesh(np.asarray(jax.devices()[:tp]), ("model",))
    sm_scale = HEAD_DIM**-0.5

    entry = jax.jit(
        lambda *a: _tq_rpa_kernel_entry(
            *a,
            None,
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=None,
            sm_scale=sm_scale,
            skip_kv_update=False,
            tq_preset=preset,
        )
    )
    new_cache, out = entry(
        c["cache"],
        c["q"],
        c["k"],
        c["v"],
        c["kv_lens"],
        c["page_indices"],
        c["cu"],
        c["distribution"],
    )
    ref_cache, ref_out = tq.tq_ragged_paged_attention(
        c["q"],
        c["k"],
        c["v"],
        c["cache"],
        c["kv_lens"],
        c["page_indices"],
        c["cu"],
        c["distribution"],
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), np.asarray(ref_cache))
    np.testing.assert_allclose(
        np.asarray(out[: c["total"]], np.float32),
        np.asarray(ref_out, np.float32),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("preset", TURBOQUANT_KV_CACHE_DTYPES)
def test_backend_page_geometry(preset):
    layout = get_tq_layout(preset, HEAD_DIM)
    page_words = layout.page_words_128(PAGE_SIZE)
    shape = PallasTurboQuantAttentionBackend.get_kv_cache_shape(
        10, PAGE_SIZE, 4, HEAD_DIM, preset
    )
    assert shape == (10, page_words, 4, 128)
    page_bytes = PallasTurboQuantAttentionBackend.get_kv_cache_page_size_bytes(
        PAGE_SIZE, 4, HEAD_DIM, preset
    )
    assert page_bytes == page_words * 4 * 128
    # Smaller than the bf16 page for the same geometry.
    assert page_bytes < PAGE_SIZE * 4 * 2 * HEAD_DIM * 2
