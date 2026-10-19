# SPDX-License-Identifier: Apache-2.0
"""Chunk boundaries preserve ragged requests, SWA visibility and cached KV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import csa2_pipeline as pipeline
from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2_cache import encode_csa2_kv
from vllm_torchtpu.kernels.deepseek_v41.csa2_metadata import prepare_swa_metadata
from vllm_torchtpu.kernels.deepseek_v41.csa2_pipeline import csa2_pipeline
from vllm_torchtpu.kernels.deepseek_v41.tuned_block_sizes import csa2_gather_chunk_size


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,pipelined",
    [(128, False), (288, True), (384, True), (512, False), (513, True)],
)
def test_default_pipeline_streams_queries_and_preserves_gathered_order(
    monkeypatch, tokens, pipelined
):
    """Decode exposes separate gather/attention chunks for device overlap."""
    events = []

    def gather(words, scales, records):
        events.append(("gather", records.shape[0]))
        return words[records], scales[records]

    def attend(q, words, scales, *args, **kwargs):
        events.append(("attention", q.shape[0]))
        return q + words[:, :1] * scales[:, :1, None]

    monkeypatch.setattr(pipeline, "gather_csa2_kv_sc", gather)
    monkeypatch.setattr(pipeline, "csa2", attend)
    q = jnp.arange(tokens, dtype=jnp.float32).reshape(tokens, 1, 1)
    words = jnp.arange(tokens, dtype=jnp.float32)[:, None]
    scales = jnp.linspace(0.1, 1, tokens)
    records = jnp.broadcast_to(jnp.arange(tokens)[:, None], (tokens, 512))
    # Exercise scheduling with lightweight stand-ins for the device kernels;
    # the TPU test below checks real attention and banked cache numerics.
    with jax.disable_jit():
        output, gathered_words, gathered_scales = csa2_pipeline(
            q,
            words,
            scales,
            records,
            words,
            jnp.array([0, tokens]),
            jnp.arange(tokens),
            jnp.zeros(1),
            q,
            jnp.zeros((1, 3)),
            None,
            jnp.full((tokens,), 512),
            chunk_size=csa2_gather_chunk_size(tokens),
            bq=6,
            scale=1.0,
            compress_ratio=2,
        )
    gathered = [size for kind, size in events if kind == "gather"]
    attended = [size for kind, size in events if kind == "attention"]
    assert gathered == attended
    assert sum(gathered) == tokens
    if pipelined:
        assert len(gathered) > 1
        assert events[:3] == [
            ("gather", gathered[0]),
            ("gather", gathered[1]),
            ("attention", attended[0]),
        ]
    else:
        assert len(gathered) == 1
    np.testing.assert_allclose(output, q + words[:, None] * scales[:, None, None])
    np.testing.assert_array_equal(gathered_words, words[records])
    np.testing.assert_array_equal(gathered_scales, scales[records])


@pytest.fixture
def sparsecore_device():
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 pipeline requires a TPU with SparseCore")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("CSA2 pipeline requires SparseCore")


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("decode", [False, True])
def test_chunked_attention_preserves_requests_and_reusable_kv(
    sparsecore_device, banked, non_causal, decode
):
    # Empty requests and padding remain valid across chunk boundaries.
    # Noncausal windows include KV beyond the end of the current chunk.
    tokens, heads, page_size = (288, 64, 32) if decode else (193, 8, 32)
    chunk_size = csa2_gather_chunk_size(tokens) if decode else 64
    rng = np.random.default_rng(31)

    def random(shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    words, scales = encode_csa2_kv(random((257, 512)))
    if decode:
        # A shortened MTP request offsets the six-query groups across chunks.
        # Noncausal visibility must still include KV beyond the chunk end.
        lengths = [0, 5, *([6] * 42), 0]
        minimum = jnp.zeros(len(lengths), jnp.int32)
        positions = jnp.array(
            [*range(330, 335)] + [*range(330, 336)] * 42 + [0] * 31, jnp.int32
        )
        starts = jnp.array(np.cumsum([0, *lengths]), jnp.int32)
        valid_tokens = 257
    else:
        starts = jnp.array([0, 0, 43, 43, 103, 155, 155], jnp.int32)
        positions = jnp.array(
            [*range(160, 203), *range(240, 300), *range(330, 382)] + [0] * 38,
            jnp.int32,
        )
        minimum = jnp.array([0, 150, 0, 200, 300, 0], jnp.int32)
        valid_tokens = 155
    counts = jnp.array(rng.integers(0, 513, tokens), jnp.int32)
    counts = jnp.where(jnp.arange(tokens) < valid_tokens, counts, 0)
    records = jnp.array(rng.integers(0, 257, (tokens, 512)), jnp.int32)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, -1)
    selected = jnp.maximum(records, 0)
    expected_words = jnp.where(records[:, :, None] >= 0, words[selected], 0)
    expected_scales = jnp.where(records >= 0, scales[selected], 0)
    requests = starts.shape[0] - 1
    pages = requests * 16 + 1
    table = jnp.arange(1, pages, dtype=jnp.int32).reshape(requests, 16)
    metadata, bounds, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        minimum,
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
        non_causal=non_causal,
    )
    swa = random((pages, page_size, 512)).at[0].set(0)
    bits = jax.lax.bitcast_convert_type(swa, jnp.uint16).astype(jnp.uint32)
    bits = bits.reshape(pages, page_size, 2, 2, 128)
    packed = bits[:, :, :, 0] | (bits[:, :, :, 1] << 16)
    swa = jax.lax.bitcast_convert_type(packed, jnp.int32).transpose(0, 2, 1, 3)
    word_banks = (words[:113], words[113:]) if banked else words
    scale_banks = (scales[:113], scales[113:]) if banked else scales
    swa_banks = (swa[:48], swa[48:]) if banked else swa
    angles = jnp.asarray(rng.standard_normal((tokens, 64)) * 0.1)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), axis=1),
            jnp.concatenate((jnp.zeros_like(angles), jnp.sin(angles)), axis=1),
        ),
        axis=1,
    )
    sink = jnp.linspace(-1, 1, heads, dtype=jnp.float32)
    kwargs = dict(scale=512**-0.5, compress_ratio=2, bq=6)

    def attend(query, selected_words, selected_scales, offset=0):
        end = offset + query.shape[0]
        return csa2(
            query,
            selected_words,
            selected_scales,
            swa_banks,
            jnp.clip(starts - offset, 0, query.shape[0]),
            positions[offset:end],
            sink,
            rope[offset:end],
            swa_metadata=metadata,
            swa_bounds=bounds if non_causal else None,
            main_lengths=counts[offset:end],
            **kwargs,
        )

    expected = attend(q, expected_words, expected_scales)
    output, gathered_words, gathered_scales = csa2_pipeline(
        q,
        word_banks,
        scale_banks,
        records,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        metadata,
        bounds if non_causal else None,
        counts,
        chunk_size=chunk_size,
        **kwargs,
    )
    np.testing.assert_allclose(output, expected, atol=2e-3, rtol=2e-2)
    np.testing.assert_array_equal(output[valid_tokens:], 0)
    np.testing.assert_array_equal(gathered_words, expected_words)
    np.testing.assert_array_equal(gathered_scales, expected_scales)

    next_q = q * jnp.bfloat16(0.7)
    reused = attend(next_q, gathered_words, gathered_scales)
    np.testing.assert_allclose(
        reused, attend(next_q, expected_words, expected_scales), atol=2e-3, rtol=2e-2
    )
