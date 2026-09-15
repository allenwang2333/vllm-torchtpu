# SPDX-License-Identifier: Apache-2.0
"""Check LONGCTX automatic VMEM budgeting with eight SPMD shards.

The compile regressions preserve the full serving cache/metadata shapes with
abstract inputs, without allocating the 26.45 GiB KV pool per rank.
The NEW_TOKENS_ONLY pass receives two local Q heads, unlike CACHE_ONLY's
sixteen gathered heads. Both output and LSE must remain live during compile.

With the original VMEM estimate, automatic selection produces the mixed
kernel RPAm-snh-p256-b2-q768-k768 and exceeds TPU7x's 64 MiB VMEM budget.
Compilation is expected to succeed after fixing the estimator; do not catch
the OOM or pin a smaller tile here.

Small numerical cases check output, LSE, and live KV writes for HND/NHD,
with and without DCP. They do not exercise distributed historical LSE merging.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)


@pytest.fixture
def mesh():
    if jax.default_backend() != "tpu":
        pytest.skip("LONGCTX compilation requires TPU")
    devices = jax.local_devices()
    if len(devices) < 8:
        pytest.skip(
            "The serving reproduction requires eight local TPU devices")
    tpu_info = pltpu.get_tpu_info()
    if (tpu_info.generation != 7
            or tpu_info.vmem_capacity_bytes != 64 * 1024**2):
        pytest.skip("This reproduction targets TPU7x with 64 MiB VMEM")
    return Mesh(np.asarray(devices[:8]), ("dcp", ))


def test_dcp8_hnd_prefill_auto_blocks_compile_spmd(mesh):
    _compile_serving_shape(mesh, cp_group_size=8, return_lse=True)


def test_tp8_hnd_prefill_auto_blocks_compile_spmd(mesh):
    _compile_serving_shape(mesh, cp_group_size=None, return_lse=False)


def _compile_serving_shape(mesh, *, cp_group_size, return_lse):

    dcp_size = 8
    num_tokens = 4096
    num_seqs = 8  # Serving pads max_num_seqs=4 to eight metadata slots.
    local_q_heads, kv_heads, head_dim = 2, 1, 256
    # Manager block_size=768 consists of three 256-token kernel pages.
    # These are the actual cache/table shapes from the failed 12-layer run.
    page_size = 256
    local_pages = 216696
    num_page_indices = 4104
    output_specs = (P(None, "dcp"), P("dcp"))
    if return_lse:
        output_specs += (P(None, "dcp"), )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("dcp"), P(None, "dcp"), P(), P(), P(), P(), P(), P()),
        out_specs=output_specs,
        check_vma=False,
    )
    def run(cache, query, key, value, seq_lens, page_indices, cu_q_lens,
            distribution):
        return wrapper.ragged_paged_attention(
            queries=query,
            keys=key,
            values=value,
            kv_cache=cache,
            kv_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=head_dim**-0.5,
            k_scale=1.0,
            v_scale=1.0,
            cp_group_size=cp_group_size,
            cp_rank=(jax.lax.axis_index("dcp").reshape(1)
                     if cp_group_size else None),
            attention_scope=(configs.AttentionScope.NEW_TOKENS_ONLY if
                             cp_group_size else configs.AttentionScope.FULL),
            return_lse=return_lse,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    cache_shape = wrapper.get_kv_cache_shape(
        local_pages * dcp_size,
        page_size,
        kv_heads,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    abstract = jax.ShapeDtypeStruct
    args = (
        abstract(cache_shape, jnp.float8_e4m3fn),
        abstract((num_tokens, local_q_heads * dcp_size, head_dim),
                 jnp.bfloat16),
        abstract((num_tokens, kv_heads, head_dim), jnp.float8_e4m3fn),
        abstract((num_tokens, kv_heads, head_dim), jnp.float8_e4m3fn),
        abstract((num_seqs, ), jnp.int32),
        abstract((num_page_indices, ), jnp.int32),
        abstract((num_seqs + 1, ), jnp.int32),
        abstract((3, ), jnp.int32),
    )
    # Donation matches the serving custom op's in-place KV-cache update.
    jax.jit(run, donate_argnums=(0, )).lower(*args).compile()


@pytest.mark.parametrize("kv_layout", list(configs.KVLayout))
@pytest.mark.parametrize("cp_group_size,return_lse", [(8, True), (None, True),
                                                      (None, False)])
def test_auto_blocks_spmd_output_and_cache(mesh, kv_layout, cp_group_size,
                                           return_lse):
    """Exercise decode + prefill, LSE padding, and rank-local KV writes."""
    num_ranks, q_heads, head_dim, page_size = 8, 16, 256, 256
    lengths = np.array([1, 257, 5], np.int32)
    starts = np.concatenate((np.array([0], np.int32), np.cumsum(lengths)))
    num_tokens = int(starts[-1])
    pages_per_seq = 2
    local_pages = len(lengths) * pages_per_seq
    page_indices = np.arange(local_pages, dtype=np.int32)
    rng = np.random.default_rng(20260915)
    query = rng.normal(0, 0.25,
                       (num_tokens, q_heads, head_dim)).astype(jnp.bfloat16)
    key = rng.normal(0, 0.25,
                     (num_tokens, 1, head_dim)).astype(jnp.float8_e4m3fn)
    value = rng.normal(0, 0.25,
                       (num_tokens, 1, head_dim)).astype(jnp.float8_e4m3fn)
    cache_shape = wrapper.get_kv_cache_shape(local_pages * num_ranks,
                                             page_size,
                                             1,
                                             head_dim,
                                             jnp.float8_e4m3fn,
                                             kv_layout=kv_layout)
    output_specs = (P(None, "dcp"), P("dcp"))
    if return_lse:
        output_specs += (P(None, "dcp"), )

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("dcp"), P(None, "dcp"), P(), P(), P(), P(), P(), P()),
        out_specs=output_specs,
        check_vma=False,
    )
    def run(cache, q, k, v, seq_lens, pages, cu_q_lens, distribution):
        return wrapper.ragged_paged_attention(
            q,
            k,
            v,
            cache,
            seq_lens,
            pages,
            cu_q_lens,
            distribution,
            sm_scale=head_dim**-0.5,
            k_scale=1.0,
            v_scale=1.0,
            cp_group_size=cp_group_size,
            cp_rank=(jax.lax.axis_index("dcp").reshape(1)
                     if cp_group_size else None),
            attention_scope=(configs.AttentionScope.NEW_TOKENS_ONLY if
                             cp_group_size else configs.AttentionScope.FULL),
            return_lse=return_lse,
            kv_layout=kv_layout,
        )

    output, cache, *lse = run(jnp.zeros(cache_shape, jnp.float8_e4m3fn),
                              jnp.asarray(query), jnp.asarray(key),
                              jnp.asarray(value), jnp.asarray(lengths),
                              jnp.asarray(page_indices),
                              jnp.asarray(starts, jnp.int32),
                              jnp.array([1, 1, 3], jnp.int32))
    query, key, value = (x.astype(np.float32) for x in (query, key, value))
    expected_out, expected_lse = [], []
    for start, end in zip(starts[:-1], starts[1:]):
        keys = np.repeat(key[start:end], q_heads, axis=1)
        values = np.repeat(value[start:end], q_heads, axis=1)
        scores = np.einsum("thd,shd->hts", query[start:end], keys)
        scores *= head_dim**-0.5
        causal = np.arange(end - start)[:, None] >= np.arange(end - start)
        scores = np.where(causal, scores, -np.inf)
        maximum = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - maximum)
        denominator = weights.sum(axis=-1, keepdims=True)
        expected_out.append(
            np.einsum("hts,shd->thd", weights / denominator, values))
        expected_lse.append((maximum + np.log(denominator))[..., 0].T)
    np.testing.assert_allclose(np.asarray(output, np.float32),
                               np.concatenate(expected_out),
                               atol=1e-2,
                               rtol=2e-2)
    if return_lse:
        np.testing.assert_allclose(np.asarray(lse[0], np.float32),
                                   np.concatenate(expected_lse),
                                   atol=4e-2,
                                   rtol=1e-2)

    # Independently pack live KV slots and check untouched pages/ranks. HND
    # writes whole pages, so the unused tail of a written page is unspecified.
    # The 257-token sequence crosses a DCP ownership boundary.
    expected_cache = np.zeros(cache_shape, np.float32)
    live = np.zeros((local_pages * num_ranks, page_size), dtype=bool)
    for rank in range(num_ranks):
        for seq, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            for pos in range(end - start):
                if cp_group_size and (pos //
                                      page_size) % cp_group_size != rank:
                    continue
                local_pos = pos
                if cp_group_size:
                    local_pos = (pos //
                                 (page_size * cp_group_size) * page_size +
                                 pos % page_size)
                page = (rank * local_pages + seq * pages_per_seq +
                        local_pos // page_size)
                offset = local_pos % page_size
                live[page, offset] = True
                for kv, data in enumerate((key, value)):
                    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                        expected_cache[page, kv, :, :,
                                       offset] = data[start + pos, 0].reshape(
                                           head_dim // 4, 4)
                    else:
                        expected_cache[page, offset, 0,
                                       kv, :] = data[start + pos, 0]
    actual_cache = np.asarray(cache, np.float32)
    untouched_pages = ~live.any(axis=1)
    np.testing.assert_array_equal(actual_cache[untouched_pages],
                                  expected_cache[untouched_pages])
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        actual_cache = actual_cache.transpose(0, 4, 1, 2, 3)
        expected_cache = expected_cache.transpose(0, 4, 1, 2, 3)
    np.testing.assert_array_equal(actual_cache[live], expected_cache[live])
