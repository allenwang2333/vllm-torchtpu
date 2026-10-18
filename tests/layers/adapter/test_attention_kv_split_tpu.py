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
"""TPU parity of the split KV read (TPU_RPA_KV_SPLIT) against the stock kernel.

Run from the repository root on a TPU host::

    PYTHONPATH=src python -m pytest -v \\
        tests/layers/adapter/test_attention_kv_split_tpu.py

Every rank of one KV-head group runs on one device, in turn: the CACHE_ONLY
pass of rank g on its page view with the group's query heads, the local
log-sum-exp merge of the G partials (the math of `cp_lse_ag_out_rs` without the
collectives), the group-size-1 NEW_TOKENS_ONLY pass with the rank's own heads,
and the final merge. The result is compared with the stock single-pass kernel
over the full cache and with a NumPy reference; the KV write of the new pass is
compared with the stock kernel's. Outputs are bf16 from two different
reduction orders, so the gate is a band (fraction of elements within tolerance
and the p99.9 absolute difference), never equality.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.adapter import attention, cp_attention, kv_split_attention

pytestmark = pytest.mark.tpu_test

_HEAD_DIM = 256
_PAGE_SIZE = 128
_OWN_Q_HEADS = 4
_K_SCALE = 0.5
_V_SCALE = 0.25
_FP8 = torch.float8_e4m3fn
_SM_SCALE = _HEAD_DIM**-0.5


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("The batched RPA kernels require a TPU")


def _write_slot(cache, layout, page, offset, key, value):
    if layout == "HND":
        cache[page, 0, :, :, offset] = key.reshape(_HEAD_DIM // 4, 4)
        cache[page, 1, :, :, offset] = value.reshape(_HEAD_DIM // 4, 4)
    else:
        cache[page, offset, 0, 0, :] = key
        cache[page, offset, 0, 1, :] = value


def _reference(query, keys, values, causal_offset):
    """Softmax attention of ``query`` rows over ``keys``.

    Row j sees positions < causal_offset + j + 1.
    """
    scores = query @ (keys * _K_SCALE).T * _SM_SCALE
    positions = np.arange(keys.shape[0])[None, :]
    allowed = positions <= (causal_offset + np.arange(query.shape[0]))[:, None]
    scores = np.where(allowed, scores, -np.inf)
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    return weights @ (values * _V_SCALE) / weights.sum(axis=-1, keepdims=True)


def _make_case(layout, group_size, q_len):
    rng = np.random.default_rng(20261018)
    # Lengths around every page boundary of the group's super page, plus long
    # sequences so each rank owns several pages.
    kv_lens = [
        page * _PAGE_SIZE + delta
        for page in range(1, group_size + 2)
        for delta in (-1, 0, 1)
    ] + [9 * _PAGE_SIZE + 37, 11 * _PAGE_SIZE + 3]
    kv_lens = [max(length, q_len) for length in kv_lens]
    num_seqs = len(kv_lens)
    pages_per_seq = 12
    group_heads = group_size * _OWN_Q_HEADS
    total_q = num_seqs * q_len

    def random_tensor(shape, dtype):
        data = rng.normal(0, 0.5, shape).astype(np.float32)
        return torch.from_numpy(data).to(dtype)

    queries = random_tensor((total_q, group_heads, _HEAD_DIM), torch.bfloat16)
    keys = random_tensor((num_seqs, max(kv_lens), _HEAD_DIM), _FP8).float().numpy()
    values = random_tensor((num_seqs, max(kv_lens), _HEAD_DIM), _FP8).float().numpy()

    num_pages = num_seqs * pages_per_seq + 4
    page_table = rng.permutation(num_pages)[: num_seqs * pages_per_seq]
    page_table = page_table.reshape(num_seqs, pages_per_seq).astype(np.int32)
    shape = attention.PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        num_pages, _PAGE_SIZE, 1, _HEAD_DIM, _FP8
    )
    cache = np.zeros(shape, dtype=np.float32)
    expected = []
    new_keys, new_values = [], []
    for seq, kv_len in enumerate(kv_lens):
        cache_len = kv_len - q_len
        for pos in range(cache_len):
            _write_slot(
                cache,
                layout,
                page_table[seq, pos // _PAGE_SIZE],
                pos % _PAGE_SIZE,
                keys[seq, pos],
                values[seq, pos],
            )
        rows = queries[seq * q_len : (seq + 1) * q_len].float().numpy()
        expected.append(
            np.stack(
                [
                    _reference(
                        rows[:, head],
                        keys[seq, :kv_len],
                        values[seq, :kv_len],
                        cache_len,
                    )
                    for head in range(group_heads)
                ],
                axis=1,
            )
        )
        new_keys.append(keys[seq, cache_len:kv_len])
        new_values.append(values[seq, cache_len:kv_len])

    args = dict(
        cache=torch.from_numpy(cache).to(_FP8),
        queries=queries,
        keys=torch.from_numpy(np.concatenate(new_keys)[:, None, :]).to(_FP8),
        values=torch.from_numpy(np.concatenate(new_values)[:, None, :]).to(_FP8),
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32),
        page_table=torch.from_numpy(page_table.reshape(-1)),
        cu_q_lens=torch.arange(num_seqs + 1, dtype=torch.int32) * q_len,
        distribution=torch.full((3,), num_seqs, dtype=torch.int32),
    )
    return args, np.concatenate(expected, axis=0)


def _jnp(tensor):
    dtype_map = {
        _FP8: jnp.float8_e4m3fn,
        torch.bfloat16: jnp.bfloat16,
        torch.int32: jnp.int32,
    }
    return jnp.asarray(tensor.float().numpy(), dtype=dtype_map[tensor.dtype])


def _kv_layout(layout):
    return (
        configs.KVLayout.SEQ_ALONG_LANE
        if layout == "HND"
        else configs.KVLayout.HEAD_ALONG_SUBLANE
    )


def _stock(args, layout, q_len, heads):
    output, updated_cache = wrapper.ragged_paged_attention(
        _jnp(args["queries"][:, heads]),
        _jnp(args["keys"]),
        _jnp(args["values"]),
        _jnp(args["cache"]),
        _jnp(args["kv_lens"]),
        _jnp(args["page_table"]),
        _jnp(args["cu_q_lens"]),
        _jnp(args["distribution"]),
        sm_scale=_SM_SCALE,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        kv_layout=_kv_layout(layout),
        decode_query_size=q_len,
    )
    return np.asarray(output.astype(jnp.float32)), np.asarray(
        updated_cache.astype(jnp.float32)
    )


def _pass(args, layout, q_len, query, pages, group_size, rank, scope):
    updated_cache, output, lse = cp_attention._pallas_rpa_kernel_dcp(
        _jnp(args["cache"]),
        _jnp(query),
        _jnp(args["keys"]),
        _jnp(args["values"]),
        _jnp(args["kv_lens"]),
        _jnp(pages),
        _jnp(args["cu_q_lens"]),
        _jnp(args["distribution"]),
        None,
        _K_SCALE,
        _V_SCALE,
        sliding_window=None,
        sm_scale=_SM_SCALE,
        soft_cap=None,
        cp_group_size=group_size,
        cp_rank_val=rank,
        attention_scope=scope,
        kv_layout=_kv_layout(layout),
        decode_query_size=q_len,
    )
    return (
        torch.from_numpy(np.asarray(output.astype(jnp.float32))),
        torch.from_numpy(np.asarray(lse.astype(jnp.float32))),
        np.asarray(updated_cache.astype(jnp.float32)),
    )


def _merge(outputs, lses):
    lses = torch.stack(lses)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - merged_lse).unsqueeze(-1)
    weighted = torch.where(weights > 0, torch.stack(outputs) * weights, 0.0)
    return weighted.sum(0), merged_lse


def _band(actual, expected, atol, rtol):
    diff = np.abs(actual - expected)
    within = diff <= atol + rtol * np.abs(expected)
    return within.mean(), np.quantile(diff, 0.999)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("group_size", [2, 4])
@pytest.mark.parametrize("q_len", [1, 4], ids=lambda q: f"q{q}")
def test_split_matches_stock_and_reference(
    layout, group_size, q_len, vllm_config_context
):
    from vllm.config import get_current_vllm_config

    get_current_vllm_config().cache_config.kv_cache_layout = (
        "LBHNC" if layout == "HND" else "LBNHC"
    )
    args, expected = _make_case(layout, group_size, q_len)
    num_seqs = args["kv_lens"].shape[0]

    ctx_outputs, ctx_lses = [], []
    for rank in range(group_size):
        pages = kv_split_attention.rank_page_view(
            args["page_table"], num_seqs, group_size, rank
        )
        output, lse, cache_after = _pass(
            args,
            layout,
            q_len,
            args["queries"],
            pages,
            group_size,
            rank,
            configs.AttentionScope.CACHE_ONLY,
        )
        np.testing.assert_array_equal(
            cache_after, args["cache"].float().numpy(), "CACHE_ONLY wrote the pool"
        )
        ctx_outputs.append(output)
        ctx_lses.append(lse)
    ctx_output, ctx_lse = _merge(ctx_outputs, ctx_lses)

    for rank in range(group_size):
        heads = slice(rank * _OWN_Q_HEADS, (rank + 1) * _OWN_Q_HEADS)
        new_output, new_lse, cache_after_new = _pass(
            args,
            layout,
            q_len,
            args["queries"][:, heads],
            args["page_table"],
            1,
            0,
            configs.AttentionScope.NEW_TOKENS_ONLY,
        )
        output, _ = cp_attention.merge_attn_results(
            ctx_output[:, heads], ctx_lse[:, heads], new_output, new_lse
        )
        output = output.to(torch.bfloat16).float().numpy()
        stock_output, stock_cache = _stock(args, layout, q_len, heads)

        # Against the NumPy reference, the DCP test's tolerance.
        np.testing.assert_allclose(output, expected[:, heads], atol=3e-3, rtol=1e-2)
        # Against the stock kernel: a band, since the two reduce in different orders.
        close, p999 = _band(output, stock_output, atol=2e-3, rtol=1e-2)
        print(
            f"[kv-split] {layout} G={group_size} q{q_len} rank {rank}: close fraction "
            f"{close:.5f}, p99.9 |diff| {p999:.3e}"
        )
        assert close >= 0.999, (close, p999)
        assert p999 <= 1.5e-2, (close, p999)
        # The new pass writes the same quantized bytes as the stock kernel.
        np.testing.assert_allclose(cache_after_new, stock_cache, atol=0, rtol=0)
