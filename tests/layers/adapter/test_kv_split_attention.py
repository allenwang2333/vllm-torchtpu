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
"""CPU tests for the split KV read over the ranks sharing a KV head (TPU_RPA_KV_SPLIT).

The kernel builder and the group are replaced; what is checked is the group
derivation, the per-rank page-table view, which inputs reach which pass, the
merge, and the adapter's routing of decode buckets. Kernel correctness on TPU
is covered by test_attention_kv_split_tpu.py.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.distributed import kv_head_group
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import (
    AttentionScope,
    KVLayout,
)
from vllm_torchtpu.layers.adapter import attention, cp_attention, kv_split_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.mark.parametrize(
    "world, tp, kv_heads, expected",
    [
        (8, 8, 2, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (8, 4, 2, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        (16, 8, 2, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
        (8, 2, 2, None),
        (8, 8, 8, None),
        (8, 8, 3, None),
    ],
)
def test_kv_head_group_ranks(world, tp, kv_heads, expected):
    assert kv_head_group.kv_head_group_ranks(world, tp, kv_heads) == expected


def test_rank_page_view_interleaves_and_pads():
    table = torch.arange(2 * 6, dtype=torch.int32) + 100
    # 6 pages per sequence, group of 4: padded to 8, rank 1 reads pages 1 and 5.
    view = kv_split_attention.rank_page_view(table, 2, 4, 1)
    assert view.tolist() == [101, 105, 107, 111]
    # Rank 3 reads page 3 and the pad.
    view = kv_split_attention.rank_page_view(table, 2, 4, 3)
    assert view.tolist() == [103, 0, 109, 0]
    # No padding when divisible.
    view = kv_split_attention.rank_page_view(table, 2, 2, 0)
    assert view.tolist() == [100, 102, 104, 106, 108, 110]


class _Group:
    """A G-rank group where every rank holds the same tensors."""

    def __init__(self, size, rank):
        self.world_size = size
        self.rank_in_group = rank
        self.calls = []

    def all_gather(self, tensor, dim):
        self.calls.append(("all_gather", dim, tuple(tensor.shape)))
        return torch.cat([tensor] * self.world_size, dim=dim)

    def reduce_scatter(self, tensor, dim):
        self.calls.append(("reduce_scatter", dim, tuple(tensor.shape)))
        return (self.world_size * tensor).chunk(self.world_size, dim)[
            self.rank_in_group
        ]


def _reference_merge(values, lses):
    lses = torch.stack(lses)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - merged_lse).unsqueeze(-1)
    return (torch.stack(values) * weights).sum(0)


def test_forward_routes_pages_heads_and_merges():
    group = _Group(4, 2)
    calls = []

    def cache_kernel(kv_cache, query, key, value, seq_lens, pages, qsl, dist, *scales):
        calls.append(("cache", tuple(query.shape), pages.tolist()))
        return torch.full_like(query, 2.0), torch.full(query.shape[:2], 1.0)

    def new_kernel(kv_cache, query, key, value, seq_lens, pages, qsl, dist, *scales):
        calls.append(("new", tuple(query.shape), pages.tolist()))
        return torch.full_like(query, 6.0), torch.full(query.shape[:2], 0.0)

    kernels = kv_split_attention.KVSplitKernels(
        group=group,
        group_size=4,
        rank=2,
        cache_kernel=cache_kernel,
        new_kernel=new_kernel,
        max_tokens=8,
    )
    num_seqs, heads, dim = 2, 4, 8
    query = torch.ones(num_seqs, heads, dim)
    table = torch.arange(num_seqs * 8, dtype=torch.int32) + 10
    attn_metadata = SimpleNamespace(
        seq_lens=torch.tensor([300, 500], dtype=torch.int32),
        block_tables=table,
        query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32),
        request_distribution=torch.full((3,), num_seqs, dtype=torch.int32),
    )
    out = kv_split_attention.forward(
        kernels,
        query=query,
        key=query[:, :1],
        value=query[:, :1],
        kv_cache=torch.zeros(4),
        attn_metadata=attn_metadata,
        q_scale=None,
        k_scale=0.5,
        v_scale=0.25,
    )
    # The cache pass sees the group's 16 heads and this rank's pages 2 and 6 of
    # each sequence; the new pass sees the local heads and the full table.
    assert calls[0] == ("cache", (num_seqs, 16, dim), [12, 16, 20, 24])
    assert calls[1] == ("new", (num_seqs, heads, dim), table.tolist())
    assert group.calls[0] == ("all_gather", 1, (num_seqs, heads, dim))
    assert group.calls[1] == ("all_gather", 0, (1, num_seqs, 16))
    assert group.calls[2] == ("reduce_scatter", 1, (num_seqs, 16, dim))
    # Four equal partials with lse 1 merge to lse 1 + ln 4 and value 2; the new
    # pass with lse 0 and value 6 then merges by log-sum-exp.
    ctx_lse = torch.full((num_seqs, heads), 1.0 + torch.log(torch.tensor(4.0)).item())
    expected = _reference_merge(
        [
            torch.full((num_seqs, heads, dim), 2.0),
            torch.full((num_seqs, heads, dim), 6.0),
        ],
        [ctx_lse, torch.zeros(num_seqs, heads)],
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def _impl(head_size=256):
    return attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=4,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )


def _config(tp=8):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 2),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )


@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC"])
def test_initialize_builds_split_kernels_for_decode_buckets(
    monkeypatch, layout, cpu_vllm_config_context
):
    monkeypatch.setenv("TPU_RPA_KV_SPLIT", "1")
    built = []

    def build_op(_name, fn):
        built.append(
            (
                fn.keywords["cp_group_size"],
                fn.keywords["cp_rank_val"],
                fn.keywords["attention_scope"],
                fn.keywords["kv_layout"],
            )
        )
        return lambda *a: None

    group = _Group(4, 3)
    monkeypatch.setattr(
        kv_split_attention, "get_kv_head_group", lambda tp, heads: group
    )
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", build_op)
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_kernel",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_schedule_op",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_scheduled_kernel",
        lambda self, *a, **k: None,
    )
    get_current_vllm_config().cache_config.kv_cache_layout = layout
    impl = _impl()
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
        impl.initialize_kernel(layer)

    expected_layout = (
        KVLayout.SEQ_ALONG_LANE if layout == "LBHNC" else KVLayout.HEAD_ALONG_SUBLANE
    )
    # Both scopes of the (4, 3) pair are registered together; only CACHE_ONLY
    # is used. The new pass is the (1, 0) pair's NEW_TOKENS_ONLY op.
    assert sorted(built, key=lambda b: (b[0], b[2].value)) == sorted(
        [
            (4, 3, AttentionScope.CACHE_ONLY, expected_layout),
            (4, 3, AttentionScope.NEW_TOKENS_ONLY, expected_layout),
            (1, 0, AttentionScope.CACHE_ONLY, expected_layout),
            (1, 0, AttentionScope.NEW_TOKENS_ONLY, expected_layout),
        ],
        key=lambda b: (b[0], b[2].value),
    )
    assert impl.kv_split is not None
    assert (impl.kv_split.group_size, impl.kv_split.rank) == (4, 3)
    # max_num_seqs 32 x decode_query_size 1 (no speculative config).
    assert impl.kv_split.max_tokens == 32


def test_initialize_without_duplicated_kv_head_leaves_split_off(
    monkeypatch, cpu_vllm_config_context
):
    monkeypatch.setenv("TPU_RPA_KV_SPLIT", "1")
    monkeypatch.setattr(kv_split_attention, "get_kv_head_group", lambda tp, heads: None)
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_kernel",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_schedule_op",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_build_rpa_scheduled_kernel",
        lambda self, *a, **k: None,
    )
    impl = _impl()
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config(tp=2)):
        impl.initialize_kernel(layer)
    assert impl.kv_split is None
