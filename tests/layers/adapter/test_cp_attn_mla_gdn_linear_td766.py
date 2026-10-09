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
"""Unit tests for TD7.66: cp_attention, cp_mla_attention, and gdn_linear."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.layers.adapter import (
    cp_attention,
    cp_mla_attention,
    gdn_linear,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# 1. cp_attention.py tests
# ---------------------------------------------------------------------------


def test_lse_weighted_combine_finite_and_non_finite() -> None:
    T, H, D = 3, 2, 4
    out_a = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [float("nan"), 1.0, 1.0, 1.0]],
            [[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]],
            [[5.0, 5.0, 5.0, 5.0], [float("inf"), -float("inf"), 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    out_b = torch.tensor(
        [
            [[3.0, 6.0, 9.0, 12.0], [2.0, 4.0, 6.0, 8.0]],
            [[0.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]],
            [[7.0, 7.0, 7.0, 7.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    lse_a = torch.tensor(
        [
            [0.0, -float("inf")],
            [1.0, 2.0],
            [-float("inf"), -float("inf")],
        ],
        dtype=torch.float32,
    )
    lse_b = torch.tensor(
        [
            [0.0, 1.5],
            [-float("inf"), 2.0],
            [-float("inf"), -float("inf")],
        ],
        dtype=torch.float32,
    )

    out, lse = cp_attention.lse_weighted_combine(out_a, lse_a, out_b, lse_b)
    assert out.shape == (T, H, D)
    assert lse.shape == (T, H)
    # Row 0, head 0: equal LSEs -> average of out_a and out_b
    torch.testing.assert_close(out[0, 0], torch.tensor([2.0, 4.0, 6.0, 8.0]))
    torch.testing.assert_close(lse[0, 0], torch.log(torch.tensor(2.0)))
    # Row 0, head 1: lse_a is -inf (with NaN in out_a) -> pure out_b
    torch.testing.assert_close(out[0, 1], out_b[0, 1])
    torch.testing.assert_close(lse[0, 1], torch.tensor(1.5))
    # Row 1, head 0: lse_b is -inf -> pure out_a
    torch.testing.assert_close(out[1, 0], out_a[1, 0])
    torch.testing.assert_close(lse[1, 0], torch.tensor(1.0))
    # Row 2: both LSEs are -inf -> zero output and -inf LSE
    torch.testing.assert_close(out[2], torch.zeros(H, D))
    assert torch.all(torch.isneginf(lse[2]))


def test_dcp_allgather_lse_combine_single_and_multi_rank() -> None:
    out0 = torch.ones(2, 3, 4, dtype=torch.float32)
    lse0 = torch.zeros(2, 3, dtype=torch.float32)

    # Fast path: dcp_world_size <= 1 does not call dcp_group.all_gather
    exploding_group = SimpleNamespace(
        all_gather=MagicMock(side_effect=AssertionError("should not be called"))
    )
    res_out, res_lse = cp_attention.dcp_allgather_lse_combine(
        out0, lse0, exploding_group, dcp_world_size=1
    )
    assert res_out is out0
    assert res_lse is lse0
    exploding_group.all_gather.assert_not_called()

    # Multi-rank path: dcp_world_size = 2
    out1 = torch.full((2, 3, 4), 3.0, dtype=torch.float32)
    lse1 = torch.zeros(2, 3, dtype=torch.float32)

    def _fake_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        if tensor.ndim == 3:
            return torch.cat([out0, out1], dim=0)
        return torch.cat([lse0, lse1], dim=0)

    group = SimpleNamespace(all_gather=_fake_all_gather)
    comb_out, comb_lse = cp_attention.dcp_allgather_lse_combine(
        out0, lse0, group, dcp_world_size=2
    )
    torch.testing.assert_close(comb_out, torch.full((2, 3, 4), 2.0))
    torch.testing.assert_close(
        comb_lse, torch.full((2, 3), float(torch.log(torch.tensor(2.0))))
    )


def test_pallas_rpa_kernel_dcp_and_kernel_builders(monkeypatch) -> None:
    # 1. Test _pallas_rpa_kernel_dcp directly by stubbing ragged_paged_attention
    recorded_kwargs: dict = {}

    def _fake_rpa(**kwargs):
        recorded_kwargs.update(kwargs)
        return "fake_out", "fake_kv_cache", "fake_lse"

    monkeypatch.setattr(
        cp_attention._batched_rpa_wrapper, "ragged_paged_attention", _fake_rpa
    )
    new_kv, out, lse = cp_attention._pallas_rpa_kernel_dcp(
        "kv_cache",
        "query",
        "key",
        "value",
        "seq_lens",
        "block_tables",
        "query_start_loc",
        "request_distribution",
        0.5,
        0.25,
        0.125,
        sliding_window=128,
        sm_scale=0.1,
        soft_cap=30.0,
        cp_group_size=2,
        cp_rank_val=1,
        attention_scope=_batched_rpa_configs.AttentionScope.CACHE_ONLY,
        kv_layout=_batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_query_size=2,
    )
    assert (new_kv, out, lse) == ("fake_kv_cache", "fake_out", "fake_lse")
    assert recorded_kwargs["cp_group_size"] == 2
    assert int(recorded_kwargs["cp_rank"][0]) == 1
    assert recorded_kwargs["return_lse"] is True
    assert recorded_kwargs["decode_query_size"] == 2

    # 2. Test _fake_dcp directly
    kv_t = torch.zeros(4, 8, dtype=torch.bfloat16)
    q_t = torch.zeros(3, 5, 16, dtype=torch.bfloat16)
    fake_kv, fake_out, fake_lse = cp_attention._fake_dcp(kv_t, q_t)
    assert fake_kv.shape == kv_t.shape
    assert fake_out.shape == q_t.shape
    assert fake_lse.shape == (3, 5)

    # 3. Test _build_dcp_kernel_op and build_dcp_kernels cache branches
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})

    def _fake_jax_op(op_name, fn, **kwargs):
        class _Op:
            def __init__(self):
                self.fake_fn = None

            def register_fake(self, f):
                self.fake_fn = f

            def __call__(self, kv_cache, query, *args, **kw):
                new_kv = kv_cache + 1.0
                out = query * 2.0
                lse = torch.zeros(query.shape[0], query.shape[1], dtype=torch.float32)
                return new_kv, out, lse

        return _Op()

    monkeypatch.setattr(cp_attention.pallas, "jax_op", _fake_jax_op)

    # Cold miss for both cache and new kernels
    cache_k, new_k = cp_attention.build_dcp_kernels(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=0,
    )
    # Warm hit for both cache and new kernels
    cache_k2, new_k2 = cp_attention.build_dcp_kernels(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=0,
    )
    assert cache_k is cache_k2
    assert new_k is new_k2

    # Partial cache hit 1: cached_cache present, cached_new missing
    base_key_rank1 = (
        None,
        1.0,
        None,
        None,
        None,
        None,
        2,
        1,
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        1,
    )
    cp_attention._DCP_KERNEL_REGISTRY[("dcp_cache", *base_key_rank1)] = cache_k
    ck_a, nk_a = cp_attention.build_dcp_kernels(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=1,
    )
    assert ck_a is cache_k
    assert callable(nk_a)

    # Partial cache hit 2: cached_cache missing, cached_new present
    base_key_rank2 = (
        None,
        1.0,
        None,
        None,
        None,
        None,
        4,
        2,
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        1,
    )
    cp_attention._DCP_KERNEL_REGISTRY[("dcp_new", *base_key_rank2)] = new_k
    ck_b, nk_b = cp_attention.build_dcp_kernels(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=4,
        cp_rank=2,
    )
    assert callable(ck_b)
    assert nk_b is new_k

    # Execute kernel_impl to cover kv_cache.copy_ and return
    kv_buf = torch.zeros(2, 4)
    q_buf = torch.ones(3, 2, 4)
    out_res, lse_res = cache_k(kv_buf, q_buf)
    torch.testing.assert_close(kv_buf, torch.ones(2, 4))
    torch.testing.assert_close(out_res, torch.full((3, 2, 4), 2.0))
    assert lse_res.shape == (3, 2)


def test_forward_with_dcp_branches(monkeypatch) -> None:
    # 1. Error branch when dcp_group is None
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: None)
    with pytest.raises(RuntimeError, match="DCP group not initialized"):
        cp_attention.forward_with_dcp(
            layer=SimpleNamespace(),
            query=torch.zeros(2, 2, 4),
            key=torch.zeros(2, 1, 4),
            value=torch.zeros(2, 1, 4),
            kv_cache=torch.zeros(4, 4),
            attn_metadata=SimpleNamespace(),
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            dcp_world_size=2,
            dcp_rank=0,
        )

    # 2. Normal execution with both is_compiling() == False and True
    class _FakeGroup:
        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: _FakeGroup())
    sync_calls: list[tuple] = []
    monkeypatch.setattr(
        cp_attention,
        "synchronize_tensors",
        lambda t, wait=False: sync_calls.append((t, wait)),
    )

    def _fake_build_dcp_kernels(**kwargs):
        def _cache_k(kv_cache, q_all, *args):
            return (
                torch.ones_like(q_all),
                torch.zeros(q_all.shape[0], q_all.shape[1]),
            )

        def _new_k(kv_cache, q_own, *args):
            return (
                torch.full_like(q_own, 3.0),
                torch.zeros(q_own.shape[0], q_own.shape[1]),
            )

        return _cache_k, _new_k

    monkeypatch.setattr(cp_attention, "build_dcp_kernels", _fake_build_dcp_kernels)

    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    meta = SimpleNamespace(
        seq_lens=torch.tensor([4, 4]),
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2]),
        request_distribution=torch.tensor([0, 0, 2]),
    )
    q = torch.ones(2, 3, 4)
    k = torch.ones(2, 1, 4)
    v = torch.ones(2, 1, 4)
    kv_cache = torch.zeros(8, 4)

    # 2a. Not compiling + quantized kv_cache_quantized_dtype
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    out = cp_attention.forward_with_dcp(
        layer=layer,
        query=q,
        key=k,
        value=v,
        kv_cache=kv_cache,
        attn_metadata=meta,
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
        dcp_world_size=2,
        dcp_rank=1,
    )
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, torch.full((2, 3, 4), 5.0 / 3.0))
    assert len(sync_calls) == 1
    assert sync_calls[0][1] is False

    # 2b. Compiling + unquantized kv_cache_quantized_dtype
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    out_comp = cp_attention.forward_with_dcp(
        layer=layer,
        query=q,
        key=k,
        value=v,
        kv_cache=kv_cache,
        attn_metadata=meta,
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        kv_cache_quantized_dtype=None,
        dcp_world_size=2,
        dcp_rank=0,
    )
    assert out_comp.shape == (2, 3, 4)
    assert len(sync_calls) == 1  # Not incremented when compiling


# ---------------------------------------------------------------------------
# 2. cp_mla_attention.py tests
# ---------------------------------------------------------------------------


def test_cp_mla_attention_all_gather_and_merge(monkeypatch) -> None:
    saved_attrs = {
        name: getattr(cp_mla_attention, name)
        for name in (
            "all_gather_heads",
            "all_gather_q_nope_pe_heads",
            "merge_lse_partials_scatter_heads",
            "_weight_by_lse",
        )
    }
    try:
        importlib.reload(cp_mla_attention)

        T, H_local, D_nope, D_pe = 3, 2, 4, 2
        q = torch.arange(T * H_local * D_nope, dtype=torch.float32).reshape(
            T, H_local, D_nope
        )
        ql_nope = q.clone()
        q_pe = torch.arange(T * H_local * D_pe, dtype=torch.float32).reshape(
            T, H_local, D_pe
        )

        # 1. No-op when group is None or world_size == 1
        monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: None)
        assert cp_mla_attention.all_gather_heads(q) is q
        nope_res, pe_res = cp_mla_attention.all_gather_q_nope_pe_heads(
            ql_nope, q_pe, D_nope
        )
        assert nope_res is ql_nope and pe_res is q_pe
        assert (
            cp_mla_attention.merge_lse_partials_scatter_heads(
                q, torch.zeros(T, H_local)
            )
            is q
        )

        group_ws1 = SimpleNamespace(world_size=1)
        monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: group_ws1)
        assert cp_mla_attention.all_gather_heads(q) is q
        nope_res1, pe_res1 = cp_mla_attention.all_gather_q_nope_pe_heads(
            ql_nope, q_pe, D_nope
        )
        assert nope_res1 is ql_nope and pe_res1 is q_pe
        assert (
            cp_mla_attention.merge_lse_partials_scatter_heads(
                q, torch.zeros(T, H_local)
            )
            is q
        )

        # 2. Multi-rank head-major dim=0 collectives
        class _HeadMajorGroup:
            world_size = 2

            def __init__(self, return_list_on_lse: bool = False):
                self.return_list_on_lse = return_list_on_lse

            def all_gather(self, tensor: torch.Tensor, dim: int = 0):
                assert dim == 0
                if self.return_list_on_lse and tensor.shape[0] == 1:
                    return [tensor, tensor + 1.0]
                return torch.cat([tensor, tensor + 10.0], dim=0)

            def reduce_scatter(
                self, tensor: torch.Tensor, dim: int = 0
            ) -> torch.Tensor:
                assert dim == 0
                c0, c1 = tensor.chunk(2, dim=0)
                return c0 + c1

        hm_group = _HeadMajorGroup(return_list_on_lse=False)
        monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: hm_group)

        q_gathered = cp_mla_attention.all_gather_heads(q)
        assert q_gathered.shape == (T, H_local * 2, D_nope)
        torch.testing.assert_close(q_gathered[:, :H_local, :], q)
        torch.testing.assert_close(q_gathered[:, H_local:, :], q + 10.0)

        nope_dcp, pe_dcp = cp_mla_attention.all_gather_q_nope_pe_heads(
            ql_nope, q_pe, D_nope
        )
        assert nope_dcp.shape == (T, H_local * 2, D_nope)
        assert pe_dcp.shape == (T, H_local * 2, D_pe)
        torch.testing.assert_close(nope_dcp[:, :H_local, :], ql_nope)
        torch.testing.assert_close(nope_dcp[:, H_local:, :], ql_nope + 10.0)
        torch.testing.assert_close(pe_dcp[:, :H_local, :], q_pe)
        torch.testing.assert_close(pe_dcp[:, H_local:, :], q_pe + 10.0)

        # 3. Test merge_lse_partials_scatter_heads & _weight_by_lse
        # Case A: f32 contiguous lse, Tensor gathered, with an all -inf token
        out_partial = torch.ones(T, H_local * 2, D_nope, dtype=torch.bfloat16)
        lse_partial = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [-float("inf"), -float("inf"), -float("inf"), -float("inf")],
                [0.5, -float("inf"), 1.5, 2.5],
            ],
            dtype=torch.float32,
        )
        merged_a = cp_mla_attention.merge_lse_partials_scatter_heads(
            out_partial, lse_partial
        )
        assert merged_a.shape == (T, H_local, D_nope)
        assert merged_a.dtype == torch.bfloat16
        # Token 1 had -inf across all ranks -> zero output
        torch.testing.assert_close(
            merged_a[1], torch.zeros(H_local, D_nope, dtype=torch.bfloat16)
        )

        # Case B: bf16 non-contiguous lse + list returned from group.all_gather
        hm_group_list = _HeadMajorGroup(return_list_on_lse=True)
        monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: hm_group_list)
        lse_bf16_noncontig = (
            lse_partial.to(torch.bfloat16).transpose(0, 1).transpose(0, 1)
        )
        weighted_b = cp_mla_attention._weight_by_lse(
            out_partial, lse_bf16_noncontig[:, ::1], hm_group_list
        )
        assert weighted_b.shape == (H_local * 2, T, D_nope)
        assert weighted_b.dtype == torch.float32
        assert weighted_b.is_contiguous()
    finally:
        for name, fn in saved_attrs.items():
            setattr(cp_mla_attention, name, fn)


# ---------------------------------------------------------------------------
# 3. gdn_linear.py tests
# ---------------------------------------------------------------------------


def _patch_tp(monkeypatch, tp_size: int, tp_rank: int) -> None:
    import vllm.distributed as distributed_mod
    from vllm.model_executor import parameter as parameter_mod
    from vllm.model_executor.layers import linear as linear_mod
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: tp_rank)
    for mod in (linear_mod, parameter_mod, distributed_mod):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda: tp_size
        )


def test_gdn_column_parallel_linear_all_branches(monkeypatch) -> None:
    _patch_tp(monkeypatch, tp_size=2, tp_rank=1)
    geometry = derive_gdn_head_geometry(num_kq_heads=1, num_v_heads=4, parallel_size=2)
    layer = gdn_linear.GdnColumnParallelLinear(
        input_size=16,
        geometry=geometry,
        d_k=8,
        d_v=8,
        include_z=True,
        quant_config=None,
        params_dtype=torch.float32,
        prefix="test_gdn_col",
    )
    # Checkpoint sizes: [8, 8, 32, 32], replication factors: [2, 2, 1, 1]
    assert layer.checkpoint_sizes == [8, 8, 32, 32]
    assert layer.replication_factors == [2, 2, 1, 1]

    # 1. PerTensorScaleParameter branch
    pt_scale = PerTensorScaleParameter(
        data=torch.zeros(4, dtype=torch.float32),
        weight_loader=layer.weight_loader,
    )
    layer.weight_loader(pt_scale, torch.tensor(1.5), loaded_shard_id=0)
    torch.testing.assert_close(pt_scale.data[0], torch.tensor(1.5))

    # 2. Normal weight parameter: loaded_shard_id=None (full fused checkpoint)
    full_ckpt = torch.arange(
        sum(layer.checkpoint_sizes) * 16, dtype=torch.float32
    ).reshape(sum(layer.checkpoint_sizes), 16)
    layer.weight_loader(layer.weight, full_ckpt, loaded_shard_id=None)

    # 3. Normal weight parameter: loaded_shard_id=(0, 1, 2) tuple + int shard 3
    qkv_ckpt = full_ckpt[: 8 + 8 + 32]
    z_ckpt = full_ckpt[8 + 8 + 32 :]
    layer.weight_loader(layer.weight, qkv_ckpt, loaded_shard_id=(0, 1, 2))
    layer.weight_loader(layer.weight, z_ckpt, loaded_shard_id=3)

    # 4. BlockQuantScaleParameter branch in _parameter_slice
    layer.weight_block_size = [8, 8]
    blk_scale = BlockQuantScaleParameter(
        data=torch.zeros(
            sum(layer.output_partition_sizes) // 8, 2, dtype=torch.float32
        ),
        output_dim=0,
        input_dim=1,
        weight_loader=layer.weight_loader,
    )
    size_s, off_s = layer._parameter_slice(blk_scale, 16, 8)
    assert (size_s, off_s) == (2, 1)


def test_gdn_interleaved_column_parallel_linear_all_branches(
    monkeypatch,
) -> None:
    _patch_tp(monkeypatch, tp_size=2, tp_rank=1)
    geometry = derive_gdn_head_geometry(num_kq_heads=1, num_v_heads=4, parallel_size=2)
    layer = gdn_linear.GdnInterleavedColumnParallelLinear(
        geometry=geometry,
        group_sizes=[8, 8, 32, 32],
        replicated_segments=[True, True, False, False],
        input_size=16,
        quant_config=None,
        prefix="test_gdn_interleaved",
    )
    assert layer.local_group_sizes == [8, 8, 16, 16]

    # 1. PerTensorScaleParameter branch
    pt_scale = PerTensorScaleParameter(
        data=torch.zeros(1, dtype=torch.float32),
        weight_loader=layer.weight_loader,
    )
    layer.weight_loader(pt_scale, torch.tensor(2.5), loaded_shard_id=0)
    torch.testing.assert_close(pt_scale.data[0], torch.tensor(2.5))

    # 2. Normal weight parameter
    ckpt_rows = sum(layer.group_sizes) * geometry.num_kq_heads
    ckpt = torch.arange(ckpt_rows * 16, dtype=torch.float32).reshape(ckpt_rows, 16)
    layer.weight.data = layer.weight.data.to(torch.float32)
    layer.weight_loader(layer.weight, ckpt)
    assert layer.weight.shape == (sum(layer.local_group_sizes), 16)

    # 3. BlockQuantScaleParameter branch: aligned vs misaligned error
    layer.weight_block_size = [8, 8]
    blk_scale = BlockQuantScaleParameter(
        data=torch.zeros(sum(layer.local_group_sizes) // 8, 2, dtype=torch.float32),
        output_dim=0,
        input_dim=1,
        weight_loader=layer.weight_loader,
    )
    blk_ckpt = torch.arange((ckpt_rows // 8) * 2, dtype=torch.float32).reshape(
        ckpt_rows // 8, 2
    )
    layer.weight_loader(blk_scale, blk_ckpt)

    # Misaligned block size raises ValueError
    layer.weight_block_size = [16, 8]
    with pytest.raises(ValueError, match="GDN interleaved scale blocks must align"):
        layer.weight_loader(blk_scale, blk_ckpt)

    # 4. ChannelQuantScaleParameter branch
    ch_scale = ChannelQuantScaleParameter(
        data=torch.zeros(sum(layer.local_group_sizes), 1, dtype=torch.float32),
        output_dim=0,
        weight_loader=layer.weight_loader,
    )
    ch_ckpt_1d = torch.arange(ckpt_rows, dtype=torch.float32)
    layer.weight_loader(ch_scale, ch_ckpt_1d)
    assert ch_scale.shape == (sum(layer.local_group_sizes), 1)

    # Ensure ModelWeightParameter import is exercised
    assert isinstance(layer.weight, ModelWeightParameter)
