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
"""CPU unit tests for MLA dispatch, kv_cache_utils, scatter_utils, and csa_mask."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter_utils
from vllm_torchtpu.kernels.mla.masked_dense import csa_mask
from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
    MaskedDensePrefillCostModel,
)


class _DummyRef:
    """Minimal array reference wrapper for exercising Pallas kernel bodies on CPU."""

    def __init__(self, array: np.ndarray):
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self):
        return jnp.dtype(self.array.dtype)

    def bitcast(self, target_dtype):
        np_dtype = np.dtype(target_dtype)
        # Fold 4 uint8 rows into 1 int32 row along axis 0 when bitcasting to int32.
        if self.array.dtype == np.uint8 and np_dtype == np.int32:
            return _DummyRef(
                self.array.reshape(-1, self.array.shape[-1]).view(np.int32)
            )
        return _DummyRef(self.array.view(np_dtype))

    def at(self, *slices):
        return (self, slices)

    def __getitem__(self, item):
        if item is Ellipsis:
            return jnp.asarray(self.array)
        return jnp.asarray(self.array[self._normalize_index(item)])

    def __setitem__(self, item, value):
        self.array[self._normalize_index(item)] = np.asarray(value)

    @staticmethod
    def _normalize_index(item):
        def _conv(idx):
            if hasattr(idx, "start") and hasattr(idx, "size"):
                start = int(idx.start)
                size = int(idx.size)
                return slice(start, start + size)
            if isinstance(idx, jax.Array):
                return int(idx)
            return idx

        if isinstance(item, tuple):
            return tuple(_conv(x) for x in item)
        return _conv(item)


class _DummyAtRef(_DummyRef):
    """Ref whose `.at[...]` indexing returns a slice descriptor."""

    class _AtHelper:
        def __init__(self, parent: "_DummyAtRef"):
            self._parent = parent

        def __getitem__(self, item):
            return (self._parent, item)

    @property
    def at(self):  # type: ignore[override]
        return self._AtHelper(self)

    def bitcast(self, target_dtype):
        np_dtype = np.dtype(target_dtype)
        if self.array.dtype == np.uint8 and np_dtype == np.int32:
            reshaped = self.array.reshape(self.array.shape[0] // 4, -1)
            return _DummyAtRef(reshaped.astype(np.int32))
        return _DummyAtRef(self.array.astype(np_dtype))


def test_dispatch_applies_and_select_tier_custom_cost_model():
    ql_nope = jnp.zeros(
        (
            mla_dispatch.MASKED_DENSE_MIN_TOKEN_BUCKET,
            mla_dispatch.GLM52_NUM_HEADS,
            mla_dispatch.GLM52_NOPE_DIM,
        ),
        dtype=jnp.bfloat16,
    )
    q_pe = jnp.zeros(
        (
            mla_dispatch.MASKED_DENSE_MIN_TOKEN_BUCKET,
            mla_dispatch.GLM52_NUM_HEADS,
            mla_dispatch.GLM52_ROPE_DIM,
        ),
        dtype=jnp.bfloat16,
    )
    kv_nope = jnp.zeros((4, mla_dispatch.GLM52_PAGE_SIZE, 4, 128), dtype=jnp.uint8)
    kv_rope = jnp.zeros((4, mla_dispatch.GLM52_PAGE_SIZE // 4, 4, 64), dtype=jnp.uint8)
    topk = jnp.zeros(
        (mla_dispatch.MASKED_DENSE_MIN_TOKEN_BUCKET, mla_dispatch.GLM52_INDEX_TOPK),
        dtype=jnp.int32,
    )

    # Fast-path rejections in `applies`
    assert not mla_dispatch.applies(
        ql_nope, q_pe, kv_nope, kv_rope, topk, cache_layout="sparsecore"
    )
    assert not mla_dispatch.applies(
        ql_nope[:16], q_pe[:16], kv_nope, kv_rope, topk[:16], cache_layout="tensorcore"
    )
    assert mla_dispatch.applies(
        ql_nope, q_pe, kv_nope, kv_rope, topk, cache_layout="tensorcore"
    )

    # Direct `split_point` without precomputed tuple and with precomputed tuple
    seq_lens = jnp.array([4096, 1024], dtype=jnp.int32)
    req_dist = jnp.array([0, 0, 2], dtype=jnp.int32)
    s0, c0 = mla_dispatch.split_point(seq_lens, req_dist, 2048)
    seq_ids = jnp.arange(2)
    s1, c1 = mla_dispatch.split_point(
        seq_lens,
        req_dist,
        2048,
        _precomputed=(seq_ids, seq_ids < 2, seq_ids < 0),
    )
    assert int(s0) == int(s1) == 1
    assert bool(c0) and bool(c1)

    # `select_tier` with explicit cost_model vs default None
    q_start = jnp.array([0, 512, 1024], dtype=jnp.int32)
    custom_model = MaskedDensePrefillCostModel(
        sparse_fixed_ns=100,
        sparse_token_ns=10,
        analytic_fixed_ns=10,
        analytic_token_ns=1,
        bitmap_fixed_ns=20,
        bitmap_token_ns=2,
        q_kv_block_ns=1,
    )
    tier_custom = mla_dispatch.select_tier(
        seq_lens, req_dist, q_start, (2048, 6144), cost_model=custom_model
    )
    tier_default = mla_dispatch.select_tier(
        seq_lens, req_dist, q_start, (0, 6144), cost_model=None
    )
    assert int(tier_custom) in (0, 1, 2)
    assert int(tier_default) in (0, 1, 2)


def test_dispatch_attention_from_parts_all_branch_counts(monkeypatch):
    num_rows = 1024
    num_heads = 4
    nope_dim = 8
    rope_dim = 4
    ql_nope = jnp.ones((num_rows, num_heads, nope_dim), dtype=jnp.bfloat16)
    q_pe = jnp.ones((num_rows, num_heads, rope_dim), dtype=jnp.bfloat16)
    kv_cache_nope = jnp.zeros((8, 1024, 4, 128), dtype=jnp.uint8)
    kv_cache_rope = jnp.zeros((8, 256, 4, 64), dtype=jnp.uint8)
    topk_indices = jnp.zeros((num_rows, 2048), dtype=jnp.int32)
    seq_lens = jnp.array([4096, 1024], dtype=jnp.int32)
    block_tables = jnp.arange(16, dtype=jnp.int32)
    query_start_loc = jnp.array([0, 512, 1024], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    monkeypatch.setattr(
        mla_dispatch.sparse_kernel,
        "prepare_q_inputs",
        lambda q: q,
    )
    monkeypatch.setattr(
        mla_dispatch.sparse_kernel,
        "prepare_gather_indices",
        lambda topk, bt, qsl, page_size, total_num_pages: (
            jnp.ones((num_rows,), dtype=jnp.int32),
            jnp.zeros((num_rows, 4), dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        mla_dispatch.sparse_kernel,
        "sparse_attention_chunk",
        lambda q, *args, **kwargs: q + 1,
    )
    monkeypatch.setattr(
        mla_dispatch,
        "masked_dense_ragged_paged_attention",
        lambda q, *args, **kwargs: q + 2,
    )

    # 1. Both tiers disabled -> len(branches) == 1
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (0, 0))
    out_sparse_only = mla_dispatch.attention_from_parts(
        ql_nope,
        q_pe,
        kv_cache_nope,
        kv_cache_rope,
        topk_indices,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sm_scale=1.0,
        k_scale=1.0,
    )
    assert out_sparse_only.shape == (num_rows, num_heads, nope_dim)

    # 2. Only bitmap tier enabled -> len(branches) == 2, tier_limits[0] == 0 (kept = 1)
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (0, 6144))
    out_two_branches_bitmap = mla_dispatch.attention_from_parts(
        ql_nope,
        q_pe,
        kv_cache_nope,
        kv_cache_rope,
        topk_indices,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sm_scale=1.0,
        k_scale=1.0,
    )
    assert out_two_branches_bitmap.shape == (num_rows, num_heads, nope_dim)

    # 3. Only analytic tier enabled -> len(branches) == 2 (kept = 0)
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (2048, 0))
    out_two_branches_analytic = mla_dispatch.attention_from_parts(
        ql_nope[:512],
        q_pe[:512],
        kv_cache_nope,
        kv_cache_rope,
        topk_indices[:512],
        jnp.array([1024, 1024], dtype=jnp.int32),
        block_tables,
        jnp.array([0, 256, 512], dtype=jnp.int32),
        request_distribution,
        sm_scale=1.0,
        k_scale=1.0,
    )
    assert out_two_branches_analytic.shape == (512, num_heads, nope_dim)

    # 4. Both tiers enabled -> len(branches) == 3 (`lax.switch`)
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (2048, 6144))
    out_three_branches = mla_dispatch.attention_from_parts(
        ql_nope,
        q_pe,
        kv_cache_nope,
        kv_cache_rope,
        topk_indices,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sm_scale=1.0,
        k_scale=1.0,
    )
    assert out_three_branches.shape == (num_rows, num_heads, nope_dim)


def test_order_prefills_by_kv_len_and_maybe_order():
    swaps: list[tuple[int, int]] = []
    req_ids: list[str | None] = ["d0", "p_short", "p_long", "p_mid"]
    num_computed = [10, 100, 500, 300]

    def swap_states(i: int, j: int) -> None:
        swaps.append((i, j))
        req_ids[i], req_ids[j] = req_ids[j], req_ids[i]
        num_computed[i], num_computed[j] = num_computed[j], num_computed[i]

    runner = SimpleNamespace(
        _masked_dense_groups_requests=True,
        input_batch=SimpleNamespace(
            req_ids=req_ids,
            num_computed_tokens_cpu=num_computed,
            swap_states=swap_states,
        ),
    )
    sched_out = SimpleNamespace(
        num_scheduled_tokens={"d0": 1, "p_short": 50, "p_long": 50, "p_mid": 50}
    )

    # Fast path: <= 1 prefill request
    mla_dispatch.order_prefills_by_kv_len(runner, 1, 2, sched_out)
    assert swaps == []

    # Spec decode tokens present -> skipped
    mla_dispatch.maybe_order_prefills_by_kv_len(
        runner, 1, 4, sched_out, spec_decode_tokens={"d0": [1]}
    )
    assert swaps == []

    # _masked_dense_groups_requests disabled -> skipped
    runner._masked_dense_groups_requests = False
    mla_dispatch.maybe_order_prefills_by_kv_len(runner, 1, 4, sched_out, None)
    assert swaps == []

    # Active sorting: [p_short(150), p_long(550), p_mid(350)] ->
    # [p_long, p_mid, p_short]
    runner._masked_dense_groups_requests = True
    mla_dispatch.maybe_order_prefills_by_kv_len(runner, 1, 4, sched_out, None)
    assert req_ids == ["d0", "p_long", "p_mid", "p_short"]
    assert len(swaps) == 2

    # Second call when already ordered -> `src == dst` branch exercised
    swaps.clear()
    mla_dispatch.order_prefills_by_kv_len(runner, 1, 4, sched_out)
    assert swaps == []


def test_dispatch_limits_rejections_and_admit_layer(monkeypatch):
    # 1. `_resolve_masked_dense_limit` and `masked_dense_limits`
    mla_dispatch.masked_dense_limits.cache_clear()
    monkeypatch.delenv("TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN", raising=False)
    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_MAX_KV_LEN", "4096")
    assert mla_dispatch.masked_dense_limits() == (
        mla_dispatch.MASKED_DENSE_ANALYTIC_MAX_KV_LEN,
        4096,
    )
    mla_dispatch.masked_dense_limits.cache_clear()
    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_MAX_KV_LEN", "500")
    with pytest.raises(ValueError, match="positive multiple"):
        mla_dispatch.masked_dense_limits()
    mla_dispatch.masked_dense_limits.cache_clear()

    # 2. `configuration_rejection`
    def _make_vllm_config(**overrides):
        hf = SimpleNamespace(
            model_type=mla_dispatch.GLM52_MODEL_TYPE,
            architectures=[mla_dispatch.GLM52_ARCHITECTURE],
            num_attention_heads=mla_dispatch.GLM52_NUM_HEADS,
            kv_lora_rank=mla_dispatch.GLM52_NOPE_DIM,
            qk_rope_head_dim=mla_dispatch.GLM52_ROPE_DIM,
            index_topk=mla_dispatch.GLM52_INDEX_TOPK,
            sliding_window=None,
        )
        for k, v in overrides.get("hf", {}).items():
            setattr(hf, k, v)
        pc = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        )
        for k, v in overrides.get("pc", {}).items():
            setattr(pc, k, v)
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf),
            parallel_config=pc,
            speculative_config=overrides.get("speculative_config"),
            kv_transfer_config=overrides.get("kv_transfer_config"),
            cache_config=SimpleNamespace(
                kv_offloading_size=overrides.get("kv_offloading_size", 0)
            ),
        )
        return cfg

    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_ENABLED", "0")
    assert (
        mla_dispatch.configuration_rejection(_make_vllm_config())
        == "TPU_MLA_MASKED_DENSE_ENABLED is off"
    )
    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_ENABLED", "1")
    assert (
        mla_dispatch.configuration_rejection(
            _make_vllm_config(hf={"model_type": "other"})
        )
        == "the model is not GLM-5.2"
    )
    assert "model geometry" in (
        mla_dispatch.configuration_rejection(
            _make_vllm_config(hf={"num_attention_heads": 32})
        )
        or ""
    )
    assert (
        mla_dispatch.configuration_rejection(
            _make_vllm_config(hf={"sliding_window": 1024})
        )
        == "sliding-window attention"
    )
    assert (
        mla_dispatch.configuration_rejection(
            _make_vllm_config(pc={"tensor_parallel_size": 2})
        )
        == "tensor_parallel_size=2"
    )
    assert (
        mla_dispatch.configuration_rejection(_make_vllm_config(speculative_config=True))
        == "speculative decoding"
    )
    assert (
        mla_dispatch.configuration_rejection(_make_vllm_config(kv_transfer_config=True))
        == "KV transfer"
    )
    assert (
        mla_dispatch.configuration_rejection(_make_vllm_config(kv_offloading_size=4))
        == "KV offloading"
    )
    assert mla_dispatch.configuration_rejection(_make_vllm_config()) is None

    # 3. `_layer_topk` and `layer_rejection`
    tc_spec = SimpleNamespace(
        layout=SimpleNamespace(value=mla_dispatch.sparse_kernel.TENSORCORE_CACHE_LAYOUT)
    )
    sc_spec = SimpleNamespace(layout=SimpleNamespace(value="sparsecore"))

    def _make_attn(**overrides):
        base = SimpleNamespace(
            non_causal_multi_token_decode=False,
            impl=SimpleNamespace(non_causal_multi_token_decode=False),
            num_heads=mla_dispatch.GLM52_NUM_HEADS,
            kv_lora_rank=mla_dispatch.GLM52_NOPE_DIM,
            qk_rope_head_dim=mla_dispatch.GLM52_ROPE_DIM,
            indexer=SimpleNamespace(topk_tokens=mla_dispatch.GLM52_INDEX_TOPK),
            topk_indices_buffer=None,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    assert (
        mla_dispatch.layer_rejection(
            _make_attn(non_causal_multi_token_decode=True),
            [tc_spec],
            mla_dispatch.GLM52_PAGE_SIZE,
        )
        == "non-causal attention"
    )
    assert (
        mla_dispatch.layer_rejection(
            _make_attn(), [sc_spec], mla_dispatch.GLM52_PAGE_SIZE
        )
        == "the KV caches are not in the TensorCore layout"
    )
    assert "layer geometry" in (
        mla_dispatch.layer_rejection(
            _make_attn(num_heads=16), [tc_spec], mla_dispatch.GLM52_PAGE_SIZE
        )
        or ""
    )
    assert mla_dispatch.layer_rejection(_make_attn(), [tc_spec], 64) == "page size 64"
    assert (
        mla_dispatch.layer_rejection(
            _make_attn(indexer=None, topk_indices_buffer=None),
            [tc_spec],
            mla_dispatch.GLM52_PAGE_SIZE,
        )
        == f"top-k None is not {mla_dispatch.GLM52_INDEX_TOPK}"
    )
    assert (
        mla_dispatch.layer_rejection(
            _make_attn(
                indexer=None,
                topk_indices_buffer=SimpleNamespace(
                    shape=(16, mla_dispatch.GLM52_INDEX_TOPK)
                ),
            ),
            [tc_spec],
            mla_dispatch.GLM52_PAGE_SIZE,
        )
        is None
    )

    # 4. `admit_layer`
    runner = SimpleNamespace(
        vllm_config=_make_vllm_config(),
        _masked_dense_groups_requests=None,
    )
    attn = _make_attn()
    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_ENABLED", "0")
    mla_dispatch.admit_layer(runner, attn, [tc_spec], mla_dispatch.GLM52_PAGE_SIZE)
    assert not hasattr(attn, "allow_masked_dense")

    monkeypatch.setenv("TPU_MLA_MASKED_DENSE_ENABLED", "1")
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (0, 0))
    mla_dispatch.admit_layer(runner, attn, [tc_spec], mla_dispatch.GLM52_PAGE_SIZE)
    assert attn.allow_masked_dense is False
    assert runner._masked_dense_groups_requests is False

    runner._masked_dense_groups_requests = None
    monkeypatch.setattr(mla_dispatch, "masked_dense_limits", lambda: (2048, 6144))
    mla_dispatch.admit_layer(runner, attn, [tc_spec], mla_dispatch.GLM52_PAGE_SIZE)
    assert attn.allow_masked_dense is True
    assert runner._masked_dense_groups_requests is True


def test_kv_cache_utils_specs_dtypes_and_scatter_wrappers(monkeypatch):
    # `get_kv_cache_shape` with `kv_packing=None` -> uses `get_dtype_packing`
    shape = kv_cache_utils.get_kv_cache_shape(
        total_num_pages=4,
        page_size=32,
        kv_dim=64,
        kv_dtype=jnp.float8_e4m3fn,
        kv_packing=None,
    )
    assert shape == (4, 8, 4, 128)

    tc_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.TENSORCORE,
        num_pages=4,
        page_size=32,
        head_dim=512,
    )
    sc_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages=4,
        page_size=32,
        head_dim=64,
    )
    assert tc_spec.torch_dtype == torch.uint8
    assert sc_spec.torch_dtype == torch.uint32

    # Unknown layout in `SparseMLAKVCacheSpec.create` (covers false branch of elif)
    custom_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        "custom_layout",  # type: ignore[arg-type]
        num_pages=4,
        page_size=32,
        head_dim=512,
    )
    assert custom_spec.head_dim == 512

    # Invalid cache_type in `.shape`
    bad_type_spec = kv_cache_utils.SparseMLAKVCacheSpec(
        cache_type="invalid",  # type: ignore[arg-type]
        layout=kv_cache_utils.KVCacheLayout.TENSORCORE,
        num_pages=4,
        page_size=32,
        head_dim=512,
        kv_packing=4,
    )
    with pytest.raises(ValueError, match="unsupported cache type"):
        _ = bad_type_spec.shape

    # Invalid layout in `_scatter_rows`
    with pytest.raises(ValueError, match="unsupported layout"):
        kv_cache_utils._scatter_rows(
            jnp.zeros((4, 32, 4, 128), dtype=jnp.uint8),
            custom_spec,
            jnp.zeros((2, 512), dtype=jnp.uint8),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
        )

    # `update_sparse_mla_kv_cache` and DCP variant via mocked scatter
    from vllm_torchtpu.kernels.mla import scatter

    scatter_calls = []

    def fake_scatter(cache, values, spec, dst_rows=None):
        scatter_calls.append((spec.cache_type, spec.layout))
        return cache

    monkeypatch.setattr(scatter, "scatter", fake_scatter)

    kv_nope = jnp.zeros(tc_spec.shape, dtype=tc_spec.jax_dtype)
    kv_rope_sc = jnp.zeros(sc_spec.shape, dtype=sc_spec.jax_dtype)
    rope_tc_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.TENSORCORE,
        num_pages=4,
        page_size=32,
        head_dim=64,
    )
    kv_rope_tc = jnp.zeros(rope_tc_spec.shape, dtype=rope_tc_spec.jax_dtype)
    kv_c = jnp.zeros((2, 512), dtype=jnp.float8_e4m3fn)
    k_pe = jnp.zeros((2, 64), dtype=jnp.float8_e4m3fn)
    seq_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([0, 1], dtype=jnp.int32)
    q_start = jnp.array([0, 2], dtype=jnp.int32)

    # Rope on SPARSECORE -> calls scatter.scatter for both nope and rope
    kv_cache_utils.update_sparse_mla_kv_cache(
        kv_nope,
        kv_rope_sc,
        kv_c,
        k_pe,
        seq_lens,
        block_tables,
        q_start,
        nope_spec=tc_spec,
        rope_spec=sc_spec,
    )
    assert scatter_calls == [
        (kv_cache_utils.KVCacheType.NOPE, kv_cache_utils.KVCacheLayout.TENSORCORE),
        (kv_cache_utils.KVCacheType.ROPE, kv_cache_utils.KVCacheLayout.SPARSECORE),
    ]

    # Rope on TENSORCORE via DCP -> scatter for nope, _scatter_rows for rope
    scatter_calls.clear()
    kv_cache_utils.update_sparse_mla_kv_cache_dcp(
        kv_nope,
        kv_rope_tc,
        kv_c,
        k_pe,
        seq_lens,
        block_tables,
        q_start,
        jnp.int32(0),
        nope_spec=tc_spec,
        rope_spec=rope_tc_spec,
        dcp_size=2,
        interleave_size=4,
    )
    assert scatter_calls == [
        (kv_cache_utils.KVCacheType.NOPE, kv_cache_utils.KVCacheLayout.TENSORCORE),
    ]


def test_scatter_utils_plan_and_kernel_cpu(monkeypatch):
    scatter_utils.split_plan.cache_clear()
    # Multi-core per block (spare > 0)
    plan_multi = scatter_utils.split_plan(num_tokens=4, num_lanes=8, num_cores=4)
    assert plan_multi.cores_per_block == 4
    assert plan_multi.lanes_per_core == 2
    assert plan_multi.blocks_per_round == 1
    assert plan_multi.total_tokens == 8

    # Single-core per block (spare == 0 when num_blocks > num_cores)
    plan_single = scatter_utils.split_plan(num_tokens=64, num_lanes=8, num_cores=2)
    assert plan_single.cores_per_block == 1
    assert plan_single.lanes_per_core == 8
    assert plan_single.blocks_per_round == 2
    assert plan_single.total_tokens == 64

    # Mock Pallas/TPU primitives to execute `scatter_rows` and `_scatter_kernel` on CPU
    fake_sc_info = SimpleNamespace(num_cores=2, num_subcores=2, num_lanes=4)
    monkeypatch.setattr(
        scatter_utils.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=fake_sc_info),
    )
    monkeypatch.setattr(
        scatter_utils.plsc,
        "VectorSubcoreMesh",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(scatter_utils.lax, "axis_index", lambda axes: jnp.int32(0))

    class _FakeCopy:
        def __init__(self, src_at, dst_at, sem_at):
            self.started = False
            self.waited = False

        def start(self):
            self.started = True

        def wait(self):
            self.waited = True

    monkeypatch.setattr(scatter_utils.pltpu, "make_async_copy", _FakeCopy)
    monkeypatch.setattr(
        scatter_utils.pl,
        "when",
        lambda cond: (lambda fn: fn() if bool(np.asarray(cond)) else None),
    )

    def fake_emit_pipeline(body, *, grid, tiling, in_specs):
        def _runner(dst_hbm_ref, src_words):
            for step in range(grid[0]):
                for spec in in_specs:
                    _ = spec.index_map(step)
                b_idx = in_specs[0].index_map(step)[0]
                L = in_specs[0].block_shape[0]
                dst_slice = _DummyAtRef(dst_hbm_ref.array[b_idx * L : (b_idx + 1) * L])
                src_slice = _DummyAtRef(src_words.array[b_idx * L : (b_idx + 1) * L])
                body(dst_slice, src_slice)

        return _runner

    monkeypatch.setattr(scatter_utils.pltpu, "emit_pipeline", fake_emit_pipeline)

    def fake_pl_kernel(kernel_fn, **kwargs):
        def _invoker(dst_rows, src_rows, cache_ref):
            sem = _DummyAtRef(np.zeros((8,), dtype=np.int32))
            kernel_fn(
                _DummyAtRef(np.asarray(dst_rows)),
                _DummyAtRef(np.asarray(src_rows)),
                _DummyAtRef(np.asarray(cache_ref[...])),
                sem,
            )

        return _invoker

    monkeypatch.setattr(scatter_utils.pl, "kernel", fake_pl_kernel)

    # 1. `tc_tiled=True` (uint8, 512B width) with padding (`num_tokens=2 < 4`)
    cache_u8 = jnp.zeros((2, 4, 4, 128), dtype=jnp.uint8)
    src_u8 = jnp.ones((2, 512), dtype=jnp.uint8)
    dst_rows_u8 = jnp.array([0, 1], dtype=jnp.int32)
    out_u8 = scatter_utils.scatter_rows(cache_u8, src_u8, dst_rows_u8)
    assert out_u8.shape == cache_u8.shape

    # 2. `tc_tiled=False` (uint32) with no padding (`num_tokens=16`, single core/block)
    cache_u32 = jnp.zeros((4, 8, 16), dtype=jnp.uint32)
    src_u32 = jnp.ones((16, 16), dtype=jnp.uint32)
    dst_rows_u32 = jnp.arange(16, dtype=jnp.int32)
    out_u32 = scatter_utils.scatter_rows(cache_u32, src_u32, dst_rows_u32)
    assert out_u32.shape == cache_u32.shape


def test_csa_mask_kernel_and_generate_mask_sc_cpu(monkeypatch):
    fake_sc_info = SimpleNamespace(num_cores=2, num_subcores=1)
    current_core = [0]
    monkeypatch.setattr(
        csa_mask.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=fake_sc_info),
    )
    monkeypatch.setattr(
        csa_mask.plsc,
        "VectorSubcoreMesh",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(csa_mask.jax.lax, "axis_index", lambda axes: current_core[0])
    monkeypatch.setattr(csa_mask.jax.lax, "psum", lambda val, axes: 2)

    def fake_fori_loop(lower, upper, body_fun, init_val, unroll=None):
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val

    monkeypatch.setattr(csa_mask.jax.lax, "fori_loop", fake_fori_loop)
    monkeypatch.setattr(
        csa_mask.jax.lax,
        "cond",
        lambda pred, true_fn, false_fn: (
            true_fn() if bool(np.asarray(pred)) else false_fn()
        ),
    )

    def fake_emit_pipeline(body, *, grid, in_specs, out_specs):
        def _runner(topk_ref, mask_out_ref):
            for r in range(grid[0]):
                in_idx = in_specs.index_map(r)
                out_idx = out_specs.index_map(r)
                row = in_idx[0]
                assert row == out_idx[0]
                topk_vmem = _DummyRef(topk_ref.array[row : row + 1])
                mask_vmem = _DummyRef(mask_out_ref.array[row : row + 1])
                body(topk_vmem, mask_vmem)

        return _runner

    monkeypatch.setattr(csa_mask.pltpu, "emit_pipeline", fake_emit_pipeline)

    def fake_pl_kernel(kernel_fn, *, out_type, **kwargs):
        def _invoker(padded_topk):
            topk_ref = _DummyRef(np.asarray(padded_topk).copy())
            out_ref = _DummyRef(np.zeros(out_type.shape, dtype=np.int32))
            for cid in range(2):
                current_core[0] = cid
                kernel_fn(topk_ref, out_ref)
            return jnp.asarray(out_ref.array)

        return _invoker

    monkeypatch.setattr(csa_mask.pl, "kernel", fake_pl_kernel)

    # 1. `topk % 8 == 0` and `pad_size > 0` (`chunk_size=1`, `num_cores=2`)
    topk_8 = jnp.array([[0, 2, -1, 4, 15, 7, -1, 1]], dtype=jnp.int32)
    mask_8 = csa_mask.generate_mask_sc.__wrapped__(topk_8, max_kv_len=8)
    assert mask_8.shape == (1, 8)
    np.testing.assert_array_equal(
        np.asarray(mask_8[0]),
        np.array([1, 1, 1, 0, 1, 0, 0, 1], dtype=np.int32),
    )

    # 2. `topk % 8 != 0` and `pad_size == 0` (`chunk_size=2`, `num_cores=2`)
    topk_3 = jnp.array([[1, -1, 3], [0, 99, 2]], dtype=jnp.int32)
    mask_3 = csa_mask.generate_mask_sc.__wrapped__(topk_3, max_kv_len=4)
    assert mask_3.shape == (2, 4)
    np.testing.assert_array_equal(
        np.asarray(mask_3[0]),
        np.array([0, 1, 0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(mask_3[1]),
        np.array([1, 0, 1, 0], dtype=np.int32),
    )
