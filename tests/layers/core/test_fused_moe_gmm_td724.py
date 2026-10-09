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

"""CPU unit tests for TD7.24 optimizations and 100% coverage of fused_moe."""

import enum
import types

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_moe as fm_adapter
from vllm_torchtpu.layers.core import fused_moe_gmm as fm_core


class _DummyEnum(enum.Enum):
    SILU = "silu"
    SITU = "situ"


class _DummyLayer(fm_adapter.TpuMoEActivationMixin):
    def __init__(self, activation, beta=1.5, linear_beta=None):
        self.activation = activation
        self.moe_config = types.SimpleNamespace(
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        )


def test_fused_moe_adapter_activations_and_mixin():
    assert fm_adapter._resolve_activation_name(_DummyEnum.SILU) == "silu"
    assert (
        fm_adapter._resolve_activation_name(types.SimpleNamespace(value="gelu"))
        == "gelu"
    )
    assert fm_adapter._resolve_activation_name("relu") == "relu"

    layer_silu = _DummyLayer(_DummyEnum.SILU)
    assert layer_silu._set_tpu_activation(layer_silu) == "silu"
    assert layer_silu._tpu_activation_str == "silu"

    layer_situ_none = _DummyLayer(_DummyEnum.SITU, beta=2.0, linear_beta=None)
    assert layer_situ_none._set_tpu_activation(layer_situ_none) == "situ:2.0:none"

    layer_situ_lin = _DummyLayer("situ", beta=1.25, linear_beta=0.5)
    assert layer_situ_lin._set_tpu_activation(layer_situ_lin) == "situ:1.25:0.5"


def test_fused_moe_adapter_fp4_ops_and_custom_op_caching(monkeypatch):
    monkeypatch.setattr(fm_adapter, "_load_kmajor_fp4_op", None)
    monkeypatch.setattr(fm_adapter, "_requant_kmajor_fp4_ops", {})
    monkeypatch.setattr(fm_adapter, "_quantize_native_fp4_kmajor_ops", {})
    monkeypatch.setattr(fm_adapter, "_fused_moe_kernel_cache", {})
    monkeypatch.setattr(fm_adapter, "_zero_experts_start_cache", {})

    registered_fakes = []

    def _fake_jax_op(name, fn):
        class _Op:
            def __init__(self):
                self.name = name
                self.fn = fn
                self.fake_fn = None
                self.calls = []

            def register_fake(self, fake_fn):
                self.fake_fn = fake_fn
                registered_fakes.append(fake_fn)

            def __call__(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return args[0]

        return _Op()

    monkeypatch.setattr(fm_adapter.pallas, "jax_op", _fake_jax_op)

    t = torch.zeros((2, 4), dtype=torch.uint8)
    s = torch.ones((2, 1), dtype=torch.float32)
    assert fm_adapter.load_kmajor_fp4(t) is t
    assert fm_adapter.load_kmajor_fp4(t) is t

    assert fm_adapter.requant_load_kmajor_fp4(t, s, block=16) is t
    assert fm_adapter.requant_load_kmajor_fp4(t, s, block=16) is t

    assert fm_adapter.quantize_native_fp4_kmajor(s, block=16, pack=True) is s
    assert (
        fm_adapter.quantize_native_fp4_fp4_cached_check(s)
        if hasattr(fm_adapter, "quantize_native_fp4_fp4_cached_check")
        else fm_adapter.quantize_native_fp4_kmajor(s, block=16, pack=True) is s
    )
    assert fm_adapter.quantize_native_fp4_kmajor(s, block=16, pack=False) is s

    fm_adapter.prebuild_fused_moe_kernel(
        topk=2,
        activation="silu",
        use_ep=False,
        rhs_quant_dtype=None,
        skip_padded_tokens=None,
    )
    assert len(registered_fakes) == 1
    hidden = torch.ones((3, 8), dtype=torch.bfloat16)
    fake_out = registered_fakes[0](hidden)
    assert fake_out.shape == hidden.shape

    w1 = torch.ones((2, 8, 8), dtype=torch.bfloat16)
    w2 = torch.ones((2, 4, 8), dtype=torch.bfloat16)
    topk_w = torch.ones((3, 2), dtype=torch.bfloat16)
    topk_i = torch.zeros((3, 2), dtype=torch.int32)

    # First non-EP call allocates and caches 0-d experts_start; second reuses it.
    out1 = fm_adapter.fused_moe_gmm(
        hidden, w1, w2, None, None, None, None, topk_w, topk_i, None, 2, "silu"
    )
    out2 = fm_adapter.fused_moe_gmm(
        hidden,
        w1,
        w2,
        None,
        None,
        None,
        None,
        topk_w,
        topk_i,
        None,
        2,
        "silu",
        skip_padded_tokens=False,
    )
    assert out1 is hidden and out2 is hidden
    op_non_ep = fm_adapter._get_fused_moe_custom_op(
        topk=2, activation="silu", use_ep=False, skip_padded_tokens=False
    )
    assert op_non_ep.calls[0][0][-1] is op_non_ep.calls[1][0][-1]

    # EP call with explicit experts_start tensor.
    exp_start = torch.tensor(4, dtype=torch.int32)
    out_ep = fm_adapter.fused_moe_gmm(
        hidden, w1, w2, None, None, None, None, topk_w, topk_i, exp_start, 2, "silu"
    )
    assert out_ep is hidden


def test_fused_moe_gmm_fp4_helpers_and_gmm_wrapper(monkeypatch):
    w_packed = jnp.arange(16, dtype=jnp.uint8).reshape(2, 4, 2)
    unpacked = fm_core.unpack_fp4_to_e2m1(w_packed)
    assert unpacked.shape == (2, 4, 4)
    assert unpacked.dtype == jnp.float4_e2m1fn

    w_req_packed = jnp.arange(64, dtype=jnp.uint8).reshape(2, 4, 8)
    scale_f = jnp.ones((2, 4, 1), dtype=jnp.float32)
    req_w, req_s = fm_core.requant_unpack_kmajor(w_req_packed, scale_f, block=16)
    assert req_w.shape == (2, 16, 4)
    assert req_w.dtype == jnp.float4_e2m1fn
    assert req_s.shape == (2, 1, 1, 4)

    w_float = jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).reshape(2, 16, 2)
    q_unpacked, s_unpacked = fm_core.quantize_to_native_fp4_kmajor(
        w_float, block=8, pack=False
    )
    assert q_unpacked.shape == (2, 16, 2)
    assert q_unpacked.dtype == jnp.float4_e2m1fn
    assert s_unpacked.shape == (2, 2, 1, 2)

    q_packed, s_packed = fm_core.quantize_to_native_fp4_kmajor(
        w_float, block=8, pack=True
    )
    assert q_packed.shape == (2, 8, 2)
    assert q_packed.dtype == jnp.uint8
    assert s_packed.shape == (2, 2, 1, 2)

    captured = []

    def _fake_gmm_v2(**kwargs):
        captured.append(kwargs)
        return kwargs["lhs"]

    monkeypatch.setattr(fm_core, "gmm_v2", _fake_gmm_v2)
    lhs = jnp.ones((4, 8), dtype=jnp.bfloat16)
    rhs_bf16 = jnp.ones((2, 8, 8), dtype=jnp.bfloat16)
    rhs_fp4 = jnp.ones((2, 8, 8), dtype=jnp.float4_e2m1fn)
    fm_core.gmm_wrapper(lhs, rhs_bf16, None, None, jnp.array([2, 2]), jnp.array([0]))
    fm_core.gmm_wrapper(lhs, rhs_fp4, None, None, jnp.array([2, 2]), jnp.array([0]))
    assert captured[0]["maybe_quantize_lhs"] is True
    assert captured[1]["maybe_quantize_lhs"] is False


def test_counting_sort_and_prepare_routed_branches(monkeypatch):
    # num_buckets > block returns (None, None).
    pos_none, counts_none = fm_core._counting_sort_positions(
        jnp.array([0, 1, 2, 3], dtype=jnp.int32), num_buckets=8
    )
    assert pos_none is None and counts_none is None

    # m = 520 > _COUNTING_SORT_BLOCK (512) exercises pad > 0 branch.
    keys_padded = jnp.arange(520, dtype=jnp.int32) % 4
    pos_pad, counts_pad = fm_core._counting_sort_positions(keys_padded, num_buckets=4)
    assert pos_pad.shape == (520,)
    assert counts_pad.shape == (4,)

    # _prepare_routed_gmm_inputs with ragged_permute where _counting_sort_positions
    # returns (None, None) (num_buckets > block) AND non-packable fallback (line 431).
    hidden = jnp.ones((2, 4), dtype=jnp.bfloat16)
    topk_idx = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    topk_w = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.bfloat16)
    monkeypatch.setattr(
        fm_core,
        "ragged_gather_v2",
        lambda x, idx, starts, ends: x[idx],
    )
    out_Fallback = fm_core._prepare_routed_gmm_inputs(
        hidden,
        topk_idx,
        topk_w,
        local_num_experts=16,
        topk=2,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    assert out_Fallback[0].shape == (4, 4)

    # Non-packable branch (line 431): local_num_experts large enough that
    # (local_num_experts + 1).bit_length() + token_bits >= 31.
    orig_one_hot = jax.nn.one_hot
    monkeypatch.setattr(
        jax.nn,
        "one_hot",
        lambda x, num_classes, **kw: orig_one_hot(x, 4, **kw),
    )
    out_unpackable = fm_core._prepare_routed_gmm_inputs(
        hidden,
        topk_idx,
        topk_w,
        local_num_experts=1 << 30,
        topk=2,
        use_ep=False,
        use_sparse_core=False,
    )
    assert out_unpackable[0].shape == (4, 4)


def test_moe_gmm_and_fused_moe_func_all_branches(monkeypatch):
    monkeypatch.setattr(fm_core, "get_packing_factor", lambda *a, **kw: 1)

    # 1. moe_gmm with gmm1_res.shape[1] != w2_in_dim (slice branch) AND
    #    topk_weights_sorted provided (line 622) AND not zero_init_gmm2 (lines 652-655).
    rows = 2306  # > 2304 so zero_init_gmm2 is False when threshold >= 2306
    num_tokens = 1153
    topk = 2
    hidden = 4
    gmm1_wide = jnp.ones((rows, hidden + 2), dtype=jnp.bfloat16)
    gmm2_out = jnp.ones((rows, hidden), dtype=jnp.bfloat16)
    outputs = iter([gmm1_wide, gmm2_out])
    monkeypatch.setattr(fm_core, "gmm_wrapper", lambda *a, **kw: next(outputs))

    res = fm_core.moe_gmm(
        x=jnp.ones((rows, hidden), dtype=jnp.bfloat16),
        w1=jnp.ones((2, hidden, hidden * 2), dtype=jnp.bfloat16),
        w1_scale=None,
        w1_bias=None,
        w2=jnp.ones((2, hidden, hidden), dtype=jnp.bfloat16),
        w2_scale=None,
        w2_bias=None,
        group_sizes=jnp.array([rows - 2, 0], dtype=jnp.int32),
        argsort_revert_indices=jnp.arange(rows, dtype=jnp.int32),
        topk_weights_flat=jnp.ones((rows,), dtype=jnp.bfloat16),
        valid_mask_flat=jnp.ones((rows,), dtype=jnp.bool_),
        token_indices_sorted=jnp.arange(rows, dtype=jnp.int32) // topk,
        activation="silu",
        num_tokens=num_tokens,
        topk=topk,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=rows,
        topk_weights_sorted=jnp.ones((rows,), dtype=jnp.bfloat16),
    )
    assert res.shape == (num_tokens, hidden)

    # 2. fused_moe_func.__wrapped__:
    #    - 3D w1_scale and w2_scale expansion (lines 730-733)
    #    - use_ep=True and experts_start is not None (lines 752-756)
    #    - dense_expert_moe path (lines 760-783)
    #    - padded_hidden_size == hidden_size vs padded_hidden_size != hidden_size
    #    - x.shape == (num_tokens, hidden_size) vs x.shape != (num_tokens, hidden_size)
    fn = fm_core.fused_moe_func.__wrapped__
    h = jnp.ones((2, 4), dtype=jnp.bfloat16)
    w1_same = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
    w2_same = jnp.ones((2, 4, 4), dtype=jnp.bfloat16)
    w1_scale_3d = jnp.ones((2, 1, 8), dtype=jnp.float32)
    w2_scale_3d = jnp.ones((2, 1, 4), dtype=jnp.float32)
    tw = jnp.ones((2, 2), dtype=jnp.float32)
    ti = jnp.array([[0, 3], [1, -1]], dtype=jnp.int32)

    # Dense-expert path when _DENSE_EXPERT_THRESHOLD >= num_tokens.
    monkeypatch.setattr(fm_core, "_DENSE_EXPERT_THRESHOLD", 16)
    monkeypatch.setattr(fm_core, "can_use_dense_expert", lambda *a, **kw: True)
    monkeypatch.setattr(fm_core, "dense_expert_moe", lambda h_in, *a, **kw: h_in)
    out_dense = fn(
        h,
        w1_same,
        w2_same,
        w1_scale_3d,
        w2_scale_3d,
        None,
        None,
        tw,
        ti,
        experts_start=jnp.array(0, dtype=jnp.int32),
        topk=2,
        activation="silu",
        use_ep=True,
        use_sparse_core=False,
    )
    assert out_dense.shape == (2, 4)

    # Routed GMM path with padded_hidden_size == hidden_size and exact output shape.
    monkeypatch.setattr(fm_core, "_DENSE_EXPERT_THRESHOLD", 0)
    monkeypatch.setattr(
        fm_core,
        "moe_gmm",
        lambda x_in, *a, **kw: jnp.ones((2, 4), dtype=jnp.bfloat16),
    )
    out_exact = fn(
        h,
        w1_same,
        w2_same,
        None,
        None,
        None,
        None,
        tw,
        ti,
        experts_start=None,
        topk=2,
        activation="silu",
        use_ep=False,
        use_sparse_core=False,
    )
    assert out_exact.shape == (2, 4)

    # Routed GMM path with padded_hidden_size > hidden_size and padded output shape.
    w1_pad = jnp.ones((2, 6, 8), dtype=jnp.bfloat16)
    w2_pad = jnp.ones((2, 4, 6), dtype=jnp.bfloat16)
    w1_scale_4d = jnp.ones((2, 1, 1, 8), dtype=jnp.float32)
    w2_scale_4d = jnp.ones((2, 1, 1, 6), dtype=jnp.float32)
    monkeypatch.setattr(
        fm_core,
        "moe_gmm",
        lambda x_in, *a, **kw: jnp.ones((4, 6), dtype=jnp.bfloat16),
    )
    out_sliced = fn(
        h,
        w1_pad,
        w2_pad,
        w1_scale_4d,
        w2_scale_4d,
        None,
        None,
        tw,
        ti,
        experts_start=None,
        topk=2,
        activation="silu",
        use_ep=False,
        use_sparse_core=False,
    )
    assert out_sliced.shape == (2, 4)


def test_remaining_fused_moe_gmm_branches(monkeypatch):
    # 1. resolve_onehot_permute_threshold branches (lines 85, 91-97).
    # Drop any value another test pinned on the lazy `envs` module, so the
    # environment variable below is what `envs.__getattr__` reads.
    monkeypatch.delitem(vars(envs), "ONEHOT_MOE_PERMUTE_THRESHOLD", raising=False)
    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "64")
    assert fm_core.resolve_onehot_permute_threshold() == 64

    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "-1")
    monkeypatch.setattr(
        fm_core.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(sparse_core=None),
    )
    assert fm_core.resolve_onehot_permute_threshold() == 0

    monkeypatch.setattr(
        fm_core.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(
            sparse_core=types.SimpleNamespace(num_lanes=8, num_cores=4, num_subcores=8)
        ),
    )
    assert fm_core.resolve_onehot_permute_threshold() == 511

    monkeypatch.setattr(
        fm_core.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(
            sparse_core=types.SimpleNamespace(num_lanes=16, num_cores=4, num_subcores=8)
        ),
    )
    assert fm_core.resolve_onehot_permute_threshold() == 4095

    # 2. _moe_gmm_tiling branches (lines 204-213).
    monkeypatch.setattr(
        fm_core,
        "calculate_tiling",
        lambda *a, **kw: fm_core.TileSizes(
            tile_m=64, tile_k=128, tile_n=128, bucket_base=64
        ),
    )
    tile_32 = fm_core._moe_gmm_tiling(
        types.SimpleNamespace(size_group=4, size_m=64, size_lhs_sublane=8),
        None,
        None,
        1024,
    )
    assert tile_32.tile_m == 32 and tile_32.bucket_base == 32
    tile_64 = fm_core._moe_gmm_tiling(
        types.SimpleNamespace(size_group=1, size_m=64, size_lhs_sublane=8),
        None,
        None,
        1024,
    )
    assert tile_64.tile_m == 64

    # 3. _counting_sort_positions (pad == 0) and prepare_routed_gmm_inputs.
    monkeypatch.setattr(
        fm_core,
        "ragged_gather_v2",
        lambda x, idx, starts, ends: x[idx],
    )
    routed_ragged = fm_core.prepare_routed_gmm_inputs(
        jnp.ones((4, 4), dtype=jnp.bfloat16),
        jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=jnp.int32),
        jnp.array([[0.5, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=jnp.bfloat16),
        local_num_experts=2,
        topk=2,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
        skip_padded_tokens=True,
    )
    assert len(routed_ragged) == 6

    # 4. _pairwise_sort_positions and onehot permute branch.
    routed_pairwise = fm_core._prepare_routed_gmm_inputs(
        jnp.ones((2, 4), dtype=jnp.bfloat16),
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        jnp.ones((2, 2), dtype=jnp.bfloat16),
        local_num_experts=2,
        topk=2,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=64,
        skip_padded_tokens=False,
    )
    assert routed_pairwise[0].shape == (4, 4)

    # 5. moe_gmm remaining branches.
    rows = 4
    hidden = 4
    monkeypatch.setattr(fm_core, "get_packing_factor", lambda *a, **kw: 1)
    monkeypatch.setattr(
        fm_core,
        "gmm_wrapper",
        lambda *a, **kw: jnp.ones((rows, hidden), dtype=jnp.bfloat16),
    )
    common_args = dict(
        x=jnp.ones((rows, hidden), dtype=jnp.bfloat16),
        w1=jnp.ones((2, hidden, hidden), dtype=jnp.bfloat16),
        w1_scale=None,
        w1_bias=None,
        w2=jnp.ones((2, hidden, hidden), dtype=jnp.bfloat16),
        w2_scale=None,
        w2_bias=None,
        group_sizes=jnp.array([2, 2], dtype=jnp.int32),
        argsort_revert_indices=jnp.arange(rows, dtype=jnp.int32),
        topk_weights_flat=jnp.ones((rows,), dtype=jnp.bfloat16),
        valid_mask_flat=jnp.ones((rows,), dtype=jnp.bool_),
        token_indices_sorted=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        activation="silu",
        num_tokens=2,
        topk=2,
    )

    # 5a. use_local_ragged_combine with get_tpu_info raising ValueError + non-EP.
    def _raise_val_err():
        raise ValueError("no tpu on cpu")

    monkeypatch.setattr(fm_core.pltpu, "get_tpu_info", _raise_val_err)
    out_non_ep = fm_core.moe_gmm(
        **common_args,
        use_ep=False,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    assert out_non_ep.shape == (2, hidden)

    # 5b. use_local_ragged_combine with valid sc_info.
    monkeypatch.setattr(
        fm_core.pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(
            sparse_core=types.SimpleNamespace(num_lanes=16),
            num_lanes=4,
            vmem_capacity_bytes=16,
        ),
    )
    monkeypatch.setattr(fm_core, "token_block_alignment", lambda *a, **kw: 2)
    monkeypatch.setattr(
        fm_core,
        "ragged_gather_reduce_v3",
        lambda *a, **kw: jnp.ones((2, hidden), dtype=jnp.bfloat16),
    )
    out_v3 = fm_core.moe_gmm(
        **common_args,
        use_ep=False,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    assert out_v3.shape == (2, hidden)

    monkeypatch.setattr(fm_core, "token_block_alignment", lambda *a, **kw: 8)
    out_v3_skip = fm_core.moe_gmm(
        **common_args,
        use_ep=False,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    assert out_v3_skip.shape == (2, hidden)

    # 5c. EP + SparseCore onehot combine with topk_weights_sorted=None.
    monkeypatch.setenv("TPU_MOE_OWNER_OUTPUT_MODE", "off")
    out_ep_onehot = fm_core.moe_gmm(
        **common_args,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=64,
        topk_weights_sorted=None,
    )
    assert out_ep_onehot.shape == (2, hidden)

    # 5d. EP + SparseCore blockwise_onehot_unpermute path (634-641).
    monkeypatch.setenv("TPU_MOE_OWNER_OUTPUT_MODE", "on")
    monkeypatch.setattr(
        fm_core, "can_use_blockwise_onehot_unpermute", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        fm_core,
        "blockwise_onehot_unpermute",
        lambda *a, **kw: jnp.ones((2, hidden), dtype=jnp.bfloat16),
    )
    out_ep_blockwise = fm_core.moe_gmm(
        **common_args,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=64,
        topk_weights_sorted=None,
    )
    assert out_ep_blockwise.shape == (2, hidden)

    # 5e. EP + SparseCore ragged_gather_reduce path (661).
    monkeypatch.setattr(
        fm_core,
        "ragged_gather_reduce",
        lambda *a, **kw: jnp.ones((2, hidden), dtype=jnp.bfloat16),
    )
    out_ep_ragged = fm_core.moe_gmm(
        **common_args,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
    )
    assert out_ep_ragged.shape == (2, hidden)
