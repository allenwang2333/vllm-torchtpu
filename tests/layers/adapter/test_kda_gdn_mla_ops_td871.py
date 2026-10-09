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
"""TD8.71: CPU coverage for the Kimi KDA and Qwen GDN custom-op bridges."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op as gdn
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kda
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

H, D, K = 2, 4, 4  # heads, head dim, conv kernel size
T, N = 6, 2  # tokens, sequences


def _operands(dtype=jnp.bfloat16):
    return dict(
        mixed_qkv=jnp.ones((T, 3 * H * D), dtype),
        raw_gate=jnp.ones((T, H * D), dtype),
        beta=jnp.ones((T, H), dtype),
        output_gate=jnp.ones((T, H * D), dtype),
        recurrent_state=jnp.zeros((N + 1, H, D, D), jnp.float32),
        a_log=jnp.zeros((H,), jnp.float32),
        dt_bias=jnp.zeros((H * D,), jnp.float32),
        norm_weight=jnp.ones((D,), dtype),
        query_start_loc=jnp.array([0, 3, 6], jnp.int32),
        state_indices=jnp.array([1, 2], jnp.int32),
        seq_lens=jnp.array([3, 3], jnp.int32),
    )


_ABI_ORDER = (
    "mixed_qkv",
    "raw_gate",
    "beta",
    "output_gate",
    "recurrent_state",
    "a_log",
    "dt_bias",
    "norm_weight",
    "query_start_loc",
    "state_indices",
    "seq_lens",
)


@pytest.mark.cpu_test
def test_check_kda_abi_accepts_and_rejects_each_operand():
    ops = _operands()
    assert kda._check_kda_abi(*(ops[k] for k in _ABI_ORDER)) == (T, N, H, D)
    bad = [
        ("mixed_qkv", jnp.ones((T, 5), jnp.bfloat16), ValueError, "activation"),
        ("raw_gate", jnp.ones((T, 3), jnp.bfloat16), ValueError, "gate shape"),
        ("beta", jnp.ones((T, 3), jnp.bfloat16), ValueError, "beta"),
        ("output_gate", jnp.ones((T, 3), jnp.bfloat16), ValueError, "output-gate"),
        ("a_log", jnp.zeros((3,), jnp.float32), ValueError, "A_log shape"),
        ("dt_bias", jnp.zeros((3,), jnp.float32), ValueError, "dt_bias shape"),
        ("norm_weight", jnp.ones((3,), jnp.bfloat16), ValueError, "norm weight"),
        ("query_start_loc", jnp.zeros((2,), jnp.int32), ValueError, "query starts"),
        ("seq_lens", jnp.zeros((3,), jnp.int32), ValueError, "sequence lengths"),
        ("beta", jnp.ones((T, H), jnp.float32), TypeError, "one common dtype"),
        (
            "recurrent_state",
            jnp.zeros((N + 1, H, D, D), jnp.bfloat16),
            TypeError,
            "recurrent state",
        ),
        ("a_log", jnp.zeros((H,), jnp.bfloat16), TypeError, "A_log and dt_bias"),
        ("seq_lens", jnp.zeros((N,), jnp.int16), TypeError, "metadata"),
    ]
    for name, value, exc, match in bad:
        args = dict(ops, **{name: value})
        with pytest.raises(exc, match=match):
            kda._check_kda_abi(*(args[k] for k in _ABI_ORDER))


@pytest.mark.cpu_test
def test_short_conv_scan_start_seq_and_4d_state():
    x = jnp.arange(T * 3, dtype=jnp.float32).reshape(T, 3)
    conv_state = jnp.ones((N + 1, K - 1, 1, 3), jnp.float32)
    weight = jnp.ones((K, 3), jnp.float32)
    qsl = jnp.array([0, 3, 6], jnp.int32)
    idx = jnp.array([1, 2], jnp.int32)
    seq_lens = jnp.array([5, 3], jnp.int32)
    out_all, state_all = kda.kimi_short_conv_scan(
        x, conv_state, weight, qsl, idx, seq_lens
    )
    out_tail, state_tail = kda.kimi_short_conv_scan(
        x, conv_state, weight, qsl, idx, seq_lens, start_seq=1
    )
    assert state_all.shape == conv_state.shape
    # Restricting to sequence 1 zeroes sequence 0's rows and leaves its slot.
    np.testing.assert_array_equal(np.asarray(out_tail[:3]), 0)
    np.testing.assert_array_equal(np.asarray(out_tail[3:]), np.asarray(out_all[3:]))
    np.testing.assert_array_equal(np.asarray(state_tail[1]), np.asarray(conv_state[1]))
    np.testing.assert_array_equal(np.asarray(state_tail[2]), np.asarray(state_all[2]))


@pytest.mark.cpu_test
def test_gated_output_norm_matches_numpy():
    out = jnp.arange(2 * H * D, dtype=jnp.float32).reshape(2, H, D) + 1
    gate = jnp.zeros((2, H * D), jnp.float32)
    got = kda._gated_output_norm(out, gate, jnp.ones((D,)), 1e-6, jnp.float32)
    ref = np.asarray(out) / np.sqrt(np.mean(np.asarray(out) ** 2, -1, keepdims=True))
    np.testing.assert_allclose(np.asarray(got), ref * 0.5, rtol=1e-5)


def _fake_kernel(calls):
    def fake(**kw):
        calls.append(kw)
        out = jnp.zeros((kw["qkv"].shape[0], kw["n_v"] * kw["d_v"]), kw["qkv"].dtype)
        return (kw["conv_state"], kw["recurrent_state"]), out

    return fake


def _core_args(ops, conv_state):
    return (
        ops["mixed_qkv"],
        ops["raw_gate"],
        ops["beta"],
        ops["output_gate"],
        conv_state,
        ops["recurrent_state"],
        jnp.ones((K, 3, H, D), jnp.bfloat16),
        ops["a_log"],
        ops["dt_bias"],
        ops["norm_weight"],
        ops["query_start_loc"],
        ops["state_indices"],
        ops["seq_lens"],
        jnp.array([0, 0, N], jnp.int32),
    )


@pytest.mark.cpu_test
def test_fused_core_verify_requires_read_offsets(monkeypatch):
    calls = []
    monkeypatch.setattr(kda.gdn_wrapper, "fused_conv1d_gdn", _fake_kernel(calls))
    ops = _operands()
    conv_state = jnp.zeros((N + 1, K - 1, 1, 3 * H * D), jnp.bfloat16)
    core = kda._build_fused_core(None, 1e-6, num_spec_tokens=2)
    with pytest.raises(ValueError, match="slot_read_offsets"):
        core(*_core_args(ops, conv_state))
    out, new_conv, new_rec = kda._build_fused_core(None, 1e-6)(
        *_core_args(ops, conv_state)
    )
    assert out.shape == (T, H, D) and new_conv is conv_state
    assert calls[-1]["read_offsets"] is None


class _FakeJaxOp:
    def __init__(self, name, fn, donate_argnums, returns):
        self.name, self.fn, self.donate = name, fn, donate_argnums
        self.returns = returns
        self.fake = None
        self.calls = []

    def register_fake(self, fn):
        self.fake = fn

    def __call__(self, *args):
        self.calls.append(args)
        return self.returns(*args)


@pytest.mark.cpu_test
@pytest.mark.parametrize("spec_tokens", [0, 3])
def test_dispatched_op_builder_wires_fake_and_impl(monkeypatch, spec_tokens):
    built = []

    def jax_op(name, fn, donate_argnums):
        op = _FakeJaxOp(
            name,
            fn,
            donate_argnums,
            lambda *a: (torch.ones(a[0].shape[0], H, D), a[4] + 1, a[5] + 2),
        )
        built.append(op)
        return op

    monkeypatch.setattr(kda.pallas, "jax_op", jax_op)
    impl = kda.build_kimi_dispatched_kda_op(
        "model.layers.0.attn",
        lower_bound=-5.0,
        eps=1e-6,
        state_dim_first=True,
        num_spec_tokens=spec_tokens,
    )
    (op,) = built
    variant = "fused_window" if spec_tokens else "fused"
    assert op.name == f"pallas::kimi_dispatched_kda_{variant}_model_layers_0_attn"
    assert op.donate == (4, 5)
    mixed = torch.zeros(T, 3 * H * D)
    conv = torch.zeros(N + 1, K - 1, 3 * H * D)
    rec = torch.zeros(N + 1, H, D, D)
    out, fconv, frec = op.fake(mixed, None, None, None, conv, rec, 1, 2, x=3)
    assert out.shape == (T, H, D) and fconv.shape == conv.shape
    assert frec.shape == rec.shape
    rest = [torch.zeros(1)] * 10
    result = impl(mixed, None, None, None, conv, rec, *rest)
    assert result.shape == (T, H, D)
    assert torch.all(conv == 1) and torch.all(rec == 2)


def _fake_layout(**_):
    return SimpleNamespace(ssm_tokens=1, conv_tokens=1, required_tokens=2)


def _pool_fakes(monkeypatch, calls):
    def gather_region(pool, idx, *, tok0, ntok, out_dtype, out_lanes=None, split=1):
        calls.append(("gather", tok0, ntok, split))
        tok_bytes = int(np.prod(pool.shape[2:])) * jnp.dtype(pool.dtype).itemsize
        if out_lanes is not None:
            rows = ntok * tok_bytes // (4 * out_lanes)
            return jnp.zeros((idx.shape[0], rows, out_lanes), out_dtype)
        elems = ntok * tok_bytes // jnp.dtype(out_dtype).itemsize
        return jnp.zeros((idx.shape[0], elems), out_dtype)

    def scatter_region(pool, vals, idx, *, tok0, ntok, split=1):
        calls.append(("scatter", tok0, ntok, split, vals.shape))
        return pool + 1

    monkeypatch.setattr(kda, "derive_pooled_gdn_state_layout", _fake_layout)
    monkeypatch.setattr(kda.pool_adapters, "gather_region", gather_region)
    monkeypatch.setattr(kda.pool_adapters, "scatter_region", scatter_region)
    monkeypatch.setattr(kda.gdn_wrapper, "fused_conv1d_gdn", _fake_kernel([]))


def _pooled_args(pool):
    ops = _operands()
    return (
        ops["mixed_qkv"],
        ops["raw_gate"],
        ops["beta"],
        ops["output_gate"],
        pool,
        jnp.ones((K, 3, H, D), jnp.bfloat16),
        ops["a_log"],
        ops["dt_bias"],
        ops["norm_weight"],
        ops["query_start_loc"],
        ops["state_indices"],
        ops["seq_lens"],
        jnp.array([0, 0, N], jnp.int32),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("pool_block_tokens,split", [(16, 1), (32, 2)])
def test_pooled_core_gathers_runs_and_scatters(monkeypatch, pool_block_tokens, split):
    calls = []
    _pool_fakes(monkeypatch, calls)
    pool = jnp.zeros((4, 16, 128), jnp.bfloat16)
    out, new_pool = kda._pooled_kda_core(
        *_pooled_args(pool),
        lower_bound=None,
        eps=1e-6,
        pool_block_tokens=pool_block_tokens,
    )
    assert out.shape == (T, H, D)
    np.testing.assert_array_equal(np.asarray(new_pool), 2)  # two scatters
    assert [c[0] for c in calls] == ["gather", "gather", "scatter", "scatter"]
    assert all(c[3] == split for c in calls)
    assert calls[2][4] == (N, 16, D)  # padded SSM region rows
    assert calls[3][4] == (N, 1, 128)  # one packed conv row


@pytest.mark.cpu_test
def test_pooled_core_sharded_conv_layout(monkeypatch):
    calls = []
    _pool_fakes(monkeypatch, calls)
    monkeypatch.setattr(
        kda,
        "derive_pooled_gdn_state_layout",
        lambda **_: SimpleNamespace(ssm_tokens=1, conv_tokens=2, required_tokens=3),
    )
    pool = jnp.zeros((4, 16, 128), jnp.bfloat16)
    out, _ = kda._pooled_kda_core(
        *_pooled_args(pool),
        lower_bound=None,
        eps=1e-6,
        pool_block_tokens=16,
        conv_shard_heads=1,
    )
    assert out.shape == (T, H, D)
    assert calls[-1][4] == (N, 2, 128)


@pytest.mark.cpu_test
def test_pooled_core_rejects_bad_geometry(monkeypatch):
    _pool_fakes(monkeypatch, [])
    pool3 = jnp.zeros((4, 16, 128), jnp.bfloat16)
    pool4 = jnp.zeros((4, 16, 3, 128), jnp.bfloat16)
    kw = dict(lower_bound=None, eps=1e-6)
    with pytest.raises(ValueError, match="whole number of pool rows"):
        kda._pooled_kda_core(*_pooled_args(pool4), pool_block_tokens=16, **kw)
    with pytest.raises(ValueError, match="whole pool blocks"):
        kda._pooled_kda_core(*_pooled_args(pool3), pool_block_tokens=24, **kw)
    monkeypatch.setattr(
        kda,
        "derive_pooled_gdn_state_layout",
        lambda **_: SimpleNamespace(ssm_tokens=1, conv_tokens=1, required_tokens=99),
    )
    with pytest.raises(ValueError, match="exceed the pool block"):
        kda._pooled_kda_core(*_pooled_args(pool3), pool_block_tokens=16, **kw)
    monkeypatch.setattr(kda, "derive_pooled_gdn_state_layout", _fake_layout)
    for heads in (0, 3):
        with pytest.raises(ValueError, match="whole source head shards"):
            kda._pooled_kda_core(
                *_pooled_args(pool3), pool_block_tokens=16, conv_shard_heads=heads, **kw
            )
    with pytest.raises(ValueError, match="whole packed pool rows"):
        kda._pooled_kda_core(
            *_pooled_args(pool3), pool_block_tokens=16, conv_shard_heads=1, **kw
        )
    # 16 bf16 lanes: the SSM region still fits in 4 rows, one conv row does not.
    monkeypatch.setattr(
        kda,
        "derive_pooled_gdn_state_layout",
        lambda **_: SimpleNamespace(ssm_tokens=4, conv_tokens=1, required_tokens=5),
    )
    small = jnp.zeros((4, 16, 16), jnp.bfloat16)
    with pytest.raises(ValueError, match="exceeds its padded pool region"):
        kda._pooled_kda_core(*_pooled_args(small), pool_block_tokens=16, **kw)


def _vllm_config(block_size=16, num_heads=H):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": num_heads})
        ),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("raiden", [False, True])
def test_pooled_op_builder(monkeypatch, raiden):
    monkeypatch.setitem(vars(tpu_envs), "TPU_USE_RAIDEN_KV_CACHE_MANAGER", raiden)
    monkeypatch.setitem(vars(tpu_envs), "TPU_RAIDEN_KIMIK3_ADMISSION", True)
    monkeypatch.setitem(vars(tpu_envs), "TPU_RAIDEN_TRANSFER_PARALLELISM", 2)
    built = []

    def jax_op(name, fn, donate_argnums):
        op = _FakeJaxOp(
            name,
            fn,
            donate_argnums,
            lambda *a: (torch.ones(a[0].shape[0], H, D), a[4] + 5),
        )
        built.append(op)
        return op

    monkeypatch.setattr(kda.pallas, "jax_op", jax_op)
    seen = {}
    monkeypatch.setattr(
        kda, "_pooled_kda_core", lambda *a, **kw: seen.update(kw) or ("out", "pool")
    )
    impl = kda.build_kimi_pooled_kda_op(
        "m.l0", lower_bound=-1.0, eps=1e-5, vllm_config=_vllm_config()
    )
    (op,) = built
    suffix = "_conv_rank_blocks_v1_h1" if raiden else ""
    assert op.name == f"pallas::kimi_pooled_kda_fused_m_l0_conv_bfloat16{suffix}"
    assert op.donate == (4,)
    assert op.fn(*range(13)) == ("out", "pool")
    assert seen["pool_block_tokens"] == 16
    assert seen["conv_shard_heads"] == (1 if raiden else None)
    mixed = torch.zeros(T, 3 * H * D)
    pool = torch.zeros(4, 16, 128)
    out, fpool = op.fake(
        mixed, None, None, None, pool, None, torch.zeros(H), None, torch.zeros(D), 1
    )
    assert out.shape == (T, H, D) and fpool.shape == pool.shape
    result = impl(mixed, None, None, None, pool, *([None] * 8))
    assert result.shape == (T, H, D) and torch.all(pool == 5)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp", [0, 3])
def test_pooled_op_builder_rejects_bad_source_tp(monkeypatch, tp):
    monkeypatch.setitem(vars(tpu_envs), "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setitem(vars(tpu_envs), "TPU_RAIDEN_KIMIK3_ADMISSION", True)
    monkeypatch.setitem(vars(tpu_envs), "TPU_RAIDEN_TRANSFER_PARALLELISM", tp)
    with pytest.raises(ValueError, match="source TP must divide"):
        kda.build_kimi_pooled_kda_op(
            "m", lower_bound=None, eps=1e-6, vllm_config=_vllm_config()
        )


# ===========================================================================
# gdn_attention_op
# ===========================================================================


def _gdn_attn(prefix="m.layers.0.linear_attn", *, tp_size=1, bias=False):
    attn = gdn.VllmGatedDeltaNetAttention.__new__(gdn.VllmGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads = 4
    attn.num_v_heads = 4
    attn.gqa_interleaved_layout = False
    attn.tp_size = tp_size
    attn.head_k_dim = 8
    attn.head_v_dim = 8
    attn.value_dim = 32
    attn.conv_kernel_size = 4
    attn.num_spec = 0
    attn.conv1d = SimpleNamespace(
        weight=torch.zeros(1), bias=torch.zeros(1) if bias else None
    )
    attn.A_log = torch.zeros(1)
    attn.dt_bias = torch.zeros(1)
    attn.model_config = SimpleNamespace(
        dtype=torch.bfloat16, architecture="Qwen3_5MoeForCausalLM"
    )
    attn.cache_config = SimpleNamespace(
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32"
    )
    attn._pcp_streaming_configured = False
    return attn


def _gdn_vllm_config(*, pcp_size=1, interleave_size=16):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        ),
        cache_config=SimpleNamespace(block_size=16),
    )


def _gdn_mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


@pytest.mark.cpu_test
def test_gdn_ssm_state_dtype_mapping():
    assert gdn._to_jax_ssm_state_dtype(torch.float32) == jnp.float32
    assert gdn._to_jax_ssm_state_dtype(torch.bfloat16) == jnp.bfloat16
    with pytest.raises(ValueError, match="Unsupported mamba_ssm_cache_dtype"):
        gdn._to_jax_ssm_state_dtype(torch.float16)


def _fake_run(seen):
    def run(*args, **kw):
        seen.append((args, kw))
        conv_in, rec = args[3], args[4]
        return (conv_in + 1, rec + 2), jnp.zeros((args[0].shape[0], 2))

    return run


@pytest.mark.cpu_test
@pytest.mark.parametrize("state_len", [3, 5])  # K - 1 and widened by 2 spec cols
def test_gdn_core_tpu_slices_and_restores_spec_tail(monkeypatch, state_len):
    seen = []
    monkeypatch.setattr(gdn, "run_jax_gdn_attention", _fake_run(seen))
    conv = jnp.zeros((2, state_len, 6))
    rec = jnp.zeros((2, 1))
    args = (jnp.zeros((3, 6)), None, None, conv, rec) + (None,) * 8
    kw = dict(mesh=None, n_kq=1, n_v=1, d_k=2, d_v=2, kernel_size=4)
    new_conv, new_rec, out = gdn.gdn_attention_core_tpu(*args, **kw)
    assert seen[0][0][3].shape == (2, 3, 6)
    assert new_conv.shape == conv.shape
    np.testing.assert_array_equal(np.asarray(new_conv[:, :3]), 1)
    np.testing.assert_array_equal(np.asarray(new_conv[:, 3:]), 0)
    np.testing.assert_array_equal(np.asarray(new_rec), 2)

    seen.clear()
    monkeypatch.setattr(gdn, "run_jax_gdn_attention_pcp_tp_prefill", _fake_run(seen))
    new_conv, _, _ = gdn.gdn_attention_core_tpu_pcp_prefill(
        *args[:13], **kw, pcp_size=2, interleave_size=16
    )
    assert seen[0][0][3].shape == (2, 3, 6) and new_conv.shape == conv.shape


@pytest.mark.cpu_test
def test_gdn_pooled_core_forwards_everything(monkeypatch):
    seen = []
    monkeypatch.setattr(
        gdn, "run_jax_gdn_attention_pooled", lambda *a, **kw: seen.append(kw) or "ok"
    )
    assert (
        gdn.gdn_attention_pooled_core_tpu(
            *range(12),
            mesh="m",
            n_kq=1,
            n_v=2,
            d_k=3,
            d_v=4,
            kernel_size=5,
            pool_block_tokens=16,
            recurrent_state_dtype=jnp.float32,
            num_spec_tokens=1,
        )
        == "ok"
    )
    assert seen[0]["pool_block_tokens"] == 16 and seen[0]["num_spec_tokens"] == 1


@pytest.mark.cpu_test
def test_gdn_localize_spec_early_return_and_mismatches(monkeypatch):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    empty = SimpleNamespace(shapes=((), (4, 8, 8)))
    assert (
        gdn._localize_gdn_mamba_spec_for_pcp(
            empty, 2, num_kq_heads=4, num_v_heads=4, d_k=8, d_v=8
        )
        is empty
    )
    wide = SimpleNamespace(shapes=((3, 99), (4, 8, 8)))
    with pytest.raises(ValueError, match="conv width"):
        gdn._localize_gdn_mamba_spec_for_pcp(
            wide, 2, num_kq_heads=4, num_v_heads=4, d_k=8, d_v=8
        )
    heads = SimpleNamespace(shapes=((3, 96), (3, 8, 8)))
    with pytest.raises(ValueError, match="recurrent heads"):
        gdn._localize_gdn_mamba_spec_for_pcp(
            heads, 2, num_kq_heads=4, num_v_heads=4, d_k=8, d_v=8
        )


@pytest.mark.cpu_test
def test_gdn_kv_cache_spec_none_and_state_dtype_per_layer(monkeypatch):
    attn = _gdn_attn()
    monkeypatch.setattr(
        gdn.QwenGatedDeltaNetAttention, "get_kv_cache_spec", lambda self, cfg: None
    )
    assert attn.get_kv_cache_spec(None) is None
    monkeypatch.setattr(
        gdn.QwenGatedDeltaNetAttention,
        "get_state_dtype",
        lambda self: (torch.bfloat16, torch.bfloat16),
    )
    monkeypatch.setattr(
        gdn, "unified_kv_layout_enabled_for_architecture", lambda arch: False
    )
    assert attn.get_state_dtype() == (torch.float32, torch.bfloat16)


class _GdnJaxOp:
    def __init__(self, *args, returns, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.returns = returns
        self.fake = None

    def register_fake(self, fn):
        self.fake = fn

    def __call__(self, *args):
        self.args_seen = args
        return self.returns(*args)


@pytest.mark.cpu_test
@pytest.mark.parametrize("pcp", [False, True])
def test_gdn_build_op_fake_and_impl(monkeypatch, pcp):
    attn = _gdn_attn(bias=True)
    built = []

    def jax_op(*args, **kwargs):
        op = _GdnJaxOp(
            *args,
            returns=lambda *a: (a[3] + 1, a[4] + 2, torch.ones(a[0].shape[0], 4, 8)),
            **kwargs,
        )
        built.append(op)
        return op

    monkeypatch.setattr(gdn.pallas, "jax_op", jax_op)
    monkeypatch.setattr(
        gdn,
        "get_or_create_pcp_mesh",
        lambda **_: SimpleNamespace(shape={"pcp": 2, "tp": 1}),
    )
    monkeypatch.setattr(gdn, "get_pcp_world_size", lambda: 2)
    with set_vllm_model_wrapper_context(
        mesh=_gdn_mesh(), vllm_config=_gdn_vllm_config(pcp_size=2)
    ):
        impl = attn._build_gdn_op(pcp_streaming=pcp)
    (op,) = built
    mixed = torch.zeros(3, 96)
    conv = torch.zeros(2, 3, 96)
    rec = torch.zeros(2, 4, 8, 8)
    f_conv, f_rec, f_out = op.fake(mixed, None, None, conv, rec, 1, x=2)
    assert f_conv.shape == conv.shape and f_rec.shape == rec.shape
    assert f_out.shape == (3, 4, 8)
    out = impl(mixed, None, None, conv, rec, *([None] * 8), "offsets")
    assert out.shape == (3, 4, 8)
    assert torch.all(conv == 1) and torch.all(rec == 2)
    assert len(op.args_seen) == (13 if pcp else 14)


@pytest.mark.cpu_test
def test_gdn_build_pcp_op_rejects_bad_config(monkeypatch):
    attn = _gdn_attn()
    monkeypatch.setattr(gdn.pallas, "jax_op", MagicMock())
    mesh = SimpleNamespace(shape={"pcp": 2, "tp": 1})
    monkeypatch.setattr(gdn, "get_or_create_pcp_mesh", lambda **_: mesh)
    cases = [
        (0, 2, 1, "cp_kv_cache_interleave_size"),
        (16, 4, 1, "PCP world size"),
        (16, 2, 2, "TP size"),
    ]
    for build in (
        lambda: attn._build_gdn_op(pcp_streaming=True),
        lambda: attn._build_pooled_pcp_gdn_op(),
    ):
        for interleave, world, tp, match in cases:
            attn.tp_size = tp
            monkeypatch.setattr(gdn, "get_pcp_world_size", lambda w=world: w)
            with (
                set_vllm_model_wrapper_context(
                    mesh=_gdn_mesh(),
                    vllm_config=_gdn_vllm_config(interleave_size=interleave),
                ),
                pytest.raises(ValueError, match=match),
            ):
                build()
    attn.tp_size = 1


@pytest.mark.cpu_test
def test_gdn_pooled_op_fake(monkeypatch):
    attn = _gdn_attn()
    built = []
    monkeypatch.setattr(gdn, "_shared_gdn_ops", {})
    monkeypatch.setattr(
        gdn.pallas,
        "jax_op",
        lambda *a, **kw: built.append(_GdnJaxOp(*a, returns=None, **kw)) or built[-1],
    )
    with set_vllm_model_wrapper_context(
        mesh=_gdn_mesh(), vllm_config=_gdn_vllm_config()
    ):
        attn._build_pooled_gdn_op()
    pool = torch.zeros(4, 16, 128)
    new_pool, out = built[0].fake(torch.zeros(3, 96), None, None, pool, 1, y=2)
    assert new_pool.shape == pool.shape and out.shape == (3, 4, 8)


@pytest.mark.cpu_test
def test_gdn_pooled_pcp_block_major_wrapped_fn(monkeypatch):
    attn = _gdn_attn()
    built = []
    monkeypatch.setattr(
        gdn.pallas,
        "jax_op",
        lambda *a, **kw: built.append(_GdnJaxOp(*a, returns=None, **kw)) or built[-1],
    )
    monkeypatch.setattr(
        gdn,
        "get_or_create_pcp_mesh",
        lambda **_: SimpleNamespace(shape={"pcp": 2, "tp": 1}),
    )
    monkeypatch.setattr(gdn, "get_pcp_world_size", lambda: 2)
    monkeypatch.setattr(gdn, "kernel_view", lambda pool: pool.reshape(-1, 4))
    monkeypatch.setattr(
        gdn,
        "run_jax_gdn_attention_pooled_pcp_prefill_projection",
        lambda *a, **kw: (a[5] + 1, "out", "z"),
    )
    with set_vllm_model_wrapper_context(
        mesh=_gdn_mesh(), vllm_config=_gdn_vllm_config(pcp_size=2)
    ):
        attn._build_pooled_pcp_gdn_op(block_major=True)
    wrapped = built[0].args[1]
    pool = jnp.zeros((2, 2, 4))
    new_pool, out, z = wrapped(*([None] * 4), pool, *([None] * 9))
    assert new_pool.shape == pool.shape and (out, z) == ("out", "z")


@pytest.mark.cpu_test
def test_gdn_pcp_projection_requires_combined_qkvz():
    attn = _gdn_attn()
    with pytest.raises(RuntimeError, match="combined QKVZ"):
        attn._require_pcp_projection_parameters()


def _gdn_forward_attn(monkeypatch, metadata, *, kv_cache, op_out_tokens=None):
    attn = _gdn_attn()
    attn.kv_cache = kv_cache
    attn.in_proj_qkvz = MagicMock(return_value=(torch.zeros(2, 96 + 32), None))
    attn.in_proj_ba = MagicMock(return_value=(torch.zeros(2, 8), None))
    attn.norm = MagicMock(side_effect=lambda x, z: x)
    attn.out_proj = MagicMock(side_effect=lambda x: (x, None))
    tokens = 2 if op_out_tokens is None else op_out_tokens
    attn.gdn_op = MagicMock(return_value=torch.zeros(tokens, 4, 8))
    attn.gdn_pooled_op = MagicMock(return_value=torch.zeros(tokens, 4, 8))
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={attn.prefix: metadata}),
    )
    monkeypatch.setattr(gdn, "is_pcp_streaming_attention_metadata", lambda md: False)
    return attn


def _gdn_metadata(**overrides):
    md = SimpleNamespace(
        mamba_state_indices=torch.tensor([3, 4]),
        query_start_loc=torch.tensor([0, 1, 2]),
        request_distribution=torch.tensor([2, 2, 2]),
        seq_lens=torch.tensor([1, 1]),
        mamba_slot_read_offsets=None,
        mamba_request_distribution=None,
        mamba_ckpt_indices=None,
    )
    vars(md).update(overrides)
    return md


@pytest.mark.cpu_test
def test_gdn_forward_without_cache_returns_projected_zeros(monkeypatch):
    attn = _gdn_forward_attn(monkeypatch, None, kv_cache=None)
    out = attn.forward(torch.zeros(2, 16))
    assert out.shape == (2, 32) and torch.all(out == 0)
    attn.gdn_op.assert_not_called()


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_spec", [0, 1])
def test_gdn_forward_unified_pool(monkeypatch, num_spec):
    mrd = torch.tensor([1, 2, 2])
    md = _gdn_metadata(
        mamba_request_distribution=mrd,
        mamba_slot_read_offsets=torch.zeros(2, dtype=torch.int32),
    )
    pool = torch.zeros(4, 16, 128)
    attn = _gdn_forward_attn(monkeypatch, md, kv_cache=(pool,))
    attn.num_spec = num_spec
    out = attn.forward(torch.zeros(2, 16))
    assert out.shape == (2, 32)
    args = attn.gdn_pooled_op.call_args.args
    assert args[3] is pool and args[10] is mrd
    assert args[8].dtype == torch.int32


@pytest.mark.cpu_test
def test_gdn_forward_unified_pool_rejects_bad_output(monkeypatch):
    attn = _gdn_forward_attn(
        monkeypatch, _gdn_metadata(), kv_cache=(torch.zeros(4),), op_out_tokens=3
    )
    with pytest.raises(RuntimeError, match="incompatible output shape"):
        attn.forward(torch.zeros(2, 16))


@pytest.mark.cpu_test
def test_gdn_forward_dense_cache_mamba_distribution_and_bad_output(monkeypatch):
    mrd = torch.tensor([1, 2, 2])
    md = _gdn_metadata(mamba_request_distribution=mrd)
    caches = (torch.zeros(4), torch.zeros(4))
    attn = _gdn_forward_attn(monkeypatch, md, kv_cache=caches)
    attn.forward(torch.zeros(2, 16))
    assert attn.gdn_op.call_args.args[11] is mrd
    attn = _gdn_forward_attn(monkeypatch, md, kv_cache=caches, op_out_tokens=3)
    with pytest.raises(RuntimeError, match="incompatible output shape"):
        attn.forward(torch.zeros(2, 16))


@pytest.mark.cpu_test
def test_gdn_fused_pooled_pcp_guards(monkeypatch):
    attn = _gdn_attn()
    monkeypatch.setattr(gdn, "is_pcp_streaming_attention_metadata", lambda md: False)
    with pytest.raises(RuntimeError, match="PCP streaming sequence layout"):
        attn._forward_fused_pooled_pcp(torch.zeros(2, 16), torch.zeros(4), None)
    monkeypatch.setattr(gdn, "is_pcp_streaming_attention_metadata", lambda md: True)
    attn.gdn_pooled_pcp_op = MagicMock(
        return_value=(torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))
    )
    attn._require_pcp_projection_parameters = lambda: (torch.zeros(1), None)
    attn.in_proj_ba = MagicMock(return_value=(torch.zeros(2, 8), None))
    with pytest.raises(RuntimeError, match="incompatible output shape"):
        attn._forward_fused_pooled_pcp(
            torch.zeros(2, 16), torch.zeros(4), _gdn_metadata()
        )
