"""Decode-time token-local q for TP-replicated layers (TPU_DEEPSEEK_V41_DECODE_LOCAL_Q).

Targets the diff of:
- src/vllm_torchtpu/envs.py (TPU_DEEPSEEK_V41_DECODE_LOCAL_Q)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py
  (`_replicated_local_q`, forward's ``qr_local`` gate, `_prepare_tpu`'s
  ``replicated_qr`` kwarg, `_init_local_q_weights` gating)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
  (``replicated_qr`` in `fp8_attention_qo`, `fp8_attention_qo_2bank` and the
  torch wrapper `sparse_fp8_attention_qo`)

The last test runs the real `fp8_attention_qo` body on an eight-device CPU
mesh (real ``shard_map`` / ``all_to_all``) and checks it is bit-identical,
per rank, to the production token-parallel sequence it replaces.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

_QO_OP = "pallas::dsv41v2_sparse_fp8_attention_qo"
_QO_2BANK_OP = "pallas::dsv41v2_sparse_fp8_attention_qo_2bank"
_OP_GLOBALS = (
    "_attention_op",
    "_attention_2bank_op",
    "_fp8_attention_op",
    "_fp8_attention_2bank_op",
    "_fp8_attention_qo_op",
    "_fp8_attention_qo_2bank_op",
)


# ==============================================================================
# Part 1: envs.py
# ==============================================================================
class TestEnvsDecodeLocalQ:
    def test_default_value(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TPU_DEEPSEEK_V41_DECODE_LOCAL_Q", raising=False)
        assert envs.environment_variables["TPU_DEEPSEEK_V41_DECODE_LOCAL_Q"]() is False

    @pytest.mark.parametrize(
        ("val_str", "expected"),
        [("1", True), ("true", True), ("0", False), ("false", False)],
    )
    def test_parsing_values(
        self, monkeypatch: pytest.MonkeyPatch, val_str: str, expected: bool
    ):
        monkeypatch.setenv("TPU_DEEPSEEK_V41_DECODE_LOCAL_Q", val_str)
        assert (
            envs.environment_variables["TPU_DEEPSEEK_V41_DECODE_LOCAL_Q"]() is expected
        )


# ==============================================================================
# Part 2: v2/attention.py _init_local_q_weights gating
# ==============================================================================
def _fake_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    other = tensor ^ 0xFF if tensor.dtype == torch.uint8 else -tensor
    return torch.cat((tensor, other), dim=dim)


def _stub_attn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    local_q: bool = False,
    decode_local_q: bool = False,
    sp_sharded: bool = False,
    ced_decoder: bool = False,
    tp: int = 2,
):
    monkeypatch.setattr(v2_attn.envs, "TPU_DEEPSEEK_V41_LOCAL_Q", local_q)
    monkeypatch.setattr(v2_attn.envs, "TPU_DEEPSEEK_V41_DECODE_LOCAL_Q", decode_local_q)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(v2_attn, "tensor_model_parallel_all_gather", _fake_all_gather)

    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._fp8_kv = True
    attn.compress_ratio = 4
    attn.n_local_groups = 1
    attn._token_parallel_attn = True
    attn._sp_sharded_input = sp_sharded
    attn._ced_decoder = ced_decoder

    gen = torch.Generator().manual_seed(7)
    wq = torch.randn(16, 8, generator=gen).to(torch.float8_e4m3fn)
    wo = torch.randn(32, 4, generator=gen).to(torch.float8_e4m3fn)
    attn.wq_b = SimpleNamespace(weight=wq, weight_scale=torch.rand(8, generator=gen))
    attn.wo_a = SimpleNamespace(weight=wo, weight_scale=torch.rand(1, 4, generator=gen))
    return attn


class TestInitLocalQWeightsGating:
    """`_replicated_local_q` follows DECODE_LOCAL_Q on TP-replicated layers only."""

    def test_class_default(self):
        assert v2_attn.DeepseekV41Attention._replicated_local_q is False
        assert v2_attn.DeepseekV41Attention._local_q_attn is False

    @pytest.mark.parametrize(
        ("kwargs", "enabled", "replicated"),
        [
            # Decoder layer at decode: replicated input, DECODE flag only.
            (dict(decode_local_q=True, ced_decoder=True), True, True),
            # Any TP-replicated layer qualifies, not only the CED decoder.
            (dict(decode_local_q=True), True, True),
            # DECODE flag does nothing for SP-sharded layers...
            (dict(decode_local_q=True, sp_sharded=True), False, False),
            # ...which the LOCAL_Q flag serves, as before.
            (dict(local_q=True, sp_sharded=True), True, False),
            (dict(local_q=True, decode_local_q=True, sp_sharded=True), True, False),
            # LOCAL_Q alone still skips the CED decoder.
            (dict(local_q=True, ced_decoder=True), False, False),
            # LOCAL_Q alone on a replicated non-decoder layer (pre-existing).
            (dict(local_q=True), True, False),
            (dict(), False, False),
            # tp == 1 never gathers.
            (dict(decode_local_q=True, tp=1), False, False),
        ],
    )
    def test_flag_matrix(
        self, monkeypatch: pytest.MonkeyPatch, kwargs, enabled, replicated
    ):
        attn = _stub_attn(monkeypatch, **kwargs)
        attn._init_local_q_weights()
        assert attn._local_q_attn is enabled
        assert attn._replicated_local_q is replicated
        assert hasattr(attn, "_wq_b_full") is enabled
        assert hasattr(attn, "_wo_a_full") is enabled


# ==============================================================================
# Part 3: forward() qr_local gate for the replicated path
# ==============================================================================
class TestForwardReplicatedLocalQ:
    def _forward_attn(self, monkeypatch, *, local_q_attn: bool):
        attn = _stub_attn(monkeypatch)
        attn._sp_sharded_input = False
        attn._replicated_local_q = True
        attn._local_q_attn = local_q_attn
        attn.padded_heads = attn.n_local_heads = 2
        attn.head_dim = 4
        attn.q_lora_rank = 3
        attn.indexer = None
        attn.compressor = None
        return attn

    def test_replicated_passes_full_qr_without_sp_gather(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = self._forward_attn(monkeypatch, local_q_attn=True)
        tokens = 4
        qr = torch.ones(tokens, 3)
        kv = torch.zeros(tokens, 4)
        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))

        def no_gather(t):
            raise AssertionError("replicated layers must not SP all-gather")

        monkeypatch.setattr(v2_attn, "sp_all_gather", no_gather)

        seen = {}
        projected = torch.full((tokens, 5), 2.0)

        def prepare(*args, qr_local=None):
            seen["qr_local"] = qr_local
            seen["qr"] = args[1]
            return projected

        attn._prepare_and_attn_fn = prepare
        attn.wo_b = lambda x: (x * 3, None)
        out = attn.forward(
            torch.zeros(tokens, dtype=torch.long), torch.zeros(tokens, 8)
        )
        assert seen["qr_local"] is qr
        assert seen["qr"] is qr
        assert torch.equal(out, projected * 3)

    def test_replicated_without_gathered_weights_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = self._forward_attn(monkeypatch, local_q_attn=False)
        tokens = 4
        qr = torch.ones(tokens, 3)
        kv = torch.zeros(tokens, 4)
        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        seen = {}

        def prepare(*args, qr_local=None):
            seen["qr_local"] = qr_local
            return None

        attn._prepare_and_attn_fn = prepare
        monkeypatch.setattr(attn, "_o_proj", lambda o, positions: "o_proj")
        out = attn.forward(
            torch.zeros(tokens, dtype=torch.long), torch.zeros(tokens, 8)
        )
        assert out == "o_proj"
        assert seen["qr_local"] is None


# ==============================================================================
# Part 4: _prepare_tpu forwards replicated_qr
# ==============================================================================
class _FakeForwardContext:
    def __init__(self, attn_metadata):
        self.attn_metadata = attn_metadata
        self.additional_kwargs = {}


def _prepare_tpu_attn(monkeypatch: pytest.MonkeyPatch, *, replicated: bool):
    metadata = {
        "swa": SimpleNamespace(
            block_table=torch.zeros((1, 4), dtype=torch.int32),
            slot_mapping=torch.zeros((4,), dtype=torch.int64),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ),
        "comp": SimpleNamespace(
            block_table=torch.zeros((1, 4), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ),
    }
    monkeypatch.setattr(
        v2_attn, "get_forward_context", lambda: _FakeForwardContext(metadata)
    )
    monkeypatch.setattr(v2_attn, "_block_table", lambda swa: swa.block_table)
    monkeypatch.setattr(
        v2_attn, "_cache_slots", lambda cache, swa, pos: swa.slot_mapping
    )
    monkeypatch.setattr(
        v2_attn,
        "_token_requests",
        lambda meta, pos: torch.zeros(pos.shape[0], dtype=torch.long),
    )
    monkeypatch.setattr(
        v2_attn,
        "_row_slots",
        lambda cache, table, reqs, logical: torch.zeros_like(
            logical, dtype=torch.int32
        ),
    )

    attn = _stub_attn(monkeypatch, local_q=True)
    attn.window_size = 64
    attn.head_dim = 128
    attn.scale = 0.125
    attn.n_local_heads = 4
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn.compressor = None
    attn._tpu_rope_cache = torch.zeros(1)
    attn.attn_sink = torch.zeros(4)
    attn._full_attn_sink = torch.zeros(8)
    attn._tp_rank_1d = torch.zeros(1, dtype=torch.int32)
    attn._swa_max_requests = 16
    attn._precompact_records = False
    attn.compressed_cache_prefix = "comp"
    attn._max_model_len = 64
    attn.indexer = None
    attn.topk_indices_buffer = torch.zeros((4, 64), dtype=torch.int32)
    fake_cache = mock.MagicMock(spec=PagedCacheRows)
    fake_cache.numel.return_value = 100
    fake_cache.row_indices = lambda slots: slots.int()
    fake_cache.pages = lambda table: None
    attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=fake_cache)
    attn.main_cache_layer = SimpleNamespace(prefix="main", kv_cache=fake_cache)
    attn._static_forward_context = {"comp": SimpleNamespace(kv_cache=fake_cache)}
    attn._rotate = lambda x, pos: x
    attn._init_local_q_weights()
    assert attn._local_q_attn is True
    attn._replicated_local_q = replicated
    return attn


class TestPrepareTpuReplicatedQr:
    @pytest.mark.parametrize("replicated", [True, False])
    def test_replicated_qr_kwarg(self, monkeypatch: pytest.MonkeyPatch, replicated):
        attn = _prepare_tpu_attn(monkeypatch, replicated=replicated)
        called = {}

        def mock_qo(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return torch.full((4, 1024), 1.0)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention_qo", mock_qo)
        monkeypatch.setattr(v2_attn.envs, "TPU_DEEPSEEK_V41_FUSED_SWA", replicated)
        qr = torch.randn(4, 16)
        res = v2_attn.DeepseekV41Attention._prepare_tpu(
            attn,
            hidden_states=None,
            qr=qr,
            kv=torch.randn(4, 128),
            qr_scale=None,
            kv_score=None,
            index_weights=None,
            positions=torch.zeros(4, dtype=torch.long),
            output=torch.zeros(4, 4, 128),
            qr_local=qr,
        )
        assert res is not None
        assert called["args"][0] is qr and called["args"][1] is qr
        assert called["kwargs"] == {
            "replicated_qr": replicated,
            "fused_swa": replicated,
        }


# ==============================================================================
# Part 5: attention_kernels.py torch wrapper and JAX bodies
# ==============================================================================
def _wrapper_args(qr, main, swa):
    main_records = torch.ones(2, 4, dtype=torch.long)
    return (
        qr,
        qr,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(qr.shape[0]),
        torch.zeros(1),
        swa,
        main,
        torch.zeros(2, 4),
        main_records,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(qr.shape[0]),
        0.125,
        torch.zeros((4, 2), dtype=torch.int32),
        torch.zeros((4,), dtype=torch.int32),
        16,
        False,
        128,
    )


class TestSparseFp8AttentionQoWrapper:
    def test_replicated_qr_is_last_tail_arg_1bank(self):
        saved = attention_kernels._fp8_attention_qo_op
        mock_op = mock.MagicMock(return_value="out")
        attention_kernels._fp8_attention_qo_op = mock_op
        try:
            main = SimpleNamespace(
                rope=torch.zeros(1), raw=torch.zeros(8, 64, 128), shape=(16, 64, 128)
            )
            swa = SimpleNamespace(raw=torch.zeros(8, 64, 128))
            qr = torch.zeros(96, 16)
            # Default keeps the SP semantics.
            assert (
                attention_kernels.sparse_fp8_attention_qo(*_wrapper_args(qr, main, swa))
                == "out"
            )
            assert mock_op.call_args[0][-3] == 128  # head_dim
            assert mock_op.call_args[0][-2] is False  # replicated_qr
            assert mock_op.call_args[0][-1] is False  # fused_swa
            attention_kernels.sparse_fp8_attention_qo(
                *_wrapper_args(qr, main, swa), replicated_qr=True, fused_swa=True
            )
            assert mock_op.call_args[0][-2] is True
            assert mock_op.call_args[0][-1] is True
            assert len(mock_op.call_args[0]) == 30
        finally:
            attention_kernels._fp8_attention_qo_op = saved

    def test_replicated_qr_is_last_tail_arg_2bank(self):
        saved = (
            attention_kernels._fp8_attention_qo_op,
            attention_kernels._fp8_attention_qo_2bank_op,
        )
        mock_2bank = mock.MagicMock(return_value="out2")
        attention_kernels._fp8_attention_qo_op = mock.MagicMock()
        attention_kernels._fp8_attention_qo_2bank_op = mock_2bank
        try:
            main = SimpleNamespace(
                rope=torch.zeros(1),
                rope_hi=torch.zeros(1),
                raw=torch.zeros(8, 64, 128),
                raw_hi=torch.zeros(8, 64, 128),
                shape=(16, 64, 128),
            )
            swa = SimpleNamespace(
                raw=torch.zeros(8, 64, 128), raw_hi=torch.zeros(8, 64, 128)
            )
            qr = torch.zeros(96, 16)
            assert (
                attention_kernels.sparse_fp8_attention_qo(
                    *_wrapper_args(qr, main, swa), replicated_qr=True
                )
                == "out2"
            )
            assert mock_2bank.call_args[0][-2] is True
            assert mock_2bank.call_args[0][-1] is False
            assert len(mock_2bank.call_args[0]) == 33
        finally:
            (
                attention_kernels._fp8_attention_qo_op,
                attention_kernels._fp8_attention_qo_2bank_op,
            ) = saved


def _capture_jax_ops(tp_size: int):
    """Re-run `initialize_paged_attention` recording every ``jax_op`` body.

    Returns a context manager yielding ``{op_name: python_fn}``; the module's
    op globals are restored on exit.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from torch_tpu._internal import pallas

        saved = {name: getattr(attention_kernels, name) for name in _OP_GLOBALS}
        captured = {}
        real_jax_op = pallas.jax_op

        def recorder(name, fn, **kwargs):
            captured[name] = fn
            return real_jax_op(name, fn, **kwargs)

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = tp_size
        try:
            with (
                set_vllm_model_wrapper_context(mesh=mesh),
                mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
                mock.patch(
                    "torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh
                ),
                mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=recorder),
            ):
                for name in _OP_GLOBALS:
                    setattr(attention_kernels, name, None)
                attention_kernels.initialize_paged_attention()
                yield captured
        finally:
            for name, value in saved.items():
                setattr(attention_kernels, name, value)

    return _ctx()


def _qo_args(tokens: int, *, q_lora: int = 32, head_dim: int = 128, tp: int = 8):
    """The 23 array args of `fp8_attention_qo` for ``tokens`` with full ``qr``."""
    qr = jnp.arange(tokens, dtype=jnp.float32)[:, None].repeat(q_lora, axis=1)
    qr = qr.astype(jnp.bfloat16)
    heads = 8
    return (
        qr,
        qr,
        jnp.zeros((q_lora, heads * head_dim)),
        jnp.zeros((heads * head_dim,)),
        jnp.zeros((q_lora, tp * heads * head_dim)),
        jnp.zeros((tp * heads * head_dim,)),
        jnp.zeros((tokens,), dtype=jnp.int32),
        jnp.zeros((256, 64)),
        jnp.zeros((16, 64, 128)),
        jnp.zeros((16, 64, 128)),
        jnp.zeros((16, 64, 64)),
        jnp.arange(tokens * 4, dtype=jnp.int32).reshape(tokens, 4),
        jnp.arange(tokens * 2, dtype=jnp.int32).reshape(tokens, 2),
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.array([0, tokens], dtype=jnp.int32),
        jnp.zeros((heads,)),
        jnp.zeros((tp * heads,)),
        jnp.array([3], dtype=jnp.int32),
        jnp.zeros((heads * head_dim, 64)),
        jnp.zeros((64,)),
        jnp.zeros((heads * head_dim, tp * 64)),
        jnp.zeros((tp * 64,)),
        jnp.ones((tokens,), dtype=jnp.int32),
    )


class TestFp8AttentionQoReplicatedBody:
    """`fp8_attention_qo(replicated_qr=True)` on a mocked single-device mesh."""

    def _run(
        self,
        fn,
        tokens,
        *,
        tp=8,
        replicated=True,
        head_dim=128,
        sp_rows=None,
        fused_swa=False,
    ):
        seen = {}

        def fake_matmul(q_in, w, ws):
            seen["matmul_in"] = q_in
            return jnp.zeros((q_in.shape[0], w.shape[1]), dtype=jnp.bfloat16)

        def fake_attn(q, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, **kwargs):
            seen["attn_q"] = q
            seen["attn_s_r"] = s_r
            seen["attn_m_r"] = m_r
            seen["attn_kwargs"] = kwargs
            return q

        def fake_o_proj(out, pos, cs, w, ws, head_dim):
            seen["o_proj_rows"] = out.shape[0]
            seen["o_proj_w"] = w
            return jnp.zeros((out.shape[0], w.shape[1]), dtype=jnp.bfloat16)

        with (
            mock.patch(
                "jax.shard_map",
                side_effect=lambda body, **kwargs: lambda *args: body(*args),
            ),
            mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x) as a2a,
            mock.patch("jax.lax.all_gather", side_effect=lambda x, **kw: x) as ag,
            mock.patch(
                "vllm_torchtpu.layers.adapter.linear_common._quantized_matmul_jax",
                side_effect=fake_matmul,
            ),
            mock.patch(
                "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
                side_effect=lambda q, pos, cs: q,
            ),
            mock.patch.object(
                attention_kernels, "sparse_fp8_attention_jax", side_effect=fake_attn
            ),
            mock.patch(
                "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
                side_effect=fake_o_proj,
            ),
        ):
            arrays = _qo_args(tokens, head_dim=head_dim, tp=tp)
            if sp_rows is not None:
                # An SP token shard of qr, as the encoder layers pass it.
                arrays = (arrays[0], arrays[0][:sp_rows], *arrays[2:])
            out = fn(*arrays, 0.125, 64, None, False, head_dim, replicated, fused_swa)
        seen["a2a_calls"] = a2a.call_count
        seen["all_gather_calls"] = ag.call_count
        seen["qr"] = arrays[0]
        return out, seen

    def test_padded_token_shard(self):
        # T=96, tp=8: tokens_local=12, pad_rows=4; rank 3 owns tokens 36..47.
        with _capture_jax_ops(8) as ops:
            out, seen = self._run(ops[_QO_OP], 96)
        assert out.shape == (96, 64)
        assert np.array_equal(seen["matmul_in"], seen["qr"][36:48])
        assert seen["attn_q"].shape == (16, 64, 128)
        assert seen["attn_s_r"].shape == (16, 4)
        assert np.array_equal(
            seen["attn_s_r"][:12], np.arange(96 * 4).reshape(96, 4)[36:48]
        )
        assert np.all(seen["attn_s_r"][12:] == -1)
        assert np.all(seen["attn_m_r"][12:] == -1)
        assert seen["o_proj_rows"] == 12
        assert seen["o_proj_w"].shape == (1024, 512)  # wo_a_full
        # Only the return all_to_all remains; no q all_to_all, no all_gather.
        assert seen["a2a_calls"] == 1
        assert seen["all_gather_calls"] == 0

    def test_aligned_token_shard_has_no_pad(self):
        # T=192, tp=8: tokens_local=24 is already a multiple of 8.
        with _capture_jax_ops(8) as ops:
            out, seen = self._run(ops[_QO_OP], 192)
        assert out.shape == (192, 64)
        assert seen["attn_q"].shape == (24, 64, 128)
        assert np.array_equal(seen["matmul_in"], seen["qr"][72:96])
        assert seen["o_proj_rows"] == 24
        assert seen["a2a_calls"] == 1
        assert seen["attn_kwargs"]["fused_swa"] is False

    def test_fused_swa_reaches_attention(self):
        with _capture_jax_ops(8) as ops:
            _, seen = self._run(ops[_QO_OP], 96, fused_swa=True)
        assert seen["attn_kwargs"]["fused_swa"] is True
        assert seen["attn_kwargs"]["max_requests"] is None
        with _capture_jax_ops(8) as ops:
            _, seen = self._run(ops[_QO_OP], 24, fused_swa=True)
        assert seen["attn_kwargs"]["fused_swa"] is True

    @pytest.mark.parametrize("tokens", [24, 50])
    def test_below_gate_uses_head_sharded_body(self, tokens):
        # T=24 (< 6 * tp) and T=50 (% tp != 0) fall back to the head-sharded
        # body with the local weights; the full qr needs no all_gather.
        with _capture_jax_ops(8) as ops:
            out, seen = self._run(ops[_QO_OP], tokens)
        assert out.shape == (tokens, 64)
        assert seen["matmul_in"].shape == (tokens, 32)
        assert seen["o_proj_w"].shape == (1024, 64)  # local wo_a
        assert seen["a2a_calls"] == 0
        assert seen["all_gather_calls"] == 0

    def test_sp_semantics_unchanged_by_default(self):
        # replicated_qr=False keeps the 128 * tp SP gate: T=96 is head-sharded.
        with _capture_jax_ops(8) as ops:
            out, seen = self._run(ops[_QO_OP], 96, replicated=False)
        assert out.shape == (96, 64)
        assert seen["o_proj_w"].shape == (1024, 64)
        assert seen["a2a_calls"] == 0

    def test_sp_local_body_takes_shard_as_is(self):
        # replicated_qr=False with T=128 * tp: the SP shard (128 rows) feeds
        # the matmul unsliced; pad_rows is 0.
        with _capture_jax_ops(8) as ops:
            out, seen = self._run(ops[_QO_OP], 1024, replicated=False, sp_rows=128)
        assert out.shape == (1024, 64)
        assert np.array_equal(seen["matmul_in"], seen["qr"][:128])
        assert seen["attn_q"].shape == (128, 64, 128)
        assert seen["o_proj_w"].shape == (1024, 512)
        assert seen["a2a_calls"] == 1

    def test_2bank_forwards_replicated_qr(self):
        with _capture_jax_ops(8) as ops:
            fn = ops[_QO_2BANK_OP]

            def two_bank(*a):
                # swa_lo, swa_hi, main_lo, main_hi, rope_lo, rope_hi.
                pools = (a[8], a[8], a[9], a[9], a[10], a[10])
                return fn(*a[:8], *pools, *a[11:])

            out, seen = self._run(two_bank, 96)
        assert out.shape == (96, 64)
        assert seen["attn_q"].shape == (16, 64, 128)
        assert seen["a2a_calls"] == 1

    def test_fake_shapes_accept_extra_static(self):
        with _capture_jax_ops(8):
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                qr = torch.empty(96, 1280)
                tensors = [torch.empty(1) for _ in range(21)]
                tensors[4] = torch.empty(96)
                tensors[16] = torch.empty(512, 1024)
                out = attention_kernels._fp8_attention_qo_op(
                    qr, qr, *tensors, 0.125, 64, None, True, 128, True, True
                )
                assert out.shape == (96, 1024)
                tensors_2bank = tensors[:8] + [torch.empty(1)] * 3 + tensors[8:]
                out2 = attention_kernels._fp8_attention_qo_2bank_op(
                    qr, qr, *tensors_2bank, 0.125, 64, None, True, 128, True, True
                )
                assert out2.shape == (96, 1024)


# ==============================================================================
# Part 6: bit-exactness vs the production token-parallel sequence (8 CPU devices)
# ==============================================================================
def test_replicated_local_q_bit_exact_vs_token_parallel_8_cpu_devices():
    """Real shard_map / all_to_all on 8 CPU devices: new path == production path."""
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "_dsv41_v2_decode_local_q_exactness.py",
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    ).strip()
    env["TPU_SKIP_MDS_QUERY"] = "true"
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(envs.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src_root, env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-P", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-6000:]}"
    )
    assert "EXACTNESS_OK" in proc.stdout, proc.stdout
    assert proc.stdout.count("bit-exact on 8 ranks") == 3
