# SPDX-License-Identifier: Apache-2.0
"""Pin upstream Qwen3Next forwards and verify trace-stable SP forward and FP8 fusions."""

import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    UPSTREAM_MTP_FORWARD,
    compute_fused_gate_and_shared_expert,
    maybe_flip_gdn_in_proj_qkvz_weight,
    maybe_patch_qwen3_next,
    maybe_setup_fused_gate_se,
    pallas_sigmoid_gate_quant,
    patched_decoder_layer_forward,
    patched_sparse_moe_forward,
    sigmoid_gate_quant_reference,
)

pytestmark = pytest.mark.cpu_test

PINNED_DECODER_LAYER_HASH = (
    "359c047f120f585e0c519a3bc438f037f996f4d48c7b64495dadddc59a709b4a"
)
PINNED_MODEL_FORWARD_HASH = (
    "a9c1265f14e88e54138394b9038db621813bb6357a9077ad50d1691db0ac4398"
)
PINNED_MTP_FORWARD_HASH = (
    "54034d76e48fccf2cd98bcf05b831207580ad9e8886e6e387193e1bd70e9e472"
)


def test_upstream_qwen3_next_forward_unchanged():
    """Fail when upstream vLLM modifies any method, requiring a re-sync."""
    actual_decoder_hash = hashlib.sha256(
        inspect.getsource(UPSTREAM_DECODER_LAYER_FORWARD).encode("utf-8")
    ).hexdigest()
    assert actual_decoder_hash == PINNED_DECODER_LAYER_HASH, (
        "Upstream Qwen3NextDecoderLayer.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )

    actual_model_hash = hashlib.sha256(
        inspect.getsource(UPSTREAM_MODEL_FORWARD).encode("utf-8")
    ).hexdigest()
    assert actual_model_hash == PINNED_MODEL_FORWARD_HASH, (
        "Upstream Qwen3NextModel.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )

    actual_mtp_hash = hashlib.sha256(
        inspect.getsource(UPSTREAM_MTP_FORWARD).encode("utf-8")
    ).hexdigest()
    assert actual_mtp_hash == PINNED_MTP_FORWARD_HASH, (
        "Upstream Qwen3NextMultiTokenPredictor.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )


def test_forward_patch_skipped_with_flag_off(monkeypatch):
    """When TPU_MOE_SEQUENCE_PARALLEL=0, methods remain the upstream objects."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    result = maybe_patch_qwen3_next()
    assert result is False
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert qwen3_next.Qwen3NextModel.forward is UPSTREAM_MODEL_FORWARD
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is UPSTREAM_MTP_FORWARD


def test_forward_patch_applied_for_qwen3_next(monkeypatch):
    """When enabled for Qwen3Next, forwards are patched with trace stability."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer,
        "forward",
        UPSTREAM_DECODER_LAYER_FORWARD,
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)
    monkeypatch.setattr(
        qwen3_next_mtp.Qwen3NextMultiTokenPredictor,
        "forward",
        UPSTREAM_MTP_FORWARD,
    )

    result = maybe_patch_qwen3_next()
    assert result is None
    assert (
        qwen3_next.Qwen3NextDecoderLayer.forward is not UPSTREAM_DECODER_LAYER_FORWARD
    )
    assert qwen3_next.Qwen3NextModel.forward is not UPSTREAM_MODEL_FORWARD
    assert (
        qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward
        is not UPSTREAM_MTP_FORWARD
    )


def test_patched_decoder_layer_forward_mtp_entry_prechunked(monkeypatch):
    """Verify MTP layer entry: residual=None with pre-chunked hidden_states."""
    full_num_tokens = 16
    tp_world_size = 8
    chunk = full_num_tokens // tp_world_size
    hidden_size = 64

    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen3_next_patch.get_tensor_model_parallel_world_size",
        lambda: tp_world_size,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen3_next_patch.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    def fake_all_gather(x, dim=0):
        return x.repeat(tp_world_size, 1)

    def fake_reduce_scatter(x, dim=0):
        return x.narrow(0, 0, chunk)

    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen3_next_patch.tensor_model_parallel_all_gather",
        fake_all_gather,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen3_next_patch.tensor_model_parallel_reduce_scatter",
        fake_reduce_scatter,
    )

    layer = MagicMock()
    layer.use_attn_reduce_scatter_for_moe = True
    layer.layer_type = "linear_attention"
    layer.layer_scale = False
    layer.input_layernorm.side_effect = lambda h, r=None: (h, r) if r is not None else h
    layer.linear_attn.side_effect = lambda hidden_states: hidden_states
    layer.post_attention_layernorm.side_effect = lambda h, r: (h, r)
    layer.mlp.side_effect = lambda h, already_sequence_parallel=False: h

    prechunked_hidden = torch.randn(chunk, hidden_size)
    positions = torch.arange(full_num_tokens).unsqueeze(0)

    out_h, out_r = patched_decoder_layer_forward(
        layer,
        hidden_states=prechunked_hidden,
        residual=None,
        positions=positions,
    )

    assert out_h.shape == (chunk, hidden_size)
    assert out_r.shape == (chunk, hidden_size)


def test_pal_gate_q_interpreter_parity():
    """Verify pallas_sigmoid_gate_quant in Pallas CPU interpreter."""
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    for m in (4, 8, 16, 64):
        a = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        g = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        q_pal, s_pal = pallas_sigmoid_gate_quant(a, g, interpret=True)
        q_ref, s_ref = sigmoid_gate_quant_reference(a, g)
        assert q_pal.shape == (m, 256) and s_pal.shape == (m, 1)
        np.testing.assert_allclose(
            np.asarray(s_pal, np.float32),
            np.asarray(s_ref, np.float32),
            rtol=1e-2,
            atol=1e-3,
        )
        deq_pal = np.asarray(q_pal, np.float32) * np.asarray(s_pal, np.float32)
        deq_ref = np.asarray(q_ref, np.float32) * np.asarray(s_ref, np.float32)
        np.testing.assert_allclose(deq_pal, deq_ref, rtol=0.15, atol=0.15)
        assert np.mean(np.abs(deq_pal - deq_ref)) < 5e-3


def test_maybe_flip_gdn_in_proj_qkvz_weight():
    """Verify GDN in_proj_qkvz weight is flipped to (k, n) only when PCP is off."""
    k_in, n_out = 256, 128
    w_nk = torch.ones((n_out, k_in), dtype=torch.float8_e4m3fn)
    w_scale = torch.full((n_out,), 0.01, dtype=torch.float32)

    def _make_gdn(pcp: bool):
        proj = nn.Module()
        proj.register_parameter(
            "weight", nn.Parameter(w_nk.clone(), requires_grad=False)
        )
        proj.register_parameter(
            "weight_scale", nn.Parameter(w_scale.clone(), requires_grad=False)
        )
        proj.bias = None
        setattr(proj, KEEP_VLLM_LAYOUT_ATTR, True)
        return SimpleNamespace(
            _pcp_streaming_configured=pcp,
            in_proj_qkvz=proj,
        )

    gdn_pcp = _make_gdn(pcp=True)
    assert maybe_flip_gdn_in_proj_qkvz_weight(gdn_pcp) is False
    assert gdn_pcp.in_proj_qkvz.weight.shape == (n_out, k_in)
    assert getattr(gdn_pcp.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, False) is True

    gdn_non_pcp = _make_gdn(pcp=False)
    assert maybe_flip_gdn_in_proj_qkvz_weight(gdn_non_pcp) is True
    assert gdn_non_pcp.in_proj_qkvz.weight.shape == (k_in, n_out)
    assert getattr(gdn_non_pcp.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, False) is False
    assert getattr(gdn_non_pcp.in_proj_qkvz, WEIGHT_FLIPPED_ATTR, False) is True


def test_cat768_fp8_fusion_and_weight_deletion():
    """Verify Cat_FP8_256 fuses [gate; gate_up_proj] into 256-column-aligned FP8
    ([4096, 768] at TP=8 I=128 and [4096, 2560] at TP=1 I=1024), folds sigmoid(se_gate)
    into down_proj per-token scale, deletes original weights, and achieves cos > 0.9988.
    """
    torch.manual_seed(0)
    hidden_dim = 4096
    n_gate = 512
    tokens = 16

    for n_inter, tp_size, rep_se in ((128, 8, False), (1024, 1, True)):
        n_se_gu = 2 * n_inter
        w_gate = torch.randn(n_gate, hidden_dim, dtype=torch.bfloat16) * 0.02
        w_se_gu_bf16 = torch.randn(hidden_dim, n_se_gu, dtype=torch.bfloat16) * 0.02
        s_se_gu = w_se_gu_bf16.float().abs().amax(dim=0) / 448.0
        w_se_gu_fp8 = (
            (w_se_gu_bf16.float() / s_se_gu.unsqueeze(0))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        w_se_down_bf16 = torch.randn(n_inter, hidden_dim, dtype=torch.bfloat16) * 0.02
        s_se_down = w_se_down_bf16.float().abs().amax(dim=0) / 448.0
        w_se_down_fp8 = (
            (w_se_down_bf16.float() / s_se_down.unsqueeze(0))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        w_se_gate = torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 0.02

        x = torch.randn(tokens, hidden_dim, dtype=torch.bfloat16)

        ref_router_logits = F.linear(x, w_gate)
        ref_gu = torch.matmul(x.float(), w_se_gu_fp8.float() * s_se_gu.unsqueeze(0)).to(
            torch.bfloat16
        )
        ref_inter = F.silu(ref_gu[:, :n_inter]) * ref_gu[:, n_inter:]
        ref_down = torch.matmul(
            ref_inter.float(), w_se_down_fp8.float() * s_se_down.unsqueeze(0)
        ).to(torch.bfloat16)
        ref_se_gate = F.linear(x, w_se_gate)
        ref_shared_out = ref_down * torch.sigmoid(ref_se_gate)

        gate_mod = nn.Module()
        gate_mod.register_parameter(
            "weight", nn.Parameter(w_gate.clone(), requires_grad=False)
        )
        gate_mod.bias = None

        gu_mod = nn.Module()
        gu_mod.register_parameter(
            "weight", nn.Parameter(w_se_gu_fp8.clone(), requires_grad=False)
        )
        gu_mod.register_parameter(
            "weight_scale", nn.Parameter(s_se_gu.float().clone(), requires_grad=False)
        )
        gu_mod.bias = None
        setattr(gu_mod, WEIGHT_FLIPPED_ATTR, True)

        down_mod = nn.Module()
        down_mod.register_parameter(
            "weight", nn.Parameter(w_se_down_fp8.clone(), requires_grad=False)
        )
        down_mod.register_parameter(
            "weight_scale", nn.Parameter(s_se_down.float().clone(), requires_grad=False)
        )
        down_mod.bias = None
        setattr(down_mod, WEIGHT_FLIPPED_ATTR, True)

        se_gate_mod = nn.Module()
        se_gate_mod.register_parameter(
            "weight", nn.Parameter(w_se_gate.clone(), requires_grad=False)
        )
        se_gate_mod.bias = None

        moe = nn.Module()
        moe.replicate_shared_expert = rep_se
        moe.use_sequence_parallel = False
        moe.tp_size = tp_size
        moe.gate = gate_mod
        moe.shared_expert = SimpleNamespace(
            gate_up_proj=gu_mod,
            down_proj=down_mod,
            act_fn=lambda t: F.silu(t[:, :n_inter]) * t[:, n_inter:],
        )
        moe.shared_expert_gate = se_gate_mod
        moe.experts = type(
            "_E",
            (),
            {
                "gate": gate_mod,
                "shared_expert_gate": se_gate_mod,
                "_fse_fuse_gate": True,
                "_shared_experts": moe.shared_expert,
                "moe_config": SimpleNamespace(
                    skip_final_all_reduce=False, is_sequence_parallel=False
                ),
                "maybe_all_reduce_tensor_model_parallel": lambda self, t: t,
                "__call__": lambda self, hidden_states, router_logits: hidden_states
                * 0.5,
            },
        )()

        assert maybe_setup_fused_gate_se(moe) is True
        assert getattr(moe, "_fused_gate_se_is_fp8", False) is True
        assert moe._fused_gate_gu_fp8_weight.shape == (hidden_dim, n_gate + n_se_gu)
        assert (n_gate + n_se_gu) % 256 == 0
        assert moe._fused_gate_gu_fp8_weight.dtype == torch.float8_e4m3fn
        assert moe._fused_se_down_fp8_weight.shape == (n_inter, hidden_dim)
        assert not hasattr(moe.gate, "weight")
        assert not hasattr(moe.shared_expert.gate_up_proj, "weight")
        assert not hasattr(moe.shared_expert.gate_up_proj, "weight_scale")
        assert not hasattr(moe.shared_expert.down_proj, "weight")
        assert not hasattr(moe.shared_expert.down_proj, "weight_scale")
        assert not hasattr(moe.shared_expert_gate, "weight")

        got_logits, got_shared = compute_fused_gate_and_shared_expert(moe, x)
        cos_logits = F.cosine_similarity(
            got_logits.float().flatten(), ref_router_logits.float().flatten(), dim=0
        ).item()
        cos_shared = F.cosine_similarity(
            got_shared.float().flatten(), ref_shared_out.float().flatten(), dim=0
        ).item()
        assert cos_logits > 0.999, f"cos_logits={cos_logits}"
        assert cos_shared > 0.9988, f"cos_shared={cos_shared}"

        got_moe_out = patched_sparse_moe_forward(moe, x)
        cos_moe = F.cosine_similarity(
            got_moe_out.float().flatten(),
            (x * 0.5 + ref_shared_out).float().flatten(),
            dim=0,
        ).item()
        assert cos_moe > 0.999, f"cos_moe={cos_moe}"
