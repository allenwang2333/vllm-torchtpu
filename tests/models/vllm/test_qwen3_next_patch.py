# SPDX-License-Identifier: Apache-2.0
"""Pin upstream Qwen3Next forwards and verify trace-stable SP forward patch."""

import hashlib
import inspect
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    UPSTREAM_MTP_FORWARD,
    maybe_patch_qwen3_next,
    patched_decoder_layer_forward,
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
