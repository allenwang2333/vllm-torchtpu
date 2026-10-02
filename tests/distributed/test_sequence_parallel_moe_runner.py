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
"""Unit tests for sequence-parallel MoE configuration and Qwen3-Next/3.5 patch."""

from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm import qwen3_next_patch


@pytest.fixture(autouse=True)
def _install_sp_patch():
    from vllm.config.parallel import ParallelConfig

    if not getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        vllm_torchtpu._patch_disable_sequence_parallel_moe()


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        (None, "off"),
        ("", "off"),
        ("0", "off"),
        ("false", "off"),
        ("1", "auto"),
        ("True", "auto"),
        ("auto", "auto"),
        (" FORCE ", "force"),
    ],
)
def test_sequence_parallel_mode_parser(monkeypatch, raw, mode):
    name = "VLLM_TPU_TEST_SP_MOE_MODE"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert envs.env_moe_sequence_parallel_mode(name)() == mode


def test_sequence_parallel_mode_parser_rejects_typos(monkeypatch):
    name = "VLLM_TPU_TEST_SP_MOE_MODE"
    monkeypatch.setenv(name, "forced")
    with pytest.raises(ValueError, match="forced"):
        envs.env_moe_sequence_parallel_mode(name)()


def _sp_moe_enabled(
    monkeypatch, *, mode, dp=1, live_dp=None, env_dp=None, tp=4, ep=True
):
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import parallel_state

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", mode)
    monkeypatch.setattr(
        parallel_state,
        "_DP",
        None if live_dp is None else SimpleNamespace(world_size=live_dp),
    )
    if env_dp is None:
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TORCH_TPU_DP_SIZE", str(env_dp))
    config = SimpleNamespace(
        enable_expert_parallel=ep,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=1,
        data_parallel_size=dp,
    )
    assert ParallelConfig.use_sequence_parallel_moe.fget is not None
    return ParallelConfig.use_sequence_parallel_moe.fget(config)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(mode="off", dp=4), False),
        # "auto" follows upstream: SP only when DP > 1.
        (dict(mode="auto", dp=1), False),
        (dict(mode="auto", dp=2), True),
        # Per-engine configs may collapse data_parallel_size to 1; the live DP
        # group or TORCH_TPU_DP_SIZE still reveal the real DP width.
        (dict(mode="auto", dp=1, live_dp=4), True),
        (dict(mode="auto", dp=1, env_dp=4), True),
        # "force" also enables DP == 1 (TP-only + EP).
        (dict(mode="force", dp=1), True),
        # Structural requirements hold in every mode.
        (dict(mode="force", dp=1, tp=1), False),
        (dict(mode="force", dp=2, ep=False), False),
    ],
)
def test_use_sequence_parallel_moe_gate(monkeypatch, kwargs, expected):
    assert _sp_moe_enabled(monkeypatch, **kwargs) is expected


@pytest.mark.parametrize(
    ("mode", "model_type", "expected"),
    [
        ("off", "qwen3_5_moe", False),
        ("auto", None, False),
        ("auto", "qwen2_5_vl", False),
        ("auto", "qwen3_next", None),
        ("force", "qwen3_5_moe", None),
        ("force", "qwen3_5_moe_text", None),
    ],
)
def test_maybe_patch_qwen3_next_model_gate(monkeypatch, mode, model_type, expected):
    from vllm.model_executor.models import qwen3_next

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", mode)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer,
        "forward",
        qwen3_next_patch.UPSTREAM_DECODER_LAYER_FORWARD,
    )
    monkeypatch.setattr(
        qwen3_next.Qwen3NextModel,
        "forward",
        qwen3_next_patch.UPSTREAM_MODEL_FORWARD,
    )
    model_config = (
        None
        if model_type is None
        else SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )
    result = qwen3_next_patch.maybe_patch_qwen3_next(model_config)
    assert result is expected
    if expected is None:
        assert (
            qwen3_next.Qwen3NextDecoderLayer.forward
            is qwen3_next_patch.patched_decoder_layer_forward
        )
        assert (
            qwen3_next.Qwen3NextModel.forward is qwen3_next_patch.patched_model_forward
        )
    else:
        assert (
            qwen3_next.Qwen3NextDecoderLayer.forward
            is qwen3_next_patch.UPSTREAM_DECODER_LAYER_FORWARD
        )
        assert (
            qwen3_next.Qwen3NextModel.forward is qwen3_next_patch.UPSTREAM_MODEL_FORWARD
        )


def test_patched_decoder_layer_forward_uses_narrow_and_reduce_scatter(monkeypatch):
    monkeypatch.setattr(
        qwen3_next_patch, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim: torch.cat([x, x + 10], dim=dim),
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim: x[: x.shape[0] // 2] + x[x.shape[0] // 2 :],
    )

    mlp_calls = []

    def mlp(x, already_sequence_parallel=False):
        mlp_calls.append((x.shape, already_sequence_parallel))
        return x * 2

    layer = SimpleNamespace(
        input_layernorm=lambda h, r=None: (h, r) if r is not None else h,
        post_attention_layernorm=lambda h, r: (h + r, r),
        use_attn_reduce_scatter_for_moe=True,
        layer_type="full_attention",
        self_attn=lambda hidden_states, positions: hidden_states + 1,
        layer_scale=False,
        mlp=mlp,
    )
    chunk = torch.ones(2, 3)
    positions = torch.arange(4)

    out, res = qwen3_next_patch.patched_decoder_layer_forward(
        layer, chunk, None, positions
    )

    assert mlp_calls == [(torch.Size([2, 3]), True)]
    assert out.shape == (2, 3)
    assert res.shape == (2, 3)
