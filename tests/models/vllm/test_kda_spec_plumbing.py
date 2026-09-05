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
"""Speculative-decode operands through the KDA op and the KDA layer.

Pallas cannot run here, so nothing below executes the kernel. What these
cover is the plumbing that is silent if wrong: whether the read-offset
buffer reaches the wrapper gathered to per-sequence, whether the static
window size reaches both the wrapper and the op name, whether the manual
path refuses a window it cannot honour, and whether the layer forwards
the mamba-specific request distribution instead of the RPA one.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.vllm.custom_ops import kda_attention_op as kda_ops
from vllm_torchtpu.models.vllm.kimi_k3 import KimiDeltaAttention

NUM_HEADS = 2
HEAD_DIM = 2
NUM_TOKENS = 2
NUM_SEQS = 1
NUM_SLOTS = 4
KERNEL_SIZE = 3


def _fused_core_operands(*, slot_read_offsets: bool):
    """A minimally valid operand set for the fused core, on CPU jnp arrays."""
    mixed_dim = 3 * NUM_HEADS * HEAD_DIM
    operands = dict(
        mixed_qkv=jnp.zeros((NUM_TOKENS, mixed_dim), jnp.float32),
        raw_gate=jnp.zeros((NUM_TOKENS, NUM_HEADS * HEAD_DIM), jnp.float32),
        beta=jnp.zeros((NUM_TOKENS, NUM_HEADS), jnp.float32),
        output_gate=jnp.zeros((NUM_TOKENS, NUM_HEADS * HEAD_DIM), jnp.float32),
        conv_state=jnp.zeros(
            (NUM_SLOTS, KERNEL_SIZE - 1, 3, NUM_HEADS, HEAD_DIM), jnp.float32),
        recurrent_state=jnp.zeros((NUM_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM),
                                  jnp.float32),
        conv_weight=jnp.zeros((KERNEL_SIZE, 3, NUM_HEADS, HEAD_DIM),
                              jnp.float32),
        a_log=jnp.zeros((NUM_HEADS, ), jnp.float32),
        dt_bias=jnp.zeros((NUM_HEADS * HEAD_DIM, ), jnp.float32),
        norm_weight=jnp.ones((HEAD_DIM, ), jnp.float32),
        query_start_loc=jnp.array([0, NUM_TOKENS], jnp.int32),
        # Slot 3 is this sequence's checkpoint-group base.
        state_indices=jnp.array([3], jnp.int32),
        seq_lens=jnp.array([NUM_TOKENS], jnp.int32),
        distribution=jnp.array([0, 0, NUM_SEQS], jnp.int32),
    )
    if slot_read_offsets:
        # Per *slot*, not per sequence: only entry 3 is this request's.
        operands["slot_read_offsets"] = jnp.array([0, 0, 0, 2], jnp.int32)
    return operands


def _stub_wrapper(monkeypatch) -> list[dict]:
    """Replace the Pallas wrapper with a recorder; returns the call log."""
    calls: list[dict] = []

    def fake_fused_conv1d_gdn(**kwargs):
        calls.append(kwargs)
        conv_state = kwargs["conv_state"]
        recurrent_state = kwargs["recurrent_state"]
        out = jnp.zeros((kwargs["qkv"].shape[0], NUM_HEADS * HEAD_DIM),
                        kwargs["qkv"].dtype)
        return (conv_state, recurrent_state), out

    monkeypatch.setattr(kda_ops.gdn_wrapper, "fused_conv1d_gdn",
                        fake_fused_conv1d_gdn)
    return calls


def test_fused_core_gathers_read_offsets_by_state_index(monkeypatch) -> None:
    calls = _stub_wrapper(monkeypatch)
    core = kda_ops._build_fused_core(lower_bound=None,
                                     eps=1e-5,
                                     num_spec_tokens=3)

    core(**_fused_core_operands(slot_read_offsets=True))

    (kwargs, ) = calls
    # The wrapper asserts `read_offsets.shape == (num_seqs,)`; handing it the
    # whole per-slot buffer is exactly the bug GDN hit when the gather was
    # dropped.
    assert kwargs["read_offsets"].shape == (NUM_SEQS, )
    assert kwargs["read_offsets"].tolist() == [2]
    assert kwargs["num_spec_tokens"] == 3
    # The dense path addresses its checkpoint group as base + offset, so it
    # must not be handed a per-checkpoint block index array.
    assert "ckpt_indices" not in kwargs


def test_fused_core_without_spec_passes_no_read_offsets(monkeypatch) -> None:
    calls = _stub_wrapper(monkeypatch)
    core = kda_ops._build_fused_core(lower_bound=None,
                                     eps=1e-5,
                                     num_spec_tokens=0)

    core(**_fused_core_operands(slot_read_offsets=False))

    (kwargs, ) = calls
    assert kwargs["read_offsets"] is None
    assert kwargs["num_spec_tokens"] == 0


@pytest.mark.parametrize(
    "shape, dtype_name, error",
    [
        ((NUM_SLOTS, ), "float32", TypeError),
        ((NUM_SLOTS, 1), "int32", ValueError),
    ],
    ids=["float-buffer", "rank-2-buffer"],
)
def test_fused_core_rejects_malformed_read_offsets(monkeypatch, shape,
                                                   dtype_name, error) -> None:
    # Built inside the test, not in the decorator: a jnp array at collection
    # time initializes the JAX backend and claims the accelerator for the
    # whole pytest process (see conftest's guard).
    _stub_wrapper(monkeypatch)
    core = kda_ops._build_fused_core(lower_bound=None,
                                     eps=1e-5,
                                     num_spec_tokens=1)
    operands = _fused_core_operands(slot_read_offsets=True)
    operands["slot_read_offsets"] = jnp.zeros(shape, getattr(jnp, dtype_name))

    with pytest.raises(error):
        core(**operands)


def _capture_op_name(monkeypatch) -> list[str]:
    names: list[str] = []

    def jax_op(name, function, donate_argnums=()):
        del donate_argnums
        names.append(name)

        def implementation(*args, **kwargs):
            del args, kwargs
            raise AssertionError("not executed")

        registered = SimpleNamespace(register_fake=lambda fake: None)
        registered.__call__ = implementation
        return registered

    monkeypatch.setattr(kda_ops.pallas, "jax_op", jax_op)
    return names


def test_op_name_encodes_num_spec(monkeypatch) -> None:
    names = _capture_op_name(monkeypatch)

    for num_spec in (0, 3):
        kda_ops.build_kimi_dispatched_kda_op(
            "layers.0.self_attn",
            lower_bound=None,
            eps=1e-5,
            state_dim_first=True,
            num_spec=num_spec,
        )

    # vLLM's compile cache is keyed on neither the env vars nor num_spec, so
    # two builds that trace different kernels must not share a name.
    assert names[0] != names[1]
    assert "spec0" in names[0]
    assert "spec3" in names[1]


def test_manual_path_refuses_a_window(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_KDA_FUSED", "0")

    # At build time, so the failure lands at model construction rather than
    # at the first verify step.
    with pytest.raises(NotImplementedError, match="manual KDA path"):
        kda_ops.build_kimi_dispatched_kda_op(
            "layers.0.self_attn",
            lower_bound=None,
            eps=1e-5,
            state_dim_first=True,
            num_spec=2,
        )


def test_manual_core_refuses_a_window_and_its_operand() -> None:
    operands = _fused_core_operands(slot_read_offsets=True)
    offsets = operands.pop("slot_read_offsets")
    positional = [
        operands[name]
        for name in ("mixed_qkv", "raw_gate", "beta", "output_gate",
                     "conv_state", "recurrent_state", "conv_weight", "a_log",
                     "dt_bias", "norm_weight", "query_start_loc",
                     "state_indices", "seq_lens", "distribution")
    ]

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        kda_ops._dispatched_kda_core(*positional,
                                     None,
                                     lower_bound=None,
                                     eps=1e-5,
                                     num_spec_tokens=2)

    # An offset buffer with no window is just as wrong: it would be dropped.
    with pytest.raises(NotImplementedError, match="slot read offsets"):
        kda_ops._dispatched_kda_core(*positional,
                                     offsets,
                                     lower_bound=None,
                                     eps=1e-5,
                                     num_spec_tokens=0)


class _Projection(nn.Module):

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        values = torch.arange(
            inputs.shape[0] * self.output_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return values.view(inputs.shape[0], self.output_size), None


class _IdentityProjection(nn.Module):

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return inputs, None


class _ConvWeight(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 1, 3))


def _kda_layer(num_spec: int) -> KimiDeltaAttention:
    """The __new__ + inject harness from test_kimi_k3.py.

    A real constructor wants a VllmConfig, TP groups and a device platform,
    which is precisely what is unavailable off-TPU.
    """
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.conv_size, layer.num_heads, layer.head_dim = 3, 2, 2
    KimiDeltaAttention.process_weights_after_loading(layer,
                                                     act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.num_spec = num_spec
    layer.kv_cache = (torch.zeros(1, 8, 3), torch.zeros(1, 2, 2, 2))
    return layer


def test_forward_prefers_the_mamba_request_distribution() -> None:
    layer = _kda_layer(num_spec=2)
    rpa_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    mamba_distribution = torch.tensor([1, 1, 1], dtype=torch.int32)
    read_offsets = torch.tensor([1], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=rpa_distribution,
        mamba_request_distribution=mamba_distribution,
        mamba_slot_read_offsets=read_offsets,
    )
    layer._metadata = lambda: metadata

    seen: list[tuple] = []

    def dispatched_kda_op(mixed_qkv, *args):
        seen.append(args)
        return mixed_qkv[:, :4].view(-1, 2, 2)

    layer.dispatched_kda_op = dispatched_kda_op
    layer(torch.arange(2), torch.ones(2, 4))

    (args, ) = seen
    # With a verify window the front segment is every windowed sequence, not
    # only the 1-token decodes RPA counts.
    assert args[12] is mamba_distribution
    assert args[12] is not rpa_distribution
    # Slot-indexed and passed whole; the op gathers it with state_indices.
    assert args[13] is read_offsets


def test_forward_passes_no_offsets_without_spec() -> None:
    layer = _kda_layer(num_spec=0)
    rpa_distribution = torch.tensor([0, 0, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=rpa_distribution,
    )
    layer._metadata = lambda: metadata

    seen: list[tuple] = []

    def dispatched_kda_op(mixed_qkv, *args):
        seen.append(args)
        return mixed_qkv[:, :4].view(-1, 2, 2)

    layer.dispatched_kda_op = dispatched_kda_op
    layer(torch.arange(2), torch.ones(2, 4))

    (args, ) = seen
    assert args[12] is rpa_distribution
    assert args[13] is None


def test_forward_requires_offsets_when_spec_is_on() -> None:
    layer = _kda_layer(num_spec=2)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata
    layer.dispatched_kda_op = lambda *args: (_ for _ in ()).throw(
        AssertionError("op must not be reached"))

    with pytest.raises(AssertionError, match="mamba_slot_read_offsets"):
        layer(torch.arange(2), torch.ones(2, 4))


@pytest.mark.parametrize("num_speculative_tokens, expected", [(3, 3),
                                                              (None, 0)])
def test_constructor_accepts_a_speculative_config(num_speculative_tokens,
                                                  expected) -> None:
    """The old NotImplementedError is gone and num_spec comes from the config.

    __init__ cannot run to completion off-TPU (it builds vLLM parallel
    linears), so drive it into the very next failure -- the missing
    linear_attn_config -- and check what it recorded on the way past where
    the guard used to be.
    """
    speculative_config = (None if num_speculative_tokens is None else
                          SimpleNamespace(
                              num_speculative_tokens=num_speculative_tokens))
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    config = SimpleNamespace(linear_attn_config=None)
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)

    with pytest.raises(ValueError, match="linear_attn_config"):
        KimiDeltaAttention.__init__(layer, config, vllm_config,
                                    "layers.0.self_attn")

    assert layer.num_spec == expected


def test_state_shape_is_not_widened_by_num_spec() -> None:
    """TPU rollback selects a checkpoint slot; it has no conv-input window.

    vLLM's own kda_state_shape adds num_spec columns to the conv cache for
    GPU rollback and the TPU GDN op slices them straight back off. This shape
    must stay identical to whatever classmethod the owning model exposes as
    get_mamba_state_shape_from_config, because vLLM sizes the pool from that
    one and addresses it through this one.
    """
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    layer.conv_size, layer.num_heads, layer.head_dim = 4, 2, 8
    layer.num_spec = 3

    conv_shape, recurrent_shape = KimiDeltaAttention.get_state_shape(layer)

    assert conv_shape == (3, 3, 2, 8)
    assert recurrent_shape == (2, 8, 8)
