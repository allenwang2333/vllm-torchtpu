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

from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.layers.common.experimental.chunked_fused_moe_gmm import \
    chunked_fused_moe_func
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func
from vllm_torchtpu.layers.vllm import fused_moe
from vllm_torchtpu.layers.vllm.fused_moe import get_fused_moe_activation


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


@pytest.mark.parametrize(
    "linear_beta,expected",
    [
        (None, "situ:4.0:none"),
        (25.0, "situ:4.0:25.0"),
    ],
)
def test_get_fused_moe_activation_encodes_situ(linear_beta, expected):
    moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
    )

    assert get_fused_moe_activation("situ", moe_config) == expected


def test_get_fused_moe_activation_requires_situ_beta():
    moe_config = SimpleNamespace(
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )

    with pytest.raises(AssertionError):
        get_fused_moe_activation("situ", moe_config)


def test_get_fused_moe_activation_compiles():

    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0


@pytest.mark.parametrize("chunk_size,expected_chunked", [(0, False),
                                                         (128, True)])
def test_build_fused_moe_custom_op_chunk_selection(monkeypatch, chunk_size,
                                                   expected_chunked):
    """Verifies that _build_fused_moe_custom_op wraps chunked_fused_moe_func when chunk_size > 0."""
    monkeypatch.setattr(fused_moe.envs, "VLLM_TPU_MOE_CHUNK_SIZE", chunk_size)

    captured_fn = None

    def fake_jax_op(name, fn):
        nonlocal captured_fn
        captured_fn = fn
        mock_op = MagicMock()
        mock_op.register_fake = MagicMock()
        return mock_op

    monkeypatch.setattr(fused_moe.pallas, "jax_op", fake_jax_op)

    fused_moe._build_fused_moe_custom_op(
        topk=2,
        activation="silu",
        use_ep=False,
        skip_padded_tokens=False,
    )

    assert captured_fn is not None
    if expected_chunked:
        assert captured_fn.func == chunked_fused_moe_func
        assert captured_fn.keywords["chunk_size"] == 128
    else:
        assert captured_fn.func == fused_moe_func
        assert "chunk_size" not in captured_fn.keywords
