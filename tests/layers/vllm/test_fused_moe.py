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

import pytest
import torch

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


def test_prebuild_and_forward_share_one_cache_entry(monkeypatch):
    """A prebuilt op must be the one the forward finds.

    The cache key is spelled at every call site, so a parameter added to the
    forward and not to the prebuild silently misses -- correct results, but the
    op gets built inside the traced forward, which is what prebuilding exists to
    avoid.
    """
    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    kwargs = dict(topk=4,
                  activation="silu",
                  use_ep=True,
                  skip_padded_tokens=True,
                  ep_size=8)
    fused_moe.prebuild_fused_moe_kernel(**kwargs)
    assert len(fused_moe._fused_moe_kernel_cache) == 1
    fused_moe._get_fused_moe_custom_op(**kwargs)
    assert len(fused_moe._fused_moe_kernel_cache) == 1
