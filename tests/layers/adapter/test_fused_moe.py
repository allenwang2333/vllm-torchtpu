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

import vllm_torchtpu.layers.adapter.fused_moe as fused_moe_mod
from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    _resolve_activation_name,
    fused_moe_gmm,
    get_fused_moe_activation,
    load_kmajor_fp4,
    prebuild_fused_moe_kernel,
    quantize_native_fp4_kmajor,
    requant_load_kmajor_fp4,
)


class _Activation(Enum):
    SILU = "silu"


class _ValueHolder:
    def __init__(self, value: str) -> None:
        self.value = value


class _Stringifyable:
    def __str__(self) -> str:
        return "gelu"


@pytest.mark.parametrize(
    "activation",
    ["silu", _Activation.SILU, _ValueHolder("silu"), _Stringifyable()],
)
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    expected = "gelu" if isinstance(activation, _Stringifyable) else "silu"
    assert _resolve_activation_name(activation) == expected
    assert get_fused_moe_activation(activation, object()) == expected


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
    has_tpu = hasattr(torch, "tpu") and torch.tpu.is_available()
    backend = "tpu" if has_tpu else "eager"
    device = "tpu" if has_tpu else "cpu"

    @torch.compile(backend=backend, fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1, device=device), _Activation.SILU)
    assert res.cpu() == 2.0


def test_tpu_moe_activation_mixin():
    class _Method(TpuMoEActivationMixin):
        pass

    method = _Method()
    assert method._tpu_activation_str is None
    layer = SimpleNamespace(
        activation="situ",
        moe_config=SimpleNamespace(
            activation_situ_beta=2.0,
            activation_situ_linear_beta=None,
        ),
    )
    resolved = method._set_tpu_activation(layer)
    assert resolved == "situ:2.0:none"
    assert method._tpu_activation_str == "situ:2.0:none"


def test_fp4_kmajor_op_caches(monkeypatch):
    registered = []

    def fake_jax_op(name, fn):
        def _op(*args):
            registered.append((name, fn, args))
            return args[0] if len(args) == 1 else (args[0], args[1])

        return _op

    monkeypatch.setattr(fused_moe_mod.pallas, "jax_op", fake_jax_op)
    monkeypatch.setattr(fused_moe_mod, "_load_kmajor_fp4_op", None)
    monkeypatch.setattr(fused_moe_mod, "_requant_kmajor_fp4_ops", {})
    monkeypatch.setattr(fused_moe_mod, "_quantize_native_fp4_kmajor_ops", {})

    w = torch.zeros(2, 4, dtype=torch.uint8)
    s = torch.ones(2, 1, dtype=torch.float32)

    assert load_kmajor_fp4(w) is w
    assert load_kmajor_fp4(w) is w
    assert [r[0] for r in registered] == [
        "pallas::nvfp4_load_kmajor",
        "pallas::nvfp4_load_kmajor",
    ]
    assert fused_moe_mod._load_kmajor_fp4_op is not None

    out_w, out_s = requant_load_kmajor_fp4(w, s, 64)
    assert out_w is w and out_s is s
    assert requant_load_kmajor_fp4(w, s, 64) == (w, s)
    assert len(fused_moe_mod._requant_kmajor_fp4_ops) == 1

    assert quantize_native_fp4_kmajor(s, 32, pack=True) is s
    assert quantize_native_fp4_kmajor(s, 32, pack=True) is s
    assert quantize_native_fp4_kmajor(s, 32, pack=False) is s
    assert set(fused_moe_mod._quantize_native_fp4_kmajor_ops.keys()) == {
        (32, True),
        (32, False),
    }


def test_build_get_prebuild_and_fused_moe_gmm(monkeypatch):
    fake_fns = []
    called_args = []

    class _FakeCustomOp:
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn
            self.fake_fn = None

        def register_fake(self, fake_fn):
            self.fake_fn = fake_fn
            fake_fns.append(fake_fn)

        def __call__(self, *args):
            called_args.append((self.name, args))
            return self.fake_fn(*args)

    monkeypatch.setattr(
        fused_moe_mod.pallas,
        "jax_op",
        lambda name, fn: _FakeCustomOp(name, fn),
    )
    monkeypatch.setattr(fused_moe_mod, "_fused_moe_kernel_cache", {})
    monkeypatch.setattr(fused_moe_mod, "_zero_experts_start_cache", {})

    # Prebuild with explicit skip_padded_tokens=False, then cache hit
    prebuild_fused_moe_kernel(
        topk=2, activation="silu", use_ep=False, skip_padded_tokens=False
    )
    assert len(fused_moe_mod._fused_moe_kernel_cache) == 1
    prebuild_fused_moe_kernel(
        topk=2, activation="silu", use_ep=False, skip_padded_tokens=False
    )
    assert len(fused_moe_mod._fused_moe_kernel_cache) == 1

    hidden = torch.ones(4, 8, dtype=torch.bfloat16)
    w1 = torch.ones(2, 8, 16, dtype=torch.bfloat16)
    w2 = torch.ones(2, 8, 8, dtype=torch.bfloat16)
    topk_weights = torch.ones(4, 2, dtype=torch.float32)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)

    # experts_start=None (allocates and caches 0-d tensor per device)
    # and default skip_padded_tokens=None
    out1 = fused_moe_gmm(
        hidden,
        w1,
        w2,
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        None,
        topk=2,
        activation="silu",
    )
    assert out1.shape == hidden.shape and out1.dtype == hidden.dtype
    assert hidden.device in fused_moe_mod._zero_experts_start_cache

    # Second call with experts_start=None hits _zero_experts_start_cache
    out2 = fused_moe_gmm(
        hidden,
        w1,
        w2,
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        None,
        topk=2,
        activation="silu",
    )
    assert out2.shape == hidden.shape

    # Call with explicit experts_start (use_ep=True)
    ep_start = torch.tensor(4, dtype=torch.int32)
    out3 = fused_moe_gmm(
        hidden,
        w1,
        w2,
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        ep_start,
        topk=2,
        activation="silu",
    )
    assert out3.shape == hidden.shape
    assert called_args[-1][1][-1] is ep_start
