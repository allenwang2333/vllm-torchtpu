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

"""Full CTracer statement & branch coverage suite for quantization/deepseek_v41."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR in sys.path:
    sys.path.remove(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    if "tpu" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(name="tpu")(lambda gm, _: gm)

    class _StubMeta(type):
        def __getattr__(cls, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)

            def _stub_fn(*args, **kwargs):
                return args[0] if args else ()

            _stub_fn.__name__ = attr
            _stub_fn.fget = _stub_fn  # type: ignore[attr-defined]
            setattr(cls, attr, _stub_fn)
            return _stub_fn

    class _StubBase(metaclass=_StubMeta):
        call_super_init: bool = False

        def __init__(self, *args, **kwargs):
            self._modules = {}

        def __getattr__(self, name: str):
            modules = self.__dict__.get("_modules")
            if modules and name in modules:
                return modules[name]
            raise AttributeError(name)

        def forward(self, hidden_states, input_ids=None, *args, **kwargs):
            return self._forward_fused_moe(hidden_states, input_ids)

    class _StubMod(types.ModuleType):
        def __iter__(self):
            return iter(())

        def __getattr__(self, name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            if name[:1].isupper():
                if name.isupper() and "_" in name:
                    val = None if name.startswith("VLLM_") else MagicMock()
                else:
                    val = _StubMeta(name, (_StubBase,), {})
            else:
                val = _StubMod(f"{self.__name__}.{name}")
            setattr(self, name, val)
            return val

        def __call__(self, *args, **kwargs):
            if self.__name__.endswith("register_quantization_config"):
                return lambda cls_or_fn: cls_or_fn
            return MagicMock()

    class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        _PREFIXES = (
            "vllm",
            "torch_tpu",
            "compressed_tensors",
            "einops",
            "requests",
        )

        def find_spec(self, fullname, path, target=None):
            if not fullname.startswith(self._PREFIXES):
                return None
            if fullname.rsplit(".", 1)[-1][:1].isupper():
                return None
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

        def create_module(self, spec):
            mod = _StubMod(spec.name)
            mod.__path__ = []
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.append(_AutoStubFinder())


from vllm_torchtpu.layers.adapter.quantization import (  # noqa: E402
    deepseek_v41 as mod,
)


def _make_moe_shell(
    *,
    sp: bool = False,
    hash_table=None,
    naive: bool = False,
    use_nn_module: bool = False,
):
    moe = object.__new__(mod.DeepseekV41MoE)
    if use_nn_module:
        gate_mod = nn.Module()
        gate_mod.tid2eid = hash_table
        experts_mod = nn.Module()
        experts_mod.do_naive_dispatch_combine = naive
        object.__setattr__(moe, "_modules", {"gate": gate_mod, "experts": experts_mod})
        moe.use_sequence_parallel = sp
    else:
        object.__setattr__(moe, "use_sequence_parallel", sp)
        object.__setattr__(moe, "gate", SimpleNamespace(tid2eid=hash_table))
        object.__setattr__(
            moe, "experts", SimpleNamespace(do_naive_dispatch_combine=naive)
        )
    return moe


def test_quant_config_registration_and_name():
    assert mod.VllmDeepseekV41Fp8Config.get_name() == "deepseek_v41_fp8"


def test_deepseek_v41_moe_init_error_and_factory_paths(monkeypatch):
    # 1. EPLB enabled raises NotImplementedError
    cfg_eplb = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=True),
        kernel_config=SimpleNamespace(moe_backend="native"),
    )
    with pytest.raises(NotImplementedError, match="does not support EPLB"):
        mod.DeepseekV41MoE(cfg_eplb)

    # 2. MegaMoE backend raises NotImplementedError
    monkeypatch.setattr(mod.v4_upstream, "MEGA_MOE_BACKENDS", {"deep_gemm_mega_moe"})
    cfg_mega = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
    )
    with pytest.raises(NotImplementedError, match="native V4 MoE backend"):
        mod.DeepseekV41MoE(cfg_mega)

    # 3. Valid initialization invokes device_hooks and wraps FusedMoEFactory
    captured_factory_opts = {}

    def fake_factory(**opts):
        captured_factory_opts.update(opts)
        return "built_experts"

    def fake_super_init(self, vllm_config, *args, **kwargs):
        object.__setattr__(self, "_modules", {})
        res = mod.v4_upstream.FusedMoEFactory(
            gate="orig_gate", apply_routed_scale_to_output=False
        )
        assert res == "built_experts"

    monkeypatch.setattr(mod.v4_upstream, "FusedMoEFactory", fake_factory)
    monkeypatch.setattr(mod.DeepseekV4MoE, "__init__", fake_super_init)
    cfg_ok = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        kernel_config=SimpleNamespace(moe_backend="native"),
    )
    inst = mod.DeepseekV41MoE(cfg_ok, prefix="model.layers.0.ffn")
    assert isinstance(inst, mod.DeepseekV41MoE)
    assert captured_factory_opts["apply_routed_scale_to_output"] is True
    assert captured_factory_opts["gate"] is None


def _make_non_sp_ep_moe(
    *,
    sp: bool = False,
    dp_size: int = 1,
    pcp_size: int = 1,
    use_ep: bool = True,
    is_monolithic: bool = True,
    supports_internal_mk: bool = False,
    custom_routing_function=None,
    gate_fn=None,
    experts_fn=None,
):
    moe = object.__new__(mod.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    if gate_fn is None:

        def _default_gate_fn(x):
            return x @ torch.ones(x.shape[-1], 8, dtype=x.dtype), None

        gate_fn = _default_gate_fn
    object.__setattr__(moe, "gate", gate_fn)
    moe_config = SimpleNamespace(
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=sp,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
    )
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            is_monolithic=is_monolithic,
            supports_internal_mk=supports_internal_mk,
        ),
        custom_routing_function=custom_routing_function,
    )
    if experts_fn is None:
        experts = SimpleNamespace(
            moe_config=moe_config,
            routed_experts=routed_experts,
            do_naive_dispatch_combine=False,
        )
    else:
        experts = experts_fn
        experts.moe_config = moe_config
        experts.routed_experts = routed_experts
    object.__setattr__(moe, "experts", experts)
    return moe


@pytest.mark.parametrize(
    "override_kwargs,ep_size,ep_rank,num_tokens,use_sim,expect_sliced",
    [
        ({}, 4, 2, 16, False, True),
        ({"dp_size": 2}, 4, 2, 16, False, True),
        ({"sp": True}, 4, 2, 16, False, False),
        ({"pcp_size": 2}, 4, 2, 16, False, False),
        ({"use_ep": False}, 4, 2, 16, False, False),
        ({"is_monolithic": False}, 4, 2, 16, False, False),
        ({"supports_internal_mk": True}, 4, 2, 16, False, False),
        ({"custom_routing_function": lambda **kw: None}, 4, 2, 16, False, False),
        ({}, 4, 2, 16, True, False),
        ({}, 1, 0, 16, False, False),
        ({}, 4, 1, 15, False, False),
    ],
)
def test_maybe_slice_gate_input_guards_and_cache(
    monkeypatch, override_kwargs, ep_size, ep_rank, num_tokens, use_sim, expect_sliced
):
    group = SimpleNamespace(world_size=ep_size, rank_in_group=ep_rank)
    monkeypatch.setattr(mod, "get_ep_group", lambda: group)
    monkeypatch.setattr(mod, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        mod.moe_routing,
        "_SIMULATION_STRATEGY",
        object() if use_sim else None,
    )
    moe = _make_non_sp_ep_moe(**override_kwargs)
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    # Call twice so both _UNRESOLVED cache miss and warm cache hit execute
    sliced = moe._maybe_slice_gate_input(hidden)
    sliced2 = moe._maybe_slice_gate_input(hidden)
    if expect_sliced:
        chunk = num_tokens // ep_size
        expected = hidden[ep_rank * chunk : (ep_rank + 1) * chunk]
        assert sliced.shape == (chunk, 4)
        assert torch.equal(sliced, expected)
        assert torch.equal(sliced2, expected)
    else:
        assert sliced is hidden
        assert sliced2 is hidden


def test_build_and_eval_gate_logits_all_branches(monkeypatch):
    monkeypatch.setattr(mod.moe_routing, "_SIMULATION_STRATEGY", None)
    monkeypatch.setattr(
        mod, "get_ep_group", lambda: SimpleNamespace(world_size=2, rank_in_group=1)
    )
    monkeypatch.setattr(mod, "_COMPILED_TPU_GATE_KERNELS", {})
    monkeypatch.setattr(mod.torch, "compile", lambda fn, **kw: fn)

    # Exercise _build_tpu_gate_kernel cold compilation + warm lookup + all inner casts
    k_slice_cast = mod._build_tpu_gate_kernel(
        (4, 4), (2, 4), torch.bfloat16, torch.float32
    )
    assert (
        mod._build_tpu_gate_kernel((4, 4), (2, 4), torch.bfloat16, torch.float32)
        is k_slice_cast
    )
    w_bf16 = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) / 4.0
    w_fp32 = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 4.0
    x_bf16 = torch.ones((4, 4), dtype=torch.bfloat16)

    assert k_slice_cast(x_bf16, w_bf16).dtype == torch.float32
    k_noslice_fp32 = mod._build_tpu_gate_kernel(
        (4, 4), None, torch.float32, torch.float32
    )
    assert k_noslice_fp32(x_bf16, w_fp32).dtype == torch.float32

    # Case 1: CPU unquantized GateLinear (both sliced and unsliced, BF16 & FP32 weights)
    moe_fast = _make_non_sp_ep_moe()
    gate_bf16 = SimpleNamespace(
        weight=w_bf16,
        bias=None,
        quant_method=object(),
        allow_dsv3_router_gemm=False,
        allow_fp32_router_gemm=False,
        allow_cublas_router_gemm=False,
        out_dtype=torch.float32,
    )
    out1 = moe_fast._eval_gate_logits(gate_bf16, x_bf16)
    out2 = moe_fast._eval_gate_logits(gate_bf16, x_bf16)
    assert out1.shape == (2, 4) and out1.dtype == torch.float32
    assert torch.equal(out1, out2)

    # Unsliced + FP32 weight path on CPU
    moe_unsliced = _make_non_sp_ep_moe(sp=True)
    gate_fp32 = SimpleNamespace(
        weight=w_fp32,
        bias=None,
        quant_method=object(),
        allow_dsv3_router_gemm=False,
        allow_fp32_router_gemm=False,
        allow_cublas_router_gemm=False,
        out_dtype=torch.float32,
    )
    out_fp32 = moe_unsliced._eval_gate_logits(gate_fp32, x_bf16)
    assert out_fp32.shape == (4, 4) and out_fp32.dtype == torch.float32

    # Case 2: Simulated TPU device tensor branch in _eval_gate_logits
    class _FakeTpuTensor:
        def __init__(self, t: torch.Tensor):
            self._t = t
            self.shape = t.shape
            self.dtype = t.dtype
            self.device = SimpleNamespace(type="tpu")

        def __getitem__(self, idx):
            return self._t[idx]

        def to(self, dt):
            return self._t.to(dt)

    fake_tpu_x = _FakeTpuTensor(x_bf16)
    out_tpu_sliced = moe_fast._eval_gate_logits(gate_bf16, fake_tpu_x)
    assert out_tpu_sliced.shape == (2, 4)
    out_tpu_unsliced = moe_unsliced._eval_gate_logits(gate_fp32, fake_tpu_x)
    assert out_tpu_unsliced.shape == (4, 4)

    # Case 3: Custom callable gate fallback (cached is None)
    moe_custom = _make_non_sp_ep_moe()
    gate_callable = lambda t: (t * 2.0, None)  # noqa: E731
    out_c1 = moe_custom._eval_gate_logits(gate_callable, x_bf16)
    out_c2 = moe_custom._eval_gate_logits(gate_callable, x_bf16)
    assert torch.equal(out_c1, out_c2)


def test_forward_fused_moe_2d_and_3d_view_handling(monkeypatch):
    ep = SimpleNamespace(world_size=4, rank_in_group=1)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    monkeypatch.setattr(mod.moe_routing, "_SIMULATION_STRATEGY", None)

    class DummyExperts2D:
        def __call__(self, *, hidden_states, router_logits, input_ids=None):
            return hidden_states.reshape(-1, hidden_states.shape[-1]) + 1.0

    moe = _make_non_sp_ep_moe(experts_fn=DummyExperts2D())
    # 2D input -> shape matches org_shape (no .view needed)
    h2d = torch.zeros(8, 4)
    out2d = moe._forward_fused_moe(h2d, input_ids=torch.arange(8))
    assert out2d.shape == (8, 4)

    # 3D input -> experts returns 2D, reshaped back to 3D org_shape
    h3d = torch.zeros(4, 2, 4)
    out3d = moe._forward_fused_moe(h3d, input_ids=torch.arange(4))
    assert out3d.shape == (4, 2, 4)


@pytest.mark.parametrize(
    "sp,hash_table,naive,gathered",
    [
        (True, torch.zeros(4, 2), True, True),
        (False, torch.zeros(4, 2), True, False),
        (True, None, True, False),
        (True, torch.zeros(4, 2), False, False),
    ],
)
def test_hash_ids_follow_sp_dispatch(monkeypatch, sp, hash_table, naive, gathered):
    ep = Mock()
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    seen = {}

    def upstream_forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
        seen["ids"] = input_ids
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", upstream_forward)
    ids = torch.tensor([1, 2])
    hidden = torch.ones(2, 3)
    moe = _make_moe_shell(sp=sp, hash_table=hash_table, naive=naive)
    assert moe.forward(hidden, ids) is hidden
    if gathered:
        ep.all_gather.assert_called_once_with(ids, dim=0)
        torch.testing.assert_close(seen["ids"], torch.tensor([1, 2, 11, 12]))
    else:
        ep.all_gather.assert_not_called()
        assert seen["ids"] is ids


def test_direct_inline_non_megamoe_forward_and_validation(monkeypatch):
    monkeypatch.setattr(mod.moe_routing, "_SIMULATION_STRATEGY", None)
    monkeypatch.setattr(
        mod, "get_ep_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )

    moe = _make_moe_shell(sp=False, hash_table=None, naive=False, use_nn_module=True)
    moe.gate.forward = lambda x: (x, None)
    moe.experts.forward = lambda *, hidden_states, router_logits, input_ids=None: (
        hidden_states + 2.0
    )

    # 1. Standard inline forward with input_ids present and input_ids=None
    x = torch.zeros(4, 4)
    out_with_ids = moe.forward(x, torch.arange(4))
    assert torch.equal(out_with_ids, torch.full((4, 4), 2.0))
    out_no_ids = moe.forward(x, None)
    assert torch.equal(out_no_ids, torch.full((4, 4), 2.0))

    # 2. Hash routing missing input_ids raises ValueError
    moe.gate.tid2eid = torch.zeros(8, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="hash MoE routing requires input_ids"):
        moe.forward(x, None)

    # 3. Vision routing missing input_ids raises ValueError
    moe.gate.tid2eid = None
    moe.gate.bias_vl = torch.zeros(8)
    with pytest.raises(ValueError, match="vision MoE routing requires input_ids"):
        moe.forward(x, None)


def test_deepseek_v41_moe_mega_gate_metadata_dual_compat(monkeypatch):
    ep = Mock()
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    moe = _make_moe_shell(sp=False, hash_table=None, naive=True)
    hidden = torch.ones(2, 3)
    ids = torch.tensor([1, 2])
    meta = {"mock": "metadata"}

    # Case 1: mega_gate_metadata is None -> calls super().forward(hidden, ids)
    seen = {}

    def forward_v030(self, hidden_states, input_ids):
        seen["args"] = (hidden_states, input_ids)
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", forward_v030)
    assert moe.forward(hidden, ids, None) is hidden
    assert seen["args"] == (hidden, ids)

    # Case 2: mega_gate_metadata is not None and upstream accepts it (v0.31)
    seen_v031 = {}

    def forward_v031(self, hidden_states, input_ids, mega_gate_metadata):
        seen_v031["args"] = (hidden_states, input_ids, mega_gate_metadata)
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", forward_v031)
    assert moe.forward(hidden, ids, meta) is hidden
    assert seen_v031["args"] == (hidden, ids, meta)

    # Case 3: mega_gate_metadata is not None but upstream raises TypeError (v0.30)
    seen_fallback = {}

    def forward_raise_typeerror(
        self, hidden_states, input_ids, mega_gate_metadata=None
    ):
        if mega_gate_metadata is not None:
            raise TypeError("forward() takes 2 positional arguments but 3 were given")
        seen_fallback["called"] = True
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", forward_raise_typeerror)
    assert moe.forward(hidden, ids, meta) is hidden
    assert seen_fallback.get("called") is True
