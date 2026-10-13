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

"""Full unit & parity test suite for deepseek_v4_mhc_op.py."""

import functools
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
_GATES = (1e-6, 1e-6, 1e-6, 2.0, 20)


class _FakeOp:
    def __init__(self, name, fn, **kwargs):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self.fake = None

    def register_fake(self, fake):
        self.fake = fake


@pytest.fixture
def mhc_env(monkeypatch: pytest.MonkeyPatch):
    if "torch_tpu" not in sys.modules:
        tt_mod = types.ModuleType("torch_tpu")
        tt_int = types.ModuleType("torch_tpu._internal")
        pallas_mod = types.ModuleType("torch_tpu._internal.pallas")
        pallas_mod.jax_op = lambda name, fn, **kwargs: _FakeOp(name, fn, **kwargs)
        tt_int.pallas = pallas_mod
        tt_mod._internal = tt_int
        monkeypatch.setitem(sys.modules, "torch_tpu", tt_mod)
        monkeypatch.setitem(sys.modules, "torch_tpu._internal", tt_int)
        monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas_mod)

    for pkg_name, rel_dir in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.adapter", "layers/adapter"),
        ("vllm_torchtpu.layers.adapter.custom_ops", "layers/adapter/custom_ops"),
        (
            "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4",
            "layers/adapter/custom_ops/deepseek_v4",
        ),
        ("vllm_torchtpu.models", "models"),
        ("vllm_torchtpu.models.vllm", "models/vllm"),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(_SRC_ROOT / rel_dir)]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)

    ctx_mod_name = "vllm_torchtpu.models.vllm.vllm_model_wrapper_context"
    if ctx_mod_name not in sys.modules:
        ctx_mod = types.ModuleType(ctx_mod_name)
        ctx_mod.get_vllm_model_wrapper_context = lambda: SimpleNamespace(mesh=None)
        monkeypatch.setitem(sys.modules, ctx_mod_name, ctx_mod)

    mod_name = "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op"
    sys.modules.pop(mod_name, None)
    mhc = importlib.import_module(mod_name)

    built = []

    def jax_op(name, fn, **kwargs):
        op = _FakeOp(name, fn, **kwargs)
        built.append(op)
        return op

    monkeypatch.setattr(mhc, "_mhc_op_cache", {})
    monkeypatch.setattr(mhc.pallas, "jax_op", jax_op)
    mesh = object()
    monkeypatch.setattr(
        mhc, "get_vllm_model_wrapper_context", lambda: SimpleNamespace(mesh=mesh)
    )
    return mhc, built, mesh


class TestDeepSeekV4MhcOpFull:
    """Achieves 100% statement and branch coverage on deepseek_v4_mhc_op.py."""

    def test_name_float_and_gate_suffix(self, mhc_env) -> None:
        mhc, _, _ = mhc_env
        assert mhc._name_float(1e-6) == "1em06"
        assert mhc._name_float(2.0) == "2"
        assert mhc._name_float(0.5) == "0p5"
        assert mhc._name_float(1e20) == "1e20"
        assert mhc._gate_suffix(*_GATES) == "_r1em06_p1em06_s1em06_a2_i20"

    def test_jax_wrappers_forward_arguments(self, monkeypatch, mhc_env) -> None:
        mhc, _, _ = mhc_env
        rec = {}
        monkeypatch.setattr(
            mhc.pre_kernel, "mhc_pre", lambda *a: rec.setdefault("pre", a) or "pre"
        )
        monkeypatch.setattr(
            mhc.post_kernel, "mhc_post", lambda *a: rec.setdefault("post", a) or "post"
        )
        monkeypatch.setattr(
            mhc.fused_post_pre_kernel,
            "mhc_fused_post_pre",
            lambda *a: rec.setdefault("fused", a) or "fused",
        )
        kw = dict(
            rms_eps=1.0,
            hc_pre_eps=2.0,
            hc_sinkhorn_eps=3.0,
            hc_post_mult_value=4.0,
            sinkhorn_repeat=5,
        )
        mhc._mhc_pre_jax("r", "f", "s", "b", **kw)
        mhc._mhc_post_jax("x", "r", "p", "c")
        mhc._mhc_fused_post_pre_jax("x", "r", "p", "c", "f", "s", "b", **kw)
        assert rec["pre"] == ("r", "f", "s", "b", 1.0, 2.0, 3.0, 4.0, 5)
        assert rec["post"] == ("x", "r", "p", "c")
        assert rec["fused"] == (
            "x",
            "r",
            "p",
            "c",
            "f",
            "s",
            "b",
            1.0,
            2.0,
            3.0,
            4.0,
            5,
        )

    def test_fake_shapes_3d_and_4d(self, mhc_env) -> None:
        mhc, _, _ = mhc_env
        # 3D residual path (len(shape) == 3)
        res_3d = torch.zeros(16, 4, 8, dtype=torch.bfloat16)
        pm3, cm3, li3 = mhc._fake_mhc_pre(res_3d, None, None, None)
        assert pm3.shape == (16, 4, 1) and pm3.dtype == torch.float32
        assert cm3.shape == (16, 4, 4) and cm3.dtype == torch.float32
        assert li3.shape == (16, 8) and li3.dtype == torch.bfloat16
        fused3 = mhc._fake_mhc_fused_post_pre(
            None, res_3d, None, None, None, None, None
        )
        assert [t.shape for t in fused3] == [
            (16, 4, 8),
            (16, 4, 1),
            (16, 4, 4),
            (16, 8),
        ]

        # 4D residual path (len(shape) != 3)
        res_4d = torch.zeros(3, 2, 4, 8, dtype=torch.bfloat16)
        pm4, cm4, li4 = mhc._fake_mhc_pre(res_4d, None, None, None)
        assert pm4.shape == (3, 2, 4, 1) and pm4.dtype == torch.float32
        assert cm4.shape == (3, 2, 4, 4) and cm4.dtype == torch.float32
        assert li4.shape == (3, 2, 8) and li4.dtype == torch.bfloat16
        post4 = mhc._fake_mhc_post(None, res_4d, None, None)
        assert post4.shape == res_4d.shape and post4.dtype == res_4d.dtype
        fused4 = mhc._fake_mhc_fused_post_pre(
            None, res_4d, None, None, None, None, None
        )
        assert [t.shape for t in fused4] == [
            (3, 2, 4, 8),
            (3, 2, 4, 1),
            (3, 2, 4, 4),
            (3, 2, 8),
        ]

    def test_get_mhc_ops_and_builders(self, monkeypatch, mhc_env) -> None:
        mhc, built, mesh = mhc_env
        suffix_calls = []
        real_suffix = mhc._gate_suffix

        def counting_suffix(*args):
            suffix_calls.append(args)
            return real_suffix(*args)

        monkeypatch.setattr(mhc, "_gate_suffix", counting_suffix)
        ops = mhc.get_mhc_ops(*_GATES)
        assert suffix_calls == [_GATES]
        suffix = real_suffix(*_GATES)
        assert ops.pre.name == "pallas::deepseek_v4_mhc_pre_v1" + suffix
        assert ops.fused.name == "pallas::deepseek_v4_mhc_fused_post_pre_v1" + suffix
        assert ops.post.name == "pallas::deepseek_v4_mhc_post_v1"
        assert ops.pre.fake is mhc._fake_mhc_pre
        assert ops.fused.fake is mhc._fake_mhc_fused_post_pre
        assert ops.post.fake is mhc._fake_mhc_post
        for op in built:
            assert op.kwargs["mesh"] is mesh
        assert ops.pre.kwargs["input_partition_specs"] is mhc._PRE_PARTITION_SPECS
        assert ops.fused.kwargs["input_partition_specs"] is mhc._FUSED_PARTITION_SPECS
        assert ops.post.kwargs["input_partition_specs"] is mhc._POST_PARTITION_SPECS
        for op, target in (
            (ops.pre, mhc._mhc_pre_jax),
            (ops.fused, mhc._mhc_fused_post_pre_jax),
        ):
            assert isinstance(op.fn, functools.partial) and op.fn.func is target

        # Warm get_mhc_ops hit returns identical MHCOps without re-formatting
        again = mhc.get_mhc_ops(*_GATES)
        assert again is ops
        assert len(suffix_calls) == 1
        assert len(built) == 3

        # Direct builder calls hit per-op cache and cover suffix=None branches
        assert mhc._mhc_pre_op(*_GATES) is ops.pre
        assert mhc._mhc_fused_post_pre_op(*_GATES) is ops.fused
        assert mhc.get_mhc_post_op() is ops.post

        # Distinct gate constants build new pre/fused ops while sharing post
        other = mhc.get_mhc_ops(1e-5, 1e-6, 1e-6, 2.0, 20)
        assert other.pre is not ops.pre and other.post is ops.post
        assert len(built) == 5
