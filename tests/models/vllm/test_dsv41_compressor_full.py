# Copyright 2025 Google LLC
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
"""100% statement & branch coverage unit tests for deepseek_v41_compressor."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel in (
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
    (
        "vllm_torchtpu.layers.adapter.custom_ops",
        "vllm_torchtpu/layers/adapter/custom_ops",
    ),
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41",
        "vllm_torchtpu/layers/adapter/custom_ops/deepseek_v41",
    ),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR) / _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg


class _StubModule(types.ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__") or name in {"compressor", "_internal"}:
            raise AttributeError(name)
        if name.isupper():
            return None
        if name[:1].isupper():
            return type(name, (), {"__init__": lambda self, *a, **k: None})
        return MagicMock()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in {"vllm", "torch_tpu"}:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        return None


try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    sys.meta_path.insert(0, _StubFinder())

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (  # noqa: E402
    deepseek_v41_compressor as mod,
)


class _DeviceViewTensor(torch.Tensor):
    _fake_device_type = "tpu"

    @property
    def device(self):  # type: ignore[override]
        return types.SimpleNamespace(type=self._fake_device_type)


def _as_tpu_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_DeviceViewTensor)


def _mock_compile(fn, **kw):
    def _wrapped(*args):
        clean_args = [
            a.as_subclass(torch.Tensor) if isinstance(a, _DeviceViewTensor) else a
            for a in args
        ]
        return fn(*clean_args)

    return _wrapped


def test_save_compress_norm_ratio1_ratio2_and_errors():
    mod._TPU_OP_CACHE.clear()
    with pytest.raises(ValueError, match="compression ratio must be 1 or 2"):
        mod.save_compress_norm(
            torch.randn(2, 5),
            torch.tensor([0, 1]),
            None,
            torch.tensor([0, 1]),
            None,
            None,
            torch.ones(4),
            1e-6,
            3,
            torch.empty(2, 4),
        )

    # Ratio 1: CPU float32 + bfloat16, and simulated TPU (cold + warm cache)
    kv_bf16 = torch.randn((3, 5), dtype=torch.bfloat16)
    slots_i64 = torch.tensor([0, -1, 2, 3], dtype=torch.int64)
    nw = torch.ones(4, dtype=torch.float32)
    lat_bf16 = torch.empty((3, 4), dtype=torch.bfloat16)
    mod.save_compress_norm(
        kv_bf16,
        torch.tensor([0, 1, 2]),
        None,
        slots_i64,
        None,
        None,
        nw,
        1e-6,
        1,
        lat_bf16,
    )
    assert torch.all(lat_bf16[1] == 0)

    lat_f32 = torch.empty((3, 4), dtype=torch.float32)
    mod.save_compress_norm(
        kv_bf16.float(),
        torch.tensor([0, 1, 2]),
        None,
        slots_i64.int(),
        None,
        None,
        nw,
        1e-6,
        1,
        lat_f32,
    )

    with patch.object(torch, "compile", side_effect=_mock_compile):
        lat_tpu = torch.empty((3, 4), dtype=torch.bfloat16)
        mod.save_compress_norm(
            _as_tpu_tensor(kv_bf16),
            torch.tensor([0, 1, 2]),
            None,
            slots_i64,
            None,
            None,
            nw,
            1e-6,
            1,
            lat_tpu,
        )
        # Warm cache hit
        mod.save_compress_norm(
            _as_tpu_tensor(kv_bf16),
            torch.tensor([0, 1, 2]),
            None,
            slots_i64,
            None,
            None,
            nw,
            1e-6,
            1,
            lat_tpu,
        )
        assert torch.allclose(lat_bf16, lat_tpu)

    # Ratio 2: CPU (with and without state_layout) and simulated TPU
    state_cache = torch.zeros((2, 4, 5), dtype=torch.float32)
    kv_r2 = torch.randn((4, 5), dtype=torch.bfloat16)
    pos_r2 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    slots_r2 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    qsl_r2 = torch.tensor([0, 4], dtype=torch.int64)
    req_r2 = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    lat_r2 = torch.empty((4, 4), dtype=torch.bfloat16)

    mod.save_compress_norm(
        kv_r2,
        pos_r2,
        state_cache,
        slots_r2,
        qsl_r2,
        req_r2,
        nw,
        1e-6,
        2,
        lat_r2,
        state_layout=None,
    )
    assert torch.all(lat_r2[0] == 0) and torch.all(lat_r2[2] == 0)

    from collections import namedtuple

    fake_layout = namedtuple("_Layout", ["page_size"])(4)
    with (
        patch.object(
            mod,
            "read_cache",
            side_effect=lambda c, s, layout=None: torch.zeros((s.shape[0], 5)),
        ),
        patch.object(mod, "write_cache"),
        patch.object(torch, "compile", side_effect=_mock_compile),
    ):
        lat_r2_tpu = torch.empty((4, 4), dtype=torch.float32)
        mod.save_compress_norm(
            _as_tpu_tensor(kv_r2.float()),
            pos_r2.int(),
            state_cache,
            slots_r2.int(),
            qsl_r2.int(),
            req_r2.int(),
            nw,
            1e-6,
            2,
            lat_r2_tpu,
            state_layout=fake_layout,
        )


def test_pack_write_csa_and_initialize_compressor_ops():
    vals = jnp.ones((2, 512), dtype=jnp.bfloat16)
    nope_words, rope_words = mod.pack_csa_values(vals)
    assert nope_words.shape == (2, 128) and rope_words.shape == (2, 32)

    nope_cache = jnp.zeros((4, 1, 128), dtype=jnp.int32)
    rope_cache = jnp.zeros((4, 1, 128), dtype=jnp.int32)
    records = jnp.array([0, 2], dtype=jnp.int32)
    up_nope, up_rope = mod.write_csa_values(nope_cache, rope_cache, records, vals)
    assert up_nope.shape == nope_cache.shape and up_rope.shape == rope_cache.shape

    # write_compressed_cache on CPU + simulated TPU
    fake_write_op = MagicMock(side_effect=lambda n, r, rec, v: (n + 1, r + 2))
    fake_write_op.register_fake = MagicMock()
    mod._write_op = None
    fake_pallas = types.ModuleType("torch_tpu._internal.pallas")
    fake_pallas.jax_op = MagicMock(return_value=fake_write_op)  # type: ignore[attr-defined]
    fake_tpu_int = types.ModuleType("torch_tpu._internal")
    fake_tpu_int.pallas = fake_pallas  # type: ignore[attr-defined]
    fake_ctx = types.ModuleType("vllm_torchtpu.models.vllm.vllm_model_wrapper_context")
    fake_ctx.get_vllm_model_wrapper_context = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        mesh=MagicMock()
    )
    with (
        patch.dict(
            sys.modules,
            {
                "torch_tpu._internal": fake_tpu_int,
                "torch_tpu._internal.pallas": fake_pallas,
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context": fake_ctx,
            },
        ),
        patch.object(mod.jax, "shard_map", side_effect=lambda fn, **kw: fn),
    ):
        mod.initialize_compressor_ops()
        assert mod._write_op is fake_write_op
        fake_fn = fake_write_op.register_fake.call_args[0][0]
        f_n, f_r = fake_fn(torch.zeros(2, 2), torch.zeros(2, 2))
        assert f_n.shape == (2, 2) and f_r.shape == (2, 2)
        # Second call returns early
        mod.initialize_compressor_ops()

    n_t = torch.zeros((4, 2, 128), dtype=torch.int32)
    r_t = torch.zeros((4, 2, 128), dtype=torch.int32)
    mod.write_compressed_cache(
        (n_t, r_t),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.randn(2, 512),
        torch.tensor([True, False]),
        2,
    )
    assert torch.all(n_t == 1) and torch.all(r_t == 2)

    with patch.object(torch, "compile", side_effect=_mock_compile):
        mod.write_compressed_cache(
            (n_t, r_t),
            _as_tpu_tensor(torch.tensor([0, 1], dtype=torch.int32)),
            torch.randn(2, 512),
            torch.tensor([True, False]),
            2,
        )


def test_compressor_state_cache_and_deepseek_compressor():
    csc = mod.CompressorStateCache.__new__(mod.CompressorStateCache)
    assert csc.get_attn_backend() is mod.AttentionBackend

    comp = mod.DeepseekCompressor()
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    w_raw = torch.randn((5, 8), dtype=torch.bfloat16)
    comp.fused_wkv_wgate = types.SimpleNamespace(weight=w_raw)
    hs = torch.randn((3, 8), dtype=torch.bfloat16)

    # First call (unflipped, bfloat16) + second call (warm weight cache + TPU op)
    out_cpu = comp.project(hs)
    assert out_cpu.shape == (3, 5)
    with patch.object(torch, "compile", side_effect=_mock_compile):
        out_tpu = comp.project(_as_tpu_tensor(hs.float()))
        assert torch.allclose(out_cpu, out_tpu)

    # Flipped float32 weight branch
    comp.fused_wkv_wgate = types.SimpleNamespace(
        weight=torch.randn((8, 5), dtype=torch.float32),
        **{WEIGHT_FLIPPED_ATTR: True},
    )
    out_flipped = comp.project(hs.float())
    assert out_flipped.shape == (3, 5)

    # forward() branches: non-dict metadata, state_cache is None,
    # and state_cache with native & non-native paths
    comp.head_dim = 4
    comp.norm = types.SimpleNamespace(weight=torch.ones(4, dtype=torch.float32))
    comp.rms_norm_eps = 1e-6
    comp.compress_ratio = 1
    comp.k_cache_prefix = "k_cache"
    comp._static_forward_context = {
        "k_cache": types.SimpleNamespace(cache_num_states=4)
    }

    with patch.object(
        mod,
        "get_forward_context",
        return_value=types.SimpleNamespace(attn_metadata=None),
    ):
        assert comp.forward(torch.randn(2, 5), torch.tensor([0, 1])) is None

    fake_meta = {
        "k_cache": types.SimpleNamespace(
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ),
        "state": types.SimpleNamespace(
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ),
    }
    with patch.object(
        mod,
        "get_forward_context",
        return_value=types.SimpleNamespace(attn_metadata=fake_meta),
    ):
        comp.state_cache = None
        lat1 = comp.forward(torch.randn(2, 5), torch.tensor([0, 1], dtype=torch.int32))
        assert lat1.shape == (2, 4)

        comp.state_cache = types.SimpleNamespace(
            prefix="state",
            has_native_cache=False,
            kv_cache=torch.zeros((1, 4, 5)),
        )
        lat2 = comp.forward(torch.randn(2, 5), torch.tensor([0, 1], dtype=torch.int32))
        assert lat2.shape == (2, 4)

        comp.state_cache = types.SimpleNamespace(
            prefix="state",
            has_native_cache=True,
            native=lambda: types.SimpleNamespace(
                layout=types.SimpleNamespace(page_size=4)
            ),
            kv_cache=torch.zeros((1, 4, 5)),
        )
        lat3 = comp.forward(torch.randn(2, 5), torch.tensor([0, 1], dtype=torch.int32))
        assert lat3.shape == (2, 4)
