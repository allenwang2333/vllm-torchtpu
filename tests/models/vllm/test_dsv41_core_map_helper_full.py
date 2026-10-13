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

import importlib
from types import SimpleNamespace

import jax.numpy as jnp

from vllm_torchtpu.kernels.sparse_core import core_map_helper


class _FakeRef:
    __slots__ = ("memory_space", "val")

    def __init__(self, val, memory_space=None) -> None:
        self.val = val
        self.memory_space = memory_space

    def __getitem__(self, idx):
        return self.val

    def __setitem__(self, idx, val) -> None:
        self.val = val


def test_core_map_helper_all_branches(monkeypatch) -> None:
    """Verify 100% statement and branch coverage of sparse_core.core_map_helper."""
    mod = importlib.reload(core_map_helper)

    # 1. _empty_out_ref with MemorySpace instance (-> None) and custom memory_space,
    #    plus _deref.
    monkeypatch.setattr(
        mod.pl_core,
        "_convert_out_shape_to_aval",
        lambda out_type: out_type,
    )
    monkeypatch.setattr(
        mod.lax,
        "empty",
        lambda shape, dtype, out_sharding=None: ("empty", shape, dtype, out_sharding),
    )
    monkeypatch.setattr(
        mod.jax_core,
        "new_ref",
        lambda val, memory_space=None: _FakeRef(val, memory_space=memory_space),
    )

    aval_default = SimpleNamespace(
        shape=(4, 8),
        dtype=jnp.float32,
        sharding=None,
        memory_space=mod.jax_core.MemorySpace.Device,
    )
    ref_default = mod._empty_out_ref(aval_default)
    assert ref_default.memory_space is None
    assert mod._deref(ref_default) == ("empty", (4, 8), jnp.float32, None)

    aval_custom = SimpleNamespace(
        shape=(2, 4),
        dtype=jnp.bfloat16,
        sharding="sharding0",
        memory_space="vmem",
    )
    ref_custom = mod._empty_out_ref(aval_custom)
    assert ref_custom.memory_space == "vmem"
    assert mod._deref(ref_custom) == ("empty", (2, 4), jnp.bfloat16, "sharding0")

    # 2. kernel with single output, tuple output, and list output.
    monkeypatch.setattr(mod.api, "jit", lambda fn: fn)

    def _fake_core_map(mesh, **kwargs):
        def _decorator(fn):
            fn("scratch0", sem="sem0")
            return fn

        return _decorator

    monkeypatch.setattr(mod.pl_core, "core_map", _fake_core_map)

    def _single_body(in0_ref, in1_ref, out_ref, scratch0, sem=None):
        out_ref[...] = (in0_ref[...], in1_ref[...], scratch0, sem)

    run_single = mod.kernel(_single_body, out_type=aval_default, mesh="fake_mesh")
    out_single = run_single("x0", "x1")
    assert out_single == ("x0", "x1", "scratch0", "sem0")

    def _multi_body(in_ref, out0_ref, out1_ref, scratch0, sem=None):
        out0_ref[...] = (in_ref[...], scratch0)
        out1_ref[...] = (in_ref[...], sem)

    run_tuple = mod.kernel(
        _multi_body,
        out_type=(aval_default, aval_custom),
        mesh="fake_mesh",
    )
    out_tuple = run_tuple("x_in")
    assert isinstance(out_tuple, tuple)
    assert out_tuple == (("x_in", "scratch0"), ("x_in", "sem0"))

    run_list = mod.kernel(
        _multi_body,
        out_type=[aval_default, aval_custom],
        mesh="fake_mesh",
    )
    out_list = run_list("x_in")
    assert isinstance(out_list, list)
    assert out_list == [("x_in", "scratch0"), ("x_in", "sem0")]
