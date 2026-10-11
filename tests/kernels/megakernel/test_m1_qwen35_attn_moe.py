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
"""M1 verification tests: Skeleton + TwoRing + GQABlock + MoEBlock on Qwen3.5 (att_every=1)."""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=8 --xla_llvm_disable_expensive_passes=true",
)

import jax
from jax.experimental import topologies
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from vllm_torchtpu.kernels.megakernel._frozen.qwen35 import megakernel as frozen_q35
from vllm_torchtpu.kernels.megakernel.models import qwen35 as lib_q35
from vllm_torchtpu.kernels.megakernel import planner, spec


def _cosine_sim(a: jax.Array, b: jax.Array) -> float:
    af = a.astype(jnp.float32).ravel()
    bf = b.astype(jnp.float32).ravel()
    num = jnp.sum(af * bf)
    den = jnp.linalg.norm(af) * jnp.linalg.norm(bf) + 1e-12
    return float(num / den)


def _alloc_random_inputs(c: lib_q35.MKConfig, tiles: spec.Tiles, mesh: Mesh, seed: int = 0):
    """Allocate deterministic random weights, states, pos, and hidden for `c`."""
    key = jax.random.key(seed)
    sh_tp = NamedSharding(mesh, P("tp"))
    sh_rep = NamedSharding(mesh, P())

    wshapes = planner.weight_shapes(c, tiles)
    sshapes = planner.state_shapes(c)
    lay, _ = planner.smalls_layout(c)

    def _rand_array(shape, dtype, k_sub):
        dt = jnp.dtype(dtype)
        if jnp.issubdtype(dt, jnp.floating):
            v = jax.random.normal(k_sub, shape, dtype=jnp.float32) * 0.08
            return v.astype(dt)
        return jnp.zeros(shape, dtype=dt)

    weights = {}
    for k, v in wshapes.items():
        key, k_sub = jax.random.split(key)
        if isinstance(v, tuple) and isinstance(v[0], spec.Spec):
            items = []
            for s in v:
                key, ks = jax.random.split(key)
                arr = _rand_array((c.R,) + s.shape, s.dtype, ks)
                if c.moe_tp:
                    # Replicate router/expert weights when needed
                    pass
                items.append(jax.device_put(arr, sh_tp))
            weights[k] = tuple(items)
        else:
            shp, dt = v
            if k in ("router", "final_norm"):
                # Replicated across TP ranks
                arr0 = _rand_array((1,) + shp, dt, k_sub)
                arr = jnp.repeat(arr0, c.R, axis=0)
            elif k == "smalls":
                # Positive scales, replicated norms
                arr0 = jnp.abs(_rand_array((1,) + shp, dt, k_sub)) + 0.05
                arr = jnp.repeat(arr0, c.R, axis=0)
            elif k == "es":
                arr = jnp.abs(_rand_array((c.R,) + shp, dt, k_sub)) + 0.05
            else:
                arr = _rand_array((c.R,) + shp, dt, k_sub)
            weights[k] = jax.device_put(arr, sh_tp)

    states = {}
    for k, (shp, dt) in sshapes.items():
        key, k_sub = jax.random.split(key)
        arr = _rand_array((c.R * shp[0],) + shp[1:], dt, k_sub)
        states[k] = jax.device_put(arr, sh_tp)

    # Pos: cached length 8 for each sequence
    pos_list = []
    for s in range(c.NS):
        for t in range(c.T):
            pos_list.append(8 + t)
    pos = jax.device_put(jnp.array(pos_list, dtype=jnp.int32), sh_rep)

    key, k_hid = jax.random.split(key)
    hid0 = _rand_array((c.M, c.H), jnp.bfloat16, k_hid)
    hid0 = hid0.at[c.B :].set(0)
    hidden = jax.device_put(jnp.tile(hid0, (c.R, 1)), sh_tp)
    return pos, hidden, weights, states


def test_m1_planner_and_vmem_match_frozen():
    """Verify M1 planner and VMEM scratch allocator match frozen Qwen3.5 byte-for-byte."""
    for B, T in [(4, 1), (16, 4)]:
        c_lib = dataclasses.replace(lib_q35.FULL_EP, B=B, T=T, att_every=1)
        c_frz = dataclasses.replace(frozen_q35.FULL_EP, B=B, T=T, att_every=1)

        _, info_lib = lib_q35.build(c_lib)
        _, info_frz = frozen_q35.build(c_frz)

        assert info_lib["T_TOTAL"] == info_frz["T_TOTAL"]
        assert info_lib["NSS"] == info_frz["NSS"]
        assert info_lib["TE"] == info_frz["TE"]
        assert info_lib["s_names"] == info_frz["s_names"]
        assert info_lib["vmem_scratch_bytes"] == info_frz["vmem_scratch_bytes"]
        assert info_lib["plan"]["tk"] == info_frz["plan"]["tk"]
        assert info_lib["plan"]["nt"] == info_frz["plan"]["nt"]


@pytest.mark.parametrize("T,B", [(1, 2), (4, 4)])
def test_m1_cpu_interpret_bit_exact_vs_frozen_and_reference(T: int, B: int):
    """Verify bit-exact equivalence vs frozen Qwen3.5 and high accuracy vs pure-JAX oracle."""
    devs = jax.devices("cpu")[:1]
    mesh = Mesh(np.array(devs), ("tp",))

    c_lib = lib_q35.MKConfig(
        R=1,
        B=B,
        T=T,
        L=2,
        att_every=1,
        H=512,
        NH=2,
        NKV=1,
        HD=128,
        ROT=64,
        KH=1,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=False,
        stack_e=True,
        ctx=256,
    )
    c_frz = frozen_q35.MKConfig(
        R=1,
        B=B,
        T=T,
        L=2,
        att_every=1,
        H=512,
        NH=2,
        NKV=1,
        HD=128,
        ROT=64,
        KH=1,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        moe_tp=False,
        stack_e=True,
        ctx=256,
    )
    t_lib = dataclasses.replace(spec.TINY_TILES, hier=0, pack=0, mb8=0, kv_blk=128, nkv=2, nbd=6, nbe=4)
    t_frz = dataclasses.replace(frozen_q35.TINY_TILES, hier=0, pack=0, mb8=0, kv_blk=128, nkv=2, nbd=6, nbe=4)

    pos, hidden, weights, states = _alloc_random_inputs(c_lib, t_lib, mesh, seed=42 + T)

    step_lib, info_lib = lib_q35.make_step(mesh, c_lib, t_lib, interpret=True)
    step_frz, info_frz = frozen_q35.make_step(mesh, c_frz, t_frz, interpret=True)

    assert info_lib["vmem_scratch_bytes"] == info_frz["vmem_scratch_bytes"]

    hid_lib, st_lib = step_lib(pos, hidden, weights, states)
    hid_frz, st_frz = step_frz(pos, hidden, weights, states)

    # 1. Bit-exact match against frozen Qwen3.5 megakernel
    assert jnp.array_equal(hid_lib, hid_frz), (
        f"max diff vs frozen: {jnp.max(jnp.abs(hid_lib.astype(jnp.float32) - hid_frz.astype(jnp.float32)))}"
    )
    assert jnp.array_equal(st_lib["kc"], st_frz["kc"])
    assert jnp.array_equal(st_lib["vc"], st_frz["vc"])

    # 2. Accuracy match against pure-JAX reference oracle
    hid_ref, st_ref = lib_q35.reference_step(c_lib, t_lib, pos, hidden, weights, states)
    cos = _cosine_sim(hid_lib[:B], hid_ref[:B])
    max_diff = float(
        jnp.max(jnp.abs(hid_lib[:B].astype(jnp.float32) - hid_ref[:B].astype(jnp.float32)))
    )
    assert cos > 0.9999, f"Expected cosine > 0.9999 vs reference_step, got {cos}"
    assert max_diff <= 2e-2, f"Expected max_diff <= 2e-2 (1 BF16 ULP) vs reference_step, got {max_diff}"
    kc_diff = float(jnp.max(jnp.abs(st_lib["kc"].astype(jnp.float32) - st_ref["kc"].astype(jnp.float32))))
    vc_diff = float(jnp.max(jnp.abs(st_lib["vc"].astype(jnp.float32) - st_ref["vc"].astype(jnp.float32))))
    assert kc_diff <= 2e-2, f"Expected kc_diff <= 2e-2 vs reference_step, got {kc_diff}"
    assert vc_diff <= 2e-2, f"Expected vc_diff <= 2e-2 vs reference_step, got {vc_diff}"


def test_m1_tpu7x_aot_compilation_t1_and_t4():
    """Verify TPU7x 8-rank AOT compilation of 60-layer Qwen3.5 (att_every=1) for T=1 and T=4."""
    topo = topologies.get_topology_desc(topology_name="tpu7x:2x2x1", platform="tpu")
    mesh = Mesh(np.array(topo.devices)[:8], ("tp",))
    sh_tp = NamedSharding(mesh, P("tp"))
    sh_rep = NamedSharding(mesh, P())

    for B, T in [(4, 1), (16, 4)]:
        c_lib = dataclasses.replace(lib_q35.FULL_EP, B=B, T=T, att_every=1)
        c_frz = dataclasses.replace(frozen_q35.FULL_EP, B=B, T=T, att_every=1)
        t_lib = spec.default_tiles(B)
        t_frz = frozen_q35.default_tiles(c_frz)

        step_lib, info_lib = lib_q35.make_step(mesh, c_lib, t_lib, interpret=False)
        _, info_frz = frozen_q35.make_step(mesh, c_frz, t_frz, interpret=False)

        assert info_lib["vmem_scratch_bytes"] == info_frz["vmem_scratch_bytes"]
        assert info_lib["vmem_scratch_bytes"] < t_lib.vmem_mib * 1024 * 1024

        wshapes = planner.weight_shapes(c_lib, t_lib)
        sshapes = planner.state_shapes(c_lib)

        pos_sds = jax.ShapeDtypeStruct((c_lib.B,), jnp.int32, sharding=sh_rep)
        hid_sds = jax.ShapeDtypeStruct((c_lib.R * c_lib.M, c_lib.H), jnp.bfloat16, sharding=sh_tp)
        w_sds = {}
        for k, v in wshapes.items():
            if isinstance(v, tuple) and isinstance(v[0], spec.Spec):
                w_sds[k] = tuple(
                    jax.ShapeDtypeStruct((c_lib.R,) + s.shape, s.dtype, sharding=sh_tp)
                    for s in v
                )
            else:
                shp, dt = v
                w_sds[k] = jax.ShapeDtypeStruct((c_lib.R,) + shp, dt, sharding=sh_tp)
        s_sds = {
            k: jax.ShapeDtypeStruct((c_lib.R * shp[0],) + shp[1:], dt, sharding=sh_tp)
            for k, (shp, dt) in sshapes.items()
        }

        compiled = jax.jit(step_lib).lower(pos_sds, hid_sds, w_sds, s_sds).compile()
        hlo = compiled.as_text()
        assert "qwen3_5_decode_megakernel" in hlo
        assert "tpu_custom_call" in hlo
