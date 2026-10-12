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
"""M1d end-to-end verification for `kernel.build/make_step` and `models.qwen35`."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.ops import F32
from vllm_torchtpu.megakernel.spec import TINY_TILES, ExecSpec


def _can_compile_tpu7x() -> bool:
    try:
        topologies.get_topology_desc(platform="tpu", topology_name="tpu7x", **{"1d": 8})
        return True
    except Exception:
        return False


def test_m1_recorded_full_ep_build_invariants():
    """Verify recorded `tk`, `nt`, `T_TOTAL`, `TE`, `NSS`, and VMEM budget."""
    g = dataclasses.replace(qwen35.FULL_EP, att_every=1)
    ex = ExecSpec(B=4, T=1, R=8, ctx=4096)
    _, info = qwen35.build(g, ex)
    p = info["plan"]

    assert p.tk == {
        "att_in": 512,
        "att_out": 512,
        "router": 2048,
        "sh13": 4096,
        "sh2": 128,
        "e13": 512,
        "e2": 256,
    }
    assert p.nt == {
        "att_in": 8,
        "att_out": 2,
        "router": 2,
        "sh13": 1,
        "sh2": 1,
        "e13": 8,
        "e2": 4,
    }
    assert info["T_TOTAL"] == 840
    assert p.block_params["moe"].TE == 12
    assert p.block_params["moe"].NSS == 3
    # GDN-free M1 omits `recb`/`sbuf`/`convb` -> 48,031,744 < frozen's 49,178,624
    assert info["vmem_scratch_bytes"] == 48_031_744 < 49_178_624
    assert "recb" not in info["vmem_shapes"]
    assert "sbuf" not in info["vmem_shapes"]
    assert "convb" not in info["vmem_shapes"]


@pytest.mark.parametrize(
    "R,B,T,NKV,moe_tp",
    [
        (1, 2, 1, 1, False),
        (1, 4, 2, 1, True),
        (2, 2, 1, 1, False),
        (2, 4, 2, 1, False),
    ],
)
def test_m1_qwen35_pack_and_step_vs_logical_reference(
    R: int,
    B: int,
    T: int,
    NKV: int,
    moe_tp: bool,
):
    """Verify `pack_inputs` + `make_step` matches the `reference_step` oracle."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=256,
        NH=2 * R,
        NKV=NKV,
        HD=128,
        ROT=32,
        KH=2 * R,
        VH=2 * R,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=moe_tp,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=256 << 10, split=True)
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(100 + R * 10 + T)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    out_hid_r = kernel.unfold(out_hid, R)
    for r in range(R):
        assert jnp.allclose(
            out_hid_r[r, :B].astype(F32),
            ref_hid.astype(F32),
            atol=2e-2,
            rtol=2e-2,
        )

    kc_out = kernel.unfold(out_st["kc"], R)
    vc_out = kernel.unfold(out_st["vc"], R)
    assert jnp.allclose(
        kc_out.astype(F32), ref_st["kc"].astype(F32), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(
        vc_out.astype(F32), ref_st["vc"].astype(F32), atol=1e-2, rtol=1e-2
    )


def test_m1_qwen35_r8_multi_rank_interpret():
    """Verify 2-layer, 8-rank EP + `G=4` split GQA + `HIER=1` AR in `interpret=True`."""
    devs = jax.devices("cpu")
    assert len(devs) >= 8, f"Requires 8 CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=8,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=1024,
        moe_tp=False,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=256 << 10, split=True, hier=1, mb8=2)
    ex = ExecSpec(B=2, T=1, R=8, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(808)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    out_hid_r = kernel.unfold(out_hid, 8)
    for r in range(8):
        assert jnp.allclose(
            out_hid_r[r, : ex.B].astype(F32),
            ref_hid.astype(F32),
            atol=3e-2,
            rtol=3e-2,
        )
    kc_out = kernel.unfold(out_st["kc"], 8)
    assert jnp.allclose(
        kc_out.astype(F32), ref_st["kc"].astype(F32), atol=1e-2, rtol=1e-2
    )


@pytest.mark.skipif(
    not _can_compile_tpu7x(), reason="TPU7x AOT topology compiler not available"
)
def test_m1_tpu7x_aot_compile_full_ep():
    """Verify `qwen35.make_step` lowers and AOT-compiles for `tpu7x` (`1d=8`)."""
    topo = topologies.get_topology_desc(
        platform="tpu", topology_name="tpu7x", **{"1d": 8}
    )
    mesh = Mesh(topo.devices, ("tp",))
    g = dataclasses.replace(qwen35.FULL_EP, L=2, att_every=1)
    ex = ExecSpec(B=4, T=1, R=8, ctx=2048)

    step, info = qwen35.make_step(mesh, g, ex, donate_states=True)
    wsh = info["wshape"]
    ssh = info["sshape"]

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    rep = NamedSharding(mesh, P())
    tp = NamedSharding(mesh, P("tp"))

    pos = jax.ShapeDtypeStruct((ex.B,), jnp.int32, sharding=rep)
    hid = jax.ShapeDtypeStruct((ex.R * ex.M, g.H), jnp.bfloat16, sharding=tp)
    weights = {
        k: tuple(
            jax.ShapeDtypeStruct((ex.R, *s.shape), s.dtype, sharding=tp) for s in v
        )
        if isinstance(v, tuple) and hasattr(v[0], "shape")
        else jax.ShapeDtypeStruct((ex.R, *v[0]), v[1], sharding=tp)
        for k, v in wsh.items()
    }
    states = {
        k: jax.ShapeDtypeStruct((ex.R * v[0][0], *v[0][1:]), v[1], sharding=tp)
        for k, v in ssh.items()
    }

    compiled = step.lower(pos, hid, weights, states).compile()
    hlo_text = compiled.as_text()
    assert "HloModule" in hlo_text
    assert "qwen3_5_decode_megakernel" in hlo_text


@pytest.mark.skipif(
    not os.environ.get("MK_FROZEN_QWEN35_DIR"),
    reason="Set MK_FROZEN_QWEN35_DIR to frozen qwen35 path for bit-exact check",
)
def test_m1_bit_exact_vs_frozen_qwen35():
    """Opt-in bit-exact comparison against frozen `qwen35_megakernel` (`c196d2500`)."""
    import sys

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    frozen_dir = pathlib.Path(os.environ["MK_FROZEN_QWEN35_DIR"])
    if str(frozen_dir) not in sys.path:
        sys.path.insert(0, str(frozen_dir))
    spec_f = importlib.util.spec_from_file_location(
        "_frozen_qwen35_mk", frozen_dir / "megakernel.py"
    )
    assert spec_f is not None and spec_f.loader is not None
    frozen_mk = importlib.util.module_from_spec(spec_f)
    sys.modules[spec_f.name] = frozen_mk
    spec_f.loader.exec_module(frozen_mk)

    devs = jax.devices("cpu")
    mesh = Mesh(devs[:1], ("tp",))
    tp = NamedSharding(mesh, P("tp"))

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
    )
    tiles_lib = dataclasses.replace(TINY_TILES, bank=256 << 10)
    ex = ExecSpec(B=2, T=1, R=1, ctx=256, tiles=tiles_lib, interpret=True)
    pos, hid_lib, w_lib, st_lib = qwen35.random_inputs(
        jax.random.PRNGKey(0), g, ex, mesh=mesh
    )
    step_lib, info_lib = qwen35.make_step(mesh, g, ex)
    out_hid_lib, out_st_lib = step_lib(pos, hid_lib, w_lib, st_lib)

    tiles_f = frozen_mk.Tiles(
        bank=256 << 10, nbd=10, nbe=6, ebank=256 << 10, kv_blk=256, nkv=3, vmem_mib=32
    )
    c_f = frozen_mk.MKConfig(
        R=1,
        B=2,
        T=1,
        L=2,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        ctx=256,
        stack_e=False,
    )
    f_wsh = frozen_mk.weight_shapes(c_f, tiles_f)
    f_ssh = frozen_mk.state_shapes(c_f)
    f_lay, f_S = frozen_mk.smalls_layout(c_f)
    l_lay = info_lib["plan"].smalls_layout

    sm_f = jnp.zeros((ex.R, g.L, 1, f_S), dtype=F32)
    for name, (lo, w) in l_lay.items():
        fo, _ = f_lay[name]
        sm_f = sm_f.at[:, :, 0, fo : fo + w].set(w_lib["smalls"][:, :, 0, lo : lo + w])

    w_f = {
        "lin_in": jax.device_put(
            jnp.zeros((ex.R, *f_wsh["lin_in"][0]), dtype=f_wsh["lin_in"][1]), tp
        ),
        "lin_out": jax.device_put(
            jnp.zeros((ex.R, *f_wsh["lin_out"][0]), dtype=f_wsh["lin_out"][1]), tp
        ),
        "att_in": w_lib["att_in"],
        "att_out": w_lib["att_out"],
        "router": w_lib["router"],
        "sh13": w_lib["sh13"],
        "sh2": w_lib["sh2"],
        "e13": w_lib["e13"],
        "e2": w_lib["e2"],
        "es": w_lib["es"],
        "smalls": jax.device_put(sm_f, tp),
        "final_norm": w_lib["final_norm"],
    }
    st_f = {
        "conv": jax.device_put(
            jnp.zeros(
                (ex.R * f_ssh["conv"][0][0], *f_ssh["conv"][0][1:]),
                dtype=f_ssh["conv"][1],
            ),
            tp,
        ),
        "rec": jax.device_put(
            jnp.zeros(
                (ex.R * f_ssh["rec"][0][0], *f_ssh["rec"][0][1:]), dtype=f_ssh["rec"][1]
            ),
            tp,
        ),
        "kc": st_lib["kc"],
        "vc": st_lib["vc"],
    }
    step_f, _ = frozen_mk.make_step(mesh, c_f, tiles_f, interpret=True)
    out_hid_f, out_st_f = jax.jit(step_f)(pos, hid_lib, w_f, st_f)

    assert jnp.array_equal(out_hid_lib, out_hid_f)
    assert jnp.array_equal(out_st_lib["kc"], out_st_f["kc"])
    assert jnp.array_equal(out_st_lib["vc"], out_st_f["vc"])
