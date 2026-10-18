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
"""M1a unit tests for `spec`, `ops`, `pool_alias`, `planner`, and `vmem`."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest
from conftest import (
    FakeAltMixerBlock,
    FakeGeometry,
    FakeMixerBlock,
    make_fake_model_spec,
)
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import planner, pool_alias, vmem
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    FP8,
    bswitch,
    rms_zc,
    rmsnorm,
    rnd,
    rope_partial,
    rup,
    silu_mul,
)
from vllm_torchtpu.megakernel.spec import (
    TINY_TILES,
    Block,
    ExecSpec,
    LayerType,
    ModelSpec,
    ScratchReq,
    SmallField,
    SwitchSlot,
    WeightDecl,
)


def test_ops_primitives():
    """Verify `rup`, `rnd`, `rmsnorm`, `rope_partial`, `silu_mul`, and `bswitch`."""
    assert rup(1, 128) == 128
    assert rup(128, 128) == 128
    assert rup(129, 128) == 256

    x = jnp.ones((4, 128), dtype=F32)
    w = jnp.zeros((1, 128), dtype=F32)
    # Zero-centered RMSNorm: w=0 gives unit scale (1.0 + w = 1.0)
    y = rms_zc(x, w, 1e-6)
    assert jnp.allclose(y, jnp.ones_like(x), atol=1e-3)

    # Standard RMSNorm: w=0.5 gives 0.5 scale, while zero-centered gives 1.5
    w_half = jnp.full((1, 128), 0.5, dtype=F32)
    y_zc = rmsnorm(x, w_half, 1e-6, zero_centered=True)
    y_std = rmsnorm(x, w_half, 1e-6, zero_centered=False)
    assert jnp.allclose(y_zc, jnp.full_like(x, 1.5), atol=1e-2)
    assert jnp.allclose(y_std, jnp.full_like(x, 0.5), atol=1e-2)

    # Partial RoPE leaves trailing columns untouched
    cos = jnp.ones((1, 16), dtype=F32)
    sin = jnp.zeros((1, 16), dtype=F32)
    rx = rope_partial(x, cos, sin, 32)
    assert jnp.allclose(rx, rnd(x))

    # SwiGLU and bswitch
    sm = silu_mul(x, x)
    assert sm.dtype == BF16
    val = bswitch(jnp.int32(2), [lambda i=i: jnp.int32(i * 10) for i in range(4)])
    assert int(val) == 20


def test_pool_alias_storage_size():
    """Verify TPU 8x128 tile storage calculation."""
    assert pool_alias.storage_size((128, 256), jnp.dtype(FP8)) == 128 * 256 * 1
    assert pool_alias.storage_size((128, 256), jnp.dtype(BF16)) == 128 * 256 * 2
    assert pool_alias.storage_size((1, 256), jnp.dtype(F32)) == 8 * 256 * 4


def test_planner_is_load_bearing_single_branch(fake_ex: ExecSpec):
    """Verify `planner.plan` derives only weights/smalls/states from active blocks."""
    ms = make_fake_model_spec(L=4, att_every=1, H=256)
    p = planner.plan(ms, fake_ex)

    # Only FakeMixerBlock and FakeFFNBlock are in the spec -> no GDN declarations
    assert p.static_kinds == ("att_in", "att_out", "router", "sh13", "sh2")
    assert p.dynamic_kinds == ("e13", "e2")
    assert "lin_in" not in p.kinds
    assert "conv" not in p.smalls_layout
    assert "rec" not in p.s_names
    assert p.s_names == ("kc", "vc")
    assert p.o_names == ("hidden", "knew", "vnew")

    wsh = planner.weight_shapes(ms, fake_ex, p)
    ssh = planner.state_shapes(ms, fake_ex, p)
    osh = planner.kernel_out_shapes(ms, fake_ex, p)

    assert set(wsh.keys()) == set(p.w_names)
    assert set(ssh.keys()) == {"kc", "vc"}
    assert set(osh.keys()) == {"hidden", "knew", "vnew"}
    # `es` has leading `(L, EW, 1, 2*IR + H)` derived from WeightDecl.lead_dims=(EW,)
    assert wsh["es"][0] == (4, 64, 1, 2 * 256 + 256)


def test_planner_hybrid_switch_slot(fake_ex: ExecSpec):
    """Verify multi-branch `SwitchSlot` derives per-branch counts and schedules."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    p = planner.plan(ms, fake_ex)

    assert p.period == 4
    assert p.branch_counts == (3, 1)
    assert p.block_layer_counts["fake_gdn"] == 3
    assert p.block_layer_counts["fake_gqa"] == 1
    assert p.block_layer_counts["fake_moe"] == 4

    wsh = planner.weight_shapes(ms, fake_ex, p)
    assert wsh["lin_in"][0][0] == 3
    assert wsh["att_in"][0][0] == 1
    assert wsh["router"][0][0] == 4

    # Shared smalls field `in_s` takes max width across branches (NAI=512, NLI=640)
    _, in_s_w = p.smalls_layout["in_s"]
    assert in_s_w == 640

    # >2 branches in a SwitchSlot raises AssertionError at plan time
    class ThirdBranchBlock(Block):
        name = "third"

    bad_sw = SwitchSlot(
        "mixer",
        branches=(FakeAltMixerBlock(), FakeMixerBlock(), ThirdBranchBlock()),
        branch_counts=(1, 1, 2),
    )
    bad_ms = ModelSpec(
        name="bad_3branch",
        config=FakeGeometry(L=4, H=256),
        layer_type=LayerType("l", slots=(bad_sw,)),
    )
    with pytest.raises(AssertionError, match="supports at most 2 active branches"):
        planner.plan(bad_ms, fake_ex)


def test_planner_custom_block_extension(fake_ex: ExecSpec):
    """Verify adding a custom `Block` updates `plan` without editing `planner.py`."""

    class CustomMLPBlock(Block):
        name = "custom_mlp"
        reduction = "allreduce"

        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [
                WeightDecl("mlp_up", K=g.H, N=512, dtype=FP8, stream_ring="static"),
                WeightDecl("mlp_down", K=512, N=g.H, dtype=FP8, stream_ring="static"),
            ]

        def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
            return [SmallField("mlp_scale", 512)]

        def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
            return {"acc": 512, "xb2": 512}

        def scratch(self, g: FakeGeometry, ex: ExecSpec, plan) -> dict[str, object]:
            return {"mlp_tmp": pltpu.VMEM((ex.MR, 512), F32)}

    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        name="custom_model",
        config=g,
        layer_type=LayerType(
            "custom_layer", slots=(FakeMixerBlock(), CustomMLPBlock())
        ),
    )
    p = planner.plan(ms, fake_ex)
    assert p.static_kinds == ("att_in", "att_out", "mlp_up", "mlp_down")
    assert p.dynamic_kinds == ()
    assert "mlp_scale" in p.smalls_layout

    cp, scr = vmem.build_scratch(ms, fake_ex, p)
    assert "mlp_tmp" in scr
    assert "epool" not in scr and "esem" not in scr
    assert cp.NACC >= 512 and cp.KX2 >= 512


def test_vmem_phase_overlay_and_conflict(fake_ex: ExecSpec):
    """Verify `ScratchReq.phase` overlays non-persistent buffers and checks budget."""
    ms_hybrid = make_fake_model_spec(L=4, att_every=4, H=256)
    cp, scr = vmem.build_scratch(ms_hybrid, fake_ex)

    # `branch_tmp` is declared by both FakeAltMixerBlock and FakeMixerBlock -> overlaid!
    assert "branch_tmp" in scr
    assert scr["branch_tmp"].shape == (fake_ex.MR * 2, 128)
    assert cp.phase_map["branch_tmp"] == "branch:gdn|branch:gqa"

    # Overlaying two SMEM phase buffers preserves SMEM memory space
    smem_scratch: dict[str, object] = {}
    smem_phases: dict[str, str] = {}
    vmem._merge_scratch(
        smem_scratch,
        smem_phases,
        [
            ScratchReq("smem_tmp", pltpu.SMEM((4,), jnp.int32), phase="branch:a"),
            ScratchReq("smem_tmp", pltpu.SMEM((8,), jnp.int32), phase="branch:b"),
        ],
    )
    assert smem_scratch["smem_tmp"].shape == (8,)
    assert smem_scratch["smem_tmp"].memory_space == pltpu.SMEM

    # Persistent name collision must raise ValueError
    class CollidingBlock(Block):
        name = "colliding"

        def scratch(self, g, ex, plan):
            return [ScratchReq("kbuf", pltpu.VMEM((16, 128), F32), phase="persistent")]

    bad_ms = ModelSpec(
        name="bad",
        config=FakeGeometry(L=2, H=256),
        layer_type=LayerType("bad_layer", slots=(FakeMixerBlock(), CollidingBlock())),
    )
    with pytest.raises(ValueError, match="Conflicting scratch buffer 'kbuf'"):
        vmem.build_scratch(bad_ms, fake_ex)

    # Exceeding tiles.vmem_mib raises MegakernelConfigError with per-buffer breakdown
    tiny_budget_ex = dataclasses.replace(
        fake_ex, tiles=dataclasses.replace(TINY_TILES, vmem_mib=1)
    )
    with pytest.raises(MegakernelConfigError, match="exceeds tiles.vmem_mib budget"):
        vmem.build_scratch(ms_hybrid, tiny_budget_ex)
