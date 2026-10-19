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
"""Unit tests for `spec`, `profiles.default`, and `errors`."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import EXACT_FLIPS, FakeGeometry

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import spec as spec_mod
from vllm_torchtpu.megakernel.profiles.default import frozen_parity_tiles

# Every opt-in performance flag added in M5 and its frozen-parity (off) value.
_M5_FLAGS_OFF = dict(
    ecopy_opt=0,
    pipe_ep=False,
    r2_direct_store=False,
    route_vec=False,
    dense_acc_pair=False,
    kv_ering=False,
    pair_kv=False,
    dmm_unroll=0,
    gq_pack=False,
    topk_lanes=False,
    conv_stage=False,
    gdn_head_major=False,
    gqa_qv=False,
    moe_unroll=False,
    rel_fetch=False,
)


def test_tile_numerics_registry_covers_every_field():
    names = {f.name for f in dataclasses.fields(mk.Tiles)}
    assert set(spec_mod.TILE_NUMERICS) == names
    assert set(spec_mod.TILE_NUMERICS.values()) <= {"exact", "approx"}
    assert spec_mod.POOL_IO_NUMERICS == {
        "fused_pool_io": "approx",
        "pool_gather": "exact",
        "pool_gather_eligible": "approx",
    }
    # Classes the plan fixes from TPU / CPU evidence.
    for f in ("kv_sub", "route_vec", "dense_acc_pair", "moe_unroll", "gdn_head_major"):
        assert spec_mod.TILE_NUMERICS[f] == "approx", f
    assert spec_mod.TILE_NUMERICS["gq_pack"] == "approx"
    for f in ("pair_kv", "kv_ering", "gqa_qv"):
        assert spec_mod.TILE_NUMERICS[f] == "exact", f


def test_exact_flip_list_covers_registry():
    exact = {k for k, v in spec_mod.TILE_NUMERICS.items() if v == "exact"}
    # kv_scatter_vec_ns only affects the non-paged path (tested separately).
    assert exact == {f for _, f, _, _, _ in EXACT_FLIPS} | {"kv_scatter_vec_ns"}


def test_default_tile_rule_lives_in_default_profile():
    for R in (1, 2, 8):
        for B in range(1, 65):
            assert mk.default_tiles(B, R=R) == frozen_parity_tiles(B, R=R)
            assert mk.frozen_parity_tiles(B, R=R) == frozen_parity_tiles(B, R=R)


def test_tiles_field_defaults_are_frozen_parity():
    t = mk.Tiles()
    for k, v in _M5_FLAGS_OFF.items():
        assert getattr(t, k) == v, k
    assert t.kv_sub == 0
    assert mk.Tiles(hier=1, pack=1, mb8=2) == spec_mod.DEFAULT_TILES


@pytest.mark.parametrize("R", [1, 8])
@pytest.mark.parametrize("moe_tp", [False, True])
def test_default_tiles_equal_frozen_parity_and_ignore_moe_tp(R: int, moe_tp: bool):
    for B in range(1, 65):
        t = mk.default_tiles(B, moe_tp=moe_tp, R=R)
        assert t == mk.frozen_parity_tiles(B, moe_tp=moe_tp, R=R)
        assert t == mk.default_tiles(B, R=R)  # no moe_tp dependence
        assert t.kv_sub == 0 and not t.route_vec and not t.dense_acc_pair
        assert not t.pipe_ep and t.pipe  # frozen PIPE stays on (TP single tile)


def test_resolved_tiles_agree_with_resolve_tiles_for_tp_geometry():
    g_tp = FakeGeometry(moe_tp=True)
    for B in (4, 8, 16, 32, 64):
        ex = mk.ExecSpec(B=B, R=8)
        assert ex.resolved_tiles == ex.resolve_tiles(g_tp) == ex.resolve_tiles()


@pytest.mark.parametrize("R", [2, 4])
def test_default_tiles_non_v7x8_ranks_use_flat_ring(R: int):
    for B in (2, 4, 8, 16, 32, 64):
        assert mk.default_tiles(B, R=R).hier == 0


def test_exec_spec_pool_gather_default_is_split():
    assert mk.ExecSpec().pool_gather is False
    assert not mk.ExecSpec(paged_kv=True, T=4, B=24).GPR
    for val in spec_mod.POOL_GATHER_VALUES:
        assert mk.ExecSpec(pool_gather=val).pool_gather == val
    with pytest.raises(ValueError, match="pool_gather must be one of"):
        mk.ExecSpec(pool_gather="sometimes")  # type: ignore[arg-type]


def test_error_classes():
    assert issubclass(mk.MegakernelLayoutError, ValueError)
    assert not issubclass(mk.MegakernelConfigError, ValueError)
    assert issubclass(mk.MegakernelConfigError, RuntimeError)
