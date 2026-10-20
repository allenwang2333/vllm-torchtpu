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
"""Unit tests for `spec`, `profiles.default` and the Qwen3.5 baseline tiles."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from conftest import EXACT_FLIPS, FakeGeometry, tile_field

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel import spec as spec_mod
from vllm_torchtpu.megakernel.blocks import GDNBlock, GQABlock, MoEBlock
from vllm_torchtpu.megakernel.models.qwen35_profile import (
    DECODE_TILES,
    baseline_tiles,
)
from vllm_torchtpu.megakernel.profiles.default import (
    DEFAULT_CHIP,
    ChipBudget,
    default_tiles,
)

# Opt-in performance flags (core `field` / `<block>.<field>`), default off.
_PERF_FLAGS_OFF = {
    "moe.ecopy_opt": 0,
    "moe.pipe_ep": False,
    "moe.r2_direct_store": False,
    "moe.route_vec": False,
    "dense_acc_pair": False,
    "gqa.kv_ering": False,
    "gqa.pair_kv": False,
    "dmm_unroll": 0,
    "gqa.gq_pack": False,
    "moe.topk_lanes": False,
    "gdn.conv_stage": False,
    "gdn.gdn_head_major": False,
    "gqa.gqa_qv": False,
    "moe.moe_unroll": False,
    "rel_fetch": False,
}
_LIBRARY_BLOCKS = (GQABlock, GDNBlock, MoEBlock)


def _library_numerics() -> dict[str, str]:
    """`spec.tile_numerics()` restricted to core and the library's blocks."""
    names = {b.name for b in _LIBRARY_BLOCKS}
    return {
        k: v
        for k, v in spec_mod.tile_numerics().items()
        if "." not in k or k.split(".", 1)[0] in names
    }


def test_core_tile_numerics_cover_every_core_field():
    names = {f.name for f in dataclasses.fields(mk.Tiles)} - {"blocks"}
    assert set(spec_mod.TILE_NUMERICS) == names
    assert set(spec_mod.TILE_NUMERICS.values()) <= {"exact", "approx"}
    assert pool.POOL_IO_NUMERICS == {
        "fused_pool_io": "approx",
        "pool_gather": "exact",
        "pool_gather_eligible": "approx",
    }


@pytest.mark.parametrize("block", _LIBRARY_BLOCKS, ids=lambda b: b.name)
def test_block_tunables_declare_numerics_for_every_field(block: type[mk.Block]):
    cls = block.Tunables
    assert cls is not None and spec_mod.tunables_class(block.name) is cls
    numerics = dict(cls.NUMERICS)
    fields = {f.name for f in dataclasses.fields(cls)}
    assert set(numerics) == fields
    assert set(numerics.values()) <= {"exact", "approx"}
    assert {f"{block.name}.{f}" for f in fields} <= set(spec_mod.tile_numerics())


def test_tile_numerics_classes_from_evidence():
    numerics = _library_numerics()
    for k in (
        "gqa.kv_sub",
        "moe.route_vec",
        "dense_acc_pair",
        "moe.moe_unroll",
        "gdn.gdn_head_major",
        "gqa.gq_pack",
    ):
        assert numerics[k] == "approx", k
    for k in ("gqa.pair_kv", "gqa.kv_ering", "gqa.gqa_qv"):
        assert numerics[k] == "exact", k


def test_tunables_registration_rejects_incomplete_or_conflicting_classes():
    @dataclasses.dataclass(frozen=True)
    class Partial:
        a: int = 0
        b: int = 1
        NUMERICS = {"a": "exact"}

    with pytest.raises(TypeError, match="missing"):
        spec_mod.register_tunables("test_partial", Partial)
    with pytest.raises(ValueError, match="already registered"):
        spec_mod.register_tunables(GQABlock.name, GDNBlock.Tunables)  # type: ignore[arg-type]


def test_tiles_block_tunables_are_canonical_and_hashable():
    gqa_cls = GQABlock.Tunables
    assert gqa_cls is not None
    t = mk.Tiles()
    assert t.block("gqa") == gqa_cls()
    same = mk.Tiles(blocks={"gqa": gqa_cls()})
    assert same == t and hash(same) == hash(t) and not same.blocks
    t2 = t.with_blocks({"gqa": {"kv_blk": 256}})
    assert t2.block("gqa").kv_blk == 256 and t2 != t
    assert hash(t2) == hash(t.with_blocks({"gqa": {"kv_blk": 256}}))
    with pytest.raises(TypeError):
        t.with_blocks({"gqa": {"no_such_field": 1}})
    with pytest.raises(KeyError):
        t.with_blocks({"no_such_block": {"x": 1}})


def test_exact_flip_list_covers_registry():
    exact = {k for k, v in _library_numerics().items() if v == "exact"}
    # kv_scatter_vec_ns only affects the non-paged path (tested separately).
    assert exact == {f for _, f, _, _, _ in EXACT_FLIPS} | {"gqa.kv_scatter_vec_ns"}


# Block and cache names core `spec.py` must not mention (in any case).
_NON_CORE_NAMES = ("gqa", "gdn", "moe", "pool", "fused", "paged", "ckpt")


def test_core_spec_mentions_no_block_or_cache_names():
    """`spec.py` holds core knobs only: no block or pool name anywhere in it."""
    text = Path(spec_mod.__file__).read_text(encoding="utf-8")
    for name in _NON_CORE_NAMES:
        hits = [
            f"{i}: {line.strip()}"
            for i, line in enumerate(text.splitlines(), 1)
            if re.search(name, line, flags=re.IGNORECASE)
        ]
        assert not hits, (name, hits)


def test_default_tile_rule_is_model_agnostic():
    """One tile set for every bucket, rank count and MoE mode; chip budgets."""
    ref = default_tiles(1)
    assert ref.hier == 0  # flat-ring all-reduce, no topology-specific mode
    assert (ref.vmem_mib, ref.smem_mib) == (
        DEFAULT_CHIP.vmem_mib,
        DEFAULT_CHIP.smem_mib,
    )
    for R in (1, 2, 4, 8, 16):
        for moe_tp in (False, True):
            for B in range(1, 65):
                assert mk.default_tiles(B, moe_tp=moe_tp, R=R) == ref
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    assert default_tiles(8, chip=small).vmem_mib == 16


def test_tiles_field_defaults_turn_perf_flags_off():
    t = mk.Tiles()
    for k, v in _PERF_FLAGS_OFF.items():
        assert tile_field(t, k) == v, k
    assert t.block("gqa").kv_sub == 0
    d = mk.default_tiles(8)
    for k, v in _PERF_FLAGS_OFF.items():
        assert tile_field(d, k) == v, k


@pytest.mark.parametrize("R", [1, 8])
@pytest.mark.parametrize("moe_tp", [False, True])
def test_qwen35_baseline_tiles_ignore_moe_tp(R: int, moe_tp: bool):
    for B in range(1, 65):
        t = baseline_tiles(B, moe_tp=moe_tp, R=R)
        assert t == baseline_tiles(B, R=R)  # no moe_tp dependence
        gqa, moe = t.block("gqa"), t.block("moe")
        assert gqa.kv_sub == 0 and not moe.route_vec and not t.dense_acc_pair
        assert not moe.pipe_ep and moe.pipe  # PIPE stays on (TP single tile)
    assert mk.Tiles(hier=1, mb8=2).with_blocks({"gqa": {"pack": 1}}) == DECODE_TILES


def test_resolved_tiles_agree_with_resolve_tiles_for_tp_geometry():
    g_tp = FakeGeometry(moe_tp=True)
    for B in (4, 8, 16, 32, 64):
        ex = mk.ExecSpec(B=B, R=8)
        assert ex.resolved_tiles == ex.resolve_tiles(g_tp) == ex.resolve_tiles()


@pytest.mark.parametrize("R", [2, 4])
def test_qwen35_baseline_non_v7x8_ranks_use_flat_ring(R: int):
    for B in (2, 4, 8, 16, 32, 64):
        assert baseline_tiles(B, R=R).hier == 0


def test_cache_exec_pool_gather_default_is_split():
    assert mk.ExecSpec().cache is None
    assert mk.cache_exec(mk.ExecSpec()).pool_gather is False
    assert not mk.CacheExec(paged_kv=True).GPR
    for val in pool.POOL_GATHER_VALUES:
        assert mk.CacheExec(pool_gather=val).pool_gather == val
    with pytest.raises(ValueError, match="pool_gather must be one of"):
        mk.CacheExec(pool_gather="sometimes")  # type: ignore[arg-type]
    ex = mk.with_cache(mk.ExecSpec(B=24, T=4), paged_kv=True, pool_gather=True)
    assert mk.cache_exec(ex).GPR and not mk.cache_exec(ex).FUSED
