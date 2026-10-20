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
"""Unit tests for `exec_spec` (core tiles, tunables registry, ExecSpec) and the
model-agnostic default tile rule (`profiles.default`)."""

from __future__ import annotations

import dataclasses
import re
import types
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import exec_spec
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.profiles.default import (
    DEFAULT_CHIP,
    ChipBudget,
    default_tiles,
)


@dataclasses.dataclass(frozen=True)
class _TestTunables:
    """Test-only block tunables (registered under `_TEST_BLOCK`)."""

    width: int = 128
    fast: bool = False
    NUMERICS = {"width": "approx", "fast": "exact"}


_TEST_BLOCK = "test_exec_spec_block"


@pytest.fixture(autouse=True)
def _test_tunables(monkeypatch):
    """Register `_TestTunables` for one test; the registry is restored after."""
    monkeypatch.setitem(exec_spec._TUNABLES, _TEST_BLOCK, _TestTunables)


def test_core_tile_numerics_cover_every_core_field():
    names = {f.name for f in dataclasses.fields(Tiles)} - {"blocks"}
    assert set(exec_spec.TILE_NUMERICS) == names
    assert set(exec_spec.TILE_NUMERICS.values()) <= {"exact", "approx"}
    numerics = exec_spec.tile_numerics()
    assert {
        k: v for k, v in numerics.items() if "." not in k
    } == exec_spec.TILE_NUMERICS
    assert numerics[f"{_TEST_BLOCK}.width"] == "approx"
    assert numerics[f"{_TEST_BLOCK}.fast"] == "exact"


def test_tunables_registration_rejects_incomplete_or_conflicting_classes():
    @dataclasses.dataclass(frozen=True)
    class Partial:
        a: int = 0
        b: int = 1
        NUMERICS = {"a": "exact"}

    @dataclasses.dataclass(frozen=True)
    class Other:
        width: int = 128
        NUMERICS = {"width": "approx"}

    with pytest.raises(TypeError, match="missing"):
        exec_spec.register_tunables("test_partial", Partial)
    with pytest.raises(ValueError, match="already registered"):
        exec_spec.register_tunables(_TEST_BLOCK, Other)
    exec_spec.register_tunables(_TEST_BLOCK, _TestTunables)  # same class: no-op
    assert exec_spec.tunables_class(_TEST_BLOCK) is _TestTunables


def test_tiles_block_tunables_are_canonical_and_hashable():
    t = Tiles()
    assert t.block(_TEST_BLOCK) == _TestTunables()
    same = Tiles(blocks={_TEST_BLOCK: _TestTunables()})
    assert same == t and hash(same) == hash(t) and not same.blocks
    t2 = t.with_blocks({_TEST_BLOCK: {"width": 256}})
    assert t2.block(_TEST_BLOCK).width == 256 and t2 != t
    assert hash(t2) == hash(t.with_blocks({_TEST_BLOCK: {"width": 256}}))
    with pytest.raises(TypeError):
        t.with_blocks({_TEST_BLOCK: {"no_such_field": 1}})
    with pytest.raises(KeyError):
        t.with_blocks({"no_such_block": {"x": 1}})


# Block and cache names `exec_spec.py` must not mention (in any case).
_NON_CORE_NAMES = ("gqa", "gdn", "moe", "pool", "fused", "paged", "ckpt")


def test_core_exec_spec_mentions_no_block_or_cache_names():
    """`exec_spec.py` holds core knobs only: no block or pool name in it."""
    text = Path(exec_spec.__file__).read_text(encoding="utf-8")
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
                assert exec_spec.default_tiles(B, moe_tp=moe_tp, R=R) == ref
    small = ChipBudget(name="small", vmem_mib=16, smem_mib=1)
    assert default_tiles(8, chip=small).vmem_mib == 16


def test_resolved_tiles_agree_with_resolve_tiles_for_tp_geometry():
    g_tp = types.SimpleNamespace(moe_tp=True)
    for B in (4, 8, 16, 32, 64):
        ex = ExecSpec(B=B, R=8)
        assert ex.resolved_tiles == ex.resolve_tiles(g_tp) == ex.resolve_tiles()
