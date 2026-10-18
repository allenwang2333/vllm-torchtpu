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
"""B1 tests: per-bucket tuning lives in tile rules, enforced numerics classes."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import ast
import dataclasses
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import profile as profile_mod
from vllm_torchtpu.megakernel import spec as spec_mod
from vllm_torchtpu.megakernel.cache import _should_use_pool_gather
from vllm_torchtpu.megakernel.models import qwen35

BLOCKS_DIR = pathlib.Path(mk.__file__).parent / "blocks"

# Row-count names that must never be compared to a literal inside `blocks/`.
_ROW_NAMES = {"B", "NS", "MR", "MB", "M", "CB"}
# Remaining literal row-count comparisons are hardware / layout limits, not
# per-bucket tuning (each one with its reason).
_ALLOWED = {
    ("moe.py", "MR <= 128"): "lane top-k keeps all MR rows in one 128-lane vreg",
    ("gqa.py", "ex.NS <= 64"): "GQ packs 128 // NS lanes per sequence (m | l) field",
}


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _row_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _ROW_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _ROW_NAMES:
        return node.attr
    return None


def _is_num(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def test_blocks_have_no_row_count_literal_comparisons():
    """Blocks read per-bucket tuning from `Tiles`, never `B <= 40`-style literals."""
    found = []
    for path in sorted(BLOCKS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = [node.left, *node.comparators]
            for a, b in zip(ops, ops[1:]):
                pair = (_row_name(a) and _is_num(b)) or (_row_name(b) and _is_num(a))
                if pair:
                    key = (path.name, ast.unparse(node))
                    if key not in _ALLOWED:
                        found.append(f"{path.name}:{node.lineno}: {key[1]}")
    assert not found, "row-count literal comparisons in blocks/:\n" + "\n".join(found)


def test_tile_numerics_registry_covers_every_field():
    names = {f.name for f in dataclasses.fields(mk.Tiles)}
    assert set(spec_mod.TILE_NUMERICS) == names
    assert set(spec_mod.TILE_NUMERICS.values()) <= {"exact", "approx"}
    assert spec_mod.POOL_IO_NUMERICS == {
        "fused_pool_io": "approx",
        "pool_gather": "exact",
    }
    # Classes the plan fixes from TPU / CPU evidence.
    for f in ("kv_sub", "route_vec", "dense_acc_pair", "moe_unroll", "gdn_head_major"):
        assert spec_mod.TILE_NUMERICS[f] == "approx", f
    assert spec_mod.TILE_NUMERICS["gq_pack"] == "approx"
    for f in ("pair_kv", "kv_ering", "gqa_qv"):
        assert spec_mod.TILE_NUMERICS[f] == "exact", f


def test_per_bucket_thresholds_come_from_tiles():
    """`ExecSpec.WIDE` and the MoE grouped-release threshold read `Tiles`."""
    assert mk.ExecSpec(B=16).WIDE is False and mk.ExecSpec(B=17).WIDE is True
    t = dataclasses.replace(mk.default_tiles(8), wide_rows=4)
    assert mk.ExecSpec(B=8, tiles=t).WIDE is True
    g = qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=4,
    )
    spec = qwen35.qwen35_spec(g)
    for B, rows, want in ((40, 40, True), (48, 40, False), (8, 4, False)):
        tiles = dataclasses.replace(
            mk.spec.TINY_TILES, bank=512 << 10, moe_group_rows=rows
        )
        ex = mk.ExecSpec(B=B, R=2, ctx=256, tiles=tiles, interpret=True)
        plan = mk.planner.plan(spec, ex)
        assert plan.block_params["moe"].group_rel is want, (B, rows)


def test_default_tile_rule_lives_in_default_profile():
    from vllm_torchtpu.megakernel.profiles.default import frozen_parity_tiles

    for R in (1, 2, 8):
        for B in range(1, 65):
            assert mk.default_tiles(B, R=R) == frozen_parity_tiles(B, R=R)
    assert mk.DEFAULT_PROFILE.tiles_for(20, R=8) == frozen_parity_tiles(20, R=8)


# ---------------------------------------------------------------------------
# Numerics classes
# ---------------------------------------------------------------------------


def test_profile_numerics_are_derived_and_enforced():
    assert profile_mod.approx_features(mk.DEFAULT_PROFILE) == ()
    c30 = mk.get_profile("qwen35-candidate30")
    feats = set(profile_mod.approx_features(c30))
    assert {"tiles.kv_sub", "tiles.gdn_head_major", "tiles.moe_unroll"} <= feats
    # Exact-class flags the profile turns on are not listed.
    assert not feats & {"tiles.pair_kv", "tiles.kv_ering", "tiles.gqa_qv"}
    fused = mk.get_profile("qwen35-candidate30-fused")
    assert fused.numerics == "approx" and fused.fused_pool_io == "auto"
    assert "fused_pool_io" in profile_mod.approx_features(fused)

    # A profile declared exact that turns on an approx feature is rejected.
    def _kv_sub_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
        return dataclasses.replace(mk.default_tiles(B, R=R), kv_sub=1024)

    with pytest.raises(ValueError, match="tiles.kv_sub"):
        mk.Profile(name="bad-exact", tiles=_kv_sub_rule)
    with pytest.raises(ValueError, match="fused_pool_io"):
        mk.Profile(name="bad-exact-fused", fused_pool_io="auto")

    # Exact-class changes keep an exact profile exact.
    def _sched_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
        return dataclasses.replace(mk.default_tiles(B, R=R), pair_kv=True, nkv=4)

    assert mk.Profile(name="sched-only", tiles=_sched_rule).numerics == "exact"


def test_overrides_relabel_and_never_silently_stay_exact():
    d = mk.DEFAULT_PROFILE
    assert d.with_overrides(fused_pool_io=False, pool_gather=False) is d
    f = d.with_overrides(fused_pool_io="auto")
    assert (f.name, f.numerics, f.fused_pool_io) == (
        "default+overrides",
        "approx",
        "auto",
    )
    g = d.with_overrides(pool_gather=True)  # GPR is exact-class
    assert (g.name, g.numerics) == ("default+overrides", "exact")
    c = mk.get_profile("qwen35-candidate30").with_overrides(fused_pool_io="auto")
    assert (c.name, c.numerics) == ("qwen35-candidate30+overrides", "approx")
    assert c.with_overrides(fused_pool_io=True).name == "qwen35-candidate30+overrides"


def test_candidate30_profiles_carry_baseline():
    for name in ("qwen35-candidate30", "qwen35-candidate30-fused"):
        b = mk.get_profile(name).baseline
        assert b is not None and b.ref == "candidate30"
        assert b.commit == "8743659cdfcfbb0d77d7fc454f519520d223e785"
        for job in (
            "j-07676b1a-25aa-48f3-a600",
            "j-ffda2788-53ab-4d8a-a419",
            "j-de863aa4-7371-49ac-b57f",
        ):
            assert job in b.job
        assert b.config and b.bench


# ---------------------------------------------------------------------------
# CPU bit-exactness of exact-class tile fields and exact profiles
# ---------------------------------------------------------------------------


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _geom() -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=4,
    )


def _tiny(t: mk.Tiles) -> mk.Tiles:
    return dataclasses.replace(
        t,
        bank=512 << 10,
        nbd=10,
        nbe=max(6, min(t.nbe, 12)),
        ebank=256 << 10,
        kv_blk=256,
        vmem_mib=32,
        kv_sub=128 if t.kv_sub else 0,
    )


_OUT_CACHE: dict[tuple, list] = {}


def _pooled_outputs(prof: mk.Profile | str, *, T: int, B: int, **tile_kw: Any) -> list:
    """Bytes of hidden + pools of one tiny paged pooled step under `prof` (cached)."""
    key = (
        prof if isinstance(prof, str) else id(prof),
        T,
        B,
        tuple(sorted(tile_kw.items())),
    )
    if key not in _OUT_CACHE:
        _OUT_CACHE[key] = _pooled_outputs_uncached(prof, T=T, B=B, **tile_kw)
    return _OUT_CACHE[key]


def _pooled_outputs_uncached(
    prof: mk.Profile | str, *, T: int, B: int, **tile_kw: Any
) -> list:
    R = 2
    mesh = _cpu_mesh(R)
    g = _geom()
    base = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=256,
        paged_kv=True,
        num_pages=512,
        stack_e=False,
        interpret=True,
    )
    ex = mk.get_profile(prof).exec_for(base, B=B, T=T, g=g)
    ex = dataclasses.replace(
        ex, tiles=dataclasses.replace(_tiny(ex.resolved_tiles), **tile_kw)
    )
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(31 + T + B), g, ex
    )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    tp = NamedSharding(mesh, P("tp"))
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    step, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    return [np.asarray(x).view(np.uint8) for x in (out, *raw)]


def _assert_same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# Exact-class fields: (field, value off the default, extra tiles the field needs).
_EXACT_FLIPS = [
    ("nbd", 8, {}),
    ("nkv", 4, {}),
    ("gdn_bufs", 2, {}),
    ("lazy", False, {}),
    ("pair_kv", True, {}),
    ("kv_ering", True, {}),
    ("gqa_qv", True, {}),
    ("rel_fetch", True, {}),
    ("bounds_checks", True, {}),
    ("vmem_mib", 48, {}),
    ("smem_mib", 2, {}),
    ("moe_group_rows", 0, {"moe_unroll": True}),
]


def test_exact_flip_list_covers_registry():
    exact = {k for k, v in spec_mod.TILE_NUMERICS.items() if v == "exact"}
    # kv_scatter_vec_ns only affects the non-paged path (tested separately).
    assert exact == {f for f, _, _ in _EXACT_FLIPS} | {"kv_scatter_vec_ns"}


@pytest.mark.parametrize(
    "field,value,extra", _EXACT_FLIPS, ids=[f for f, _, _ in _EXACT_FLIPS]
)
def test_exact_tile_fields_reproduce_default(field: str, value: Any, extra: dict):
    """Each `exact` tile field, moved alone, keeps hidden + pools bit-identical."""
    ref = _pooled_outputs("default", T=2, B=4, **extra)
    _assert_same(_pooled_outputs("default", T=2, B=4, **extra, **{field: value}), ref)


def test_exact_kv_scatter_vec_ns_non_paged():
    """`kv_scatter_vec_ns` only switches the host KV update between equivalent forms."""
    R, B = 2, 4
    mesh = _cpu_mesh(R)
    g = _geom()
    outs = []
    for ns in (16, 0):
        tiles = dataclasses.replace(
            mk.spec.TINY_TILES, bank=512 << 10, kv_scatter_vec_ns=ns
        )
        ex = mk.ExecSpec(B=B, T=1, R=R, ctx=256, tiles=tiles, interpret=True)
        pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(5), g, ex)
        pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
            pos, hid_l, w_l, st_l, g, ex, mesh=mesh
        )
        step, _ = qwen35.make_step(mesh, g, ex)
        out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)
        outs.append(
            [
                np.asarray(x).view(np.uint8)
                for x in (out_hid, out_st["kc"], out_st["vc"])
            ]
        )
    _assert_same(outs[1], outs[0])


def test_every_exact_profile_reproduces_default():
    """Every registered `exact` profile matches `default` bit-exactly (CPU)."""
    exact = [n for n in mk.list_profiles() if mk.get_profile(n).numerics == "exact"]
    assert "default" in exact
    for T, B in ((1, 8), (2, 4)):
        ref = _pooled_outputs("default", T=T, B=B)
        for name in exact:
            _assert_same(_pooled_outputs(name, T=T, B=B), ref)


# ---------------------------------------------------------------------------
# pool_gather semantics and config errors
# ---------------------------------------------------------------------------


def test_profile_pool_gather_true_means_where_eligible():
    g = _geom()
    spec = qwen35.qwen35_spec(g)
    pb = spec.pool_backend
    prof = mk.DEFAULT_PROFILE.with_overrides(pool_gather=True)
    base = mk.ExecSpec(R=2, ctx=256, paged_kv=True, num_pages=512, interpret=True)
    ex1 = prof.exec_for(base, B=4, T=1, g=g)
    assert ex1.pool_gather == "eligible" and ex1.GPR is False
    assert _should_use_pool_gather(spec, pb, ex1, use_fused=False) is False  # no raise
    ex4 = prof.exec_for(base, B=24, T=4, g=g)
    assert _should_use_pool_gather(spec, pb, ex4, use_fused=False) is True
    # An explicit (strict) request on an ineligible bucket is a config error.
    with pytest.raises(mk.MegakernelConfigError, match="pool_gather=True"):
        _should_use_pool_gather(
            spec, pb, dataclasses.replace(ex1, pool_gather=True), use_fused=False
        )
    with pytest.raises(ValueError, match="pool_gather"):
        mk.Profile(name="bad-pg", pool_gather="eligible")  # type: ignore[arg-type]


def test_exec_for_explicit_base_settings_win():
    g = _geom()
    gpr = mk.DEFAULT_PROFILE.with_overrides(pool_gather=None)
    base = mk.ExecSpec(R=2, ctx=256, paged_kv=True, interpret=True)
    assert gpr.exec_for(base, B=24, T=4, g=g).pool_gather is None
    for explicit in (True, "eligible"):
        b2 = dataclasses.replace(base, pool_gather=explicit)
        assert mk.DEFAULT_PROFILE.exec_for(b2, B=24, T=4, g=g).pool_gather == explicit
    t = mk.spec.TINY_TILES
    assert gpr.exec_for(dataclasses.replace(base, tiles=t), B=8, T=1, g=g).tiles is t


@pytest.mark.parametrize("field,match", [("vmem_mib", "VMEM"), ("smem_mib", "SMEM")])
def test_scratch_budget_overflow_is_config_error(field: str, match: str):
    """VMEM / SMEM budget overflow is a configuration error, not a layout error."""
    tiles = dataclasses.replace(_tiny(mk.default_tiles(4, R=2)), **{field: 0})
    ex = mk.ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        paged_kv=True,
        num_pages=512,
        interpret=True,
        tiles=tiles,
    )
    with pytest.raises(mk.MegakernelConfigError, match=match):
        qwen35.build(_geom(), ex)
    assert not issubclass(mk.MegakernelConfigError, ValueError)


def test_load_builtins_sets_flag_only_after_success(monkeypatch: pytest.MonkeyPatch):
    from vllm_torchtpu.megakernel import profiles

    monkeypatch.setattr(profile_mod, "_BUILTINS_LOADED", False)

    def _boom() -> None:
        raise ImportError("synthetic")

    monkeypatch.setattr(profiles, "load_builtin_profiles", _boom)
    with pytest.raises(ImportError, match="synthetic"):
        profile_mod.list_profiles()
    assert profile_mod._BUILTINS_LOADED is False
    monkeypatch.undo()
    assert "qwen35-candidate30-fused" in profile_mod.list_profiles()
