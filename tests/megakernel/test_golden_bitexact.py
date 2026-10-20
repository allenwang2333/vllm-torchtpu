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
"""Behaviour-preservation gate: golden hashes of hidden + every pool output.

Each case runs one tiny paged pooled step (CPU interpret) and hashes the hidden
output and every returned pool (the pools carry all KV / GDN state). The goldens
in `golden/b1_bitexact.json` were generated at the B1 head; Phase B refactor PRs
must reproduce them bit-for-bit. Only public APIs are used (profiles,
`default_pool_layout`, `states_to_pool`, `PallasWriter`), so the harness itself
stays unchanged across refactors. Regenerate only for an intentional numerics
change, and say so in the PR: `MK_REGEN_GOLDEN=1 pytest <this file>`.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.models import qwen35

GOLDEN = pathlib.Path(__file__).parent / "golden" / "b1_bitexact.json"
R = 2

CASES: dict[str, dict[str, Any]] = {
    # Qwen3.5 default profile (qwen35-baseline), production-size and tiny tiles
    "def_t1_b4_prod": dict(profile="qwen35-baseline", T=1, B=4, tiles_mode="prod"),
    "def_t1_b8": dict(profile="qwen35-baseline", T=1, B=8),
    "def_t1_b20_wide": dict(profile="qwen35-baseline", T=1, B=20),
    "def_t2_b4": dict(profile="qwen35-baseline", T=2, B=4),
    "def_t4_b16": dict(profile="qwen35-baseline", T=4, B=16),
    "def_t4_b24_gch": dict(profile="qwen35-baseline", T=4, B=24),
    "def_t4_b48": dict(profile="qwen35-baseline", T=4, B=48),
    "def_t1_b8_moetp": dict(profile="qwen35-baseline", T=1, B=8, moe_tp=True),
    "def_t1_b8_eplb": dict(profile="qwen35-baseline", T=1, B=8, e_slots=72),
    "def_t2_b4_stacke": dict(profile="qwen35-baseline", T=2, B=4, stack_e=True),
    "def_t4_b24_gpr": dict(profile="qwen35-baseline", T=4, B=24, pool_gather=True),
    # qwen35-candidate30 (approx), split / both sides of moe_group_rows
    "c30_t1_b4": dict(profile="qwen35-candidate30", T=1, B=4),
    "c30_t1_b4_prod": dict(profile="qwen35-candidate30", T=1, B=4, tiles_mode="prod"),
    "c30_t4_b8": dict(profile="qwen35-candidate30", T=4, B=8),
    "c30_t4_b16": dict(profile="qwen35-candidate30", T=4, B=16),
    "c30_t4_b24": dict(profile="qwen35-candidate30", T=4, B=24),
    "c30_t4_b40": dict(profile="qwen35-candidate30", T=4, B=40),
    "c30_t4_b48": dict(profile="qwen35-candidate30", T=4, B=48),
    "c30_t1_b8_moetp": dict(profile="qwen35-candidate30", T=1, B=8, moe_tp=True),
    "c30_t1_b8_eplb": dict(profile="qwen35-candidate30", T=1, B=8, e_slots=72),
    # qwen35-candidate30 tiles + fused pool I/O (= qwen35-candidate30-fused)
    "c30_fused_t1_b4": dict(profile="qwen35-candidate30", T=1, B=4, fused=True),
    "c30_fused_t4_b8": dict(profile="qwen35-candidate30", T=4, B=8, fused=True),
    "c30_fused_t4_b24": dict(profile="qwen35-candidate30", T=4, B=24, fused=True),
}


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _geom(moe_tp: bool = False, e_slots: int = 0) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4, moe_tp=moe_tp, e_slots=e_slots,
    )  # fmt: skip


def _tiny(t: mk.Tiles, ctx: int) -> mk.Tiles:
    """Shrink ring / bank sizes to the CPU geometry, keep every schedule flag."""
    return dataclasses.replace(
        t,
        bank=512 << 10,
        nbd=10,
        nbe=max(6, min(t.nbe, 12)),
        ebank=256 << 10,
        vmem_mib=32,
    ).with_blocks(
        {
            "gqa": {
                "kv_blk": min(256, ctx),
                "kv_sub": 128 if t.block("gqa").kv_sub else 0,
            }
        }
    )


def _h(a: Any) -> str:
    return hashlib.sha256(np.asarray(a).view(np.uint8).tobytes()).hexdigest()[:24]


def run_case(
    *,
    profile: str,
    T: int,
    B: int,
    fused: bool = False,
    pool_gather: Any = None,
    tiles_mode: str = "tiny",
    moe_tp: bool = False,
    e_slots: int = 0,
    stack_e: bool = False,
) -> dict[str, Any]:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _geom(moe_tp=moe_tp, e_slots=e_slots)
    ctx = 256
    base = mk.ExecSpec(
        R=R, B=B, T=T, ctx=ctx, stack_e=stack_e, interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=512 if B <= 8 else 2048),
    )  # fmt: skip
    ex = mk.get_profile(profile).exec_for(base, B=B, T=T, g=g)
    t = ex.resolved_tiles
    if tiles_mode == "tiny":
        t = _tiny(t, ctx)
    else:  # production ring / bank sizes; only the KV block must fit `ctx`
        gqa = t.block("gqa")
        t = t.with_blocks(
            {
                "gqa": {
                    "kv_blk": min(gqa.kv_blk, ctx),
                    "kv_sub": min(gqa.kv_sub, ctx // 2) if gqa.kv_sub else 0,
                }
            }
        )
    ex = dataclasses.replace(ex, tiles=t)
    if pool_gather is not None:
        ex = mk.with_cache(ex, pool_gather=pool_gather)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.key(900 + 10 * T + B), g, ex
    )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    if T == 1:
        n_read = int(ckpt.max()) - int(ckpt.min()) + 1
        ckpt = jnp.concatenate([ckpt, ckpt + n_read + 4], axis=-1)
        roff = roff.at[:, 1].set(-1)
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt[..., :T], jnp.maximum(roff, 0))
    )
    step, _ = mk.PallasWriter(fused_pool_io=fused).make_step(
        mesh, spec, ex, donate=fused
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    return {
        "hid": _h(out),
        "pools": [_h(r) for r in raw],
        "gpr": bool(mk.cache_exec(ex).GPR),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bitexact(name: str):
    """Hidden + every pool output reproduces the B1 golden hash bit-for-bit."""
    got = run_case(**CASES[name])
    if os.environ.get("MK_REGEN_GOLDEN"):
        gold = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        gold[name] = got
        GOLDEN.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")
        pytest.skip("regenerated golden")
    gold = json.loads(GOLDEN.read_text())[name]
    assert got["hid"] == gold["hid"], (name, "hidden", got["hid"], gold["hid"])
    assert got["pools"] == gold["pools"], (name, "pools", got["pools"], gold["pools"])


def test_golden_covers_every_case():
    assert set(json.loads(GOLDEN.read_text())) == set(CASES)
