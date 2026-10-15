# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The V-space DCP merge (`TPU_DCP_MERGE_VSPACE`) over real collectives,
8 ranks, composed onto the trial-4 s1 merge env (a2a <= 64 tokens with pack
+ real rows; chunk 8 from 1024 tokens; one lse gather <= 2048; bf16
reduce-scatter; WEIGHTED_HM from 2048; MERGE_PIECES=2).

Every rank builds its own partial `[T, 64, 512]` (bf16, NaN where its lse is
-inf on some rows) from per-(rank, head block) seeds, so each rank can rebuild
every peer's partial for its own head block and form the f64 reference of
`(sum_r softmax_r(lse) o_r) @ W_UV` for its heads. Arms (all compiled with
the TPU backend; off/on also eager):

* off: VSPACE off -- the latent merge, then this rank's W_UV (`bmm`, as
  `PallasMLAttentionBackendImpl.forward` does).
* on: VSPACE on -- W_UV for every head before the weighted reduce-scatter.
* on_rs32: on with TPU_DCP_MERGE_RS_BF16=0 (the f32 wire): must differ from
  `on`, i.e. RS_BF16=1 is in force on the V-space operand.
* ctl_swap / ctl_unw: on with the operand's W_UV heads reversed / the weight
  multiply skipped -- each must break the bound.

Chunked buckets T in {1024, 2048, 4096}: `on` must be within RS_BF16's
bound (7 bf16 ulps of sum_r w_r |o_r W_UV| against f64, the scale
`test_t3_final_merge_multichip` uses) and no less accurate than `off` in
mean absolute error; the controls must break the bound.
Reported: max scale-ulp per arm, max |on - off| in bf16 ulps, % of elements
differing. Decode bucket T=16 and the drafter (is_backbone False) at 2048:
on must be bit for bit off.

`VSPACE_HLO_DUMP=<dir>`: rank 0 dumps the optimized HLO; each arm's modules
are listed in `<dir>/arms.json` for the reduce-scatter byte count.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
HEADS = 64
C = 512
V = 256
CHUNKED_TOKENS = (1024, 2048, 4096)
DECODE_TOKENS = 16
LAUNCH_TIMEOUT = 3000
# RS_BF16=1's bound over the f32 merge (test_cp_mla_merge_multichip).
RS_BF16_MAX_SCALE_ULP = 7.0
SCALE_FLOOR = 2.0**-10

S1_MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    "TPU_DCP_MERGE_CHUNKED": 8,
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 1024,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 2048,
    "TPU_DCP_MERGE_RS_BF16": 1,
    "TPU_DCP_MERGE_WEIGHTED_HM": True,
    "TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS": 2048,
    "TPU_DCP_MLA_MERGE_PIECES": 2,
    "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS": 2048,
    "TPU_DCP_MERGE_RS_LAG": 0,
    "TPU_MTP_DCP_MERGE_SIZED": False,
    "TPU_DCP_MERGE_VSPACE": False,
    "TPU_DCP_MERGE_VSPACE_MIN_TOKENS": 1024,
}


def _lses(tokens, world=WORLD_SIZE, seed=20261014):
    import torch

    g = torch.Generator().manual_seed(seed + tokens)
    lses = torch.randn(world, tokens, HEADS, generator=g) * 4 + 3
    # Ranks that own none of a row's top-k: -inf (about 6 % of entries).
    own_none = torch.rand(world, tokens, HEADS, generator=g) < 0.06
    lses[own_none] = float("-inf")
    lses[:, 3] = float("-inf")  # nobody owns token 3
    lses[1:, 2] = float("-inf")  # rank 0 owns token 2 alone
    lses[:, 0] += 200.0
    return lses


def _block(r, hb, tokens, local, lses):
    """Rank r's partial for head block hb, `[T, local, C]` bf16, NaN where
    rank r's lse is -inf on some of those rows (zero-weight dummy-key rows)."""
    import torch

    g = torch.Generator().manual_seed(1_000_003 * (r + 1) + 7919 * hb + tokens)
    o = torch.randn(tokens, local, C, generator=g).to(torch.bfloat16)
    dead = ~torch.isfinite(lses[r, :, hb * local : (hb + 1) * local])
    o[dead & (torch.arange(tokens)[:, None] % 2 == 0)] = float("nan")
    return o


def _w_uv(seed=4242):
    import torch

    g = torch.Generator().manual_seed(seed)
    return (torch.randn(HEADS, C, V, generator=g) / C**0.5).to(torch.bfloat16)


def _partial(rank, tokens, lses, world=WORLD_SIZE):
    import torch

    local = HEADS // world
    return torch.cat(
        [_block(rank, hb, tokens, local, lses) for hb in range(world)], dim=1
    ).contiguous()


def _reference(rank, tokens, lses, w_uv, world=WORLD_SIZE):
    """f64 `[T, local, V]` of this rank's heads and the scale
    sum_r w_r |o_r @ W_UV_h|."""
    import numpy as np

    local = HEADS // world
    sl = slice(rank * local, (rank + 1) * local)
    lse = lses[:, :, sl].numpy().astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0.0, den, 1.0)
    wv = w_uv[sl].float().numpy().astype(np.float64)  # [local, C, V]
    ref = np.zeros((tokens, local, V))
    scale = np.zeros((tokens, local, V))
    for r in range(world):
        o = _block(r, rank, tokens, local, lses).float().numpy().astype(np.float64)
        o = np.where(w[r][..., None] > 0, np.nan_to_num(o), 0.0)
        v = np.matmul(o.transpose(1, 0, 2), wv).transpose(1, 0, 2)  # [T, local, V]
        ref += v * w[r][..., None]
        scale += np.abs(v) * w[r][..., None]
    return ref, np.maximum(scale, SCALE_FLOOR)


def _scale_ulp(got, ref, scale):
    import numpy as np

    err = np.abs(got.float().numpy().astype(np.float64) - ref)
    e = np.nan_to_num(err / np.exp2(np.floor(np.log2(scale)) - 7), nan=np.inf)
    return float(e.max()), float(np.nan_to_num(err, nan=np.inf).mean())


def _ulp(a, b):
    import torch

    def line(x):
        x = x.view(torch.int16).to(torch.int32)
        return torch.where(x < 0, -(x & 0x7FFF), x)

    return (line(a) - line(b)).abs()


def _set(cp_mla, hw, **kw):
    env = dict(S1_MERGE_ENV)
    env.update(kw)
    for k, v in env.items():
        setattr(cp_mla.envs, k, v)
    hw.VSPACE_FAULT = None


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _list_dump(d):
    return set(os.listdir(d)) if d and os.path.isdir(d) else set()


def _check(rank, world, cp_mla, hw):
    import torch

    local = HEADS // world
    w_uv = _w_uv()
    w_loc = w_uv[rank * local : (rank + 1) * local].contiguous().to("tpu")
    w_all = w_uv.contiguous().to("tpu")
    dump = os.environ.get("VSPACE_HLO_DUMP") if rank == 0 else None
    arms_hlo = {}
    report = {}

    def run_arm(name, fn, o, lse, compiled, tag):
        before = _list_dump(dump)
        f = _compile(fn) if compiled else fn
        out = f(o, lse).cpu()
        if dump and compiled:
            arms_hlo[tag] = sorted(_list_dump(dump) - before)
        return out

    for tokens in (*CHUNKED_TOKENS, DECODE_TOKENS):
        lses = _lses(tokens, world)
        o = _partial(rank, tokens, lses, world).to("tpu")
        lse = lses[rank].contiguous().to("tpu")
        qsl = torch.tensor([0, tokens], dtype=torch.int32).to("tpu")

        def f_off(o, lse, qsl=qsl, backbone=True):
            m = cp_mla.merge_lse_partials_scatter_heads(
                o, lse, is_backbone=backbone, query_start_loc=qsl
            )
            m_t = m.reshape(-1, local, C).transpose(0, 1)
            return torch.bmm(m_t, w_loc).transpose(0, 1).contiguous()

        def f_on(o, lse, qsl=qsl, backbone=True):
            m, projected = cp_mla.merge_lse_partials_scatter_heads_vspace(
                o, lse, w_all, None, is_backbone=backbone, query_start_loc=qsl
            )
            if projected:
                return m.reshape(-1, local, V).contiguous()
            m_t = m.reshape(-1, local, C).transpose(0, 1)
            return torch.bmm(m_t, w_loc).transpose(0, 1).contiguous()

        ref, scale = _reference(rank, tokens, lses, w_uv, world)
        hows = ("compiled", "eager")
        for how in hows:
            key = f"T{tokens}/{how}"
            try:
                comp = how == "compiled"
                res = {}
                _set(cp_mla, hw)
                res["off"] = run_arm("off", f_off, o, lse, comp, f"T{tokens}_off")
                _set(cp_mla, hw, TPU_DCP_MERGE_VSPACE=True)
                res["on"] = run_arm("on", f_on, o, lse, comp, f"T{tokens}_on")
                applies = cp_mla.vspace_merge_applies(tokens, True)
                c = {
                    "tokens": tokens,
                    "vspace": bool(applies),
                    "elements": int(res["on"].numel()),
                }
                if comp and tokens in CHUNKED_TOKENS:
                    _set(cp_mla, hw, TPU_DCP_MERGE_VSPACE=True, TPU_DCP_MERGE_RS_BF16=0)
                    res["on_rs32"] = run_arm(
                        "on_rs32", f_on, o, lse, comp, f"T{tokens}_on_rs32"
                    )
                    for fault, nm in (
                        ("swap_heads", "ctl_swap"),
                        ("unweighted", "ctl_unw"),
                    ):
                        _set(cp_mla, hw, TPU_DCP_MERGE_VSPACE=True)
                        hw.VSPACE_FAULT = fault
                        res[nm] = run_arm(nm, f_on, o, lse, comp, f"T{tokens}_{nm}")
                        hw.VSPACE_FAULT = None
                if tokens == 2048 and comp:
                    # The drafter never compares the token count: latent merge.
                    _set(cp_mla, hw)
                    d_off = _compile(lambda a, b: f_off(a, b, backbone=False))(
                        o, lse
                    ).cpu()
                    _set(cp_mla, hw, TPU_DCP_MERGE_VSPACE=True)
                    d_on = _compile(lambda a, b: f_on(a, b, backbone=False))(
                        o, lse
                    ).cpu()
                    c["drafter_on_vs_off_ndiff"] = int((_ulp(d_on, d_off) > 0).sum())
                for nm, r_ in res.items():
                    c[f"{nm}_scale_ulp"], c[f"{nm}_mean_abs"] = _scale_ulp(
                        r_, ref, scale
                    )
                u = _ulp(res["on"], res["off"])
                c["on_vs_off_ndiff"] = int((u > 0).sum())
                c["on_vs_off_maxulp"] = int(u.max())
                if "on_rs32" in res:
                    c["on_vs_rs32_ndiff"] = int(
                        (_ulp(res["on"], res["on_rs32"]) > 0).sum()
                    )
                c["on_finite"] = bool(torch.isfinite(res["on"].float()).all())
                report[key] = c
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    if dump:
        Path(dump, "arms.json").write_text(json.dumps(arms_hlo, indent=1))
    return report


def _run_worker(result_dir: Path) -> None:
    rank = int(os.environ["RANK"])
    if rank == 0 and os.environ.get("VSPACE_HLO_DUMP"):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_dump_to={os.environ['VSPACE_HLO_DUMP']}"
            + " --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=^$"
        )
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla
        from vllm_torchtpu.layers.adapter import dcp_weighted_hm as hw

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="vspace_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        result["check"] = _check(rank, world, cp_mla, hw)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_dcp_merge_vspace_on_real_collectives(tmp_path):
    import subprocess

    import test_cp_mla_merge_multichip as mm

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--result-dir",
        str(tmp_path),
    ]
    done = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=mm._prepare_worker_env(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=LAUNCH_TIMEOUT,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-6000:],
        (done.stderr or "")[-6000:],
    )
    total = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(key, {"tokens": c["tokens"], "vspace": c["vspace"]})
            for k, v in c.items():
                if k in ("tokens", "vspace"):
                    continue
                if k.endswith("ndiff") or k == "elements":
                    t[k] = t.get(k, 0) + v
                elif k.endswith("finite"):
                    t[k] = t.get(k, True) and v
                elif k.endswith("mean_abs"):
                    t[k] = t.get(k, 0) + v / len(results)
                else:
                    t[k] = max(t.get(k, 0), v)
    failures = []
    for key, t in sorted(total.items()):
        t["on_vs_off_pct"] = 100.0 * t["on_vs_off_ndiff"] / t["elements"]
        print(f"VSPACEMERGE {key}: {json.dumps(t)}")
        if t["tokens"] in CHUNKED_TOKENS:
            # In V space the f64 scale is sum_r w_r |o_r W_UV|; RS_BF16's own
            # bound applies to it directly. (The latent arm's error is set by
            # the latent scale pushed through W_UV, so its max over this
            # scale is large where V cancels: reported, not a bound.)
            bound = RS_BF16_MAX_SCALE_ULP
            ok = t["vspace"] and t["on_finite"] and t["on_scale_ulp"] <= bound
            ok &= t["on_mean_abs"] <= t["off_mean_abs"]
            if key.endswith("compiled"):
                ok &= t["ctl_swap_scale_ulp"] > bound and t["ctl_unw_scale_ulp"] > bound
                ok &= t["on_vs_rs32_ndiff"] > 0
            if "drafter_on_vs_off_ndiff" in t:
                ok &= t["drafter_on_vs_off_ndiff"] == 0
        else:
            ok = (not t["vspace"]) and t["on_vs_off_ndiff"] == 0
        if not ok:
            failures.append((key, t))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
