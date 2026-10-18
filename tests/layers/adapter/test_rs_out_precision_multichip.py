# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DCP_RS_OUT_PRECISION (bf16 reduce-scatter for the unchunked merge
buckets, upstream #1808) on real attention partials over real collectives
(8 ranks), eager and compiled.

Reads `T4S2_PARTIALS` (the mixed-256 / mixed-2048 off+on partials
test_t4_stack2_composition writes). Merge env = the trial-5 s4 merge with the
a2a cap at 32: the 256 step is an unchunked reduce-scatter bucket, the 2048
step a chunked one (TPU_DCP_MERGE_RS_BF16=1 already). Arms on the "on"
partials:

* f32: TPU_DCP_RS_OUT_PRECISION=0 (s4 today).
* bf16: TPU_DCP_RS_OUT_PRECISION=1.
* ctl: bf16 with every rank's lse rolled by one token (must break the bound).

Unchunked step: bf16 is held to the bf16-wire bound already accepted for the
chunked merge (RS_BF16_MAX_SCALE_ULP bf16 scale-ulps over f32's own error vs
f64) and must differ from f32 (else the flag did nothing); max / mean |err|
vs f64 for both arms are printed (RSOPMERGE lines). Chunked step: bitwise
unchanged by the flag.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_cp_mla_merge_multichip as mm  # noqa: E402
import test_t3_final_merge_multichip as t3m  # noqa: E402
import test_t4_combined_merge_multichip as t4m  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
MERGE_ENV = dict(
    t4m.S23_MERGE_ENV,
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS=32,
    TPU_DCP_LSE_MERGE_A2A_PACK=True,
    TPU_DCP_LSE_MERGE_A2A_REAL_ROWS=True,
    TPU_DCP_MERGE_RS_BF16=1,
    TPU_DCP_MERGE_WEIGHTED_HM=True,
    TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS=2048,
    TPU_DCP_MLA_MERGE_PIECES=2,
    TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS=2048,
    TPU_DCP_MERGE_CHUNKED_SMALL=2,
    TPU_DCP_MERGE_RS_LAG=0,
    TPU_MTP_DCP_MERGE_SIZED=False,
    TPU_DCP_MERGE_VSPACE=False,
)


def _err_stats(got, outputs, lses, rank, world):
    import numpy as np

    local = outputs.shape[2] // world
    sl = slice(rank * local, (rank + 1) * local)
    out = outputs[:, :, sl].float().numpy().astype(np.float64)
    lse = lses[:, :, sl].numpy().astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0.0, den, 1.0)
    ref = (out * w[..., None]).sum(axis=0)
    err = np.abs(got.float().numpy().astype(np.float64) - ref)
    return {
        "sum_abs": float(err.sum()),
        "max_abs": float(err.max()),
        "n": int(err.size),
        "ref_max_abs": float(np.abs(ref).max()),
        "ref_sum_abs": float(np.abs(ref).sum()),
    }


def _check(rank, cp_mla, path):
    import numpy as np
    import torch

    z = np.load(path)
    dcp = int(z["dcp"])
    assert dcp == WORLD_SIZE, dcp
    for k, v in MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    merge = cp_mla.merge_lse_partials_scatter_heads
    report = {}
    for step in range(int(z["steps"])):
        real = int(z[f"s{step}_real"])
        for how in ("eager", "compiled"):
            key = f"s{step}/{how}"
            try:
                qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")

                def fn0(o, lse, qsl=qsl):
                    return merge(o, lse, query_start_loc=qsl)

                res = {}
                for name, flag, roll in (
                    ("f32", False, False),
                    ("bf16", True, False),
                    ("ctl", True, True),
                ):
                    cp_mla.envs.TPU_DCP_RS_OUT_PRECISION = flag
                    fn = fn0 if how == "eager" else mm._compile(fn0)
                    o, lse = t4m._partials(z, step, "on", rank, dcp, roll)
                    res[name] = fn(o.to("tpu"), lse.to("tpu")).cpu()[:real]
                cp_mla.envs.TPU_DCP_RS_OUT_PRECISION = False
                bucket = int(z[f"s{step}_on_out"].shape[0]) // dcp
                outs, lses = t3m._all_partials(z, step, "on", dcp, real)
                u = t4m._ulp(res["bf16"], res["f32"])
                c = {
                    "tokens": bucket,
                    "real": real,
                    "chunked": bool(cp_mla._merge_chunks(bucket) > 1),
                    "elements": int(u.numel()),
                    "bf16_vs_f32_ndiff": int((u > 0).sum()),
                    "bf16_vs_f32_maxulp": int(u.max()) if u.numel() else 0,
                }
                for name in ("f32", "bf16", "ctl"):
                    c[f"{name}_scale"] = t3m._scale_err(
                        res[name], outs, lses, rank, dcp
                    )["max_ulp"]
                    for k, v in _err_stats(res[name], outs, lses, rank, dcp).items():
                        c[f"{name}_{k}"] = v
                d = (res["bf16"].float() - res["f32"].float()).abs()
                c["bf16_vs_f32_max_abs"] = float(d.max())
                c["bf16_vs_f32_sum_abs"] = float(d.sum())
                report[key] = c
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:400]}
    return report


def _run_worker(result_dir: Path, path: str) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
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

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="rsop_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        result["check"] = _check(rank, cp_mla, path)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_rs_out_precision_on_real_collectives(tmp_path):
    path = os.environ.get("T4S2_PARTIALS")
    if not path or not os.path.exists(path):
        pytest.skip("T4S2_PARTIALS (npz from test_t4_stack2_composition) unset")
    import subprocess

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
        "--partials",
        path,
    ]
    done = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=mm._prepare_worker_env(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=mm.LAUNCH_TIMEOUT_SECONDS,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-8000:],
    )
    total = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key, {"tokens": c["tokens"], "real": c["real"], "chunked": c["chunked"]}
            )
            for k, v in c.items():
                if k in ("tokens", "real", "chunked"):
                    continue
                if (
                    k.endswith(("ndiff", "sum_abs"))
                    or k in ("elements",)
                    or k.endswith("_n")
                ):
                    t[k] = t.get(k, 0) + v
                else:
                    t[k] = max(t.get(k, 0), v)
    failures = []
    for key, t in sorted(total.items()):
        for name in ("f32", "bf16", "ctl"):
            t[f"{name}_mean_abs"] = t[f"{name}_sum_abs"] / max(t[f"{name}_n"], 1)
        t["ref_mean_abs"] = t["f32_ref_sum_abs"] / max(t["f32_n"], 1)
        t["bf16_vs_f32_mean_abs"] = t["bf16_vs_f32_sum_abs"] / max(t["elements"], 1)
        print(f"RSOPMERGE {key}: {json.dumps(t)}")
        if t["chunked"]:
            ok = t["bf16_vs_f32_ndiff"] == 0
        else:
            bound = t["f32_scale"] + t3m.RS_BF16_MAX_SCALE_ULP
            ok = (
                t["bf16_scale"] <= bound
                and t["ctl_scale"] > bound
                and t["bf16_vs_f32_ndiff"] > 0
            )
        if not ok:
            failures.append((key, t))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--partials", required=True)
    a = p.parse_args()
    _run_worker(a.result_dir, a.partials)
