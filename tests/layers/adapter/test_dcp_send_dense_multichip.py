# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DCP_MLA_SEND_DENSE over real collectives, eager and compiled.

On 8 ranks, each rank builds a synthetic kernel send buffer
(`sparse_mla_kernel.send_layout_shape`: per destination SEND_PAD_ROWS - 1 rows
of junk, the lse word row, then T * local partial rows, the padding tokens'
rows junk too) and merges it through `merge_send_layout` three ways: the
ragged exchange (TPU_DCP_MLA_SEND_LAYOUT_UNPACK, the served path), the ragged
exchange with torch views (no flag), and the dense exchange
(TPU_DCP_MLA_SEND_DENSE). Every output element must agree bit for bit, at
decode buckets 4 / 8 / 16 with all, all-but-one and one real token(s).

Positive controls the comparison must catch: the dense op handed every row as
live (padding tokens' junk merged instead of zero), and the send buffer with
its lse word row's lo halves zeroed.
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

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
HEADS = 32
VDIM = 512
CASES = [(4, 4), (4, 3), (4, 1), (8, 8), (8, 7), (8, 1), (16, 16), (16, 12), (16, 1)]
MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    "TPU_DCP_MLA_SEND_LAYOUT": True,
}


def _send_buffer(rank, world, tokens, real, pad):
    """One rank's int16 `[world, pad + tokens * local, V]` send buffer."""
    import numpy as np

    local = HEADS // world
    tl = tokens * local
    rng = np.random.default_rng(7000 + 97 * rank + 13 * tokens + real)
    buf = rng.integers(-32768, 32767, size=(world, pad + tl, VDIM), dtype=np.int16)
    o = rng.standard_normal((world, tl, VDIM)).astype(np.float32)
    o[:, real * local :] *= 1e30  # padding tokens: finite junk that would show
    bf = (o.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    buf[:, pad:] = bf
    lse = (rng.standard_normal((world, tl)) * 4).astype(np.float32)
    if rank == 3:
        lse[:, ::5] = -np.inf  # rows this rank owns no position of
    bits = lse.view(np.int32)
    buf[:, pad - 1, :tl] = (bits >> 16).astype(np.int16)
    buf[:, pad - 1, tl : 2 * tl] = (bits & 0xFFFF).astype(np.uint16).view(np.int16)
    return buf


def _check(rank, world, cp_mla):
    import torch

    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS as PAD

    for k, v in MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    local = HEADS // world
    report = {}
    for tokens, real in CASES:
        tl = tokens * local
        send = torch.from_numpy(_send_buffer(rank, world, tokens, real, PAD))
        bad = send.clone()
        bad[:, PAD - 1, tl : 2 * tl] = 0
        qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
        qsl_all = torch.tensor([0, tokens], dtype=torch.int32).to("tpu")
        for how in ("eager", "compiled"):
            key = f"T{tokens}_r{real}/{how}"
            try:
                wrap = mm._compile if how == "compiled" else (lambda f: f)

                def run(s, q, unpack, dense, tokens=tokens):
                    cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = unpack
                    cp_mla.envs.TPU_DCP_MLA_SEND_DENSE = dense
                    try:
                        f = wrap(lambda s, q: cp_mla.merge_send_layout(s, q, tokens))
                        return f(s.contiguous().to("tpu"), q).cpu()
                    finally:
                        cp_mla.envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK = False
                        cp_mla.envs.TPU_DCP_MLA_SEND_DENSE = False

                want = run(send, qsl, True, False)  # served: ragged + unpack op
                plain = run(send, qsl, False, False)  # ragged, torch views
                got = run(send, qsl, False, True)
                got_u = run(send, qsl, True, True)  # DENSE wins over UNPACK
                ctl_rows = run(send, qsl_all, False, True)  # no padding mask
                ctl_lse = run(bad, qsl, False, True)

                def nd(a):
                    return int((a.view(torch.int16) != want.view(torch.int16)).sum())

                report[key] = {
                    "elements": int(want.numel()),
                    "plain_n_diff": nd(plain),
                    "n_diff": nd(got),
                    "unpack_too_n_diff": nd(got_u),
                    "nonzero_real": bool((want[:real] != 0).any()),
                    "pad_rows_zero": bool((want[real:] == 0).all()),
                    "ctl_rows_n_diff": nd(ctl_rows),
                    "ctl_lse_n_diff": nd(ctl_lse),
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    return report


def _run_worker(result_dir: Path) -> None:
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
        from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="t5d_dense",
        )
        cp_mla.get_dcp_group = lambda: group
        from vllm_torchtpu.distributed import dcp as dcp_mod

        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world, SEND_PAD_ROWS - 1)
        cp_mla.prebuild_send_unpack(world)
        cp_mla.prebuild_send_dense(world)
        result["check"] = _check(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_send_dense_matches_ragged_on_real_collectives(tmp_path):
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
    total: dict[str, dict] = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key,
                dict(elements=0, plain=0, n_diff=0, unpack_too=0, nonzero=False,
                     pad_zero=True, ctl_rows=0, ctl_lse=0),
            )
            t["elements"] += c["elements"]
            t["plain"] += c["plain_n_diff"]
            t["n_diff"] += c["n_diff"]
            t["unpack_too"] += c["unpack_too_n_diff"]
            t["nonzero"] |= c["nonzero_real"]
            t["pad_zero"] &= c["pad_rows_zero"]
            t["ctl_rows"] += c["ctl_rows_n_diff"]
            t["ctl_lse"] += c["ctl_lse_n_diff"]
    for key, t in sorted(total.items()):
        print(f"SENDDENSE {key}: {t}")
        tokens, real = (int(x[1:]) for x in key.split("/")[0].split("_"))
        assert t["plain"] == 0, ("ragged torch-view path drifted", key, t)
        assert t["n_diff"] == 0, ("TPU_DCP_MLA_SEND_DENSE not bitwise", key, t)
        assert t["unpack_too"] == 0, (key, t)
        assert t["nonzero"] and t["pad_zero"], (key, t)
        if real < tokens:
            assert t["ctl_rows"] > 0, ("unmasked padding rows not caught", key, t)
        assert t["ctl_lse"] > 0, ("dropped lse lo halves not caught", key, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
