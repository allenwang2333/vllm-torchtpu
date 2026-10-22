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
"""TPU_DCP_A2A_INKERNEL over real collectives (8 ranks, one host), compiled.

The served stack's DCP merge exchange, flag off vs on, must agree bit for bit:

  backbone  `merge_send_layout` with the served env (send layout + UNPACK +
            GROUP_ROWS=8 + the Pallas small all-to-all capped at 8). Off, the
            buckets T = 16 / 32 take XLA's row-group ragged all-to-all; on, the
            Pallas one-round remote-DMA all-to-all. T = 8 (below the small cap)
            takes the small all-to-all either way. All, some and one real
            token(s); the padding tokens' rows carry finite junk that would show.
  drafter   the sized merge (TPU_MTP_DCP_MERGE_SIZED + _SMALL_A2A capped at 8).
            Off, T = 16 / 32 take XLA's all_to_all; on, the small all-to-all.
            -inf rows, a rank owning nothing and NaN partials of weight 0.

Routing is asserted, not assumed: the flag-on backbone trace must call the
small all-to-all op and the flag-off one must not; the drafter op must carry
the `_ik` name. Positive controls the comparison must catch (faults injected
into the flag-on path only):

  wrong_slot   the small all-to-all handed rank ^ 1 as its DCP index: every
               rank pushes into the wrong slot of its peers (a kernel
               addressing fault; arrival counts still balance, so no hang)
  lse_lo       the send buffer's lse word row with its lo halves zeroed
  rows_live    every padding token's row handed to the unpack as live
  rotated      the drafter's received blocks rotated by one row
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
SEND_CASES = [(8, 8), (16, 16), (16, 12), (16, 1), (32, 32), (32, 31), (32, 1)]
DRAFT_TOKENS = (4, 8, 16, 32)
SERVED_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 32,
    "TPU_DCP_LSE_MERGE_A2A_PACK": True,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    "TPU_DCP_MLA_SEND_LAYOUT": True,
    "TPU_DCP_MLA_SEND_LAYOUT_UNPACK": True,
    "TPU_DCP_MLA_SEND_GROUP_ROWS": 8,
    "TPU_DCP_MLA_SEND_SMALL_A2A": True,
    "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS": 8,
    "TPU_DCP_INDEXER_SMALL_AG": True,
    "TPU_SC_TOPK_DCP_MERGE_FLAT": True,
    "TPU_DCP_RANK_OPERAND": True,
    "TPU_SMALL_COLL_HBM_OPERAND": True,
    "TPU_MTP_DCP_MERGE_SIZED": True,
    "TPU_MTP_DCP_MERGE_SMALL_A2A": True,
    "TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS": 8,
    "TPU_DCP_A2A_INKERNEL": False,
    "TPU_DCP_A2A_INKERNEL_MAX_TOKENS": 32,
}


def _send_buffer(rank, world, tokens, real, pad):
    """One rank's int16 `[world, pad + tokens * local, V]` kernel send buffer
    (as test_dcp_send_dense_multichip: junk pad rows, the lse word row at
    pad - 1, then the partial rows; padding tokens' rows finite junk)."""
    import numpy as np

    local = HEADS // world
    tl = tokens * local
    rng = np.random.default_rng(9100 + 97 * rank + 13 * tokens + real)
    buf = rng.integers(-32768, 32767, size=(world, pad + tl, VDIM), dtype=np.int16)
    o = rng.standard_normal((world, tl, VDIM)).astype(np.float32)
    o[:, real * local :] *= 1e30
    buf[:, pad:] = (o.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    lse = (rng.standard_normal((world, tl)) * 4).astype(np.float32)
    if rank == 3:
        lse[:, ::5] = -np.inf  # rows this rank owns no position of
    bits = lse.view(np.int32)
    buf[:, pad - 1, :tl] = (bits >> 16).astype(np.int16)
    buf[:, pad - 1, tl : 2 * tl] = (bits & 0xFFFF).astype(np.uint16).view(np.int16)
    return buf


def _check_send(rank, world, cp_mla):
    import torch

    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS as PAD

    local = HEADS // world
    small_op = cp_mla._SEND_SMALL_OPS[world]
    calls = {"n": 0}

    def counted(*a):
        calls["n"] += 1
        return small_op(*a)

    cp_mla._SEND_SMALL_OPS[world] = counted
    good_rank = cp_mla._DCP_RANK
    wrong_rank = (good_rank.cpu() ^ 1).to("tpu")
    report = {}
    for tokens, real in SEND_CASES:
        tl = tokens * local
        key = f"send_T{tokens}_r{real}"
        try:
            send = torch.from_numpy(_send_buffer(rank, world, tokens, real, PAD))
            bad = send.clone()
            bad[:, PAD - 1, tl : 2 * tl] = 0
            qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
            qsl_all = torch.tensor([0, tokens], dtype=torch.int32).to("tpu")

            def run(s, q, on, rank_t=good_rank, tokens=tokens):
                cp_mla.envs.TPU_DCP_A2A_INKERNEL = on
                cp_mla._DCP_RANK = rank_t
                calls["n"] = 0
                try:
                    f = mm._compile(lambda s, q: cp_mla.merge_send_layout(s, q, tokens))
                    return f(s.contiguous().to("tpu"), q).cpu(), calls["n"]
                finally:
                    cp_mla.envs.TPU_DCP_A2A_INKERNEL = False
                    cp_mla._DCP_RANK = good_rank

            want, n_off = run(send, qsl, False)
            got, n_on = run(send, qsl, True)
            ctl_lse, _ = run(bad, qsl, True)
            ctl_rows, _ = run(send, qsl_all, True)
            ctl_slot, _ = run(send, qsl, True, wrong_rank)

            def nd(a):
                return int((a.view(torch.int16) != want.view(torch.int16)).sum())

            report[key] = {
                "elements": int(want.numel()),
                "small_calls_off": n_off,
                "small_calls_on": n_on,
                "n_diff": nd(got),
                "nonzero_real": bool((want[:real] != 0).any()),
                "pad_rows_zero": bool((want[real:] == 0).all()),
                "finite": bool(torch.isfinite(got.float()).all()),
                "ctl_lse_n_diff": nd(ctl_lse),
                "ctl_rows_n_diff": nd(ctl_rows),
                "ctl_slot_n_diff": nd(ctl_slot),
            }
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            traceback.print_exc()
            report[key] = {"error": repr(exc)[:600]}
    cp_mla._SEND_SMALL_OPS[world] = small_op
    return report


def _check_draft(rank, world, cp_mla):
    import torch

    base_heads = mm.HEADS
    local = base_heads // world

    def arm(on):
        cp_mla.envs.TPU_DCP_A2A_INKERNEL = on
        cp_mla._SIZED_MERGE_OPS.pop(world, None)
        cp_mla.prebuild_sized_merge(world)
        return lambda o, l: cp_mla.merge_lse_partials_scatter_heads(
            o, l, is_backbone=False
        )

    def partials(tokens):
        outputs, lses = mm._global_partials(world, max(tokens, 8))
        return outputs[:, :tokens].contiguous(), lses[:, :tokens].contiguous()

    real = cp_mla._small_exchange
    calls = {"n": 0}

    def counted(send, **kw):
        calls["n"] += 1
        return real(send, **kw)

    report = {}
    for tokens in DRAFT_TOKENS:
        key = f"draft_T{tokens}"
        try:
            outputs, lses = partials(tokens)
            poisoned = outputs.clone()
            poisoned[~torch.isfinite(lses)] = float("nan")
            ref = mm._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
            o_dev, p_dev = outputs[rank].to("tpu"), poisoned[rank].to("tpu")
            l_dev = lses[rank].to("tpu")
            got, n = {}, {}
            cp_mla._small_exchange = counted
            for name, on in (("off", False), ("on", True)):
                calls["n"] = 0
                fn = mm._compile(arm(on))
                got[name] = (fn(o_dev, l_dev).cpu(), fn(p_dev, l_dev).cpu())
                n[name] = calls["n"]
            cp_mla._small_exchange = real
            c = {
                "small_calls_off": n["off"],
                "small_calls_on": n["on"],
                "shape": list(got["on"][0].shape),
                **mm._error_stats(got["on"][0], ref),
            }
            c["bitwise"] = bool(
                torch.equal(
                    got["on"][0].view(torch.int16), got["off"][0].view(torch.int16)
                )
            )
            c["nan_bitwise"] = bool(
                torch.equal(
                    got["on"][1].view(torch.int16), got["off"][1].view(torch.int16)
                )
            )
            c["nan_input_bitwise"] = bool(
                torch.equal(
                    got["on"][1].view(torch.int16), got["on"][0].view(torch.int16)
                )
            )
            report[key] = c
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            report[key] = {"error": repr(exc)[:600]}
        finally:
            cp_mla._small_exchange = real
            cp_mla.envs.TPU_DCP_A2A_INKERNEL = False
    # Positive control at T = 16 (above the small cap: only the flag routes it
    # through the small exchange, so a caught rotation also proves the route).

    def rotated(send, **kw):
        import jax.numpy as jnp

        return jnp.roll(real(send, **kw), 1, axis=1)

    try:
        outputs, lses = partials(16)
        ref = mm._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
        off = mm._compile(arm(False))(
            outputs[rank].to("tpu"), lses[rank].to("tpu")
        ).cpu()
        cp_mla._small_exchange = rotated
        y = mm._compile(arm(True))(outputs[rank].to("tpu"), lses[rank].to("tpu")).cpu()
        c = mm._error_stats(y, ref)
        c["bitwise"] = bool(torch.equal(y.view(torch.int16), off.view(torch.int16)))
        report["draft_ctl_rotated_T16"] = c
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report["draft_ctl_rotated_T16"] = {"error": repr(exc)[:600]}
    finally:
        cp_mla._small_exchange = real
        cp_mla.envs.TPU_DCP_A2A_INKERNEL = False
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
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="dcp_a2a_inkernel",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        for k, v in SERVED_ENV.items():
            setattr(cp_mla.envs, k, v)
        cp_mla.prebuild_ragged_all_to_all(world, SEND_PAD_ROWS - 1)
        cp_mla.prebuild_send_unpack(world)
        cp_mla.prebuild_send_group(world, 8)
        cp_mla.prebuild_send_small(world)
        result["send"] = _check_send(rank, world, cp_mla)
        result["draft"] = _check_draft(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
    if "error" in result:
        raise SystemExit(1)


def assert_results(results):
    assert not [r for r in results if "error" in r], results
    total: dict[str, dict] = {}
    for r in results:
        for key, c in r["send"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key,
                dict(
                    elements=0,
                    n_diff=0,
                    nonzero=False,
                    pad_zero=True,
                    finite=True,
                    ctl_lse=0,
                    ctl_rows=0,
                    ctl_slot=0,
                    calls_off=0,
                    calls_on=0,
                ),
            )
            t["elements"] += c["elements"]
            t["n_diff"] += c["n_diff"]
            t["nonzero"] |= c["nonzero_real"]
            t["pad_zero"] &= c["pad_rows_zero"]
            t["finite"] &= c["finite"]
            t["ctl_lse"] += c["ctl_lse_n_diff"]
            t["ctl_rows"] += c["ctl_rows_n_diff"]
            t["ctl_slot"] += c["ctl_slot_n_diff"]
            t["calls_off"] += c["small_calls_off"]
            t["calls_on"] += c["small_calls_on"]
    for key, t in sorted(total.items()):
        print(f"A2AINK {key}: {t}")
        tokens, real = (int(x[1:]) for x in key.split("_")[1:])
        assert t["n_diff"] == 0, ("TPU_DCP_A2A_INKERNEL not bitwise", key, t)
        assert t["nonzero"] and t["pad_zero"] and t["finite"], (key, t)
        assert t["calls_on"] > 0, ("flag on did not route to the small a2a", key, t)
        if tokens > SERVED_ENV["TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS"]:
            assert t["calls_off"] == 0, ("flag off already took the small a2a", key, t)
        assert t["ctl_lse"] > 0, ("lse lo-halves control not caught", key, t)
        assert t["ctl_slot"] > 0, ("wrong-slot control not caught", key, t)
        if real < tokens:
            assert t["ctl_rows"] > 0, ("padding rows control not caught", key, t)
    for r in results:
        for key, c in r["draft"].items():
            print(f"A2AINK rank{r['rank']} {key}: {c}")
            assert "error" not in c, (r["rank"], key, c)
            if "ctl" in key:
                assert not c["bitwise"] and not c["within_1ulp"], (
                    "control not caught",
                    r["rank"],
                    key,
                    c,
                )
                continue
            assert c["bitwise"] and c["nan_bitwise"] and c["nan_input_bitwise"], (
                r["rank"],
                key,
                c,
            )
            assert c["within_1ulp"], (r["rank"], key, c)
            assert c["small_calls_on"] > 0, (
                "flag on did not route the drafter",
                r["rank"],
                key,
                c,
            )
            tokens = int(key.split("_T")[1])
            if tokens > SERVED_ENV["TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS"]:
                assert c["small_calls_off"] == 0, (r["rank"], key, c)


def test_dcp_a2a_inkernel_bitwise_on_real_collectives(tmp_path):
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
    if any("error" in r for r in results):
        print((done.stdout or "")[-8000:], (done.stderr or "")[-8000:])
    assert_results(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
