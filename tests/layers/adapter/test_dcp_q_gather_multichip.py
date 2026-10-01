# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DCP_Q_GATHER_MODE on real devices and real collectives.

The unit tests in test_mla_attention.py pin the head order with a simulated
group. This pins what only hardware can answer: that every mode produces the
same kernel query -- bit for bit -- through vLLM's real TPU all-gather inside
a compiled graph, and how long each takes at the served shapes.

The world group stands in for the DCP group (8 ranks, one host, 4 chips x 2
cores), with GLM-5.2's per-rank geometry rescaled so the group still holds 64
heads: 8 local heads x 8 ranks, nope 192, rope 64, latent 512. The function
under test is the q path of `PallasMLAttentionBackendImpl.forward` from the
`W_UK` absorb through the kernel's [T, 64, 576] concat (padded to 640 as the
kernel's wrapper does), built from the same helpers forward calls.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
LOCAL_HEADS = 8
NOPE, ROPE, LATENT = 192, 64, 512
# Served decode step and the mixed twin step's token count.
TOKEN_COUNTS = (16, 4096)
MODES = ("split", "fused", "pre_absorb")
# Chained copies per compiled call, so host dispatch is amortized; more at
# decode, where one copy is tens of microseconds.
REPS = {16: 32, 4096: 4}
ITERS = 10
LAUNCH_TIMEOUT_SECONDS = 1200


def _worker_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
            prepare_tpu_environment,
        )
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")

    prepared_keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    original_values = {key: os.environ.get(key) for key in prepared_keys}
    try:
        for key in prepared_keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    env.pop("LIBTPU_INIT_ARGS", None)
    return env


def _q_path(mode, q_nope, q_pe, w_uk_local, w_uk_group):
    """forward's q path for one mode: [T, h, nope], [T, h, rope] -> [T, H, 640]."""
    import torch

    from vllm_torchtpu.layers.adapter.cp_mla_attention import (
        all_gather_heads,
        all_gather_heads_packed,
    )

    def absorb(qn, w):
        return torch.bmm(qn.transpose(0, 1), w).transpose(0, 1)

    if mode == "pre_absorb":
        q_nope, q_pe = all_gather_heads_packed(q_nope, q_pe)
        ql_nope = absorb(q_nope, w_uk_group)
    elif mode == "fused":
        ql_nope, q_pe = all_gather_heads_packed(absorb(q_nope, w_uk_local), q_pe)
    else:
        ql_nope = all_gather_heads(absorb(q_nope, w_uk_local))
        q_pe = all_gather_heads(q_pe)
    q = torch.cat([ql_nope, q_pe], dim=-1)
    return torch.nn.functional.pad(q, (0, 640 - q.shape[-1]))


def _measure(rank: int) -> dict[str, object]:
    import torch
    from torch_tpu._internal import compile as tpu_compile
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla
    from vllm_torchtpu.utils import synchronize_tensors

    # The world group carries no device communicator; build an all-ranks
    # group that does, the way the intra-chip group is built, and stand it in
    # for the DCP group.
    world_group = get_world_group()
    world = init_model_parallel_group(
        [list(range(WORLD_SIZE))],
        world_group.local_rank,
        dist.get_backend(world_group.device_group),
        group_name="dcp_q_gather_test",
    )
    assert int(world.rank_in_group) == rank
    cp_mla.get_dcp_group = lambda: world
    group_heads = LOCAL_HEADS * WORLD_SIZE
    heads = slice(rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS)

    torch.manual_seed(1234)  # same full tensors on every rank
    w_uk_group = (torch.randn(group_heads, NOPE, LATENT) * 0.05).to(torch.bfloat16)
    w_uk_group_tpu = w_uk_group.to("tpu")
    w_uk_local_tpu = w_uk_group[heads].contiguous().to("tpu")

    out: dict[str, object] = {"rank": rank, "cases": []}
    for tokens in TOKEN_COUNTS:
        reps = REPS[tokens]
        q_nope_all = torch.randn(reps, tokens, group_heads, NOPE).to(torch.bfloat16)
        q_pe_all = torch.randn(reps, tokens, group_heads, ROPE).to(torch.bfloat16)
        q_nope = q_nope_all[:, :, heads].contiguous().to("tpu")
        q_pe = q_pe_all[:, :, heads].contiguous().to("tpu")

        results = {}
        for mode in MODES:

            def step(qn, qp, mode=mode, reps=reps):
                # Each copy waits on the previous one through a one-element
                # dependency, so the collectives run back to back the way one
                # layer follows another, not overlapped across copies.
                outs, dep = [], None
                for i in range(reps):
                    qn_i = qn[i] if dep is None else qn[i] + dep * 0
                    o = _q_path(mode, qn_i, qp[i], w_uk_local_tpu, w_uk_group_tpu)
                    outs.append(o)
                    dep = o[:1, :1, :1]
                return outs

            fn = torch.compile(
                step, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
            )
            first = fn(q_nope, q_pe)
            first_cpu = [t.cpu() for t in first]
            torch.accelerator.synchronize()
            times = []
            for _ in range(ITERS):
                world_group.barrier()
                t0 = time.perf_counter()
                res = fn(q_nope, q_pe)
                synchronize_tensors(res)
                times.append((time.perf_counter() - t0) / reps * 1e6)
            times.sort()
            results[mode] = (first_cpu, times[len(times) // 2], times[0])

        base = results["split"][0]
        # Anchor the reference on the replicated device absorb, so a head
        # permutation shared by every mode would still be caught.
        ref_ql = (
            torch.bmm(
                q_nope_all[0].to("tpu").transpose(0, 1), w_uk_group_tpu
            )
            .transpose(0, 1)
            .cpu()
        )
        ref_max = float(
            (base[0][..., :LATENT].float() - ref_ql.float()).abs().max().item()
        )
        pe_exact = bool(torch.equal(base[0][..., LATENT : LATENT + ROPE], q_pe_all[0]))
        case = {
            "tokens": tokens,
            "ref_ql_max_abs": ref_max,
            "ref_ql_scale": float(ref_ql.float().abs().max().item()),
            "q_pe_exact": pe_exact,
        }
        for mode in MODES:
            got, med, best = results[mode]
            diffs = [
                float((g.float() - b.float()).abs().max().item())
                for g, b in zip(got, base)
            ]
            case[mode] = {"max_abs_vs_split": max(diffs), "us_med": med, "us_min": best}
        out["cases"].append(case)
    return out


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result = _measure(rank)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def test_dcp_q_gather_modes_bit_exact_on_device(tmp_path):
    result_dir = tmp_path / "dcp_q_gather"
    result_dir.mkdir()
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)

    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    assert process.returncode == 0 and not errors, (
        f"worker failure: returncode={process.returncode}, errors={errors}\n"
        f"output tail:\n{(output or '')[-12000:]}"
    )
    print("DCP_Q_GATHER_RESULTS " + json.dumps(results))
    out_dir = os.environ.get("CDK_OUTPUT_DIR")
    if out_dir:
        Path(out_dir, "dcp_q_gather_results.json").write_text(json.dumps(results))
    for r in results:
        for case in r["cases"]:
            assert case["q_pe_exact"], (r["rank"], case)
            # Device absorb of 8-head slices vs one 64-head bmm: same per-head
            # contraction; allow at most a bf16 ULP-scale slack.
            assert case["ref_ql_max_abs"] <= 1e-2 * case["ref_ql_scale"], case
            for mode in ("fused", "pre_absorb"):
                assert case[mode]["max_abs_vs_split"] == 0.0, (r["rank"], mode, case)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        raise ValueError("expected --worker")
    _run_worker(args.result_dir)


if __name__ == "__main__":
    _main()
