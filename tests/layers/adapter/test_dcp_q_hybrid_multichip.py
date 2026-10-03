# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""`TPU_DCP_Q_HYBRID` over real collectives: the two DCP query paths agree.

Under query replication (`VLLM_DCP_Q_REPLICATE=1`) every rank projects the
whole DCP group's heads with the replicated q_b weight and absorbs them with
the replicated `W_UK_T`. The hybrid's small-bucket path instead projects only
this rank's heads -- a column slice (view) of that same replicated weight,
through the production `_q_proj(local=True)` -- absorbs with the local
`W_UK_T`, and all-gathers `(ql_nope, q_pe)` in one collective
(`all_gather_q_nope_pe_heads`). Downstream code is shared, so the two must
hand the attention the same group-wide q and produce the same output.

Per rank and token count (decode 64, mixed 4096), eager and compiled:

  1. the local q projection equals this rank's heads of the full projection;
  2. the attention inputs `(ql_nope_dcp, q_pe_dcp)` agree;
  3. the full chain -- a dense stand-in for the sparse kernel over this rank's
     KV shard, the production lse merge, `W_UV` -- agrees to bf16 tolerance;
  4. positive control: the local path with the wrong rank's column slice fails.

Run on one host (4 chips x 2 cores = 8 ranks, one DCP group of 8).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
LOCAL_HEADS = 4  # GLM-5.2 at TP16: 64 heads / 16
Q_LORA = 2048
NOPE, ROPE = 192, 64
KV_LORA = 512
V_HEAD = 256
KV_PER_RANK = 128
TOKENS = (64, 4096)
LAUNCH_TIMEOUT_SECONDS = 1500


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
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


# --------------------------------------------------------------------- worker
def _inputs(world, tokens, seed=20261003):
    """Every rank's inputs, identical on all ranks (KV shards per rank)."""
    import torch

    g = torch.Generator().manual_seed(seed + tokens)
    heads = LOCAL_HEADS * world
    q_c = torch.randn(tokens, Q_LORA, generator=g).to(torch.bfloat16)
    w_f = torch.randn(Q_LORA, heads * (NOPE + ROPE), generator=g) * 0.02
    w_s = w_f.abs().amax(dim=0) / 448.0
    w_q = (w_f / w_s).to(torch.float8_e4m3fn)
    w_uk_t = (torch.randn(heads, NOPE, KV_LORA, generator=g) * 0.07).to(torch.bfloat16)
    w_uv = (torch.randn(heads, KV_LORA, V_HEAD, generator=g) * 0.05).to(torch.bfloat16)
    kv_c = torch.randn(world, KV_PER_RANK, KV_LORA, generator=g).to(torch.bfloat16)
    k_pe = torch.randn(world, KV_PER_RANK, ROPE, generator=g).to(torch.bfloat16)
    return q_c, w_q, w_s.to(torch.float32), w_uk_t, w_uv, kv_c, k_pe


class _FakeQB:
    """The replicated q_b projection as `VllmFp8LinearMethodTPU` leaves it."""

    def __init__(self, weight, scale, rank_in_group, group_size, num_heads):
        from vllm_torchtpu.layers.adapter.linear_common import (
            WEIGHT_FLIPPED_ATTR,
            quantized_matmul,
        )
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            VllmFp8LinearMethodTPU,
        )

        self.weight = weight  # (k, n) = [Q_LORA, group heads * qk_head_dim]
        self.weight_scale = scale  # [n]
        self.bias = None
        self.rank_in_group = rank_in_group
        self.group_size = group_size
        self.quant_method = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self._mm = quantized_matmul
        self._num_heads = num_heads

    def __call__(self, x):
        return self._mm(x, self.weight, self.weight_scale), None

    def _local_view(self, out):
        n = out.shape[-2] // self.group_size
        start = self.rank_in_group * n
        return out[..., start : start + n, :].contiguous()


def _paths(rank, world, cp_mla, wrapper_cls):
    """(qrep, hybrid) functions of q_c -> dict of intermediate tensors."""
    import torch

    def build(q_c, w_q, w_s, w_uk_t, w_uv, kv_c, k_pe, slice_rank):
        layer = _FakeQB(w_q, w_s, slice_rank, world, LOCAL_HEADS)
        me = SimpleNamespace(
            q_lora_rank=Q_LORA,
            q_b_proj=layer,
            q_proj=None,
            dcp_group_size=world,
            num_heads=LOCAL_HEADS,
            qk_head_dim=NOPE + ROPE,
        )
        me._q_proj_layer = lambda: layer
        lo, hi = rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS
        w_uk_local, w_uv_local = w_uk_t[lo:hi].contiguous(), w_uv[lo:hi].contiguous()
        scale = (NOPE + ROPE) ** -0.5

        def attend_merge_project(ql_dcp, pe_dcp):
            # Dense stand-in for the sparse DCP kernel over this rank's shard.
            s = torch.einsum("thd,sd->ths", ql_dcp.float(), kv_c.float())
            s = s + torch.einsum("thd,sd->ths", pe_dcp.float(), k_pe.float())
            s = s * scale
            lse = torch.logsumexp(s, dim=-1)
            p = torch.exp(s - lse.unsqueeze(-1))
            partial = torch.einsum("ths,sd->thd", p, kv_c.float()).to(torch.bfloat16)
            merged = cp_mla.merge_lse_partials_scatter_heads(partial, lse)
            out = torch.bmm(merged.transpose(0, 1), w_uv_local).transpose(0, 1)
            return out.reshape(out.shape[0], -1)

        def qrep(q_c):
            q = wrapper_cls._q_proj(me, q_c, False)
            q = q.view(-1, LOCAL_HEADS * world, NOPE + ROPE)
            q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
            ql = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)
            q_pe = q_pe.contiguous()
            return {
                "q_local": layer._local_view(q),
                "ql_dcp": ql,
                "pe_dcp": q_pe,
                "out": attend_merge_project(ql, q_pe),
            }

        def hybrid(q_c):
            q = wrapper_cls._q_proj(me, q_c, True)
            q = q.view(-1, LOCAL_HEADS, NOPE + ROPE)
            q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
            ql = torch.bmm(q_nope.transpose(0, 1), w_uk_local).transpose(0, 1)
            ql_dcp, pe_dcp = cp_mla.all_gather_q_nope_pe_heads(ql, q_pe, KV_LORA)
            return {
                "q_local": q,
                "ql_dcp": ql_dcp,
                "pe_dcp": pe_dcp,
                "out": attend_merge_project(ql_dcp, pe_dcp),
            }

        return qrep, hybrid

    return build


def _stats(got, ref):
    import numpy as np

    g = got.float().cpu().numpy().astype(np.float64)
    r = ref.float().cpu().numpy().astype(np.float64)
    err = np.abs(g - r)
    rms = float(np.sqrt((r * r).mean())) or 1.0
    # bf16 tolerance: two ulps of the value, plus an ulp of the tensor's scale
    # for entries near zero.
    tol = 2.0**-7 * np.abs(r) + 2.0**-8 * rms
    return {
        "shape": list(got.shape),
        "finite": bool(np.isfinite(g).all()),
        "max_abs_err": float(err.max()),
        "rel_max_err": float(err.max() / (np.abs(r).max() or 1.0)),
        "frac_over_tol": float((err > tol).mean()),
        "bitwise_equal": bool((g == r).all()),
        "pass": bool(np.isfinite(g).all() and (err > tol).mean() <= 1e-3),
    }


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _check(rank, world, cp_mla, wrapper_cls):
    import torch

    report = {}
    build = _paths(rank, world, cp_mla, wrapper_cls)
    for tokens in TOKENS:
        host = _inputs(world, tokens)
        q_c, w_q, w_s, w_uk_t, w_uv, kv_c, k_pe = (t.to("tpu") for t in host)
        kv_c, k_pe = kv_c[rank], k_pe[rank]
        for how in ("eager", "compiled"):
            for arm, slice_rank in (("hybrid", rank), ("control_wrong_slice", (rank + 1) % world)):
                key = f"T{tokens}/{how}/{arm}"
                try:
                    qrep, hybrid = build(
                        q_c, w_q, w_s, w_uk_t, w_uv, kv_c, k_pe, slice_rank
                    )
                    if how == "compiled":
                        qrep, hybrid = _compile(qrep), _compile(hybrid)
                    ref = qrep(q_c)
                    got = hybrid(q_c)
                    report[key] = {k: _stats(got[k], ref[k]) for k in ref}
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:600]}
                if rank == 0:
                    print(f"CHECK {key}: {json.dumps(report[key])}", flush=True)
    return report


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
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
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla
        from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
            VllmTPUMultiHeadLatentAttentionWrapper,
        )

        world_group = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="dcp_q_hybrid_test",
        )
        cp_mla.get_dcp_group = lambda: group
        # The production merge as served (reduce-scatter; a2a off).
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A = False
        result["check"] = _check(
            rank, world, cp_mla, VllmTPUMultiHeadLatentAttentionWrapper
        )
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


# --------------------------------------------------------------------- pytest
def _launch(result_dir: Path) -> subprocess.CompletedProcess[str]:
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
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _results(result_dir: Path, world: int) -> list[dict]:
    out = []
    for rank in range(world):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_hybrid_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    for r in results:
        check = r["check"]
        for tokens in TOKENS:
            for how in ("eager", "compiled"):
                c = check[f"T{tokens}/{how}/hybrid"]
                assert "error" not in c, (r["rank"], tokens, how, c)
                for k in ("q_local", "ql_dcp", "pe_dcp", "out"):
                    assert c[k]["pass"], (r["rank"], tokens, how, k, c[k])
                ctl = check[f"T{tokens}/{how}/control_wrong_slice"]
                assert "error" not in ctl, (r["rank"], tokens, how, ctl)
                assert not ctl["out"]["pass"], (
                    "positive control: the wrong rank's q_b slice must fail",
                    r["rank"],
                    tokens,
                    how,
                    ctl["out"],
                )


def test_q_hybrid_paths_agree_on_real_collectives(tmp_path):
    result_dir = tmp_path / "dcp_q_hybrid"
    result_dir.mkdir()
    completed = _launch(result_dir)
    results = _results(result_dir, WORLD_SIZE)
    for line in (completed.stdout or "").splitlines():
        if line.startswith("CHECK "):
            print(line)
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    assert_hybrid_results(results)


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
