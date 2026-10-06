# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained TPU smoke test script for Qwen3Next attention-to-MoE overlap.

Can be run directly via `python3 tests/models/vllm/tpu_smoke_attn_moe_overlap.py`
or via `python3 -m pytest tests/models/vllm/tpu_smoke_attn_moe_overlap.py -v`.
When run on TPU silicon with 8 cores, executes 8-process compiled TPU test
across all 8 physical ranks and asserts TPU_8RANK_COMPILED_PASS.
On CPU fallback, runs mathematical equivalence test and prints TPU_SMOKE_PASS.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn


class _SimpleRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        var = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps) * self.weight
        return normed, x


class _Smoke4LayerModel(nn.Module):
    def __init__(
        self,
        d: int,
        tp_size: int,
        mode: str,
        attn_weights: list[torch.Tensor],
        moe_weights: list[torch.Tensor],
        gate_weights: list[torch.Tensor],
    ) -> None:
        super().__init__()
        self.d = d
        self.tp_size = tp_size
        self.mode = mode
        self.norms_in = nn.ModuleList([_SimpleRMSNorm(d) for _ in range(4)])
        self.norms_post = nn.ModuleList([_SimpleRMSNorm(d) for _ in range(4)])
        self.attn_weights = nn.ParameterList([nn.Parameter(w) for w in attn_weights])
        self.moe_weights = nn.ParameterList([nn.Parameter(w) for w in moe_weights])
        self.gate_weights = nn.ParameterList([nn.Parameter(w) for w in gate_weights])
        self.final_norm = _SimpleRMSNorm(d)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.distributed import (
            tensor_model_parallel_all_gather,
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_reduce_scatter,
        )

        curr_h = x
        curr_res = None
        tp_size = self.tp_size
        n = x.shape[0]

        for layer_idx in range(4):
            if curr_res is None:
                curr_res = curr_h
                h_in, _ = self.norms_in[layer_idx](curr_h)
            else:
                h_in, curr_res = self.norms_in[layer_idx](curr_h, curr_res)

            a_partial = h_in @ self.attn_weights[layer_idx]

            if self.mode == "base":
                a = tensor_model_parallel_all_reduce(a_partial)
                h_post, curr_res = self.norms_post[layer_idx](a, curr_res)
                logits = h_post @ self.gate_weights[layer_idx]
                p_partial = (h_post @ self.moe_weights[layer_idx]) * (
                    logits.sum(-1, keepdim=True) * 0.01 + 1
                )
                curr_h = tensor_model_parallel_all_reduce(p_partial)
            else:
                # Rank-invariant overlap
                x_loc = tensor_model_parallel_reduce_scatter(
                    a_partial + curr_res.to(a_partial.dtype) * (1.0 / tp_size), 0
                )
                h_loc, _ = self.norms_post[layer_idx](x_loc)
                logits_loc = h_loc @ self.gate_weights[layer_idx]
                logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(0, 0, n)
                h_full = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, n)
                p_partial = (h_full @ self.moe_weights[layer_idx]) * (
                    logits.sum(-1, keepdim=True) * 0.01 + 1
                )
                curr_h = tensor_model_parallel_all_reduce(p_partial + a_partial)

        final_out, _ = self.final_norm(curr_h, curr_res)
        return final_out


def _run_8rank_worker(
    rank: int,
    world_size: int,
    n: int,
    d: int,
    shared_attn_weights: list[list[torch.Tensor]],
    shared_moe_weights: list[list[torch.Tensor]],
    shared_gate_weights: list[torch.Tensor],
    x_input: torch.Tensor,
) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="tpu_dist")
    torch.empty((1,), device="tpu").cpu()

    from vllm.distributed import parallel_state

    parallel_state.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    parallel_state._TP = parallel_state.init_model_parallel_group(
        [list(range(world_size))],
        rank,
        backend="tpu_dist",
        use_device_communicator=True,
        group_name="tp",
    )

    rank_attn = [shared_attn_weights[i][rank] for i in range(4)]
    rank_moe = [shared_moe_weights[i][rank] for i in range(4)]

    base_model = _Smoke4LayerModel(
        d=d,
        tp_size=world_size,
        mode="base",
        attn_weights=rank_attn,
        moe_weights=rank_moe,
        gate_weights=shared_gate_weights,
    ).to("tpu")

    overlap_model = _Smoke4LayerModel(
        d=d,
        tp_size=world_size,
        mode="overlap",
        attn_weights=rank_attn,
        moe_weights=rank_moe,
        gate_weights=shared_gate_weights,
    ).to("tpu")

    x = x_input.to("tpu")

    compiled_base = torch.compile(base_model)
    compiled_overlap = torch.compile(overlap_model)

    out_base = compiled_base(x)
    out_overlap = compiled_overlap(x)

    rel_rmse = (torch.norm(out_overlap - out_base) / torch.norm(out_base)).item()
    assert not torch.isnan(out_overlap).any(), f"Rank {rank} overlap output has NaN"
    assert not torch.isinf(out_overlap).any(), f"Rank {rank} overlap output has Inf"
    assert rel_rmse < 2e-3, f"Rank {rank} rel_rmse {rel_rmse} exceeds 2e-3"

    dist.barrier()
    if rank == 0:
        print(f"TPU 8-rank verification successful: rel_rmse={rel_rmse:.6e}")
        print("TPU_8RANK_COMPILED_PASS")

    dist.destroy_process_group()


def run_cpu_fallback_smoke() -> None:
    """CPU mathematical equivalence and Dynamo capture fallback."""
    torch.manual_seed(42)
    tp_size = 8
    d = 64
    n = 16
    x = torch.randn(n, d)

    norms_in = [_SimpleRMSNorm(d) for _ in range(4)]
    norms_post = [_SimpleRMSNorm(d) for _ in range(4)]
    attn_weights = [[torch.randn(d, d) for _ in range(tp_size)] for _ in range(4)]
    moe_weights = [[torch.randn(d, d) for _ in range(tp_size)] for _ in range(4)]
    gate_weights = [torch.randn(d, 8) for _ in range(4)]
    final_norm = _SimpleRMSNorm(d)

    # 1. Base simulation
    curr_h = x
    curr_res = None
    for layer_idx in range(4):
        if curr_res is None:
            curr_res = curr_h
            h_in, _ = norms_in[layer_idx](curr_h)
        else:
            h_in, curr_res = norms_in[layer_idx](curr_h, curr_res)

        a_partials = [h_in @ attn_weights[layer_idx][r] for r in range(tp_size)]
        a = sum(a_partials)
        h_post, curr_res = norms_post[layer_idx](a, curr_res)
        logits = h_post @ gate_weights[layer_idx]
        p_partials = [
            (h_post @ moe_weights[layer_idx][r])
            * (logits.sum(-1, keepdim=True) * 0.01 + 1)
            for r in range(tp_size)
        ]
        curr_h = sum(p_partials)
    base_out, _ = final_norm(curr_h, curr_res)

    # 2. Rank-invariant overlap simulation
    curr_h = x
    curr_res = None
    m = n // tp_size

    for layer_idx in range(4):
        if curr_res is None:
            curr_res = curr_h
            h_in, _ = norms_in[layer_idx](curr_h)
        else:
            h_in, curr_res = norms_in[layer_idx](curr_h, curr_res)

        a_partials = [h_in @ attn_weights[layer_idx][r] for r in range(tp_size)]
        h_locs = []
        logits_locs = []
        for r in range(tp_size):
            x_loc = (
                sum(a_partials[k] + curr_res * (1.0 / tp_size) for k in range(tp_size))
            )[r * m : (r + 1) * m]
            h_loc, _ = norms_post[layer_idx](x_loc)
            logits_loc = h_loc @ gate_weights[layer_idx]
            h_locs.append(h_loc)
            logits_locs.append(logits_loc)

        logits = torch.cat(logits_locs, dim=0)
        h_full = torch.cat(h_locs, dim=0)
        out_partials = []
        for r in range(tp_size):
            p = (h_full @ moe_weights[layer_idx][r]) * (
                logits.sum(-1, keepdim=True) * 0.01 + 1
            )
            out_partials.append(p + a_partials[r])

        curr_h = sum(out_partials)

    overlap_out, _ = final_norm(curr_h, curr_res)
    assert torch.allclose(overlap_out, base_out, atol=1e-4, rtol=1e-3)
    print("TPU_SMOKE_PASS")


def main() -> None:
    has_tpu = False
    num_tpu_devices = 0
    if hasattr(torch, "tpu") and torch.tpu.is_available():
        has_tpu = True
        num_tpu_devices = torch.tpu.device_count()
    elif os.environ.get("USE_TPU", "0") == "1":
        has_tpu = True
        num_tpu_devices = 8

    if has_tpu and num_tpu_devices >= 8:
        if "LOCAL_RANK" not in os.environ:
            print("LOCAL_RANK not set; re-executing via torchrun --nproc-per-node=8...")
            import subprocess

            subprocess.run(["torchrun", "--nproc-per-node=8", __file__], check=True)
            return

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.manual_seed(42)
        n = 16384
        d = 64
        x_input = torch.randn(n, d)
        shared_attn = [[[torch.randn(d, d) for _ in range(8)] for _ in range(4)]][0]
        shared_moe = [[[torch.randn(d, d) for _ in range(8)] for _ in range(4)]][0]
        shared_gate = [torch.randn(d, 8) for _ in range(4)]

        _run_8rank_worker(
            rank, world_size, n, d, shared_attn, shared_moe, shared_gate, x_input
        )
    else:
        print("Running CPU fallback smoke test...")
        run_cpu_fallback_smoke()


@pytest.mark.cpu_test
def test_tpu_smoke_script() -> None:
    run_cpu_fallback_smoke()


if __name__ == "__main__":
    main()
