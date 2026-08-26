# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 tensor-layout adapter for sequence parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_torchtpu.distributed.sequence_parallel import (
    SequenceParallelMesh, resolve_sequence_parallel_size)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KimiSequenceParallel:
    """Sequence-parallel layout transitions used by the Kimi-K3 model."""

    def __init__(
            self,
            *,
            mesh: SequenceParallelMesh | None = None,
            token_sizes: tuple[int, ...] = (),
    ) -> None:
        self.mesh = mesh
        self.token_sizes = token_sizes

    @property
    def enabled(self) -> bool:
        return self.mesh is not None

    @classmethod
    def disabled(cls) -> KimiSequenceParallel:
        return cls()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> KimiSequenceParallel:
        sequence_parallel_size = resolve_sequence_parallel_size(vllm_config)
        if sequence_parallel_size == 1:
            return cls.disabled()

        parallel_config = vllm_config.parallel_config
        dp_size = int(parallel_config.data_parallel_size)
        pp_size = int(parallel_config.pipeline_parallel_size)
        if dp_size != 1 or pp_size != 1:
            raise ValueError(
                "Kimi-K3 sequence parallelism requires DP1/PP1, got "
                f"DP{dp_size}/PP{pp_size}")

        compile_sizes = vllm_config.compilation_config.compile_sizes
        if not compile_sizes:
            raise ValueError(
                "Kimi-K3 sequence parallelism requires static TPU "
                "compile_sizes")

        ordered_compile_sizes = tuple(
            sorted({int(size)
                    for size in compile_sizes}))
        incompatible_sizes = tuple(size for size in ordered_compile_sizes
                                   if size % sequence_parallel_size)
        if incompatible_sizes:
            raise ValueError(
                "Kimi-K3 compile sizes must be divisible "
                f"by sequence_parallel_size={sequence_parallel_size}; got "
                f"{incompatible_sizes}")

        mesh = SequenceParallelMesh.from_tensor_parallel_group(
            sequence_parallel_size)
        configured_tp_size = int(parallel_config.tensor_parallel_size)
        if configured_tp_size != len(mesh.plan.tp_ranks):
            raise RuntimeError(
                "Kimi-K3 sequence parallelism saw inconsistent TP sizes: "
                f"config={configured_tp_size}, group="
                f"{len(mesh.plan.tp_ranks)}")
        return cls(mesh=mesh, token_sizes=ordered_compile_sizes)

    def shard_model_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.shard_tokens(
            hidden_states,
            full_token_sizes=self.token_sizes,
        )

    def gather_attention_input(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.all_gather_tokens(hidden_states)

    def reduce_attention_output(self,
                                hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.reduce_attention_output(hidden_states)

    def gather_feed_forward_input(self,
                                  hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.all_gather_tokens(hidden_states)

    def shard_feed_forward_output(self,
                                  hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.shard_tokens(
            hidden_states,
            full_token_sizes=self.token_sizes,
        )

    def gather_model_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.mesh is not None
        return self.mesh.all_gather_tokens(hidden_states)
