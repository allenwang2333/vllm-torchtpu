# SPDX-License-Identifier: Apache-2.0
"""Configuration resolver for heterogeneous disaggregated TPU serving tiers."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from typing import Any, Literal

from vllm_torchtpu import envs as tpu_envs

HardwareGeneration = Literal["v6e", "v7x", "v5p", "unknown"]
ServingRole = Literal["prefill", "decode"]


@dataclasses.dataclass(frozen=True)
class HeteroTierConfig:
    """Immutable hardware and serving configuration for one tier of an HDS pair."""

    role: ServingRole
    hardware_generation: HardwareGeneration
    kv_role: Literal["kv_producer", "kv_consumer"]
    kv_cache_dtype: str
    moe_requantize_weight_dtype: str
    requantize_weight_dtype: str
    enable_quantized_matmul_kernel: bool
    tensor_parallel_size: int
    data_parallel_size: int
    prefill_context_parallel_size: int
    expert_parallel: bool
    allow_cross_generation: bool

    def as_env_dict(self) -> dict[str, str]:
        """Convert tier quantization and feature flags into process environment variables."""
        return {
            "MOE_REQUANTIZE_WEIGHT_DTYPE": self.moe_requantize_weight_dtype,
            "REQUANTIZE_WEIGHT_DTYPE": self.requantize_weight_dtype,
            "ENABLE_QUANTIZED_MATMUL_KERNEL": "1" if self.enable_quantized_matmul_kernel else "0",
            "TPU_RAIDEN_ALLOW_CROSS_GENERATION": "1" if self.allow_cross_generation else "0",
        }


class HeteroTierResolver:
    """Pure deterministic resolver for heterogeneous TPU serving tier settings."""

    @staticmethod
    def resolve(
        *,
        role: ServingRole,
        hardware_generation: str,
        kv_cache_dtype: str = "fp8",
        allow_cross_generation: bool | None = None,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 8,
        prefill_context_parallel_size: int = 8,
        expert_parallel: bool = True,
        force_fp8_weights: bool = False,
        allow_pure_tp: bool = False,
    ) -> HeteroTierConfig:
        """Resolve tier execution and quantization settings.

        Prefill on v6e: Auto-selects INT8 weight requantization unless forced to FP8,
        unlocking 1,836 TOPS while keeping fp8 KV cache on the wire.
        Decode on v7x: Retains native FP8 execution.
        """
        gen_normalized = hardware_generation.strip().lower()
        if gen_normalized not in ("v6e", "v7x", "v5p"):
            gen_normalized = "unknown"

        cross_gen_enabled = (
            allow_cross_generation
            if allow_cross_generation is not None
            else bool(getattr(tpu_envs, "TPU_RAIDEN_ALLOW_CROSS_GENERATION", False))
        )

        is_prefill = role == "prefill"
        kv_role: Literal["kv_producer", "kv_consumer"] = "kv_producer" if is_prefill else "kv_consumer"

        # Wire format is strictly normalized to valid vLLM cli choices
        norm_kv_cache_dtype = "fp8_e4m3" if kv_cache_dtype in ("fp8", "float8_e4m3fn", "fp8_e4m3") else kv_cache_dtype

        if is_prefill and gen_normalized == "v6e" and not force_fp8_weights:
            # v6e Trillium native INT8 MXU optimization
            moe_weight_dtype = "int8"
            linear_weight_dtype = "int8"
            quant_matmul = True
        else:
            # v7x GhostFish native FP8 or fallback
            moe_weight_dtype = "float8_e4m3fn"
            linear_weight_dtype = "float8_e4m3fn"
            quant_matmul = False

        if is_prefill and tensor_parallel_size > 1 and prefill_context_parallel_size == 1 and not allow_pure_tp:
            raise ValueError(
                f"Inefficient prefill sharding for {gen_normalized}: TP={tensor_parallel_size} at "
                "PCP=1 causes HBM streaming boundness. Prefer PCP >= 4 or DP/EP sharding."
            )

        return HeteroTierConfig(
            role=role,
            hardware_generation=gen_normalized,  # type: ignore[arg-type]
            kv_role=kv_role,
            kv_cache_dtype=norm_kv_cache_dtype,
            moe_requantize_weight_dtype=moe_weight_dtype,
            requantize_weight_dtype=linear_weight_dtype,
            enable_quantized_matmul_kernel=quant_matmul,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size if is_prefill else 1,
            expert_parallel=expert_parallel,
            allow_cross_generation=cross_gen_enabled,
        )


__all__ = [
    "HardwareGeneration",
    "HeteroTierConfig",
    "HeteroTierResolver",
    "ServingRole",
]
