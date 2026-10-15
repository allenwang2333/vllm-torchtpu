# SPDX-License-Identifier: Apache-2.0
"""Unit verification suite for HeteroTierResolver and HeteroTierConfig."""

from __future__ import annotations

import unittest

from vllm_torchtpu.distributed.kv_transfer.hetero_config import (
    HeteroTierConfig,
    HeteroTierResolver,
)


class HeteroTierConfigTest(unittest.TestCase):
    """Tests resolution, flag gating, and quantization rules for HDS."""

    def test_v6e_prefill_auto_configures_int8_weights_and_fp8_kv(self):
        """v6e Prefill selects INT8 weights (2x MXU speedup) while keeping FP8 KV cache."""
        config = HeteroTierResolver.resolve(
            role="prefill",
            hardware_generation="v6e",
            kv_cache_dtype="fp8",
            allow_cross_generation=True,
            prefill_context_parallel_size=8,
        )

        self.assertEqual(config.role, "prefill")
        self.assertEqual(config.kv_role, "kv_producer")
        self.assertEqual(config.hardware_generation, "v6e")
        self.assertEqual(config.moe_requantize_weight_dtype, "int8")
        self.assertEqual(config.requantize_weight_dtype, "int8")
        self.assertTrue(config.enable_quantized_matmul_kernel)
        self.assertEqual(config.kv_cache_dtype, "fp8_e4m3")
        self.assertTrue(config.allow_cross_generation)

        env = config.as_env_dict()
        self.assertEqual(env["MOE_REQUANTIZE_WEIGHT_DTYPE"], "int8")
        self.assertEqual(env["ENABLE_QUANTIZED_MATMUL_KERNEL"], "1")
        self.assertEqual(env["TPU_RAIDEN_ALLOW_CROSS_GENERATION"], "1")

    def test_v7x_decode_retains_native_fp8(self):
        """v7x Decode retains native FP8 weights and consumer role."""
        config = HeteroTierResolver.resolve(
            role="decode",
            hardware_generation="v7x",
            kv_cache_dtype="fp8",
            allow_cross_generation=True,
            data_parallel_size=8,
        )

        self.assertEqual(config.role, "decode")
        self.assertEqual(config.kv_role, "kv_consumer")
        self.assertEqual(config.hardware_generation, "v7x")
        self.assertEqual(config.moe_requantize_weight_dtype, "float8_e4m3fn")
        self.assertFalse(config.enable_quantized_matmul_kernel)
        self.assertEqual(config.kv_cache_dtype, "fp8_e4m3")

        env = config.as_env_dict()
        self.assertEqual(env["MOE_REQUANTIZE_WEIGHT_DTYPE"], "float8_e4m3fn")
        self.assertEqual(env["ENABLE_QUANTIZED_MATMUL_KERNEL"], "0")

    def test_v6e_staging_force_fp8_baseline(self):
        """Staging flag force_fp8_weights preserves pure FP8 on v6e for baseline comparison."""
        config = HeteroTierResolver.resolve(
            role="prefill",
            hardware_generation="v6e",
            force_fp8_weights=True,
            allow_cross_generation=True,
        )
        self.assertEqual(config.moe_requantize_weight_dtype, "float8_e4m3fn")
        self.assertFalse(config.enable_quantized_matmul_kernel)

    def test_flag_off_preserves_strict_isolation(self):
        """Flag OFF disables cross-generation allowance."""
        config = HeteroTierResolver.resolve(
            role="prefill",
            hardware_generation="v6e",
            allow_cross_generation=False,
        )
        self.assertFalse(config.allow_cross_generation)
        env = config.as_env_dict()
        self.assertEqual(env["TPU_RAIDEN_ALLOW_CROSS_GENERATION"], "0")

    def test_guard_rejects_inefficient_prefill_sharding(self):
        """Resolver fails closed if prefill is misconfigured with TP > 1 and PCP = 1."""
        with self.assertRaisesRegex(ValueError, "Inefficient prefill sharding"):
            HeteroTierResolver.resolve(
                role="prefill",
                hardware_generation="v6e",
                tensor_parallel_size=4,
                prefill_context_parallel_size=1,
            )


if __name__ == "__main__":
    unittest.main()
