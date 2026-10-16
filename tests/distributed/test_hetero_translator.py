# SPDX-License-Identifier: Apache-2.0
"""Unit verification suite for HeteroTranslator and sub-translators."""

from __future__ import annotations

import unittest

import torch

from vllm_torchtpu.distributed.kv_transfer.hetero_config import HeteroTierConfig, HeteroTierResolver
from vllm_torchtpu.distributed.kv_transfer.hetero_translator import (
    DTypeTranslator,
    HeteroTranslator,
    MemoryLayoutTranslator,
    ShardingTranslator,
    TransportMetadataTranslator,
)


class HeteroTranslatorTest(unittest.TestCase):
    """Tests cross-hardware sharding, layout, dtype, and transport translation."""

    def test_sharding_translator_v6e_tp8_to_v7x_tp4(self):
        """v6e (TP=8) to v7x (TP=4) for Qwen3-32B (8 KV heads): 2:1 gather pattern."""
        translator = ShardingTranslator(
            producer_tp=8,
            consumer_tp=4,
            num_kv_heads=8,
            producer_cores_per_chip=1,
            consumer_cores_per_chip=2,
        )

        self.assertEqual(translator.communication_pattern, "gather")
        self.assertEqual(translator.heads_per_producer_rank, 1)
        self.assertEqual(translator.heads_per_consumer_rank, 2)

        # Producer rank head slices
        self.assertEqual(translator.get_producer_head_slice(0), (0, 1))
        self.assertEqual(translator.get_producer_head_slice(1), (1, 2))
        self.assertEqual(translator.get_producer_head_slice(7), (7, 8))

        # Consumer rank head slices
        self.assertEqual(translator.get_consumer_head_slice(0), (0, 2))
        self.assertEqual(translator.get_consumer_head_slice(1), (2, 4))
        self.assertEqual(translator.get_consumer_head_slice(3), (6, 8))

        # Channel mappings
        mappings = translator.build_channel_mappings()
        self.assertEqual(len(mappings), 8)

        # Producer rank 0 -> Consumer rank 0, offset 0
        self.assertEqual(mappings[0].producer_rank, 0)
        self.assertEqual(mappings[0].consumer_rank, 0)
        self.assertEqual(mappings[0].head_indices, (0,))
        self.assertEqual(mappings[0].consumer_head_offset, 0)

        # Producer rank 1 -> Consumer rank 0, offset 1
        self.assertEqual(mappings[1].producer_rank, 1)
        self.assertEqual(mappings[1].consumer_rank, 0)
        self.assertEqual(mappings[1].head_indices, (1,))
        self.assertEqual(mappings[1].consumer_head_offset, 1)

        # Producer rank 2 -> Consumer rank 1, offset 0
        self.assertEqual(mappings[2].producer_rank, 2)
        self.assertEqual(mappings[2].consumer_rank, 1)
        self.assertEqual(mappings[2].head_indices, (2,))
        self.assertEqual(mappings[2].consumer_head_offset, 0)

    def test_sharding_translator_point_to_point(self):
        """TP=4 to TP=4 is point-to-point."""
        translator = ShardingTranslator(
            producer_tp=4,
            consumer_tp=4,
            num_kv_heads=8,
        )
        self.assertEqual(translator.communication_pattern, "point_to_point")
        mappings = translator.build_channel_mappings()
        self.assertEqual(len(mappings), 4)
        for i, m in enumerate(mappings):
            self.assertEqual(m.producer_rank, i)
            self.assertEqual(m.consumer_rank, i)
            self.assertEqual(m.consumer_head_offset, 0)

    def test_memory_layout_translator_block_calculation_and_alignment(self):
        """Block allocation and HBM burst alignment."""
        layout = MemoryLayoutTranslator(
            producer_block_size=16,
            consumer_block_size=16,
            head_dim=128,
            producer_alignment_bytes=128,
            consumer_alignment_bytes=256,
        )

        # 50 tokens with block_size=16 requires 4 blocks
        self.assertEqual(layout.calculate_num_blocks(50, 16), 4)

        # Alignment: 100 bytes -> 128B for producer, 256B for consumer
        self.assertEqual(layout.align_buffer_bytes(100, is_consumer=False), 128)
        self.assertEqual(layout.align_buffer_bytes(100, is_consumer=True), 256)

        # Strides for canonical layout
        strides = layout.get_strides(
            layout="canonical",
            num_blocks=4,
            block_size=16,
            num_heads=2,
            head_dim=128,
            dtype_bytes=1,
        )
        # canonical: [num_blocks, block_size, num_heads, head_dim]
        # s3 = 1
        # s2 = 128 * 1 = 128
        # s1 = 2 * 128 = 256
        # s0 = 16 * 256 = 4096
        self.assertEqual(strides, (4096, 256, 128, 1))

    def test_dtype_translator_element_sizes(self):
        """Dtype element size and lossless checks."""
        dtype = DTypeTranslator(producer_dtype="float8_e4m3fn", consumer_dtype="fp8")
        self.assertEqual(dtype.producer_bytes, 1)
        self.assertEqual(dtype.consumer_bytes, 1)
        self.assertTrue(dtype.is_lossless)

        dtype_mixed = DTypeTranslator(producer_dtype="bfloat16", consumer_dtype="float8_e4m3fn")
        self.assertEqual(dtype_mixed.producer_bytes, 2)
        self.assertEqual(dtype_mixed.consumer_bytes, 1)
        self.assertFalse(dtype_mixed.is_lossless)

    def test_transport_metadata_tags(self):
        """Raiden channel tag generation."""
        tag = TransportMetadataTranslator.build_raiden_channel_tag(
            request_id="abc-123",
            layer_idx=5,
            kv_component="key",
            head_idx=2,
        )
        self.assertEqual(tag, "req_abc-123_l5_key_h2")

    def test_hetero_translator_full_e2e_plan(self):
        """End-to-end plan generation from HeteroTierConfigs."""
        producer_tier = HeteroTierResolver.resolve(
            role="prefill",
            hardware_generation="v6e",
            tensor_parallel_size=8,
            prefill_context_parallel_size=8,
            allow_cross_generation=True,
        )
        consumer_tier = HeteroTierResolver.resolve(
            role="decode",
            hardware_generation="v7x",
            tensor_parallel_size=4,
            allow_cross_generation=True,
        )

        translator = HeteroTranslator.from_tier_configs(
            producer_tier=producer_tier,
            consumer_tier=consumer_tier,
            num_kv_heads=8,
            head_dim=128,
        )

        # Plan for 512 input tokens across 64 transformer layers
        plan = translator.plan_transfer(
            request_id="test-req-001",
            num_tokens=512,
            num_layers=64,
        )

        self.assertEqual(plan.request_id, "test-req-001")
        self.assertEqual(plan.producer_tp, 8)
        self.assertEqual(plan.consumer_tp, 4)
        self.assertEqual(plan.communication_pattern, "gather")
        self.assertEqual(len(plan.channel_mappings), 8)

        # Bytes per token per layer: 2 (K & V) * 8 heads * 128 head_dim * 1 byte (FP8) = 2,048 bytes
        self.assertEqual(plan.bytes_per_token_per_layer, 2048)

        # Total transfer bytes: 512 tokens * 64 layers * 2048 bytes = 67,108,864 bytes (~67.1 MB)
        self.assertEqual(plan.total_transfer_bytes, 67_108_864)

        # Wire transfer time estimate at 100 Gbps:
        # 67,108,864 / (100e9 / 8) = 67,108,864 / 12,500,000,000 = ~0.00537 s = ~5.37 ms
        latency_ms = plan.estimated_transfer_latency_ms(dcn_bandwidth_gbps=100.0)
        self.assertAlmostEqual(latency_ms, 5.3687, places=3)

    def test_translate_pallas_kv_buffer_fp8_to_bfloat16_and_back(self):
        """Verify sublane-packed Pallas KV buffer translation between fp8_e4m3fn and bfloat16."""
        num_blocks = 2
        page_size = 16
        head_dim = 128
        # 1 KV head per rank:
        # fp8_e4m3fn has logical (B, 16, 1, 4, 128) -> physical (B, 16, 1, 128, 4) with [K, V, 0, 0]
        # bfloat16 has logical (B, 16, 1, 2, 128) -> physical (B, 16, 1, 128, 2) with [K, V]
        src_phys = torch.zeros(
            (num_blocks, page_size, 1, head_dim, 4), dtype=torch.bfloat16
        )
        src_phys[..., 0] = -4.5  # K
        src_phys[..., 1] = 0.125  # V
        src_fp8 = src_phys.to(torch.float8_e4m3fn).contiguous()
        src_bytes = memoryview(src_fp8.view(torch.uint8).numpy().tobytes())

        dst_buf = bytearray(num_blocks * page_size * 1 * 2 * head_dim * 2)
        DTypeTranslator.translate_pallas_kv_buffer(
            src_bytes,
            dst_buf,
            dst_offset=0,
            num_blocks=num_blocks,
            src_dtype="torch.float8_e4m3fn",
            dst_dtype=torch.bfloat16,
            src_block_shape=(page_size, 1, 4, head_dim),
            dst_block_shape=(page_size, 1, 2, head_dim),
        )

        dst_bf16 = torch.frombuffer(dst_buf, dtype=torch.bfloat16).view(
            num_blocks, page_size, 1, head_dim, 2
        )
        self.assertTrue(torch.all(dst_bf16[..., 0] == -4.5))
        self.assertTrue(torch.all(dst_bf16[..., 1] == 0.125))


if __name__ == "__main__":
    unittest.main()
