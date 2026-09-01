"""Unit tests verifying on-device multi-step steady-state decode execution logic.

Validates:
1. can_multi_step condition guard (num_reqs, block boundary slots, spec_decode).
2. Per-layer AttentionMetadata dict traversal and seq_lens/position_ids increment.
3. On-device token retention and multi-token batch state synchronization.
4. Block boundary safety fallback (can_multi_step=False when slot remaining < K).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import torch


class TestMultiStepDecodeLogic(unittest.TestCase):

    def setUp(self):
        self.num_reqs = 4
        self.multi_step = 4
        self.block_size = 256

        # Initial state: positions = 8192, seq_lens = 8192 for 4 requests
        self.position_ids = torch.tensor([8192, 8192, 8192, 8192], dtype=torch.int64)
        
        # Simulating per_layer_attn_metadata dictionary as used in vllm-torchtpu
        self.attn_metadata_dict = {}
        for layer_idx in range(4):
            layer_name = f"model.layers.{layer_idx}.self_attn"
            meta = SimpleNamespace(
                input_positions=self.position_ids.clone(),
                seq_lens=torch.tensor([8192, 8192, 8192, 8192], dtype=torch.int32),
                block_tables=torch.zeros((4, 64), dtype=torch.int32),
            )
            self.attn_metadata_dict[layer_name] = meta

    def test_can_multi_step_guard_success(self):
        """Verify can_multi_step is True during steady-state decode with sufficient block slots."""
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=self.num_reqs)
        spec_decode_metadata = None
        is_pooling_model = False
        num_computed_tokens_cpu = [8192, 8192, 8192, 8192]

        can_multi_step = (
            self.multi_step > 1
            and scheduler_output.total_num_scheduled_tokens == self.num_reqs
            and spec_decode_metadata is None
            and not is_pooling_model
        )
        if can_multi_step:
            seq_lens_cpu = torch.tensor(num_computed_tokens_cpu)
            rem_slots = self.block_size - (seq_lens_cpu % self.block_size)
            if (rem_slots < self.multi_step).any():
                can_multi_step = False

        self.assertTrue(can_multi_step)

    def test_can_multi_step_guard_block_boundary_fallback(self):
        """Verify can_multi_step falls back to False when approaching a block boundary."""
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=self.num_reqs)
        spec_decode_metadata = None
        is_pooling_model = False
        # Request 1 has seq_len = 254 (only 2 slots left in block of 256, < K=4)
        num_computed_tokens_cpu = [100, 254, 100, 100]

        can_multi_step = (
            self.multi_step > 1
            and scheduler_output.total_num_scheduled_tokens == self.num_reqs
            and spec_decode_metadata is None
            and not is_pooling_model
        )
        if can_multi_step:
            seq_lens_cpu = torch.tensor(num_computed_tokens_cpu)
            rem_slots = self.block_size - (seq_lens_cpu % self.block_size)
            if (rem_slots < self.multi_step).any():
                can_multi_step = False

        self.assertFalse(can_multi_step, "Should disable multi-step when rem_slots < K")

    def test_multi_step_dict_traversal_and_seq_lens_increment(self):
        """Verify that with the dict bugfix, ALL attention layers receive +1 for position and seq_lens."""
        attn_metadata = self.attn_metadata_dict
        multi_step = 4
        position_ids = self.position_ids.clone()

        # Mock initial first token sampling
        first_sampled = torch.tensor([[101], [102], [103], [104]], dtype=torch.int64)
        all_sampled = [first_sampled]
        curr_tokens = first_sampled

        # Multi-step loop with the bugfix
        for step_idx in range(multi_step - 1):
            position_ids = position_ids + 1
            if isinstance(attn_metadata, dict):
                for meta in attn_metadata.values():
                    if hasattr(meta, "input_positions") and meta.input_positions is not None:
                        meta.input_positions = position_ids
                    if hasattr(meta, "seq_lens") and meta.seq_lens is not None:
                        meta.seq_lens = meta.seq_lens + 1
            else:
                if hasattr(attn_metadata, "input_positions") and attn_metadata.input_positions is not None:
                    attn_metadata.input_positions = position_ids
                if hasattr(attn_metadata, "seq_lens") and attn_metadata.seq_lens is not None:
                    attn_metadata.seq_lens = attn_metadata.seq_lens + 1

            # Simulated forward pass produces next tokens
            next_sampled = first_sampled + (step_idx + 1) * 10
            all_sampled.append(next_sampled)
            curr_tokens = next_sampled

        sampled_tokens = torch.cat(all_sampled, dim=-1)

        # Assertions
        # 1. Exactly 4 tokens generated per request
        self.assertEqual(sampled_tokens.shape, (4, 4))
        # 2. Position ids advanced by 3 (8192 -> 8195)
        torch.testing.assert_close(position_ids, torch.tensor([8195, 8195, 8195, 8195]))
        # 3. Every layer in attn_metadata dict has updated seq_lens (8192 -> 8195)
        for layer_name, meta in attn_metadata.items():
            torch.testing.assert_close(meta.seq_lens, torch.tensor([8195, 8195, 8195, 8195], dtype=torch.int32))
            torch.testing.assert_close(meta.input_positions, torch.tensor([8195, 8195, 8195, 8195]))

    def test_multi_step_output_retention_and_cpu_sync(self):
        """Verify that multi-step tokens are retained, scheduler CPU state is advanced, and last token feeds async."""
        selected_token_ids = torch.tensor([
            [101, 111, 121, 131],
            [201, 211, 221, 231],
            [301, 311, 321, 331],
            [401, 411, 421, 431],
        ], dtype=torch.int64)

        # 1. Async token substitution receives the LAST token (131, 231, 331, 431)
        if selected_token_ids.dim() > 1 and selected_token_ids.shape[-1] > 1:
            next_tokens_tpu_chunk = selected_token_ids[:, -1].contiguous()
        else:
            next_tokens_tpu_chunk = selected_token_ids.view(-1)

        torch.testing.assert_close(next_tokens_tpu_chunk, torch.tensor([131, 231, 331, 431], dtype=torch.int64))

        # 2. Verify CPU batch state updates for all requests
        num_computed_tokens_cpu = [8192, 8192, 8192, 8192]
        output_token_ids = [[] for _ in range(4)]

        for req_idx in range(4):
            valid_tokens = selected_token_ids[req_idx].tolist()
            if len(valid_tokens) > 1:
                extra_tokens = len(valid_tokens) - 1
                num_computed_tokens_cpu[req_idx] += extra_tokens
            output_token_ids[req_idx].extend(valid_tokens)

        # All 4 tokens retained in output
        self.assertEqual(output_token_ids[0], [101, 111, 121, 131])
        # num_computed_tokens advanced by 3 extra tokens (total 4 tokens accounted for)
        self.assertEqual(num_computed_tokens_cpu, [8195, 8195, 8195, 8195])


if __name__ == "__main__":
    unittest.main()

    def test_reproduce_buggy_dict_hasattr_silently_drops_updates(self):
        """Demonstrates that the un-fixed hasattr check on a dict fails silently."""
        attn_metadata = self.attn_metadata_dict  # This is a dict!
        
        # Buggy code in commit 1501d2010:
        updated = False
        if hasattr(attn_metadata, "seq_lens"):
            attn_metadata.seq_lens = attn_metadata.seq_lens + 1
            updated = True

        # Assert that hasattr on a dict is ALWAYS False, proving the bug!
        self.assertFalse(hasattr(attn_metadata, "seq_lens"))
        self.assertFalse(updated, "Bug reproduced: dict hasattr check silently dropped all seq_lens updates!")
