# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp MoE using the upstream Qwen3Next and TPU quantization paths."""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Keep the upstream MoE implementation and Qwen4Exp HC constraint."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
        if vllm_config.kv_transfer_config is not None:
            # Handoffs require the same routed sum on the producer and consumer.
            assert isinstance(self.experts, VllmMoERunner)
            self.experts.combine_dtype = torch.float32
