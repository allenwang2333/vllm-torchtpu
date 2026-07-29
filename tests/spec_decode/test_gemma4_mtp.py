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

from types import SimpleNamespace
from unittest import mock

import torch

from vllm.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    CompilationConfig,
    set_current_vllm_config,
)
from vllm_torchtpu.models.vllm.gemma4_mtp import (
    Gemma4MTP,
    Gemma4MTPMaskedEmbedder,
    Gemma4MultiTokenPredictor,
    Gemma4MTPAttention,
)


def test_masked_embedder_forward():
    """Verify that the TPU-optimized Gemma4MTPMaskedEmbedder executes and masks correctly."""
    hidden_size = 64
    vocab_size = 200
    num_centroids = 10
    centroid_intermediate_top_k = 2

    # Initialize the embedder
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=centroid_intermediate_top_k,
    )

    # Initialize a linear ordering of tokens
    token_ordering = torch.arange(vocab_size, dtype=torch.long)
    embedder.token_ordering.copy_(token_ordering)

    # Dummy hidden states and embedding weights
    num_tokens = 4
    hidden_states = torch.randn(num_tokens, hidden_size)
    lm_head_weight = torch.randn(vocab_size, hidden_size)

    # Forward pass
    logits = embedder(hidden_states, lm_head_weight)

    # Assert shape: [num_tokens, vocab_size]
    assert logits.shape == (num_tokens, vocab_size)

    # Assert that non-selected centroids are masked to the minimum float value
    min_val = torch.finfo(logits.dtype).min
    num_masked = (logits == min_val).sum().item()
    
    # Each token selects centroid_intermediate_top_k centroids. Each centroid has (vocab_size / num_centroids) tokens.
    vocab_per_centroid = vocab_size // num_centroids
    num_selected = centroid_intermediate_top_k * vocab_per_centroid
    expected_masked = num_tokens * (vocab_size - num_selected)
    
    assert num_masked == expected_masked


def test_masked_embedder_get_top_tokens():
    """Verify that get_top_tokens correctly returns the sparse argmax tokens."""
    hidden_size = 32
    vocab_size = 100
    num_centroids = 5
    centroid_intermediate_top_k = 2

    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_centroids=num_centroids,
        centroid_intermediate_top_k=centroid_intermediate_top_k,
    )
    embedder.token_ordering.copy_(torch.arange(vocab_size, dtype=torch.long))

    num_tokens = 2
    hidden_states = torch.randn(num_tokens, hidden_size)
    lm_head_weight = torch.randn(vocab_size, hidden_size)

    top_tokens = embedder.get_top_tokens(hidden_states, lm_head_weight)

    # Assert shape: [num_tokens]
    assert top_tokens.shape == (num_tokens,)
    assert top_tokens.dtype == torch.long


def _get_vllm_test_config(hf_config):
    # Omit model and tokenizer to prevent HuggingFace network validation lookup.
    model_config = ModelConfig(
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_model_len=1024,
        is_encoder_decoder=False,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    
    draft_model_config = SimpleNamespace(hf_config=hf_config)
    draft_model_config.try_get_generation_config = lambda: None
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        method="mtp",
        compute_hash=lambda: "gemma4_mtp_test_hash",
        model="gemma4_mtp",
        draft_tensor_parallel_size=1,
        num_speculative_tokens=3,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        observability_config={},
        additional_config={},
    )
    vllm_config.speculative_config = speculative_config
    vllm_config.compilation_config = CompilationConfig(
        custom_ops=["none"],
        mode="VLLM_COMPILE",
    )
    return vllm_config


def test_gemma4_mtp_initialization():
    """Verify that Gemma4MTP model class can initialize correctly with structural properties."""
    # Build a minimal HF config for the draft model
    hf_config = SimpleNamespace(
        hidden_size=128,
        vocab_size=1000,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=512,
        attention_bias=False,
        rms_norm_eps=1e-6,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=128,
        rope_parameters={},
        tie_word_embeddings=True,
        use_ordered_embeddings=True,
        num_centroids=10,
        centroid_intermediate_top_k=2,
        backbone_hidden_size=256,
        intermediate_size=256,
        hidden_activation="gelu",
    )

    vllm_config = _get_vllm_test_config(hf_config)

    # Mock tensor parallel group coordinator
    fake_tp_group = SimpleNamespace(world_size=1, rank_in_group=0, all_reduce=lambda x: x)

    # Patch distributed and helper functions to prevent real GPU/TPU cluster checks
    with mock.patch("vllm.distributed.parallel_state.get_tp_group", return_value=fake_tp_group), \
         mock.patch("vllm_torchtpu.models.vllm.gemma4_mtp.get_tensor_model_parallel_world_size", return_value=1), \
         mock.patch("vllm_torchtpu.models.vllm.gemma4_mtp.get_draft_quant_config", return_value=None):
        
        with set_current_vllm_config(vllm_config):
            model = Gemma4MTP(vllm_config=vllm_config)
            
            # Verify module layout
            assert isinstance(model.model, Gemma4MultiTokenPredictor)
            assert model.model.hidden_size == 128
            assert model.model.backbone_hidden_size == 256
            assert len(model.model.layers) == 2
            
            # Verify sub-module classes
            assert isinstance(model.model.layers[0].self_attn, Gemma4MTPAttention)
            assert isinstance(model.masked_embedding, Gemma4MTPMaskedEmbedder)
