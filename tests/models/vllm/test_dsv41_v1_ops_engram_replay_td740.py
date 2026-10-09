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
"""Unit and 100% coverage tests for TD7.40 v1/ops, v1/engram, and v1/replay_plan."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v1 import engram as v1_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import ops as v1_ops
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import replay_plan as v1_replay


def test_v1_compact_rope_caches_and_rotary() -> None:
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        v1_ops.compact_rope_caches([], 0)

    large_table = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    small_table = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    attn1 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=large_table))
    attn2 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=large_table))
    attn3 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=small_table))

    v1_ops.compact_rope_caches([attn1, attn2, attn3], 4)
    assert attn1._tpu_rope_cache.shape == (4, 4)
    assert attn1._tpu_rope_cache is attn2._tpu_rope_cache
    assert attn3._tpu_rope_cache is small_table

    x_bf16 = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
    x_f32 = x_bf16.float()
    pos_i64 = torch.tensor([0, 1, 2], dtype=torch.int64)
    pos_f32 = pos_i64.float()

    with pytest.raises(ValueError, match="invalid rotary dimension"):
        v1_ops.rotary(x_bf16, pos_i64, {"qk_rope_head_dim": 3}, compressed=False)

    cfg = {
        "qk_rope_head_dim": 4,
        "rope_theta": 10000.0,
        "compress_rope_theta": 50000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 4.0,
        },
    }

    # Uncompressed + bfloat16 + int64 positions
    out_uncomp = v1_ops.rotary(x_bf16, pos_i64, cfg, compressed=False, inverse=False)
    assert out_uncomp.shape == x_bf16.shape
    assert out_uncomp.dtype == torch.bfloat16

    # Compressed + YaRN + float32 x + float32 positions + inverse=True
    out_comp_inv = v1_ops.rotary(x_f32, pos_f32, cfg, compressed=True, inverse=True)
    assert out_comp_inv.shape == x_f32.shape
    assert out_comp_inv.dtype == torch.float32


def test_v1_engram_ngram_hash_state(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_writes: list[tuple] = []
    monkeypatch.setattr(
        v1_engram,
        "write_cache",
        lambda c, s, v, val: recorded_writes.append((c, s, v, val)),
    )

    state = v1_engram.NgramHashState.__new__(v1_engram.NgramHashState)
    state._cache = None
    state._kv_cache_ref = None
    state.use_slot_cache = True
    state.block_size = 16

    c_t = torch.zeros((4, 8), dtype=torch.int32)
    s_t = torch.tensor([0, 1], dtype=torch.int32)
    v_t = torch.ones((2, 8), dtype=torch.int32)
    val_t = torch.tensor([True, False])
    state.write_cache(c_t, s_t, v_t, val_t)
    assert len(recorded_writes) == 1

    # 1. torch.compiler.is_compiling() == True delegates to super().ensure_cache()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        v1_engram.common.NgramHashState, "ensure_cache", lambda self: True
    )
    assert state.ensure_cache() is True

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    # 2. Empty bound_cache -> returns False and resets _cache / _kv_cache_ref
    state.swa_cache_module = SimpleNamespace(
        bound_cache=torch.empty((0, 16), dtype=torch.int32),
        block_size=16,
    )
    assert state.ensure_cache() is False
    assert state._cache is None
    assert state._kv_cache_ref is None

    # 3. Non-empty bound_cache with use_slot_cache == False -> returns True
    bound = torch.zeros((4, 16), dtype=torch.int32)
    state.swa_cache_module = SimpleNamespace(bound_cache=bound, block_size=16)
    state.use_slot_cache = False
    assert state.ensure_cache() is True

    # 4. use_slot_cache == True with matching block_size -> allocates _cache
    state.use_slot_cache = True
    assert state.ensure_cache() is True
    assert state._cache is not None and state._cache.shape == (64,)

    # 5. Second call with same bound_cache and block_size -> hits weakref fast path
    first_cache = state._cache
    assert state.ensure_cache() is True
    assert state._cache is first_cache

    # 6. Changing block_size invalidates _kv_cache_ref and reallocates _cache
    state.swa_cache_module.block_size = 32
    assert state.ensure_cache() is True
    assert state.block_size == 32
    assert state._cache is not None and state._cache.shape == (128,)


def test_v1_plan_decoder_replay_branches() -> None:
    # None metadata fields -> returns None
    meta_none = SimpleNamespace(
        query_start_loc_cpu=None, seq_lens_cpu=[2], prompt_lens_cpu=[2]
    )
    assert v1_replay.plan_decoder_replay(meta_none, 4, 8, (4, 8)) is None

    # Invalid window / mismatched lengths
    meta_bad_len = SimpleNamespace(
        query_start_loc_cpu=[0, 2], seq_lens_cpu=[2, 3], prompt_lens_cpu=[2]
    )
    with pytest.raises(ValueError, match="Invalid decoder replay scheduling"):
        v1_replay.plan_decoder_replay(meta_bad_len, 4, 8, (4, 8))

    # Query boundaries exceed input token bucket
    meta_oob = SimpleNamespace(
        query_start_loc_cpu=[0, 10], seq_lens_cpu=[10], prompt_lens_cpu=[10]
    )
    with pytest.raises(ValueError, match="exceed the input token bucket"):
        v1_replay.plan_decoder_replay(meta_oob, 4, 8, (4, 8))

    # Invalid scheduled query or prompt length
    meta_neg = SimpleNamespace(
        query_start_loc_cpu=[0, 2], seq_lens_cpu=[1], prompt_lens_cpu=[2]
    )
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt"):
        v1_replay.plan_decoder_replay(meta_neg, 4, 8, (4, 8))

    # All-kept fast path (skip == 0 for every request) with tuple compile_sizes
    meta_all_kept = SimpleNamespace(
        query_start_loc_cpu=[0, 2, 5],
        seq_lens_cpu=[2, 3],
        prompt_lens_cpu=[2, 3],
    )
    plan_kept = v1_replay.plan_decoder_replay(meta_all_kept, 4, 8, (4, 8, 16))
    assert plan_kept == v1_replay.DecoderReplayPlan(
        indices=(0, 1, 2, 3, 4),
        query_start_loc=(0, 2, 5),
        replay_starts=(0, 0),
        padded_num_tokens=8,
    )

    # Partial-skip path: first request kept (skip == 0), second request skips 2 tokens,
    # with list compile_sizes (exercises non-tuple compile_sizes conversion)
    meta_partial = SimpleNamespace(
        query_start_loc_cpu=[0, 2, 6],
        seq_lens_cpu=[2, 4],
        prompt_lens_cpu=[2, 4],
    )
    plan_partial = v1_replay.plan_decoder_replay(meta_partial, 2, 8, [4, 8])
    assert plan_partial == v1_replay.DecoderReplayPlan(
        indices=(0, 1, 4, 5),
        query_start_loc=(0, 2, 4),
        replay_starts=(0, 2),
        padded_num_tokens=4,
    )

    # Zero-kept path (kept == 0 -> padded == 0)
    meta_zero_kept = SimpleNamespace(
        query_start_loc_cpu=[0, 2],
        seq_lens_cpu=[2],
        prompt_lens_cpu=[6],
    )
    plan_zero = v1_replay.plan_decoder_replay(meta_zero_kept, 2, 8, (4, 8))
    assert plan_zero == v1_replay.DecoderReplayPlan(
        indices=(),
        query_start_loc=(0, 0),
        replay_starts=(4,),
        padded_num_tokens=0,
    )
