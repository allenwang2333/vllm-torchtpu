# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% full-module coverage tests for Track TD7.26:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.ops
- vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence
- vllm_torchtpu.models.vllm.deepseek_v41.v2.engram
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import async_fence as v2_fence
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops


def test_async_fence_install_and_execution(monkeypatch: pytest.MonkeyPatch):
    from vllm_torchtpu.runner import tpu_runner

    called_original: list[int] = []

    def dummy_original(runner_self, num_reqs: int):
        called_original.append(num_reqs)
        return torch.tensor([1, 2, 3])

    monkeypatch.setattr(
        tpu_runner.TPUModelRunner,
        "_prepare_lookback_token_ids",
        dummy_original,
        raising=False,
    )

    v2_fence.install_lookback_fence()
    patched_fn = tpu_runner.TPUModelRunner._prepare_lookback_token_ids
    assert getattr(patched_fn, "_deepseek_v41_fenced", False) is True

    # Idempotent second install
    v2_fence.install_lookback_fence()
    assert tpu_runner.TPUModelRunner._prepare_lookback_token_ids is patched_fn

    runner = SimpleNamespace()
    mock_event = MagicMock()
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: mock_event), raising=False
    )

    res1 = patched_fn(runner, 4)
    assert torch.equal(res1, torch.tensor([1, 2, 3]))
    assert called_original == [4]
    mock_event.record.assert_called_once()
    assert getattr(runner, "_deepseek_v41_lookback_fence", None) is mock_event

    mock_event.reset_mock()
    mock_event2 = MagicMock()
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: mock_event2), raising=False
    )
    res2 = patched_fn(runner, 2)
    assert torch.equal(res2, torch.tensor([1, 2, 3]))
    assert called_original == [4, 2]
    mock_event.synchronize.assert_called_once()
    mock_event2.record.assert_called_once()
    assert runner._deepseek_v41_lookback_fence is mock_event2


def test_v2_engram_ngram_hash_state_write_cache(monkeypatch: pytest.MonkeyPatch):
    written: list[tuple[object, object, object, object]] = []
    monkeypatch.setattr(
        v2_engram,
        "write_cache",
        lambda cache, slots, values, valid: written.append(
            (cache, slots, values, valid)
        ),
    )
    state = object.__new__(v2_engram.NgramHashState)
    cache = torch.zeros((4, 8))
    slots = torch.tensor([0, 1])
    values = torch.ones((2, 8))
    valid = torch.tensor([True, False])
    state.write_cache(cache, slots, values, valid)
    assert len(written) == 1
    assert written[0] == (cache, slots, values, valid)


def test_ops_device_hooks_and_bind_device_ops():
    mod = SimpleNamespace(alpha=10, beta="hello")
    with v2_ops.device_hooks(mod, alpha=99, beta="world"):
        assert mod.alpha == 99
        assert mod.beta == "world"
    assert mod.alpha == 10
    assert mod.beta == "hello"

    with (
        pytest.raises(RuntimeError, match="test_err"),
        v2_ops.device_hooks(mod, alpha=100),
    ):
        assert mod.alpha == 100
        raise RuntimeError("test_err")
    assert mod.alpha == 10

    def sample_fn(x, factor=2):
        return custom_add(x * factor, offset)  # noqa: F821

    bound = v2_ops.bind_device_ops(sample_fn, custom_add=lambda a, b: a + b, offset=5)
    assert bound(10) == 25
    assert bound(10, factor=3) == 35


def test_ops_compact_rope_caches():
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        v2_ops.compact_rope_caches([], 0)

    shared_cache = torch.randn(100, 16)
    att1 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_cache))
    att2 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_cache))
    small_cache = torch.randn(30, 16)
    att3 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=small_cache))

    v2_ops.compact_rope_caches([att1, att2, att3], max_model_len=50)
    assert att1._tpu_rope_cache.shape == (50, 16)
    assert att1._tpu_rope_cache is att2._tpu_rope_cache
    assert torch.equal(att1._tpu_rope_cache, shared_cache[:50])
    assert att3._tpu_rope_cache is small_cache


def test_ops_rotary_all_branches():
    x_3d = torch.randn(1, 2, 16, dtype=torch.bfloat16)
    pos_i64 = torch.tensor([0, 1], dtype=torch.int64)
    pos_f32 = torch.tensor([0.0, 1.0], dtype=torch.float32)

    for bad_dim in (1, 7, 64):
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v2_ops.rotary(
                x_3d,
                pos_i64,
                {"qk_rope_head_dim": bad_dim, "rope_theta": 10000.0},
                compressed=False,
            )

    cfg = {"qk_rope_head_dim": 8, "rope_theta": 10000.0}
    out_bf16 = v2_ops.rotary(x_3d, pos_i64, cfg, compressed=False, inverse=False)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16[..., :-8], x_3d[..., :-8])

    # Float32 x and float32 positions fast path (no redundant casts)
    x_f32 = x_3d.float()
    out_f32 = v2_ops.rotary(x_f32, pos_f32, cfg, compressed=False, inverse=False)
    assert out_f32.dtype == torch.float32
    inv_f32 = v2_ops.rotary(out_f32, pos_f32, cfg, compressed=False, inverse=True)
    assert torch.allclose(inv_f32, x_f32, atol=1e-5)

    # Full-width rotary (dim == x.shape[-1]) skips torch.cat with empty prefix
    cfg_full = {"qk_rope_head_dim": 16, "rope_theta": 10000.0}
    out_full = v2_ops.rotary(x_f32, pos_f32, cfg_full, compressed=False, inverse=False)
    assert out_full.shape == x_f32.shape
    inv_full = v2_ops.rotary(
        out_full, pos_f32, cfg_full, compressed=False, inverse=True
    )
    assert torch.allclose(inv_full, x_f32, atol=1e-5)

    # Compressed with YaRN (rope_scaling, rope_parameters, and <=0 original len)
    for key in ("rope_scaling", "rope_parameters"):
        cfg_yarn = {
            "qk_rope_head_dim": 8,
            "compress_rope_theta": 50000.0,
            key: {
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 2.0,
            },
        }
        out_y = v2_ops.rotary(x_f32, pos_f32, cfg_yarn, compressed=True)
        assert out_y.shape == x_f32.shape

    cfg_no_corr = {
        "qk_rope_head_dim": 8,
        "compress_rope_theta": 50000.0,
        "rope_scaling": {"original_max_position_embeddings": 0},
    }
    assert v2_ops.rotary(x_f32, pos_f32, cfg_no_corr, compressed=True).shape == (
        1,
        2,
        16,
    )


def test_ops_round_fp8_and_rms_norm_dtypes_and_parity():
    with pytest.raises(
        ValueError, match="FP8 block size must divide contraction dimension"
    ):
        v2_ops.round_fp8_activation(torch.randn(2, 40), block_size=32)

    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(3, 64, dtype=dtype)
        rounded = v2_ops.round_fp8_activation(x, block_size=32)
        assert rounded.shape == x.shape
        assert rounded.dtype == dtype

    torch.manual_seed(0)
    for x_dtype in (torch.bfloat16, torch.float32):
        for w_dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(4, 16, dtype=x_dtype)
            w = torch.randn(16, dtype=w_dtype)
            eps = 1e-6
            ref = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                * w.float()
            ).to(x_dtype)
            got = v2_ops.rms_norm(x, w, eps)
            assert got.dtype == x_dtype
            assert torch.equal(got, ref)
