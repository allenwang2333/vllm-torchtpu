# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v1/ops.py."""

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class _StubModule(ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        return MagicMock()


for _mod_name in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.model_executor.models.config",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _StubModule(_mod_name)

v1_ops = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v1.ops")


def _ref_rotary(x, positions, config, *, compressed, inverse=False):
    dim = config["qk_rope_head_dim"]
    base = config["compress_rope_theta"] if compressed else config["rope_theta"]
    frequencies = 1 / base ** (
        torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    yarn = (
        (config.get("rope_scaling") or config.get("rope_parameters"))
        if compressed
        else None
    )
    if yarn and yarn["original_max_position_embeddings"] > 0:
        original = yarn["original_max_position_embeddings"]

        def correction(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(yarn["beta_fast"])), 0)
        high = min(math.ceil(correction(yarn["beta_slow"])), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=x.device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequencies = frequencies * (1 - ramp + ramp / yarn["factor"])
    pos_f32 = (
        positions
        if positions.device == x.device and positions.dtype == torch.float32
        else positions.to(device=x.device, dtype=torch.float32)
    )
    angles = pos_f32[:, None] * frequencies
    shape = (1, len(positions), *((1,) * (x.ndim - 3)), dim // 2)
    cosine, sine = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sine = -sine
    x_tail = x[..., -dim:] if x.dtype == torch.float32 else x[..., -dim:].float()
    pairs = x_tail.unflatten(-1, (-1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), -1
    ).flatten(-2)
    rotated_out = rotated if x.dtype == torch.float32 else rotated.to(x.dtype)
    return torch.cat((x[..., :-dim], rotated_out), -1)


def test_compact_rope_caches_validation_and_dedup() -> None:
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        v1_ops.compact_rope_caches([], 0)
    with pytest.raises(ValueError, match="max_model_len must be positive"):
        v1_ops.compact_rope_caches([], -4)

    shared_large = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    small = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    a1 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_large))
    a2 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_large))
    a3 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=small))
    v1_ops.compact_rope_caches([a1, a2, a3], 8)
    assert a1._tpu_rope_cache.shape == (8, 4)
    assert a1._tpu_rope_cache is a2._tpu_rope_cache
    assert a3._tpu_rope_cache is small


def test_rotary_branches_cache_and_bit_exactness() -> None:
    v1_ops._rotary_freq_cache.clear()
    x_f32 = torch.randn(1, 6, 4, 16, dtype=torch.float32)
    x_bf16 = x_f32.to(torch.bfloat16)
    pos_i64 = torch.arange(6, dtype=torch.int64)
    pos_f32 = pos_i64.float()

    with pytest.raises(ValueError, match="invalid rotary dimension"):
        v1_ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 0}, compressed=False)
    with pytest.raises(ValueError, match="invalid rotary dimension"):
        v1_ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 3}, compressed=False)
    with pytest.raises(ValueError, match="invalid rotary dimension"):
        v1_ops.rotary(x_f32, pos_f32, {"qk_rope_head_dim": 32}, compressed=False)

    # 1. Partial rotary dimension (dim < x.shape[-1]), uncompressed + cache hit:
    cfg_partial = {"qk_rope_head_dim": 8, "rope_theta": 10000.0}
    out_p1 = v1_ops.rotary(
        x_bf16, pos_i64, cfg_partial, compressed=False, inverse=False
    )
    out_p2 = v1_ops.rotary(
        x_bf16, pos_f32, cfg_partial, compressed=False, inverse=False
    )
    ref_p = _ref_rotary(x_bf16, pos_i64, cfg_partial, compressed=False, inverse=False)
    assert torch.equal(out_p1.view(torch.int16), ref_p.view(torch.int16))
    assert torch.equal(out_p2.view(torch.int16), ref_p.view(torch.int16))

    # 2. Full rotary dim (dim == x.shape[-1]), compressed YaRN + inverse + cache hit:
    cfg_full = {
        "qk_rope_head_dim": 16,
        "compress_rope_theta": 50000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 4.0,
        },
    }
    out_f1 = v1_ops.rotary(x_f32, pos_f32, cfg_full, compressed=True, inverse=True)
    out_f2 = v1_ops.rotary(x_f32, pos_f32, cfg_full, compressed=True, inverse=True)
    ref_f = _ref_rotary(x_f32, pos_f32, cfg_full, compressed=True, inverse=True)
    assert torch.equal(out_f1.view(torch.int32), ref_f.view(torch.int32))
    assert torch.equal(out_f2.view(torch.int32), ref_f.view(torch.int32))

    # 3. Compressed with original_max_position_embeddings <= 0 via rope_parameters:
    cfg_zero_yarn = {
        "qk_rope_head_dim": 8,
        "compress_rope_theta": 50000.0,
        "rope_parameters": {"original_max_position_embeddings": 0},
    }
    out_z = v1_ops.rotary(
        x_bf16, pos_i64, cfg_zero_yarn, compressed=True, inverse=False
    )
    ref_z = _ref_rotary(x_bf16, pos_i64, cfg_zero_yarn, compressed=True, inverse=False)
    assert torch.equal(out_z.view(torch.int16), ref_z.view(torch.int16))
