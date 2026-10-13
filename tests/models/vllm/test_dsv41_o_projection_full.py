# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch unit tests for DeepSeek-V4/V4.1 o_projection kernel."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"


def _largest_divisor(x: int, cap: int) -> int:
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


@functools.cache
def _dtype_max(dtype: jnp.dtype) -> float:
    info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)
    return float(info.max)


for _pkg_name, _pkg_dir in (
    ("vllm_torchtpu", _SRC_ROOT),
    ("vllm_torchtpu.kernels", _SRC_ROOT / "kernels"),
    ("vllm_torchtpu.kernels.deepseek_v4", _SRC_ROOT / "kernels" / "deepseek_v4"),
    ("vllm_torchtpu.layers", _SRC_ROOT / "layers"),
    ("vllm_torchtpu.layers.core", _SRC_ROOT / "layers" / "core"),
):
    _pkg = types.ModuleType(_pkg_name)
    _pkg.__path__ = [str(_pkg_dir)]
    sys.modules.setdefault(_pkg_name, _pkg)

_utils_mod = types.ModuleType("vllm_torchtpu.utils")
_utils_mod.largest_divisor = _largest_divisor  # type: ignore[attr-defined]
sys.modules.setdefault("vllm_torchtpu.utils", _utils_mod)

_quant_mod = types.ModuleType("vllm_torchtpu.layers.core.quantization")
_quant_mod.dtype_max = _dtype_max  # type: ignore[attr-defined]
_quant_mod.FP8_E4M3_MAX = _dtype_max(jnp.float8_e4m3fn)  # type: ignore[attr-defined]
sys.modules.setdefault("vllm_torchtpu.layers.core.quantization", _quant_mod)

from vllm_torchtpu.kernels.deepseek_v4 import o_projection as op  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.rope import (  # noqa: E402
    LANE,
    _rotate_gptj,
)


class _MockRef:
    def __init__(self, arr: jax.Array) -> None:
        self.arr = arr

    @property
    def dtype(self) -> jnp.dtype:
        return self.arr.dtype

    def __getitem__(self, idx: object) -> jax.Array:
        if idx is Ellipsis:
            return self.arr
        return self.arr[idx]  # type: ignore[index]

    def __setitem__(self, idx: object, val: jax.Array) -> None:
        if idx is Ellipsis:
            self.arr = jnp.asarray(val, dtype=self.arr.dtype)
        else:
            self.arr = self.arr.at[idx].set(jnp.asarray(val, dtype=self.arr.dtype))  # type: ignore[index]


def test_largest_divisor() -> None:
    assert op._largest_divisor(128, cap=128) == 128
    assert op._largest_divisor(100, cap=16) == 10
    assert op._largest_divisor(7, cap=6) == 1
    assert op._largest_divisor(0, cap=10) == 1
    assert op._largest_divisor(10, cap=0) == 1


def test_rope_heads_parity() -> None:
    rng = np.random.default_rng(0)
    for head_dim in (128, 256, 512):
        x = jnp.asarray(rng.standard_normal((8, 8, head_dim)), dtype=jnp.bfloat16)
        cs = jnp.asarray(rng.standard_normal((8, 2 * LANE)), dtype=jnp.float32)
        out = op._rope_heads(x, cs, head_dim=head_dim)
        lo = head_dim - LANE
        tail = x[:, :, lo:].astype(jnp.float32)
        expected_tail = (
            tail * cs[:, :LANE][:, None, :]
            + _rotate_gptj(tail) * cs[:, LANE:][:, None, :]
        ).astype(x.dtype)
        expected = (
            expected_tail
            if lo == 0
            else jnp.concatenate([x[:, :, :lo], expected_tail], axis=-1)
        )
        assert bool(jnp.array_equal(out, expected))


def test_kernel_single_and_multi_subchunk() -> None:
    rng = np.random.default_rng(1)
    h_per_group = 8
    tile_r = 16
    for head_dim in (128, 256):
        d_dim = h_per_group * head_dim
        w_arr = jnp.asarray(rng.standard_normal((d_dim, tile_r)), dtype=jnp.float32)
        scale_arr = jnp.asarray(
            rng.uniform(0.5, 1.5, size=(1, tile_r)), dtype=jnp.float32
        )
        for tile_t, num_sub_t in ((4, 1), (8, 2), (16, 8)):
            x_arr = jnp.asarray(
                rng.standard_normal((tile_t, h_per_group, head_dim)), dtype=jnp.float32
            )
            cs_arr = jnp.asarray(
                rng.standard_normal((tile_t, 2 * LANE)), dtype=jnp.float32
            )
            for quant in (True, False):
                out_ref = _MockRef(jnp.zeros((tile_t, tile_r), dtype=jnp.float32))
                op._kernel(
                    _MockRef(x_arr),  # type: ignore[arg-type]
                    _MockRef(w_arr),  # type: ignore[arg-type]
                    _MockRef(scale_arr),  # type: ignore[arg-type]
                    _MockRef(cs_arr),  # type: ignore[arg-type]
                    out_ref,  # type: ignore[arg-type]
                    tile_t=tile_t,
                    num_sub_t=num_sub_t,
                    quantize_activations=quant,
                    head_dim=head_dim,
                )
                assert out_ref.arr.shape == (tile_t, tile_r)
                assert bool(jnp.all(jnp.isfinite(out_ref.arr)))


def test_wo_a_projection_assertions_and_execution() -> None:
    num_tokens = 16
    num_groups = 2
    h_per_group = 8
    head_dim = 128
    d_dim = h_per_group * head_dim
    lora_rank = 32
    out_features = num_groups * lora_rank

    valid_x = jnp.zeros(
        (num_tokens, num_groups * h_per_group, head_dim), dtype=jnp.bfloat16
    )
    valid_w = jnp.zeros((d_dim, out_features), dtype=jnp.int8)
    valid_scale = jnp.zeros((out_features,), dtype=jnp.float32)
    valid_cs = jnp.zeros((num_tokens, 2 * LANE), dtype=jnp.float32)

    with pytest.raises(AssertionError):
        op.wo_a_projection(
            jnp.zeros((num_tokens, num_groups * h_per_group), dtype=jnp.bfloat16),
            valid_w,
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            jnp.zeros(
                (num_tokens, num_groups * h_per_group, head_dim), dtype=jnp.float32
            ),
            valid_w,
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            jnp.zeros((d_dim, out_features, 1), dtype=jnp.int8),
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            valid_w,
            jnp.zeros((out_features, 1), dtype=jnp.float32),
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            valid_w,
            jnp.zeros((out_features + 1,), dtype=jnp.float32),
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            jnp.zeros((d_dim + 1, out_features), dtype=jnp.int8),
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            jnp.zeros((4 * head_dim, out_features), dtype=jnp.int8),
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            jnp.zeros((num_tokens, 15, head_dim), dtype=jnp.bfloat16),
            valid_w,
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            jnp.zeros((d_dim, 65), dtype=jnp.int8),
            jnp.zeros((65,), dtype=jnp.float32),
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            valid_x,
            valid_w,
            valid_scale,
            jnp.zeros((num_tokens, LANE), dtype=jnp.float32),
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(
            jnp.zeros((num_tokens, num_groups * h_per_group, 100), dtype=jnp.bfloat16),
            jnp.zeros((h_per_group * 100, out_features), dtype=jnp.int8),
            valid_scale,
            valid_cs,
        )
    with pytest.raises(AssertionError):
        op.wo_a_projection(valid_x, valid_w, valid_scale, valid_cs, tile_r=7)
    with pytest.raises(AssertionError):
        op.wo_a_projection(valid_x, valid_w, valid_scale, valid_cs, tile_t=5)
    with pytest.raises(AssertionError):
        op.wo_a_projection(valid_x, valid_w, valid_scale, valid_cs, tile_t=16, sub_t=6)

    captured: dict[str, object] = {}

    def _fake_pallas_call(
        kernel: object,
        out_shape: jax.ShapeDtypeStruct,
        grid: tuple[int, ...],
        in_specs: list[object],
        out_specs: object,
        compiler_params: object,
        name: str,
    ) -> object:
        captured["grid"] = grid
        captured["in_specs"] = in_specs
        captured["out_specs"] = out_specs

        def _runner(*_args: object) -> jax.Array:
            return jnp.ones(out_shape.shape, dtype=out_shape.dtype)

        return _runner

    with patch.object(op.pl, "pallas_call", side_effect=_fake_pallas_call):
        res_default = op.wo_a_projection.__wrapped__(
            valid_x, valid_w, valid_scale, valid_cs
        )
        assert res_default.shape == (num_tokens, out_features)
        assert captured["grid"] == (num_groups, 1, 1)
        in_specs = captured["in_specs"]
        out_specs = captured["out_specs"]
        assert in_specs[0].index_map(1, 2, 3) == (2, 1, 0)  # type: ignore[index]
        assert in_specs[1].index_map(1, 2, 0) == (0, 1)  # type: ignore[index]
        assert in_specs[2].index_map(1, 2, 0) == (0, 1)  # type: ignore[index]
        assert in_specs[3].index_map(1, 2, 3) == (2, 0)  # type: ignore[index]
        assert out_specs.index_map(1, 2, 0) == (2, 1)  # type: ignore[attr-defined]

        res_custom = op.wo_a_projection.__wrapped__(
            valid_x,
            valid_w,
            valid_scale,
            valid_cs,
            tile_t=8,
            tile_r=16,
            sub_t=4,
            quantize_activations=False,
            name="custom_wo_a",
        )
        assert res_custom.shape == (num_tokens, out_features)
        assert captured["grid"] == (num_groups, 2, 2)


def test_fused_reverse_rope_wo_a_projection() -> None:
    num_tokens = 16
    num_groups = 2
    h_per_group = 8
    head_dim = 128
    d_dim = h_per_group * head_dim
    lora_rank = 32
    out_features = num_groups * lora_rank

    x = jnp.zeros((num_tokens, num_groups * h_per_group, head_dim), dtype=jnp.bfloat16)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    cache = jnp.zeros((64, 64), dtype=jnp.float32)
    w = jnp.zeros((d_dim, out_features), dtype=jnp.int8)
    scale = jnp.zeros((out_features,), dtype=jnp.float32)

    with pytest.raises(AssertionError):
        op.fused_reverse_rope_wo_a_projection(
            jnp.zeros((num_tokens, num_groups * h_per_group), dtype=jnp.bfloat16),
            positions,
            cache,
            w,
            scale,
            head_dim=head_dim,
        )
    with pytest.raises(AssertionError):
        op.fused_reverse_rope_wo_a_projection(
            x,
            positions,
            cache,
            w,
            scale,
            head_dim=head_dim + 16,
        )

    fake_cs = jnp.zeros((num_tokens, 2 * LANE), dtype=jnp.float32)
    fake_out = jnp.zeros((num_tokens, out_features), dtype=jnp.bfloat16)
    with (
        patch.object(op, "gather_cos_sin", return_value=fake_cs) as mock_gather,
        patch.object(op, "wo_a_projection", return_value=fake_out) as mock_proj,
    ):
        res = op.fused_reverse_rope_wo_a_projection.__wrapped__(
            x,
            positions,
            cache,
            w,
            scale,
            head_dim=head_dim,
            inverse=True,
            name="test_fused",
        )
        assert res.shape == (num_tokens, out_features)
        mock_gather.assert_called_once_with(
            positions,
            cache,
            inverse=True,
            name="test_fused_gather_cos_sin",
        )
        mock_proj.assert_called_once()
