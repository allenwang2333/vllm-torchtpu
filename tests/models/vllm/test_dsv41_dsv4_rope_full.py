# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch unit tests for DeepSeek-V4/V4.1 RoPE kernels."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
sys.modules.setdefault("vllm_torchtpu.layers.core.quantization", _quant_mod)

from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_mod  # noqa: E402


class _MockArrayRef:
    def __init__(self, arr: jax.Array) -> None:
        self.arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype

    def __getitem__(self, idx) -> jax.Array:
        if idx is Ellipsis:
            return self.arr
        return self.arr[idx]

    def __setitem__(self, idx, val: jax.Array) -> None:
        if idx is Ellipsis:
            self.arr = val
        else:
            self.arr = self.arr.at[idx].set(val)

    @property
    def at(self) -> _MockArrayRef:
        return self


def test_cos_sin_lanes_rotate_gptj_and_split_cos_sin_math() -> None:
    rng = np.random.default_rng(123)
    cos_sin_raw = jnp.asarray(rng.standard_normal((8, 64), dtype=np.float32))

    cos_fwd, sin_fwd = rope_mod._cos_sin_lanes(
        cos_sin_raw, rotary_dim=64, inverse=False, premultiply_rotate_sign=False
    )
    cos_inv, sin_inv = rope_mod._cos_sin_lanes(
        cos_sin_raw, rotary_dim=64, inverse=True, premultiply_rotate_sign=False
    )
    cos_pre, sin_pre = rope_mod._cos_sin_lanes(
        cos_sin_raw, rotary_dim=64, inverse=False, premultiply_rotate_sign=True
    )

    assert cos_fwd.shape == (8, 128) and sin_fwd.shape == (8, 128)
    np.testing.assert_allclose(np.asarray(cos_fwd[:, :64]), 1.0)
    np.testing.assert_allclose(np.asarray(sin_fwd[:, :64]), 0.0)
    np.testing.assert_allclose(np.asarray(cos_fwd), np.asarray(cos_inv))
    np.testing.assert_allclose(np.asarray(cos_fwd), np.asarray(cos_pre))
    np.testing.assert_allclose(
        np.asarray(sin_fwd[:, 64:]), -np.asarray(sin_inv[:, 64:])
    )

    # Check _rotate_gptj both with and without sign application
    x_2d = jnp.asarray(rng.standard_normal((4, 128), dtype=np.float32))
    rot_signed = rope_mod._rotate_gptj(x_2d, apply_sign=True)
    rot_unsigned = rope_mod._rotate_gptj(x_2d, apply_sign=False)
    ref_rot = jnp.stack([-x_2d[:, 1::2], x_2d[:, 0::2]], axis=-1).reshape(4, 128)
    ref_perm = jnp.stack([x_2d[:, 1::2], x_2d[:, 0::2]], axis=-1).reshape(4, 128)
    np.testing.assert_array_equal(np.asarray(rot_signed), np.asarray(ref_rot))
    np.testing.assert_array_equal(np.asarray(rot_unsigned), np.asarray(ref_perm))

    # Verify bitwise identity of 3D sign-folding optimization:
    # (partner * sign) * sin == partner * (sin * sign)
    x_3d = jnp.asarray(rng.standard_normal((8, 16, 128), dtype=np.float32))
    table_ref = jnp.concatenate([cos_fwd, sin_fwd], axis=-1)
    table_opt = jnp.concatenate([cos_pre, sin_pre], axis=-1)
    c2, s2 = rope_mod._split_cos_sin(table_ref, 2)
    assert c2.shape == (8, 128) and s2.shape == (8, 128)

    c3_ref, s3_ref = rope_mod._split_cos_sin(table_ref, 3)
    c3_opt, s3_opt = rope_mod._split_cos_sin(table_opt, 3)
    y_ref = x_3d * c3_ref + rope_mod._rotate_gptj(x_3d, apply_sign=True) * s3_ref
    y_opt = x_3d * c3_opt + rope_mod._rotate_gptj(x_3d, apply_sign=False) * s3_opt
    np.testing.assert_array_equal(np.asarray(y_ref), np.asarray(y_opt))


def test_rope_bodies_2d_3d_quant_and_qnorm_branches() -> None:
    rng = np.random.default_rng(456)
    cos_sin_raw = jnp.asarray(rng.standard_normal((4, 64), dtype=np.float32))
    cos_sin_out = _MockArrayRef(jnp.zeros((4, 256), dtype=jnp.float32))
    rope_mod._cos_sin_body(
        _MockArrayRef(cos_sin_raw),
        cos_sin_out,
        rotary_dim=64,
        inverse=False,
        out_dtype=jnp.float32,
    )

    # 2D in-place _rope_body
    x_2d_init = jnp.asarray(
        rng.standard_normal((4, 128), dtype=np.float32), dtype=jnp.bfloat16
    )
    x_2d_ref = _MockArrayRef(x_2d_init)
    rope_mod._rope_body(
        x_2d_ref,
        cos_sin_out,
        out_dtype=jnp.bfloat16,
        quant_dtype=None,
    )
    assert x_2d_ref.arr.dtype == jnp.bfloat16 and x_2d_ref.arr.shape == (4, 128)

    # 3D quantized _rope_body (including zero row for scale_inv == 0 branch)
    x_3d_init = (
        jnp.asarray(
            rng.standard_normal((4, 2, 128), dtype=np.float32), dtype=jnp.bfloat16
        )
        .at[0, 0]
        .set(0.0)
    )
    q_out = _MockArrayRef(jnp.zeros((4, 2, 128), dtype=jnp.float8_e4m3fn))
    s_out = _MockArrayRef(jnp.zeros((4, 2), dtype=jnp.float32))
    rope_mod._rope_body(
        _MockArrayRef(x_3d_init),
        cos_sin_out,
        q_out,
        s_out,
        out_dtype=jnp.bfloat16,
        quant_dtype=jnp.float8_e4m3fn,
    )
    assert float(s_out.arr[0, 0]) == 0.0

    # _qnorm_rope_body with head_dim == LANE (128) and head_dim == 256 (> LANE)
    for hd in (128, 256):
        qn_in = _MockArrayRef(
            jnp.asarray(
                rng.standard_normal((4, 2, hd), dtype=np.float32), dtype=jnp.bfloat16
            )
        )
        qn_out = _MockArrayRef(jnp.zeros((4, 2, hd), dtype=jnp.bfloat16))
        rope_mod._qnorm_rope_body(
            qn_in,
            cos_sin_out,
            qn_out,
            eps=1e-6,
            out_dtype=jnp.bfloat16,
        )
        assert qn_out.arr.shape == (4, 2, hd) and qn_out.arr.dtype == jnp.bfloat16


def test_cos_sin_ref_and_pallas_kernels_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise CosSinRef.create, copy_in, and wait_in
    dummy_fields: dict[str, object] = {
        name: _MockArrayRef(jnp.zeros((2, 4, 64), dtype=jnp.float32))
        for name in rope_mod._BUFFERED_REF_FIELD_NAMES
    }
    dummy_fields["_buffer_count"] = 2
    dummy_fields["_buffer_type"] = pltpu.BufferType.INPUT
    dummy_fields["copy_in_slot"] = jnp.uint32(0)
    dummy_fields["wait_in_slot"] = jnp.uint32(0)

    monkeypatch.setattr(
        pltpu.BufferedRef,
        "create",
        classmethod(lambda cls, **kw: SimpleNamespace(**dummy_fields)),
    )
    monkeypatch.setattr(
        pltpu.BufferedRef,
        "input",
        classmethod(
            lambda cls, spec, dtype, buffer_count=2: SimpleNamespace(spec=spec)
        ),
    )
    monkeypatch.setattr(
        pltpu.BufferedRef,
        "output",
        classmethod(
            lambda cls, spec, dtype, buffer_count=2: SimpleNamespace(spec=spec)
        ),
    )
    monkeypatch.setattr(
        pltpu.BufferedRef,
        "input_output",
        classmethod(
            lambda cls, spec, dtype, buffer_count=2: SimpleNamespace(spec=spec)
        ),
    )

    copy_events: list[str] = []

    class _FakeAsyncCopy:
        def start(self) -> None:
            copy_events.append("start")

        def wait(self) -> None:
            copy_events.append("wait")

    monkeypatch.setattr(
        pltpu, "make_async_copy", lambda src, dst, sem: _FakeAsyncCopy()
    )
    monkeypatch.setattr(
        pl,
        "loop",
        lambda lower, upper, unroll=False: lambda body_fn: [
            body_fn(i) for i in range(lower, upper)
        ],
    )
    monkeypatch.setattr(
        pl,
        "with_scoped",
        lambda allocations: lambda run_fn: lambda: run_fn(allocations),
    )

    cs_ref = rope_mod.CosSinRef.create(
        spec=pl.BlockSpec(
            block_shape=(2, 64), memory_space=pltpu.VMEM, index_map=lambda i: (i, 0)
        ),
        dtype=jnp.float32,
        buffer_count=2,
        tile_n=2,
    )
    cs_cache_ref = _MockArrayRef(jnp.ones((16, 64), dtype=jnp.float32))
    pos_ref = _MockArrayRef(jnp.array([1, 3], dtype=jnp.int32))
    cs_ref.copy_in((cs_cache_ref, pos_ref), (0,))
    cs_ref.wait_in((cs_cache_ref, pos_ref), (0,))
    assert copy_events == ["start", "start", "wait"]

    def _fake_emit_pipeline(body, *, grid, in_specs, out_specs):
        # Exercise all BlockSpec index_map callables
        for spec in (*in_specs, *out_specs):
            if spec.index_map is not None:
                spec.index_map(0)

        def _runner(*args, allocations=None):
            del args, allocations

        return _runner

    monkeypatch.setattr(pltpu, "emit_pipeline", _fake_emit_pipeline)

    def _fake_pallas_call(kernel_fn, *, out_shape, **kwargs):
        del kwargs

        def _invoker(*args):
            if isinstance(out_shape, tuple):
                outs = tuple(jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape)
                kernel_fn(*args, *outs)
                return outs
            out = jnp.zeros(out_shape.shape, dtype=out_shape.dtype)
            kernel_fn(*args, out)
            return out

        return _invoker

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)

    positions = jnp.arange(8, dtype=jnp.int32)
    cos_sin_cache = jnp.ones((32, 64), dtype=jnp.float32)

    # 2D rope, 3D rope (<=16 and >16 heads), qnorm_rope, and rope_quant
    out_2d = rope_mod.rope(
        jnp.ones((8, 128), dtype=jnp.bfloat16), positions, cos_sin_cache
    )
    assert out_2d.shape == (8, 128)

    out_3d_small = rope_mod.rope(
        jnp.ones((8, 16, 256), dtype=jnp.bfloat16),
        positions,
        cos_sin_cache,
        inverse=True,
    )
    assert out_3d_small.shape == (8, 16, 256)

    out_3d_large = rope_mod.rope(
        jnp.ones((8, 32, 256), dtype=jnp.bfloat16), positions, cos_sin_cache
    )
    assert out_3d_large.shape == (8, 32, 256)

    qn_small = rope_mod.qnorm_rope(
        jnp.ones((8, 16, 256), dtype=jnp.bfloat16), positions, cos_sin_cache
    )
    assert qn_small.shape == (8, 16, 256)

    qn_large = rope_mod.qnorm_rope(
        jnp.ones((8, 64, 256), dtype=jnp.bfloat16), positions, cos_sin_cache
    )
    assert qn_large.shape == (8, 64, 256)

    rq, rs = rope_mod.rope_quant(
        jnp.ones((8, 16, 128), dtype=jnp.bfloat16), positions, cos_sin_cache
    )
    assert rq.shape == (8, 16, 128) and rs.shape == (8, 16)

    # Error validation branches in _rope_call
    with pytest.raises(ValueError, match="x must be rank 2 or 3"):
        rope_mod._rope_call(
            jnp.ones((8,), dtype=jnp.bfloat16),
            positions,
            cos_sin_cache,
            inverse=False,
            quant_dtype=None,
            name="bad_rank",
        )

    with pytest.raises(ValueError, match="quantization requires head_dim == 128"):
        rope_mod._rope_call(
            jnp.ones((8, 16, 256), dtype=jnp.bfloat16),
            positions,
            cos_sin_cache,
            inverse=False,
            quant_dtype=jnp.float8_e4m3fn,
            name="bad_quant_hd",
        )
