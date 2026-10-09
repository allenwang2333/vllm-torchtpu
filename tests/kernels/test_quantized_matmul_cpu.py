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
"""CPU unit tests for quantized_matmul kernels, utilities, and tuned block sizes."""

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.quantized_matmul import (
    blockwise_kernel,
    kernel,
    tuned_block_sizes,
    util,
)

jax.config.update("jax_numpy_dtype_promotion", "standard")


class _FakeRef:
    """Minimal mutable buffer emulating a Pallas Ref on CPU."""

    def __init__(self, array: jax.Array):
        self.array = array

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self.array
        return self.array[idx]

    def __setitem__(self, idx, val):
        if idx is Ellipsis:
            self.array = jnp.asarray(val, dtype=self.array.dtype)
        else:
            self.array = self.array.at[idx].set(val)


def _install_cpu_pallas_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch pl.pallas_call to execute the Pallas grid in pure JAX on CPU."""
    state = {"pids": (0, 0, 0), "grid": (1, 1, 1)}

    monkeypatch.setattr(pl, "program_id", lambda axis: state["pids"][axis])
    monkeypatch.setattr(pl, "num_programs", lambda axis: state["grid"][axis])

    def _fake_pallas_call(kernel_fn, *, grid_spec, out_shape, compiler_params=None):
        del compiler_params

        def _call(*args: jax.Array) -> jax.Array:
            grid = grid_spec.grid
            state["grid"] = grid
            out = jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            scratch_refs = [
                _FakeRef(jnp.zeros(s.shape, dtype=s.dtype)) if s is not None else None
                for s in grid_spec.scratch_shapes
            ]

            for b, o, i in itertools.product(
                range(grid[0]), range(grid[1]), range(grid[2])
            ):
                state["pids"] = (b, o, i)
                in_refs = []
                for arg, spec in zip(args, grid_spec.in_specs):
                    idx = spec.index_map(b, o, i)
                    slices = tuple(
                        slice(c * bs, (c + 1) * bs)
                        for c, bs in zip(idx, spec.block_shape)
                    )
                    in_refs.append(_FakeRef(arg[slices]))

                out_idx = grid_spec.out_specs.index_map(b, o, i)
                out_slices = tuple(
                    slice(c * bs, (c + 1) * bs)
                    for c, bs in zip(out_idx, grid_spec.out_specs.block_shape)
                )
                out_ref = _FakeRef(out[out_slices])
                kernel_fn(*in_refs, out_ref, *scratch_refs)
                out = out.at[out_slices].set(out_ref.array)

            return out

        return _call

    monkeypatch.setattr(pl, "pallas_call", _fake_pallas_call)


def test_tuned_block_sizes_and_device_vmem_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default CPU path: get_tpu_version() == -1
    assert tuned_block_sizes.get_tpu_version() == -1
    assert tuned_block_sizes.get_device_vmem_limit() == 96 * 1024 * 1024

    # Fallback tuned value when key is missing
    default_val = tuned_block_sizes.get_tuned_block_sizes(128, 128, 128, "int8", "int8")
    assert default_val == tuned_block_sizes.TunedValue(128, 128, 128)

    # Simulated TPU v6 and v7 paths
    monkeypatch.setattr(
        jax, "devices", lambda: [SimpleNamespace(device_kind="TPU v7x")]
    )
    assert tuned_block_sizes.get_tpu_version() == 7
    assert tuned_block_sizes.get_device_vmem_limit() == 48 * 1024 * 1024

    monkeypatch.setattr(
        jax, "devices", lambda: [SimpleNamespace(device_kind="TPU v6e")]
    )
    assert tuned_block_sizes.get_tpu_version() == 6
    assert tuned_block_sizes.get_device_vmem_limit() == 96 * 1024 * 1024

    hit = tuned_block_sizes.get_tuned_block_sizes(1024, 1024, 4096, "int8", "int8")
    assert hit == tuned_block_sizes.TunedValue(1024, 256, 4096)


def test_util_functions_and_validation() -> None:
    # unfold_args with both Python bool and JAX scalar bool array
    recorded = []
    util.unfold_args(
        (True, jnp.asarray(True), jnp.asarray(False)),
        (),
        lambda a, b, c: recorded.append((a, b, c)),
    )
    assert (True, True, False) in recorded

    # quantize_tensor per-channel and blockwise (for both float and int dtypes)
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) - 4.0
    q_int8, s_int8 = util.quantize_tensor(x, jnp.int8)
    assert q_int8.dtype == jnp.int8
    assert s_int8.shape == (4, 1)

    q_fp8, s_fp8 = util.quantize_tensor(x, jnp.float8_e4m3fn, block_size=2)
    assert q_fp8.dtype == jnp.float8_e4m3fn
    assert s_fp8.shape == (2, 1, 4)

    # quantize_array in util and kernel
    x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True).T
    qa_util, sa_util = util.quantize_array(x, x_abs_max, jnp.int8)
    qa_kern, sa_kern = kernel.quantize_array(x, x_abs_max, jnp.int8)
    assert qa_util.shape == x.shape and sa_util.shape == (4, 1)
    assert qa_kern.shape == x.shape and sa_kern.shape == (4, 1)

    # get_kernel_name & next_multiple
    tv = tuned_block_sizes.TunedValue(128, 256, 512)
    assert util.get_kernel_name(tv) == "quantized_matmul_kernel_128_256_512"
    assert util.next_multiple(130, 128) == 256

    # xla_quantized_matmul (quantize_activation=True/False, int8/fp8)
    w_q_int8 = jnp.ones((4, 6), dtype=jnp.int8)
    w_scale = jnp.ones((6,), dtype=jnp.float32)
    out_q_int = util.xla_quantized_matmul(
        x, w_q_int8, w_scale, quantize_activation=True
    )
    out_noq = util.xla_quantized_matmul(x, w_q_int8, w_scale, quantize_activation=False)
    w_q_fp8 = jnp.ones((4, 6), dtype=jnp.float8_e4m3fn)
    out_q_fp = util.xla_quantized_matmul(x, w_q_fp8, w_scale, quantize_activation=True)
    assert out_q_int.shape == (4, 6)
    assert out_noq.shape == (4, 6)
    assert out_q_fp.shape == (4, 6)

    # xla_quantized_batched_matmul:
    # 1) perm == [0, 1, 2] (batch=(0,), lhs_free=(1,), contract=(2,))
    xb = jnp.ones((2, 3, 4), dtype=jnp.bfloat16)
    wb_int8 = jnp.ones((2, 5, 4), dtype=jnp.int8)
    ws = jnp.ones((5,), dtype=jnp.float32)
    out_b1 = util.xla_quantized_batched_matmul(
        xb, wb_int8, ws, (((2,), (2,)), ((0,), (0,))), quantize_activation=True
    )
    assert out_b1.shape == (2, 3, 5)

    # 2) perm != [0, 1, 2] (batch=(1,), lhs_free=(0,), contract=(2,)) + fp8 + 3D scale
    wb_fp8 = jnp.ones((5, 3, 4), dtype=jnp.float8_e4m3fn)
    ws_3d = jnp.ones((3, 1, 5), dtype=jnp.float32)
    out_b2 = util.xla_quantized_batched_matmul(
        xb, wb_fp8, ws_3d, (((2,), (2,)), ((1,), (1,))), quantize_activation=True
    )
    assert out_b2.shape == (3, 2, 5)

    # 3) quantize_activation=False
    out_b3 = util.xla_quantized_batched_matmul(
        xb, wb_fp8, ws, (((2,), (2,)), ((1,), (1,))), quantize_activation=False
    )
    assert out_b3.shape == (3, 2, 5)

    # get_vmem_limit with single-block vs multi-block and save_acc/save_x_q combinations
    v1 = util.get_vmem_limit(
        n_batch=1,
        n_out=1,
        n_in=1,
        batch_block_size=128,
        out_block_size=128,
        in_block_size=128,
        x_dtype=jnp.bfloat16,
        x_q_dtype=jnp.int8,
        w_q_dtype=jnp.int8,
        scale_dtype=jnp.float32,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.int32,
        save_acc=False,
        save_x_q=False,
        upper_limit_bytes=96 * 1024 * 1024,
    )
    v2 = util.get_vmem_limit(
        n_batch=2,
        n_out=2,
        n_in=2,
        batch_block_size=128,
        out_block_size=128,
        in_block_size=128,
        x_dtype=jnp.bfloat16,
        x_q_dtype=jnp.int8,
        w_q_dtype=jnp.int8,
        scale_dtype=jnp.float32,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.int32,
        save_acc=True,
        save_x_q=True,
        upper_limit_bytes=96 * 1024 * 1024,
    )
    assert v2 > v1 > 0

    # util.validate_inputs error and valid branches
    x_v = jnp.ones((8, 16), dtype=jnp.bfloat16)
    wq_v = jnp.ones((8, 16), dtype=jnp.int8)
    ws_2d = jnp.ones((1, 8), dtype=jnp.float32)
    ws_3d_ok = jnp.ones((2, 1, 8), dtype=jnp.float32)
    x_abs = jnp.ones((1, 8), dtype=jnp.float32)

    util.validate_inputs(x_v, wq_v, ws_2d, x_abs, 4, 4, 8)
    util.validate_inputs(x_v, wq_v, ws_3d_ok, None, 4, 4, 8)

    with pytest.raises(ValueError, match="x.shape\\[1\\]"):
        util.validate_inputs(
            x_v, jnp.ones((8, 8), dtype=jnp.int8), ws_2d, x_abs, 4, 4, 8
        )
    with pytest.raises(ValueError, match="w_q.shape\\[0\\]"):
        util.validate_inputs(
            x_v, wq_v, jnp.ones((2, 1, 4), dtype=jnp.float32), x_abs, 4, 4, 8
        )
    with pytest.raises(ValueError, match="x_abs_max.shape"):
        util.validate_inputs(
            x_v, wq_v, ws_2d, jnp.ones((1, 4), dtype=jnp.float32), 4, 4, 8
        )
    with pytest.raises(ValueError, match="batch_block_size"):
        util.validate_inputs(x_v, wq_v, ws_2d, x_abs, 6, 4, 8)
    with pytest.raises(ValueError, match="out_block_size"):
        util.validate_inputs(x_v, wq_v, ws_2d, x_abs, 4, 6, 8)
    with pytest.raises(ValueError, match="in_block_size"):
        util.validate_inputs(x_v, wq_v, ws_2d, x_abs, 4, 4, 6)


def test_kernel_validate_inputs_and_errors() -> None:
    x = jnp.ones((8, 16), dtype=jnp.bfloat16)
    wq = jnp.ones((8, 16), dtype=jnp.int8)
    ws = jnp.ones((1, 8), dtype=jnp.float32)
    x_abs = jnp.ones((1, 8), dtype=jnp.float32)

    with pytest.raises(ValueError, match="same int or float type"):
        kernel.validate_inputs(x, wq, ws, x_abs, jnp.float8_e4m3fn, 4, 4, 8)
    with pytest.raises(ValueError, match="x.shape\\[1\\]"):
        kernel.validate_inputs(
            x, jnp.ones((8, 8), dtype=jnp.int8), ws, x_abs, jnp.int8, 4, 4, 8
        )
    with pytest.raises(ValueError, match="w_q.shape\\[0\\]"):
        kernel.validate_inputs(
            x, wq, jnp.ones((1, 4), dtype=jnp.float32), x_abs, jnp.int8, 4, 4, 8
        )
    with pytest.raises(ValueError, match="x_abs_max.shape"):
        kernel.validate_inputs(
            x, wq, ws, jnp.ones((1, 4), dtype=jnp.float32), jnp.int8, 4, 4, 8
        )
    with pytest.raises(ValueError, match="batch_block_size"):
        kernel.validate_inputs(x, wq, ws, x_abs, jnp.int8, 6, 4, 8)
    with pytest.raises(ValueError, match="out_block_size"):
        kernel.validate_inputs(x, wq, ws, x_abs, jnp.int8, 4, 6, 8)
    with pytest.raises(ValueError, match="in_block_size"):
        kernel.validate_inputs(x, wq, ws, x_abs, jnp.int8, 4, 4, 6)

    with pytest.raises(NotImplementedError, match="zero_point"):
        kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((8,), dtype=jnp.float32), w_zp=jnp.ones((8,))
        )
    with pytest.raises(NotImplementedError, match="block_size"):
        kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((8,), dtype=jnp.float32), block_size=4
        )


def test_per_channel_kernel_execution_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_pallas_runner(monkeypatch)

    x = jnp.ones((8, 16), dtype=jnp.bfloat16)
    wq_int8 = jnp.ones((8, 16), dtype=jnp.int8)
    ws_bf16 = jnp.ones((8,), dtype=jnp.bfloat16)

    with jax.disable_jit():
        # 1) save_x_q=True (quantize_activation=True, n_in=1, n_out=2), aligned shapes
        out1 = kernel.quantized_matmul_kernel(
            x,
            wq_int8,
            ws_bf16,
            x_q_dtype=jnp.int8,
            tuned_value=tuned_block_sizes.TunedValue(4, 4, 16),
        )
        assert out1.shape == (8, 8)

        # 2) save_acc=True (n_in=2, n_out=2), unaligned shapes requiring padding + slice
        x_unaligned = jnp.ones((6, 12), dtype=jnp.bfloat16)
        wq_unaligned = jnp.ones((6, 12), dtype=jnp.float8_e4m3fn)
        ws_f32 = jnp.ones((6,), dtype=jnp.float32)
        out2 = kernel.quantized_matmul_kernel(
            x_unaligned,
            wq_unaligned,
            ws_f32,
            x_q_dtype=jnp.float8_e4m3fn,
            tuned_value=tuned_block_sizes.TunedValue(4, 4, 8),
        )
        assert out2.shape == (6, 6)

        # 3) quantize_activation=False (x_q_dtype=None), tuned_value=None
        out3 = kernel.quantized_matmul_kernel(
            x,
            wq_int8,
            jnp.ones((8,), dtype=jnp.float32),
            x_q_dtype=None,
            tuned_value=None,
        )
        assert out3.shape == (8, 8)

        # 4) Exercise matmul_kernel when save_x_q=True and save_acc=True simultaneously
        # (including quant=False with is_last_step=False and is_last_step=True)
        x_ref = _FakeRef(jnp.ones((4, 8), dtype=jnp.bfloat16))
        w_q_ref = _FakeRef(jnp.ones((4, 8), dtype=jnp.int8))
        w_scale_ref = _FakeRef(jnp.ones((1, 4), dtype=jnp.float32))
        x_abs_max_ref = _FakeRef(jnp.ones((1, 4), dtype=jnp.float32))
        out_ref = _FakeRef(jnp.zeros((4, 4), dtype=jnp.bfloat16))
        acc_scratch = _FakeRef(jnp.zeros((4, 4), dtype=jnp.int32))
        x_q_scratch = _FakeRef(jnp.zeros((4, 8), dtype=jnp.int8))
        x_scale_scratch = _FakeRef(jnp.ones((4, 1), dtype=jnp.float32))

        monkeypatch.setattr(pl, "num_programs", lambda axis: 2)
        for out_idx, in_idx in [(0, 0), (1, 0), (1, 1)]:
            monkeypatch.setattr(
                pl, "program_id", lambda axis, o=out_idx, i=in_idx: (0, o, i)[axis]
            )
            kernel.matmul_kernel(
                x_ref,
                w_q_ref,
                w_scale_ref,
                x_abs_max_ref,
                out_ref,
                acc_scratch,
                x_q_scratch,
                x_scale_scratch,
                x_q_dtype=jnp.int8,
                save_acc=True,
                save_x_q=True,
            )


def test_blockwise_kernel_validation_and_execution_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cpu_pallas_runner(monkeypatch)

    x = jnp.ones((8, 16), dtype=jnp.bfloat16)
    wq = jnp.ones((512, 16), dtype=jnp.float8_e4m3fn)
    ws = jnp.ones((2, 1, 512), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Block size was not specified"):
        blockwise_kernel.quantized_matmul_kernel(x, wq, ws, block_size=None)
    with pytest.raises(ValueError, match="Block size must be positive"):
        blockwise_kernel.quantized_matmul_kernel(x, wq, ws, block_size=0)
    with pytest.raises(NotImplementedError, match="zero_point"):
        blockwise_kernel.quantized_matmul_kernel(
            x, wq, ws, w_zp=jnp.ones((512,)), block_size=8
        )
    with pytest.raises(ValueError, match="w_scale must be rank 3"):
        blockwise_kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((2, 512)), block_size=8
        )
    with pytest.raises(ValueError, match="w_scale.shape\\[1\\] must be 1"):
        blockwise_kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((2, 2, 512)), block_size=8
        )
    with pytest.raises(ValueError, match="w_scale output dim"):
        blockwise_kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((2, 1, 256)), block_size=8
        )
    with pytest.raises(ValueError, match="must be divisible by block_size"):
        blockwise_kernel.quantized_matmul_kernel(x, wq, ws, block_size=6)
    with pytest.raises(ValueError, match="w_scale block dim"):
        blockwise_kernel.quantized_matmul_kernel(
            x, wq, jnp.ones((4, 1, 512)), block_size=8
        )
    with pytest.raises(ValueError, match="Channelwise weight scales require"):
        blockwise_kernel.quantized_matmul_kernel(
            x,
            wq,
            jnp.ones((1, 1, 512), dtype=jnp.float32),
            block_size=16,
            tuned_value=tuned_block_sizes.TunedValue(8, 512, 8, 1),
        )
    with pytest.raises(ValueError, match="in_block_size=.*must be divisible"):
        blockwise_kernel.quantized_matmul_kernel(
            x,
            wq,
            ws,
            block_size=8,
            tuned_value=tuned_block_sizes.TunedValue(8, 512, 4, 1),
        )
    with pytest.raises(ValueError, match="must be a multiple of in_block_size"):
        blockwise_kernel.quantized_matmul_kernel(
            x,
            wq,
            ws,
            block_size=8,
            tuned_value=tuned_block_sizes.TunedValue(8, 512, 32, 1),
        )
    with pytest.raises(ValueError, match="out_block_size=.*must be divisible"):
        blockwise_kernel.quantized_matmul_kernel(
            x,
            wq,
            ws,
            block_size=8,
            tuned_value=tuned_block_sizes.TunedValue(8, 512, 16, 4),
        )

    with jax.disable_jit():
        # 1) Aligned execution with steps_k=2 (block_size=4, in_block_size=8),
        # n_in=2 (orig_n_in=16 -> pid_k=0 and pid_k=1), int8 x int8 -> acc_dtype=int32
        wq_int8 = jnp.ones((512, 16), dtype=jnp.int8)
        ws_bf16 = jnp.ones((4, 1, 512), dtype=jnp.bfloat16)
        out_aligned = blockwise_kernel.quantized_matmul_kernel(
            x,
            wq_int8,
            ws_bf16,
            block_size=4,
            x_q_dtype=jnp.int8,
            tuned_value=tuned_block_sizes.TunedValue(8, 256, 8, 1),
        )
        assert out_aligned.shape == (8, 512)

        # 2) Unaligned batch & out dimensions (padding + slice) with channelwise scale
        # (block_size == orig_n_in == 128), x_q_dtype=None, tuned_value=None
        monkeypatch.setattr(
            tuned_block_sizes,
            "_DEFAULT_TUNED_VALUE",
            tuned_block_sizes.TunedValue(8, 256, 128, 1),
        )
        x_un = jnp.ones((6, 128), dtype=jnp.bfloat16)
        wq_un = jnp.ones((200, 128), dtype=jnp.float8_e4m3fn)
        ws_un = jnp.ones((1, 1, 200), dtype=jnp.float32)
        out_unaligned = blockwise_kernel.quantized_matmul_kernel(
            x_un,
            wq_un,
            ws_un,
            block_size=128,
            x_q_dtype=None,
            tuned_value=None,
        )
        assert out_unaligned.shape == (6, 200)
