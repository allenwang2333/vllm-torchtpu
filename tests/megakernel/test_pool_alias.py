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
"""Unit tests for `pool_alias`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel import pool_alias
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8


def test_pool_alias_storage_size():
    """Verify TPU 8x128 tile storage calculation."""
    assert pool_alias.storage_size((128, 256), jnp.dtype(FP8)) == 128 * 256 * 1
    assert pool_alias.storage_size((128, 256), jnp.dtype(BF16)) == 128 * 256 * 2
    assert pool_alias.storage_size((1, 256), jnp.dtype(F32)) == 8 * 256 * 4


def test_pool_alias_reinterpret_mosaic_lowering():
    """Verify `pool_alias.Reinterpret` lowers to TPU MLIR ops (`jax==0.11.2`).

    Exercises the `jax._src.pallas.mosaic.lowering._reshape_memref` hook (pinned
    against `jax==0.11.2` / `jaxlib==0.11.2` in `pyproject.toml`). If a future JAX
    upgrade changes the signature or behavior of `_reshape_memref`,
    `tpu.reinterpret_cast`, or `tpu.erase_memref_layout`, this test will fail on CPU.
    """
    from jax._src.lib import tpu
    from jax._src.lib.mlir import ir
    from jax._src.pallas.mosaic import lowering

    for dtype, target_shape in [(F32, (16, 256)), (BF16, (32, 256)), (FP8, (64, 256))]:
        nbytes = pool_alias.storage_size(target_shape, jnp.dtype(dtype))
        itemsize = jnp.dtype(dtype).itemsize
        flat_elems = nbytes // itemsize
        flat_shape = (1, 8, flat_elems // 8)

        with ir.Context() as ctx, ir.Location.unknown():
            tpu.register_dialect(ctx)
            module = ir.Module.create()
            vmem_attr = ir.Attribute.parse("#tpu.memory_space<vmem>")
            if dtype == F32:
                elem_type = ir.F32Type.get()
            elif dtype == BF16:
                elem_type = ir.BF16Type.get()
            else:
                elem_type = ir.Float8E4M3FNType.get()
            src_memref = ir.MemRefType.get(
                flat_shape, elem_type, memory_space=vmem_attr
            )
            with ir.InsertionPoint(module.body):
                func_type = ir.FunctionType.get([src_memref], [])
                func_op = ir.Operation.create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("test_reinterpret"),
                        "function_type": ir.TypeAttr.get(func_type),
                    },
                    regions=1,
                )
                block = func_op.regions[0].blocks.append(src_memref)
                with ir.InsertionPoint(block):
                    transform = pool_alias.Reinterpret(target_shape)
                    aval = jax.ShapeDtypeStruct(flat_shape, dtype)
                    out_val, out_shape = lowering._reshape_memref(
                        block.arguments[0], transform, aval, flat_shape
                    )
                    assert out_shape == target_shape
                    out_type = ir.MemRefType(out_val.type)
                    assert tuple(out_type.shape) == target_shape
                    assert out_type.memory_space == vmem_attr
                asm = str(func_op)
                assert "tpu.reinterpret_cast" in asm
                assert "tpu.erase_memref_layout" in asm
