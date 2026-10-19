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
"""Portability contract and private-JAX (`jax._src.*`) compatibility tests."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel import pool_alias
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8


def test_megakernel_portability_ast_guard():
    """Enforce relative internal imports, 0 vllm/torch deps, and jax._src allowlist.

    Checks all `.py` files under `src/vllm_torchtpu/megakernel/`:
    - Zero imports from `vllm`, `vllm_torchtpu`, or `torch`.
    - All absolute (`level == 0`) imports come strictly from Python stdlib, `jax`,
      or `numpy` (enforcing relative imports for all internal modules).
    - Private `jax._src.*` imports are restricted to `{"pool_alias.py"}` (plus
      `"cache_experimental.py"` if added in M3).
    - Dynamic import calls (`__import__`, `import_module`) are forbidden.
    """
    pkg_root = Path(pool_alias.__file__).resolve().parent
    jax_src_allowlist = {"pool_alias.py"}
    forbidden_roots = {"vllm", "vllm_torchtpu", "torch"}
    allowed_external_roots = sys.stdlib_module_names | {"jax", "numpy"}

    py_files = sorted(pkg_root.rglob("*.py"))
    rel_paths = {p.relative_to(pkg_root).as_posix() for p in py_files}
    required_modules = {
        "__init__.py",
        "comm.py",
        "errors.py",
        "ops.py",
        "planner.py",
        "pool_alias.py",
        "spec.py",
        "stream.py",
        "vmem.py",
        "profiles/__init__.py",
        "profiles/default.py",
    }
    assert required_modules <= rel_paths, required_modules - rel_paths

    for path in py_files:
        rel = path.relative_to(pkg_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else None)
                )
                assert func_name not in {"__import__", "import_module"}, (
                    f"{rel}:{node.lineno} calls dynamic import '{func_name}'; "
                    "use static relative imports within megakernel/."
                )
                continue

            if isinstance(node, ast.Import):
                level = 0
                modules = [alias.name for alias in node.names]
                imported_names: list[str] = []
            elif isinstance(node, ast.ImportFrom):
                level = node.level
                modules = [node.module] if node.module else []
                imported_names = [alias.name for alias in node.names]
            else:
                continue

            # `__init__.py` is the public package facade and may re-export
            # model helpers; all other core modules outside `models/` must not.
            facade_allowlist = {"__init__.py"}
            if rel not in facade_allowlist and not rel.startswith("models/"):
                for mod in modules:
                    assert mod != "models" and not mod.startswith("models."), (
                        f"{rel}:{node.lineno} imports '{mod}'; core megakernel "
                        "modules must be model-agnostic and never import models/."
                    )
                if not modules:
                    assert "models" not in imported_names, (
                        f"{rel}:{node.lineno} imports 'models'; core megakernel "
                        "modules must be model-agnostic and never import models/."
                    )

            for mod in modules:
                root = mod.split(".")[0]
                assert root not in forbidden_roots, (
                    f"{rel}:{node.lineno} imports forbidden module '{mod}'; "
                    "use relative imports and keep 0 vllm/torch dependencies."
                )
                if level == 0:
                    assert root in allowed_external_roots, (
                        f"{rel}:{node.lineno} uses non-relative or disallowed import "
                        f"'{mod}'; internal megakernel imports must be relative and "
                        "external imports must be Python stdlib, jax, or numpy."
                    )
                if mod == "jax._src" or mod.startswith("jax._src."):
                    assert rel in jax_src_allowlist, (
                        f"{rel}:{node.lineno} imports private JAX API '{mod}', "
                        f"which is restricted to allowlist {sorted(jax_src_allowlist)}."
                    )


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
