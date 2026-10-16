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
"""Bridge between `megakernel` (`make_flat_pooled_step`) and PyTorch/vLLM.

Provides:
- `build_pooled_step_op`: wraps `cache.make_flat_pooled_step` into a
  `sharded_jax_callable` + `@torch.library.custom_op` (taking `list[Tensor]`
  for `weights` and `pools` to stay under the PyTorch dispatcher's 64-argument
  schema limit), with automatic CPU `interpret=True` execution when called with
  CPU `torch.Tensor` operands in unit tests.
- `execute_pooled_step_inplace`: invokes the custom op and copies the donated
  pool outputs back into `pools` via `dst.copy_(src)`, preserving `dst.data_ptr()`
  (Raiden pointer identity) and folding into an XLA `input_output_alias` /
  `tf.aliasing_output` under `torch.compile(backend="tpu")`.
- `verify_pooled_step_hlo`: lowers/compiles the JAX pooled step (defaulting to
  compiled post-optimization HLO on TPU) and validates pool buffer donation via
  `cache.check_pooled_step_hlo`.
- `verify_torchtpu_compiled_step_ir`: inspects the compiled `torch_tpu` forward
  graph's StableHLO MLIR / HLO (`TpuCompileDebug.stablehlo_forward_text`) to
  verify that `dst.copy_(src)` on all pools folded into `tf.aliasing_output` /
  `input_output_alias` with zero full-pool copies.
- `export_pooled_step`: offline inspection/serialization helper that exports the
  StableHLO module and aliasing manifest (`PooledStepExport`) via `jax.export`
  (serving traces directly through `build_pooled_step_op` / `JaxCallable`).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import cache
from vllm_torchtpu.megakernel.spec import ExecSpec, ModelSpec

_STOCK_PLACEHOLDER_LINE = """      output_shapes = [
          torch_placeholder(aval, mesh=self.mesh) for aval in lowered.out_avals
      ]"""
_SHARDED_PLACEHOLDER_LINE = (
    """      output_shapes = _sharded_output_placeholders(lowered, self.mesh)"""
)
_SHARDED_CALLABLE_CLS: Any = None


def _leaf_output_specs(output_partition_specs: Any) -> list[P]:
    """Flatten `output_partition_specs` to one `PartitionSpec` per output leaf."""
    return jax.tree.leaves(output_partition_specs, is_leaf=lambda s: isinstance(s, P))


def _placeholder(pallas_impl: Any, aval: Any, spec: Any, mesh: Any) -> torch.Tensor:
    """Build a torch placeholder tensor shaped for this rank's local shard."""
    if spec is None or not isinstance(aval, jax.core.ShapedArray):
        return pallas_impl.torch_placeholder(aval, mesh=mesh)
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(f"Unsupported dtype for pallas kernels: {aval.dtype}")
    shape = tuple(pallas_impl.get_local_shape(aval.shape, mesh, spec))
    if torch_dtype == torch.float4_e2m1fn_x2 and shape:
        if shape[-1] % 2:
            raise ValueError(
                "expected the last dimension of the JAX local shape to be even "
                f"to pack into {torch_dtype}, got {shape[-1]} (local={shape}, "
                f"global={aval.shape}, spec={spec})"
            )
        shape = shape[:-1] + (shape[-1] // 2,)
    return torch.empty(shape, dtype=torch_dtype, device="tpu")


def _make_sharded_output_placeholders(
    pallas_impl: Any,
) -> Callable[[Any, Any], list[torch.Tensor]]:
    def _sharded_output_placeholders(lowered: Any, mesh: Any) -> list[torch.Tensor]:
        shardings = getattr(lowered, "_out_named_shardings", None)
        if shardings is None or len(shardings) != len(lowered.out_avals):
            shardings = (None,) * len(lowered.out_avals)
        return [
            _placeholder(pallas_impl, aval, getattr(sharding, "spec", None), mesh)
            for aval, sharding in zip(lowered.out_avals, shardings)
        ]

    return _sharded_output_placeholders


def _get_sharded_callable_cls(pallas_impl: Any) -> Any:
    """Return a `JaxCallable` class that sizes output placeholders per local shard."""
    global _SHARDED_CALLABLE_CLS
    if _SHARDED_CALLABLE_CLS is not None:
        return _SHARDED_CALLABLE_CLS
    init_sig = inspect.signature(pallas_impl.JaxCallable.__init__)
    if "output_partition_specs" in init_sig.parameters:
        _SHARDED_CALLABLE_CLS = pallas_impl.JaxCallable
        return _SHARDED_CALLABLE_CLS
    src = inspect.getsource(pallas_impl.JaxCallable.__call__)
    if _STOCK_PLACEHOLDER_LINE not in src:
        _SHARDED_CALLABLE_CLS = pallas_impl.JaxCallable
        return _SHARDED_CALLABLE_CLS
    patched = "class _Holder:\n" + src.replace(
        _STOCK_PLACEHOLDER_LINE, _SHARDED_PLACEHOLDER_LINE, 1
    )
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        compile(patched, pallas_impl.__file__, "exec"),
        {
            **pallas_impl.__dict__,
            "_sharded_output_placeholders": _make_sharded_output_placeholders(
                pallas_impl
            ),
        },
        namespace,
    )
    _SHARDED_CALLABLE_CLS = type(
        "_ShardedJaxCallable",
        (pallas_impl.JaxCallable,),
        {"__call__": namespace["_Holder"].__call__},
    )
    return _SHARDED_CALLABLE_CLS


def sharded_jax_callable(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[P],
    output_partition_specs: Any,
    donate_argnums: Sequence[int] | None = None,
) -> Any:
    """Wrap an SPMD JAX function over `mesh` into a `torch_tpu` `JaxCallable`.

    Self-contained so `exported_bridge` does not depend on any `vllm` or
    `vllm_torchtpu.distributed` modules when extracted to a standalone repo.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name {name} must be 'namespace::op', with one '::'.")

    callable_cls = _get_sharded_callable_cls(pallas_impl)
    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, P),
    )

    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        out_shardings=out_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(output_partition_specs),
        },
    )
    extra_kwargs: dict[str, Any] = {}
    init_sig = inspect.signature(callable_cls.__init__)
    if "output_partition_specs" in init_sig.parameters:
        extra_kwargs["output_partition_specs"] = _leaf_output_specs(
            output_partition_specs
        )
    return callable_cls(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        **extra_kwargs,
    )


@dataclass(frozen=True)
class PooledStepExport:
    """Offline serialized StableHLO export and aliasing manifest for a pooled step."""

    op_name: str
    exported: Any
    mlir_module_serialized: bytes
    in_specs: tuple[P, ...]
    out_specs: tuple[P, ...]
    donate_argnums: tuple[int, ...]
    input_output_aliases: dict[int, int]


_OP_COUNTER = 0


def _torch_dtype_to_jax(dt: torch.dtype) -> Any:
    if dt == torch.float8_e4m3fn:
        return jnp.float8_e4m3fn
    if dt == torch.bfloat16:
        return jnp.bfloat16
    if dt == torch.float32:
        return jnp.float32
    if dt == torch.int32:
        return jnp.int32
    if dt == torch.int64:
        return jnp.int64
    raise TypeError(f"Unsupported torch.dtype for megakernel bridge: {dt}")


def _torch_cpu_to_jax(
    t: torch.Tensor, sharding: NamedSharding | None = None
) -> jax.Array:
    """Convert a CPU `torch.Tensor` to a `jax.Array` without precision loss."""
    t_c = t.detach().cpu().contiguous()
    if t_c.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t_c.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t_c.dtype == torch.bfloat16:
        arr = jnp.asarray(t_c.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t_c.numpy())
    if sharding is not None:
        arr = jax.device_put(arr, sharding)
    return arr


def _jax_to_torch_cpu(a: jax.Array, like: torch.Tensor) -> torch.Tensor:
    """Convert a `jax.Array` to a CPU `torch.Tensor` matching `like.dtype`."""
    np_arr = np.asarray(a)
    if like.dtype == torch.float8_e4m3fn:
        out = torch.from_numpy(np_arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    elif like.dtype == torch.bfloat16:
        out = torch.from_numpy(np_arr.view(np.uint16).copy()).view(torch.bfloat16)
    else:
        out = torch.from_numpy(np_arr.copy()).to(like.dtype)
    return out


def make_typed_step_wrapper(
    fn_name: str,
    step_fn: Callable[..., Any],
    num_inputs: int,
    num_outputs: int,
) -> Callable[..., Any]:
    """Synthesize a fixed-arity `jax.Array`-annotated wrapper for `JaxCallable`.

    `torch_tpu`'s `JaxCallable` inspects the function's explicit positional
    parameters and rejects `*args` signatures.
    """
    args = [f"x{i}" for i in range(num_inputs)]
    ret = "tuple[" + ", ".join(["jax.Array"] * num_outputs) + "]"
    src = (
        f"def {fn_name}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
        f"    return _step_fn({', '.join(args)})\n"
    )
    ns: dict[str, Any] = {"jax": jax, "_step_fn": step_fn}
    exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102
    return ns[fn_name]


def _build_global_avals(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
) -> list[jax.ShapeDtypeStruct]:
    """Construct global `jax.ShapeDtypeStruct` descriptors for HLO lowering/export."""
    r_cnt = int(mesh.devices.size)
    g: Any = spec.config
    b_tok = ex.B
    t_tok = ex.T
    k_ckpt = int(ckpt_cols) if ckpt_cols is not None else t_tok
    nseq = ex.NS
    m_rows = ex.M
    g_per_att = max(1, int(getattr(g, "att_every", 4)) - 1)

    def _global_sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
        dt = _torch_dtype_to_jax(t.dtype)
        return jax.ShapeDtypeStruct((r_cnt * int(t.shape[0]),) + tuple(t.shape[1:]), dt)

    return [
        jax.ShapeDtypeStruct((b_tok,), jnp.int32),
        jax.ShapeDtypeStruct((r_cnt * m_rows, int(g.H)), jnp.bfloat16),
        *(_global_sds(w) for w in w_flat),
        *(_global_sds(p) for p in raw_pools),
        jax.ShapeDtypeStruct((nseq, int(max_pages_per_seq)), jnp.int32),
        jax.ShapeDtypeStruct((g_per_att, nseq, k_ckpt), jnp.int32),
        jax.ShapeDtypeStruct((g_per_att, nseq), jnp.int32),
    ]


def verify_pooled_step_hlo(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    compile_hlo: bool | None = None,
) -> tuple[bool, str]:
    """Lower or compile `make_flat_pooled_step` and verify pool donation in HLO.

    When `compile_hlo` is `None` (default), compiles post-optimization HLO on
    TPU (`jax.default_backend() == "tpu" and not ex.interpret`) so XLA buffer
    assignment and copy insertion have run.
    """
    if compile_hlo is None:
        compile_hlo = bool(jax.default_backend() == "tpu" and not ex.interpret)
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    avals = _build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    j_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    lowered = j_fn.lower(*avals)
    hlo_text = lowered.compile().as_text() if compile_hlo else lowered.as_text("hlo")
    num_pages = int(raw_pools[0].shape[0])
    page_size = int(raw_pools[0].shape[-1])
    g: Any = spec.config
    kvh = int(g.nkv(ex.R)) if hasattr(g, "nkv") else 1
    hd = int(getattr(g, "HD", 256))
    return cache.check_pooled_step_hlo(
        hlo_text,
        page_size=page_size,
        kvh=kvh,
        hd=hd,
        min_pool_pages=max(16, num_pages),
    )


def verify_torchtpu_compiled_step_ir(
    ir_text: str,
    *,
    num_pages: int,
    page_size: int = 128,
    kvh: int = 1,
    hd: int = 256,
    num_pools: int = 15,
) -> tuple[bool, str]:
    """Verify that a compiled `torch_tpu` forward step aliases all pools in-place.

    Inspects the StableHLO MLIR (`TpuCompileDebug.stablehlo_forward_text`) or
    HLO text produced by `torch.compile(backend="tpu")` for the outer step graph
    that invokes `execute_pooled_step_inplace` (`op(...)` + `dst.copy_(src)`):
    1. Confirms the custom Pallas kernel call is present.
    2. Confirms all `num_pools` pool inputs carry `tf.aliasing_output` (MLIR) or
       `input_output_alias` (HLO) so `dst.copy_(src)` is folded into an in-place
       buffer alias.
    3. Confirms no `copy`, `copy-start`, `dynamic_update_slice`, or
       `as_strided_copy_inverse` materializes a full-pool buffer.
    """
    if not ir_text or not ir_text.strip():
        return False, "empty torch_tpu compiled IR"

    has_custom_call = (
        "custom_call" in ir_text
        or "tpu_custom_call" in ir_text
        or "megakernel_runtime" in ir_text
    )
    if not has_custom_call:
        return False, "missing custom kernel call in torch_tpu compiled IR"

    n_mlir_alias = ir_text.count("tf.aliasing_output")
    n_hlo_alias = ir_text.count("input_output_alias")
    if n_mlir_alias < num_pools and n_hlo_alias == 0:
        return (
            False,
            f"expected >= {num_pools} tf.aliasing_output attributes on donated "
            f"pool arguments in torch_tpu IR, found {n_mlir_alias}",
        )

    mlir_pool_pat = re.compile(rf"\b(\d+)x{2 * kvh}x{hd // 4}x4x{page_size}xf8E4M3FN\b")
    hlo_pool_suffix = f",{2 * kvh},{hd // 4},4,{page_size}]"
    copy_ops = (
        "stablehlo.copy",
        "mhlo.copy",
        "copy-start(",
        "dynamic_update_slice",
        "dynamic-update-slice(",
        "as_strided_copy_inverse",
    )
    for line in ir_text.splitlines():
        if "func.func" in line or "ENTRY" in line or "custom_call" in line:
            continue
        is_copy_line = any(op in line for op in copy_ops) or (
            "= f8e4m3fn[" in line and "copy(" in line
        )
        if not is_copy_line:
            continue
        for m in mlir_pool_pat.finditer(line):
            if int(m.group(1)) >= num_pages:
                return (
                    False,
                    f"unexpected full-pool copy in torch_tpu IR: {line.strip()}",
                )
        if hlo_pool_suffix in line:
            prefix = line.split(hlo_pool_suffix, 1)[0]
            dim_str = prefix.rsplit("[", 1)[-1].rsplit(",", 1)[-1].strip()
            if not dim_str.isdigit() or int(dim_str) >= num_pages:
                return (
                    False,
                    f"unexpected full-pool copy in torch_tpu HLO: {line.strip()}",
                )
    return True, "ok"


def export_pooled_step(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    op_name: str = "megakernel_runtime::exported_pooled_step",
    platforms: Sequence[str] = ("tpu",),
) -> PooledStepExport:
    """Export `make_flat_pooled_step` to a serialized StableHLO module.

    Note: At serving time, `build_pooled_step_op` traces through
    `sharded_jax_callable` (`JaxCallable`). This helper is provided for offline
    StableHLO inspection and serialization tests.
    """
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    avals = _build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    j_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    try:
        with jax._src.config.export_ignore_forward_compatibility(True):
            exported: Any = jax.export.export(
                j_fn,
                platforms=list(platforms),
                disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
            )(*avals)
        mlir_bytes = exported.mlir_module_serialized
    except NotImplementedError:
        if not ex.interpret:
            raise
        lowered = j_fn.lower(*avals)
        exported = lowered
        mlir_bytes = lowered.as_text("stablehlo").encode("utf-8")
    n_w = len(w_flat)
    n_p = len(raw_pools)
    aliases = {2 + n_w + i: 1 + i for i in range(n_p)}
    return PooledStepExport(
        op_name=op_name,
        exported=exported,
        mlir_module_serialized=mlir_bytes,
        in_specs=in_specs,
        out_specs=out_specs,
        donate_argnums=donate_argnums,
        input_output_aliases=aliases,
    )


def build_pooled_step_op(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    *,
    op_name: str | None = None,
    donate_pools: bool = True,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """Build a PyTorch custom op wrapping `cache.make_flat_pooled_step`.

    Returns `(op, in_specs, out_specs, donate_argnums)` where `op` takes
    `(pos, hidden, page_table, ckpt_indices, read_offsets, weights, pools)`.
    """
    global _OP_COUNTER
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=donate_pools
    )
    _OP_COUNTER += 1
    if op_name is None:
        op_name = (
            f"megakernel_runtime::step_t{ex.T}_b{ex.B}_p{ex.num_pages}_{_OP_COUNTER}"
        )
    elif "::" not in op_name:
        op_name = f"megakernel_runtime::{op_name}"

    fn_name = op_name.replace("::", "_")
    typed_fn = make_typed_step_wrapper(
        fn_name, step_fn, num_inputs=len(in_specs), num_outputs=len(out_specs)
    )
    flat_callable = sharded_jax_callable(
        op_name,
        typed_fn,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=donate_argnums,
    )

    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    cpu_jit_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )

    @torch.library.custom_op(op_name, mutates_args=())
    def op(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        all_tensors = [
            pos,
            hidden,
            *weights,
            *pools,
            page_table,
            ckpt_indices,
            read_offsets,
        ]
        if pos.device.type == "cpu":
            # CPU interpret execution path for unit tests without a TPU host.
            jax_inputs = [
                _torch_cpu_to_jax(t, sh)
                for t, sh in zip(all_tensors, in_sh, strict=True)
            ]
            jax_outs = cpu_jit_fn(*jax_inputs)
            like_tensors = [hidden, *pools]
            return [
                _jax_to_torch_cpu(o, lk)
                for o, lk in zip(jax_outs, like_tensors, strict=True)
            ]
        return list(flat_callable(*all_tensors))

    @op.register_fake
    def _(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [torch.empty_like(hidden)] + [torch.empty_like(p) for p in pools]

    return op, in_specs, out_specs, donate_argnums


def execute_pooled_step_inplace(
    op: Callable[..., list[torch.Tensor]],
    pos: torch.Tensor,
    hidden: torch.Tensor,
    page_table: torch.Tensor,
    ckpt_indices: torch.Tensor,
    read_offsets: torch.Tensor,
    weights: Sequence[torch.Tensor],
    pools: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Execute `op`, copy donated outputs into `pools` in-place, and return `hid_out`.

    Preserves `pool.data_ptr()` and `pool.storage().data_ptr()` on every pool
    tensor so `raiden`-pinned buffers remain valid without re-registration.
    """
    outs = op(
        pos,
        hidden,
        page_table,
        ckpt_indices,
        read_offsets,
        list(weights),
        list(pools),
    )
    n_p = len(pools)
    for dst, src in zip(pools, outs[1 : 1 + n_p]):
        dst.copy_(src)
    return outs[0]
