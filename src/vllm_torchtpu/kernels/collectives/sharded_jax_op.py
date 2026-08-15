# SPDX-License-Identifier: Apache-2.0
"""``pallas.jax_op`` for kernels that run under a multi-device ``shard_map``.

Stock ``torch_tpu``'s ``pallas.jax_op`` builds its ``jax.jit`` without
``out_shardings``.  For an ordinary single-device kernel that is fine, but a
``shard_map`` whose body has side effects (any collective does) hits two
problems:

* ``jax.export`` records shard_map outputs in ``out_avals`` as *replicated*, so
  the torch output tensors get allocated at the global shape instead of this
  device's shard;
* XLA's SPMD partitioner refuses the program outright --
  ``Side-effect HLO must have sharding`` (``spmd_partitioner.cc``) -- because
  the side-effecting custom call carries no sharding annotation.

Passing explicit ``out_shardings`` to the ``jax.jit`` fixes both.

This mirrors ``kernels/experimental/pcp_streaming_rpa/vllm_adapter.py``, which
solved the same problem for context-parallel attention.  It is kept separate
rather than imported from there so that a collective used by every model does
not depend on an experimental attention kernel; the PCP adapter additionally
carries kernel-iteration hot-reload support, which this does not need.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

__all__ = ["sharded_jax_op"]


def _torch_placeholder_with_sharding(
        aval: jax.core.ShapedArray | None,
        sharding: jax.sharding.NamedSharding | None,
        mesh: jax.sharding.Mesh | None) -> torch.Tensor | None:
    """Allocate this device's *local* shard of a shard_map output."""
    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(
            f"Unsupported dtype for pallas kernels: {aval.dtype}")
    spec = getattr(sharding, "spec", None)
    if spec is None:
        spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return torch.empty(pallas_impl.get_local_shape(aval.shape, mesh, spec),
                       dtype=torch_dtype,
                       device="tpu")


class _ShardedJaxCallable(pallas_impl.JaxCallable):
    """``JaxCallable`` that sizes its outputs from explicit out_shardings."""

    def __call__(self, *args, **kwargs):
        self._validate_args(*args)

        kernel_key = pallas_impl._get_kernel_invocation_key(
            self.trace_key, args, kwargs, self.static_argnums)
        output_shapes, out_tree = self.output_shapes.get(
            kernel_key, (None, None))
        kernel_exists = pallas_impl.tpu_torch_pallas.lookup_custom_kernel(
            self.name, kernel_key)
        if not output_shapes or not kernel_exists:
            jax_args = pallas_impl.jax_placeholders(
                args,
                mesh=self.mesh,
                partition_specs=self.input_partition_specs,
            )
            with jax._src.config.export_ignore_forward_compatibility(True):
                lowered = self.exported(*jax_args, **kwargs)
            pallas_impl.tpu_torch_pallas.register_custom_kernel(
                self.name,
                kernel_key,
                serialized_mlir_module=lowered.mlir_module_serialized,
            )
            out_shardings = getattr(lowered, "_out_named_shardings", None)
            if (out_shardings is None
                    or len(out_shardings) != len(lowered.out_avals)):
                out_shardings = [None] * len(lowered.out_avals)
            output_shapes = [
                _torch_placeholder_with_sharding(aval, sharding, self.mesh)
                for aval, sharding in zip(lowered.out_avals, out_shardings)
            ]
            out_tree = lowered.out_tree
            self.output_shapes[kernel_key] = (output_shapes, out_tree)

        tensor_args = [
            arg for i, arg in enumerate(args)
            if arg is not None and i not in self.static_argnums
        ]
        results = pallas_impl.tpu_torch_pallas.call_custom_kernel(
            self.name,
            kernel_key,
            inputs=tensor_args,
            output_shapes=output_shapes,
            donate_argnums=self.donate_argnums,
        )

        for in_idx, out_idx in self.input_output_aliases.items():
            tensor_args[in_idx].copy_(results[out_idx])

        return out_tree.unflatten(results)


def _named_shardings(mesh: jax.sharding.Mesh,
                     partition_specs: Sequence[PartitionSpec]):
    return tuple(
        jax.sharding.NamedSharding(mesh, spec) for spec in partition_specs)


def build_sharded_callable(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec],
    donate_argnums: Sequence[int] | None = None,
) -> _ShardedJaxCallable:
    signature = inspect.signature(fn, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    output_shardings = _named_shardings(mesh, tuple(output_partition_specs))
    jit_fn = jax.jit(fn,
                     static_argnums=static_argnums,
                     donate_argnums=donate_argnums_tuple,
                     out_shardings=output_shardings)
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums":
            static_argnums,
            "donate_argnums":
            donate_argnums_tuple,
            "output_partition_specs":
            tuple(map(str, tuple(output_partition_specs))),
        },
    )
    # Export in Shardy, JAX 0.10's default, and leave it that way: the XLA
    # inside torch_tpu has the Shardy partitioner enabled, and feeding it a
    # GSPMD-lowered module fails with
    #     expected CustomCallOp with a sharding attribute ... the model has
    #     been lowered targeting GSPMD, but Shardy was then enabled in XLA
    # The two dialects have to agree; this is the half that is not ours to
    # choose.
    return _ShardedJaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )


def sharded_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec],
    donate_argnums: Sequence[int] | None = None,
):
    """Register a multi-device ``shard_map`` JAX function as a torch custom op.

    ``input_partition_specs`` / ``output_partition_specs`` describe how each
    torch tensor (a per-device shard) maps onto the global array over ``mesh``.
    """
    wrapped_fn = build_sharded_callable(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
        donate_argnums=donate_argnums,
    )

    result = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args, **kwargs):
        jax_args = pallas_impl.jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        out_shardings = getattr(lowered, "_out_named_shardings", None)
        if out_shardings is None or len(out_shardings) != len(
                lowered.out_avals):
            out_shardings = [None] * len(lowered.out_avals)
        return lowered.out_tree.unflatten(
            _torch_placeholder_with_sharding(aval, sharding, mesh)
            for aval, sharding in zip(lowered.out_avals, out_shardings))

    result.register_fake(fake_fn)
    return result
