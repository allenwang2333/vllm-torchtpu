# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch side of ``kernels/deepseek_v41/host_row_gather.py``.

``HostRowTable`` moves a CPU ``[rows, row_bytes]`` table into TPU
``pinned_host`` memory (``torch.ops.ao.offload`` is the only TorchTPU entry
point that yields a pinned-host tensor; ``pin_memory().to("tpu")`` does not)
and gathers rows from it inside ``torch.compile`` graphs.

Two TorchTPU gaps are patched once, process-wide (``_install``):

- ``pallas.jax_placeholder`` drops the memory kind, so a pinned-host operand
  is traced as an HBM array; it now carries ``memory_kind="pinned_host"``.
- ``_backend._mark_pinned_host_inputs`` marks only ``ao.reload`` sources; the
  compiled graph's example inputs are fake tensors that lost the pinned flag,
  so placeholders feeding a ``HOST_OP_NAMESPACE`` op are marked too. Without
  it the operand is laid out in HBM and Mosaic fails with "Unsupported
  operand memory space".

Keep the chunks as plain tensor attributes: parameters and buffers can be
embedded into the compiled graph as constants.
"""

import functools
import inspect
from typing import Any

import jax
import torch

from vllm_torchtpu.kernels.deepseek_v41 import host_row_gather as hrg

HOST_OP_NAMESPACE = "dsv41_host_rows"
# Bytes of one pinned_host chunk. Each chunk is staged through HBM once at
# conversion, so this bounds the transient HBM use.
CHUNK_BYTES = 2 << 30


def _pinned_sharding(device: jax.Device) -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")


@functools.cache
def _install() -> None:
    import torch._functorch._activation_offloading.offload_ops  # noqa: F401
    from torch_tpu._internal.compile import _backend as tpu_backend
    from torch_tpu._internal.compile import compiler as tpu_compiler
    from torch_tpu._internal.pallas import pallas as tpu_pallas

    base_placeholder = tpu_pallas.jax_placeholder

    def jax_placeholder(tensor, mesh=None, partition_spec=None):
        result = base_placeholder(tensor, mesh, partition_spec)
        if (
            mesh is None
            and isinstance(result, jax.ShapeDtypeStruct)
            and tpu_compiler.is_pinned_host_tensor(tensor)
        ):
            result = jax.ShapeDtypeStruct(
                result.shape, result.dtype, sharding=_pinned_sharding(jax.devices()[0])
            )
        return result

    base_mark = tpu_backend._mark_pinned_host_inputs

    def mark_pinned_host_inputs(graph_module, example_inputs):
        base_mark(graph_module, example_inputs)
        placeholders = {
            node: i
            for i, node in enumerate(
                n for n in graph_module.graph.nodes if n.op == "placeholder"
            )
        }
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or HOST_OP_NAMESPACE not in str(node.target):
                continue
            # `lookup_and_dequant` takes (rows, valid, weight_scale_inv, *chunks);
            # `gather` takes (row_ids, *chunks).
            chunk_start = 3 if "lookup_and_dequant" in str(node.target) else 1
            for arg in node.args[chunk_start:]:
                index = placeholders.get(arg)
                if index is not None and isinstance(
                    example_inputs[index], torch.Tensor
                ):
                    example_inputs[index]._is_pinned_host = True

    tpu_pallas.jax_placeholder = jax_placeholder
    tpu_backend._mark_pinned_host_inputs = mark_pinned_host_inputs


@functools.cache
def _gather_op(num_chunks: int, row_bytes: int):
    """``jax_op`` gathering ``row_bytes`` rows from ``num_chunks`` host chunks.

    ``jax_op`` requires one annotated parameter per tensor, so the signature
    is built for the chunk count.
    """
    from torch_tpu._internal.pallas import pallas as tpu_pallas

    def gather(*args):
        return hrg.gather_rows_jax(args[0], *args[1:], row_bytes=row_bytes)

    gather.__signature__ = inspect.Signature(  # pyrefly: ignore[missing-attribute]
        [
            inspect.Parameter(
                f"arg{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
            )
            for i in range(1 + num_chunks)
        ],
        return_annotation=jax.Array,
    )
    op = tpu_pallas.jax_op(
        f"{HOST_OP_NAMESPACE}::gather_c{num_chunks}_b{row_bytes}", gather
    )

    def fake(row_ids, *chunks):
        # The default fake kernel rejects symbolic (bucketed token) dims.
        return row_ids.new_empty((row_ids.shape[0], hrg.row_words(row_bytes)))

    op.register_fake(fake)
    return op


@functools.cache
def _lookup_and_dequant_op(num_chunks: int, row_bytes: int, block_size: int):
    """``jax_op`` gathering ``row_bytes`` rows and dequantizing to bf16."""
    from torch_tpu._internal.pallas import pallas as tpu_pallas

    def fn(*args):
        return hrg.lookup_and_dequant_rows_jax(
            args[0],
            args[1],
            args[2],
            *args[3:],
            row_bytes=row_bytes,
            block_size=block_size,
        )

    fn.__signature__ = inspect.Signature(  # pyrefly: ignore[missing-attribute]
        [
            inspect.Parameter(
                f"arg{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
            )
            for i in range(3 + num_chunks)
        ],
        return_annotation=jax.Array,
    )
    op = tpu_pallas.jax_op(
        f"{HOST_OP_NAMESPACE}::lookup_and_dequant_c{num_chunks}_b{row_bytes}_s{block_size}",
        fn,
    )

    def fake(row_ids, valid, weight_scale_inv, *chunks):
        return row_ids.new_empty((row_ids.shape[0], row_bytes), dtype=torch.bfloat16)

    op.register_fake(fake)
    return op


def chunk_majors(row_bytes: int, chunk_bytes: int = CHUNK_BYTES) -> int:
    """Host-layout majors per chunk (``[majors, 8, 1024]`` int32 each)."""
    return max(1, chunk_bytes // (hrg.SUBLANES * hrg.MAJOR_LANES * 4))


class HostRowTable:
    """A ``[num_rows, row_bytes]`` table in TPU pinned_host memory."""

    def __init__(
        self,
        rows: torch.Tensor,
        device: torch.device,
        chunk_bytes: int = CHUNK_BYTES,
        capacity_rows: int = 0,
    ) -> None:
        """Pack and offload the CPU ``rows`` (any 1-byte dtype) chunk by chunk.

        Chunk count and shape depend only on ``max(num_rows, capacity_rows)``
        and ``row_bytes``, and every chunk has the same shape. Pass the
        largest row count any TP rank holds as ``capacity_rows``: chunk
        shapes are compiled into the graph, and ranks whose programs differ
        hang the engine (40L Engram, TD3.9a). Rows past ``num_rows`` are zero.

        The caller frees ``rows`` afterwards; peak extra host memory is one
        packed chunk.
        """
        assert rows.device.type == "cpu" and rows.dim() == 2, (rows.device, rows.shape)
        assert rows.element_size() == 1, rows.dtype
        _install()
        self.num_rows, self.row_bytes = rows.shape
        per_major = hrg.TILES_PER_MAJOR * hrg.rows_per_tile(self.row_bytes)
        total_majors = max(1, -(-max(self.num_rows, capacity_rows) // per_major))
        num_chunks = -(-total_majors // chunk_majors(self.row_bytes, chunk_bytes))
        majors = -(-total_majors // num_chunks)
        rows_per_chunk = majors * per_major
        host = rows.view(torch.uint8).numpy()
        self.chunks: list[torch.Tensor] = []
        for start in range(0, num_chunks * rows_per_chunk, rows_per_chunk):
            packed = hrg.pack_rows(host[start : start + rows_per_chunk], majors)
            staged = torch.from_numpy(packed).to(device)
            self.chunks.append(torch.ops.ao.offload(staged))
            del staged, packed
        # Registered here, outside tracing: Dynamo cannot trace `jax_op`.
        self._op = _gather_op(len(self.chunks), self.row_bytes)
        self._lookup_ops: dict[int, Any] = {}

    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        """``int32[n]`` in-range row ids -> ``uint8[n, row_bytes]`` on device."""
        if row_ids.dtype != torch.int32:
            row_ids = row_ids.to(torch.int32)
        return self._op(row_ids, *self.chunks).view(torch.uint8)

    def lookup_and_dequant(
        self,
        row_ids: torch.Tensor,
        valid: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Gather ``row_ids`` from ``pinned_host`` chunks and dequantize to bf16."""
        op = self._lookup_ops.get(block_size)
        if op is None:
            op = _lookup_and_dequant_op(len(self.chunks), self.row_bytes, block_size)
            self._lookup_ops[block_size] = op
        if row_ids.dtype != torch.int32:
            row_ids = row_ids.to(torch.int32)
        if valid.dtype != torch.bool:
            valid = valid.to(torch.bool)
        return op(
            row_ids,
            valid,
            weight_scale_inv,
            *self.chunks,
        )
