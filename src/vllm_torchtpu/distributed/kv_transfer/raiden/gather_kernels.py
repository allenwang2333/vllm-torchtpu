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
"""Row copies of the head-unique landing gather: HBM rows between two tensors.

The kernel copies ``num_chunks`` leading-dim rows ``src[src_rows[i]]`` into
``dst[dst_rows[i]]`` with one async DMA per row. ``dst`` is donated to the
op and comes back as its output (``copy_rows``); writing into a pool a KV
connector registered with raiden goes through ``copy_rows_into``, the
runner's own pattern (``copy_pool_blocks``): donate, then ``copy_`` the
output back into the registered tensor. A write into a kernel input that is
not donated does not land (measured on tpu7x), so there is no such path.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_jax_import_error: BaseException | None = None
try:
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas as ttpu_pallas
except Exception as e:  # pragma: no cover
    _jax_import_error = e
    pl = cast(Any, None)
    pltpu = cast(Any, None)
    ttpu_pallas = None  # type: ignore[assignment]


def kernels_available() -> bool:
    return _jax_import_error is None


if _jax_import_error is None:

    def _copy_rows_body(num_chunks_ref, src_rows_ref, dst_rows_ref, src_ref, dst_ref):
        def body(sem):
            @pl.loop(0, num_chunks_ref[0])
            def _start(i):
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(src_rows_ref[i], 1)],
                    dst_ref.at[pl.ds(dst_rows_ref[i], 1)],
                    sem,
                ).start()

            @pl.loop(0, num_chunks_ref[0])
            def _wait(i):
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(src_rows_ref[i], 1)],
                    dst_ref.at[pl.ds(dst_rows_ref[i], 1)],
                    sem,
                ).wait()

        pl.run_scoped(body, pltpu.SemaphoreType.DMA)

    def _copy_rows_aliased_body(
        num_chunks_ref, src_rows_ref, dst_rows_ref, src_ref, dst_in_ref, dst_out_ref
    ):
        del dst_in_ref
        _copy_rows_body(
            num_chunks_ref, src_rows_ref, dst_rows_ref, src_ref, dst_out_ref
        )

    def _jax_copy_rows(
        num_chunks: jax.Array,
        src_rows: jax.Array,
        dst_rows: jax.Array,
        src: jax.Array,
        dst: jax.Array,
    ) -> jax.Array:
        return pl.pallas_call(
            _copy_rows_aliased_body,
            out_shape=jax.ShapeDtypeStruct(shape=dst.shape, dtype=dst.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                grid=(1,),
                num_scalar_prefetch=3,
                in_specs=[
                    pl.BlockSpec(memory_space=pl.ANY),
                    pl.BlockSpec(memory_space=pl.ANY),
                ],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
            ),
            input_output_aliases={4: 0},
        )(num_chunks, src_rows, dst_rows, src, dst)

    # Donating the destination is what lets XLA alias the output to it; a
    # copy that needs the destination's memory twice does not fit a decode
    # leg's HBM.
    _copy_rows_op = ttpu_pallas.jax_op(
        "pallas::head_unique_copy_rows", _jax_copy_rows, donate_argnums=(4,)
    )

    def _fake_copy_rows(num_chunks, src_rows, dst_rows, src, dst):
        del num_chunks, src_rows, dst_rows, src
        return torch.empty_like(dst)

    _copy_rows_op.register_fake(_fake_copy_rows)

else:  # pragma: no cover
    _copy_rows_op = cast(Any, None)


def chunk_args(
    src_rows: tuple[int, ...], dst_rows: tuple[int, ...], device: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernels' scalar-prefetch arguments for one row list."""
    return (
        torch.tensor([len(src_rows)], dtype=torch.int32, device=device),
        torch.tensor(src_rows, dtype=torch.int32, device=device),
        torch.tensor(dst_rows, dtype=torch.int32, device=device),
    )


def copy_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    chunks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Returns ``dst`` with rows of ``src`` copied in; ``dst`` is donated and must not be used again."""
    if _jax_import_error is not None:
        raise RuntimeError(f"gather kernels unavailable: {_jax_import_error!r}")
    num_chunks, src_rows, dst_rows = chunks
    return _copy_rows_op(num_chunks, src_rows, dst_rows, src, dst)


def copy_rows_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    chunks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Copies rows of ``src`` into the registered tensor ``dst``, which keeps its buffer."""
    out = copy_rows(src, dst, chunks)
    if out is not dst:
        dst.copy_(out)
