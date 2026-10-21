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
"""A stand-in step kernel that writes the bound caches the way the real one
does: a communicating pallas_call (collective_id, a barrier across the mesh)
DMA-writing cache pages, either in place (`inplace`: the caches are plain
inputs, none returned) or into outputs aliased to donated inputs.

Every cache array: page `ids[0] % n` <- page `(ids[0] + 1) % n` when any
request is active. hidden = (ids, the kernel's busy flag, 0, ...); aux[a] =
(a + 2) * hidden. TPU only (Mosaic DMA).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import allreduce as AR
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI


def page_copy_call(n_ranks: int, cache_shapes, inplace: bool):
    n_c = len(cache_shapes)
    n_out_c = 0 if inplace else n_c

    def kernel(meta, ids, seq, *refs):
        caches_in, done = refs[:n_c], refs[n_c]
        caches = caches_in if inplace else refs[n_c + 1 : 2 * n_c + 1]
        sem = refs[-1]
        if n_ranks > 1:
            AR.barrier(n_ranks, meta[KI.META_RANK])
        busy = jnp.int32(0)
        for r in range(seq.shape[0]):
            busy = jnp.maximum(busy, (seq[r] > 0).astype(jnp.int32))

        @pl.when(busy > 0)
        def _():
            for c in caches:
                n = c.shape[0]
                dst, src = ids[0] % n, (ids[0] + 1) % n
                cp = pltpu.make_async_copy(c.at[src], c.at[dst], sem)
                cp.start()
                cp.wait()

        done[...] = jnp.full(done.shape, busy, jnp.int32)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    n_in = 3 + n_c
    return pl.pallas_call(
        kernel,
        in_specs=[smem] * 3 + [anyspace] * n_c,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [anyspace] * n_out_c,
        out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.int32)]
        + [jax.ShapeDtypeStruct(s, d) for s, d in cache_shapes[:n_out_c]],
        input_output_aliases={n_in - n_c + i: 1 + i for i in range(n_out_c)},
        scratch_shapes=[pltpu.SemaphoreType.DMA(())],
        compiler_params=pltpu.CompilerParams(collective_id=AR.COLLECTIVE_ID),
        name="page_copy_inplace" if inplace else "page_copy_aliased",
    )


class PageCopyKernel:
    """`KernelImpl` over `page_copy_call`; weights are passed through unread."""

    def __init__(self, inplace: bool, n_weights: int, hidden: int):
        self.cache_inplace = inplace
        self.n_weights = n_weights
        self.hidden = hidden

    def collect_weights(self, model, spec):
        raise NotImplementedError("weights are handed to the adapter directly")

    def build_step(self, cfg: KI.StepConfig, bucket: KI.Bucket):
        M, n_bt = bucket.own_rows, len(cfg.group_widths)
        n_aux = len(cfg.model.aux_layers)
        shapes = [(tuple(s), jnp.dtype(d)) for s, d in cfg.cache_shapes]
        call = page_copy_call(cfg.model.num_ranks, shapes, self.cache_inplace)
        skip = n_bt + self.n_weights

        def fn(meta, ids, pos, seq, *rest):
            caches = rest[skip:]
            outs = call(meta, ids, seq, *caches)
            done = outs[0]
            h = jnp.zeros((M, self.hidden), jnp.float32)
            h = h.at[:, 0].set(ids.astype(jnp.float32))
            h = h.at[:, 1].set(done[0, 0].astype(jnp.float32))
            head = [h] + [h * (a + 2) for a in range(n_aux)]
            return head if self.cache_inplace else head + list(outs[1:])

        return fn
