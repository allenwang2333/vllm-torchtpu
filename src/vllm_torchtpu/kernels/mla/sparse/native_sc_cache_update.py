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
"""Aliased token writes into the native SparseCore cache representation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper


def update_cache(nope, rope, nope_words, rope_words, page, slot):
    """Write unique token destinations, preserving all other cache bytes.

    The persistent caches keep their gather-native shape and tiling. In
    particular, RoPE DMA writes only one 32-word quarter; independent tokens
    sharing a four-token group never read or overwrite each other's words.
    Out-of-range pages denote padding and issue no DMA.
    """
    if page.shape[0] == 0:
        return nope, rope
    tpu_info = pltpu.get_tpu_info()
    assert tpu_info is not None and tpu_info.sparse_core is not None
    info = tpu_info.sparse_core
    mesh = plsc.VectorSubcoreMesh(num_cores=info.num_cores,
                                  num_subcores=info.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    owners = info.num_cores * info.num_subcores
    tokens = page.shape[0]
    # Metadata DMA covers one SIMD vector even for a short token bucket.
    padded = pl.cdiv(tokens, 16) * 16
    page = jnp.pad(page, (0, padded - tokens), constant_values=nope.shape[0])
    slot = jnp.pad(slot, (0, padded - tokens))

    def body(nope_in, rope_in, n_in, r_in, p_in, s_in, nope_out, rope_out,
             n_buf, r_buf, p_buf, s_buf, sem):
        del nope_in, rope_in
        owner = jax.lax.axis_index(("core", "subcore"))

        @pl.loop(0, pl.cdiv(tokens, owners))
        def step(i):
            token = owner + i * owners

            @pl.when(token < tokens)
            def write():
                pltpu.sync_copy(p_in.at[pl.ds(token // 16 * 16, 16)], p_buf)
                pltpu.sync_copy(s_in.at[pl.ds(token // 16 * 16, 16)], s_buf)
                dst_page = plsc.load_gather(
                    p_buf, [jnp.full((16, ), token % 16, jnp.int32)])[0]
                dst_slot = plsc.load_gather(
                    s_buf, [jnp.full((16, ), token % 16, jnp.int32)])[0]

                @pl.when((dst_page >= 0) & (dst_page < nope_out.shape[0]))
                def valid():
                    n_copy = pltpu.make_async_copy(n_in.at[token, :], n_buf,
                                                   sem.at[0])
                    r_copy = pltpu.make_async_copy(r_in.at[token, :], r_buf,
                                                   sem.at[1])
                    n_copy.start()
                    r_copy.start()
                    n_copy.wait()
                    r_copy.wait()
                    n_copy = pltpu.make_async_copy(
                        n_buf, nope_out.at[dst_page, dst_slot, :], sem.at[0])
                    r_copy = pltpu.make_async_copy(
                        r_buf, rope_out.at[dst_page, dst_slot // 4,
                                           pl.ds(dst_slot % 4 * 32, 32)],
                        sem.at[1])
                    n_copy.start()
                    r_copy.start()
                    n_copy.wait()
                    r_copy.wait()

    return core_map_helper.kernel(
        body,
        out_type=(jax.ShapeDtypeStruct(nope.shape, nope.dtype),
                  jax.ShapeDtypeStruct(rope.shape, rope.dtype)),
        mesh=mesh,
        input_output_aliases={
            0: 0,
            1: 1
        },
        scratch_types={
            "n_buf": pltpu.VMEM((128, ), jnp.uint32),
            "r_buf": pltpu.VMEM((32, ), jnp.uint32),
            "p_buf": pltpu.VMEM((16, ), jnp.int32),
            "s_buf": pltpu.VMEM((16, ), jnp.int32),
            "sem": pltpu.SemaphoreType.DMA((2, )),
        },
        # Explicit vector gathers use native SC lowering without layout passes.
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=False,
                                             needs_layout_passes=False),
        name="sc_native_cache_update",
    )(nope, rope, nope_words, rope_words, page, slot)
