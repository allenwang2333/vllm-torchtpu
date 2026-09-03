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
"""Ref bundles of the fused permute + grouped matmul.

The scratch keeps gmm_v2's layout verbatim, so `inner_kernel` and `zero_out_*`
keep their contract; the gather ring is spliced in ahead of the zero-init pair,
which must stay last.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.megablox.fused_permute_gmm import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GatherRefs:
    """The ring the per-row DMAs land in: one slot per in-flight tile."""
    buf: Any  # [num_slots, tile_m, ring_chunks, lanes] pool dtype
    sem: Any  # DMA[num_slots]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineScratchRef:
    """What `emit_pipeline` hands the body: gmm_v2's scratch, then the ring."""
    partial_out: Any  # [sublane, tile_n] out dtype
    acc: Any  # [tile_m, acc_columns] acc dtype
    metadata: gmm_v2.MetadataRef
    gather: GatherRefs


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScratchRef:
    pipeline: PipelineScratchRef
    zero: Any | None  # [zero_tile_rows, lanes] out dtype, under zero_initialize
    zero_sem: Any | None  # DMA[1], under zero_initialize

    @classmethod
    def create_scratch_types(cls, cfg: config.Config) -> "ScratchRef":
        tiles, dims = cfg.tiles, cfg.dims
        pipeline = PipelineScratchRef(
            partial_out=pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n),
                                   cfg.gmm.out_dtype),
            acc=pltpu.VMEM((tiles.tile_m, cfg.acc_columns), cfg.gmm.acc_dtype),
            metadata=gmm_v2.MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((cfg.max_num_gm, ), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((cfg.max_num_gm + 1, ),
                                             jnp.int32),
            ),
            gather=GatherRefs(
                buf=pltpu.VMEM(cfg.gather_buffer_shape,
                               cfg.gmm.lhs_cfgs.dtype),
                sem=pltpu.SemaphoreType.DMA((cfg.tuning.num_slots, )),
            ),
        )
        if not cfg.gmm.zero_init:
            return cls(pipeline=pipeline, zero=None, zero_sem=None)
        return cls(
            pipeline=pipeline,
            zero=pltpu.VMEM((cfg.zero_tile_rows, cfg.num_lanes),
                            cfg.gmm.out_dtype),
            zero_sem=pltpu.SemaphoreType.DMA((1, )),
        )
