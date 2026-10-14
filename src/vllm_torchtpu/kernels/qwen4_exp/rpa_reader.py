# SPDX-License-Identifier: Apache-2.0
"""Qwen decode reader deriving regular DMA fields from the schedule base."""

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    bref_override,
    configs,
    schedule,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class QwenDecodeKVRef(bref_override.KVBufferedRefHeadAlongSublane):
    """Keep the shared buffer/wait/write contracts and per-page sizes."""

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ) -> None:
        if not (
            self.cfgs.mode == configs.RpaCase.DECODE
            and self.cfgs.serve.skip_kv_update
            and self.cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
            and self.cfgs.serve.page_size == 4
            and self.cfgs.serve.dtype_kv == jnp.float8_e4m3fn
            and self.cfgs.bkv_sz == 256
            and self.cfgs.batch_size == 8
        ):
            raise ValueError(
                "Qwen reader requires page4 FP8 skip-update HEAD decode, K256/batch8."
            )
        cache, new_kv, metadata, pages = src_ref
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        block = jnp.maximum(grid_indices[0], 0)
        dst = cast(jax.Ref, self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride])
        # Pallas refs support reshape; jax.Ref omits that method from its type.
        # pyrefly: ignore[missing-attribute]
        flat = cache.reshape(-1, *cache.shape[2:])
        cache_copies, new_copies = [], []
        for batch in range(self.cfgs.batch_size):
            base = metadata.dma_kv_cache[block, batch, 0, 0]
            for page in range(self.cfgs.bkv_p_cache):
                size = metadata.dma_kv_cache[block, batch, page, 2]
                # k_loop clamps src_i; zero-filled padding also yields legal
                # empty views. Keep sizes, selected page order and issue order.
                index = jnp.minimum(base + page, self.cfgs.serve.num_page_indices - 1)
                offset = page << self.cfgs.serve.page_size_log2
                source = pages[index] * self.cfgs.serve.page_size
                cache_copies.append((source, offset, size, batch))
            first = metadata.dma_kv_new[block, batch, 0]
            source = first.fetch_hbm[...]
            offset = first.fetch_vmem[...]
            size = 0
            for page in range(self.cfgs.bkv_p_new):
                entry = metadata.dma_kv_new[block, batch, page]
                assert isinstance(entry, schedule.HeadAlongSublaneDmaNew)
                size += entry.fetch_val
            new_copies.append((source, offset, size, batch))
        for source, offset, size, batch in cache_copies:
            pltpu.make_async_copy(
                flat.at[pl.ds(source, size)], dst.at[batch, pl.ds(offset, size)], sem
            ).start()
        for source, offset, size, batch in new_copies:
            pltpu.make_async_copy(
                new_kv.at[pl.ds(source, size)], dst.at[batch, pl.ds(offset, size)], sem
            ).start()
