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
"""Custom BufferedRef overrides for StreamIndex Top-K kernel."""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexKVBufferedRef(pltpu.BufferedRef):
    """Handles fetching paged KV cache into VMEM buffers across sequence sublanes."""

    bkv_p: int = dataclasses.field(default=1, metadata=dict(static=True))
    page_size_per_kv_packing: int = dataclasses.field(
        default=64, metadata=dict(static=True))
    kv_packing: int = dataclasses.field(default=4, metadata=dict(static=True))
    pages_per_seq: int = dataclasses.field(
        default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(
        default=1, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bkv_p: int,
        page_size_per_kv_packing: int,
        kv_packing: int,
        pages_per_seq: int,
        seq_batch_size: int,
        grid_rank: int = 1,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
        )

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        cache_kv_hbm, page_indices_ref, meta_ref = src_ref
        p_id = grid_indices[0]
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        assert isinstance(cache_kv_hbm, jax.Ref)
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error

        reshaped_cache_hbm_ref: Any = cache_kv_hbm.reshape(  # pytype: disable=attribute-error
            cache_kv_hbm.shape[0] * self.page_size_per_kv_packing,
            self.kv_packing,
            cache_kv_hbm.shape[-1],
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
        num_page_indices = page_indices_ref.shape[0]
        effective_seq_idx = batch_tile_idx

        for batch_idx in range(self.seq_batch_size):
            kv_p_start = bkv_idx * self.bkv_p
            page_indices_offset = (
                effective_seq_idx +
                batch_idx) * self.pages_per_seq + kv_p_start
            for i in range(self.bkv_p):
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)
                safe_page_offset = jnp.minimum(
                    page_indices_ref[page_idx] * self.page_size_per_kv_packing,
                    jnp.maximum(0,
                                max_hbm_pages - self.page_size_per_kv_packing),
                )
                pltpu.make_async_copy(
                    reshaped_cache_hbm_ref.at[pl.ds(
                        safe_page_offset, self.page_size_per_kv_packing)],
                    vmem_dst.at[  # pytype: disable=attribute-error
                        batch_idx,
                        pl.ds(
                            i * self.page_size_per_kv_packing,
                            self.page_size_per_kv_packing,
                        ),
                    ],
                    sem,
                ).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref, Any],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error
        pltpu.make_async_copy(
            vmem_dst,
            vmem_dst,
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexQBufferedRef(pltpu.BufferedRef):
    """Handles fetching query slices into VMEM buffers."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(
        default=1, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
    ):
        assert buffer_type == pltpu.BufferType.INPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def input(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
        )

    def copy_in(  # pytype: disable=attribute-error
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        q_hbm, cu_q_lens_ref, meta_ref = src_ref
        p_id = grid_indices[0]
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bq_idx = meta_ref.bq_idx[p_id]

        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error
        effective_seq_idx = batch_tile_idx

        if self.seq_batch_size > 1:
            sz = 1
            q_len_start = cu_q_lens_ref[effective_seq_idx]
        elif self.bq_sz == 1:
            sz = 1
            q_len_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
        else:
            q_len_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
            curr_q_end = cu_q_lens_ref[effective_seq_idx + 1]
            sz = jnp.maximum(0,
                             jnp.minimum(self.bq_sz,
                                         curr_q_end - q_len_start))

        total_sz = self.seq_batch_size * sz

        pltpu.make_async_copy(
            q_hbm.at[pl.ds(q_len_start, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],  # pytype: disable=attribute-error
            sem,
        ).start()

    def wait_in(
        self,
        src_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_input
        if not self.is_buffered:
            return
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_dst: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error
        p_id = grid_indices[0]
        meta_ref = src_ref[2]
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bq_idx = meta_ref.bq_idx[p_id]
        effective_seq_idx = batch_tile_idx

        if self.seq_batch_size > 1:
            total_sz = self.seq_batch_size
        elif self.bq_sz == 1:
            total_sz = self.seq_batch_size * self.bq_sz
        else:
            _, cu_q_lens_ref, _ = src_ref
            q_len_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
            curr_q_end = cu_q_lens_ref[effective_seq_idx + 1]
            sz = jnp.maximum(0,
                             jnp.minimum(self.bq_sz,
                                         curr_q_end - q_len_start))
            total_sz = self.seq_batch_size * sz

        pltpu.make_async_copy(
            vmem_dst.at[pl.ds(0, total_sz)],
            vmem_dst.at[pl.ds(0, total_sz)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StreamIndexOBufferedRef(pltpu.BufferedRef):
    """Handles scattering computed Top-K scores back to HBM."""

    bq_sz: int = dataclasses.field(default=16, metadata=dict(static=True))
    num_sublanes: int = dataclasses.field(default=1, metadata=dict(static=True))
    seq_batch_size: int = dataclasses.field(
        default=1, metadata=dict(static=True))

    @classmethod
    def create(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
    ):
        assert buffer_type == pltpu.BufferType.OUTPUT
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=grid_rank,
            use_lookahead=use_lookahead,
        )
        return cls(
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    @classmethod
    def output(  # pytype: disable=signature-mismatch
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: Any,
        buffer_count: int,
        use_lookahead: bool,
        bq_sz: int,
        num_sublanes: int,
        seq_batch_size: int = 1,
        grid_rank: int = 1,
        **kwargs: Any,
    ):
        return cls.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=pltpu.BufferType.OUTPUT,
            buffer_count=buffer_count,
            use_lookahead=use_lookahead,
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            grid_rank=grid_rank,
        )

    def copy_out(  # pytype: disable=attribute-error
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        scores_hbm, cu_q_lens_ref, meta_ref = dst_ref
        p_id = grid_indices[0]
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bq_idx = meta_ref.bq_idx[p_id]
        bkv_idx = meta_ref.bkv_idx[p_id]

        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error
        effective_seq_idx = batch_tile_idx

        if self.seq_batch_size > 1:
            sz = 1
            token_start = cu_q_lens_ref[effective_seq_idx]
        elif self.bq_sz == 1:
            sz = 1
            token_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
        else:
            token_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
            curr_q_end = cu_q_lens_ref[effective_seq_idx + 1]
            sz = jnp.maximum(0,
                             jnp.minimum(self.bq_sz,
                                         curr_q_end - token_start))

        total_sz = self.seq_batch_size * sz
        sublane_start = bkv_idx * self.num_sublanes

        pltpu.make_async_copy(
            vmem_src.at[pl.ds(0, total_sz)],  # pytype: disable=attribute-error
            scores_hbm.at[
                pl.ds(token_start, total_sz),
                pl.ds(sublane_start, self.num_sublanes),
            ],
            sem,
        ).start()

    def wait_out(
        self,
        dst_ref: tuple[Any, ...],
        grid_indices: tuple[int | jax.Array, ...],
    ):
        assert self.is_output
        if not self.is_buffered:
            return
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_src: Any = self.window_ref.at[
            slot]  # pytype: disable=attribute-error
        p_id = grid_indices[0]
        meta_ref = dst_ref[2]
        batch_tile_idx = meta_ref.batch_tile_idx[p_id]
        bq_idx = meta_ref.bq_idx[p_id]
        effective_seq_idx = batch_tile_idx

        if self.seq_batch_size > 1:
            total_sz = self.seq_batch_size
        elif self.bq_sz == 1:
            total_sz = self.seq_batch_size * self.bq_sz
        else:
            _, cu_q_lens_ref, _ = dst_ref
            token_start = cu_q_lens_ref[effective_seq_idx] + bq_idx * self.bq_sz
            curr_q_end = cu_q_lens_ref[effective_seq_idx + 1]
            sz = jnp.maximum(0,
                             jnp.minimum(self.bq_sz,
                                         curr_q_end - token_start))
            total_sz = self.seq_batch_size * sz

        pltpu.make_async_copy(
            vmem_src.at[pl.ds(0, total_sz)],
            vmem_src.at[pl.ds(0, total_sz)],
            sem,
        ).wait()
