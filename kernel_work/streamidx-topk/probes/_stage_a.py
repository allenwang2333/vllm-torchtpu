"""Shared helper for P1 and P4: launch stage A's `pallas_call` in isolation.

`run_scores_kernel` is a closure nested inside `streamindex_topk`
(`src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py:607-739`), so it
cannot be imported.  The launch below is a **verbatim transcription** of that
closure with the enclosing scope's values passed in explicitly.  It imports the
incumbent's `kernel`, `MlaCase`, `align_to`, `get_dtype_packing`,
`prepare_q_inputs` and `prepare_index_weights` rather than re-implementing them,
so the *body* under study is the incumbent's own code; only the launch wrapper
is transcribed.

Nothing in `streamindex_topk.py` is modified.  `source_md5()` prints the md5 of
the module this was transcribed from, so drift is detectable from the job log.

Why the transcription is needed at all: `report.md` §6 records that "stage A
cannot be invoked in isolation without editing the incumbent", which is what
forced its time to be *derived* by subtraction.  This file removes that
limitation for the two probes that need a stage-A-only launch (P1's LLO capture
and P4's empty-grid timing) without touching the incumbent.
"""

from __future__ import annotations

import functools
import hashlib
import pathlib

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sx
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DEFAULT_VMEM_LIMIT_BYTES, MlaCase, align_to, kernel, prepare_index_weights,
    prepare_q_inputs)


def source_md5() -> str:
    p = pathlib.Path(sx.__file__)
    return hashlib.md5(p.read_bytes()).hexdigest()


def source_path() -> str:
    return sx.__file__


def make_stage_a_fn(
    *,
    # static geometry, all read off the inputs by the incumbent
    num_q_heads: int,
    head_dim: int,
    page_size_per_kv_packing: int,
    kv_packing: int,
    lkv_dim: int,
    num_sublanes_total: int,
    # the three per-call statics
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    static_q_len: int | None,
    seq_batch_size: int,
    case: MlaCase = MlaCase.DECODE,
    compression_ratio: int = 1,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    out_dtype=jnp.float32,
):
    """Returns `f(q, weights, cache_kv, scores_init, seq_lens, page_indices,
    cu_q_lens, start_seq_idx, end_seq_idx) -> scores`.

    `start_seq_idx` / `end_seq_idx` stay **traced device scalars**, exactly as
    `_common_path` passes them, so the grid is dynamic and the same executable
    serves every grid extent -- including 0.  That is the property P4 needs.
    """
    page_size = page_size_per_kv_packing * kv_packing

    def f(q, prepared_indexer_weights, cache_kv, scores_init, seq_lens,
          page_indices, cu_q_lens, start_seq_idx, end_seq_idx):
        # ---- verbatim from run_scores_kernel, streamindex_topk.py:607-739 ----
        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # scores_init
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        bkv_double_buf = pltpu.VMEM(
            (2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing,
             lkv_dim), cache_kv.dtype)
        bq_double_bufq = pltpu.VMEM(
            (2, seq_batch_size, bq_sz, num_q_heads, head_dim), q.dtype)
        bq_weights_double_buf = pltpu.VMEM(
            (2, seq_batch_size, bq_sz, num_q_heads),
            prepared_indexer_weights.dtype)
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), out_dtype)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),
            jnp.full((2, ), -1, jnp.int32),
        )

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
        )
        # ---------------------------- end verbatim ---------------------------

    return f


# --------------------------------------------------------------------------
# The nightly (baseline-of-record) decode operating point.
#
#   GLM-5.2, nightly perf config: DP=16, TP=1, max_num_seqs=64,
#   max_model_len=9216, page size 1024, kv_packing 4, lkv_dim 256,
#   H=32, D=128, k=2048, compression_ratio=1, num_kv_pages_per_block=(2,2,2),
#   num_queries_per_block=(1,128,128), decode_req_batch_size=4.
#
#   ctx 8704 is the row report.md models: 64 seqs x 1 token, 9 pages/seq,
#   num_sublanes_total = align_to(9,2)*1024//128 = 80, num_positions = 10240,
#   score buffer = 64*80*128*4 B = 2.62 MB.
# --------------------------------------------------------------------------

NIGHTLY = dict(
    num_seqs=64,
    q_len=1,
    context_len=8704,
    max_model_len=9216,
    page_size=1024,
    kv_packing=4,
    lkv_dim=256,
    h_idx=32,
    d_idx=128,
    k=2048,
    compression_ratio=1,
    decode_req_batch_size=4,
    bkv_p=2,
    bq_per_block=1,
)


def nightly_geometry():
    g = dict(NIGHTLY)
    g["pages_per_seq"] = -(-g["max_model_len"] // g["page_size"])  # 9
    g["num_sublanes_total"] = (align_to(g["pages_per_seq"], g["bkv_p"]) *
                               g["page_size"] // 128)  # 80
    g["num_positions"] = g["num_sublanes_total"] * 128  # 10240
    g["num_tokens"] = g["num_seqs"] * g["q_len"]  # 64
    g["total_pages"] = g["num_seqs"] * g["pages_per_seq"]  # 576
    g["score_bytes"] = (g["num_tokens"] * g["num_sublanes_total"] * 128 * 4)
    return g


def build_nightly_inputs(seed: int = 0):
    """Device-resident nightly-decode inputs, same construction as the
    step-1 instrument's `build_inputs` (which is read-only and is not edited
    here; the record layout is reproduced so this file has no import-time
    dependency on it)."""
    g = nightly_geometry()
    keys = jax.random.split(jax.random.key(seed), 4)

    q = jax.random.normal(keys[0], (g["num_tokens"], g["h_idx"], g["d_idx"]),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    weights = jax.random.uniform(keys[1], (g["num_tokens"], g["h_idx"]),
                                 jnp.float32, 0.25,
                                 1.75).astype(jnp.bfloat16)

    shape = (g["total_pages"], g["page_size"] // g["kv_packing"],
             g["kv_packing"], g["lkv_dim"])
    raw = jax.random.bits(keys[2], shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(g["lkv_dim"], dtype=jnp.int32)
    cache_kv = jnp.where(lane < g["d_idx"], vals,
                         jnp.where(lane == g["d_idx"], jnp.uint8(127),
                                   jnp.uint8(0)))

    block_table = jax.random.permutation(
        keys[3], jnp.arange(g["total_pages"], dtype=jnp.int32)).reshape(
            g["num_seqs"], g["pages_per_seq"])

    seq_lens = jnp.full((g["num_seqs"], ), g["context_len"], jnp.int32)
    cu_q_lens = jnp.arange(g["num_seqs"] + 1, dtype=jnp.int32) * g["q_len"]

    return g, dict(
        q=prepare_q_inputs(q),
        weights=prepare_index_weights(weights, q.dtype),
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=block_table.reshape(-1),
        cu_q_lens=cu_q_lens,
    )


def nightly_decode_call1(g):
    """The launcher for stage A's *running* decode call (call 1 of three):
    `seq_batch_size = decode_req_batch_size = 4`, `bq_sz = 1`, `bkv_p = 2`.
    Grid = (end - start) // 4, so 64 sequences => grid 16."""
    return make_stage_a_fn(
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=1,
        seq_batch_size=g["decode_req_batch_size"],
        case=MlaCase.DECODE,
        compression_ratio=g["compression_ratio"],
    )


def nightly_decode_call2(g):
    """The launcher for stage A's *remainder* decode call (call 2 of three):
    `seq_batch_size = 1`.  This is the call that is empty by construction at
    every pure-decode step, because `mla_attention_op.py:183` builds
    `distribution = stack([rd[0], rd[0], rd[2]])`."""
    return make_stage_a_fn(
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=1,
        seq_batch_size=1,
        case=MlaCase.DECODE,
        compression_ratio=g["compression_ratio"],
    )
