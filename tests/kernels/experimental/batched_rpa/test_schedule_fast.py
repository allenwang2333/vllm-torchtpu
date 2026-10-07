# SPDX-License-Identifier: Apache-2.0
"""TPU_RPA_SCHEDULE_FAST and decode_only produce the incumbent's results.

`schedule_fast` changes how `rpa_metadata_schedule` fills the schedule (a
branch-free descriptor path for full KV blocks, a per-task cache-page count for
the kv-in wait, and an HBM write of only the produced steps). It must not
change a single word of the schedule in the steps the RPA kernel reads
(`< actual_steps`), for either KV layout, a table-end clamp, a padded lane or a
schedule that overflows SMEM and flushes more than once.

`decode_only` drops the MIXED pass; for a batch without MIXED sequences the
output and the KV cache are bit-identical to running both passes.

Positive controls: a different kv length changes the compared schedule, and
decode_only on a batch WITH a MIXED sequence changes the output.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule, wrapper

HEAD_DIM = 256
NUM_Q_HEADS = 4
PAGE_SIZE = 128
DQS = 4


def _cfgs(mode, layout, num_seqs, pages_per_seq, fast, block=None):
    model = configs.ModelConfigs(
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        mask_value=float(jnp.finfo(jnp.bfloat16).min),
        sm_scale=HEAD_DIM**-0.5,
    )
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=PAGE_SIZE,
        total_q_tokens=1024,
        num_page_indices=num_seqs * pages_per_seq,
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.float8_e4m3fn),
        dtype_out=jnp.dtype(jnp.bfloat16),
        scale_k=1.0,
        scale_v=1.0,
        kv_layout=layout,
        decode_query_size=DQS,
        schedule_fast=fast,
    )
    vmem = pltpu.get_tpu_info().vmem_capacity_bytes
    dec, pre = wrapper.calculate_block_sizes(
        model, serve, vmem, decode_query_size=DQS, decode_batch_size=2
    )
    if block is None:
        block = dec if mode == configs.RpaCase.DECODE else pre
    return configs.RpaConfigs(
        block=block, model=model, serve=serve, mode=mode, vmem_limit_bytes=vmem
    )


def _schedule(cfgs, q_lens, kv_lens, distribution, page_table):
    n = cfgs.serve.num_seqs
    ql = np.zeros(n, np.int32)
    kl = np.zeros(n, np.int32)
    ql[: len(q_lens)] = q_lens
    kl[: len(kv_lens)] = kv_lens
    cu = np.concatenate([[0], np.cumsum(ql)]).astype(np.int32)
    cache_lens = kl - ql
    out = schedule.generate_rpa_metadata(
        jnp.asarray(cu),
        jnp.asarray(cache_lens),
        jnp.asarray(cache_lens),
        jnp.asarray(ql),
        jnp.asarray(np.asarray(distribution, np.int32)),
        jnp.asarray(page_table.reshape(-1)),
        cfgs=cfgs,
    )
    return jax.block_until_ready(out)


def _used_rows(sched, cfgs):
    """Every leaf of the schedule restricted to the steps the kernel reads."""
    steps = int(np.asarray(sched.actual_steps)[0])
    rows = cfgs.max_steps_ub * cfgs.max_schedule_size_multiplier
    out = {"actual_steps": steps}
    for path, leaf in jax.tree_util.tree_leaves_with_path(sched):
        a = np.asarray(leaf)
        if a.size == 1:
            continue
        es = a.size // rows
        out[jax.tree_util.keystr(path)] = a[: steps * es]
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


SEQ = configs.KVLayout.SEQ_ALONG_LANE
HEAD = configs.KVLayout.HEAD_ALONG_SUBLANE

# (q_lens, kv_lens (incl. new), distribution)
CASES = {
    "decode": ((4, 4, 3, 4, 1), (93000, 4100, 7, 30000, 14336), (5, 5, 5)),
    "decode_padded_lane": ((4, 4, 4), (14336 * 2 + 4, 14340, 900), (3, 3, 3)),
    "mixed": ((4, 4, 300, 40), (60000, 134, 1200, 40), (2, 2, 4)),
    "max_batch": (
        (4,) * 8,
        (5000, 7, 129, 40960, 777, 2048, 60000, 255),
        (8, 8, 8),
    ),
}


def _table(num_seqs, pages_per_seq, seed=8228):
    rng = np.random.default_rng(seed)
    return (
        rng.permutation(num_seqs * pages_per_seq + 64)[: num_seqs * pages_per_seq]
        .astype(np.int32)
        .reshape(num_seqs, pages_per_seq)
    )


@pytest.mark.parametrize("layout", [SEQ, HEAD], ids=["seq", "head"])
@pytest.mark.parametrize("mode", [configs.RpaCase.DECODE, configs.RpaCase.MIXED])
@pytest.mark.parametrize("case", sorted(CASES))
def test_schedule_fast_same_schedule(layout, mode, case):
    q_lens, kv_lens, dist = CASES[case]
    num_seqs, pps = 8, 520  # 520 pages: not a multiple of a 112-page block
    table = _table(num_seqs, pps)
    a = _used_rows(
        _schedule(_cfgs(mode, layout, num_seqs, pps, False), q_lens, kv_lens, dist, table),
        _cfgs(mode, layout, num_seqs, pps, False),
    )
    b = _used_rows(
        _schedule(_cfgs(mode, layout, num_seqs, pps, True), q_lens, kv_lens, dist, table),
        _cfgs(mode, layout, num_seqs, pps, True),
    )
    _assert_same(a, b)


@pytest.mark.parametrize("layout", [SEQ, HEAD], ids=["seq", "head"])
def test_schedule_fast_table_end_clamp(layout):
    # The last sequence slot's last block runs past the table end: the general
    # (clamped) path, next to full blocks that take the fast path.
    num_seqs, pps = 4, 300
    table = _table(num_seqs, pps)
    # The long request sits in the last slot (empty rows in front).
    q = (0, 0, 4, 4)
    kv = (0, 0, 20000, pps * PAGE_SIZE)
    dist = (4, 4, 4)
    args = (q, kv, dist, table)
    a = _used_rows(_schedule(_cfgs(configs.RpaCase.DECODE, layout, num_seqs, pps, False), *args),
                   _cfgs(configs.RpaCase.DECODE, layout, num_seqs, pps, False))
    b = _used_rows(_schedule(_cfgs(configs.RpaCase.DECODE, layout, num_seqs, pps, True), *args),
                   _cfgs(configs.RpaCase.DECODE, layout, num_seqs, pps, True))
    assert a["actual_steps"] > 0
    _assert_same(a, b)


def test_schedule_fast_multi_flush():
    # bkv = 128 tokens (1 page) -> one task per page: 8 x 594 pages = 4752 tasks
    # = 2376 steps > max_steps_ub (2304 on tpu7x at this size): two flushes.
    num_seqs, pps = 8, 600
    table = _table(num_seqs, pps)
    block = configs.BlockSizes(bq_sz=DQS, bq_c_sz=DQS, bkv_sz=128, batch_size=2, n_buffer=3)
    q, kv, dist = (4,) * 8, (76000,) * 8, (8, 8, 8)
    ca = _cfgs(configs.RpaCase.DECODE, SEQ, num_seqs, pps, False, block)
    cb = _cfgs(configs.RpaCase.DECODE, SEQ, num_seqs, pps, True, block)
    a = _used_rows(_schedule(ca, q, kv, dist, table), ca)
    b = _used_rows(_schedule(cb, q, kv, dist, table), cb)
    assert a["actual_steps"] > ca.max_steps_ub
    _assert_same(a, b)


def test_schedule_compare_can_fail():
    # Positive control: one kv length shorter changes the compared rows.
    num_seqs, pps = 8, 520
    table = _table(num_seqs, pps)
    q_lens, kv_lens, dist = CASES["decode"]
    c = _cfgs(configs.RpaCase.DECODE, SEQ, num_seqs, pps, True)
    a = _used_rows(_schedule(c, q_lens, kv_lens, dist, table), c)
    kv2 = (kv_lens[0] - 129,) + kv_lens[1:]
    b = _used_rows(_schedule(c, q_lens, kv2, dist, table), c)
    with pytest.raises(AssertionError):
        _assert_same(a, b)


# ------------------------------------------------------------------ decode_only

MAX_SEQS = 8
PPS = 160


def _rpa_inputs(q_lens, ctx_lens, distribution, seed=8229):
    rng = np.random.default_rng(seed)
    total = sum(q_lens)
    num_tokens = max(64, -(-total // 64) * 64)
    num_pages = MAX_SEQS * PPS + 16
    shape = wrapper.get_kv_cache_shape(
        num_pages, PAGE_SIZE, 1, HEAD_DIM, jnp.float8_e4m3fn, kv_layout=SEQ
    )

    def rand(s, dt):
        return np.asarray((rng.standard_normal(s) * 0.5).astype(dt))

    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(q_lens)] = np.add(ctx_lens, q_lens)
    cu = np.full(MAX_SEQS + 1, total, np.int32)
    cu[: len(q_lens) + 1] = np.cumsum((0, *q_lens))
    table = rng.permutation(num_pages)[: MAX_SEQS * PPS].astype(np.int32)
    return (
        rand((num_tokens, NUM_Q_HEADS, HEAD_DIM), jnp.bfloat16),
        rand((num_tokens, 1, HEAD_DIM), jnp.float8_e4m3fn),
        rand((num_tokens, 1, HEAD_DIM), jnp.float8_e4m3fn),
        rand(shape, jnp.float8_e4m3fn),
        kv_lens,
        table,
        cu,
        np.asarray(distribution, np.int32),
    ), total


def _call(args, **kw):
    f = functools.partial(
        wrapper.ragged_paged_attention,
        sm_scale=HEAD_DIM**-0.5,
        kv_layout=SEQ,
        decode_query_size=DQS,
        **kw,
    )
    out, cache = f(*(jnp.asarray(a) for a in args))
    out, cache = jax.block_until_ready((out, cache))
    return np.asarray(out.astype(jnp.float32)), np.asarray(cache).view(np.uint8)


DECODE_BATCH = ((4, 4, 3, 1), (16000, 4100, 7, 14335), (4, 4, 4))
MIXED_BATCH = ((4, 4, 300), (16000, 130, 900), (2, 2, 3))


@pytest.mark.parametrize("fast", [False, True], ids=["sched", "sched_fast"])
def test_decode_only_bit_exact(fast):
    args, total = _rpa_inputs(*DECODE_BATCH)
    o0, c0 = _call(args)
    o1, c1 = _call(args, decode_only=True, schedule_fast=fast)
    np.testing.assert_array_equal(o0[:total], o1[:total])
    np.testing.assert_array_equal(c0, c1)


def test_schedule_fast_bit_exact_mixed():
    args, total = _rpa_inputs(*MIXED_BATCH)
    o0, c0 = _call(args)
    o1, c1 = _call(args, schedule_fast=True)
    np.testing.assert_array_equal(o0[:total], o1[:total])
    np.testing.assert_array_equal(c0, c1)


def test_decode_only_with_mixed_rows_differs():
    # Positive control: the MIXED row's output (and its new KV) is not computed.
    args, total = _rpa_inputs(*MIXED_BATCH)
    o0, c0 = _call(args)
    o1, c1 = _call(args, decode_only=True)
    assert not np.array_equal(o0[:total], o1[:total])
