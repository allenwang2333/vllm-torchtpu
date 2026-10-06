# SPDX-License-Identifier: Apache-2.0
"""The schedule resolves page ids; the RPA kernel no longer reads the page table.

`rpa_metadata_schedule` writes HBM page ids (read from `page_indices`) into the
KV descriptors, so the RPA launches do not scalar-prefetch the whole
`max_num_seqs * pages_per_seq` table into SMEM. These tests pin that:

- the RPA pallas_calls take no page-table operand, the schedule does;
- relabelling physical pages (permuted cache + matching page table) leaves the
  output bit-identical and the logical KV cache bit-identical, for decode-only,
  mixed, small-bucket prefill, prefill-only and a full batch;
- the output matches a float32 reference and new K/V land in the right pages;
- positive controls: a page table shifted by one row (a wrong slice) is caught by
  the oracle, while garbage in rows past `distribution[2]` changes nothing.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jex_core

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
PAGE_SIZE = 128
HEAD_DIM = 256
NUM_Q_HEADS = 4
MAX_SEQS = 8
DECODE_QUERY_SIZE = 4
KV_DTYPE = jnp.float8_e4m3fn

# (q_lens, ctx_lens (cached tokens before this step), distribution).
DECODE_ONLY = ((4, 4, 3, 4), (1021, 4100, 7, 300), (4, 4, 4))
MIXED = ((4, 4, 300, 40), (2000, 130, 900, 0), (2, 2, 4))
SMALL_BUCKET_PREFILL = ((4, 4, 7), (1025, 4100, 520), (2, 2, 3))
PREFILL_ONLY = ((33, 120), (600, 0), (0, 0, 2))
MAX_BATCH = (
    (4,) * MAX_SEQS,
    (5000, 3, 129, 4096, 777, 2048, 6000, 255),
    (MAX_SEQS, MAX_SEQS, MAX_SEQS),
)
CASES = {
    "decode_only": DECODE_ONLY,
    "mixed": MIXED,
    "small_bucket_prefill": SMALL_BUCKET_PREFILL,
    "prefill_only": PREFILL_ONLY,
    "max_batch": MAX_BATCH,
}
PAGES_PER_SEQ = 64  # 8192 tokens per row; every case fits with room to spare.
EXTRA_PAGES = 16


def _inputs(case, seed=8228):
    q_lens, ctx_lens, distribution = case
    rng = np.random.default_rng(seed)
    total = sum(q_lens)
    num_tokens = max(64, -(-total // 64) * 64)
    num_pages = MAX_SEQS * PAGES_PER_SEQ + EXTRA_PAGES
    cache_shape = wrapper.get_kv_cache_shape(
        num_pages, PAGE_SIZE, 1, HEAD_DIM, KV_DTYPE, kv_layout=LAYOUT
    )

    def rand(shape, dtype):
        return np.asarray((rng.standard_normal(shape) * 0.5).astype(dtype))

    q = rand((num_tokens, NUM_Q_HEADS, HEAD_DIM), jnp.bfloat16)
    k = rand((num_tokens, 1, HEAD_DIM), KV_DTYPE)
    v = rand((num_tokens, 1, HEAD_DIM), KV_DTYPE)
    cache = rand(cache_shape, KV_DTYPE)
    page_table = rng.permutation(num_pages)[: MAX_SEQS * PAGES_PER_SEQ]
    page_table = page_table.reshape(MAX_SEQS, PAGES_PER_SEQ).astype(np.int32)
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(q_lens)] = np.add(ctx_lens, q_lens)
    assert kv_lens.max() <= PAGES_PER_SEQ * PAGE_SIZE
    cu_q_lens = np.full(MAX_SEQS + 1, total, np.int32)
    cu_q_lens[: len(q_lens) + 1] = np.cumsum((0, *q_lens))
    return dict(
        q=q,
        k=k,
        v=v,
        cache=cache,
        kv_lens=kv_lens,
        page_table=page_table,
        cu_q_lens=cu_q_lens,
        distribution=np.asarray(distribution, np.int32),
        q_lens=q_lens,
        ctx_lens=ctx_lens,
    )


_rpa = functools.partial(
    wrapper.ragged_paged_attention,
    sm_scale=HEAD_DIM**-0.5,
    kv_layout=LAYOUT,
    decode_query_size=DECODE_QUERY_SIZE,
)


def _run(inp, page_table=None, cache=None):
    pt = inp["page_table"] if page_table is None else page_table
    out, new_cache = _rpa(
        jnp.asarray(inp["q"]),
        jnp.asarray(inp["k"]),
        jnp.asarray(inp["v"]),
        jnp.asarray(inp["cache"] if cache is None else cache),
        jnp.asarray(inp["kv_lens"]),
        jnp.asarray(pt.reshape(-1)),
        jnp.asarray(inp["cu_q_lens"]),
        jnp.asarray(inp["distribution"]),
    )
    out, new_cache = jax.block_until_ready((out, new_cache))
    total = sum(inp["q_lens"])
    return np.asarray(out[:total].astype(jnp.float32)), np.asarray(new_cache)


def _logical_pages(cache, page_table, inp):
    """Every live page of every scheduled sequence, in logical order (raw bytes)."""
    rows = []
    for s, (ql, cl) in enumerate(zip(inp["q_lens"], inp["ctx_lens"])):
        n = -(-(cl + ql) // PAGE_SIZE)
        rows.append(cache[page_table[s, :n]].view(np.uint8))
    return rows


def _reference(inp):
    """float32 attention over the pre-step cache plus this step's new K/V."""
    cache = inp["cache"].astype(np.float32)  # (P, 2, D/packing, packing, page)
    q = inp["q"].astype(np.float32)
    k = inp["k"].astype(np.float32)[:, 0]
    v = inp["v"].astype(np.float32)[:, 0]
    pt = inp["page_table"]
    outs = []
    start = 0
    for s, (ql, cl) in enumerate(zip(inp["q_lens"], inp["ctx_lens"])):
        npg = -(-cl // PAGE_SIZE)
        g = cache[pt[s, :npg]].transpose(0, 4, 1, 2, 3)
        g = g.reshape(npg * PAGE_SIZE, 2, HEAD_DIM)[:cl]
        kk = np.concatenate([g[:, 0], k[start : start + ql]])
        vv = np.concatenate([g[:, 1], v[start : start + ql]])
        logits = np.einsum("qhd,kd->hqk", q[start : start + ql], kk) * HEAD_DIM**-0.5
        mask = (cl + np.arange(ql))[:, None] >= np.arange(cl + ql)[None, :]
        logits = np.where(mask[None], logits, -np.inf)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        outs.append(np.einsum("hqk,kd->qhd", p, vv))
        start += ql
    return np.concatenate(outs)


def _rel_err(out, ref):
    return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))


def _new_tokens_misplaced(inp, new_cache, page_table):
    """Count new K/V tokens not found in their (logical) page slot."""
    nc = new_cache.astype(np.float32)
    k = inp["k"].astype(np.float32)[:, 0]
    v = inp["v"].astype(np.float32)[:, 0]
    bad = 0
    start = 0
    for s, (ql, cl) in enumerate(zip(inp["q_lens"], inp["ctx_lens"])):
        for t in range(ql):
            pos = cl + t
            slot = nc[page_table[s, pos // PAGE_SIZE], :, :, :, pos % PAGE_SIZE]
            slot = slot.reshape(2, HEAD_DIM)
            bad += not (
                np.array_equal(slot[0], k[start + t])
                and np.array_equal(slot[1], v[start + t])
            )
        start += ql
    return bad


def _pallas_operand_shapes(jaxpr, found=None):
    if found is None:
        found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append([tuple(x.aval.shape) for x in eqn.invars])
            continue
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                if isinstance(sub, jex_core.ClosedJaxpr):
                    _pallas_operand_shapes(sub.jaxpr, found)
                elif isinstance(sub, jex_core.Jaxpr):
                    _pallas_operand_shapes(sub, found)
    return found


def test_only_schedule_takes_the_page_table():
    inp = _inputs(MIXED)
    args = (
        inp["q"],
        inp["k"],
        inp["v"],
        inp["cache"],
        inp["kv_lens"],
        inp["page_table"].reshape(-1),
        inp["cu_q_lens"],
        inp["distribution"],
    )
    calls = _pallas_operand_shapes(jax.make_jaxpr(_rpa)(*args).jaxpr)
    pt_shape = (MAX_SEQS * PAGES_PER_SEQ,)
    # The schedule is the only pallas_call that reads `distribution` (s32[3]).
    sched = [c for c in calls if (3,) in c]
    rpa = [c for c in calls if (3,) not in c]
    assert len(sched) == 2 and len(rpa) == 2, calls
    for shapes in sched:
        assert pt_shape in shapes, shapes
    for shapes in rpa:
        assert pt_shape not in shapes, shapes


@pytest.mark.parametrize("case", list(CASES))
def test_page_relabel_is_bit_exact(case):
    """Physical page placement must not matter: bit-identical out and cache."""
    inp = _inputs(CASES[case])
    out, new_cache = _run(inp)

    assert np.isfinite(out).all()
    ref = _reference(inp)
    assert _rel_err(out, ref) < 2e-2
    assert _new_tokens_misplaced(inp, new_cache, inp["page_table"]) == 0

    # Relabel: physical page p moves to sigma[p].
    sigma = np.random.default_rng(1).permutation(inp["cache"].shape[0])
    cache2 = np.empty_like(inp["cache"])
    cache2[sigma] = inp["cache"]
    pt2 = sigma[inp["page_table"]].astype(np.int32)
    out2, new_cache2 = _run(inp, page_table=pt2, cache=cache2)

    np.testing.assert_array_equal(out2, out)
    for a, b in zip(
        _logical_pages(new_cache2, pt2, inp),
        _logical_pages(new_cache, inp["page_table"], inp),
    ):
        np.testing.assert_array_equal(a, b)


def test_inactive_rows_are_not_read():
    """Rows at or past distribution[2] may hold anything (bit-exact)."""
    inp = _inputs(SMALL_BUCKET_PREFILL)
    out, new_cache = _run(inp)
    pt = inp["page_table"].copy()
    n = int(inp["distribution"][2])
    # Valid page ids, but pointing at other sequences' pages.
    # Valid page ids aliasing sequence 0's pages: a read would change the output
    # of nothing scheduled, but any write through them would corrupt sequence 0.
    pt[n:] = pt[0]
    out2, new_cache2 = _run(inp, page_table=pt)
    np.testing.assert_array_equal(out2, out)
    np.testing.assert_array_equal(new_cache2.view(np.uint8), new_cache.view(np.uint8))


@pytest.mark.parametrize("case", ["decode_only", "max_batch"])
def test_wrong_slice_is_caught(case):
    """Positive control: a page table one row off must fail the oracle."""
    inp = _inputs(CASES[case])
    shifted = np.roll(inp["page_table"], 1, axis=0)
    out, new_cache = _run(inp, page_table=shifted)
    ref = _reference(inp)
    caught_out = _rel_err(out, ref) > 0.2
    caught_cache = _new_tokens_misplaced(inp, new_cache, inp["page_table"]) > 0
    assert caught_out and caught_cache
