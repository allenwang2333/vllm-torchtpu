"""Option B -- the chunked stage-A / stage-B pipeline, built outside `src/`.

Nothing in `src/vllm_torchtpu/` is modified.  `streamindex_topk.py`,
`sparsecore_topk.py` and `mla_attention_op.py` are imported unedited; the only
transcription is the stage-A `pallas_call` launcher, and that is not
transcribed here either -- it is imported from `probes/_stage_a.py`, which is
already a verbatim copy of `run_scores_kernel` committed at step 2.

What this module adds
---------------------
1. `chunk_plan(phase, G)` -- how the row axis splits.  Decode splits over
   *sequences*; prefill splits one chunked-prefill chunk over its *query
   tokens*, presented to stage A as G pseudo-sequences whose `seq_lens`
   advance by `q_len/G` each.  Both are exactly semantics-preserving: the
   per-token causal position and the per-token `eff_row_length` are unchanged
   by the split, which is checked in `probe_flip3.py --arm plan`.

2. `stage_b(...)` -- stage B for one chunk, with the two hard assertions §5
   requires:
      * `_pick_partition` is never called on a chunked path  (`p_tripwire`)
      * `num_waves * 32 == b_pad`                            (`waves_guard`)

3. `build_body(...)` -- the three timing arms, all inside one `jax.jit`,
   which P3 established is the only shape in which TC and SC overlap at all.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_PROBES = _HERE.parent.parent / "probes"
if str(_PROBES) not in sys.path:
    sys.path.insert(0, str(_PROBES))

import _stage_a  # noqa: E402  (probes/_stage_a.py, committed at step 2)
from vllm_torchtpu.kernels.deepseek_v4 import \
    sparsecore_topk as scmod  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4 import \
    streamindex_topk as simod  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    MlaCase, _effective_row_lengths, align_to, prepare_index_weights,
    prepare_q_inputs)

LANES = 16


def cdiv(a, b):
    return -(-a // b)


def source_md5(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def module_md5s() -> dict:
    return {
        "streamindex_topk.py": {
            "path": simod.__file__,
            "md5": source_md5(simod.__file__)
        },
        "sparsecore_topk.py": {
            "path": scmod.__file__,
            "md5": source_md5(scmod.__file__)
        },
        "_stage_a.py": {
            "path": _stage_a.__file__,
            "md5": source_md5(_stage_a.__file__)
        },
        "pipeline.py": {
            "path": str(_HERE / "pipeline.py"),
            "md5": source_md5(_HERE / "pipeline.py")
        },
    }


# ---------------------------------------------------------------------------
# Geometry -- the two production operating points from
# `instruments/bench_streamidx_topk.py` (`prod_decode_b64_ctx8704` and
# `prod_prefill_q1024_ctx9216`), which are the rows `report.md` models.
# ---------------------------------------------------------------------------

BASE = dict(
    max_model_len=9216,
    page_size=1024,
    kv_packing=4,
    lkv_dim=256,
    h_idx=32,
    d_idx=128,
    k=2048,
    compression_ratio=1,
    bkv_p=2,
    total_pages=576,  # 64 x 9; prefill borrows 9 of them so its DMA is not
    #                   reading a 9-page toy region
)


def geometry(phase: str) -> dict:
    g = dict(BASE)
    g["phase"] = phase
    g["pages_per_seq"] = cdiv(g["max_model_len"], g["page_size"])  # 9
    g["num_sublanes_total"] = (align_to(g["pages_per_seq"], g["bkv_p"]) *
                               g["page_size"] // 128)  # 80
    g["num_positions"] = g["num_sublanes_total"] * 128  # 10240
    if phase == "decode":
        g.update(num_seqs=64, q_len=1, context_len=8704, seq_batch_size=4,
                 bq_per_block=1, static_q_len=1, case=MlaCase.DECODE)
    elif phase == "prefill":
        g.update(num_seqs=1, q_len=1024, context_len=9216, seq_batch_size=1,
                 bq_per_block=128, static_q_len=None, case=MlaCase.MIXED)
    else:
        raise ValueError(phase)
    g["num_tokens"] = g["num_seqs"] * g["q_len"]
    return g


def legal_chunk_counts(phase: str) -> tuple[int, ...]:
    return (1, 2, 4)


def chunk_plan(phase: str, G: int) -> list[dict]:
    """Static (numpy-level) description of the G chunks.

    Decode: chunk `i` owns sequences `[i*S, (i+1)*S)`, `S = num_seqs / G`.
    Prefill: chunk `i` owns query tokens `[i*T, (i+1)*T)` of the one prefill
    chunk, `T = q_len / G`, presented as a pseudo-sequence of length
    `context_len - q_len + (i+1)*T`.  Token `t` of chunk `i` then sits at
    absolute position `context_len - q_len + i*T + t`, which is exactly where
    it sits in the unchunked call.
    """
    g = geometry(phase)
    plan = []
    if phase == "decode":
        S, r = divmod(g["num_seqs"], G)
        assert r == 0, (g["num_seqs"], G)
        assert S % g["seq_batch_size"] == 0, (
            f"chunk of {S} seqs is not a multiple of decode_req_batch_size "
            f"{g['seq_batch_size']}")
        for i in range(G):
            plan.append(
                dict(idx=i, tok0=i * S, num_tokens=S, num_seqs=S,
                     seq0=i * S,
                     seq_lens=np.full((S, ), g["context_len"], np.int32),
                     cu_q_lens=np.arange(S + 1, dtype=np.int32),
                     distribution=np.array([S, S, S], np.int32)))
    else:
        T, r = divmod(g["q_len"], G)
        assert r == 0, (g["q_len"], G)
        assert T % g["bq_per_block"] == 0 or T >= g["bq_per_block"], T
        for i in range(G):
            sl = g["context_len"] - g["q_len"] + (i + 1) * T
            plan.append(
                dict(idx=i, tok0=i * T, num_tokens=T, num_seqs=1, seq0=0,
                     seq_lens=np.array([sl], np.int32),
                     cu_q_lens=np.array([0, T], np.int32),
                     distribution=np.array([0, 0, 1], np.int32)))
    return plan


def build_inputs(phase: str, seed: int = 0):
    """Device-resident inputs for the whole (unchunked) op, plus the per-chunk
    slices of them.  Construction copied from `_stage_a.build_nightly_inputs`
    so the score distribution matches every earlier probe."""
    g = geometry(phase)
    keys = jax.random.split(jax.random.key(seed), 4)

    q = jax.random.normal(keys[0], (g["num_tokens"], g["h_idx"], g["d_idx"]),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    weights = jax.random.uniform(keys[1], (g["num_tokens"], g["h_idx"]),
                                 jnp.float32, 0.25, 1.75).astype(jnp.bfloat16)

    shape = (g["total_pages"], g["page_size"] // g["kv_packing"],
             g["kv_packing"], g["lkv_dim"])
    raw = jax.random.bits(keys[2], shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(g["lkv_dim"], dtype=jnp.int32)
    cache_kv = jnp.where(
        lane < g["d_idx"], vals,
        jnp.where(lane == g["d_idx"], jnp.uint8(127), jnp.uint8(0)))

    perm = np.asarray(
        jax.random.permutation(keys[3],
                               jnp.arange(g["total_pages"], dtype=jnp.int32)))
    pps = g["pages_per_seq"]

    q_prep = prepare_q_inputs(q)
    w_prep = prepare_index_weights(weights, q.dtype)

    shared = dict(cache_kv=jax.block_until_ready(cache_kv),
                  q=jax.block_until_ready(q_prep),
                  weights=jax.block_until_ready(w_prep),
                  page_perm=perm)
    return g, shared


def chunk_arrays(phase: str, G: int, shared: dict) -> list[dict]:
    """Per-chunk device arrays.  Every array is a *slice* of the whole-op
    input, so the chunked path reads exactly the bytes the unchunked one does."""
    g = geometry(phase)
    pps = g["pages_per_seq"]
    plan = chunk_plan(phase, G)
    out = []
    for c in plan:
        t0, nt = c["tok0"], c["num_tokens"]
        if phase == "decode":
            pages = shared["page_perm"][c["seq0"] * pps:(c["seq0"] +
                                                         c["num_seqs"]) * pps]
        else:
            pages = shared["page_perm"][:pps]
        eff = _effective_row_lengths(
            seq_lens=jnp.asarray(c["seq_lens"]),
            cu_q_lens=jnp.asarray(c["cu_q_lens"]),
            distribution=jnp.asarray(c["distribution"]),
            num_tokens=nt,
            num_positions=g["num_positions"],
            compression_ratio=g["compression_ratio"],
        )
        out.append(
            dict(
                idx=c["idx"],
                num_tokens=nt,
                q=jax.block_until_ready(shared["q"][t0:t0 + nt]),
                weights=jax.block_until_ready(shared["weights"][t0:t0 + nt]),
                seq_lens=jnp.asarray(c["seq_lens"]),
                page_indices=jnp.asarray(pages, jnp.int32),
                cu_q_lens=jnp.asarray(c["cu_q_lens"]),
                eff=jax.block_until_ready(eff),
                start=jnp.int32(0),
                end=jnp.int32(c["num_seqs"]),
            ))
    return out


def stage_a_fn(phase: str):
    g = geometry(phase)
    return _stage_a.make_stage_a_fn(
        num_q_heads=g["h_idx"],
        head_dim=g["d_idx"],
        page_size_per_kv_packing=g["page_size"] // g["kv_packing"],
        kv_packing=g["kv_packing"],
        lkv_dim=g["lkv_dim"],
        num_sublanes_total=g["num_sublanes_total"],
        num_kv_pages_per_block=g["bkv_p"],
        num_queries_per_block=g["bq_per_block"],
        static_q_len=g["static_q_len"],
        seq_batch_size=g["seq_batch_size"],
        case=g["case"],
    )


# ---------------------------------------------------------------------------
# The two hard assertions
# ---------------------------------------------------------------------------


class PickPartitionCalled(AssertionError):
    pass


@contextlib.contextmanager
def p_tripwire(active: bool = True):
    """`_pick_partition` must never be reached on a chunked path.  Any call
    raises, so the assertion is enforced by the tracer rather than inspected
    afterwards."""
    orig = scmod._pick_partition

    def boom(b, n, k):
        raise PickPartitionCalled(
            f"_pick_partition({b=}, {n=}, {k=}) was called on a chunked path")

    if active:
        scmod._pick_partition = boom
    try:
        yield
    finally:
        scmod._pick_partition = orig


def waves_guard(b: int) -> dict:
    """`num_waves * 32 == b_pad` -- the guard that stops a chunk leaving half
    the 32 subcores idle.  Not a threshold: a chunk that fails it must not be
    chunked to."""
    b_pad = align_to(b, LANES)
    num_waves = cdiv(b, 32)
    return dict(b=b, b_pad=b_pad, num_waves=num_waves,
                lhs=num_waves * 32, ok=bool(num_waves * 32 == b_pad))


def stage_b(scores2d, k, eff, *, force_p1: bool):
    """Stage B for one chunk.

    `force_p1=True` calls `_sc_topk_direct` -- the single-stage kernel -- so
    `_pick_partition` is not consulted at all and no `take_along_axis` gather
    can be introduced.  Option E established on hardware (426 cases x every
    feasible `p` = 1100 arms, 0 index-set mismatches) that lowering `p` is
    exactness-preserving, which is what makes this legal.
    """
    if force_p1:
        return scmod._sc_topk_direct(scores2d, k, eff, write_empty=True)
    return scmod.sparsecore_topk(scores2d, k, row_lengths=eff,
                                 write_empty_rows=True)


# ---------------------------------------------------------------------------
# The timing arms.
#
# Every arm chains R copies inside ONE `jax.jit` with a real data dependency,
# so the slope of t(R) is dispatch-free device time -- the method established
# in `probes/stage-b-ksweep.md` §0.  The two requirements are in tension only
# apparently: the *overlap* needs stage A and stage B inside one program, and
# chain-slope needs R copies inside one program.  Both are satisfied by one
# program containing R copies of the whole G-chunk pipeline.
#
# Two carriers link consecutive copies, and neither is foldable:
#
#   minA(s)  = min(argmax(s[0, 0]), 0)   -- identically 0, from a stage-A
#                                           score buffer.  Requires XLA to
#                                           know argmax >= 0 to fold it.
#   minB(tk) = min(tk[0, 0], 0)          -- identically 0, from a top-k index
#                                           output.  The carrier
#                                           `probe_stage_b_ksweep.py` used,
#                                           where 35/35 fits came back
#                                           r^2 >= 0.9997.
#
# They are used so that arm `a` and arm `pipe` pay the *same* number of
# `-inf` score-buffer broadcasts (R x G) and the same number of `minA`
# reductions, and arm `pipe`'s carrier is exactly arm `a`'s plus arm `b`'s.
# `hidden = t_a + t_b - t_pipe` therefore has every non-kernel term cancel.
# ---------------------------------------------------------------------------

ARMS = ("a", "b", "b_chained", "pipe")

CHUNK_FIELDS = ("q", "weights", "seq_lens", "page_indices", "cu_q_lens",
                "eff", "start", "end")


def flat_args(chunk_arrs, cache_kv, premade=()):
    """The jit argument tuple: (per-chunk arrays, cache_kv, premade scores)."""
    chunks = tuple(
        tuple(c[f] for f in CHUNK_FIELDS) for c in chunk_arrs)
    return (chunks, cache_kv, tuple(premade))


def _minA(s):
    return jnp.minimum(jnp.argmax(s[0, 0]).astype(jnp.int32), jnp.int32(0))


def _minB(tk):
    return jnp.minimum(tk[0, 0], jnp.int32(0))


def build_body(phase, G, arm, R, *, force_p1, return_outputs=False):
    """Returns `f(chunks, cache_kv, premade)`.

      a           G stage-A calls per copy, no stage B.
      b           G stage-B calls per copy on pre-made scores, no stage A.
                  The G calls within a copy are mutually independent.
      b_chained   as `b`, but the G calls within a copy are chained to each
                  other -- the control that measures how much SparseCore
                  overlaps *itself*.
      pipe        the pipeline: stage B of chunk i consumes stage A of chunk
                  i's real output.  Chunk i+1's stage A does not depend on
                  chunk i's stage B, so XLA is free to overlap them; copy r+1
                  depends on every one of copy r's stage-B outputs, so copies
                  cannot overlap and the measured saving is the within-copy
                  one only.
    """
    assert arm in ARMS, arm
    g = geometry(phase)
    k = g["k"]
    n = g["num_positions"]
    sub = g["num_sublanes_total"]
    sa = stage_a_fn(phase)
    plan = chunk_plan(phase, G)

    def body(chunks, cache_kv, premade):
        eps = jnp.int32(0)
        tail = []
        for r in range(R):
            acc = jnp.int32(0)
            tail = []
            for i, c in enumerate(chunks):
                (q, w, seq_lens, page_indices, cu_q_lens, eff, start,
                 end) = c
                nt = plan[i]["num_tokens"]
                if arm in ("a", "pipe"):
                    init = jnp.full((nt, sub, 128), -jnp.inf, jnp.float32)
                    s = sa(q, w, cache_kv, init, seq_lens, page_indices,
                           cu_q_lens, start + eps, end + eps)
                    acc = acc + _minA(s)
                    if arm == "a":
                        tail.append(s)
                        continue
                    tk = stage_b(s.reshape(nt, n), k, eff, force_p1=force_p1)
                else:
                    rl = eff + eps
                    tk = stage_b(premade[i], k, rl, force_p1=force_p1)
                    if arm == "b_chained":
                        eps = _minB(tk)
                acc = acc + _minB(tk)
                tail.append(tk)
            eps = acc
        if return_outputs:
            return (eps, ) + tuple(tail)
        return (eps, )

    return body
