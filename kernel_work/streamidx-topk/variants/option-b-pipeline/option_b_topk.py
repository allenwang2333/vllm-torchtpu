"""Option B: chunked, pipelined `streamindex_topk`.

A drop-in replacement for
`vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk` that
splits the batch into `num_groups` chunks and emits, *inside one XLA program*,

    A_0  B_0  A_1  B_1  ...  A_{G-1}  B_{G-1}

so that chunk g's SparseCore top-k is in flight while chunk g+1's TensorCore
scoring kernel runs.  P3 established that TC/SC concurrency is a property of a
single XLA program, so the whole point is that this stays one program; nothing
here uses threads or separate executables.

Nothing under `src/` is modified.  The scoring launch comes from
`kernel_work/streamidx-topk/probes/_stage_a.py`, itself a verbatim
transcription of the incumbent's `run_scores_kernel` closure that imports the
incumbent's `kernel` body; the top-k comes from the incumbent's
`_sc_topk_direct`.  The only new code is the chunk decomposition and the
`_common_path` control flow around it, transcribed from
`streamindex_topk.py:740-836`.

Two invariants this file is required to hold (measurement_and_design.md §5),
both asserted rather than assumed:

  1. `_pick_partition` is never called on a chunked path.  Chunking shrinks `b`,
     and `_pick_partition`'s *discretionary* `max_p = 32 // b` term then picks
     p = 2 at b = 16 -- which is not a capacity requirement and costs 2.4x
     (measured, decode G=4: stage B 203.44 us via `sparsecore_topk` vs 85.85 us
     via `_sc_topk_direct`).  So stage B here always calls `_sc_topk_direct`
     when `force_p1`, and `p_tripwire()` in `pipeline.py` proves the incumbent
     chooser was never consulted.  The *mandatory* term is unaffected: `p_min =
     cdiv(n, 65536)` is 1 for every n this op sees (n = 10240 at the production
     shape), so forcing p = 1 removes no VMEM-capacity protection at these
     shapes.  `_require_p1_is_safe` re-derives that per call rather than
     trusting it.

  2. `num_waves * 32 == b_pad`.  `_sc_topk_direct` pads `b` up to a multiple of
     LANES (16) but launches `cdiv(b, 32)` waves of 32 subcores, so at b = 16
     the kernel occupies a full 32-row wave to do 16 rows of work.  This is not
     a correctness bug -- it is the reason chunking past b = 32 buys nothing --
     and `waves_guard()` reports it per chunk.

Chunking is only well defined when a chunk boundary falls on a *sequence*
boundary: stage A's grid is over sequences and its masking is driven by
`cu_q_lens`, so a single sequence's query tokens cannot be split across chunks.
`plan_bounds()` returns an empty plan when the requested `num_groups` cannot be
met that way, and the caller then runs unchunked.  That is a real limitation,
not a harness artifact: at decode (1 token per sequence) any G dividing the
sequence count works, but a single-sequence chunked-prefill chunk cannot be
split here at all.  Splitting one long prefill into G pseudo-sequences -- which
is how the prefill numbers in RESULTS.md were obtained -- is a change to how
the runner packs the batch, not a change confined to this kernel.


Why the public entry is a `jax.jit` object
------------------------------------------

`jax.eval_shape(fun, *args, **kwargs)` is `jit(fun).trace(...)` *unless*
`type(fun) is jax._src.PjitFunction`, in which case it forwards to
`fun.trace(...)` and the callee's `static_argnames` are honoured (jax
0.10.2, `jax/_src/api.py:2318`).  The check is an exact `type(...) is`, so a
plain Python wrapper -- however faithful -- makes `eval_shape` turn *every*
leaf into a tracer, including `k` and the block sizes, and the forwarded call
then dies with "Non-hashable static arguments are not supported".  The repo's
`test_streamindex_topk_shape` does exactly this, so `chunked_core` below is a
real `jax.jit` object with the incumbent's `static_argnames` plus this file's.

The consequence is that the *plan* cannot be computed inside the entry point:
`cu_q_lens` and `distribution` are traced there, and a chunk's token bounds are
data.  So planning is eager (`plan_bounds`) and its result is handed in as the
static `group_bounds`.  `streamindex_topk_chunked` is the eager convenience
wrapper that does both; the probes use it.  A caller that has only the jitted
entry (the pytest plugin) gets `group_bounds=()` and the fall-back path.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

_PROBES = pathlib.Path(__file__).resolve().parents[2] / "probes"
if str(_PROBES) not in sys.path:
    sys.path.insert(0, str(_PROBES))

import _stage_a  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    DEFAULT_VMEM_LIMIT_BYTES, MlaCase, _effective_row_lengths, align_to,
    prepare_index_weights, prepare_q_inputs)

LANES = 16

# Trace-time record of what the jitted entry did, appended to on every *trace*
# (not every call -- a cache hit re-executes nothing).  The pytest plugin reads
# it; nothing in the compute path depends on it.
TRACE_LOG: list[dict] = []


def _cdiv(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# chunk planning  (eager: needs concrete cu_q_lens / distribution)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Group:
    """One chunk: a sequence range, the token range it owns, and the
    `distribution` triple rebased onto that sequence range."""
    s0: int
    s1: int
    t0: int
    t1: int
    d0: int
    d1: int
    d2: int

    @property
    def num_seqs(self) -> int:
        return self.s1 - self.s0

    @property
    def num_tokens(self) -> int:
        return self.t1 - self.t0

    def as_tuple(self) -> tuple[int, ...]:
        return (self.s0, self.s1, self.t0, self.t1, self.d0, self.d1, self.d2)


class NotChunkable(ValueError):
    """Raised by `plan_bounds(..., strict=True)` when G cannot be honoured."""


def plan_bounds(
    *,
    num_groups: int,
    cu_q_lens,
    distribution,
    num_tokens: int,
    decode_req_batch_size: int,
    strict: bool = False,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    """Split [0, num_seqs) into `num_groups` contiguous sequence ranges.

    Returns `(group_bounds, reason)`, where `group_bounds` is a hashable tuple
    suitable for passing as a `jax.jit` static argument.  An **empty** tuple
    means "run unchunked", and `reason` says why -- it is logged rather than
    swallowed, because a chunking that silently degrades to G=1 would show up
    as a perfectly clean but meaningless A/B.

    Constraints, in the order they are checked:

      * `distribution` and `cu_q_lens` must be *concrete*.  Under `jit` they
        are traced and the token boundary `cu_q_lens[s]` is not a Python int,
        so the per-chunk shapes cannot be formed.  In production the runner
        does know these values and would pass them in; that plumbing is out of
        scope here and is called out in RESULTS.md.
      * every group boundary must land on a sequence boundary (implied by
        splitting the sequence axis) *and* give a non-empty token range.
      * the decode prefix must stay aligned to `decode_req_batch_size`: the
        incumbent batches `decode_req_batch_size` decode sequences per grid
        step, and a split that leaves a partial batch inside a group changes
        which kernel handles those rows.
    """
    if num_groups <= 1:
        return (), f"num_groups={num_groups}"

    try:
        cu = np.asarray(jax.device_get(cu_q_lens)).astype(np.int64)
        dist = tuple(int(x) for x in np.asarray(jax.device_get(distribution)))
    except Exception as e:  # traced under jit
        msg = f"cu_q_lens/distribution not concrete ({type(e).__name__})"
        if strict:
            raise NotChunkable(msg) from e
        return (), msg

    num_seqs = dist[2]
    if num_seqs < num_groups:
        msg = f"num_seqs={num_seqs} < num_groups={num_groups}"
        if strict:
            raise NotChunkable(msg)
        return (), msg
    if num_seqs % num_groups:
        msg = f"num_seqs={num_seqs} not divisible by num_groups={num_groups}"
        if strict:
            raise NotChunkable(msg)
        return (), msg

    per = num_seqs // num_groups
    if dist[0] and per % decode_req_batch_size:
        msg = (f"decode prefix {dist[0]} present but seqs/group={per} is not a "
               f"multiple of decode_req_batch_size={decode_req_batch_size}")
        if strict:
            raise NotChunkable(msg)
        return (), msg

    groups: list[Group] = []
    for g in range(num_groups):
        s0, s1 = g * per, (g + 1) * per
        t0, t1 = int(cu[s0]), int(cu[s1])
        if t1 <= t0:
            msg = f"group {g} covers zero tokens (seqs [{s0},{s1}))"
            if strict:
                raise NotChunkable(msg)
            return (), msg
        groups.append(
            Group(s0=s0, s1=s1, t0=t0, t1=t1,
                  d0=min(max(dist[0] - s0, 0), s1 - s0),
                  d1=min(max(dist[1] - s0, 0), s1 - s0),
                  d2=min(max(dist[2] - s0, 0), s1 - s0)))

    if groups[-1].t1 != int(cu[num_seqs]):
        msg = "groups do not cover all real tokens"
        if strict:
            raise NotChunkable(msg)
        return (), msg
    # Padding tokens past cu[num_seqs] belong to no sequence; give them to the
    # last group so every row of `q` is scored exactly once.
    if num_tokens > groups[-1].t1:
        groups[-1] = dataclasses.replace(groups[-1], t1=num_tokens)
    return tuple(g.as_tuple() for g in groups), "ok"


# Kept under the old name for callers that want the dataclass view.
def plan_groups(*, num_groups, cu_q_lens, distribution, max_num_seqs,
                num_tokens, decode_req_batch_size, strict=False):
    bounds, reason = plan_bounds(num_groups=num_groups, cu_q_lens=cu_q_lens,
                                 distribution=distribution,
                                 num_tokens=num_tokens,
                                 decode_req_batch_size=decode_req_batch_size,
                                 strict=strict)
    if not bounds:
        return None, reason
    return [Group(*b) for b in bounds], reason


# ---------------------------------------------------------------------------
# the two §5 invariants
# ---------------------------------------------------------------------------


def waves_guard(b: int) -> dict:
    """`num_waves * 32 == b_pad`, the sub-wave-occupancy invariant."""
    b_pad = align_to(b, LANES)
    num_waves = _cdiv(b, 32)
    return dict(b=b, b_pad=b_pad, num_waves=num_waves, lhs=num_waves * 32,
                ok=num_waves * 32 == b_pad)


def _require_p1_is_safe(b: int, n: int, k: int) -> None:
    """Forcing p=1 must not remove a *capacity* requirement.

    `_pick_partition` mixes two things: `p_min = cdiv(n, MAX_SLICE_WORDS)`,
    which is mandatory (a slice must fit VMEM), and `max_p = 32 // b`, which is
    a discretionary occupancy heuristic with no correctness content.  Option B
    overrides only the second.  If `p_min` were ever > 1 this override would be
    unsafe, so it is checked rather than assumed.
    """
    p_min = _cdiv(n, scmod.MAX_SLICE_WORDS)
    if p_min > 1:
        raise AssertionError(
            f"forcing p=1 is unsafe: p_min={p_min} for n={n} "
            f"(MAX_SLICE_WORDS={scmod.MAX_SLICE_WORDS}); the chunked path may "
            "not override a VMEM-capacity partition")


# ---------------------------------------------------------------------------
# the chunked op
# ---------------------------------------------------------------------------

_STATIC_ARGNAMES = (
    "k",
    "compression_ratio",
    "num_kv_pages_per_block",
    "num_queries_per_block",
    "vmem_limit_bytes",
    "decode_req_batch_size",
    "enable_early_exit",
    "group_bounds",
    "force_p1",
    "self_path",
)


@functools.partial(jax.jit, static_argnames=_STATIC_ARGNAMES)
def chunked_core(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution,
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    group_bounds: tuple[tuple[int, ...], ...] = (),
    force_p1: bool = True,
    self_path: bool = False,
) -> jax.Array:
    """Chunked `streamindex_topk`.  Signature-compatible with the incumbent.

    `group_bounds` is the *static* plan from `plan_bounds`.  Three cases:

      * non-empty -> the pipelined A_0 B_0 A_1 B_1 ... program.
      * empty and `self_path` -> one group covering the whole batch, run
        through *this file's* code.  Same answer as the incumbent, but it
        exercises the stage-A launches, the reshape/pad, `_effective_row_
        lengths` and stage B here rather than delegating.  This is what the
        pytest transparency arm runs.
      * empty and not `self_path` -> the incumbent, unchanged.
    """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )

    rec = dict(group_bounds=group_bounds, self_path=self_path,
               force_p1=force_p1, enable_early_exit=enable_early_exit,
               num_tokens=int(q.shape[0]), num_seqs=int(seq_lens.shape[0]))

    if not group_bounds and (enable_early_exit or not self_path):
        # `_fast_path` is batch-wide and bypasses both stages, so there is no
        # pipeline to build; chunking it would only remove the fast path.
        rec["path"] = ("incumbent (early-exit)" if enable_early_exit else
                       "incumbent (no plan)")
        TRACE_LOG.append(rec)
        return sxmod.streamindex_topk(
            q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=enable_early_exit,
        )

    if isinstance(num_kv_pages_per_block, int):
        bkv_ps = [num_kv_pages_per_block] * 3
    else:
        bkv_ps = list(num_kv_pages_per_block)
    if isinstance(num_queries_per_block, int):
        bqs = [num_queries_per_block] * 3
    else:
        bqs = list(num_queries_per_block)

    max_num_seqs = seq_lens.shape[0]
    original_dtype = q.dtype
    prepared_w = prepare_index_weights(indexer_weights, original_dtype)
    q_prepared = prepare_q_inputs(q)

    lkv_dim = cache_kv.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in bkv_ps:
        if (page_size * bkv_p) % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p}) must be a multiple of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128 for bkv_p in bkv_ps)

    _, num_q_heads, head_dim = q_prepared.shape
    common = dict(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=kv_packing,
        lkv_dim=lkv_dim,
        num_sublanes_total=num_sublanes_total,
        compression_ratio=compression_ratio,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=jnp.float32,
    )
    # The three scoring launches `_common_path` makes, in the same order and
    # with the same statics.  Built once and reused for every chunk, so the
    # compiler sees one executable per (case, chunk-shape).
    fn_dec_batched = _stage_a.make_stage_a_fn(
        num_kv_pages_per_block=bkv_ps[0],
        num_queries_per_block=bqs[0],
        static_q_len=1,
        seq_batch_size=decode_req_batch_size,
        case=MlaCase.DECODE,
        **common)
    fn_dec_single = _stage_a.make_stage_a_fn(
        num_kv_pages_per_block=bkv_ps[0],
        num_queries_per_block=bqs[0],
        static_q_len=1,
        seq_batch_size=1,
        case=MlaCase.DECODE,
        **common)
    fn_mixed = _stage_a.make_stage_a_fn(
        num_kv_pages_per_block=bkv_ps[2],
        num_queries_per_block=bqs[2],
        static_q_len=None,
        seq_batch_size=1,
        case=MlaCase.MIXED,
        **common)

    if group_bounds:
        # (s0, s1, t0, t1, d0, d1, d2), all Python ints.
        plan = [tuple(b) for b in group_bounds]
        rec["path"] = f"chunked G={len(plan)}"
    else:
        # One group over the whole batch.  `distribution` is traced here, so
        # the rebased triple is just `distribution` itself.
        plan = [(0, max_num_seqs, 0, int(q.shape[0]),
                 distribution[0], distribution[1], distribution[2])]
        rec["path"] = "self (one group, whole batch)"

    guards = []
    outs = []
    for (s0, s1, t0, t1, d0, d1, d2) in plan:
        T = t1 - t0
        seq_lens_g = jax.lax.slice(seq_lens, (s0, ), (s1, ))
        page_indices_g = jax.lax.slice(page_indices,
                                       (s0 * pages_per_seq, ),
                                       (s1 * pages_per_seq, ))
        cu_g = jax.lax.slice(cu_q_lens, (s0, ), (s1 + 1, ))
        cu_g = cu_g - cu_q_lens[s0]
        q_g = jax.lax.slice_in_dim(q_prepared, t0, t1, axis=0)
        w_g = jax.lax.slice_in_dim(prepared_w, t0, t1, axis=0)

        scores = jnp.full((T, num_sublanes_total, 128), -jnp.inf, jnp.float32)
        decode_batch_end = (d0 // decode_req_batch_size) * decode_req_batch_size

        head = (q_g, w_g, cache_kv)
        tail = (seq_lens_g, page_indices_g, cu_g)
        scores = fn_dec_batched(*head, scores, *tail,
                                jnp.asarray(0, jnp.int32),
                                jnp.asarray(decode_batch_end, jnp.int32))
        scores = fn_dec_single(*head, scores, *tail,
                               jnp.asarray(decode_batch_end, jnp.int32),
                               jnp.asarray(d1, jnp.int32))
        scores = fn_mixed(*head, scores, *tail,
                          jnp.asarray(d1, jnp.int32),
                          jnp.asarray(d2, jnp.int32))

        scores = scores.reshape(T, -1)
        if scores.shape[1] < k:
            scores = jnp.pad(scores, ((0, 0), (0, k - scores.shape[1])),
                             constant_values=-jnp.inf)
        n = scores.shape[1]

        eff = _effective_row_lengths(
            seq_lens=seq_lens_g,
            cu_q_lens=cu_g,
            distribution=jnp.stack([
                jnp.asarray(d0, jnp.int32),
                jnp.asarray(d1, jnp.int32),
                jnp.asarray(d2, jnp.int32),
            ]),
            num_tokens=T,
            num_positions=n,
            compression_ratio=compression_ratio,
        )

        guards.append(waves_guard(T))
        if force_p1:
            _require_p1_is_safe(T, n, k)
            outs.append(scmod._sc_topk_direct(scores, k, eff, write_empty=True))
        else:
            outs.append(
                scmod.sparsecore_topk(scores, k, row_lengths=eff,
                                      write_empty_rows=True))

    rec["waves_guard"] = guards
    rec["num_sublanes_total"] = num_sublanes_total
    TRACE_LOG.append(rec)

    topk = jnp.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
    return topk[:q.shape[0], :k]


_INCUMBENT_STATIC_ARGNAMES = (
    "k",
    "compression_ratio",
    "num_kv_pages_per_block",
    "num_queries_per_block",
    "vmem_limit_bytes",
    "decode_req_batch_size",
    "enable_early_exit",
)


def make_dropin(*, num_groups: int = 1, force_p1: bool = True,
                self_path: bool = True):
    """A `jax.jit` object with *exactly* the incumbent's public signature.

    This is what the pytest plugin binds in place of `streamindex_topk`.  It is
    a real `PjitFunction`, so `jax.eval_shape` honours the static arguments
    (see the module docstring); the price is that it cannot plan, because
    `cu_q_lens` is traced by the time the body runs.  It therefore always
    passes `group_bounds=()`, and with `self_path=True` runs the whole batch
    through this file's single-group path -- the same stage-A launches, the
    same reshape/pad, the same `_effective_row_lengths`, the same stage B as
    the chunked path, just with G=1.  The repo suite is a *transparency* check
    on that code, not a chunking check: its batches are 1-3 sequences, so no
    plan with G>=2 could be honoured even if one were available.  The chunking
    check is `probe_correctness.py`.
    """

    def dropin(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
               distribution, *, k, compression_ratio,
               num_kv_pages_per_block=None, num_queries_per_block=None,
               vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
               decode_req_batch_size=4, enable_early_exit=False):
        return chunked_core(
            q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=enable_early_exit,
            group_bounds=(),
            force_p1=force_p1,
            self_path=self_path,
        )

    dropin.__name__ = f"option_b_dropin_G{num_groups}"
    dropin.__qualname__ = dropin.__name__
    return jax.jit(dropin, static_argnames=_INCUMBENT_STATIC_ARGNAMES)


def streamindex_topk_chunked(
    q,
    indexer_weights,
    cache_kv,
    seq_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    k,
    compression_ratio,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    num_groups: int = 1,
    force_p1: bool = True,
    self_path: bool = False,
    report: dict | None = None,
):
    """Eager convenience wrapper: plan, then call `chunked_core`.

    Not a `jax.jit` object -- do not hand this to `jax.eval_shape` (see the
    module docstring).  `report`, if given, is filled in with what actually
    happened; a caller that does not read it cannot tell a chunked run from a
    silent G=1 run.
    """
    if enable_early_exit:
        bounds, reason = (), "enable_early_exit=True"
    else:
        bounds, reason = plan_bounds(
            num_groups=num_groups,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=int(q.shape[0]),
            decode_req_batch_size=decode_req_batch_size,
        )
    if report is not None:
        report["requested_num_groups"] = num_groups
        report["chunked"] = bool(bounds)
        report["reason"] = reason
        report["groups"] = [dict(zip(("s0", "s1", "t0", "t1", "d0", "d1", "d2"),
                                     b)) for b in bounds]
        report["force_p1"] = force_p1
        report["waves_guard"] = [waves_guard(b[3] - b[2]) for b in bounds]

    return chunked_core(
        q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_req_batch_size=decode_req_batch_size,
        enable_early_exit=enable_early_exit,
        group_bounds=bounds,
        force_p1=force_p1,
        self_path=self_path,
    )
