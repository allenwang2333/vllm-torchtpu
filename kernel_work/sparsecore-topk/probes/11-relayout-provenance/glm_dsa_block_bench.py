"""M2: sweep the untuned block-tuning constants on GLM-5.2's DSA attention path.

Two kernels are timed, both per-DP-rank local (TP=1, DP=16), so a single-core
measurement is representative and the DP16 factor enters only as the token
count:

* `kernels/mla/sparse/kernel.py::sparse_ragged_paged_attention` -- the live
  sparse-MLA kernel. Runs 78x/step (every layer). Its
  `gather_and_attention_chunk_size=64`, `attention_kernel_batch_size=16` and
  `vmem_limit_bytes` are first-fit values, never swept.
* `kernels/deepseek_v4/streamindex_topk.py::streamindex_topk` -- runs 21x/step
  (`indexer_types` is 21 `full` / 57 `shared`). Its `num_kv_pages_per_block=8`
  and `num_queries_per_block=32` carry an explicit "NOT tuned" comment at the
  callsite in `mla_attention_op.py`.

The dead twin at `kernels/deepseek_v4/core_attention/sparse_mla.py:599` is
deliberately not touched.

Harness rules are copied method-for-method from `microbench/qmm_bench2.py`,
because they are what make a 2% delta on a shared pod mean anything:

* launch `reps` calls and `block_until_ready` **once** -- a per-call block
  costs ~138 us and would swamp every decode-sized shape here;
* two rep counts, flagging any row whose implied per-call host cost exceeds 5%
  (`hostbound_pct`) as dispatch-bound rather than kernel-bound;
* **3 interleaved rounds** of all-variants-then-all-variants, never
  all-reps-of-A-then-all-reps-of-B -- pod-level drift is ~2.5%, larger than
  several of the deltas being looked for, and grouping by variant aliases that
  drift straight onto the variant;
* every variant compiled up front, so compile time never lands inside a timed
  region;
* a null-jit dispatch floor printed first, as the bar every row is read
  against.

Shapes are derived from GLM-5.2's real `config.json` and from the backend
source, not guessed:

* `page_size` = 1024 from `PallasMLAttentionBackend.get_page_size`
  (`layers/vllm/attention.py:1102`); the indexer backend delegates to it
  (`:1145`), so both caches share it.
* nope cache `(pages, page_size, 4, 128)` uint8 and rope cache
  `(pages, page_size // 4, 4, 128)` uint8, from
  `sparse_mla_kernel.get_kv_cache_shape` folded through `dsa_gather`'s tile
  layout exactly as `tests/layers/common/test_sparse_mla_attention.py` does it.
* indexer cache `(pages, page_size // 4, 4, 256)` uint8 -- `index_head_dim` 128
  plus one e8m0 scale byte, padded to a 128 multiple, per the layout comment on
  `VllmTPUSparseAttnIndexer.forward_oot`.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import vllm_torchtpu
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.mla.sparse import dsa_gather
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

# ---------------------------------------------------------------- GLM-5.2 ---
# From the checkpoint's config.json, verified on disk.
NUM_HEADS = 64  # num_attention_heads; TP=1 so every rank holds all of them
LKV_DIM = 512  # kv_lora_rank
ROPE_DIM = 64  # qk_rope_head_dim
CSA_TOPK = 2048  # index_topk
INDEX_HEADS = 32  # index_n_heads
INDEX_HEAD_DIM = 128  # index_head_dim
NUM_LAYERS = 78  # num_hidden_layers -> sparse-MLA multiplier
NUM_INDEXER_LAYERS = 21  # indexer_types: 21 'full', 57 'shared'

# ------------------------------------------------------------- job shape ---
PAGE_SIZE = 1024  # PallasMLAttentionBackend.get_page_size
MAX_MODEL_LEN = 9216  # --max-model-len
MAX_NUM_SEQS = 16  # --max-num-seqs, per DP engine
PAGES_PER_SEQ = math.ceil(MAX_MODEL_LEN / PAGE_SIZE)  # 9
TOTAL_PAGES = 512  # >= MAX_NUM_SEQS * PAGES_PER_SEQ, with realistic spread
SEQ_LEN = 8192  # ISL; decode sits here, prefill's last chunk ends here
PREFILL_CHUNK = 1024  # --max-num-batched-tokens

SM_SCALE = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
K_SCALE = 1.0

KV_PACKING = 4  # get_dtype_packing(uint8) = 32 // 8
INDEXER_LANE = 256  # align_to(index_head_dim + 1, 128)

# ---------------------------------------------------------------- harness ---
REPS_LO = 20
REPS_HI = 60
ROUNDS = 3
HOSTBOUND_PCT_LIMIT = 5.0

SEED = 20260829

# Sourced by the pod's RUN_CMD to run the functional suite at the swept winner.
WINNER_ENV_PATH = "/tmp/dsa_winner.env"


def _p(*args):
    print(*args, flush=True)


# ------------------------------------------------------------- input prep ---


def _rng():
    return np.random.default_rng(SEED)


def _topk_rows(num_tokens: int, real_tokens: int,
               positions: np.ndarray) -> np.ndarray:
    """i32[num_tokens, CSA_TOPK] of in-sequence positions, -1 = padding.

    `positions[i]` is the absolute position of real token `i`. A token at
    position `p` attends to `min(CSA_TOPK, p + 1)` earlier positions; at
    `p >= CSA_TOPK` -- which is every token in this job's decode phase and in
    its last prefill chunk -- the row is completely full, so this is both the
    realistic case and the worst one.

    Indices are drawn without replacement and sorted. Sorted is the assumption
    a streaming top-k makes plausible and it is the friendlier case for gather
    DMA locality; `--unsorted` in the sweep below carries one control row so
    the assumption is measured rather than trusted.
    """
    rng = _rng()
    rows = np.full((num_tokens, CSA_TOPK), -1, np.int32)
    for i in range(real_tokens):
        pos = int(positions[i])
        n = min(CSA_TOPK, pos + 1)
        if n >= pos + 1:
            sel = np.arange(pos + 1, dtype=np.int32)
        else:
            sel = np.sort(
                rng.choice(pos + 1, size=n, replace=False).astype(np.int32))
        rows[i, :n] = sel
    return rows


def _shuffle_rows(rows: np.ndarray) -> np.ndarray:
    """Same rows, valid entries permuted -- the gather-locality control."""
    rng = np.random.default_rng(SEED + 1)
    out = rows.copy()
    for i in range(out.shape[0]):
        valid = out[i] >= 0
        n = int(valid.sum())
        if n > 1:
            out[i, :n] = rng.permutation(out[i, :n])
    return out


def _caches():
    """Zeroed nope/rope caches in dsa_gather's native tiled uint8 layouts."""
    nope_shape, rope_shape = (sparse_mla_kernel.get_kv_cache_shape(
        TOTAL_PAGES, PAGE_SIZE, dim, jnp.float8_e4m3fn)
                              for dim in (LKV_DIM, ROPE_DIM))
    blocks, ps_div, kv_packing, nope_dim = nope_shape
    lane = dsa_gather.TILE_LANE_BYTES
    nope = jnp.zeros((blocks, ps_div * kv_packing, nope_dim // lane, lane),
                     jnp.uint8)
    rope = jnp.zeros(rope_shape, jnp.uint8)
    return nope, rope


def _page_indices():
    """Distinct physical pages per sequence slot, permuted."""
    rng = _rng()
    need = MAX_NUM_SEQS * PAGES_PER_SEQ
    assert TOTAL_PAGES >= need, (TOTAL_PAGES, need)
    return jnp.asarray(rng.permutation(TOTAL_PAGES)[:need].astype(np.int32))


class Case:
    """One phase of the step, with every kernel input built for it."""

    def __init__(self, name: str, num_tokens: int, q_lens: list[int],
                 seq_lens: list[int], distribution: tuple[int, int, int],
                 positions: np.ndarray, shuffle: bool):
        self.name = name
        self.num_tokens = num_tokens
        self.real_tokens = int(sum(q_lens))
        rng = _rng()

        cu = np.zeros(MAX_NUM_SEQS + 1, np.int32)
        cu[1:len(q_lens) + 1] = np.cumsum(q_lens)
        cu[len(q_lens) + 1:] = cu[len(q_lens)]
        self.cu_q_lens = jnp.asarray(cu)

        sl = np.zeros(MAX_NUM_SEQS, np.int32)
        sl[:len(seq_lens)] = seq_lens
        self.seq_lens = jnp.asarray(sl)

        self.distribution = jnp.asarray(np.asarray(distribution, np.int32))
        self.page_indices = _page_indices()

        rows = _topk_rows(num_tokens, self.real_tokens, positions)
        if shuffle:
            rows = _shuffle_rows(rows)
        self.topk_indices = jnp.asarray(rows)

        # sparse-MLA q: [T, heads, LKV_DIM + ROPE_DIM] bf16
        self.q = jnp.asarray(rng.standard_normal(
            (num_tokens, NUM_HEADS, LKV_DIM + ROPE_DIM)).astype(np.float32),
                             dtype=jnp.bfloat16)
        self.nope_cache, self.rope_cache = _caches()

        # streamindex q: fp8 [T, INDEX_HEADS, INDEX_HEAD_DIM], plus weights
        self.idx_q = jnp.asarray(
            rng.standard_normal(
                (num_tokens, INDEX_HEADS, INDEX_HEAD_DIM)).astype(np.float32),
            dtype=jnp.float8_e4m3fn)
        self.idx_weights = jnp.asarray(rng.standard_normal(
            (num_tokens, INDEX_HEADS)).astype(np.float32),
                                       dtype=jnp.bfloat16)
        self.idx_cache = jnp.zeros(
            (TOTAL_PAGES, PAGE_SIZE // KV_PACKING, KV_PACKING, INDEXER_LANE),
            jnp.uint8)


def build_cases() -> list[Case]:
    cases = []
    # Decode: MAX_NUM_SEQS sequences, one token each, mid-generation at
    # SEQ_LEN. The token bucket pads beyond the 16 real tokens.
    for bucket in (16, 32):
        cases.append(
            Case(f"decode_T{bucket}",
                 num_tokens=bucket,
                 q_lens=[1] * MAX_NUM_SEQS,
                 seq_lens=[SEQ_LEN] * MAX_NUM_SEQS,
                 distribution=(MAX_NUM_SEQS, MAX_NUM_SEQS, MAX_NUM_SEQS),
                 positions=np.full(MAX_NUM_SEQS, SEQ_LEN - 1),
                 shuffle=False))
    # Prefill: one chunked-prefill sequence, the last chunk of an ISL-8192
    # prompt, so every token is past CSA_TOPK and the topk rows are full.
    prefill_positions = np.arange(SEQ_LEN - PREFILL_CHUNK, SEQ_LEN)
    cases.append(
        Case("prefill_T1024",
             num_tokens=PREFILL_CHUNK,
             q_lens=[PREFILL_CHUNK],
             seq_lens=[SEQ_LEN],
             distribution=(0, 1, 1),
             positions=prefill_positions,
             shuffle=False))
    # Gather-locality control: identical to decode_T16 but with the valid
    # topk entries permuted, so "sorted indices" stops being an assumption.
    cases.append(
        Case("decode_T16_unsorted",
             num_tokens=16,
             q_lens=[1] * MAX_NUM_SEQS,
             seq_lens=[SEQ_LEN] * MAX_NUM_SEQS,
             distribution=(MAX_NUM_SEQS, MAX_NUM_SEQS, MAX_NUM_SEQS),
             positions=np.full(MAX_NUM_SEQS, SEQ_LEN - 1),
             shuffle=True))
    return cases


# ------------------------------------------------------------- timing core --


def _time(fn, reps: int) -> float:
    """Total wall seconds for `reps` launches with a single block at the end."""
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def dispatch_floor() -> float:
    """Cost of issuing a trivial jitted call -- the bar for every row below."""
    x = jnp.zeros((8, 128), jnp.float32)
    f = jax.jit(lambda a: a + 1.0)
    jax.block_until_ready(f(x))
    _time(lambda: f(x), REPS_HI)  # warm
    lo = _time(lambda: f(x), REPS_LO)
    hi = _time(lambda: f(x), REPS_HI)
    per = (hi - lo) / (REPS_HI - REPS_LO)
    return per * 1e6


#: Visiting order actually used by the most recent `measure` call, one list
#: per round. Read it after the call and record it in the JSON; it is the only
#: evidence that the shuffle happened and the only way to check, after the
#: fact, whether a suspicious row was always visited early.
LAST_ORDER: list[list[str]] = []


def measure(variants: dict[str, object], rounds: int = ROUNDS,
            shuffle_seed: int = 20260902) -> dict:
    """Interleaved timing of every variant. `variants[name]` is a 0-arg thunk.

    Returns `{name: {"us": ..., "hostbound_pct": ...}}`. Rounds alternate
    across variants so pod drift lands on all of them equally instead of on
    whichever happened to run last.

    Two properties this function is careful about, both of which were wrong
    before and both of which showed up as a 4.2% cross-job disagreement on
    `decode_r46/kernel` between `j-686ef8af` and `j-bdefb210`:

    **The visiting order is reshuffled every round.** It used to be
    `variants.items()`, i.e. identical in every round and in both the lo and
    hi passes, which confounds position-in-pass with arm: whatever is
    registered first is always measured first. In `j-686ef8af` the three rows
    running the *unmodified* computation (the baseline, the algebraically-inert
    `nmi`, and `#null`) read 74.40-76.51 us while every real arm reproduced to
    0.4%, so position was a live suspect and could not be separated. The seed
    is fixed and recorded so a pass is still reproducible.

    **The slope is PAIRED.** It used to be `(min_r hi_r - min_r lo_r)`, whose
    two minima can come from different rounds; that is both higher-variance
    and biased low, since `min(hi) - min(lo) <= min(hi - lo)`. At decode
    `lo = 20 x 73 us = 1.46 ms`, so perturbing `lo` alone by 1% moves the
    reported figure 0.5% -- the right order to have produced the outlier.
    `us_unpaired` is emitted alongside so the two estimators can be compared
    on any pass rather than taken on faith.
    """
    rng = random.Random(shuffle_seed)
    lo_best: dict[str, float] = {}
    hi_best: dict[str, float] = {}
    slopes: dict[str, list[float]] = {}
    order_log: list[list[str]] = []
    # Warm EVERY variant before round 1. A thunk that compiles on its first
    # call otherwise puts the whole compile inside round 1's `lo`, making that
    # round's paired slope `hi - lo_1` equal to minus the compile time -- and
    # `min` over rounds then selects precisely that round.
    #
    # This is not hypothetical: job `j-2a092b0d` read every `/kernel#null` row
    # at -89000 to -149000 us/layer, uniformly across all five passes. Those
    # are compile times, not measurements (-96820 us x 40 reps = -3.87 s at
    # decode, -5.98 s at prefill). The arms escaped it only by accident -- they
    # are warmed incidentally by the feasibility/SKIP probe at their
    # registration sites -- while the `--drift-null` rows have no such probe
    # and so were the only thunks entering this loop cold.
    #
    # The old unpaired estimator hid the bug rather than avoiding it: taking
    # `min` of `lo` and of `hi` independently across rounds silently dropped
    # the contaminated round. So this warmup, not the choice of estimator, is
    # what actually makes a recompiling variant measurable.
    for name in variants:
        jax.block_until_ready(variants[name]())
    for _ in range(rounds):
        names = list(variants)
        rng.shuffle(names)
        order_log.append(names)
        lo_r: dict[str, float] = {}
        for name in names:
            lo_r[name] = _time(variants[name], REPS_LO)
            lo_best[name] = min(lo_best.get(name, math.inf), lo_r[name])
        for name in names:
            hi = _time(variants[name], REPS_HI)
            hi_best[name] = min(hi_best.get(name, math.inf), hi)
            # Paired within the round: this `hi` against the `lo` measured a
            # few hundred milliseconds ago, not against the best `lo` of any
            # round. Drift between rounds cancels instead of accumulating.
            slopes.setdefault(name, []).append(hi - lo_r[name])
    out = {}
    for name in variants:
        lo, hi = lo_best[name], hi_best[name]
        # Slope removes the fixed per-launch host cost; the intercept is that
        # host cost, and its share of the naive per-call time is how much of
        # the row is dispatch rather than kernel.
        #
        # THREE aggregators, all emitted, because which one is right is an
        # empirical question this harness has never asked. `min` is the
        # traditional timing idiom and is correct when the contamination is
        # one-sided (another process stealing the core); it is the wrong choice
        # under symmetric jitter, where it chases the lower tail instead of
        # averaging -- modelled at sigma=1%/call, going 3 -> 20 rounds moves
        # `min`'s null spread only 1.73 -> 1.28 sd, while `mean` at 8 rounds
        # already reaches 0.785. `us` stays `min` so every historical row
        # remains comparable; pick the winner from the three-null spread that
        # `--drift-null` now measures on real hardware, not from that model.
        s = sorted(slopes[name])
        n = len(s)
        med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        span = REPS_HI - REPS_LO
        per_device = s[0] / span
        naive = hi / REPS_HI
        host = max(naive - per_device, 0.0)
        out[name] = {
            "us": per_device * 1e6,
            "us_median": med / span * 1e6,
            "us_mean": (sum(s) / n) / span * 1e6,
            "us_unpaired": (hi - lo) / span * 1e6,
            "hostbound_pct": (host / naive * 100.0) if naive > 0 else 0.0,
        }
        # A slope can only go negative if `hi` came in below `lo`, which no
        # amount of jitter explains at these rep counts -- it means that round's
        # `lo` contained something `hi` did not, and in practice that is a
        # compile. `us` is `min(slopes)`, so it SELECTS such a round: the row
        # then reports minus the compile time and is the fastest thing on the
        # board. Job `j-2a092b0d` printed -89000 to -149000 us/layer this way on
        # every `#null` row, in all five passes, and the table looked ordinary
        # apart from those rows.
        #
        # The warmup above is meant to make this unreachable. This flag is here
        # because the warmup is easy to bypass by accident -- any future variant
        # registered outside the `variants` dict, or a thunk that recompiles on
        # a later call (a changed static argument, a new shape), lands right
        # back here. A smaller compile would shift a row a few percent instead
        # of making it negative, and would read as a win.
        out[name]["suspect"] = bool(
            min(s) <= 0.0 or out[name]["us"] <= 0.0)
        if out[name]["suspect"]:
            print(f"  !! SUSPECT {name}: min slope {min(s) / span * 1e6:.1f} "
                  f"us/call <= 0 -- almost certainly an unwarmed recompile. "
                  f"`us` is invalid for this row; read `us_median` "
                  f"({out[name]['us_median']:.2f}) and check it against the "
                  f"dispatch floor before using it.", file=sys.stderr)
    # Side channel, NOT a key in `out`: eight call sites across the microbench
    # scripts iterate `measure(...).items()` and treat every key as a variant,
    # so a metadata entry here would be printed as a row and ranked as an arm.
    global LAST_ORDER
    LAST_ORDER = order_log
    return out


# --------------------------------------------------------------- sweeps -----

# At T=16 and T=32 `num_chunks = cdiv(T, chunk)` (kernel.py:634) collapses to 1
# for any chunk >= 32, so the decode space is genuinely small; the real room is
# at T=1024.
MLA_CHUNKS = {
    "decode_T16": (16, 32, 64),
    "decode_T32": (16, 32, 64),
    "decode_T16_unsorted": (16, 32, 64),
    "prefill_T1024": (32, 64, 128, 256),
}
MLA_BATCHES = (4, 8, 16, 32)
MLA_DEFAULT = (64, 16)

STREAMIDX_KV_PAGES = (2, 4, 8)
STREAMIDX_QUERIES = (16, 32, 64)
STREAMIDX_DEFAULT = (8, 32)

MIB = 1024 * 1024
VMEM_VALUES = (None, 64 * MIB, 96 * MIB, 128 * MIB)


def mla_thunk(case: Case, chunk: int, batch: int, vmem: int | None):
    kwargs = dict(sm_scale=SM_SCALE,
                  k_scale=K_SCALE,
                  gather_and_attention_chunk_size=chunk,
                  attention_kernel_batch_size=batch)
    if vmem is not None:
        kwargs["vmem_limit_bytes"] = vmem

    def run():
        return sparse_mla_kernel.sparse_ragged_paged_attention(
            case.q, case.nope_cache, case.rope_cache, case.topk_indices,
            case.page_indices, case.cu_q_lens, case.distribution, **kwargs)

    return run


def streamidx_thunk(case: Case, kv_pages: int, queries: int,
                    vmem: int | None):
    kwargs = dict(k=CSA_TOPK,
                  compression_ratio=1,
                  num_kv_pages_per_block=kv_pages,
                  num_queries_per_block=queries)
    if vmem is not None:
        kwargs["vmem_limit_bytes"] = vmem

    def run():
        return streamindex_topk(case.idx_q, case.idx_weights, case.idx_cache,
                                case.seq_lens, case.page_indices,
                                case.cu_q_lens, case.distribution, **kwargs)

    return run


def compile_variants(builder, case: Case, combos) -> tuple[dict, list[dict]]:
    """Build and warm every variant, recording compile failures as SKIP rows.

    Compiling up front is not a nicety: a Pallas compile inside a timed region
    would be attributed to the kernel, and a VMEM-exceeded configuration must
    be reported as infeasible rather than silently dropped.
    """
    ok, skipped = {}, []
    for label, args in combos:
        thunk = builder(case, *args)
        try:
            jax.block_until_ready(thunk())
        except Exception as exc:  # noqa: BLE001 - any compile failure is a SKIP
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            _p(f"  SKIP {case.name} {label}: {msg}")
            skipped.append({"case": case.name, "variant": label, "skip": msg})
            continue
        ok[label] = thunk
    return ok, skipped


def sweep(kernel: str, builder, cases: list[Case], combos_for, default_label,
          multiplier: int) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        combos = combos_for(case)
        _p(f"\n--- {kernel} / {case.name}: {len(combos)} variants ---")
        t0 = time.perf_counter()
        variants, skipped = compile_variants(builder, case, combos)
        rows.extend({
            "kernel": kernel,
            "case": case.name,
            **s
        } for s in skipped)
        _p(f"  compiled {len(variants)}/{len(combos)} in "
           f"{time.perf_counter() - t0:.1f}s")
        if not variants:
            continue
        stats = measure(variants)
        base = stats.get(default_label, {}).get("us")
        for label, s in sorted(stats.items(), key=lambda kv: kv[1]["us"]):
            saving = (base - s["us"]) if base is not None else None
            row = {
                "kernel": kernel,
                "case": case.name,
                "variant": label,
                "us": round(s["us"], 2),
                "hostbound_pct": round(s["hostbound_pct"], 1),
                "layers": multiplier,
                "ms_per_step": round(s["us"] * multiplier / 1000.0, 3),
                "vs_default_ms_per_step":
                (round(saving * multiplier / 1000.0, 3)
                 if saving is not None else None),
                "is_default": label == default_label,
            }
            rows.append(row)
            mark = "  <-- default" if row["is_default"] else ""
            warn = "  >5% hostbound" if s["hostbound_pct"] > \
                HOSTBOUND_PCT_LIMIT else ""
            _p(f"  {label:<28} {s['us']:9.2f} us  "
               f"{row['ms_per_step']:8.3f} ms/step  "
               f"hb={s['hostbound_pct']:5.1f}%{mark}{warn}")
    return rows


def main():
    _p(f"vllm_torchtpu from {vllm_torchtpu.__file__}")
    _p(f"jax {jax.__version__}  devices={jax.devices()}")
    _p(f"PAGE_SIZE={PAGE_SIZE} PAGES_PER_SEQ={PAGES_PER_SEQ} "
       f"TOTAL_PAGES={TOTAL_PAGES} MAX_NUM_SEQS={MAX_NUM_SEQS} "
       f"SEQ_LEN={SEQ_LEN} CSA_TOPK={CSA_TOPK}")
    _p(f"sparse-MLA defaults: chunk={MLA_DEFAULT[0]} "
       f"batch={MLA_DEFAULT[1]} vmem={sparse_mla_kernel.DEFAULT_VMEM_LIMIT_BYTES}")
    _p(f"streamidx defaults:  kv_pages={STREAMIDX_DEFAULT[0]} "
       f"queries={STREAMIDX_DEFAULT[1]}")

    floor = dispatch_floor()
    _p(f"\nnull/dispatch issue rate: {floor:.2f} us/call "
       f"(floor for every row below)")

    cases = build_cases()
    for c in cases:
        _p(f"case {c.name}: tokens={c.num_tokens} real={c.real_tokens} "
           f"q={c.q.shape} nope={c.nope_cache.shape} rope={c.rope_cache.shape} "
           f"idx_cache={c.idx_cache.shape} topk={c.topk_indices.shape}")

    rows: list[dict] = []

    # Stage 1: (chunk, batch) at the kernel's own default vmem.
    #
    # Every variant costs a Pallas compile, so the grid is only spent where a
    # question is actually open. The unsorted control asks one question --
    # "does gather locality matter at all?" -- and answers it with the default
    # config alone; a full grid there would buy nothing.
    def mla_combos(case: Case):
        if case.name == "decode_T16_unsorted":
            return [(f"chunk{MLA_DEFAULT[0]}_batch{MLA_DEFAULT[1]}",
                     (*MLA_DEFAULT, None))]
        return [(f"chunk{c}_batch{b}", (c, b, None))
                for c in MLA_CHUNKS[case.name] for b in MLA_BATCHES]

    rows += sweep("sparse_mla",
                  mla_thunk,
                  cases,
                  mla_combos,
                  default_label=f"chunk{MLA_DEFAULT[0]}_batch{MLA_DEFAULT[1]}",
                  multiplier=NUM_LAYERS)

    # streamidx runs 21x/step against sparse-MLA's 78x, so it gets the two
    # phases that matter and not the two redundant decode variants: T=32 is
    # the same 16 real tokens with padding, and locality is a sparse-MLA
    # question, not an indexer one.
    def streamidx_combos(case: Case):
        if case.name not in ("decode_T16", "prefill_T1024"):
            return []
        return [(f"kvp{p}_q{q}", (p, q, None)) for p in STREAMIDX_KV_PAGES
                for q in STREAMIDX_QUERIES]

    rows += sweep(
        "streamidx",
        streamidx_thunk,
        cases,
        streamidx_combos,
        default_label=f"kvp{STREAMIDX_DEFAULT[0]}_q{STREAMIDX_DEFAULT[1]}",
        multiplier=NUM_INDEXER_LAYERS)

    # Stage 2: vmem, only at each case's stage-1 winner. Sweeping vmem across
    # the whole (chunk, batch) grid would quadruple the compile count for a
    # knob that is a ceiling, not a shape.
    best: dict[str, tuple[int, int]] = {}
    for case in cases:
        cand = [
            r for r in rows
            if r["kernel"] == "sparse_mla" and r["case"] == case.name
            and "us" in r
        ]
        if cand:
            win = min(cand, key=lambda r: r["us"])["variant"]
            c, b = win.replace("chunk", "").split("_batch")
            best[case.name] = (int(c), int(b))
    _p(f"\nstage-1 sparse_mla winners: {best}")

    # Emit the decode winner as a sourceable env file so the same pod can run
    # the functional suite through the new `TPU_DSA_*` overrides at exactly the
    # constants this sweep recommends. That is what makes candidate #2's
    # eventual E2E arm correctness-verified without a separate accuracy job --
    # a perf number for an unvalidated kernel config is worth nothing.
    if "decode_T16" in best:
        c, b = best["decode_T16"]
        with open(WINNER_ENV_PATH, "w") as f:
            f.write(f"export TPU_DSA_GATHER_CHUNK={c}\n")
            f.write(f"export TPU_DSA_ATTN_BATCH={b}\n")
        _p(f"wrote {WINNER_ENV_PATH}: TPU_DSA_GATHER_CHUNK={c} "
           f"TPU_DSA_ATTN_BATCH={b}")

    def vmem_combos(case: Case):
        if case.name not in best or case.name == "decode_T16_unsorted":
            return []
        c, b = best[case.name]
        return [(f"chunk{c}_batch{b}_vmem"
                 f"{'default' if v is None else v // MIB}", (c, b, v))
                for v in VMEM_VALUES]

    rows += sweep("sparse_mla_vmem",
                  mla_thunk,
                  cases,
                  vmem_combos,
                  default_label=None,
                  multiplier=NUM_LAYERS)

    _p("\nJSON_BEGIN")
    _p(
        json.dumps(
            {
                "dispatch_floor_us": round(floor, 2),
                "page_size": PAGE_SIZE,
                "pages_per_seq": PAGES_PER_SEQ,
                "seq_len": SEQ_LEN,
                "max_num_seqs": MAX_NUM_SEQS,
                "reps": [REPS_LO, REPS_HI],
                "rounds": ROUNDS,
                "rows": rows,
            },
            indent=None))
    _p("JSON_END")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
