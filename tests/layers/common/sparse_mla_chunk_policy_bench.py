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
"""Preflight bench for `TPU_DSA_CHUNK_BY_BUCKET` at the LAYER level.

Not collected by pytest (no `test_` prefix): run it as a script.

What this measures that nothing else does
-----------------------------------------
`gather_and_attention_chunk_size` has already been swept on one node at the
*kernel* entry point (`sparse_ragged_paged_attention`) at these same
production shapes. That sweep excludes everything the layer wrapper
`attention_interface.sparse_mla_attention` adds -- `shard_map`, the fused
`update_sparse_mla_kv_cache` insert, the `q = concat(ql_nope, q_pe)` -- all of
which are chunk-INDEPENDENT and therefore dilute the kernel-level delta by an
amount nobody has measured. This bench drives the real wrapper, through the
real `DSA_CHUNK_BY_BUCKET` module switch, so the number it prints is the
fraction of the kernel win that survives to the layer.

It also does three things the in-tree functional test cannot, because that
test runs at a toy geometry (16 pages of 32 tokens, 16 heads, topk 64):

  * bit-exactness of ON vs OFF at the PRODUCTION geometry (415 pages of 1024,
    64 heads, topk 2048) and at the two token buckets production actually
    lands on;
  * proof, read out of the lowered HLO rather than from a `mock.patch`, that
    the bucket policy put the intended chunk size into the kernel call and
    that the kernel's own attention call count changed accordingly;
  * a feasibility check that the doubled gather staging buffer at chunk 128
    (chunk x topk x 512 B = 128 MiB, against 64 MiB shipping) compiles and
    runs at the production topk.

Shape provenance
----------------
Every constant below is derived from a production artifact, not inherited
from another harness:

  PAGE_SIZE, MAX_MODEL_LEN, MAX_NUM_SEQS, PREFILL_CHUNK
      scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh
      (--max-model-len 9216, --max-num-batched-tokens 1024, --max-num-seqs 64)
  TOTAL_PAGES, RUNNING_REQS
      the engine log of the profiled 2-node run j-5c9c011c
      (424,960 KV tokens / 1024; steady-state "Running: 46 reqs")
  DECODE_T
      _get_padded_token_len(_get_token_paddings(16, 1024), 46) == 64.
      The buckets are [16, 32, 64, 128, 256, 512, 1024]; this is the value
      the whole change hinges on, since `n <= 64` is the small-chunk branch.
  NUM_HEADS, LKV_DIM, ROPE_DIM, CSA_TOPK
      GLM-5.2 config: num_attention_heads 64 (TP=1, so every rank holds all
      of them), kv_lora_rank 512, qk_rope_head_dim 64, index_topk 2048.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.mla.sparse import dsa_gather
from vllm_torchtpu.layers.common import attention_interface

# ------------------------------------------------------------------ shapes --
PAGE_SIZE = 1024
MAX_MODEL_LEN = 9216
PAGES_PER_SEQ = math.ceil(MAX_MODEL_LEN / PAGE_SIZE)  # 9
MAX_NUM_SEQS = 64
PAGE_SLOTS = MAX_NUM_SEQS * PAGES_PER_SEQ  # 576
TOTAL_PAGES = 415
RUNNING_REQS = 46
DECODE_T = 64
PREFILL_CHUNK = 1024
PREFILL_CTX = 1024  # one of the six context offsets the profiled run swept
ISL = 8192

NUM_HEADS = 64
LKV_DIM = 512
ROPE_DIM = 64
CSA_TOPK = 2048

SM_SCALE = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
K_SCALE = 1.0

#: How many wrapper calls are chained inside one jit. Per-call dispatch on
#: this pod costs O(100 us) and a decode wrapper call is O(200 us), so a
#: single call is not measurable; chaining puts the whole run inside one
#: executable. Chosen per case so a measurement is >= ~30x the dispatch floor
#: without unrolling so many pallas_calls that compilation dominates.
CHAIN = {"decode": 4, "prefill": 2}


def _fp8_bytes(rng, shape):
    """Random uint8 that decode to small finite float8_e4m3fn values.

    Bytes 0x00-0x3F have sign 0 and exponent field <= 7, i.e. magnitudes in
    [0, 1.875]. That deliberately excludes 0x7F/0xFF, the two e4m3fn NaN
    encodings -- a NaN anywhere in the cache would make the ON-vs-OFF
    elementwise comparison below vacuous, since NaN never equals NaN and
    `assert_array_equal` would report a difference the chunk size did not
    cause.
    """
    return rng.integers(0, 0x40, size=shape, dtype=np.uint8)


def build_case(case: str):
    """Production-shaped inputs for `sparse_mla_attention`."""
    rng = np.random.default_rng(20260904)

    if case == "decode":
        num_tokens = DECODE_T
        n_seqs = RUNNING_REQS
        q_lens = [1] * n_seqs
        seq_lens_real = [ISL] * n_seqs
        distribution = (n_seqs, n_seqs, n_seqs)
        # Each decode token sits at position ISL-1 and may attend anywhere in
        # [0, ISL); all CSA_TOPK slots are genuinely valid at this context.
        visible = [ISL] * n_seqs
    elif case == "prefill":
        num_tokens = PREFILL_CHUNK
        n_seqs = 1
        q_lens = [PREFILL_CHUNK]
        seq_lens_real = [PREFILL_CTX + PREFILL_CHUNK]
        distribution = (0, 0, 1)
        visible = None  # per-token, computed below
    else:
        raise ValueError(case)

    # --- seq_lens / query_start_loc / block_tables, padded the way the
    # --- runner pads them: max_num_seqs entries regardless of how many are
    # --- live, so `cu_q_lens.shape[0] - 1 == max_num_seqs`.
    seq_lens = np.zeros(MAX_NUM_SEQS, np.int32)
    seq_lens[:n_seqs] = seq_lens_real
    qsl = np.zeros(MAX_NUM_SEQS + 1, np.int32)
    qsl[1:n_seqs + 1] = np.cumsum(q_lens)
    qsl[n_seqs + 1:] = qsl[n_seqs]

    # Distinct physical pages per sequence: 46 x 9 = 414 <= TOTAL_PAGES.
    perm = rng.permutation(TOTAL_PAGES)
    block_tables = np.zeros(PAGE_SLOTS, np.int32)
    for s in range(MAX_NUM_SEQS):
        lo = (s * PAGES_PER_SEQ) % (TOTAL_PAGES - PAGES_PER_SEQ)
        block_tables[s * PAGES_PER_SEQ:(s + 1) * PAGES_PER_SEQ] = \
            perm[lo:lo + PAGES_PER_SEQ]

    # --- topk_indices: valid entries inside each token's own window, -1 tail.
    topk = np.full((num_tokens, CSA_TOPK), -1, np.int32)
    for t in range(num_tokens):
        if case == "decode":
            if t < n_seqs:
                win = visible[t]
            else:
                # Padded token. Kernel contract: >= 1 valid entry.
                topk[t, 0] = 0
                continue
        else:
            if t < PREFILL_CHUNK:
                win = PREFILL_CTX + t + 1
            else:
                topk[t, 0] = 0
                continue
        k = min(win, CSA_TOPK)
        topk[t, :k] = rng.choice(win, size=k, replace=False)

    # --- caches, queries, this step's latents.
    nope_cache = jnp.asarray(
        _fp8_bytes(rng,
                   (TOTAL_PAGES, PAGE_SIZE, dsa_gather.TILE_SUBROWS,
                    dsa_gather.TILE_LANE_BYTES)))
    rope_cache = jnp.asarray(
        _fp8_bytes(rng, (TOTAL_PAGES, PAGE_SIZE // dsa_gather.TILE_SUBROWS,
                         dsa_gather.TILE_SUBROWS, dsa_gather.TILE_LANE_BYTES)))

    ql_nope = jnp.asarray(
        rng.standard_normal((num_tokens, NUM_HEADS, LKV_DIM), np.float32),
        dtype=jnp.bfloat16)
    q_pe = jnp.asarray(
        rng.standard_normal((num_tokens, NUM_HEADS, ROPE_DIM), np.float32),
        dtype=jnp.bfloat16)
    kv_c = jax.lax.bitcast_convert_type(
        jnp.asarray(_fp8_bytes(rng, (num_tokens, LKV_DIM))),
        jnp.float8_e4m3fn)
    k_pe = jax.lax.bitcast_convert_type(
        jnp.asarray(_fp8_bytes(rng, (num_tokens, ROPE_DIM))),
        jnp.float8_e4m3fn)

    return dict(
        case=case,
        num_tokens=num_tokens,
        args=(ql_nope, q_pe, kv_c, k_pe, nope_cache, rope_cache,
              jnp.asarray(topk), jnp.asarray(seq_lens),
              jnp.asarray(block_tables), jnp.asarray(qsl),
              jnp.asarray(np.asarray(distribution, np.int32))),
    )


def make_chain(mesh, depth: int):
    """`depth` wrapper calls, output fed back in as the next `ql_nope`.

    The feedback is what stops XLA from CSE-ing the repeats into one call,
    and it mirrors a layer stack: `output` and `ql_nope` have the same shape
    `[T, NUM_HEADS, LKV_DIM]`.
    """

    def chain(ql_nope, q_pe, kv_c, k_pe, nope_cache, rope_cache, topk,
              seq_lens, block_tables, qsl, dist):
        out = ql_nope
        for _ in range(depth):
            nope_cache, rope_cache, out = \
                attention_interface.sparse_mla_attention(
                    out, q_pe, kv_c, k_pe, nope_cache, rope_cache, topk,
                    seq_lens, block_tables, qsl, dist, mesh,
                    sm_scale=SM_SCALE, k_scale=K_SCALE)
            out = out.astype(ql_nope.dtype)
        return out

    return jax.jit(chain)


def _time(fn, reps):
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / reps * 1e6  # us per chained call


def dispatch_floor():
    x = jnp.zeros((8, 128), jnp.float32)
    f = jax.jit(lambda a: a + 1.0)
    jax.block_until_ready(f(x))
    lo, hi = 20, 200
    _time(lambda: f(x), hi)
    t_lo = _time(lambda: f(x), lo) * lo
    t_hi = _time(lambda: f(x), hi) * hi
    return (t_hi - t_lo) / (hi - lo)


def gcs_tags(lowered_text):
    """`{chunk: count}` from the `sparse-mla-gcs_<n>` named scopes in the HLO.

    This is the proof that the policy reached the kernel. It reads what was
    lowered, not what a Python-level patch claims: the tag is emitted by the
    call site in `sparse_mla_attention` from the value it actually passed as
    `gather_and_attention_chunk_size`.
    """
    import re
    counts = {}
    for m in re.finditer(r"sparse-mla-gcs_(\d+)", lowered_text):
        counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
    return counts


def mla_call_count(lowered_text, chunk):
    """Attention pallas_calls in the HLO carrying this chunk's own scope.

    The kernel builds its scope inside `run_mla_kernel`, whose `cache_kv_*`
    parameters hold the GATHERED buffers, not the caches -- so
    `MLA-p_<topk>-bz_<kernel_batch_size>-gcz_<chunk_size>`. `gcz_` is the
    chunk size, not the KV page count (the comment at the new call site in
    `attention_interface.py` has this backwards). That makes the kernel's own
    scope a sufficient, independent read-back of which chunk size the policy
    actually delivered, and the count is `depth * cdiv(T, chunk)`.
    """
    return lowered_text.count(f"MLA-p_{CSA_TOPK}-bz_16-gcz_{chunk}")


def run_case(case: str, rounds: int, reps: int, report: dict):
    data = build_case(case)
    depth = CHAIN[case]
    n = data["num_tokens"]
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

    # ---- arms. `#null1..3` are OFF as well: identical work under a different
    # ---- name, so their spread is this case's floor. One null is a number,
    # ---- three are a floor.
    arms = {
        "off": False,
        "on": True,
        "off#null1": False,
        "off#null2": False,
        "off#null3": False,
    }

    # ONE `jax.jit` OBJECT PER ARM, and this is not a style choice. The chunk
    # size is read from a module global at trace time; `jax.jit`'s cache key
    # is (shapes, dtypes, ...) and does NOT include that global. A single
    # shared `chain` would trace once under whichever flag ran first and then
    # serve the SAME executable to every later arm -- five arms, one
    # compilation, a perfect null result, and nothing in the output would say
    # so. A fresh wrapper per arm has a fresh cache.
    chains = {arm: make_chain(mesh, depth) for arm in arms}

    resolved = {}
    lowered = {}
    outputs = {}
    saved = attention_interface.DSA_CHUNK_BY_BUCKET
    try:
        for arm, flag in arms.items():
            attention_interface.DSA_CHUNK_BY_BUCKET = flag
            resolved[arm] = attention_interface.sparse_mla_gather_chunk_size(n)
            chain = chains[arm]
            if arm in ("off", "on"):
                try:
                    txt = chain.lower(*data["args"]).as_text(debug_info=True)
                except TypeError:
                    txt = chain.lower(*data["args"]).as_text()
                lowered[arm] = {
                    "gcs_tags": gcs_tags(txt),
                    "mla_calls": mla_call_count(txt, resolved[arm]),
                    "mla_calls_other": {
                        c: mla_call_count(txt, c)
                        for c in (16, 32, 64, 128) if c != resolved[arm]
                    },
                }
            outputs[arm] = jax.block_until_ready(chain(*data["args"]))

        # ---- correctness: ON must be bit-identical to OFF.
        def _bits(x):
            return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))

        a = np.asarray(outputs["off"].astype(jnp.float32))
        b = np.asarray(outputs["on"].astype(jnp.float32))
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        off_bits, on_bits = _bits(outputs["off"]), _bits(outputs["on"])
        bit_identical = bool(np.array_equal(off_bits, on_bits))
        max_abs_diff = float(np.max(np.abs(a - b))) if finite else float("nan")
        # Anti-vacuity: a run whose output is a constant (all-zero, say) would
        # be bit-identical across arms for a reason that has nothing to do
        # with the chunk size.
        distinct_values = int(np.unique(off_bits).size)

        # ---- negative control: the SAME comparison, on a copy of the OFF
        # ---- output with one element perturbed. If this does not reject,
        # ---- the green result above is unfalsifiable.
        poisoned = off_bits.copy()
        poisoned[0, 0, 0] ^= np.uint16(1)
        control_rejects = not np.array_equal(off_bits, poisoned)

        # ---- timing, arms interleaved and reshuffled every round.
        rng = random.Random(20260904)
        per_round = {k: [] for k in arms}
        for _ in range(rounds):
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                attention_interface.DSA_CHUNK_BY_BUCKET = arms[arm]
                c = chains[arm]
                per_round[arm].append(_time(lambda: c(*data["args"]), reps))
    finally:
        attention_interface.DSA_CHUNK_BY_BUCKET = saved

    med = {k: statistics.median(v) / depth for k, v in per_round.items()}
    base = med["off"]
    delta = {k: (v - base) / base * 100.0 for k, v in med.items()}
    nulls = [delta[k] for k in ("off#null1", "off#null2", "off#null3")]

    rec = dict(
        case=case,
        num_tokens=n,
        chain_depth=depth,
        rounds=rounds,
        reps=reps,
        resolved_chunk=resolved,
        lowered=lowered,
        us_per_call_median=med,
        per_round_us=per_round,
        delta_pct_vs_off=delta,
        null_floor_pct=[min(nulls), max(nulls)],
        null_span_pts=max(nulls) - min(nulls),
        finite=finite,
        bit_identical=bit_identical,
        max_abs_diff=max_abs_diff,
        distinct_output_values=distinct_values,
        negative_control_rejects=control_rejects,
    )
    report["cases"].append(rec)

    print(f"\n=== {case}  T={n}  chain={depth}  rounds={rounds} reps={reps}")
    print(f"  resolved chunk: off={resolved['off']} on={resolved['on']}")
    for arm in ("off", "on"):
        print(f"  lowered[{arm}]: gcs_tags={lowered[arm]['gcs_tags']} "
              f"mla_pallas_calls@gcz_{resolved[arm]}="
              f"{lowered[arm]['mla_calls']} "
              f"other={lowered[arm]['mla_calls_other']}")
    for k in arms:
        print(f"  {k:12s} med {med[k]:10.2f} us/call   {delta[k]:+7.3f}%")
    print(f"  null floor n=3 [{min(nulls):+.3f}%, {max(nulls):+.3f}%] "
          f"span {max(nulls) - min(nulls):.3f} pts")
    print(f"  ON vs OFF: finite={finite} bit_identical={bit_identical} "
          f"max_abs_diff={max_abs_diff} "
          f"negative_control_rejects={control_rejects}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="decode,prefill")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if jax.devices()[0].platform != "tpu":
        print("GATE: FAIL not a TPU")
        return 2

    report = {
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dispatch_floor_us": dispatch_floor(),
        "cases": [],
    }
    print(f"dispatch floor: {report['dispatch_floor_us']:.2f} us")

    # What the bucket policy is SUPPOSED to resolve to, per case. Stated here
    # rather than read back from the code, so a policy edit that changes the
    # branch boundaries fails this bench instead of silently redefining it.
    want = {"decode": (64, 16), "prefill": (64, 128)}

    fails = []
    for case in args.cases.split(","):
        rec = run_case(case, args.rounds, args.reps, report)
        exp_off, exp_on = want[case]
        if (rec["resolved_chunk"]["off"], rec["resolved_chunk"]["on"]) != \
                (exp_off, exp_on):
            fails.append(f"{case}: policy resolved "
                         f"{rec['resolved_chunk']} want off={exp_off} "
                         f"on={exp_on}")
        for arm, exp in (("off", exp_off), ("on", exp_on)):
            # `gcs_tags` come from the new `jax.named_scope` and only survive
            # into the text if this jax build emits debug info; empty is
            # tolerated, wrong is not. The `gcz_` count below is the check
            # that has to hold -- it is the KERNEL's own scope, so it cannot
            # be satisfied by a call site that computed the right number and
            # then failed to pass it.
            tags = rec["lowered"][arm]["gcs_tags"]
            if tags and set(tags) != {exp}:
                fails.append(f"{case}/{arm}: HLO carries gcs tags {tags}, "
                             f"expected only {exp}")
            want_calls = rec["chain_depth"] * math.ceil(
                rec["num_tokens"] / exp)
            if rec["lowered"][arm]["mla_calls"] != want_calls:
                fails.append(
                    f"{case}/{arm}: {rec['lowered'][arm]['mla_calls']} "
                    f"MLA pallas_calls at gcz_{exp} in the HLO, expected "
                    f"{want_calls} (depth {rec['chain_depth']} x "
                    f"cdiv({rec['num_tokens']}, {exp}))")
            stray = {c: k
                     for c, k in rec["lowered"][arm]["mla_calls_other"].items()
                     if k}
            if stray:
                fails.append(f"{case}/{arm}: HLO also contains MLA calls at "
                             f"other chunk sizes {stray}")
        if exp_off == exp_on:
            fails.append(f"{case}: the two arms resolve to the SAME chunk, "
                         "so this case discriminates nothing")
        if not rec["finite"]:
            fails.append(f"{case}: non-finite output")
        if rec["distinct_output_values"] < 1000:
            fails.append(
                f"{case}: only {rec['distinct_output_values']} distinct "
                "output values -- a near-constant output makes the "
                "bit-identity result vacuous")
        if not rec["bit_identical"]:
            fails.append(f"{case}: ON is NOT bit-identical to OFF "
                         f"(max_abs_diff {rec['max_abs_diff']})")
        if not rec["negative_control_rejects"]:
            fails.append(f"{case}: negative control did not reject; the "
                         "bit-identity result above is unfalsifiable")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"wrote {args.json_out}")

    if fails:
        for f in fails:
            print(f"GATE: FAIL {f}")
        return 1
    print("GATE: PASS policy resolved as specified, reached the kernel "
          "(HLO tags + call counts), output bit-identical, control rejects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
