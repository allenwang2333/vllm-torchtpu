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

Harness fidelity: three axes, and what each one was worth
---------------------------------------------------------
The first revision of this bench reported the OPPOSITE sign from the
kernel-level sweep at decode (+24.8% here against -13.4% there) while agreeing
with it to 6.8% on the one-stage arm. An artifact that hits only the arm with
more pipeline stages is what suppressed cross-layer overlap looks like. Three
such divergences have now been found and each is a command-line axis rather
than an assumption, so its size can be read off instead of argued about.

MEASURED (job j-e1bff428, decode, per-layer marginal `D`): fixing the first
two moved the ON penalty from +28.1% to +25.7%, i.e. almost not at all -- they
were real (they cut `D_off` 178.41 -> 113.67 us, 36%) but they cut BOTH arms
alike, so they were never the sign flip. The third one is the candidate that
remains, and it is the only one that is arm-asymmetric by construction.

  --cache-mode per-layer|shared
      Production builds one `pallas::sparse_mla_attention_<layer>` op per
      layer, each with its own KV cache pair; 78 layers, 78 caches, no layer's
      gather waiting on another layer's cache write. `shared` (the first
      revision's behaviour) threads ONE pair through every iteration and so
      serialises what the chunk size exists to overlap.
  --no-donate
      Production's op is built with `donate_argnums=(0, 1)`
      (`layers/vllm/attention.py:1541`). The first revision used a plain
      `jax.jit`, so a 272 MB cache pair was copied per invocation.
  --serial-topk
      `topk` is a loop invariant here, so no iteration's gather depends on
      anything the chain produces and XLA may hoist all `depth` gathers and
      schedule them against each other. Production computes `topk_indices`
      per layer from that layer's own hidden states
      (`layers/vllm/custom_ops/mla_attention_op.py:185`), so layer `i`'s
      gather cannot start until layer `i-1` has finished. This is the one
      divergence that CANNOT cancel in the ON/OFF ratio: free cross-layer
      overlap is worth much more to the arm with one big exposed gather than
      to the arm that already overlaps internally. The signature is in the
      data -- chaining cut the isolated kernel-level per-layer cost by 40.3%
      at chunk 64 but by only 13.4% at chunk 16.

`--cache-mode shared --no-donate` is therefore the POSITIVE CONTROL, and it
has a job to do: the depth axis had no positive control before, so a null
there was uninterpretable. It has since done it -- that combination reproduced
the first revision's decode penalty to +28.1% against +28.0%, so the first
revision is reproducible and the disagreement is about fidelity, not noise.

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

    ql_nope = jnp.asarray(rng.standard_normal((num_tokens, NUM_HEADS, LKV_DIM),
                                              np.float32),
                          dtype=jnp.bfloat16)
    q_pe = jnp.asarray(rng.standard_normal((num_tokens, NUM_HEADS, ROPE_DIM),
                                           np.float32),
                       dtype=jnp.bfloat16)
    kv_c = jax.lax.bitcast_convert_type(
        jnp.asarray(_fp8_bytes(rng, (num_tokens, LKV_DIM))), jnp.float8_e4m3fn)
    k_pe = jax.lax.bitcast_convert_type(
        jnp.asarray(_fp8_bytes(rng, (num_tokens, ROPE_DIM))),
        jnp.float8_e4m3fn)

    return dict(
        case=case,
        num_tokens=num_tokens,
        # Everything EXCEPT the KV caches. The caches are built separately, on
        # demand, by `make_caches` -- see there for why they cannot live in a
        # shared tuple any more.
        static=(ql_nope, q_pe, kv_c, k_pe, jnp.asarray(topk),
                jnp.asarray(seq_lens), jnp.asarray(block_tables),
                jnp.asarray(qsl),
                jnp.asarray(np.asarray(distribution, np.int32))),
    )


#: Index of `topk_indices` inside the `static` tuple above, used by the
#: rolled-topk negative control.
STATIC_TOPK_IDX = 4

NOPE_SHAPE = (TOTAL_PAGES, PAGE_SIZE, dsa_gather.TILE_SUBROWS,
              dsa_gather.TILE_LANE_BYTES)
ROPE_SHAPE = (TOTAL_PAGES, PAGE_SIZE // dsa_gather.TILE_SUBROWS,
              dsa_gather.TILE_SUBROWS, dsa_gather.TILE_LANE_BYTES)
CACHE_PAIR_BYTES = int(np.prod(NOPE_SHAPE) + np.prod(ROPE_SHAPE))


def make_caches(n_pairs: int):
    """`n_pairs` INDEPENDENT KV cache pairs, as fresh device buffers.

    Two reasons this is a function rather than a field of `build_case`.

    Distinctness. Production builds one `pallas::sparse_mla_attention_<layer>`
    op per layer (`layers/vllm/attention.py:1535-1541`), each closing over that
    layer's own cache pair; GLM-5.2 has 78 of them and no layer's attention
    waits on another layer's cache write. A chain that threads ONE pair through
    every iteration imposes a serialisation edge production does not have, and
    that edge lands only on the arm with more pipeline stages -- which is
    exactly the shape of the artifact this bench is now being re-run to rule in
    or out. `--cache-mode per-layer` gives iteration `i` the pair at slot `2i`.

    Freshness. Under `donate_argnums` a consumed buffer is invalidated, so an
    arm cannot be handed the buffers another arm already ran on, and the timing
    loop has to thread each call's output caches into the next call -- which is
    also what production does step to step.

    Seeds differ per pair so the layers hold different data, as in production,
    but they are fixed, so every arm sees byte-identical caches and the ON-vs-
    OFF bit comparison stays meaningful.
    """
    out = []
    for i in range(n_pairs):
        r = np.random.default_rng(20260904 + 1000 * i)
        out.append(jnp.asarray(_fp8_bytes(r, NOPE_SHAPE)))
        out.append(jnp.asarray(_fp8_bytes(r, ROPE_SHAPE)))
    return tuple(out)


def make_chain(mesh,
               depth: int,
               per_layer: bool,
               donate: bool,
               serial_topk: bool = False):
    """`depth` wrapper calls, output fed back in as the next `ql_nope`.

    The feedback is what stops XLA from CSE-ing the repeats into one call,
    and it mirrors a layer stack: `output` and `ql_nope` have the same shape
    `[T, NUM_HEADS, LKV_DIM]`.

    Two axes, both of which the first version of this bench got wrong in the
    same direction, and both of which are now settable so the error can be
    measured rather than argued about:

    `per_layer`
        `True` gives iteration `i` its own cache pair, as production does.
        `False` threads one pair through all `depth` iterations, which is what
        the first version did -- iteration `i+1`'s gather then depends on
        iteration `i`'s cache write, suppressing exactly the cross-layer
        overlap that a smaller chunk size exists to create. `False` is kept as
        the POSITIVE CONTROL: it must reproduce the earlier +28% decode
        penalty, or the earlier job is simply irreproducible and neither
        number means anything.

    `donate`
        Production's op is built with `donate_argnums=(0, 1)`
        (`layers/vllm/attention.py:1541`). Without donation a 272 MB cache pair
        is copied on every invocation -- 201-453 us at plausible HBM rates,
        which brackets the first version's fitted PER-INVOCATION constant of
        357.2 us. (That figure was originally mislabelled a per-layer
        marginal; it is the coefficient on `1/depth`, which is where a
        per-invocation copy belongs, so the copy explanation is the coherent
        one. Measured: donation moves the per-layer marginal by 0.6%, from
        178.41 to 177.40 us, and the whole 272 MB copy lands in `C`, which
        drops 298.61 -> 15.97 us/call.)

    `serial_topk`
        The third divergence, found only after `per_layer` and `donate` were
        both fixed and the decode penalty did not move. `topk` is passed in as
        a loop invariant, so every iteration's gather depends on nothing the
        chain produces and XLA is free to hoist all `depth` gathers and
        schedule them against each other. Production cannot do that: each
        layer's `topk_indices` comes from that layer's own hidden states
        (`layers/vllm/custom_ops/mla_attention_op.py:185`, `topk_op(...)`),
        so layer `i`'s gather cannot start until layer `i-1` has finished.

        The free cross-layer overlap is worth far more to the arm with one big
        exposed gather than to the arm that already overlaps internally, so it
        does not cancel in the ratio -- chaining cut the isolated per-layer
        cost by 40.3% at chunk 64 and by only 13.4% at chunk 16. `True`
        restores the dependency edge without perturbing a single index.
    """

    def chain(caches, ql_nope, q_pe, kv_c, k_pe, topk, seq_lens, block_tables,
              qsl, dist):
        caches = list(caches)
        out = ql_nope
        for i in range(depth):
            j = 2 * i if per_layer else 0
            tk = topk
            if serial_topk:
                # Make this layer's gather indices depend on this layer's
                # input, as production's per-layer indexer does. The predicate
                # is true at runtime so `tk` is `topk` exactly, and the two
                # branches differ so XLA cannot fold the select away and lose
                # the edge. Reading one element is enough: the data dependency
                # is on the buffer, not on how much of it is consumed.
                ok = jnp.isfinite(out[0, 0, 0].astype(jnp.float32))
                tk = jnp.where(ok, topk, topk * 0)
            caches[j], caches[j + 1], out = \
                attention_interface.sparse_mla_attention(
                    out, q_pe, kv_c, k_pe, caches[j], caches[j + 1], tk,
                    seq_lens, block_tables, qsl, dist, mesh,
                    sm_scale=SM_SCALE, k_scale=K_SCALE)
            out = out.astype(ql_nope.dtype)
        # The updated caches are returned only when they are donated, and the
        # two go together on purpose. Donated, returning them is free -- XLA
        # aliases the output onto the input buffer -- and it is REQUIRED,
        # because a donated buffer is invalidated and the next call needs the
        # new one. Not donated, returning them would add a copy-out the
        # original version of this bench did not have, and `--no-donate` exists
        # precisely to reproduce that original. So `--no-donate` returns just
        # the activation, exactly as before.
        return (tuple(caches), out) if donate else out

    return jax.jit(chain, donate_argnums=(0, ) if donate else ())


def _call(chain, caches, static, donate):
    """One chain invocation; returns `(next_caches_or_None, activation)`."""
    if donate:
        new_caches, out = chain(caches, *static)
        return new_caches, out
    return None, chain(caches, *static)


def _time(chain, caches, static, reps, donate):
    """Mean of `reps` invocations; returns `(caches, us_per_chain_call)`.

    Under donation the caches must be threaded: call `n`'s output pair is
    call `n+1`'s input pair, because the input was invalidated. That is also
    what production does from one decode step to the next, so the dependency
    it creates is a real one rather than a harness artifact. Without donation
    the same buffers are reused every rep, as in the original version.
    """
    new, r = _call(chain, caches, static, donate)
    jax.block_until_ready((new, r) if donate else r)
    caches = new if donate else caches
    t0 = time.perf_counter()
    for _ in range(reps):
        new, r = _call(chain, caches, static, donate)
        if donate:
            caches = new
    jax.block_until_ready((caches, r) if donate else r)
    return caches, (time.perf_counter() - t0) / reps * 1e6


def dispatch_floor():
    """Per-invocation dispatch slope on a trivial one-argument jit.

    Informational only, and deliberately labelled as such in the report: it
    prices neither the chain's eleven arguments nor its 272 MB cache pair, so
    it is a LOWER bound on this bench's own per-invocation constant, not an
    estimate of it. The `C` term of the depth fit is the real measurement.
    """
    x = jnp.zeros((8, 128), jnp.float32)
    f = jax.jit(lambda a: a + 1.0)

    def bare(reps):
        jax.block_until_ready(f(x))
        t0 = time.perf_counter()
        for _ in range(reps):
            r = f(x)
        jax.block_until_ready(r)
        return time.perf_counter() - t0

    lo, hi = 20, 200
    bare(hi)
    return (bare(hi) - bare(lo)) / (hi - lo) * 1e6


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
    chunk size, not the KV page count. That makes the kernel's own scope a
    sufficient, independent read-back of which chunk size the policy actually
    delivered. (An earlier comment at the call site in
    `attention_interface.py` had this backwards; it has since been corrected
    there, and the `jax.named_scope` tag kept as belt and braces.)

    The returned NUMBER is informational only, and in particular it is NOT
    `depth * cdiv(T, chunk)`. MLIR prints repeated source locations through a
    deduplicated alias table (`#loc7 = loc("...")` defined once, referenced as
    `loc(#loc7)` thereafter), so a raw string count over `as_text()` counts
    unique location DEFINITIONS, not op instances. Measured on
    j-b77c8a7e-8da5-4236-9dba: decode depth 4 gave 3 (off) and 6 (on) against
    an op-count expectation of 4 and 16, and the `sparse-mla-gcs_` tag count
    was 2 in all four lowerings -- constant across chain depths 4 and 2, which
    a genuine per-op count cannot be. What IS load-bearing, and what the gate
    below checks, is EXCLUSIVITY: the lowering must contain this chunk size
    and zero occurrences of any other, which alias-dedup cannot fake because a
    chunk size that never appears has no location to alias.
    """
    return lowered_text.count(f"MLA-p_{CSA_TOPK}-bz_16-gcz_{chunk}")


#: The shipped policy function, captured before any arm overwrites the module
#: attribute. `off`/`on`/the nulls all run THIS function (with the module flag
#: set), so they test the real thing; a forced-chunk arm such as `c32` swaps in
#: a constant, which is the only way to reach a chunk size the bucket rule
#: never emits.
_REAL_POLICY = attention_interface.sparse_mla_gather_chunk_size


def _apply_arm(name: str, force):
    if force is None:
        attention_interface.sparse_mla_gather_chunk_size = _REAL_POLICY
        attention_interface.DSA_CHUNK_BY_BUCKET = (name == "on")
    else:
        attention_interface.DSA_CHUNK_BY_BUCKET = False
        attention_interface.sparse_mla_gather_chunk_size = \
            lambda n, _c=force: _c


def run_case(case: str,
             rounds: int,
             reps: int,
             report: dict,
             depth: int = 0,
             per_layer: bool = True,
             donate: bool = True,
             serial_topk: bool = False,
             extra_chunks=(),
             max_cache_gb: float = 40.0,
             data=None):
    data = data if data is not None else build_case(case)
    # `depth` is the number of MLA layer calls chained back to back. It is the
    # one knob that decides whether the gather of layer i+1 can overlap the
    # attention of layer i -- production has 78 layers in a row, the earlier
    # kernel-level sweep had exactly one call in isolation, and the two gave
    # OPPOSITE signs at decode. Sweeping it is how you tell which end of that
    # range production sits at.
    depth = depth or CHAIN[case]
    n = data["num_tokens"]
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

    # ---- arms. `#null1..3` are OFF as well: identical work under a different
    # ---- name, so their spread is this case's floor. One null is a number,
    # ---- three are a floor.
    arms = {"off": None, "on": None}
    for c in extra_chunks:
        arms[f"c{c}"] = int(c)
    arms.update({"off#null1": None, "off#null2": None, "off#null3": None})
    off_like = [k for k in arms if k.startswith("off")]

    n_pairs = depth if per_layer else 1
    # Every arm holds its own cache set for the whole interleaved timing loop,
    # so they are all resident at once. Refuse rather than OOM the pod: a job
    # that dies here reports nothing, and the depth sweep it was part of loses
    # its other points too.
    total_gb = len(arms) * n_pairs * CACHE_PAIR_BYTES / 1e9
    print(f"\n[{case} depth={depth} per_layer={per_layer} donate={donate}] "
          f"caches: {n_pairs} pair(s)/arm x {len(arms)} arms = "
          f"{total_gb:.2f} GB")
    if total_gb > max_cache_gb:
        print(f"GATE: FAIL {case} depth={depth}: cache footprint "
              f"{total_gb:.2f} GB exceeds --max-cache-gb {max_cache_gb}")
        return None

    # ONE `jax.jit` OBJECT PER ARM, and this is not a style choice. The chunk
    # size is read from a module global at trace time; `jax.jit`'s cache key
    # is (shapes, dtypes, ...) and does NOT include that global. A single
    # shared `chain` would trace once under whichever flag ran first and then
    # serve the SAME executable to every later arm -- five arms, one
    # compilation, a perfect null result, and nothing in the output would say
    # so. A fresh wrapper per arm has a fresh cache.
    chains = {
        arm: make_chain(mesh, depth, per_layer, donate, serial_topk)
        for arm in arms
    }
    # A SIXTH chain, never timed, that exists only to run the rolled-topk
    # negative control. The first version ran that control through
    # `chains["off"]`, which made `off` the one arm to execute an extra
    # invocation and a recompile-free re-dispatch -- so `off` was not
    # exchangeable with its own nulls, and the null floor it anchored was not
    # a floor for it. Semantically identical to `off`; asserted so below.
    poison_chain = make_chain(mesh, depth, per_layer, donate, serial_topk)

    resolved = {}
    lowered = {}
    outputs = {}
    saved_flag = attention_interface.DSA_CHUNK_BY_BUCKET
    try:
        for arm, force in arms.items():
            _apply_arm(arm, force)
            resolved[arm] = attention_interface.sparse_mla_gather_chunk_size(n)
            chain = chains[arm]
            if arm == "off" or arm == "on" or force is not None:
                probe = make_caches(n_pairs)
                try:
                    txt = chain.lower(probe,
                                      *data["static"]).as_text(debug_info=True)
                except TypeError:
                    txt = chain.lower(probe, *data["static"]).as_text()
                lowered[arm] = {
                    "gcs_tags": gcs_tags(txt),
                    "mla_calls": mla_call_count(txt, resolved[arm]),
                    "mla_calls_other": {
                        c: mla_call_count(txt, c)
                        for c in (8, 16, 32, 64, 128) if c != resolved[arm]
                    },
                }
                del probe
            caches = make_caches(n_pairs)
            new_caches, out = _call(chain, caches, data["static"], donate)
            outputs[arm] = jax.block_until_ready(out)
            del caches, new_caches

        # ---- correctness: every other arm must be bit-identical to OFF.
        def _bits(x):
            return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))

        a = np.asarray(outputs["off"].astype(jnp.float32))
        off_bits = _bits(outputs["off"])
        finite = bool(np.isfinite(a).all())
        bit_identical = True
        max_abs_diff = 0.0
        for arm in arms:
            if arm == "off":
                continue
            b = np.asarray(outputs[arm].astype(jnp.float32))
            finite = finite and bool(np.isfinite(b).all())
            bit_identical = bit_identical and bool(
                np.array_equal(off_bits, _bits(outputs[arm])))
            if finite:
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))))
        # Anti-vacuity: a run whose output is a constant (all-zero, say) would
        # be bit-identical across arms for a reason that has nothing to do
        # with the chunk size.
        distinct_values = int(np.unique(off_bits).size)

        # ---- negative control, run through the WHOLE pipeline rather than on
        # ---- a numpy copy. Execute an OFF-equivalent chain with
        # ---- `topk_indices` rolled by one row: same shapes, same executable,
        # ---- no recompile, but every token now attends to its neighbour's
        # ---- window. If the comparison above cannot see THAT, then
        # ---- `bit_identical=True` is unfalsifiable and says nothing about the
        # ---- chunk size.
        # ---- (The earlier form of this control flipped a bit in a numpy copy
        # ---- of the OFF output, which only proved `np.array_equal` works.)
        _apply_arm("off", None)
        clean = jax.block_until_ready(
            _call(poison_chain, make_caches(n_pairs), data["static"],
                  donate)[1])
        # The control is only evidence about `off` if the vehicle carrying it
        # IS `off`. Cheap to prove, so prove it rather than assume it.
        poison_vehicle_matches_off = bool(
            np.array_equal(off_bits, _bits(clean)))
        poison_static = list(data["static"])
        poison_static[STATIC_TOPK_IDX] = jnp.roll(
            poison_static[STATIC_TOPK_IDX], 1, axis=0)
        poison_out = jax.block_until_ready(
            _call(poison_chain, make_caches(n_pairs), tuple(poison_static),
                  donate)[1])
        control_rejects = not np.array_equal(off_bits, _bits(poison_out))
        control_frac_changed = float(np.mean(off_bits != _bits(poison_out)))
        del clean, poison_out

        # ---- timing, arms interleaved and reshuffled every round.
        live = {arm: make_caches(n_pairs) for arm in arms}
        rng = random.Random(20260904)
        per_round = {k: [] for k in arms}
        for _ in range(rounds):
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                _apply_arm(arm, arms[arm])
                live[arm], us = _time(chains[arm], live[arm], data["static"],
                                      reps, donate)
                per_round[arm].append(us)
        del live
    finally:
        attention_interface.DSA_CHUNK_BY_BUCKET = saved_flag
        attention_interface.sparse_mla_gather_chunk_size = _REAL_POLICY

    med = {k: statistics.median(v) / depth for k, v in per_round.items()}
    delta = {k: (v - med["off"]) / med["off"] * 100.0 for k, v in med.items()}
    # The floor is the spread of the OFF-equivalent arms about THEIR OWN mean,
    # not about `off`. Measuring it relative to `off` forces `off`'s deviation
    # to zero by construction and understates the span -- on the first run of
    # this bench by 9.3x at DECODE depth 2 (0.101 pts reported against 0.945
    # pts actual). `off` is one draw from the null, not the null.
    ref = statistics.mean([med[k] for k in off_like])
    null_dev = {k: (med[k] - ref) / ref * 100.0 for k in off_like}
    null_span_pts = max(null_dev.values()) - min(null_dev.values())
    delta_vs_null_mean = {k: (v - ref) / ref * 100.0 for k, v in med.items()}

    rec = dict(
        case=case,
        num_tokens=n,
        chain_depth=depth,
        per_layer_caches=per_layer,
        donate=donate,
        serial_topk=serial_topk,
        cache_gb=total_gb,
        rounds=rounds,
        reps=reps,
        resolved_chunk=resolved,
        lowered=lowered,
        us_per_call_median=med,
        per_round_us=per_round,
        delta_pct_vs_off=delta,
        delta_pct_vs_null_mean=delta_vs_null_mean,
        null_dev_pct=null_dev,
        null_floor_pct=[min(null_dev.values()),
                        max(null_dev.values())],
        null_span_pts=null_span_pts,
        finite=finite,
        bit_identical=bit_identical,
        max_abs_diff=max_abs_diff,
        distinct_output_values=distinct_values,
        negative_control_rejects=control_rejects,
        control_frac_changed=control_frac_changed,
        poison_vehicle_matches_off=poison_vehicle_matches_off,
    )
    report["cases"].append(rec)

    print(f"=== {case}  T={n}  chain={depth}  per_layer={per_layer}  "
          f"donate={donate}  rounds={rounds} reps={reps}")
    print(
        f"  resolved chunk: "
        f"{ {k: v for k, v in resolved.items() if not k.startswith('off#')} }")
    for arm in lowered:
        print(f"  lowered[{arm}]: gcs_tags={lowered[arm]['gcs_tags']} "
              f"mla_pallas_calls@gcz_{resolved[arm]}="
              f"{lowered[arm]['mla_calls']} "
              f"other={lowered[arm]['mla_calls_other']}")
    for k in arms:
        print(
            f"  {k:12s} med {med[k]:10.2f} us/layer  vs_off {delta[k]:+7.3f}%"
            f"  vs_nullmean {delta_vs_null_mean[k]:+7.3f}%")
    print(f"  null floor n={len(off_like)} (about their own mean) "
          f"[{min(null_dev.values()):+.3f}%, {max(null_dev.values()):+.3f}%] "
          f"span {null_span_pts:.3f} pts")
    print(f"  ON vs OFF: finite={finite} bit_identical={bit_identical} "
          f"max_abs_diff={max_abs_diff} distinct_out={distinct_values} "
          f"negative_control_rejects={control_rejects} "
          f"control_frac_changed={control_frac_changed:.4f} "
          f"poison_vehicle_matches_off={poison_vehicle_matches_off}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="decode,prefill")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--depth",
                    type=int,
                    default=0,
                    help="override the chain depth (0 = the per-case default)")
    ap.add_argument("--depths",
                    default="",
                    help="comma-separated chain depths to sweep; overrides "
                    "--depth. Three or more points make the C + depth*D fit "
                    "identifiable, which one or two cannot.")
    ap.add_argument("--cache-mode",
                    default="per-layer",
                    choices=("per-layer", "shared"),
                    help="per-layer matches production (one cache pair per "
                    "MLA call); shared reproduces the first version of this "
                    "bench and is the positive control")
    ap.add_argument("--no-donate",
                    action="store_true",
                    help="drop donate_argnums. Production donates "
                    "(attention.py:1541); this is for the positive control.")
    ap.add_argument("--serial-topk",
                    action="store_true",
                    help="make each layer's gather indices depend on the "
                    "previous layer's output, as production's per-layer "
                    "indexer does (mla_attention_op.py:185). Without it XLA "
                    "hoists every gather in the chain and schedules them "
                    "against each other, an overlap production cannot have.")
    ap.add_argument("--extra-chunks",
                    default="",
                    help="comma-separated chunk sizes to add as forced arms, "
                    "e.g. 32 -- values the bucket rule never emits")
    ap.add_argument("--max-cache-gb",
                    type=float,
                    default=0.0,
                    help="0 = 55%% of the device's reported HBM limit. The "
                    "per-layer cache mode scales as depth x arms, so a deep "
                    "sweep can ask for more HBM than the chip has; refusing "
                    "loudly beats an OOM that also loses the other depths.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if jax.devices()[0].platform != "tpu":
        print("GATE: FAIL not a TPU")
        return 2

    per_layer = args.cache_mode == "per-layer"
    donate = not args.no_donate
    extra = [int(c) for c in args.extra_chunks.split(",") if c.strip()]

    report = {
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dispatch_floor_us_informational": dispatch_floor(),
        "cache_mode": args.cache_mode,
        "donate": donate,
        "serial_topk": args.serial_topk,
        "extra_chunks": extra,
        "cases": [],
    }
    try:
        report["hbm"] = {
            k: v
            for k, v in jax.local_devices()[0].memory_stats().items()
            if "bytes" in k
        }
    except Exception as exc:  # pragma: no cover - platform dependent
        report["hbm"] = f"unavailable: {exc}"
    print(f"dispatch floor (informational): "
          f"{report['dispatch_floor_us_informational']:.2f} us")
    print(f"cache_mode={args.cache_mode} donate={donate} "
          f"serial_topk={args.serial_topk} extra_chunks={extra}")
    print(f"hbm: {report['hbm']}")

    max_cache_gb = args.max_cache_gb
    if not max_cache_gb:
        limit = (report["hbm"] or {}).get("bytes_limit") \
            if isinstance(report["hbm"], dict) else None
        max_cache_gb = 0.55 * limit / 1e9 if limit else 20.0
    report["max_cache_gb"] = max_cache_gb
    print(f"max_cache_gb: {max_cache_gb:.1f}")

    # What the bucket policy is SUPPOSED to resolve to, per case. Stated here
    # rather than read back from the code, so a policy edit that changes the
    # branch boundaries fails this bench instead of silently redefining it.
    want = {"decode": (64, 16), "prefill": (64, 128)}

    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    fails = []
    for case in args.cases.split(","):
        data = build_case(case)
        for depth in (depths or [args.depth]):
            rec = run_case(case,
                           args.rounds,
                           args.reps,
                           report,
                           depth,
                           per_layer=per_layer,
                           donate=donate,
                           serial_topk=args.serial_topk,
                           extra_chunks=extra,
                           max_cache_gb=max_cache_gb,
                           data=data)
            if rec is None:
                fails.append(f"{case} depth={depth}: skipped, see above")
                continue
            fails.extend(check_case(rec, want[case]))
        fails.extend(
            depth_fit(case, report, per_layer, donate, args.serial_topk))

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


def depth_fit(case, report, per_layer, donate, serial_topk):
    """Least-squares `T(depth) = C + depth*D` over this case's depth points.

    `D` is the marginal cost of one more pipeline stage and is the ONLY figure
    that should be quoted as the per-layer effect: the raw `us/layer` ratio at
    a single depth carries `C/depth`, a per-invocation constant that has
    nothing to do with the chunk size and that shrinks as depth grows. Needs
    >= 2 depths to fit at all and >= 3 to be worth believing.

    The `dD` column for the three null arms is the floor on `D` itself, and it
    is the number to read before believing any `dD` on `on`: a fit through two
    points inherits both points' noise, so a null spread wider than the ON
    effect means the depth axis has not resolved anything.
    """
    recs = [
        r for r in report["cases"]
        if r["case"] == case and r["per_layer_caches"] == per_layer
        and r["donate"] == donate and r["serial_topk"] == serial_topk
    ]
    if len({r["chain_depth"] for r in recs}) < 2:
        return []
    out = {}
    for arm in recs[0]["us_per_call_median"]:
        xs = [float(r["chain_depth"]) for r in recs]
        ys = [r["us_per_call_median"][arm] * r["chain_depth"] for r in recs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx)**2 for x in xs)
        d = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        out[arm] = {"D_us_per_layer": d, "C_us_per_call": my - d * mx}
    key = (f"{case}/per_layer={per_layer}/donate={donate}"
           f"/serial_topk={serial_topk}")
    report.setdefault("depth_fit", {})[key] = {
        "depths": [r["chain_depth"] for r in recs],
        "per_arm": out,
    }
    base = out["off"]["D_us_per_layer"]
    print(f"\n--- {key} depth fit over "
          f"{[r['chain_depth'] for r in recs]}  (T = C + depth*D)")
    for arm, v in out.items():
        print(f"  {arm:12s} D {v['D_us_per_layer']:9.2f} us/layer  "
              f"C {v['C_us_per_call']:9.2f} us/call  "
              f"dD {(v['D_us_per_layer'] - base) / base * 100.0:+7.3f}%")
    return []


def check_case(rec, want_pair):
    fails = []
    case, depth = rec["case"], rec["chain_depth"]
    tag = f"{case} depth={depth}"
    exp_off, exp_on = want_pair
    if (rec["resolved_chunk"]["off"], rec["resolved_chunk"]["on"]) != \
            (exp_off, exp_on):
        fails.append(f"{tag}: policy resolved {rec['resolved_chunk']} "
                     f"want off={exp_off} on={exp_on}")
    # Forced arms are checked against their own name, so a typo'd
    # `--extra-chunks` cannot silently benchmark the default a second time.
    expect = {"off": exp_off, "on": exp_on}
    for arm in rec["lowered"]:
        if arm.startswith("c"):
            expect[arm] = int(arm[1:])
    for arm, exp in expect.items():
        if rec["resolved_chunk"][arm] != exp:
            fails.append(f"{tag}/{arm}: resolved chunk "
                         f"{rec['resolved_chunk'][arm]}, expected {exp}")
        # `gcs_tags` come from the new `jax.named_scope` and only survive
        # into the text if this jax build emits debug info; empty is
        # tolerated, wrong is not. The `gcz_` count below is the check
        # that has to hold -- it is the KERNEL's own scope, so it cannot
        # be satisfied by a call site that computed the right number and
        # then failed to pass it.
        tags = rec["lowered"][arm]["gcs_tags"]
        if tags and set(tags) != {exp}:
            fails.append(f"{tag}/{arm}: HLO carries gcs tags {tags}, "
                         f"expected only {exp}")
        # Presence, not count -- see `mla_call_count`: MLIR alias-dedups
        # repeated locations, so the count is not `depth * cdiv(T, chunk)`
        # and never could be. Presence + the exclusivity check below is
        # the whole of what the HLO can tell us, and it is enough: it
        # cannot be satisfied by a call site that computed the right
        # number and then failed to pass it to the kernel.
        if rec["lowered"][arm]["mla_calls"] < 1:
            fails.append(
                f"{tag}/{arm}: no MLA pallas_call at gcz_{exp} in the "
                "HLO -- the resolved chunk never reached the kernel")
        stray = {
            c: k
            for c, k in rec["lowered"][arm]["mla_calls_other"].items() if k
        }
        if stray:
            fails.append(f"{tag}/{arm}: HLO also contains MLA calls at "
                         f"other chunk sizes {stray}")
    if exp_off == exp_on:
        fails.append(f"{tag}: the two arms resolve to the SAME chunk, "
                     "so this case discriminates nothing")
    if not rec["finite"]:
        fails.append(f"{tag}: non-finite output")
    # `distinct_output_values` is INFORMATIONAL. It is a proxy for
    # "the output is not a constant", and it is a poor one: at prefill
    # every output element is a softmax-weighted mean of 2048 random fp8
    # latents, which concentrates hard and then quantizes to a handful of
    # bf16 codes (52 measured on j-b77c8a7e-8da5-4236-9dba) while still
    # being fully data-dependent. The real anti-vacuity check is the
    # rolled-`topk` control below, which perturbs the INPUT and requires
    # the comparison to see it.
    if not rec["bit_identical"]:
        fails.append(f"{tag}: an arm is NOT bit-identical to OFF "
                     f"(max_abs_diff {rec['max_abs_diff']})")
    if not rec["poison_vehicle_matches_off"]:
        fails.append(f"{tag}: the negative control's chain does not reproduce "
                     "OFF on clean inputs, so the control says nothing "
                     "about OFF")
    if not rec["negative_control_rejects"]:
        fails.append(f"{tag}: rolled-topk control did not change the "
                     "output; the bit-identity result above is unfalsifiable")
    if rec["control_frac_changed"] < 0.10:
        fails.append(
            f"{tag}: rolled-topk control changed only "
            f"{rec['control_frac_changed']:.4f} of output elements -- too "
            "few for bit-identity to be meaningful evidence")
    return fails


if __name__ == "__main__":
    sys.exit(main())
