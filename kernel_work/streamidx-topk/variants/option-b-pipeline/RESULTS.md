# Option B — chunked stage-A / stage-B pipeline

Step 3 for Option B of `measurement_and_design.md` §3.4, gated on Flip 3 (§4).

**Verdict: Flip 3 passes on its stated criterion, and Option B is still not worth adopting at
decode. It is a prefill win of +16.4 % (G = 2) / +24.4 % (G = 4) on the op, and a decode wash of
+4.7 % (G = 2) / −5.1 % (G = 4). The strongest argument against adopting it is not a number: at
prefill, the split B needs is not expressible inside the kernel at all** — see
[§7](#7-the-argument-against-adopting-b).

Everything below is measured on TPU7x unless it says `modelled` or `static`. Nothing under
`src/vllm_torchtpu/` was modified; `streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532`
and `sparsecore_topk.py` md5 `8bc9e25b8701a53a97704564b9fbc658` were read back on the pod in every
job below and are the checkout's, not the image's.

| Job | What | Result |
| --- | --- | --- |
| `j-c397124b-3d9c-476b-a03e` | Flip 3 gate: overlap, HLO dump, plan, contention anchors | Succeeded, 4/4 shards rc=0 |
| `j-67b9804d-e532-4281-9a04` | correctness (first attempt) | Failed — harness bugs, no measurement |
| `j-399e77d1-5c42-4fc4-aaf0` | correctness (second attempt) | Failed — harness bugs; did produce the tripwire self-test and a `20 passed` control |
| `j-562532b4-8705-4ad2-945f` | correctness: chunking + repo suite × 2 patched arms + control | Succeeded, 4/4 shards rc=0 |

---

## 1. What was measured, and how

### 1.1 The harness

`pipeline.py` builds, inside **one** `jax.jit`, `R` chained copies of the whole `G`-chunk program:

```
copy r:   A_0  B_0  A_1  B_1  …  A_{G-1}  B_{G-1}
```

Stage A is `probes/_stage_a.py` — the step-2 verbatim transcription of the incumbent's
`run_scores_kernel`, importing the incumbent's `kernel` body. Stage B is the incumbent's
`_sc_topk_direct` (`force_p1`) or `sparsecore_topk` (`nop1`). P3 established that TC/SC concurrency
is a property of a *single XLA program*, so staying inside one program is the whole point.

Four arms per (phase, G):

| arm | what it contains |
| --- | --- |
| `a` | the `G` stage-A calls only |
| `b` | the `G` stage-B calls only, on a pre-made score matrix; mutually **independent** |
| `b_chained` | as `b`, but the `G` calls are chained to each other — the control for SparseCore overlapping *itself* |
| `pipe` | the real pipeline: chunk *i*'s stage B consumes chunk *i*'s stage-A output |

### 1.2 Timing method

Device time is the **slope of a chained-`R` fit**, per `probes/stage-b-ksweep.md` §0: `R ∈
{1,2,3,4}`, best-of-9 per point, least squares on `t(R)`. The slope is dispatch-free device time and
the host term lands in the intercept, so no host floor is subtracted anywhere. `probe_flip3.py`
rejects any fit with `r² < 0.999`.

**Every fit reported here has `r² ≥ 0.99981`; the lowest is `b_G2` at decode, `r² = 0.9998157`.**

The brief's two requirements compose without tension: the overlap needs stage A and stage B inside
one program, chain-slope needs `R` copies inside one program, and one program containing `R` copies
of the whole `G`-chunk pipeline satisfies both. Because every arm is a slope, the fixed per-program
cost is in the intercept and P3's `F = 2·t_a(1) − t_a(2)` correction is unnecessary. **No wall clock
is reported as a result anywhere in this file.**

Consecutive copies are linked by two zero-valued but unfoldable carriers — `min(argmax(s[0,0]), 0)`
from a stage-A score buffer and `min(tk[0,0], 0)` from a top-k index — so XLA cannot hoist a copy
out, and copy *r+1* depends on **every** one of copy *r*'s stage-B outputs, so copies cannot
overlap each other and the measured saving is the within-copy one only. That second property is
verified from the compiled HLO, not assumed (§2).

### 1.3 Contention control

The pod was fanned out 4 ways and other agents were running jobs. Wall clocks are therefore
worthless and are not used. As a check that the *slopes* are not inflated, shard 3 re-measured
isolated `_sc_topk_direct` at three batch sizes and compared with `probes/stage-b-ksweep.md`:

| b | this job (µs) | ksweep (µs) | Δ |
| --- | --- | --- | --- |
| 64 | 27.36 | 29.0 | −5.6 % |
| 32 | 17.30 | 17.95 | −3.6 % |
| 16 | 17.42 | 17.35 | +0.4 % |

All three within 6 %, and two of the three *faster* than the reference — the fan-out is not
inflating the measurement.

### 1.4 Geometry

Read back from the job's own dump, not assumed.

| | decode | prefill |
| --- | --- | --- |
| `num_seqs` | 64 | 1 |
| `q_len` | 1 | 1024 |
| `context_len` | 8704 | 9216 |
| `num_tokens` (= b at G=1) | 64 | 1024 |
| `k` | 2048 | 2048 |
| `num_positions` (n) | 10240 | 10240 |
| stage-A case | `DECODE`, `seq_batch_size=4`, `bq=1` | `MIXED`, `seq_batch_size=1`, `bq=128` |

One caveat on `t_a`: the harness launches **one** stage-A kernel per chunk — the case that actually
does the work at that phase — where the incumbent's `_common_path` always launches three (two of
which have empty ranges at these shapes). Chunking multiplies whichever launches exist by `G`, so
the *launch-count* term is understated here by the two empty launches per chunk. P4 measured an
empty launch at 2.9 µs (TC), so at decode G = 4 that is ~23 µs unaccounted for — which makes the
decode conclusion below more negative, not less.

---

## 2. The dependency is real, and it survived into the compiled program

This is the failure mode the brief names: if the compiler can prove arm B does not need arm A's
data, it schedules them independently and the "overlap" is an artifact.

**How it was verified.** `probe_flip3.py --arm hlo` dumps the *scheduled* HLO
(`is_scheduled=true`) for all six (phase, G) programs at R = 2. `verify_hlo.py` then parses the
ENTRY computation offline and answers four questions from the operand DAG. All six files pass all
four.

An in-job checker written first reported the opposite and was **wrong**: it looked for stage B as a
`custom-call`, but the SparseCore top-k compiles to a `call-start` / `call-done` pair into an
`async_execution_thread="sparsecore"` computation. The only name-matching `custom-call`s in the
module are the callee's ROOT (which takes parameters, not the caller's buffers) and the
`PrepareAsyncCallStart` / `PrepareAsyncCallDone` pair (which have no data operands at all). Its
`every_B_depends_on_an_A=False` was a bug in the checker, not a property of the module. This is
recorded because it is exactly the shape of false negative that would have killed Option B for the
wrong reason.

What `verify_hlo.py` establishes, per file:

1. **`B → A` dependency survives.** Every stage-B `call-start` reaches a stage-A `tpu_custom_call`
   named `%StreamIdxTC-*` through its data operands.
2. **The carrier is layout-only, hence unfoldable.** The operand path is
   `['bitcast', 'bitcast-convert']` and nothing else — stage B literally reads the buffer stage A
   wrote. There is no arithmetic on the path for the compiler to constant-fold, and no `x*0` term
   anywhere. Example, decode G = 4:
   `%sc_topk_direct_b16_n10240_k2048.10.cloned.1.call-start -> stage A: ['bitcast', 'bitcast-convert']`.
3. **Pairing is 1:1.** One stage B per stage A, using the direct carrier walk. (The *transitive*
   reachable set is larger — copy 1's stage B reaches copy 0's stage A's through the inter-copy
   carrier — and is recorded separately as `pairing_transitive`; using it for the 1:1 test was a
   second, benign, bug in an earlier version of the verifier.)
4. **Copies are fully serialized.** Copy *r+1*'s first stage A reaches *every* one of copy *r*'s
   stage-B `call-done`s, so no cross-copy overlap contaminates the within-copy measurement.

And an independent structural confirmation of the timings, from the same parse — the number of
stage-A kernels scheduled inside each stage-B async window:

| | G = 1 | G = 2 | G = 4 |
| --- | --- | --- | --- |
| decode | `[0, 0]` | `[1, 0, 1, 0]` | `[3, 2, 1, 0, 3, 2, 1, 0]` |
| prefill | `[0, 0]` | `[1, 0, 1, 0]` | `[3, 2, 1, 0, 3, 2, 1, 0]` |

Zero at G = 1 — there is nothing to overlap, and the measured `hidden` is indeed ≈ 0. Non-zero and
increasing with G, matching the measured overlap fractions. The schedule and the clock agree.

Artifacts: `data/verify_hlo.json`; the six `*.hlo.txt` are in the job's
`command-container/hlo/`.

---

## 3. Flip 3 — the gate

`overlap_fraction = hidden / ideal_saving`, where `hidden = (t_a + t_b) − t_pipe` at the same `G`
and `ideal_saving = min(t_a, t_b)/G × (G−1)`. All times µs, all slopes.

### Decode (64 sequences, ctx 8704)

| G | t_a | t_b | t_a+t_b | t_pipe | hidden | ideal | **overlap** | vs G=1 pipe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 114.11 | 34.90 | 149.01 | 149.80 | −0.79 | 0 | n/a | 0 |
| 2 | 120.34 | 44.11 | 164.45 | 142.80 | 21.65 | 22.06 | **98.2 %** | **+4.67 %** |
| 4 | 133.27 | 85.85 | 219.12 | 157.43 | 61.70 | 64.39 | **95.8 %** | **−5.09 %** |
| 4, no p1 | 133.27 | 203.44 | 336.72 | 288.16 | 48.56 | 99.95 | 48.6 % | −92.36 % |

### Prefill (1 sequence, q_len 1024, ctx 9216)

| G | t_a | t_b | t_a+t_b | t_pipe | hidden | ideal | **overlap** | vs G=1 pipe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 263.32 | 440.77 | 704.09 | 727.67 | −23.58 | 0 | n/a | 0 |
| 2 | 272.99 | 449.28 | 722.27 | 608.34 | 113.93 | 136.50 | **83.5 %** | **+16.40 %** |
| 4 | 288.09 | 466.34 | 754.43 | 549.84 | 204.59 | 216.07 | **94.7 %** | **+24.44 %** |

### Verdict on the gate

The bars are **≥ 20 % at decode** and **≥ 7 % at prefill** (§3.4 Fallback Predicate 2). Measured:
**95.8–98.2 % at decode and 83.5–94.7 % at prefill.** Both cleared, by roughly a factor of five and
a factor of twelve.

**Flip 3 passes.** P3's 41.6 % was not an artifact of its missing dependency — the real dependency
overlaps *better* than P3's independent arms did, which is the opposite of the risk §4 named. That
is worth stating plainly: the concern was that XLA would serialise chunk *g*'s stage B against
chunk *g+1*'s stage A on the aliased score buffer, as `area1 §A8` shows it doing for the three
stage-A calls. It does not. Those three share one aliased buffer; the chunks here own disjoint
buffers, and the scheduler exploits that.

Two things this does **not** say.

- `b_chained` vs `b` — decode 43.70 vs 44.11 (G=2), 86.26 vs 85.85 (G=4); prefill 451.47 vs 449.28
  (G=2), 465.22 vs 466.34 (G=4). Within noise in every case. SparseCore does **not** overlap
  itself, so all of the measured saving is genuine TC/SC concurrency and none of it is stage B
  pipelining against stage B.
- `hidden` is negative at G = 1 (−0.79 decode, −23.58 prefill): the `pipe` program at G = 1 is
  slightly *slower* than `a` plus `b` measured apart, because its stage B reads a buffer stage A
  has just written rather than a resident pre-made one. That is why the "vs baseline" column is
  taken against `t_pipe` at G = 1 and not against `t_a + t_b`. It is the honest serial reference.

---

## 4. Why a passing gate still does not make decode worth it

The overlap *fraction* is near ideal. The *net* is not, because chunking inflates both stages:

| | t_a G1→G2→G4 | t_b G1→G2→G4 |
| --- | --- | --- |
| decode | 114.11 → 120.34 (+5.5 %) → 133.27 (**+16.8 %**) | 34.90 → 44.11 (+26.4 %) → 85.85 (**+146.0 %**) |
| prefill | 263.32 → 272.99 (+3.7 %) → 288.09 (+9.4 %) | 440.77 → 449.28 (+1.9 %) → 466.34 (+5.8 %) |

At decode, `G = 4` inflates the serial work by 47 % (149.01 → 219.12 µs) to buy a 95.8 %-efficient
overlap of 64 µs. The arithmetic does not close.

**The mechanism is the 32-row wave quantum, and it is measured directly.** `_sc_topk_direct` pads
`b` up to LANES = 16 but launches `cdiv(b, 32)` waves of 32 subcores. Per-chunk in-program stage B:
**22.06 µs at b = 32 and 21.46 µs at b = 16** — halving the rows buys 2.7 %. The isolated anchors
agree: 17.30 µs at b = 32, 17.42 µs at b = 16, against 27.36 µs at b = 64. So `t_b` roughly halves
going 64 → 32 and then *doubles* going to 4 × 16.

Prefill never touches the quantum: `b` is 1024 / 512 / 256, all comfortably above 32, so chunking
is nearly free and the overlap is nearly all profit. **Option B is a prefill option that happens to
be expressible at decode** — exactly what §3.4 concluded, now measured rather than modelled.

---

## 5. Does forcing `p = 1` unlock `G = 4`?

**At decode: it removes the cliff it was aimed at, and `G = 4` is still not a win.**

`_pick_partition(b=16, n=10240, k=2048)` returns **2** — read back from the incumbent chooser in
`--arm plan`. That second partition is a pure occupancy heuristic (`max_p = 32 // b`), not a
capacity requirement: `p_min = cdiv(10240, 65536) = 1`. Overriding it is therefore safe, and
`option_b_topk._require_p1_is_safe` re-derives that per call rather than trusting it.

| decode, G = 4 | forced p = 1 | incumbent `_pick_partition` = 2 |
| --- | --- | --- |
| stage B | **85.85 µs** | 203.44 µs |
| pipeline | **157.43 µs** | 288.16 µs |
| vs G = 1 | −5.09 % | −92.36 % |

So forcing `p = 1` is worth **2.37×** on stage B and turns a −92 % catastrophe into a −5 % wash. It
does not make `G = 4` a win.

**The direct answer to the question §3.4 asked.** The two gathers `p = 2` introduces *were* a
binding constraint, and they *are* removable — that part of §3.4's pessimism was right and its
remedy works. But `p_min` does **not** rise to meet it: it stays at 1. A *different* cliff survives,
and it is the one in §5's own second invariant — the 32-row wave quantum. At `b = 16` a full
32-subcore wave does 16 rows of work, so per-chunk stage B does not shrink and total stage B
doubles. **B's decode ceiling is not higher than §3.4 assumed; it is set by `b < 32`, not by the
gathers.** The practical consequence is that `G` at decode is bounded by `num_tokens / 32`, which at
the production decode batch of 64 means `G ≤ 2`.

**At prefill there is nothing to unlock.** `_pick_partition` already returns 1 at every `G`
(`b` = 1024 / 512 / 256, so `max_p = 1`), read back in `--arm plan`. The prefill numbers in §3 are
unaffected by `force_p1`.

---

## 6. The two hard assertions §5 requires

### 6.1 `_pick_partition` is never called on a chunked path — **PASS**

Enforced at trace time by `pipeline.p_tripwire`, which rebinds `sparsecore_topk._pick_partition` to
a raiser. It was armed on every `force_p1` arm of the Flip-3 job and on every `G` of all six
correctness shapes. It never fired.

**The tripwire is not inert**, which matters more than the negative result: `probe_correctness.py`
self-tests it by arming it around a deliberate `sparsecore_topk` call, which must raise.

```
[correctness] tripwire self-test: PASS -- tripwire fired on a deliberate sparsecore_topk call, as required
```

(job `j-562532b4-8705-4ad2-945f`, shard 0.)

One arm *does* call `_pick_partition`: `G4_nop1`, deliberately, to measure the cliff in §5. It runs
with the tripwire disarmed and is the only such arm.

### 6.2 `num_waves × 32 == b_pad` — **FAILS whenever `b < 32`, and that is the finding**

| where | b | b_pad | num_waves | `num_waves×32` | ok |
| --- | --- | --- | --- | --- | --- |
| decode G=1 | 64 | 64 | 2 | 64 | ✅ |
| decode G=2 | 32 | 32 | 1 | 32 | ✅ |
| **decode G=4** | **16** | **16** | **1** | **32** | ❌ |
| prefill G=1/2/4 | 1024 / 512 / 256 | same | 32 / 16 / 8 | same | ✅ |
| correctness shapes | 4, 8, 16 | 16 | 1 | 32 | ❌ |
| correctness shapes | 32 | 32 | 1 | 32 | ✅ |

This is not a wrongness — the index sets are exact at every one of those chunk sizes (§6.3) — it is
the occupancy statement, and it is the same fact as §4's 32-row quantum seen statically. Under
§3.4's own Fallback Predicate 1 (`b % 32 != 0` ⇒ refuse to chunk), **decode `G = 4` must be refused
by the design's gate**, and the measurement in §4 confirms the gate is right to refuse it. Option B
as designed is `G ≤ num_tokens / 32`.

### 6.3 Correctness

Job **`j-562532b4-8705-4ad2-945f`**, `JobStatus: Succeeded`, 4/4 shards rc = 0.

**Shard 0 — chunking correctness (`probe_correctness.py`), `OVERALL: PASS`.** Six shapes × `G ∈
{1, 2, 4}`. Every `G` produces **identical top-k index sets** to the unmodified incumbent, row by
row, ignoring order and `-1` padding; and every `G` is checked directly against every other `G`
(`G1_vs_G2: True, G2_vs_G4: True` on all six). `G = 1` is not a free pass: it runs Option B's own
single-group path, not the incumbent. The KV cache is built with the repo test file's own
`quantize_fp8_ue8m0` / `_to_byte_lane` record layout so the scores are real numbers rather than
garbage that would make tie-breaking the thing under test.

| shape | tokens | chunks at G=2 / G=4 | index sets match |
| --- | --- | --- | --- |
| `decode_b16_s512` | 16 | b=8×2 / b=4×4 | ✅ ✅ ✅ |
| `decode_b64_s1024` | 64 | b=32×2 / b=16×4 | ✅ ✅ ✅ |
| `prefill_b8_q8_s256` | 64 | b=32×2 / b=16×4 | ✅ ✅ ✅ |
| `prefill_b16_q4_s512` | 64 | b=32×2 / b=16×4 | ✅ ✅ ✅ |
| `mixed_b16_d8` (8 decode + 8×4-token) | 40 | b=8,32 / b=4,4,16,16 | ✅ ✅ ✅ |
| `decode_b16_s512_c2` (compression_ratio 2) | 16 | b=8×2 / b=4×4 | ✅ ✅ ✅ |

**Shards 1–3 — the repo suite.** Test file md5 **`becf9f23ace9424a7d67cd3b34de9afd`** in all three,
matching the checkout; the discovery scan reports **`distinct copies on disk: 1`**, so the image's
stale copy is not present under the mapped-in `tests/`. Collection count **20** in all three, so
this is not the stale 16-test file.

| shard | arm | result |
| --- | --- | --- |
| 1 | patched, `OPTION_B_G=1 OPTION_B_FORCE_P1=1` | `20 passed, 14 warnings in 41.43s` |
| 2 | patched, `OPTION_B_G=4 OPTION_B_FORCE_P1=0` | `20 passed, 14 warnings in 41.54s` |
| 3 | control, unpatched incumbent | `20 passed, 14 warnings in 42.21s` |

**What the patched arms do and do not prove.** They rebind `streamindex_topk` in the test module to
`option_b_topk.make_dropin(...)`, which routes 19 of 23 traces through Option B's own code —
`self (one group, whole batch)` — and 4 through the incumbent (`enable_early_exit=True`, which is a
batch-wide fast path with no pipeline to build). So the Option B stage-A launches, reshape/pad,
`_effective_row_lengths` and stage B are checked against the repo's NumPy ground truth at 20
parametrised shapes, under both settings of `force_p1`. They do **not** prove chunking, and cannot:
every case in that file has 1–3 sequences. The plugin says so itself in the terminal summary rather
than letting a green run read as more than it is. §6.3's shard 0 is the chunking evidence.

**Why the drop-in is a `jax.jit` object.** `test_streamindex_topk_shape` calls
`jax.eval_shape(streamindex_topk, …)`. `jax.eval_shape` honours the callee's `static_argnames` only
when `type(fun) is PjitFunction` (an exact type check, jax 0.10.2 `api.py:2318`); for anything else
it wraps the callable in a fresh `jit` and every leaf becomes a tracer, `k` included, and the
forwarded call dies with "Non-hashable static arguments are not supported". Seven tests failed that
way in `j-399e77d1` before the entry point was made a real `jax.jit`. The cost of that fix is
architectural and is the second half of §7: the jitted entry **cannot plan**, because `cu_q_lens`
and `distribution` are traced by the time its body runs.

---

## 7. The argument against adopting B

Not the numbers. The numbers say prefill is a +16.4 % / +24.4 % win on this op, which is real.

**The split B needs at prefill is not a kernel-local change.** Stage A's grid is over *sequences*
and its masking is driven by `cu_q_lens`, so a chunk boundary must fall on a sequence boundary. At
decode that is free — one token per sequence, any `G` dividing the batch works (subject to
`decode_req_batch_size` alignment and, per §6.2, `b ≥ 32`). At prefill the production shape is
**one sequence of 1024 query tokens**, and a single sequence cannot be split at all.

The prefill numbers in §3 were obtained by presenting that chunk to stage A as **`G`
pseudo-sequences** whose `seq_lens` advance by `q_len/G` — semantics-preserving (the per-token
causal position and per-token `eff_row_length` are unchanged, checked in `--arm plan`), but it is a
change to **how the runner packs the batch**, not a change confined to the kernel. Option B's entire
prize sits behind a modification to `mla_attention_op.py`'s caller and the scheduler's chunk
packing, which is a different review, a different blast radius, and a different set of things that
can regress.

**And a second, independent integration cost.** The plan is data — it needs concrete `cu_q_lens`
and `distribution`. vLLM calls this op under `jax.jit`, where both are traced. A production Option B
therefore needs the runner to hand the chunk plan in as static metadata. That is plumbing rather
than research, but it is plumbing in the same file the first problem is in, and it means the thing
that was pitched as "a change confined to `streamindex_topk.py`" touches neither less nor more than
the runner.

Weigh that against: decode gains nothing (`G ≤ 2` by §6.2, and `G = 2` is +4.67 %, inside the
±8.5 % end-to-end spread); the +24.4 % prefill figure is on an op, not on the model, and no
end-to-end run was made (out of scope); and §1.4's understated launch count makes decode's number
optimistic by ~23 µs at `G = 4`.

**Recommendation: do not build Option B into `src/` on the strength of this. If the prefill packing
change is being made for another reason, Option B becomes nearly free to add and should be
reconsidered then, at `G = 4`, with the `b ≥ 32` guard and `force_p1` both in place.**

---

## 8. Files

| file | what |
| --- | --- |
| `pipeline.py` | geometry, chunk plan, the two §5 guards, the four timing arms |
| `probe_flip3.py` | the arms: `plan`, `decode`, `prefill`, `hlo`, `anchors` |
| `verify_hlo.py` | offline verifier of the compiled HLO — the four checks in §2 |
| `option_b_topk.py` | the drop-in: eager planner + jitted `chunked_core` + `make_dropin` |
| `option_b_patch.py` | pytest plugin that rebinds `streamindex_topk` in the repo suite |
| `probe_correctness.py` | the chunking correctness suite of §6.3 |
| `shard.sh` | shard entry point; reads back both kernel md5s and the test-file md5 |
| `data/verify_hlo.json` | §2's per-file verdicts |
| `data/flip3-{decode,prefill}.shard*.json` | §3's raw slopes, intercepts and r² — every number in §3 and §4 recomputable from these |
| `data/flip3-hlo.shard2.json` | the in-job checker's output, kept so the false negative described in §2 stays on the record |
| `data/flip3-plan.shard3.json` | `_pick_partition` read-backs, the §6.2 guard tuples, and the pseudo-sequence equivalence check of §7 |
| `data/flip3-anchors.shard3.json` | §1.3's contention control |
| `data/option-b-correctness.json` | §6.3's per-shape, per-`G` index-set and guard results |

The six scheduled `*.hlo.txt` modules of §2 are **not** committed (≈40 MB); they are in job
`j-c397124b-3d9c-476b-a03e`'s `command-container/hlo/`, and `probe_flip3.py --arm hlo` regenerates
them.

## 9. What was not done

- **No end-to-end model benchmark.** Out of scope by instruction; all numbers here are
  microbenchmarks on a 4-chip pod.
- **Nothing was logged to the shared experiment log.** Out of scope by instruction.
- `streamindex_topk.py`, `sparsecore_topk.py` and `mla_attention_op.py` are unmodified, and nothing
  already in `instruments/` was touched.
- `t_a` counts one stage-A launch per chunk, not the incumbent's three (§1.4).
- Speculative decode (`static_q_len > 1`) was not exercised; `plan_bounds` handles it only through
  the generic `cu_q_lens` path and has no test for it.
