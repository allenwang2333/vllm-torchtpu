# Variant A1 — **drop the ReLU**. Gate result: **FAILED on all five regimes.**

`area6-algorithm.md` row A1 requires the accuracy gate **before any perf number**. It was run
first, it failed, and per that instruction **no A1 kernel was built and no perf number was
taken**. There is no `Δ time %` table in this file because nothing was timed.

**Job** `j-5acf2993-b56b-4a38-9abe` — `JobStatus=Succeeded`, `state=Complete`,
`03:11:50Z → 03:13:24Z` (94 s). `tpu-fanout: 4/4 shards succeeded`, every shard `rc=0`.

```
===== tpu-fanout EXIT CODES =====
shard0   chips=0        rc=0    bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh 0
shard1   chips=1        rc=0    bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh 1
shard2   chips=2        rc=0    bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh 2
shard3   chips=3        rc=0    bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh 3
tpu-fanout: 4/4 shards succeeded
```

`RUN_CMD`, **as read back from `cdk job recipe j-5acf2993-b56b-4a38-9abe --no-color`**:

```
cd /root/cloud-devkit/vllm-torchtpu && export PYTHONPATH=/workspace:$PYTHONPATH
  && bash /root/fanout/tpu-fanout.sh "bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh
  0" "bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh
  1" "bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh
  2" "bash /workspace/kernel_work/streamindex_topk/variants/drop-relu/a1_gate_shards.sh
  3"
```

`image:`, read back from the same recipe, is the requested one:
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`

`--map-dir` flags as submitted (both printed a `Packaging dir …` line):

```
--map-dir /home/annyan_google_com/workspaces/vllm-torchtpu/kernel_work:tpu-job:/workspace/kernel_work
--map-dir /home/annyan_google_com/workspaces/copy1/vllm-optimization-agent/.claude/skills/run-command-in-cdk/fanout:tpu-job:/root/fanout
```

**Checkout state, captured in the same shell invocation as the submit:**

```
=== vllm-torchtpu HEAD ===
c6c1acab11a3e36942d75226789bbb037593ebbb
annyan/streamidx-scores
=== vllm-torchtpu git status --porcelain ===
 M kernel_work/streamindex_topk/variants/narrow-record/narrow_inputs.py
 M kernel_work/streamindex_topk/variants/narrow-record/streamindex_topk_narrow.py
 M kernel_work/streamindex_topk/variants/right-size/run_correctness.py
 M kernel_work/streamindex_topk/variants/right-size/run_perf.py
 M kernel_work/streamindex_topk/variants/right-size/run_sc_coverage.py
 M kernel_work/streamindex_topk/variants/right-size/run_topk_width.py
?? kernel_work/streamindex_topk/variants/drop-relu/
?? kernel_work/streamindex_topk/variants/half-width-key/
?? kernel_work/streamindex_topk/variants/right-size/shard.sh
?? optimization_work/
=== cloud-devkit ===
4c6383fbd1578d56afa876f56904535715155cd2
?? recipes/experimental/gdn-comparison.yaml
```

Every modified/untracked path above belongs to a **concurrent variant agent**, not to this task.
This task touched only `kernel_work/streamindex_topk/variants/drop-relu/`. No kernel source was
changed by it, and none was needed: see "What was measured, and why no kernel" below.

## Sign convention

Binding, though nothing here exercises it: every percentage would be a change in **time**,
`(new − old)/old`, negative faster, signed only in a `Δ time %` column beside the two absolute
times it came from. **This file reports no timing.** The percentages that do appear are shares of
arithmetic work and are labelled as such.

## Tags

`measured (job id)` · `modelled` · `assumed` · `unmodelled`.

---

# 1. The gate

## 1.1 What was measured, and why no kernel

A1 replaces

```
score[t,s] = sum_h relu(q[t,h] . kv[s]) * w[t,h]        (the true function)
```

with

```
score[t,s] = (sum_h w[t,h] * q[t,h]) . kv[s]            (A1's function)
```

That is **a different function of the same inputs**, not a different way of evaluating the same
one. Its accuracy is therefore decided by arithmetic and by the input distribution — not by
Pallas, tiling, dtype, block size or the MXU. So the gate was run against the **idealised** A1:
exact float64 arithmetic, an exact top-k, and the identical causal visibility mask lifted directly
off the float64 ReLU oracle. Every real A1 kernel — fp8 operands, an f32 accumulator, a
re-quantised collapsed query vector, `sparsecore_topk`'s selection — is strictly worse than this.

The logic is one-directional and it is the reason one shard settles the candidate:

* **idealised A1 fails ⇒ every A1 kernel fails.** No implementation recovers it.
* idealised A1 passes ⇒ nothing is proven about a real kernel, which would still have to be built
  and re-gated.

The reference, the criterion, `RANK_SLACK = 256` and `MIN_RECALL = 0.99` all come from the
read-only instrument `instruments/test_correctness.py`, whose `oracle_scores` and `check` are
called directly. Nothing is re-derived. The single instrument interaction is a rebind of the
module attribute `TC.TOKEN_CHUNK` from the driver script, to bound host memory at `kv_len=65536`;
no instrument file was edited.

**The collapse itself is exact and was verified, not assumed.** `sum_h w_h (q_h . kv) ==
(sum_h w_h q_h) . kv` reproduces to `6.5e-16` relative in float64, and the driver's `qw` path is
what was checked. *The algebra is not what fails.* What fails is that the function the algebra
gives you is a poor **ranker** of the true score: Pearson correlation between the two score
fields is `0.679` on this fixture. So — answering the standing question — **the head-axis collapse
does hold in this kernel's dataflow. A1 does not die on the collapse; it dies on the gate.**

## 1.2 Verdict, per regime

**A1 FAILS on all five regimes, on both criteria, by 3.4× to 24.9× the allowed rank.**
Every quoted number is verbatim from the job log,
`measured (j-5acf2993-b56b-4a38-9abe)`, seed 0, one seed per regime.

| regime | S (`kv_len`) | rows | rows bad | `worst_rank=` (allowed 2304) | over by | `min_recall=` (required 0.99) | mean recall | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `decode_prod` | 8192 | 64 | **64 of 64** | **7873** | 3.42× | **0.69189** | 0.72653 | **FAIL** |
| `mixed_prod` | 8192 | 1024 | **1024 of 1024** | **7777** | 3.37× | **0.67578** | 0.73711 | **FAIL** |
| `mixed_32k` | 32768 | 1024 | **1024 of 1024** | **30832** | 13.4× | **0.48047** | 0.58197 | **FAIL** |
| `mixed_64k` | 65536 | 1024 | **1024 of 1024** | **57391** | 24.9× | **0.39844** | 0.51999 | **FAIL** |
| `decode_64k` | 65536 | 6 | **6 of 6** | **40939** | 17.8× | **0.50293** | 0.52555 | **FAIL** |

Not one row of 3138 passed on any regime. Log lines, verbatim:

```
[decode_prod A1] FAIL -- 64 of 64 rows bad (worst_rank=7873 of allowed 2304, min_recall=0.69189 of required 0.99)
[mixed_prod A1]  FAIL -- 1024 of 1024 rows bad (worst_rank=7777 of allowed 2304, min_recall=0.67578 of required 0.99)
[mixed_32k A1]   FAIL -- 1024 of 1024 rows bad (worst_rank=30832 of allowed 2304, min_recall=0.48047 of required 0.99)
[mixed_64k A1]   FAIL -- 1024 of 1024 rows bad (worst_rank=57391 of allowed 2304, min_recall=0.39844 of required 0.99)
[decode_64k A1]  FAIL -- 6 of 6 rows bad (worst_rank=40939 of allowed 2304, min_recall=0.50293 of required 0.99)
```

and a sample failing row, `mixed_64k`-scale, from `decode_64k`:

```
row 0: worst_rank=36535 (allowed 2304), recall=0.50293 (min 0.99), 1018 of 2048 indices outside the true top-2048
```

**Half the returned set is wrong.** The incumbent's margin was `worst_rank=2109 / min_recall=0.99561`
at `decode_prod` — 206 of the 256 slack unused. A1 does not consume the remaining 50 ranks; it
overshoots the whole band by an order of magnitude.

**It gets worse as context grows**, monotonically: mean recall `0.737 → 0.582 → 0.520` as S goes
`8192 → 32768 → 65536`. Picking a fixed 2048 out of a larger pool is a harder ranking problem, and
`mixed_64k` is the regime the end-to-end A/B targets. That is the opposite of the direction the
candidate needed.

## 1.3 Negative control — fired on **every** row of **every** regime

Same mutation as the instrument: zero the q rows of the second half of the heads, score with the
mutated q, compare against the **unmutated** oracle. Under A1 as under the ReLU form this is
exactly row A6 (16 of 32 heads), because `w_h · 0 = 0` just as `relu(0 · kv) = 0`.

| regime | control fired on | control `worst_rank=` | control `min_recall=` |
| --- | --- | ---: | ---: |
| `decode_prod` | **64 of 64 rows** | 8178 | 0.45020 |
| `mixed_prod` | **1024 of 1024 rows** | 8167 | 0.37207 |
| `mixed_32k` | **1024 of 1024 rows** | 32684 | 0.16260 |
| `mixed_64k` | **1024 of 1024 rows** | 65328 | 0.11328 |
| `decode_64k` | **6 of 6 rows** | 64513 | 0.19873 |

No PASS anywhere in this file is vacuous.

Two readings of that table. First, **A1 is a better ranker than dropping half the heads and still
nowhere near the bar** — at `mixed_64k`, `min_recall` 0.398 against the control's 0.113. Being
better than the known-broken arm is not the criterion. Second, the control's `decode_prod` point
(`8178 / 0.45020`) sits beside the sheet's recorded A6 rejection (`8177 / 0.49512`,
`measured (j-6d675a58-7ecd-4b44-9171)`), which is a cross-check that this harness lands where the
kernel-side one did. The two are not identical arms — that one drops heads inside the ReLU form,
this one inside A1's — so the near-agreement on `worst_rank` is partly `worst_rank` saturating;
the recall figures differ as expected.

## 1.4 The harness self-check

A third arm per regime scores with the **oracle itself** and selects exactly. It must be a perfect
pass or nothing else on that regime means anything.

```
[decode_prod self-ref] PASS -- worst_rank=2048, min_recall=1.00000
[mixed_prod  self-ref] PASS -- worst_rank=2048, min_recall=1.00000
[mixed_32k   self-ref] PASS -- worst_rank=2048, min_recall=1.00000
[mixed_64k   self-ref] PASS -- worst_rank=2048, min_recall=1.00000
[decode_64k  self-ref] PASS -- worst_rank=2048, min_recall=1.00000
```

`worst_rank = 2048` of `k = 2048` is the floor: mask handling, the `-1`-padded output format, the
oracle and `check` are all sound, so the A1 failures are the approximation and not the plumbing.
Separately, the fast rank-matrix checker used for the M sweep is asserted identical to
`instruments/test_correctness.check` on the A1 arm of every regime before any sweep number is
emitted; the log carries that line per regime.

## 1.5 What the gate cannot see

Quoted forward from `measurement_and_design.md` §5.3 rather than discovered here.
`instruments/inputs.py:170-179` draws `q` and `kv` from `standard_normal` and `w` from
`uniform(−1.5, 1.5)`; the real checkpoint's `w` is ~50% negative with `min = −3.33, max = 2.81`,
close enough that the fixture is defensible, but **the score distribution under real `q`/`kv` is
not established** — `assumed`. One seed (0) per regime.

That caveat does not rescue A1 and should not be used to. The gate is missed by 3.4× to 24.9× on
`worst_rank` and by 0.30 to 0.59 absolute on recall, uniformly across five regimes and 3138 rows,
with a monotone trend in S. No plausible change of input distribution closes a gap of that size,
and the same fixture is the one on which the incumbent passes and A6 was rejected.

---

# 2. A3 — the two-stage fallback

Run in the same job on the same oracle, because the oracle is the expensive part and computing A3
alongside cost nothing. Row A3: score all S with A1's cheap form, keep the top **M**, exact-rescore
the shortlist with the real ReLU form, take the top `k = 2048`.

The rescore here reads the **float64 oracle** at the shortlist positions — again the idealised best
case, so these M values are **lower bounds** on what a real A3 would need. Recall after such a
rescore is exactly `|shortlist ∩ true top-k| / k`, since the rescore orders the shortlist by the
true score.

## 2.1 The sweep

`measured (j-5acf2993-b56b-4a38-9abe)`. `M*` = the smallest swept M that passes both criteria on
every row. The grid is coarse, so `M*` is an upper bound and the bracket below it is given.

| regime | S | M* (passes) | **M\*/S** | at M\*: `worst_rank=` | at M\*: `min_recall=` | last failing M (bracket) | at that M: `min_recall=` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_prod` | 8192 | **6144** | **0.750** | 2059 | 0.99463 | 5120 (33 of 64 rows bad) | 0.97705 |
| `mixed_prod` | 8192 | **6144** | **0.750** | 2066 | 0.99121 | 5120 (272 of 1024 bad) | 0.97754 |
| `mixed_32k` | 32768 | **16384** | **0.500** | 2066 | 0.99170 | 12288 (340 of 1024 bad) | 0.96924 |
| `mixed_64k` | 65536 | **32768** | **0.500** | 2055 | 0.99658 | 24576 (3 of 1024 bad) | 0.98828 |
| `decode_64k` | 65536 | **24576** | **0.375** | 2056 | 0.99609 | 16384 (5 of 6 bad) | 0.97949 |

The full 13-point sweep per regime is in the job log and, extracted from it verbatim, in
`a1_gate_results.json` beside this file. The shape is the same everywhere: recall climbs
slowly and the last stretch to 0.99 is expensive — at `mixed_64k`, `M = 8192` (`M/S = 0.125`)
gives `min_recall = 0.81006`, `M = 16384` gives `0.95654`, and `M = 24576` is still 3 rows short at
`0.98828`.

`min_recall` is the binding criterion at every regime; `worst_rank` clears 2304 slightly earlier.

## 2.2 What fraction of A1's saving survives at M\*

`modelled`, from a MAC count, with the assumption stated so it can be argued with. Per scored
position the incumbent does 32 head dot-products of dim 128 = 4096 MACs; A1's cheap form does one,
128 MACs. A3 pays the cheap form over S and the full form over M:

```
cost(A3)/cost(incumbent) = 1/32 + M/S
cost(A1)/cost(incumbent) = 1/32
surviving fraction = (1 − 1/32 − M/S) / (1 − 1/32)
```

| regime | M*/S | A3 scoring-MAC ratio | scoring MACs removed | **fraction of A1's saving surviving** |
| --- | ---: | ---: | ---: | ---: |
| `decode_prod` | 0.750 | 0.781 | 21.9% | **22.6%** |
| `mixed_prod` | 0.750 | 0.781 | 21.9% | **22.6%** |
| `mixed_32k` | 0.500 | 0.531 | 46.9% | **48.4%** |
| `mixed_64k` | 0.500 | 0.531 | 46.9% | **48.4%** |
| `decode_64k` | 0.375 | 0.406 | 59.4% | **61.3%** |

**Read that table with its currency, because the currency is wrong.** MACs are not what this step
costs. Shard 3 of `j-b54e9716-3db5-4f1f-b8d4` measured the MIXED inner step as MXU 1.084 µs
(14.8%), relu/`w`/scale/masks 2.401 µs (32.7%), 32-head-sum bundle 3.858 µs (52.5%), and showed
that **doubling the matmul's FLOPs costs +2.18% of step time**. So the 32× MAC reduction was never
the prize; the prize was the 85.2% of the step sitting on the VPU. Under A3 that VPU chain still
runs, on M positions instead of S — which scales it by `M/S`, i.e. **0.5 at both long-context
regimes and 0.75 at both production regimes**. In the currency that was measured, A3 at
`mixed_64k` removes about half of the 85%, before paying for anything it adds.

## 2.3 What A3 adds, which is unpriced here and is probably decisive

Three costs A3 introduces that this measurement does not carry. They are named rather than
estimated.

1. **A top-M-of-S selection stage that does not exist today** — `unmodelled`. At `M/S = 0.5` this
   is a selection problem *of the same order as the one it is meant to make cheaper*. The whole
   premise of a shortlist is `M ≪ k`-ish relative to S; `M = S/2` is not a shortlist. Shard 2 of
   `j-b54e9716-3db5-4f1f-b8d4` priced the cheapest possible thing in this family at the A/B's
   scores width: one compare-and-count pass is `232.31 µs measured`, and `sparsecore_topk` at that
   width is `3071.85 µs measured`. A3 would run a selection **twice** — once for the shortlist,
   once for the final top-2048.
2. **An irregular gather of M scattered KV positions** for the rescore — `unmodelled`, and against
   a paged, shuffled block table. The incumbent's KV reads are sequential over `bkv` blocks.
3. **A second scores buffer**, or a re-pass over the first. The scores buffer is already the
   block's largest tensor (276.8 MB at `mixed_64k`).

**Recommendation, stated as judgement and not as a measurement: do not build A3.** At the two
regimes the A/B targets it needs `M/S = 0.5`, halves a VPU tail while adding a full extra
selection pass over the same data, and its own gate margin at that M (`min_recall = 0.99658`
against 0.99 required) is thinner than the incumbent's. What would overturn this is a priced
top-M-of-S at `M = S/2` coming in far below the ~232 µs single-pass floor — which shard 2's
per-pass traffic argument says is not available on this hardware.

---

# 3. Consequences to carry upward

1. **A1 is dead. Row A1 in `step2/sheets/area6-algorithm.md` should be marked
   `fails, measured, job j-5acf2993-b56b-4a38-9abe`,** the way row A6 already is, with the five
   regimes' margins. `measurement_and_design.md` §7.5 item 1 — *"A1 … inserted above all of them if
   its recall gate passes"* — resolves to **not inserted**. §7.5's ranking under the confirmed VPU
   hypothesis reverts to **A, E, B, C, D** with no sixth option.
2. **The 85% of the MIXED inner step that A1 was aimed at is still there and now has no
   candidate.** Shard 3 measured relu/`w`/masks at 32.7% and the 32-head-sum bundle at 52.5%; A1
   was the only proposal that removed both, and it removed them by computing a different function.
   Whatever comes next has to attack that 85% *exactly*. The two exact routes left on the area-6
   sheet are **E1** (fold `w` into `q` at the quantizer — removes one VPU pass over
   `[T,32,S]`, exact iff `w ≥ 0`, which is open and probably false) and **E2** (re-tile to
   `(position, head)` so the head sum becomes a lane reduction). Neither removes the head sum;
   they make it cheaper. Open shard 5 of the decider job — split the 3.858 µs bundle into
   reduction / store / fixed cost with a `NUM_HEADS` ladder — is the prerequisite for pricing
   either, and it is now the highest-value cheap probe in the space.
3. **This gate is reusable and costs one shard.** `a1_gate.py` gates *any* rescoring of the
   scoring half against the float64 ReLU oracle on all five regimes, with the negative control and
   a harness self-check, in 94 s of wall clock and without writing a kernel. Row **A2** (clamp
   `w ← max(w,0)`, which makes E1's precondition true by construction) is the obvious next user:
   it is a one-line change to the surrogate and it has never been gated. Point it at A2 before
   anyone builds E1.
4. **The gate machinery now has a second real rejection on the record.** `06-algorithm.md`'s gap
   *"no probe measures a genuine approximation being rejected by a gate"* was closed once by A6.
   A1 closes it again, on five regimes rather than two, and this time against an idealised
   float64 implementation of the approximation rather than a kernel — which is the cheaper and
   stronger form of the test.

## Probe record — the five fields

```
probe:    Does A1 (drop the ReLU, collapse the head axis into the contraction) preserve
          membership of the causal top-2048 well enough to pass the named accuracy gate?
          And if not, at what shortlist size M does the A3 two-stage form reach the bar?
config:   GLM-5.2 indexer shapes, all five regimes (decode_prod, mixed_prod, mixed_32k,
          mixed_64k, decode_64k), k=2048, RANK_SLACK=256, MIN_RECALL=0.99, seed 0,
          float64 oracle + float64 idealised A1 + exact top-k, inputs from
          instruments/inputs.py (q,kv ~ N(0,1) after fp8 quantization; w ~ U(-1.5,1.5)),
          image vllm-torchtpu-annyan_google_com:13617694, host CPU (no TPU compute)
job:      j-5acf2993-b56b-4a38-9abe   (JobStatus=Succeeded, 4/4 shards rc=0)
answer:   FAIL on 5 of 5 regimes, 3138 of 3138 rows. worst_rank 7873 / 7777 / 30832 / 57391 /
          40939 against an allowed 2304; min_recall 0.69189 / 0.67578 / 0.48047 / 0.39844 /
          0.50293 against a required 0.99. Negative control fired on every row of every
          regime; oracle self-reference arm perfect (worst_rank=2048, min_recall=1.0)
          everywhere. A3 reaches the bar only at M/S = 0.750 (both prod regimes),
          0.500 (mixed_32k, mixed_64k) and 0.375 (decode_64k).
decided:  A1 is rejected without a kernel being written and without a perf number being taken.
          §7.5 item 1 resolves to "not inserted"; the ranking stays A, E, B, C, D. A3 is not
          recommended: at M/S=0.5 it keeps only ~48% of A1's modelled saving while adding an
          unpriced top-M-of-S pass at a width where one compare-and-count pass alone measures
          232.31 us (j-b54e9716-3db5-4f1f-b8d4). The 85% of the MIXED inner step A1 targeted
          reverts to having no candidate.
```

## Files

| file | what |
| --- | --- |
| `a1_gate.py` | the gate driver: idealised float64 A1, the negative control, the harness self-reference arm, and the A3 M sweep. Calls `instruments/test_correctness.py`'s `oracle_scores` and `check` directly; the only instrument interaction is a `TC.TOKEN_CHUNK` module-attribute rebind. Has a `--dry` mode (tiny shape, `JAX_PLATFORMS=cpu`) used before submitting. |
| `a1_gate_shards.sh` | four-way fan-out driver: shard 0 `decode_prod`, 1 `mixed_prod`, 2 `mixed_32k`, 3 `mixed_64k,decode_64k`. |
| `a1_gate_results.json` | every arm and every sweep point of `j-5acf2993-b56b-4a38-9abe`, extracted verbatim from the four shards' `=== JSON ===` blocks in the job log. |

**No diff.** The gate failed, so no kernel was written and there is nothing to apply.
