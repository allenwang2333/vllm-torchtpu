# Step 3, first job: the four deciders

**Job** `j-b54e9716-3db5-4f1f-b8d4` — `JobStatus=Succeeded`, `state=Complete`, 2 m 13 s
(`02:50:52Z` → `02:53:05Z`). `tpu-fanout: 4/4 shards succeeded`, every shard `rc=0`.

```
===== tpu-fanout EXIT CODES =====
shard0   chips=0        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh 0
shard1   chips=1        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh 1
shard2   chips=2        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh 2
shard3   chips=3        rc=0    bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh 3
tpu-fanout: 4/4 shards succeeded
```

`RUN_CMD`, **as read back from `cdk job recipe j-b54e9716-3db5-4f1f-b8d4`**:

```
cd /root/cloud-devkit/vllm-torchtpu && export PYTHONPATH=/workspace:$PYTHONPATH
  && bash /root/fanout/tpu-fanout.sh "bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh
  0" "bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh
  1" "bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh
  2" "bash /workspace/kernel_work/streamindex_topk/probes/step3_deciders.sh
  3"
```

`image:` read back as
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`
— the requested one.

**Checkout state, captured in the same shell invocation as the submit:**

```
=== vllm-torchtpu HEAD ===
06c6bfae1ad25f7d2999fef581f10ac73a28ec08
annyan/streamidx-scores
=== vllm-torchtpu git status --porcelain ===
?? kernel_work/streamindex_topk/variants/
?? optimization_work/
=== cloud-devkit ===
4c6383fbd1578d56afa876f56904535715155cd2
?? recipes/experimental/gdn-comparison.yaml
```

The dispatch named HEAD `950974c2`. It had moved two commits by submit time and moves once
more with this file. `950974c2..06c6bfae` is `11f26d75` (a `measurement_and_design.md` edit
withdrawing §10 item 1's claim that no `(8,8,8)` arm exists) plus this task's own two probe
commits. **No kernel source changed** between `950974c2` and the submitted tree:
`git diff --name-only 950974c2 HEAD` at submit time listed only files under
`kernel_work/streamindex_topk/`. The two untracked directories are the concurrent variant
agents' work and were copied into the pod by `--map-dir` but are read by nothing here.

## Sign convention

Every percentage is a change in **time**, `(new − old)/old`. Negative is faster. Signed
percentages appear only in tables under a `Δ time %` column, and every one sits in the same row
as the two absolute times it was computed from. Prose says the direction in words.

## Tags

`measured (job id)` · `modelled` · `assumed` · `unmodelled`. `unmodelled` means the floor has no
term for the thing; the word appears with no number beside it.

## A caveat that applies to every absolute number below

Four shards ran concurrently on four chips of one host. They share HBM and 16 CPUs. The
inflation is not uniform: shard 1's `decode_64k` baseline reads `0.20202844752930105` ms here
against `0.19925692467950284` ms at `j-3aa755dd` (1.4% slower) and its `mixed_prod` baseline
`0.6724997947458178` against `0.6666850706096739` (0.9% slower) — small. Shard 2's `sc_topk`
reads **541.71 µs** at `mixed_prod`'s width against the **399.60 µs** of `j-5f8f4a08` — **35.6%
slower**, which is large. So **ratios taken inside one shard are the trustworthy quantity and
every conclusion below is built on one**; absolute µs are contended and are quoted for
re-derivation, not for cross-job comparison.

---

# Shard 0 — the flipping assumption: the decode:mixed step mix at the A/B shape

**Question.** At the A/B shape, what fraction of `streamindex_topk` invocations, and of its time,
is the DECODE band versus the MIXED band? §9.2 makes rank 1 turn on this.

**Method: `modelled`, not measured.** The real server cannot be run in this job. What follows is a
derivation from the scheduler's chunking arithmetic at the A/B's settings, with every step of the
arithmetic shown. It is labelled `modelled` throughout and there is no `measured` number anywhere
in this shard. What it does **not** cover is listed at the end and the list is not short.

## The settings

GLM-5.2, ISL 65536, OSL 1024, `max_model_len` 66560, `--max-concurrency=64`, DP=16,
`--max-num-batched-tokens=1024`, `--no-enable-prefix-caching`.

`--max-concurrency=64` over DP=16 is **4 concurrent requests per engine** [`assumed`: even DP
routing].

## The step mix

In a step that contains a prefill chunk, one of the four slots is prefilling and the other three
are decoding, so the 1024-token budget is `3` decode tokens + **`1021`** prefill tokens
[`modelled`]. In a pure-decode step all four slots decode, 4 tokens.

Let `N` = model steps in steady state, `M` = those containing a prefill chunk, `R` = requests
completed.

Prefill-token balance — each request needs 65536 prefill tokens at 1021 per step:

```
1021 · M = 65536 · R        →  M = 64.1881 · R
```

Decode-token balance — each request emits 1024 output tokens, the first of which comes out of its
last prefill chunk, so it needs **1023** decode steps. Pure-decode steps emit 4 tokens, prefill
steps emit 3:

```
4(N − M) + 3M = 1023 · R
4N − M        = 1023 · R
4N            = 1023 R + 64.1881 R = 1087.1881 R
N             = 271.7970 · R
```

Cross-check: a slot is occupied for `4N/R = 1087.19` steps per request, and a request needs
`64.19 + 1023 = 1087.19` steps. Consistent.

```
M / N       = 64.1881 / 271.7970 = 0.236168   → 23.62% of steps contain a prefill chunk
(N − M) / N = 0.763832                        → 76.38% are pure decode
```

## Which bands execute, and how often

`streamindex_topk` makes **three** `pallas_call`s, reading `num_kv_pages_per_blocks[0]`, `[0]`,
`[2]` (`streamindex_topk.py:765 / 783 / 801`): DECODE batched, DECODE remainder, MIXED. Slot `1`
is read by no executing band. The split is driven by `distribution`
(`mla_attention_op.py:183` → `tpu_runner.py:3175-3183`, `distribution = [chunk_num_decode,
chunk_num_decode, num_reqs]`).

A source detail this shard turned up that is not in `measurement_and_design.md`:
`decode_batch_end = (distribution[0] // decode_req_batch_size) * decode_req_batch_size` with
`decode_req_batch_size = 4`. So

- **pure-decode step**, `chunk_num_decode = 4`: `decode_batch_end = 4`. The batched call covers
  `[0, 4)`; the remainder call covers `[4, 4)` and is empty.
- **prefill-containing step**, `chunk_num_decode = 3`: `decode_batch_end = 3 // 4 * 4 = 0`. The
  batched call covers `[0, 0)` and is **empty**; all three decoding sequences fall to the
  **remainder** path at `seq_batch_size = 1`, which is the un-batched, one-sequence-at-a-time
  shape.

Either way exactly one decode-band call executes per step, and the MIXED call executes only in the
`M` prefill steps. So, as a fraction of executing band invocations [`modelled`]:

| band | invocations per N steps | share |
| --- | --- | --- |
| DECODE (batched or remainder) | `N` | **80.90%** |
| MIXED | `M = 0.236168 N` | **19.11%** |

`1 / 1.236168 = 0.80895`.

## And how much time

Invocation share is not time share, and here they are almost opposites. Per-step costs, using the
measured regimes as stand-ins [`assumed`, see the coverage list]:

- pure-decode step ≈ `decode_64k` = **0.20202844752930105 ms** `measured (j-b54e9716-3db5-4f1f-b8d4)`
  at `(2,2,2)`. (`decode_64k` is 6 sequences / 6 tokens; the A/B's pure-decode step is 4 / 4, so
  this over-states the decode side slightly, which is conservative for the conclusion.)
- prefill-containing step ≈ `mixed_64k` = **4.76284 ms** `measured (j-5f8f4a08-20b4-4fa0-bfc3)` at
  full kv.

The prefilling request's kv grows from 1021 to 65536 across its 64 chunks, so a mean over chunks
is below the full-kv figure. That gives a **bracket** rather than a point:

**HIGH end** — every prefill step costs the full `mixed_64k`, 4.76284 ms.

**LOW end** — split `mixed_64k` into selection (constant: the scores buffer is a fixed 528
sublanes wide whatever the prefiller's kv is) `3.00969` ms and scoring (∝ `num_bkv`) `1.75316` ms,
and scale the scoring by the mean chunk fraction. With `bkv_sz = 2048` and chunk `c` carrying
`1021c` kv:

```
num_bkv(c) = ceil(1021c / 2048),  c = 1..64
           = 1,1,2,2,3,3, ... ,32,32     (exactly two chunks per value)
    sum    = 2 · (1 + 2 + ... + 32) = 2 · 528 = 1056
    mean   = 1056 / 64 = 16.5   of 32   →  0.515625
```

```
LOW mean prefill step = 3.00969 + 0.515625 · 1.75316 = 3.00969 + 0.90397 = 3.91366 ms
```

⚠ **The LOW end rests on a decomposition `report.md` marks unusable.** `mixed_64k`'s
`scoring by difference` comes from a control arm with **133.1% spread**, and `report.md` says in
terms: *"The two long-KV `scoring by difference` figures are not usable."* It is used here only to
push the bracket's low end as far toward decode as the data allows — i.e. deliberately against the
conclusion. The bracket is what is quoted; neither end is a point estimate.

| end | prefill-step ms | × 0.236168 | decode-step ms | × 0.763832 | **w_decode** | **w_mixed** |
| --- | --- | --- | --- | --- | --- | --- |
| LOW  | 3.91366 | 0.924233 | 0.20202844752930105 | 0.154315 | **14.31%** | 85.69% |
| HIGH | 4.76284 | 1.124747 | 0.20202844752930105 | 0.154315 | **12.06%** | 87.94% |

## Answer

**DECODE is 80.9% of the kernel's invocations and 12.1–14.3% of its time. MIXED is 19.1% of
invocations and 85.7–87.9% of its time.** [all `modelled`] The bracket is narrow: it does not
matter which end you take.

The mechanism is not subtle. A prefill step scores 1024 rows and a pure-decode step scores 4;
selection runs over a padded row count in both. Three quarters of the steps do a two-hundredth of
the work.

## The ranking consequence, in §9.2's own words

> If mixed dominates the A/B's indexer time … then **E's net effect at the A/B is near zero or
> slightly negative and B is rank 1 outright, with A second.**

Mixed dominates at 85.7–87.9%. **So: B is rank 1 outright, A second, and E's net effect at the A/B
is near zero.** §9.1's `E ≈ B, then A, C, D` does not survive.

## E's sign, and which end of the bracket flips it

E is a trade, not a free win: `-11.69%` at `decode_64k` against `+0.97%` at `mixed_prod` for
`(4,4,4)` `measured (j-b54e9716-3db5-4f1f-b8d4)`, and variant E's own drift-controlled job
reports mixed regressions of `+2.02%` and `+2.63%` `measured (j-ce9a19a4-4639-4a8f-9059)`.

Net at the A/B is `w_decode · d + w_mixed · m`. E is net faster iff the mixed regression `m` stays
below a **breakeven** `m* = d · w_decode / w_mixed`:

| bracket end | `w_decode / w_mixed` | `m*` at `d = −11.69%` ((4,4,4)) | `m*` at `d = −13.96%` ((8,4,4)) |
| --- | --- | --- | --- |
| LOW  (w_decode 14.31%) | 0.166953 | **1.95%** | 2.33% |
| HIGH (w_decode 12.06%) | 0.137197 | **1.60%** | 1.92% |

Worked, for `(4,4,4)` at each end and each candidate mixed number:

| `w_decode` | `d` (decode_64k) | `m` (mixed) | `w_d·d` | `w_m·m` | **net Δ time %** |
| --- | --- | --- | --- | --- | --- |
| 14.31% | −11.69 | +0.97 `(j-b54e9716)` | −1.673 | +0.831 | **−0.84** (faster) |
| 12.06% | −11.69 | +0.97 `(j-b54e9716)` | −1.410 | +0.853 | **−0.56** (faster) |
| 14.31% | −11.69 | +2.02 `(j-ce9a19a4)` | −1.673 | +1.731 | **+0.06** (slower) |
| 12.06% | −11.69 | +2.02 `(j-ce9a19a4)` | −1.410 | +1.776 | **+0.37** (slower) |
| 14.31% | −11.69 | +2.63 `(j-ce9a19a4)` | −1.673 | +2.254 | **+0.58** (slower) |
| 12.06% | −11.69 | +2.63 `(j-ce9a19a4)` | −1.410 | +2.313 | **+0.90** (slower) |

**Neither end of the bracket flips E's sign.** Read down any column of fixed `m` and both ends
agree. What flips the sign is the **mixed regression crossing ~1.6–1.9%**, and the measured mixed
regressions straddle that line: `+0.97%` and `+1.61%` fall below it, `+1.67%` (`mixed_64k`,
`(4,4,4)`, `j-3aa755dd`) sits on it, and the two drift-controlled arms at `+2.02%` and `+2.63%`
fall above.

So the honest answer to "which end flips E's sign" is **neither** — the split is not the
uncertainty that matters any more, the mixed regression is. And the whole span is under one
percent of indexer time either way: **E's net at the A/B is within ±1% of zero**, and the
`streamindex_topk` kernel is itself a fraction of the model. E should be ranked on that, not on its
decode number.

## What this shard does not cover

Stated plainly because a derivation invites more confidence than it has earned:

1. **It is not an observation.** Nothing here ran the scheduler. An instrumented dispatch counting
   real `distribution` triples over a real serving run would replace it and should.
2. It assumes the prefill budget is **always filled** and that **exactly one** request prefills at
   a time. `M/N = 0.236` is the expected number of prefilling requests per step, so the mean-field
   is self-consistent, but steps with two prefillers exist and are not modelled.
3. Homogeneous requests at exactly ISL 65536 / OSL 1024, even DP routing to 4 per engine, no
   preemption, no prefix caching, steady state — the `t=0` burst where all four slots prefill at
   once is ignored.
4. The per-step costs are **regime stand-ins**, not the A/B's own shapes: `decode_64k` is 6/6 and
   the A/B's decode step is 4/4; `mixed_64k` is 6 sequences and the A/B's mixed step is 4.
5. The LOW end's scoring/selection split is `report.md`'s **unusable** 133.1%-spread control, used
   only to widen the bracket in the direction that would weaken the conclusion.
6. E's `(4,4,4)` decode number is from `decode_64k` and its mixed numbers from `mixed_prod`
   (kv 8192), not `mixed_64k` (kv 65536). Pairing across regimes is `assumed`.

---

# Shard 1 — `bkv_p = 8` correctness, and the `(8,4,4)` arm

## Correctness, and its negative control

Verbatim, `measured (j-b54e9716-3db5-4f1f-b8d4)`:

```
shipped NUM_KV_PAGES_PER_BLOCK = (2, 2, 2); mode=correct; regimes=['decode_64k', 'mixed_prod']
[decode_64k tuple=(8, 4, 4)] PASS -- all 6 rows a valid top-2048 vs the float64 oracle. worst_rank=2105 of allowed 2304, min_recall=0.99512 of required 0.99
[mixed_prod tuple=(8, 4, 4)] PASS -- all 1024 rows a valid top-2048 vs the float64 oracle. worst_rank=2098 of allowed 2304, min_recall=0.99512 of required 0.99

shipped NUM_KV_PAGES_PER_BLOCK = (2, 2, 2); mode=control; regimes=['decode_64k', 'mixed_prod']
[decode_64k tuple=(8, 4, 4)] NEGATIVE CONTROL OK -- fired on 6 of 6 rows. worst_rank=62644 of allowed 2304, min_recall=0.26953 of required 0.99
[mixed_prod tuple=(8, 4, 4)] NEGATIVE CONTROL OK -- fired on 1024 of 1024 rows. worst_rank=8159 of allowed 2304, min_recall=0.42578 of required 0.99
```

The controls fired on **every** row at both regimes, so neither PASS is vacuous. The margins are
bit-identical to `(4,2,2)`'s at the same regimes (`decode_64k` 2105 / 0.99512, `mixed_prod` 2098 /
0.99512, `j-3aa755dd`) — raising `bkv_p` changes the blocking, not the arithmetic.

**§10 item 3 — "`bkv_p = 8` is not value-checked" — is closed for the DECODE slot, at both a short
and a long regime.** `(8,4,4)` puts `8` in slot `[0]`, which is the slot both decode calls read.

**And it closes the coordinator's `(8,2,2)` at `decode_64k` too, by construction rather than by
running it.** At `decode_64k` the arm `(8,2,2)` and the arm `(8,4,4)` are the same computation:
slots `1` and `2` are read by no band that executes at a pure-decode regime, and
`num_sublanes_total` is already saturated by slot `0`
— `align_to(65, 8) · 1024 / 128 = 576` for both, against `528` for `align_to(65, 2)`. The sweep
below confirms it empirically from the other side: `(8,4,4)` and `(8,8,8)` at `decode_64k` differ
only in slots 1 and 2 and land at the same `subl=576`, the same `num_bkv=8`, and times
`0.17382489750161767` vs `0.17355122254230082` ms — 0.16% apart, inside both arms' spreads. **So
running `(8,2,2)` at `decode_64k` would re-measure an arm that already ran under another name.**

**What is still open** is `8` in the **MIXED** slot: only `(8,8,8)` puts `8` in slot `[2]`, and
`(8,8,8)` was never correctness-checked. It is also the worst timing arm measured, so nothing will
ship it — but the gap is real and should be recorded as a gap, not closed by inference.

## The sweep

`measured (j-b54e9716-3db5-4f1f-b8d4)`, median ms/call, 1 TensorCore, `reps=10`, all four arms and
both regimes on **the same chip in the same job**, so every `Δ time %` is against a baseline
measured beside it.

| regime | tuple | subl | scores MB | num_bkv | baseline ms `(2,2,2)` | arm ms | spread% | **Δ time %** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode_64k` | (2,2,2) | 528 | 1.62 | 32 | 0.20202844752930105 | 0.20202844752930105 | 0.9186871940980673 | 0.00 |
| `decode_64k` | (4,4,4) | 544 | 1.67 | 16 | 0.20202844752930105 | 0.17841560184024274 | 0.47114067610312343 | **−11.69** |
| `decode_64k` | (8,4,4) | 576 | 1.77 | 8 | 0.20202844752930105 | 0.17382489750161767 | 0.5127768946675958 | **−13.96** |
| `decode_64k` | (8,8,8) | 576 | 1.77 | 8 | 0.20202844752930105 | 0.17355122254230082 | 0.652739565935232 | **−14.10** |
| `mixed_prod` | (2,2,2) | 80 | 41.94 | 4 | 0.6724997947458178 | 0.6724997947458178 | 2.8311920625326144 | 0.00 |
| `mixed_prod` | (4,4,4) | 96 | 50.33 | 2 | 0.6724997947458178 | 0.6789957056753336 | 1.1587049713102968 | **+0.97** |
| `mixed_prod` | (8,4,4) | 128 | 67.11 | 2 | 0.6724997947458178 | 0.7057664450258017 | **491.23465258716567** ⚠ | **+4.95** ⚠ |
| `mixed_prod` | (8,8,8) | 128 | 67.11 | 1 | 0.6724997947458178 | 0.723214476602152 | 2.261826194518308 | **+7.54** |

### ⚠ The row whose spread swamps its effect

**`mixed_prod` / `(8,4,4)` carries a 491.23% spread against a 4.95% effect — a hundredfold. That
row is not a usable point estimate and must not be quoted as one.** The median is plausible only
because the two arms that bracket it are clean: `(4,4,4)` at `+0.97%` (1.16% spread) below and
`(8,8,8)` at `+7.54%` (2.26% spread) above. So the honest statement is **`(8,4,4)` costs
`mixed_prod` somewhere in `[+0.97%, +7.54%]`, and the run did not narrow it.** Every other arm in
this job is under 2.9% spread, so this is one arm's outlier, not a job-wide problem; re-running
that single arm would settle it.

### What the sweep says

- **`(8,4,4)` closes gap #4.** It is the best decode arm that is not `(8,8,8)`: `−13.96%` at
  `decode_64k`, a further `−2.6%` beyond `(4,4,4)`. Its mixed cost is the un-narrowed bracket
  above.
- **Nothing contradicts `measurement_and_design.md`.** `decode_64k` reproduces `j-3aa755dd`
  closely — `(4,2,2)` there read `−11.33%` and `(4,4,4)` here reads `−11.69%`; `(8,2,2)` there
  read `−13.79%` and `(8,4,4)` here reads `−13.96%`. `mixed_prod` regresses at every raised tuple
  in both jobs. This is now **eight** mixed arms across three jobs all on the slower side.
- **`(8,8,8)` is confirmed the worst mixed arm**, at `+7.54%` here against `+7.08%` read across
  jobs at `j-98399433`. §10 item 1's withdrawal (commit `11f26d75`) stands, and the number
  reproduces within half a point in a same-job comparison.
- **The coupling behaves exactly as §2.4 says.** `mixed_prod`'s `subl` climbs 80 → 96 → 128 as the
  *decode* slot is raised, and its scores buffer with it, 41.9 → 50.3 → 67.1 MB — a regime whose
  MIXED band never reads slot `[0]` still pays for it. There is no tuple that gives decode `8`
  without mixed paying the width.

### Ranking consequence

This shard does not move the ranking by itself; it feeds shard 0. Combined with shard 0's weights,
`(4,4,4)` is E's best clean arm and its net at the A/B is the ±1%-of-zero band computed above.
`(8,4,4)`'s extra `−2.27%` at decode buys, against `w_decode ≈ 13%`, about `−0.3%` of indexer time
— and costs an unbounded amount at mixed until that 491%-spread row is re-run. **E's ceiling is
not raised by `(8,4,4)`.**

---

# Shard 2 — the TensorCore-side selection strike test

**Question.** §9.4 item 1 / §6.3: option D moves top-2048-of-N selection off the SparseCore onto
the TensorCore, and that cost is `unmodelled`. Struck, D's `mixed_prod` goes `669.11 → 275.54` µs,
`−58.82`, and D is rank 1. **Priced above ~394 µs, D loses.**

`measured (j-b54e9716-3db5-4f1f-b8d4)`, 1024 rows, k = 2048, random-normal scores (not the
all-ties buffer `microbench.py` warns about):

| width | arm | µs | spread% | read MB | HBM read floor µs | **Δ time % vs `sc_topk`** |
| --- | --- | --- | --- | --- | --- | --- |
| 10240 | `sc_topk` (incumbent) | 541.7149513959885 | 4.2916676695400415 | 41.94 | 14.1 | 0.00 |
| 10240 | `xla_topk` | 4010.36546099931 | 0.5247994725618271 | 41.94 | 14.1 | **+640.31** |
| 10240 | `xla_approx` | 3530.2165197208524 | 0.6162939334947857 | 41.94 | 14.1 | **+551.67** |
| 10240 | `pallas_topk` | — | — | 41.94 | 14.1 | ERROR, see below |
| 10240 | `pass_count` (one pass) | 153.6225900053978 | 7.561033276548794 | 41.94 | 14.1 | −71.64 |
| 10240 | `pass_copy` **CONTROL** | 151.2804301455617 | 10.204934401137677 | 41.94 | 14.1 | not a finding |
| 67584 | `sc_topk` (incumbent) | 3071.8480702489614 | 0.5997390985415758 | 276.82 | 93.2 | 0.00 |
| 67584 | `xla_topk` | 46132.05697387457 | 0.9002578616886744 | 276.82 | 93.2 | **+1401.77** |
| 67584 | `xla_approx` | 41751.20301079005 | 0.8220003773532264 | 276.82 | 93.2 | **+1259.16** |
| 67584 | `pallas_topk` | — | — | 276.82 | 93.2 | ERROR, see below |
| 67584 | `pass_count` (one pass) | 232.30945225805044 | 6.888269391031825 | 276.82 | 93.2 | −92.44 |
| 67584 | `pass_copy` **CONTROL** | 188.74299712479115 | 5.704839973107834 | 276.82 | 93.2 | not a finding |

`W = 10240` is `mixed_prod`'s scores width (`num_sublanes_total` 80 × 128); `W = 67584` is the A/B
shape's (528 × 128). `pass_copy` is a **control arm** — same `BlockSpec`, so the same bytes are
DMA'd, with the compare and the reduce removed. Its absolute time is engineered to be subtracted
and is not a result.

## Answer: D loses, by an order of magnitude

The threshold is ~394 µs at `mixed_prod`, i.e. at `W = 10240`. The cheapest **complete** TC-side
top-2048 measured there is `xla_approx` at **3530.22 µs — 8.96× the threshold**. The exact one,
`xla_topk`, is 4010.37 µs — 10.2×. At the A/B's own width the gap widens to 106× and 117×
respectively against the same threshold, and to +1259% and +1402% against the SparseCore measured
beside them.

This survives the contention caveat with room to spare. `sc_topk` reads 35.6% slow in this job
against `j-5f8f4a08`'s 399.60 µs; deflating `xla_approx` by the same 1.356 gives **2604 µs, still
6.6× the threshold**.

## The reusable primitive, and why no cleverer scheme rescues D

The task asked for the primitive, not the variant, so here is the part that outlives option D.

| width | `pass_count` µs | `pass_copy` **CONTROL** µs | compare+reduce, isolated | HBM read floor µs |
| --- | --- | --- | --- | --- |
| 10240 | 153.6225900053978 | 151.2804301455617 | **2.34** | 14.1 |
| 67584 | 232.30945225805044 | 188.74299712479115 | **43.57** | 93.2 |

**The selection arithmetic is nearly free; the traffic is the whole cost.** One compare-and-count
pass over 1024 × 67584 f32 costs 232.31 µs, of which 188.74 is moving the bytes and 43.57 is the
compare and the row reduction. The pass is 2.49× its HBM read floor.

Any TC-side scheme that finds a top-2048 threshold by bisection pays that traffic **once per
pass**. A bisection over the 32 bits of a monotone int32 key is 32 passes:

- at `W = 10240`: `32 × 153.62 = 4915.9 µs` [`modelled` from a measured pass] — **12.5× the
  394 µs threshold**, and worse than `xla_topk`.
- at `W = 67584`: `32 × 232.31 = 7433.9 µs` [`modelled`] — 2.4× the SparseCore beside it.

Run the budget the other way: at `W = 10240` the 394 µs threshold buys **two** full passes
(307 µs) and no more. There is no top-2048-of-10240 in two passes. **The threshold is not
reachable by any multi-pass TC formulation on this hardware**, and that conclusion does not depend
on which top-k op you pick — it falls out of the per-pass traffic alone.

## ⚠ One arm errored, and it is a probe defect, not a hardware verdict

`pallas_topk` — `jax.lax.top_k` inside a `pallas_call` — failed identically at both widths. Full
error string, verbatim:

```
ValueError("The Pallas TPU lowering currently requires that the last two dimensions of your block
shape are divisible by 8 and 128 respectively, or be equal to the respective dimensions of the
overall array. Block spec for args[0] in pallas_call body at
/workspace/kernel_work/streamindex_topk/probes/tc_topk.py:181 has block shape
(Blocked(block_size=1), Blocked(block_size=10240)), array shape (1024, 10240), and index_map
{ lambda ; a:i32[]. let  in (a, 0:i32[]) }, in memory space <MemorySpace.DEFAULT: 'default'>.
See details at https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#pallas-blockspec")
```

(the `67584` instance is the same string with `block_size=67584` / `array shape (1024, 67584)`.)

**Read this correctly.** It is *not* "Mosaic cannot lower `top_k`", which would have been a hard
structural answer for D. It is my `BlockSpec`: I gave a block of `(1, W)` against an array of
`(1024, W)`, and the sublane dimension 1 is neither divisible by 8 nor equal to 1024. The probe is
wrong; the op was never reached. **Whether `jax.lax.top_k` lowers inside a `pallas_call` is
therefore UNRESOLVED, not UNSUPPORTED.** The one-line fix is a block of `(8, W)`.

Per the standing instruction not to retry a failed arm inside this job, it was not re-run. It also
would not move the verdict: the fused-Pallas formulation would have to beat 394 µs at `W = 10240`,
and the per-pass primitive above shows that a single unavoidable read of the scores is already
153.62 µs before any selection happens. Recording it as an open arm rather than as evidence.

## Ranking consequence, in §6.3's own terms

> Price it at anything above ~394 µs and D loses.

It prices at 3530 µs at best. **D loses. §6.3's strike does not fire, D's `−58.82` at `mixed_prod`
is withdrawn, and D stays last.** The `unmodelled` term that spanned the whole ranking is now
`measured` and it is not close.

---

# Shard 3 — the VPU-saturation strike test

**Question.** §7.4: is the MIXED band's inner step VPU-saturated? Target `269.51 µs ÷ 32 inner
iterations = 8.42 µs per inner step` `measured (j-5f8f4a08-20b4-4fa0-bfc3)`. Verdict rule, stated
in the probe's docstring before the run: **CONFIRMED if the non-MXU part lands near 8.4 µs;
REFUTED if it is below ~2 µs.**

Standing instruction honoured: **no spill-reduction candidate is opened on this result, whichever
way it lands.**

## Why not §7.4's literal two arms

§7.4 asks for a compile-time constant `s` (arm 1) and a slice epilogue (arm 2). A constant `s`
lets the compiler hoist the whole VPU chain out of the grid, so arm 1 reads ≈0 and the hypothesis
is "refuted" by dead-code elimination; a slice epilogue lets DCE delete 31 of the matmul's 32
head-rows, so arm 2 reads 1/32 of the MXU and the residual is "confirmed" by the same mechanism
pointing the other way. Instead: a **K-ladder** that cannot be folded, since MXU FLOPs scale with
`K` and nothing else in the step does. §7.4's literal arm 1 is kept as `gen=bcast`, a cross-check.

## Raw arms

`measured (j-b54e9716-3db5-4f1f-b8d4)`, µs per inner step, `grid=32`, shape
`(bq_sz=128, heads=32, bkv_sz=2048)`, fp8 operands.

| gen | epilogue | K | µs/step | spread% | GFLOP/step |
| --- | --- | --- | --- | --- | --- |
| mxu | full | 128 | 7.342914614127949 | **550.5565780901256** ⚠ | 2.147 |
| mxu | full | 256 | 7.218867722258437 | 3.3557807138990197 | 4.295 |
| mxu | full | 512 | 10.59588339558104 | 1.6239414325660537 | 8.590 |
| mxu | sum **CONTROL** | 128 | 4.815605279873125 | 2.9622192583157783 | 2.147 |
| mxu | sum **CONTROL** | 256 | 4.920359060633928 | 3.1664002460632106 | 4.295 |
| mxu | sum **CONTROL** | 512 | 7.688320692977868 | 0.8321150855549448 | 8.590 |
| bcast | full | 128 | 6.306484465312678 | 2.0401920781395817 | — |
| bcast | sum **CONTROL** | 128 | 2.598409992060624 | **40.88986209923618** ⚠ | — |

Every `sum` row is a **control arm**: the 32-head reduction alone, the minimal epilogue that still
consumes every row of the matmul so the MXU is not DCE'd. It exists to be subtracted and its
absolute time is not a finding.

## ⚠ Two rows whose spread is comparable to or larger than the effect

- **`mxu/full/K=128` at 550.56% spread.** This is the row `t(K_lo)` and therefore the row the
  published intercept is anchored on. Its median is not trustworthy.
- **`bcast/sum/K=128` at 40.89% spread.** The cross-check's subtrahend.

The probe's own decomposition, which uses `K ∈ {128, 512}`, is printed below for completeness, but
**it is not the reading this shard rests on**, precisely because of the first of those two rows:

```
epilogue     t(Klo) us   t(Khi) us     mxu us    rest us   mxu Tops/s
full             7.343      10.596      1.084      6.259       1980.5
sum              4.816       7.688      0.958      3.858       2242.6

relu + w-multiply + scale + mask, isolated = 6.259 - 3.858 = 2.401 us/step
head sum + out-store + fixed per-step      = 3.858 us/step
non-MXU total at K=128                     = 6.259 us/step
MXU at K=128                               = 1.084 us/step
```

(The `1980.5 Tops/s` is itself a flag that the two-point slope is not measuring what it claims —
it is at or above the fp8 MXU peak, which a real kernel does not reach.)

## The reading that does not depend on that row: `t(K)` is not linear

| epilogue | K=128 µs | K=256 µs | **Δ time % 128→256** | K=512 µs | **Δ time % 256→512** |
| --- | --- | --- | --- | --- | --- |
| full | 7.342914614127949 | 7.218867722258437 | **−1.69** | 10.59588339558104 | **+46.78** |
| sum **CONTROL** | 4.815605279873125 | 4.920359060633928 | **+2.18** | 7.688320692977868 | **+56.25** |

**Doubling the matmul's FLOPs from K=128 to K=256 changes the step time by −1.69% (full) and
+2.18% (control), both inside the arms' own spreads. Doubling again to K=512 costs +46.78% and
+56.25%.** The ladder is flat and then it is steep; the MXU crosses over between K=256 and K=512.

The `full` reading rests on the 550%-spread row, so take the **control** arm's `+2.18%`, whose two
rows are at 2.96% and 3.17% spread: **twice the matmul, +2.18% of step time.** At the real
`head_dim = 128`, the MXU is *not on the critical path* — it is hidden behind the rest of the step.
That is a within-shard ratio, so it is also the reading most robust to the fan-out's contention.

Consistency check with the steep leg: the marginal MXU cost on the 256→512 leg of the control is
`(7.688320692977868 − 4.920359060633928) / 256 = 0.010812 µs per unit K`, so a K=128 matmul is
`1.384 µs` of MXU work — comfortably under the `4.816 µs` the control step takes, which is exactly
the condition for it to hide. At K=512 the same figure is `5.54 µs`, now larger than the non-MXU
work, and the step duly jumps. The overlap model holds at all three rungs.

## Verdict

Applying the pre-stated rule: the non-MXU part of the step is **6.26 µs** by the published
intercept, and **≥7.22 µs** by the K=128→256 invariance (which says essentially *all* of the
7.22 µs step at K=256 is non-MXU). Against a target of 8.42 µs and a refutation bar of ≲2 µs:

**CONFIRMED. The MIXED band's inner step is not MXU-bound.** The MXU contributes at most ~1.1 µs
of a 7.34 µs step, and at the real K it contributes nothing measurable at all.

Fidelity check: the probe's whole step at K=128 is **7.343 µs** against the real kernel's
**8.42 µs/step** — the microbenchmark reproduces 87% of the real inner step. The missing ~1.08
µs/step is `unmodelled`.

Cross-check against §7.4's literal arm 1: `bcast/full` reads **6.306 µs/step** against the
ladder's `rest(full)` of **6.259 µs** — agreement to **0.75%**. So the broadcast was *not* hoisted,
and the two independent constructions of "the step without its matmul" land on the same number.
(The `bcast` *difference* of 3.708 µs disagrees with the ladder's 2.401 µs, but its subtrahend is
the 40.89%-spread control row; that comparison is not usable.)

## What is actually the dominant term — and it is not the ReLU

The decomposition inside the non-MXU part, from the `full` and `sum` intercepts:

| term | µs/step | share of the 7.343 µs step |
| --- | --- | --- |
| MXU at K=128 | 1.084 | 14.8% |
| relu + `w` multiply + scale + two masks + `where` | **2.401** | 32.7% |
| 32-head sum + output store + fixed per-grid-step | **3.858** | 52.5% |

The largest single term is not the elementwise chain. It is `rest(sum)` — and that bundle's output
store is small: 128 × 2048 × 4 = 1 MiB per step, which at the achieved 2.97 TB/s is ~0.35 µs. So
**the 32-way head reduction is the biggest thing in the step**, at roughly 3.5 µs, against 2.4 µs
for the entire relu/weight/mask chain and 1.1 µs for the matmul. The split of `rest(sum)` into
reduction, store and fixed per-step overhead is `unmodelled` — this probe measures the bundle, not
its parts.

## Ranking consequence, in §7.5's own terms

§7.5 re-ranks under the VPU hypothesis and puts **A1, "drop the ReLU", at the top**. The
hypothesis is confirmed, so that re-ranking is live — **but this shard says A1 is aimed at the
wrong third of the step.** The ReLU is one op inside a 2.401 µs bundle that is 32.7% of the inner
step; removing it removes some fraction of a third. The 52.5% sitting in the head reduction is
where the step's time actually is, and no option A through E targets it.

That is a finding for step 3's next wave, not a candidate opened here. Two things it is **not**:
it is not a spill result (no spill-reduction candidate is opened, per the standing instruction),
and the 624 µs op-slot model from area 5 remains `modelled` static analysis — this shard neither
confirms nor uses it.

---

# Open shards

Named so they are not mistaken for answered.

1. **`8` in the MIXED slot has never been value-checked.** Only `(8,8,8)` puts `8` in slot `[2]`,
   and only its *timing* was measured. Low priority: at `+7.54%` at `mixed_prod` it is the worst
   arm across three jobs and will not ship.
2. **`mixed_prod` / `(8,4,4)` needs one re-run.** 491.23% spread against a 4.95% effect. Its true
   cost is bracketed `[+0.97%, +7.54%]` by its neighbours and nothing narrows it. This is the only
   number blocking a clean price on E's best decode arm.
3. **`jax.lax.top_k` inside a `pallas_call` is UNRESOLVED, not UNSUPPORTED.** The arm died on my
   `BlockSpec` (`(1, W)` against a `(1024, W)` array), never reaching the op. Fix is a `(8, W)`
   block. It cannot change shard 2's verdict — a single unavoidable read of the scores is already
   153.62 µs against a 394 µs budget — but the feasibility question is still open on the record.
4. **Shard 0 is a derivation and wants an observation.** An instrumented dispatch counting real
   `distribution` triples over a serving run would replace `modelled` with `measured` on the number
   that now decides rank 1.
5. **The `rest(sum)` bundle wants splitting.** 3.858 µs/step is the largest term in the MIXED inner
   step and it is three things — head reduction, output store, fixed per-grid-step cost — measured
   as one. A ladder over `NUM_HEADS` would separate the reduction from the other two.
