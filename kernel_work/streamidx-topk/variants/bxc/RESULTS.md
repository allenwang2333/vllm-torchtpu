# B × C — do Option B (chunked pipeline) and Option C (head-major layout) compose at prefill?

**Sign convention for this whole file: signed microseconds, and `−` means *faster*.** Percentages
are `(cell − baseline) / baseline`, so a negative percentage is also faster. Two earlier documents
in this exercise use opposite conventions — `option-b-pipeline/RESULTS.md` writes `+16.4 %` for the
same effect this file writes as `−16.41 %` — so every number below is given in µs as well.

**Verdict: they compose, but sub-additively, and the shortfall is not noise — it is mechanical.**
B × C is a real win over either alone (`−26.18 %` at G = 4 against `−24.49 %` for B alone and
`−6.77 %` for C alone), but it delivers **84 %** of what adding the two standalone wins predicts.
The mechanism is that **C shortens stage A, stage A is the leg B hides, so the two options are
drawing on the same budget.** The sharpest measured form of that: **C's marginal value on top of B
scales as `Δt_a / G`**, at a ratio of 0.96 at G = 1, 2 and 4 alike (§5.2). The two options are not
fighting the scheduler; they are both spending stage-A time, and stage-A time can only be spent
once.

**Correctness: PASS.** All seven cells produce top-k index sets identical to the unchunked
incumbent, 0 of 1024 rows differing, job `j-caf973f4-163d-42e0-89e6`.

**The `num_bq` mechanism the brief flagged did not bite.** C's stage-A saving is flat in absolute
terms across G — `−50.94 / −50.12 / −51.25 µs` at G = 1 / 2 / 4 — because the *total* number of bq
blocks over the op is `q_len / bq_sz = 8` regardless of how the tokens are chunked. See §6.

---

## 0. The one job

| Job | What | Result |
| --- | --- | --- |
| `j-caf973f4-163d-42e0-89e6` | `plan`, `correct`, `perf` — everything in this file | `JobStatus: Succeeded`, `combined rc=0` |

Image (prebuilt, not rebuilt):
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b`

`RUN_CMD`, read back from `cdk job recipe j-caf973f4-163d-42e0-89e6 --no-color`:

```
python -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)" && bash /root/kw/variants/bxc/bxc_shard.sh all
```

Two `--map-dir` flags, both of which printed a `Packaging dir …` line:

```
--map-dir <checkout>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src
--map-dir <checkout>/kernel_work/streamidx-topk:tpu-job:/root/kw
```

The `src` map is **required**, not belt-and-braces: the image commit's
`streamindex_topk.py` has md5 `b0099c0a27ad014fc814a1f543b32313` and it ships **no**
`sparsecore_topk.py` at all. Read back on the pod, in each of the three arms:

```
MODULE /root/cloud-devkit/vllm-torchtpu/src/.../streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532
MODULE /root/cloud-devkit/vllm-torchtpu/src/.../sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658
```

Both are the checkout's. Nothing under `src/vllm_torchtpu/` or `instruments/` was modified.

**One unpinned process on the whole 4-chip pod. No fan-out**, so no cross-shard contention on the
wall clocks the chain slopes are fitted to. Another agent was running a job elsewhere on the
cluster; that job was neither reused, inspected nor deleted.

---

## 1. What was measured, and how

### 1.1 The harness

`variants/bxc/pipeline_c.py` is deliberately thin. Everything that defines Option B's experiment —
geometry, chunk plan, the pseudo-sequence presentation, the two guards, the stage-B wrapper, the
two unfoldable chain carriers, and the shape of the four timing arms — is **imported from**
`variants/option-b-pipeline/pipeline.py`, not re-typed. `build_body` is a copy of
`pipeline.build_body` with exactly one edit: `sa` comes from Option C's launcher instead of
Option B's.

| | Option B alone | B × C |
| --- | --- | --- |
| stage-A launcher | `probes/_stage_a.py::make_stage_a_fn` | `variants/option-c-head-major/stage_a_c.py::make_stage_a_fn(arm)` |
| kernel body | `src/…/streamindex_topk.py::kernel` | `streamindex_topk_c.py::kernel` with `q_layout=<arm>` |

`stage_a_c.make_stage_a_fn` is Option C's own transcription of the same `run_scores_kernel`
closure that `_stage_a.py` transcribes. The two differ in exactly two neutral places: the extra
static `q_layout=` kwarg (suppressed entirely for the `incumbent` arm, which launches `src`'s
kernel and produces the identical `scope_name`), and an explicit `max_num_tokens=` in place of
`q.shape[0]` — every chunk at a given G has the same token count, so this module passes
`plan[0]["num_tokens"]` and the two are the same number.

For `q_layout="head_major"` the transpose is **in registers inside the kernel**; only
`head_major_hbm` (which does not compile, and is not used here) touches the HBM layout. So
`pipeline.build_inputs` / `chunk_arrays` are reused unchanged and both q layouts read the same
HBM bytes.

**What I changed to make the two harnesses meet:** nothing in either. The whole adaptation is the
new `pipeline_c.py`; no file under `option-b-pipeline/`, `option-c-head-major/`, `probes/`,
`instruments/` or `src/` was edited. `probe_bxc.py` imports `chain_slope` and `linfit` **from
`probe_flip3.py`** rather than re-implementing the fit.

### 1.2 Timing method — unchanged from Flip 3

Chain slope: `R ∈ {1,2,3,4}` chained copies of the whole G-chunk program inside **one** `jax.jit`,
best-of-9 per point, least squares on `t(R)`, fits below `r² = 0.999` rejected. The slope is
dispatch-free device time and the host term lands in the intercept. A per-call wall clock is
`max(device, host)`, so the host dispatch floor is a clamp and **nothing is subtracted anywhere**.

**Every one of the 25 fits was accepted. The lowest `r²` is `0.9998967` (`head_major/a_G2`); 20 of
25 are ≥ 0.99999.** The five below that bar are `head_minor/a_G2` (0.9999633), `head_minor/a_G4`
(0.9999576), `head_major/a_G2` (0.9998967), `head_major/a_G4` (0.9999265) and `head_major/pipe_G2`
(0.9999856) — the stage-A arms are the noisier ones. Full table in §7.

### 1.3 The grid

`incumbent` at G = 1 is a **drift control on Option C's verbatim copy**, not a row of the grid: it
launches the unmodified `src` kernel through Option C's wrapper. It agrees with `head_minor` to
`0.0016 %` on stage A (263.122 vs 263.118 µs) and `0.15 %` on the pipeline (729.39 vs 728.30 µs),
so the copy has not drifted and the wrapper is neutral.

`head_minor` also reproduces Option B's own Flip-3 prefill numbers, measured on a different day in
a different (fanned-out) job. **All nine comparisons, not a selection:**

| | Flip 3 (`j-c397124b`) | here (`j-caf973f4`) | Δ |
| --- | --- | --- | --- |
| `t_a` G1 | 263.32 | 263.12 | −0.076 % |
| `t_a` G2 | 272.99 | 272.29 | −0.256 % |
| `t_a` G4 | 288.09 | 288.30 | +0.073 % |
| `t_b` G1 | 440.77 | 441.25 | +0.109 % |
| `t_b` G2 | 449.28 | 449.36 | +0.018 % |
| `t_b` G4 | 466.34 | 464.32 | **−0.433 %** |
| `t_pipe` G1 | 727.67 | 728.30 | +0.087 % |
| `t_pipe` G2 | 608.34 | 608.78 | +0.072 % |
| `t_pipe` G4 | 549.84 | 549.92 | +0.015 % |

The worst disagreement is **0.433 %** (2.02 µs, `t_b` at G = 4); the median is 0.073 %. An earlier
draft of this file quoted "≤ 0.1 %" from the four `t_pipe`/`t_a`-G1 rows alone, which was a
selection. Take 0.43 % as the cross-job reproducibility of a single slope here.

### 1.4 Geometry, read back in `--arm plan`

1 sequence, `q_len` 1024, `context_len` 9216, `max_model_len` 9216, page 1024, `kv_packing` 4,
`lkv_dim` 256, H 32, D 128, `k` 2048, `bkv_p` 2, `bq_sz` 128, `seq_batch_size` 1, case `MIXED`,
`num_positions` 10240, `num_bkv` 5.

| G | tokens/chunk | bq blocks per chunk | pseudo-sequence `seq_lens` | `waves_guard` | `_pick_partition` |
| --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | `[9216]` | PASS (32×32 = 1024) | 1 |
| 2 | 512 | 4 | `[8704, 9216]` | PASS (16×32 = 512) | 1 |
| 4 | 256 | 2 | `[8448, 8704, 8960, 9216]` | PASS (8×32 = 256) | 1 |

The prefill chunk is presented to stage A as G pseudo-sequences whose `seq_lens` advance by
`q_len/G`, per Option B §7 — semantics-preserving, and §3 below shows it produces the incumbent's
exact indices. `force_p1=True` on every arm and the `_pick_partition` tripwire was armed on every
one; it never fired. `_pick_partition` returns 1 at every G anyway, so this changes nothing at
prefill. **No runner change was made and `mla_attention_op.py` was not touched.**

---

## 2. The 2 × 3 grid

Device time is the `pipe` arm's chain slope — the whole op, stage A feeding stage B, at that
(G, layout). Baseline is `head_minor` G = 1, **728.295 µs**.

| | `head_minor` | `head_major` |
| --- | --- | --- |
| **G = 1** (unchunked) | **728.30 µs** — baseline, `+0.00 µs` / `+0.00 %` | **678.98 µs** — C alone, **`−49.32 µs` / `−6.77 %`** |
| **G = 2** | **608.78 µs** — B alone, **`−119.51 µs` / `−16.41 %`** | **584.76 µs** — B × C, **`−143.54 µs` / `−19.71 %`** |
| **G = 4** | **549.92 µs** — B alone, **`−178.37 µs` / `−24.49 %`** | **537.60 µs** — B × C, **`−190.70 µs` / `−26.18 %`** |

The best cell is B × C at G = 4: **537.60 µs, `−190.70 µs` / `−26.18 %` against the baseline.**

Two readings of the same table. They are **not independent evidence** — each is algebraically the
§5 additivity miss with a sign, and they reproduce it bit for bit, so read them as two views of one
quantity rather than as two confirmations of it:

- **What C adds on top of B**, cell minus the `head_minor` cell at the same G:
  `−49.32 µs` at G = 1, `−24.02 µs` at G = 2, `−12.32 µs` at G = 4.
  C's marginal value **falls by 4× as G goes 1 → 4.**
- **What B adds on top of C**, cell minus the `head_major` cell at G = 1:
  `−94.22 µs` at G = 2, `−141.38 µs` at G = 4 — against `−119.51` / `−178.37 µs` for B on its own.
  B's prize shrinks by `+25.29` / `+37.00 µs`, which is the same lost overlap seen from the other
  side: `ΔC − (C on top of B) = ΔB − (B on top of C) = miss`, identically.

Both bullets are in µs only, deliberately. Percentages here would be against **different
denominators** — C-on-top-of-B at G = 2 is 3.95 % of the `head_minor` G = 2 cell (608.78) but
2.24 % at G = 4 is against 549.92, while the grid's percentages are all against the 728.295 µs
baseline — and the three are not comparable to each other.

---

## 3. Correctness — PASS

**Reference: `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py::streamindex_topk`, called
on the whole unchunked op** — one sequence, 1024 query tokens, ctx 9216, `distribution=[0,0,1]`,
`enable_early_exit=False`. That is the incumbent as production calls it: three stage-A launches
(two of them empty at this shape) followed by one `sparsecore_topk`. Output `i32[1024, 2048]`,
index range `[0, 9212]`.

Comparison is per-row **index sets**, ignoring order and `-1` padding, because the kernel documents
its output as unsorted. No recall-vs-f64 gate is used anywhere.

| cell | rows | rows differing | verdict |
| --- | --- | --- | --- |
| `incumbent` G = 1 | 1024 | **0** | identical |
| `head_minor` G = 1 | 1024 | **0** | identical |
| `head_minor` G = 2 | 1024 | **0** | identical |
| `head_minor` G = 4 | 1024 | **0** | identical |
| `head_major` G = 1 (C alone) | 1024 | **0** | identical |
| **`head_major` G = 2 (B × C)** | 1024 | **0** | **identical** |
| **`head_major` G = 4 (B × C)** | 1024 | **0** | **identical** |

`OVERALL: PASS`. There were no mismatches, so the tie-adjudication path (which quotes the f32
scores at every reference-only and candidate-only index) never fired and no scores are quoted.

**The scores are *not* bitwise identical, and the indices are.** This is worth stating plainly
rather than leaving it implied by a green result. Comparing `head_minor` and `head_major` stage-A
score buffers element by element, at every G:

| | value |
| --- | --- |
| elements compared | 10 485 760 (1024 × 10240) |
| elements differing | **36 973** (0.353 %) |
| max abs difference | **6.104 × 10⁻⁵** |
| `isfinite` pattern identical | **yes** |
| bitwise identical | no |

Identical at all three G, to the element. That is the expected signature of a different f32
reduction order — C replaces a `vrot.slane` tree over adjacent sublanes with a vreg add chain — and
it is small enough, at k = 2048 out of 8193–9216 live positions, that it moved no selection
boundary in any of the 7 168 rows checked. It is not zero, so a shape where two positions sit
within 6 × 10⁻⁵ of the k-th score could in principle flip; that would be a tie broken differently,
not an error, but it has not been observed here.

---

## 4. Arm breakdown — where the shortfall lives

`hidden = t_a + t_b − t_pipe`; `ideal = min(t_a, t_b)/G × (G−1)`; `overlap = hidden / ideal`. All
figures are chain slopes in µs.

### `head_minor` (reproduces Option B §3)

| G | `t_a` | `t_b` | `t_a+t_b` | `t_pipe` | `hidden` | `ideal` | **overlap** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 263.12 | 441.25 | 704.37 | 728.30 | −23.93 | 0 | n/a |
| 2 | 272.29 | 449.36 | 721.65 | 608.78 | 112.87 | 136.15 | **82.90 %** |
| 4 | 288.30 | 464.32 | 752.62 | 549.92 | 202.70 | 216.22 | **93.75 %** |

### `head_major` — the row the brief asked for

| G | `t_a` | `t_b` | `t_a+t_b` | `t_pipe` | `hidden` | `ideal` | **overlap** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 212.18 | 439.90 | 652.08 | 678.98 | −26.90 | 0 | n/a |
| 2 | 222.17 | 450.33 | 672.50 | 584.76 | 87.74 | 111.08 | **78.99 %** |
| 4 | 237.05 | 464.47 | 701.52 | 537.60 | 163.92 | 177.79 | **92.20 %** |

Three things to read off these.

1. **`t_b` is very nearly untouched by the layout**, as expected — stage B is a SparseCore radix
   select on a score matrix whose *values* barely moved (§3). There are three differences, not
   four: `−0.306 %` at G = 1 (441.25 → 439.90), `+0.216 %` at G = 2, `+0.032 %` at G = 4. All are
   ≤ 0.31 %, none is systematic in sign, and all are inside the 0.43 % cross-job reproducibility of
   §1.3. "Untouched" is right to ~0.3 % and no better: the 1.355 µs `t_b` gap at G = 1 turns out to
   carry 45.6 % of §5's `−2.98 µs` serial-residual term, so it is small but not negligible in the
   decomposition.
2. **The overlap *fraction* barely changes** — 82.90 → 78.99 % at G = 2, 93.75 → 92.20 % at G = 4.
   B still works under C. What shrinks is not the efficiency, it is the window.
3. **`hidden` falls by almost exactly the amount `ideal` falls.**

| G | `ideal` lost to C *(algebraic)* | `hidden` lost to C *(measured)* | measured additivity miss |
| --- | --- | --- | --- |
| 2 | 136.15 − 111.08 = **25.06** | 112.87 − 87.74 = **25.12** | **25.29** |
| 4 | 216.22 − 177.79 = **38.43** | 202.70 − 163.92 = **38.78** | **37.00** |

**The first column is a tautology and carries no evidential weight.** Since `t_a < t_b` in both
layouts at every G, `ideal ≡ t_a/G × (G−1)`, so `ideal` lost to C is *by definition*
`(G−1)/G × Δt_a` — 0.5 × 50.12 and 0.75 × 51.25, reproducing to < 1e−9. It restates the stage-A
measurement; it does not corroborate it. What is a real, non-trivial finding is that the
**second** column (`hidden`, which involves `t_pipe` and could have come out anywhere) lands within
0.06 µs and 0.35 µs of it — that is the hardware agreeing that the overlap window really did shrink
by the full amount C removed from stage A. The third column is the miss it is meant to explain, and
the fit there is looser: 0.9 % at G = 2 but **3.9 % at G = 4**. §5 accounts for the gap.

### `b_chained` — SparseCore still does not overlap itself

The control that would invalidate the whole overlap reading if it moved:

| | `b` | `b_chained` | Δ |
| --- | --- | --- | --- |
| `head_minor` G = 2 | 449.36 | 450.63 | +0.28 % |
| `head_minor` G = 4 | 464.32 | 467.07 | +0.59 % |
| `head_major` G = 2 | 450.33 | 451.51 | +0.26 % |
| `head_major` G = 4 | 464.47 | 466.56 | +0.45 % |

Chaining the G stage-B calls to each other costs nothing and saves nothing, in both layouts. All of
the measured saving is genuine TensorCore/SparseCore concurrency; none of it is stage B pipelining
against stage B.

---

## 5. Does B × C compose additively? **No — it is sub-additive, by 25–37 µs.**

Taking deltas against the G = 1 / `head_minor` baseline of 728.295 µs:

| G | `ΔB` (B alone) | `ΔC` (C alone) | additive prediction | **`ΔB×C` measured** | **miss** | fraction realised |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | −119.51 | −49.32 | −168.83 | **−143.54** | **+25.29 µs** (+3.47 pp of baseline) | **85.0 %** |
| 4 | −178.37 | −49.32 | −227.69 | **−190.70** | **+37.00 µs** (+5.08 pp of baseline) | **83.8 %** |

**Direction: B × C is *slower* than additive.** It realises 85.0 % of the predicted saving at
G = 2 and 83.8 % at G = 4. In percentage terms the prediction is `−23.18 %` / `−31.26 %` and the
measurement is `−19.71 %` / `−26.18 %`.

### 5.1 Why — the mechanism

B's ceiling is `ideal(G) = min(t_a, t_b)/G × (G−1)`, and at prefill `t_a < t_b` at every G, so the
ceiling is set by **`t_a`**. C's entire prize is a reduction in `t_a`. So C lowers the ceiling of
B's prize by `(G−1)/G × Δt_a` before either has done anything wrong:

```
(G−1)/G × Δt_a   at G=2:  0.5 × 50.12 = 25.06 µs      measured hidden lost: 25.12 µs
                 at G=4:  0.75 × 51.25 = 38.43 µs     measured hidden lost: 38.78 µs
```

Two cautions on how much that proves.

**The left-hand side is a definition, not a prediction.** `(G−1)/G × Δt_a` *is* `Δideal` whenever
`t_a < t_b`, exactly (verified to < 1e−9), so writing them next to each other is arithmetic. The
content is on the right: `hidden` is a `t_pipe`-dependent measurement that was free to land
anywhere, and it lands 0.24 % / 0.91 % away.

**And this quantity is close to the miss, not equal to it.** Against the thing it is offered to
explain — the +25.29 / +37.00 µs additivity miss — the agreement is **0.92 % at G = 2 but 3.88 % at
G = 4** (2.12 % / 3.30 % if `median_us` is used instead of the slope). The earlier "within 1 %" in
this file compared against `hidden_lost` rather than against the miss and so understated the G = 4
error by 4×. The decomposition below is where the remaining ~1.4 µs at G = 4 goes.

**The decomposition.** This is an exact algebraic rearrangement of `miss = ΔB×C − (ΔB + ΔC)` under
`hidden ≡ t_a + t_b − t_pipe` — it holds identically, residual `0.00e+00`, for *any* numbers you
put in it. It is a way of apportioning the miss, **not** a hardware result:

```
miss(G) = [ hidden(G,minor) − hidden(G,major) ]                # overlap C removed
        − [ hidden(1,minor) − hidden(1,major) ]                # −2.98 µs, the G=1 serial residual
        + [ S(1) − S(G) ]                                      # drift in C's serial saving
```

where `S(G) = (t_a+t_b)(G,minor) − (t_a+t_b)(G,major)` is C's saving on the serial work
(52.29 / 49.15 / 51.10 µs at G = 1 / 2 / 4). Apportioned:

| G | term 1 (overlap C removed) | term 2 (G = 1 residual) | term 3 (serial drift) | total | miss |
| --- | --- | --- | --- | --- | --- |
| 2 | +25.12 (**99.3 %**) | −2.98 (**−11.8 %**) | +3.14 (**+12.4 %**) | 25.28 | 25.29 |
| 4 | +38.78 (**104.8 %**) | −2.98 (**−8.0 %**) | +1.19 (**+3.2 %**) | 36.99 | 37.00 |

An earlier draft called terms 2 and 3 "rounding". They are not: they are 8–12 % of the miss each
and at G = 2 they very nearly cancel, which is the only reason term 1 alone looks exact there. Term
2 is also not purely a stage-A effect — **45.6 % of that −2.98 µs is the 1.355 µs `t_b` gap** that
§4 calls untouched. The honest summary is that term 1 dominates and the other two are real but
partly cancelling.

### 5.2 The sharper statement: C's marginal value scales as `Δt_a / G`

The cleanest form of the mechanism, and the one that predicts rather than restates:

| G | C's marginal value on top of B (measured) | `Δt_a / G` | ratio |
| --- | --- | --- | --- |
| 1 | −49.32 | −50.94 | **0.968** |
| 2 | −24.02 | −25.06 | **0.959** |
| 4 | −12.32 | −12.81 | **0.962** |

C returns a constant ~96 % of `Δt_a / G` at all three G. `Δt_a` is flat (§6), so **C's marginal
value halves every time G doubles.** The 4 % shortfall is stable and is the serial residual of
§5.1. This is one relationship over three points, not a law, but it is the form to carry forward:
extrapolated, C would be worth ~6 µs on top of B at G = 8.

**So the collision is real and it is structural, not an interaction bug.** The two options are not
competing for a resource or fighting the scheduler — they are *both spending stage-A time*, and
stage-A time can only be spent once. C spends it by not doing the work; B spends it by hiding the
work behind SparseCore. Whichever runs first, the second one finds less left.

**It is still worth doing both.** C on top of B at G = 4 is `−12.32 µs`, a further 2.24 % on an op
that is already 24.5 % faster, and it costs nothing but a static kernel parameter. But **the two
wins must not be quoted added together**: `−24.49 % + −6.77 %` is not `−31.3 %`, it is `−26.18 %`.

**One reading this does *not* support.** The brief's other candidate mechanism — that C would get
*worse* under chunking — is not what happened. C's contribution to stage A never degrades (§6);
what degrades is how much of that contribution survives the overlap. If B were ever *removed* from
a build that had both, C would immediately be worth its full `−49.32 µs` again.

---

## 6. Does C's win degrade as G rises? **No — the `num_bq` mechanism did not bite.**

Measured on the stage-A-only arm (`a`), which is exactly the quantity Option C's own document
reports:

| G | tokens/chunk | **bq blocks per chunk** | `t_a` `head_minor` | `t_a` `head_major` | Δ | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | **8** | 263.12 | 212.18 | **−50.94 µs** | −19.36 % |
| 2 | 512 | **4** | 272.29 | 222.17 | **−50.12 µs** | −18.41 % |
| 4 | 256 | **2** | 288.30 | 237.05 | **−51.25 µs** | −17.78 % |

**In absolute µs the saving is flat**: −50.94, −50.12, −51.25, a total spread of 1.13 µs (2.2 % of
the saving).

The argument that this is flat rather than declining rests on **shape, not on a noise bound**. The
sequence is **non-monotonic** — it dips at G = 2 and comes back up at G = 4, so the largest saving
of the three is at the largest G — and it is non-monotonic under all four estimators the raw data
supports (slope, `median_us`, `min_us`, and a refit dropping R = 1). The `num_bq` mechanism predicts
a *monotone decline* as blocks per chunk go 8 → 4 → 2. A monotone prediction cannot be rescued by a
non-monotonic measurement whichever way the error bars fall, so the noise size does not need to be
settled to reject it. (For the record, 2.2 % is a spread of a *difference* of two slopes and is not
comparable to the 0.15 % agreement of a single slope in §1.3 — an earlier draft made that
comparison and it was wrong.)

**External corroboration**, from a different job on a different harness:
`variants/option-c-head-major/RESULTS.md` measures C's unchunked stage-A saving at **50.89 µs**,
against the 50.94 / 50.12 / 51.25 measured here. Three independent numbers within 0.7 % of each
other, one of them from a job this file did not run.

The percentage drifts from −19.36 % to −17.78 % **only because the denominator grows** — chunking
inflates stage A itself (263.12 → 272.29 → 288.30 µs, Option B §4's launch-count and
re-DMA cost, which is a property of B and is present in the `head_minor` column too). The numerator
does not move.

**Why the feared mechanism does not apply.** The concern was that C's transpose is paid once per bq
block, and G = 4 leaves only 2 bq blocks per chunk. But the transpose is amortised over the **bkv
loop inside** each bq block, not over the number of bq blocks: `num_bkv = ceil(9216 / 2048) = 5`,
and chunking does not change it — every chunk still sweeps the whole 9216-token context. Nor does
chunking change the *total* count of transposes over the op: `G × num_bq_per_chunk = 8` at every G,
because it is just `q_len / bq_sz`. The transpose:compute ratio C was measured at is 8 shuffles to
40 `(bq, bkv)` pairs, and it is **8 : 40 at G = 1, G = 2 and G = 4 alike.** C's amortisation was
measured over `num_bkv` because `num_bkv` is the axis it actually amortises over; the brief's worry
substituted `num_bq`, and `num_bq` is not that axis.

This also means the result should extrapolate: C's stage-A saving should stay ~51 µs for any G that
keeps `q_len/G ≥ bq_sz = 128`, i.e. up to G = 8 at this shape. Not measured beyond G = 4.

---

## 7. Every fit

All 25 accepted, all `r² ≥ 0.9998967`. Slope = device µs per copy; the host term is in the
intercept and is not subtracted from anything.

| key | slope µs | `r²` | intercept µs |
| --- | --- | --- | --- |
| `incumbent/a_G1` | 263.12 | 0.9999975 | 6.73 |
| `incumbent/b_G1` | 441.33 | 0.9999991 | 78.95 |
| `incumbent/pipe_G1` | 729.39 | 0.9999990 | 20.06 |
| `head_minor/a_G1` | 263.12 | 0.9999960 | 6.28 |
| `head_minor/b_G1` | 441.25 | 0.9999976 | 79.53 |
| `head_minor/pipe_G1` | 728.30 | 0.9999985 | 22.58 |
| `head_minor/a_G2` | 272.29 | 0.9999633 | 7.69 |
| `head_minor/b_G2` | 449.36 | 0.9999941 | 38.53 |
| `head_minor/b_chained_G2` | 450.63 | 0.9999987 | 38.46 |
| `head_minor/pipe_G2` | 608.78 | 0.9999953 | 12.89 |
| `head_minor/a_G4` | 288.30 | 0.9999576 | 17.17 |
| `head_minor/b_G4` | 464.32 | 0.9999993 | 30.64 |
| `head_minor/b_chained_G4` | 467.07 | 0.9999984 | 26.97 |
| `head_minor/pipe_G4` | 549.92 | 0.9999954 | 19.00 |
| `head_major/a_G1` | 212.18 | 0.9999999 | 7.76 |
| `head_major/b_G1` | 439.90 | 0.9999989 | 80.52 |
| `head_major/pipe_G1` | 678.98 | 0.9999992 | 20.48 |
| `head_major/a_G2` | 222.17 | **0.9998967** | 6.22 |
| `head_major/b_G2` | 450.33 | 0.9999945 | 36.57 |
| `head_major/b_chained_G2` | 451.51 | 0.9999986 | 37.30 |
| `head_major/pipe_G2` | 584.76 | 0.9999856 | 10.91 |
| `head_major/a_G4` | 237.05 | 0.9999265 | 18.17 |
| `head_major/b_G4` | 464.47 | 0.9999995 | 30.36 |
| `head_major/b_chained_G4` | 466.56 | 0.9999978 | 28.20 |
| `head_major/pipe_G4` | 537.60 | 0.9999943 | 17.38 |

---

## 8. Files

| file | what |
| --- | --- |
| `pipeline_c.py` | Option B's `pipeline.py` with Option C's stage-A launcher swapped in; plus the unchunked-incumbent reference call |
| `probe_bxc.py` | the three arms — `plan`, `correct`, `perf`. Imports `chain_slope` from `probe_flip3.py` |
| `bxc_shard.sh` | job entry point; reads back the kernel md5s |
| `data/bxc-perf.shardx.json` | every slope, intercept, `r²` and raw `t(R)` point behind §2, §4, §5, §6, §7 |
| `data/bxc-correct.shardx.json` | §3's per-cell index-set verdicts and the score-difference census |
| `data/bxc-plan.shardx.json` | §1.4's geometry, guards and `_pick_partition` read-backs |

## 9. What was not done

- **No end-to-end model benchmark.** Out of scope by instruction; everything here is a
  microbenchmark on one 4-chip pod.
- **Nothing was logged to the shared experiment log.** Out of scope by instruction.
- `src/vllm_torchtpu/`, `instruments/`, `option-b-pipeline/` and `option-c-head-major/` are all
  unmodified. `mla_attention_op.py` was not touched and **no runner change was made** — the
  prefill split is still the pseudo-sequence presentation, which Option B §7 records as needing a
  runner change to ship.
- **Decode was not measured.** Both options are prefill options and the question was about prefill.
- **`head_major_hbm` was not attempted** — it does not compile.
- **G > 4 was not measured**, so §6's extrapolation to G = 8 is an inference, not a result.
- No `hlo_stats` cross-check of the chain slopes was taken. The slopes reproduce Flip 3's to
  **≤ 0.433 %** across all nine comparisons (§1.3) and the `incumbent` control agrees with
  `head_minor` to 0.15 % (§1.3), which is the drift evidence this file rests on instead.
- **Each cell was compiled once.** The 9 trials per `(cell, R)` point resample execution, not
  compilation, so a per-compile layout or scheduling artifact would be inside every trial of that
  cell and invisible to the `r²`. The Flip-3 and `option-c-head-major` cross-job agreements above
  are what bound this, at ~0.4–0.7 %.
- **The empirical floor on the `pipe` arm is 1.10 µs** (largest residual across the 4 fitted
  points, any pipe cell). Differences of a few µs — term 2 and term 3 of §5.1 — are only a few
  times that, which is the second reason not to over-read the decomposition.
- **Status: `provisional`, not `accepted`.** Correctness is established for this shape (§3) and the
  performance is a microbenchmark, not an end-to-end result.
