# `streamindex_topk`: one composed shipping candidate

**Job `j-74f9b184-8cf7-4db2-9907`, `JobStatus = Succeeded`, 4/4 shards `rc=0`, 5m09s.**
Image `vllm-torchtpu-annyan_google_com:13617694`. Branch `annyan/streamidx-scores`,
candidate at commit `4dc06a33`. Every arm of every comparison below ran in the **same
process on the same chip**, timed by the read-only `instruments/microbench.py`.

Sign convention, binding on every table: **`Δ time %` = `(arm − incumbent)/incumbent × 100`,
so negative is faster.** Each delta sits in the same row as the two absolute times it came
from. `speedup ×` (>1 is faster) is given where it is the more readable form. Every number
is tagged `measured (job)`, `modelled`, `assumed` or `unmodelled`.

---

## 1. What the candidate is

`kernel_work/streamindex_topk/final/streamindex_topk_composed.py`, built by
`final/compose.py` from Option B's file plus Option A's twenty hunks. Three changes:

| | Change | Where |
| --- | --- | --- |
| **B** | Size the scores buffer from a static bucket instead of `max_model_len`; delete the `-1`/`-inf` fill | the kernel file |
| **A** | Split the packed KV record into a value plane and a `u8[num_pages, page_size//128, 128]` scale plane | the kernel file + `mla_attention_op.py` writer |
| **E** | `num_kv_pages_per_block` `(2,2,2)` → `(4,2,4)` | `mla_attention_op.py` only — **not** a kernel-file change |

A and B are not independent in two places, both encoding the operand index of
`scores_init`: B's `None`-splice `n_before_scores_in` and B's conditional
`input_output_aliases` both move `+3 → +4` because A inserts `cache_scale` ahead of it.
`compose.py` asserts an exact match count per hunk, so a silently half-applied variant
cannot be measured as a composition.

The composed module docstring also discharges the §6.5 **layout opt-out** obligation: this
kernel does dynamic-offset DMA, in-kernel metadata construction and selective triple
buffering, so it takes the sanctioned opt-out explicitly instead of continuing to half-use
the layout sheet in silence.

### The bucket is a constant, not a per-request value — and this is new

Step 3 measured B with the driver passing the **exact `kv_len`**. That is not reachable.
`kv_len_bucket` must be a static Python `int`; vLLM's backbone is Dynamo-traced **once**
with all guards dropped (`vllm/compilation/wrapper.py:104-122`; this repo's
`compilation/shape_variants.py:1-31` says the same in its own words), so a Python int
reaching the custom op is **frozen at its first-traced value**. The warmed ladder that does
exist keys on token count, not kv length. `pallas.jax_op` itself is fine — it infers
`static_argnums` from annotations (`pallas.py:382-403`, note keyword-only params are
skipped) and recompiles per distinct value — the blocker is above it.

So the candidate ships **`bucket = min(max_model_len, MAX_SLICE_WORDS) = min(mml, 65536)`**,
computed once at op-build time. Everything in §2 is measured at that bucket. §5 prices the
per-step version that does not exist.

---

## 2. The composed measurement — `measured (j-74f9b184-8cf7-4db2-9907)`

Full call, median ms, one TensorCore, `reps=10`, `early_exit=False` unless noted.
`drift %` is a second incumbent arm against the first, bracketing all four arms: it is the
**noise floor**, and a delta below it is not a finding.

| regime | ee | inc ms | B ms | BA ms | BAE ms | B Δ% | BA Δ% | **BAE Δ%** | BAE × | drift % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mixed_64k` | F | 4.782 | 3.745 | 3.735 | 3.722 | −21.69 | −21.90 | **−22.17** | 1.285 | 0.37 |
| `mixed_64k` | **T** | 4.772 | 3.761 | 3.762 | 3.703 | −21.20 | −21.18 | **−22.40** | 1.289 | 0.25 |
| `mixed_confound` | F | 2.296 | 0.831 | 0.828 | 0.828 | −63.82 | −63.91 | **−63.94** | 2.773 | −0.10 |
| `mixed_32k` | F | 2.004 | 1.936 | 1.956 | 1.936 | −3.39 | −2.39 | **−3.39** | 1.035 | −0.82 |
| `mixed_prod` | F | 0.669 | 0.658 | 0.649 | 0.649 | −1.64 | −2.93 | **−3.00** | 1.031 | −0.03 |
| `decode_64k` | F | 0.202 | 0.197 | 0.184 | 0.161 | −2.40 | −8.69 | **−20.31** | 1.255 | −0.72 |
| `decode_prod` | F | 0.167 | 0.166 | 0.147 | 0.134 | −0.98 | −11.89 | **−19.99** | 1.250 | −0.62 |

Within-arm spreads are 0.5–3.8% everywhere except one cell: B at `mixed_confound` reported
**47.52%**. Its median (0.831 ms) is corroborated by the BA arm at the same shape (0.828 ms,
spread 0.45%), so the conclusion holds, but do not quote that single arm's spread as clean.

`ee=T` is the checked-in call-site value (`shapes.CALLSITE_ENABLE_EARLY_EXIT`). `ee=F` is
what every step-3 number was taken at, and is carried so the comparison below is like-for-like.

### Composition is ADDITIVE. Nothing was lost.

The attribution arm is what says so. Each contribution is measured against the arm it is
stacked on, not against the incumbent:

| regime | A in composition (BA vs B) | E in composition (BAE vs BA) | drift % | reading |
| --- | --- | --- | --- | --- |
| `mixed_64k` ee=F | −0.26 | −0.34 | 0.37 | both inside the noise floor — no finding either way |
| `mixed_64k` ee=T | +0.02 | **−1.55** | 0.25 | E is real at the shipping early-exit setting |
| `mixed_confound` | −0.26 | −0.06 | −0.10 | B carries this shape alone |
| `mixed_32k` | **+1.03** | −1.02 | −0.82 | A is marginally *negative* here |
| `mixed_prod` | **−1.31** | −0.07 | −0.03 | A real, E absent |
| `decode_64k` | **−6.45** | **−12.73** | −0.72 | both large |
| `decode_prod` | **−11.02** | **−9.19** | −0.62 | both large |

- **B alone reproduces step 3 to 0.03 pp**: −21.69% here against −21.72% then, at
  `mixed_64k`, in a different job. Cross-job incumbent stability at that regime is now
  4.782 / 4.784 / 4.786 / 4.789 ms across four jobs — 0.15% spread.
- **A's contribution in composition brackets its standalone numbers.** Step 3 measured A's
  reader at −10.45% (`decode_prod`) and −7.66% (`decode_64k`) standalone; in composition it
  is −11.02% and −6.45%. Both mixed regimes were inconclusive in step 3 and remain so here.
- **No interaction loss anywhere.** There is no regime where the composed arm is worse than
  the best single component by more than drift.

### At the A/B shape

Weighting by the `modelled` decode:mixed time split at the A/B shape (DECODE 12.1–14.3% of
indexer time, MIXED 85.7–87.9% — `results.md` §2), using `mixed_64k` and `decode_64k`:

| candidate | indexer-time change at the A/B shape | tag |
| --- | --- | --- |
| B alone | −18.9% to −19.4% | `modelled` from `measured (j-74f9b184)` |
| B + A | −20.0% to −20.3% | same |
| **B + A + E** | **−21.9% to −22.0%** | same |

A is therefore worth **0.93–1.10%** of indexer time and E a further **~1.7 pp**. Both are
consistent with the corrected step-3 figures.

**Do not read this as an end-to-end number.** `results.md` §7 puts this block at
**2.69% of the decode step and 1.02% of the prefill step at the checked-in production
shape**, against an end-to-end `spread_pct` of **2.77%**. A 22% saving on a 2.69% block is
~0.6% of the decode step — at or below the noise of one end-to-end run, needing ~190 runs
per arm to resolve. That is the honest ceiling, and it is a *different configuration* from
the one the 85.7/12.1 split was taken at: the two must not be multiplied together. The A/B
runs at long context precisely because the indexer's cost is O(resident KV positions) while
the sparse-MLA it feeds saturates at `k = 2048`, so its share should be larger there — how
much larger is `unmodelled` until the A/B profile lands.

---

## 3. Correctness — `measured (j-74f9b184-8cf7-4db2-9907)`, red = 0

Float64 oracle, `RANK_SLACK = 256`, `MIN_RECALL = 0.99`, plus **index-set equality against
the incumbent**, which is strictly stronger. The lane counter and `neg_control_unwritten`
are **imported** from `probes/bucket_correctness/` (the B-alone gate, `j-6454262e-acd2-4371-b3ba`)
rather than rebuilt.

### 3a. `bucket > kv_len` — THE SHIPPING CONFIGURATION. **PASS.**

This is the sentence the recommendation rests on, and before this job it did not exist for
the composed candidate. Unwritten sublanes are **counted off the device buffer**, not
predicted, and agree with the analytic prediction in every row:

| shape | bucket | buffer | **unwritten** | first–last | cells below kv_len | composed | poison | `neg_control_unwritten` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `unwritten_decode_8000` | 16384 | 128 sub | **64** | 64–127 | 0 | PASS, set_diff 0 | PASS, set_diff 0 | **FAIL** ✓ |
| `unwritten_mixed_24576` | 32768 | 256 sub | **64** | 192–255 | 0 | PASS, set_diff 0 | PASS, set_diff 0 | **FAIL** ✓ |
| `mixed_confound` | 65536 | 512 sub | **448** | 64–511 | 0 | PASS, set_diff 0 | PASS, set_diff 0 | **FAIL** ✓ |
| `decode_prod` / `mixed_prod` | 9216 | 72 sub | **8** | 64–71 | 0 | PASS, set_diff 0 | PASS, set_diff 0 | **FAIL** ✓ |
| `mask_mixed_8000` / `mask_decode_8000` | 9216 | 72 sub | **8** | 64–71 | 0 | PASS, set_diff 0 | PASS, set_diff 0 | **FAIL** ✓ |

The first three ran at **both** `(2,2,2)` and Option E's `(4,2,4)`, with identical results —
so E's tuple is covered, not assumed. `worst_rank` and `min_recall` are identical **to the
digit** between incumbent, composed and poison at every shape (e.g. `unwritten_mixed_24576`:
2129 / 0.99268 on all three; allowed 2304).

What makes this non-vacuous: `neg_control_unwritten` runs the **same poisoned buffer with
only the reachability gate removed** and FAILS at every shape where the measured unwritten
count is > 0. Where the count is 0 it is reported **vacuous, not passing**. The ordinary
`neg_control` (half the query heads zeroed) FAILS everywhere: `worst_rank` 7921–62644 against
an allowed 2304, `min_recall` 0.25–0.52.

### 3b. `bucket == kv_len` — the masked partial-block store tail. **PASS.**

`tail_static_mixed_8300` (bucket 9216 = 4 slabs + an 8-sublane tail, the tail slab reached
because `num_bkv = 5 = max_bkv`) and the two `mask_*_8000` shapes at bucket 8000
(63-sublane width): composed and poison PASS with `set_diff 0`, `neg_control` FAILS.

**The two risks are structurally mutually exclusive** and cannot compound: the tail slab is
only visited when `num_bkv == max_bkv`, which is exactly the case in which the kernel writes
the buffer's whole width. Confirmed empirically here (`tail_static_mixed_8300` measures 0
unwritten) and independently in the B-alone gate.

### 3c. Two honest limits on the poison evidence

1. The poison arm necessarily runs with **`aliased = True`** (it needs a `scores_init` to
   poison), while the shipped candidate's first band runs `aliased = False`. It is a
   faithful proxy for the **coverage** question — does selection ever read a lane the store
   did not write — not a literal execution of the shipped path.
2. This is a **single-chip microbenchmark with synthetic inputs**, not an accuracy check on
   a served model. Option A's bit-exactness is additionally true *by construction* (e8m0 is
   a bare 8-bit exponent; bf16 and f32 both have 8), and `set_diff = 0` on every arm at
   every shape is consistent with that.

---

## 4. Does E earn its place? **Yes — include it.**

Step 3 left E's sign unresolved (−0.84% to +0.90%) and required it be measured *composed
with B*, not summed. Composed, it is:

- **−1.55%** at `mixed_64k` with `early_exit=True` — the checked-in call-site value — against
  a 0.25% drift floor. At `ee=False` it is −0.34% against 0.37% drift, i.e. nothing. The
  shipping setting is the one that matters, and E wins there.
- **−9.19%** and **−12.73%** on the two decode regimes, far above drift.
- Never a regression: the worst cell is −0.06%.

Do not walk further up the tuple ladder. `(8,4,4)` is `measured` **4.2–4.7% slower** at
`mixed_prod` — 0.67238 → 0.70413 ms, +4.72%, bootstrap 95% CI [+4.23, +5.09], 59 of 60
paired blocks agreeing on the sign (`results.md` §8.4). Its buffer is 128 sublanes against
`(4,2,4)`'s 80.

---

## 5. Gaps, priced

**Recompile / jit-cache cost — was `unmodelled`, now `measured`. The gap is closed and it
is not a cost.** AOT `lower().compile()` on a contended shard:

| | compile s | Δ s vs incumbent |
| --- | --- | --- |
| incumbent | 2.42 | — |
| **composed at the shipping bucket 65536** | **2.39** | **−0.04** |
| all 8 hypothetical ladder rungs together | 18.3 | +15.9 |

The shipping configuration compiles **one** bucket and *replaces* the incumbent's compile,
so its marginal warmup cost is −0.04 s. Even a full 8-rung per-step ladder would add ~16 s
of one-time warmup. Compile-time seconds on a contended shard: an order of magnitude, not a
benchmark.

**Call-site plumbing — still open, and now bounded.** A per-step bucket would be worth, at
`mixed_confound`: `measured` −73.10% (B, exact bucket 8192, 2.305 → 0.620 ms) against
`measured` −63.82% (B, static bucket 65536, 2.296 → 0.831 ms) — an extra **1.34×** on top of
what ships, at a shape chosen to maximise the gap. Reaching it needs a static int that
varies per serving step, which the single-trace/guards-dropped compilation model does not
admit. A host `int` upper bound is cheap to compute in the runner (`seq_lens_np` is already
host-resident and never read back from device) but does not reach `forward_oot`;
`AttentionMetadata.seq_lens_cpu` is a **dead declaration**, never assigned anywhere in the
repo. The plausible route is riding the existing two-way `VLLM_TPU_MOST_MODEL_LEN` split,
which is already host-decided per step and already warmed per token bucket — a 2-rung
ladder, not 8. **Not attempted; not part of this candidate.**

---

## 6. Recommendation

Ship **B + A + E composed**, at `bucket = min(max_model_len, 65536)`.
`modelled` −21.9% to −22.0% of indexer time at the A/B shape, from
`measured (j-74f9b184-8cf7-4db2-9907)`; correctness PASS in both bucket configurations with
a non-vacuous negative control in each. Next step is `/preflight-check` and then the
end-to-end benchmark — and go in expecting **~0.6% of step time**, not 22%.

Note A's shipping half is two coupled changes: the kernel reader *and* the writer at
`mla_attention_op.py:103-104`. Landing the reader without the writer changes the cache
format contract. The writer is `measured` **3.342×** faster at T=1024 (`results.md` §5b-v2);
its absolute microseconds are explicitly not quotable.

---

## `carried:` — measured improvements that are not `streamindex_topk` kernel edits

1. **A's writer change, `mla_attention_op.py:103-104`** — `measured` 3.342× at T=1024.
   *Route:* ships with A, same PR. *Next:* include in the preflight diff; it is not optional.
2. **Option E is a one-line call-site change** — `mla_attention_op.py` `(2,2,2)` → `(4,2,4)`.
   *Route:* same PR. *Next:* note `deepseek_v4_indexer.py:93` uses `(3,2,2)` and was **not**
   measured; leave it alone or measure it separately.
3. **Two duplicate independent `sparsecore_topk` unit tests**, written by the B and C agents.
   *Route:* consolidate into one and land under `tests/`. *Next:* pick one, delete the other.
4. **`W_BD`, the producer-side handoff** — fails its numerics gate as written.
   *Route:* its own exercise. *Next:* re-derive the gate before writing more code.
5. **`references/` promotions**, each `measured` in this exercise:
   Mosaic tiles the **last two** dims of an HBM memref (a 2-D scale plane is rejected);
   fp16 not bf16 for a top-k key; `--xla_allow_excess_precision` defaults true and silently
   deletes a bf16 round trip; the `jax._src.linear_util` jaxpr memo survives
   `f.clear_cache()`; spare MXU MACs are not spare MXU issue bandwidth.
   *Route:* `write-tpu-kernels/references/`. *Next:* one PR, generic phrasing only.
6. **`instruments/shapes.py` has a footgun worth a comment**: `num_sublanes_total`'s default
   `bkv_ps=NUM_KV_PAGES_PER_BLOCK` binds at def time, so rebinding the module attribute
   changes what the kernel gets but not what the instrument reports. It is read-only, so
   this is a note for whoever next revises it, not an edit.
