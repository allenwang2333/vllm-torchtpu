# Probe 05 — `refinement-count`: how many of the 3 refinement levels actually run?

**Question.** `ref_body` is a full pass over `L` and there are 3 of them, each skipped per row once
`done` is set (`sparsecore_topk.py:221-265`). So the compute term swings 2× depending on a number
nothing in the kernel reports. Report §5d multiplies that count by a measured per-pass price; a 2×
error there reorders the entire step-2 candidate list.

**Why it did NOT need hardware.** `done` is a deterministic function of the score data — no timing,
no scheduling, no compiler behaviour enters it. The exact trip count is obtainable by replaying the
kernel's own integer arithmetic. This is the "never infer a structural fact from a timing delta when
a static artifact states it" rule applied one step further: the artifact here is the algorithm.

## Arms

`probe.py`, on the host. Replays lines 183–265 per row: `monotone_key` → 256-bucket L0 histogram at
shift 24 → inclusive prefix sum → `find_bucket(total−k)` → `done = keep_all | quota==cnt`, then per
level the masked re-histogram and `done |= (quota2==cnt2) | (quota2==0)`.

Three distributions × `b ∈ {64, 1024}` at `n=10240, k=2048`, seed 20260904. `heavy_ties` (17 distinct
values) is the degenerate arm — it forces every level, which is what the rule about putting the
degenerate value of the scaling parameter in the case list asks for.

**The reported statistic is per-subcore, not per-row.** There is no `subcore_barrier` anywhere in
the file (lines 19 and 528 both advertise this), and subcore `flat` owns rows `flat, flat+32, …`
(line 157), so the kernel costs `max over 32 subcores of the sum of its own rows` — not the mean,
and not the sum of per-wave maxima.

## Config

Host, no TPU, no job. `probe.py` → `host-run.txt`. Runs in about a second; rerun it rather than
trusting the table.

## Answer

```
dist             b  mean/row  crit/row   imbal  waves  passes/row
random          64      1.95      2.50    1.28x      2        5.50
random        1024      1.92      2.03    1.06x     32        5.03
uniform         64      2.05      2.50    1.22x      2        5.50
uniform       1024      2.06      2.25    1.09x     32        5.25
heavy_ties      64      3.00      3.00    1.00x      2        6.00
heavy_ties    1024      3.00      3.00    1.00x     32        6.00
```

Two findings, one of which was not the question:

1. **≈2.0 refinement levels at prefill, 2.50 at decode** — so ≈5.0 and 5.5 total passes over `L`,
   the top of the 3–6 range rather than the middle.
2. **Subcore imbalance is a decode problem, not a prefill one** — 1.06× at 32 rows/subcore against
   **1.28× at 2**. With only two rows each, one subcore drawing two 3-level rows sets the wave time
   for all 32. Roughly a fifth of decode's refinement work is the critical subcore waiting on
   nothing. This is orthogonal to the DMA question and is carried to step 2 separately.

## Decided

- Report §5d fits its single free parameter against `crit/row`, giving **114.3 µs per refinement
  pass = 0.98× a dense scatter**, and validates out of sample at `b=64` to **−0.14%**.
- Report §9 ranks the refinement walk (51.4% of the kernel) above the caller's DMA mechanism (7.4%).

**Caveat carried forward.** The count is exact *given a distribution*, and the distribution is
assumed. `uniform` moves the prefill figure 2.03 → 2.25 (11% on half the kernel) and `heavy_ties`
pins it at 3.00. The real input is a DSA score vector and nothing here says which row it resembles.
A step-2 candidate whose prize depends on the **count** must re-derive it; one whose prize depends
on the per-pass **cost** — which is what candidate 1 is — does not.
