# Step-3 variant brief — identical for every variant agent

You are building **one** candidate from `design.md`. Everything in this file applies to every
variant agent equally; your dispatch message names *which* candidate and nothing else that differs.

Repo: `/mnt/pd/home-mp/vllm-torchtpu-test3/vllm-torchtpu` (branch `hning/sctopk-pipeline`).
Exercise: `kernel_work/sparsecore-topk/`.
Skill reference root: `/mnt/pd/home-mp/vllm-torchtpu-test3/.claude/skills/write-tpu-kernels/`.

---

## 0. One sign convention — verbatim, and it is binding

> Every percentage is `(new − old)/old` of **time**. Negative is faster. A signed percentage
> appears only in a column headed `Δ time %`. Never write "+20% faster". Never quote an unsigned
> percentage as an improvement.

This holds in your code comments, your `README.md`, your risk table, and anything you return.

## 1. Your folder, and the deliberate duplication

`kernel_work/sparsecore-topk/variants/<your-variant>/` already exists and already contains a
**verbatim copy of the incumbent subject** `sparsecore_topk.py`
(md5 `44353b347c96e0eedbd742c710d4540a`, commit `b11b6eb6`). Edit **that copy**, in place.

- ⛔ **Do not edit `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`.** The live tree stays
  at the incumbent for the whole of step 3; variants reach the pod via `--map-dir`.
- ⛔ **Do not read or edit any other variant's folder,** and do not factor a helper out into a file
  two variants share. The duplication is deliberate: a shared helper adjusted for one variant
  silently moves the other's number.
- Everything you produce lands in your folder. Nothing you produce lands anywhere else.

## 2. ⛔ Instruments are READ-ONLY — this is the one that voids a result

`kernel_work/sparsecore-topk/instruments/` holds the frozen unit test, the reference and the
microbenchmark, chmod `a-w` on purpose:

| file | md5 |
| --- | --- |
| `test_dsv4_sparsecore_topk.py` | `b3241859e471fc87fa6e7a50e4ffec4b` |
| `sctopk_2stage_bench.py` | `bdd978e6a76a9a73043b0a72c6a7a1d7` |
| `glm_dsa_block_bench.py` | `08375d7068365e1a776c58fef09cd6d0` |

**Read them. Do not modify them, do not modify the live copies they were frozen from, and do not
relax a tolerance, swap a reference, or narrow a `-k` selection anywhere.** A variant that changes
the test, the reference or the tolerance **has not passed — it has moved the gate**, and its number
is thrown away. If your candidate genuinely needs an arm the frozen bench does not have, say so in
your report and hand the requirement to the measurer; do not build it yourself.

You may **read** `instruments/README.md` — it lists three prints of the harness that are traps, and
what the instruments cannot measure.

## 3. What you must read before writing a line

1. **Your candidate's own section in `design.md`** — the plan section (§9, Plan A/B/C/E) or the §5
   version block, whichever your dispatch names. Read the **whole** section including its
   **file plan (§4 "Files")** and its **§7 risk table**.
2. **`references/code-style.md`** — *in full*, under the skill reference root. Binding. You return
   its checklist **item by item**, each `✅` or `❌ — because …`. Two classes are **blocking**:
   - hardware constants are **read from the platform, not hardcoded**;
   - what the kernel is tested against, and **where its tolerance is written**.
3. **The area files your plan leans on**, named in your dispatch, under `references/`.
4. **`report.md` §4 "Shape derivation"** — the step-1 shapes. The ones that matter:

   | | value |
   | --- | --- |
   | `k` | 2048 |
   | `n` | **10240** (padded page-aligned KV capacity; *not* 133120) |
   | `b` prefill | **1024** (32 waves) |
   | `b` decode | **64** (2 waves) |
   | `P` | **1** at both shipping shapes — single stage, no gathers |
   | ρ = `E[eff_len]/n` | prefill **0.40005**, decode **0.850** |

   ⚠ `n` is a capacity, not a live length; live length enters only via `row_lengths`. ⚠ The bench
   defaults to `--ctx 132096`; every step-1 number is at **`--ctx 9216`** and so is every step-3
   number.
5. **`design.md` §8b** — the pattern that killed two nodes, and the rule that follows from it.

## 4. ⛔ §8b's rule: every arm measures the whole kernel, not the region it edits

`v5` and `v6` both died the same death: each deleted work from a region correctly, and each had the
saving absorbed by a term the model had **no entry for** — for `v5`, shrinking `NUM_BUCKETS` shrank
the sweep and *grew* the `addupdate_scatter` histogram faster. Neither was visible to any amount of
care applied *inside* the model.

So: your candidate is judged on **end-to-end kernel time at the production shapes**, never on the
region it edits. Do not propose, and do not report, a region-local saving as your number.

## 5. The two halves of your mandate, in order

**First, build the plan as written.** The design predicted a number for the plan as specified; that
prediction is scored against what the plan-as-written achieves, so it has to exist as a thing that
was measured. Do not "improve" it on the way in.

**Then own an iteration loop, and make it a search.** Open the kernel with `/xprof-llo` before every
change — never iterate on a number you have not opened. An attribution names where the time went;
what closes the gap is a **move**, and the move is frequently one the design could not have chosen
because the design priced a model and this is a reading of emitted code. Walk this list against
what the reading actually showed and take what it points at: fewer operations for the same answer;
**more FLOPs on an idle unit**; memory layout, and arranging so a relayout never exists; the grid,
the tile and the pipeline; what crosses the kernel edge; the scaffolding (spills, a scalar loop
that should be SMEM metadata, an `arbitrary` axis that is really parallel, a VMEM limit).

Two rules bound this: **the unit test runs after every one of these, without exception**, and **a
move that changes what the variant fundamentally is has become a different candidate** — give it
its own version node, and its own folder if it is going to be iterated on, so the plan's number
survives instead of being overwritten by its own descendant.

⚠ **In this wave, your iteration budget is static-only** (see §6). You do not own a timing job.

## 6. What you may and may not run

| | |
| --- | --- |
| ✅ **AOT compile with no TPU** | `jax.jit(f).lower(...).compile()` against `jax.experimental.topologies.get_topology_desc("tpu7x:2x2x1")` runs the real XLA-TPU + Mosaic **SparseCore** backend on the host, in seconds, for zero cluster time. `probes/17` established it. Evidence class **`static-compile`**. It needs a **negative control**: an 8 MiB VMEM scratch must fail with `E3000: CompileTimeSparseCoreAllocationFailure … memory space spmem`. Without that control the compile proves nothing. Use `uv_venv/bin/python3`. |
| ✅ **Host-side reference simulation** in NumPy — pass counts, densities, bucket occupancies, break-even arithmetic. Free. |
| ✅ **Reading** the optimized HLO and the Mosaic MLIR your AOT compile emits. |
| ⛔ **You do not run a timing job.** Writing fans out; **ranking timings do not**. One measurer owns every timing that decides the ranking, so that all arms share a pool, an interleave and a resolution floor. |
| ⛔ **You do not run the frozen unit test on a pod yourself** in this wave — a single correctness runner fans all variants over one pod's chips. You state exactly how your variant must be invoked for it. |
| ⛔ You do not run `/load-experiment` or `/log-experiment`. This skill neither reads nor writes the shared log. |

If you believe you need a pod job, **stop and ask the main thread** rather than submitting one.

## 7. Facts about this exercise that will otherwise cost you a job

- **The kernel sits behind a per-step runtime gate** in `streamindex_topk.py:858-864`:
  `lax.cond(max(seq_lens)//compression_ratio <= k, _fast_path, _common_path)`. JAX orders
  `branches=[false_fun, true_fun]`, so **branch 0 = `_common_path`** (your kernel) and
  **branch 1 = `_fast_path`**. The predicate reduces to `max(seq_lens) <= 2048`. The kernel runs on
  6 of 8 prefill chunks; **1/4 of prefill rows have `token_pos < 2048`.** Never phrase this as
  "2 of 8 chunks".
- **A docker image tag is mutable.** Tag `fa63d6df` shipped md5 `44353b34…` when probes 07–09
  measured, and now ships `1a89d451…`. Nothing in a recipe read-back detects this. Anything that
  runs on a pod carries an **in-job md5 gate on the subject file**; it has caught this four times
  with zero false positives.
- **The denominator, stated once.** Harness **189.98 µs**; production **155.82 µs firm**; a single
  scale factor **1.219** from harness to production. The `1.466` leg is closed and refuted.
- ⚠ **Every percentage in `design.md` has a `modelled` denominator.** Every probe ran at ρ=1;
  nothing has ever run at the production ρ=0.40005. The measurer is buying that arm in the first
  job. Until it lands, a design percentage is a ratio of two predictions — do not treat one as
  measured.
- **`instruments/sctopk_2stage_bench.py` has no bug.** `probes/15` claimed its dense-histogram arm
  was ~50% dense; that claim is **retracted** (`design.md` §7). The mask at `:856` runs over
  monotone keys written at `:826`, `KEY_NEG_INF` is the monotone key of −inf, and the measured
  density is 1.000000 under both readings. Do not re-derive it and do not act on it.
- **`find_bucket` is unresolved at every radix** (0.047 / −1.385 / 3.634 µs, floor 1.859). Stop
  quoting `probes/09`'s 1.200 µs.
- ⛔ **Do not commit a bare microbench number into source.** State the mechanism in a comment; the
  number lives in `results.md`.
- **Do not read** `docs/`, `.prior_sessions_DO_NOT_READ/`, `.scratch/`, `.sctopk-log/`, `logs/`,
  `joblogs/`, `mbdump/`, `refdump_*/`, `microbench/`, root `*.md` other than `CLAUDE.md`/`README.md`,
  and never run a repo-wide `grep` over `vllm/` or `vllm-torchtpu/`.

## 8. What you return, and it is also what you write to `variants/<you>/README.md`

Write the file **and** return the same content. Sections, in this order:

1. **What you built** — one paragraph, and the exact diff summary against the incumbent
   (`diff -u` line count per hunk is enough; the code itself is in your folder).
2. **The file plan, scored.** Your plan's §4 "Files" table said which files hold what and roughly
   how many lines. Put the plan's row beside what you actually produced, and explain every
   divergence. "Same, ~6 lines as planned" is a complete row.
3. **`code-style.md` checklist, item by item**, `✅` or `❌ — because …`. Any `❌` in the two blocking
   classes (hardware constants from the platform; what it is tested against and where the tolerance
   is written) **blocks the variant** — say so plainly rather than shipping with a note.
4. **The §7 risk table, returned with an outcome on every row.** One of exactly three per row:
   - `did not happen — <the reading that shows it>`
   - `happened — <what it cost, and the fallback taken>`
   - `still unchecked — <why the signal was not observable>`
   A `threatens: correctness` row is discharged by the unit-test result **and by naming which
   assertion in the frozen test would have fired**. An unchecked correctness row blocks the variant
   exactly like a blocking `❌`.
5. **`static-compile` evidence** — that it lowers, with the negative control, and what the optimized
   HLO / Mosaic MLIR shows changed against the incumbent (op counts, VMEM high-water, spills, the
   ENTRY instruction count).
6. **Your prediction, restated as a number to be scored** — the `Δ time %` the design predicted, in
   the harness denominator and in the production denominator, and *what mechanism* it rests on.
   If your build diverged from the plan such that the prediction no longer applies, say the new one
   and say why.
7. **How the measurer runs you** — exact invocation: which file to map where, which arm names,
   which shapes (`b ∈ {64, 1024}`, `--ctx 9216`), any checksum arm your candidate needs because it
   changes what the DMA lands, and any compile-time constant of yours that deserves its own rival
   arm in the same block.
8. **How the correctness runner runs you** — exact `pytest` invocation and `--map-dir`, unmodified
   frozen test, and which of the 68 test ids exercise your change hardest.
9. **Anything you found that is not your candidate** — a `PARKED` idea with its reason and route, a
   non-Pallas improvement (constant, flag, default, call-site) with its measured or modelled prize,
   or a fact about the platform that belongs in `references/`. ⛔ "Nobody owns it" is not an
   outcome; give it a route.
10. **Open terms** — anything your candidate's verdict still rests on that is `assumed`,
    `unmodelled`, or predicted at zero. Be explicit: a term tagged `unmodelled`, predicted at zero,
    or credited as a *saving* **cannot fail** and will be recorded as `unmodelled`, not as a pass.

Keep raw output out of the README: conclusions, paths and (if any) job ids. Raw artifacts go in
your folder as files.
