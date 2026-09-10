# `v2` — `R=1`, one shared histogram (design.md §9 Plan A)

Sign convention, binding throughout: every percentage is `(new − old)/old` of **time**; negative is
faster; a signed percentage appears only under a `Δ time %` heading.

| | |
| --- | --- |
| subject | `variants/v02-r1-shared-hist/sparsecore_topk.py`, md5 **`c0ca54a66a549020cfc584e2d8d97abe`** |
| incumbent | commit `b11b6eb6`, md5 `44353b347c96e0eedbd742c710d4540a` |
| optional rival arm | `sparsecore_topk_bigscratch.py`, md5 **`70f9873e2941bd94601c15cea1dd9d55`** (§7) |
| evidence class | `static-compile` (AOT, no TPU) + host NumPy census + one **prior** measured job `j-300d4690` |
| jobs I ran | **none.** No timing job, no pytest on a pod. |

---

## 1. What you built

`v2` deletes the incumbent's 16× lane-private replication of the 256-bucket L0/refinement histogram.
The incumbent gives each of the 16 lanes its own 256-entry copy (`addr = lane_iota * NUM_BUCKETS +
bucket`) so that two lanes of the same vector can never target the same word, then folds the sixteen
copies with a 16-deep accumulate. `v2` scatters into **one** shared 256-entry histogram
(`addr = bucket`), which is safe only because `plsc.addupdate_scatter` resolves intra-vector address
conflicts in hardware; the fold then collapses to a copy, the pre-pass clear shrinks from 4096 words
to 256, and the scratch allocation falls from 16 KiB to 1 KiB per subcore. The clear is where most of
the prize is: `probes/09` split the level-0 sweep as `zero_priv` 9.238 µs of 18.866 µs total.

`diff -u` against the incumbent — **7 hunks, 12 source lines changed, 20 comment lines added**:

| hunk | incumbent line | change | lines |
| --- | ---: | --- | ---: |
| 1 | `:72` | `priv_vmem` shape comment `i32[16 * NUM_BUCKETS + LANES]` → `i32[NUM_BUCKETS + LANES]` | 1 −, 1 + |
| 2 | `:128-135` | `fold_priv` 16-deep accumulate → a copy | 4 −, 5 + (+4 comment) |
| 3 | `:185` | `zero_priv(NUM_BUCKETS)` → `zero_priv(NUM_BUCKETS // LANES)` | 1 −, 1 + (+2 comment) |
| 4 | `:191` | `hist0_body`: `addr = lane_iota * NUM_BUCKETS + bucket` → `addr = bucket` | 1 −, 1 + (+4 comment) |
| 5 | `:224` | `zero_priv(jnp.where(done, 0, NUM_BUCKETS))` → `… NUM_BUCKETS // LANES` | 1 −, 1 + (+5 comment) |
| 6 | `:238` | `ref_body`: same address change as hunk 4 | 1 −, 1 + |
| 7 | `:456` | `scratch_types`: `VMEM((LANES * NUM_BUCKETS + LANES,))` → `VMEM((NUM_BUCKETS + LANES,))` | 1 −, 1 + (+3 comment) |

**Those seven line numbers are the composition-conflict surface** — see §4 row 2.

### Is this the same mechanism as `probes/08/arms/arm-R1.py`? — asked explicitly, answered explicitly

**Yes for the mechanism; no for the code, in four named places.** `arm-R1.py` differs from the
incumbent in exactly four lines. Three of them are hunks 3, 4 and 6 above, character for character.
The differences:

| # | line | `arm-R1.py` | `v2` | why |
| --- | --- | --- | --- | --- |
| a | `:48` | `REFINEMENT_LEVELS = ()` | production `((16,0xFF),(8,0xFF),(0,0xFF))` | probe-only scaffolding. **This is the important one — see below.** |
| b | `:133` | `for m in range(1)` over a zero-seeded accumulator | a plain copy | same value (`0 + priv[0*NB+off]`); statically **cheaper** — see §5 |
| c | `:456` | **unchanged**, still `LANES * NUM_BUCKETS + LANES` (16 KiB) | `NUM_BUCKETS + LANES` (1 KiB) | the `v2` design node specifies "V4 falls from 16 KiB to 1 KiB"; `arm-R1` never did it |
| d | `:224` | **unchanged**, still `zero_priv(jnp.where(done, 0, NUM_BUCKETS))` | `… NUM_BUCKETS // LANES` | forced by (c), and unreachable in the probe because of (a) |

(d) is the finding of this build. **Plan A §2's edit list is three edits and, taken literally
together with the `v2` node's 1 KiB allocation, it is memory-corrupting**: with `priv_vmem` shrunk to
272 words, leaving `zero_priv(NUM_BUCKETS)` at `:224` writes `LANES * NUM_BUCKETS = 4096` words into
a 272-word ref on every unconverged refinement level, over the top of `glob_vmem`. The probe could
not see it — with `REFINEMENT_LEVELS = ()` the line never executes, and with the 16 KiB buffer still
allocated it would have been harmless anyway. Route in §9.

(c) is the second substantive gap: **the 12.195 µs measured in `j-300d4690` contains no
tile-memory effect at all**, because the arm kept the full 16 KiB allocation. §7 supplies a rival arm
that isolates it.

---

## 2. The file plan, scored

Plan A §4 "Files" has exactly one row.

| plan row | what I produced | divergence |
| --- | --- | --- |
| `sparsecore_topk.py` — "the three edits above", **~6 changed lines**, ⛔ no bare microbench constant | `sparsecore_topk.py` — **four** mechanism edits + 3 bookkeeping edits, **12 changed lines** + 20 comment lines | **Three divergences, all explained.** (i) A fourth mechanism edit at `:224` the plan omitted — §1(d); without it the variant corrupts memory. (ii) `scratch_types` at `:456`, which the `v2` version node requires ("16 KiB → 1 KiB", live set 72,192 → 57,152 B) but which Plan A §2's call-chain list does not name. (iii) The `:72` shape comment, required by `code-style.md` §2 — a ref comment that lies about its shape is exactly the failure that section exists for. |
| — | ⛔ **no bare number in source.** ✅ Every one of the 20 comment lines states a *mechanism* ("there is one histogram, not LANES of them"; "a bound of NUM_BUCKETS here would write LANES * NUM_BUCKETS words past it"). No µs, no percentage, no job id in the kernel. | — |
| — (no plan row) | `static_compile.py` + `hlo-*.txt` / `mosaic-*.txt`, `aot_single_arm.py` + `solo-*.txt` + `xdump-{inc,v2}/` (§5); `collision_census.py` + `.txt` (§4); `sparsecore_topk_bigscratch.py` (§7) | Correct that the plan has no row: none of these ship. They are evidence and one optional arm, all inside my folder. |

---

## 3. `code-style.md` checklist, item by item

The file is 560 lines inherited verbatim from the incumbent; I changed 12 of them. I score the file
as it stands and mark what is inherited, because a `❌` that is identical in the control arm cannot
move an A/B — but per the brief I state the two blocking classes plainly rather than as a note.

| # | item | |
| --- | --- | --- |
| 1 | Dimension glossary at the top; uppercase logical, `t`-tile, `n`-grid; padded extents own letter | ❌ — because there is no `# Dimensions.` block. `b`, `n`, `k`, `P`, `n_p`, `slice_len`, `walk_chunks`, `b_pad` are never defined in one place, and `b_pad` is precisely the "padded extent with its own letter" the rule asks for and gets no glossary line. **Inherited.** |
| 2 | Every ref parameter has a trailing `# [shape] dtype` comment, block vs whole array | ✅ — all eleven refs carry one, and **this is the item my change could have silently broken**: hunk 1 updates `priv_vmem`'s comment in the same edit that changes its shape. |
| 3 | Extents from the refs, not the config; no bare `x.shape[2]` in arithmetic | ✅ — `b_pad = len_vmem.shape[0]` (`:99`); no bare integer axis index anywhere. (`num_subcores = 16` at `:94` is item 15's problem, not this one's.) |
| 4 | jaxtyping on the public wrapper, bare refs inside; never `jax.Array` on a ref | ⚠ half — refs inside are correctly bare (✅, and I added none). The public wrapper `sparsecore_topk` has plain untyped parameters, so ❌ on the outer half. **Inherited.** |
| 5 | Signature carries `# Inputs. / # Outputs. / # Scratch.` | ❌ — `_topk_body`'s eleven refs are one flat commented list; the input/output/scratch boundary is exactly the invisible thing §6 exists to fix. **Inherited.** |
| 6 | Body order: indices → shapes → semaphores → init guard → loads → compute → masked store → epilogue | ✅ — `core`/`sub`/`flat` (`:93-98`), `b_pad` (`:99`), the length DMA and `sems` aliasing (`:101-107`), then `wave_body`. My four edits each sit in the position their predecessor occupied. |
| 7 | Single characters only for loop indices, annotated math tensors, uppercase extents, `d<x>` | ⚠ mostly ✅ (`i`, `j`, `w`, `m` are loop indices). ❌ on `s = row_c * n` (`:163`) — a computed byte offset with a one-letter name. **Inherited**; I did not add one. |
| 8 | Memory space in the ref name where two spaces share a signature; never `_ptr` | ✅ — `_hbm` / `_vmem` suffixes throughout, no `_ptr`. The file uses `_vmem` rather than `_vmem_ref`; consistent within itself, and I kept it. |
| 9 | Comments name the compromise; one-line module docstring | ⚠ split. ✅ on comments — all 20 lines I added name *why* (why one histogram is safe, why the `:224` bound is a separate edit, why the buffer shrinks), and the comment density in the region I touched went up ~4×. ❌ on the docstring: it is a 16-line paragraph. **Inherited.** |
| 10 | Unsupported features raise `NotImplementedError`, not `ValueError` | ✅ vacuous — `v2` adds no capability gate and removes none. |
| 11 | Four files `api/base/pallas_mosaic_tpu/…_kernel` | ❌ — one 560-line file holding the `pl.kernel`, the shape derivation and the public wrapper. §10's "when not to use the framework at all" is arguably the right call for a kernel doing dynamic-offset DMA and in-kernel metadata, but then the rule is *say so in the module docstring and leave*, and the docstring does not say so. **Inherited.** |
| 12 | Where it splits further, the boundary is what the module may touch (arrays / refs / DMA / ints) | ❌ — same file; `_cdiv`/`_align_to` (ints) sit beside `_topk_body` (refs + DMA). **Inherited.** |
| 13 | Config is a frozen kw_only slots pydantic dataclass; cross-field checks in `__post_init__` | n/a — there is no `Config`; the static knobs are keyword args on the wrapper. |
| 14 | **BLOCKING — tested against the XLA `_fwd`; tolerance visible at the call site** | ✅ **and in the strongest available form.** The reference is `instruments/test_dsv4_sparsecore_topk.py:75 _reference_scores`, plain NumPy in the same repo, not a golden file. The tolerance is written **at the comparison**, `test_dsv4_sparsecore_topk.py:131-136`: *"Exact: this is a selection, so a correct kernel returns the same score values bit for bit. No relative tolerance."* → `if diff != 0.0`. There is no tolerance to loosen. The `Precision.HIGHEST` sub-item is n/a: a top-k contains no contraction. |
| 15 | **BLOCKING — hardware constants from the platform, never hardcoded** | ❌ **— and I am stating it plainly as a blocking `❌`, not shipping it with a note.** `LANES = 16` (`:41`), `MAX_SLICE_WORDS = 64 * 1024` (`:45`, the 512 KiB tile expressed in words), `num_subcores = 16` (`:94`) and `num_sc_cores = 2` (`:95`) are hardcoded, while `plsc.get_sparse_core_info().num_lanes`, `sc.num_cores`, `sc.num_subcores` and `sc.vmem_capacity_bytes` exist — and the *launcher* already reads them at `:445-459` to build the mesh. So the same four numbers are read from the platform in one place and written literally in another, which is the drift `references/08-sparsecore.md:92-95` names outright ("a kernel that hard-codes 8 is a v5p kernel"). |

**On item 15, three things the reviewer needs and one route.** (a) It is inherited verbatim: byte for
byte identical in the incumbent, in the control arm, and — because every variant folder started from
the same md5 — in every other variant in this wave. It therefore cannot move any A/B in step 3.
(b) `v2` moves the file *toward* the rule, not away: deleting the replication removes `LANES` from
the histogram's address arithmetic and from its allocation, so one of the four constants now has
strictly fewer load-bearing uses. (c) It nonetheless **blocks by the brief's letter, and I am not
fixing it here**, because a change to `LANES`'s provenance rewrites lines every arm shares and would
have to land in the control too or it becomes a confound. **Route:** a `[mutate]` node against the
*incumbent*, "read the SparseCore geometry from `plsc.get_sparse_core_info()` in the kernel body",
owned by step 4 and applied to the base so the control moves with it.

---

## 4. Plan A §7 risk table, with an outcome on every row

### Row 1 — `addupdate_scatter`'s conflict resolution is shape- or count-dependent and fails at a shape not yet run · threatens **correctness**

> **`did not happen` for the mechanism's core, at both production shapes — and the coverage is
> quantified, not asserted. ⏳ Three named sub-cases remain for the correctness runner.**

**The reading that shows it.** `probes/08` (`j-300d4690`) compared `R1` against `B` row by row:
**1024/1024 bit-identical at b=1024 and 64/64 at b=64.** That is only evidence if the input actually
produced conflicts, so I counted them on the host — `collision_census.py`, NumPy only, replaying the
kernel's own `monotone_key` and `key >> L0_SHIFT` on the bench's own `SEED = 20260904`:

| input, n=10240 | vectors with ≥1 intra-vector bucket collision | colliding lane pairs |
| --- | --- | ---: |
| `random` (what `j-300d4690` ran), b=1024 | **655,360 of 655,360 — 100.0%** | **6,734,859** |
| `random`, b=64 | 40,960 of 40,960 — 100.0% | 421,174 |
| `heavy_ties`, b=1024 | 100.0%, and only **4** occupied L0 buckets | 8,362,161 |
| `strict_desc`, b=1024 | 100.0%, 9 occupied buckets | 9,827,328 |

Only 16–17 of 256 L0 buckets are ever occupied by `standard_normal` (the exponent byte dominates the
monotone key), so with 16 lanes per vector a collision is not rare — it is universal. A single
dropped duplicate undercounts `total`, which moves `b_star = find_bucket(total − k)`, which moves
`quota`, which changes the emitted set. **Bit-identical output over 6.73 M colliding lane pairs is a
direct measurement that the hardware resolves the conflict**, and it is bit-identical, so it also
closes every value-mediated mechanism for the delta.

**Which assertion in the frozen test would fire — asked explicitly, and the answer is that ones
do.** Not "none would". A dropped duplicate reaches, in order:

1. `test_dsv4_sparsecore_topk.py:126-128` — `if not (n_ref == n_got).all(): return f"row {bad}:
   returned {n_got[bad]} valid entries, expected {n_ref[bad]}"`, the wrong-cardinality path, which is
   what a wrong `quota` produces;
2. failing that, `:133-136` — `diff = np.abs(ref[finite] - got[finite]).max(); if diff != 0.0`, the
   exact-score path.

Both are reached from `test_exact_topk_full_rows` (`:231-236`) via
`assert _mismatch(scores, _run(...), k, row_lengths) is None`, and `_mismatch` is proven capable of
going red by `test_comparison_detects_a_wrong_index` (`:383-…`), which perturbs one index and
requires a failure. The out-of-bounds hazard of §1(d), had I shipped Plan A §2 literally, lands on
the same two assertions by way of a corrupted `glob_vmem`.

**⏳ What the probe could not reach, and so is on the correctness runner:** (i) the *refinement-level*
scatter at `:238`, never executed under `R=1` because every probe arm had `REFINEMENT_LEVELS = ()` —
its mask is `match`, i.e. strictly sparser than L0's `valid`, so it has strictly fewer conflicts and
the L0 result bounds it, but that is an argument, not a reading; (ii) `heavy_ties`, which
concentrates every key into 4 buckets and is the true worst case for conflict density; (iii)
`strict_desc`, the negative-heavy input this risk row names — see the §8 test ids. **Until the
correctness runner returns, treat this row as discharged-in-the-core and open at the edges.**

**Fallback if it goes red:** revert. It is 12 lines.

### Row 2 — the win does not survive composition with Plan B or D, which touch the same clear · threatens **speed**

> **`still unchecked — the signal is per-arm absolutes in one job, and I do not own a timing job.
> Step 4 owns it.** What I can hand over is the conflict surface, made checkable.**

**Lines `v2` touches** (incumbent numbering, from §1): `:72`, `:128-135`, `:185`, `:191`, `:224`,
`:238`, `:456`. Of these, `:185`, `:224` and `:456` are the *clear and its allocation*; `:191` and
`:238` are the *scatter address*; `:128-135` is the *fold*.

Two named interactions, with the direction I predict:

- **Plan B / `v3` (fused emit)** adds a conditional store inside the level-0 walk. Its edit lands in
  `hist0_body`'s body around `:186-196`, i.e. **textually adjacent to hunk 4 but not overlapping it**
  — `v3` changes what the walk *emits*, `v2` changes where the scatter *addresses*. I predict a clean
  textual merge and near-additive timing, because `v2`'s prize is in `zero_priv`/`fold_priv`
  (`:185`, `:128-135`) which `v3` does not enter. Not verified; step 4.
- **Plan D / `v5`,`v7` (radix `NUM_BUCKETS`)** is the real hazard and `design.md` §8 already records
  it: `9.238` of `v2`'s `12.195 µs` is `zero_priv` shrinking, and shrinking `NUM_BUCKETS` shrinks the
  *same* term. They are **not additive**. My sharper prediction: after `v2`, `zero_priv` is 15/16
  smaller, so a `NUM_BUCKETS` reduction has ~16× less left to win there — while `probes/16` measured
  that shrinking `NUM_BUCKETS` *grows* the scatter histogram by more. **`v2` should make `v5`/`v7`
  strictly less attractive**, and that is a falsifiable claim step 4 can settle with the two arms in
  one job.

---

## 5. `static-compile` evidence

`static_compile.py` (five arms, one process) and `aot_single_arm.py` (one arm, one process — the
cross-check); raw output `static_compile.txt`, artifacts `hlo-*.txt`, `mosaic-*.txt`,
`solo-{hlo,jaxpr}-*.txt`, `xdump-{inc,v2}/`. No TPU attached:
`jax.experimental.topologies.get_topology_desc("tpu7x:2x2x1")` with `probes/17`'s pinned
`jax._src.pallas.mosaic.tpu_info` prologue, so `.compile()` runs the real XLA-TPU + Mosaic SparseCore
pipeline on the host. jax 0.10.2. Shape: b=1024, n=10240, k=2048 — the production prefill shape at
`--ctx 9216`. **No timing is produced or implied.**

**Negative control, first, because without it none of the below counts.** ⚠ It has to be refused by
the *SparseCore allocator*, not by the MLIR layer — the tempting `o[...] = jnp.zeros_like(o)` form
fails with `NotImplementedError: Swap does not support storing to HBM`, which proves nothing about
allocation. Mine (`static_compile.py:148-182`) is the working form: an 8 MiB VMEM scratch DMA'd to
HBM, asserted on all three of `E3000` / `spmem` / `SparseCoreAllocationFailure`. It fails as
required:

```
NEG control FAILED as required: True
INTERNAL: Failed to run MLO pass pipeline. aot_negctl_oversized_vmem.6:49:1: error:
'memref.alloca' op E3000: CompileTimeSparseCoreAllocationFailure: current allocation offset upper
bound (33554431 words) exceeds the legitimate user allocatable offset upper bound (2097151 words)
in memory space spmem
```

The SparseCore allocator is live and this pipeline is the one that would reject an over-sized
histogram.

**Arms.** `inc` = incumbent via `git show b11b6eb6:…`; `v2` = this folder's file; `v2f` = `v2` with
`arm-R1`'s fold spelling *and* the incumbent's 16 KiB scratch (i.e. `arm-R1` with production
`REFINEMENT_LEVELS`); `v2fold` = `v2` with only the fold spelling reverted; `bigscratch` = `v2` with
only the allocation reverted (§7). All five compile.

| | `inc` | `v2` | `v2f` (= arm-R1) | `bigscratch` |
| --- | ---: | ---: | ---: | ---: |
| ENTRY instructions | 13 | 13 | 13 | 13 |
| ENTRY identical to `inc`? | — | **yes** | yes | yes |
| optimized HLO chars | 42,708 | 37,863 | 38,360 | 37,902 |
| Pallas jaxpr equations | 1,555 | **1,013** (−34.9%) | 1,037 | 1,013 |
| `add` equations | 314 | **50** (−84.1%) | 66 | 50 |
| `broadcast_in_dim` | 38 | 30 | 38 | 30 |
| VMEM refs `i32[4112]` (16 KiB histogram) | **×32** | **absent** | ×32 | ×32 |
| VMEM refs `i32[272]` (1 KiB) | ×38 | **×70** | ×38 | ×38 |
| spill lines in optimized HLO | 0 | 0 | 0 | 0 |

Four things this settles.

1. **ENTRY is byte-identical across all arms.** The whole change is inside the single
   `tpu_custom_call`; the surrounding XLA graph — including the untiling `copy` that makes the
   harness denominator 1.219× the production one — is untouched. So the harness→production scale
   factor carries over unchanged and my §6 production number is not double-counting a graph change.
2. **The layout change is exactly what the `v2` design node specified.** Every `i32[4112]` reference
   disappears; every histogram reference is `i32[272]`. 4112 → 272 words is **−15,360 B per subcore**
   of the 512 KiB tile, matching the node's "V4 falls from 16 KiB to 1 KiB". 0 spills before and
   after, so nothing was pushed out to make room and nothing came back.
3. **The 264 deleted `add` equations are the 16-deep fold chain**, which is the mechanism, not an
   incidental. `mul` is 84 → 76 and `scf.for` is 25 → 25: no loop was deleted, only the body of one.
4. **Fidelity to the measured arm.** `v2` and `v2f`/`v2fold` are *not* jaxpr-identical: `arm-R1`'s
   `for m in range(1)` over a zero-seeded accumulator emits 5 operations per fold body (broadcast a
   zero, add 0 to the offset, load, add the zeros, store) where `v2`'s copy emits 2, a difference of
   24 equations and 16 `add`s. They compute the same value, and **`v2` is strictly the cheaper of the
   two**, so `v2` cannot be slower than the arm that measured 12.195 µs on this account.
   `bigscratch` has *identical* jaxpr equation counts to `v2` (1,013 / 50) and differs only in the
   VMEM census — which is what makes it a clean isolator in §7.

### Cross-check: one arm per process — because a five-arm table in one process can be a lie

`jax.jit` keys its cache on `(function object, avals)`. An in-model A/B done by rebinding a module
attribute therefore compiles the first arm and then **hands the same executable back for the second**
— silently, with no error, producing byte-identical HLO for two different kernels. That failure mode
is sharper for `v2` than for anyone else in this wave: **I am the harness's known answer, so a `v2`
that quietly compiled as the incumbent would look like a perfectly clean lift.**

`static_compile.py` should be immune — it loads each arm as its own module object via
`importlib.util.spec_from_file_location` under a distinct `sys.modules` name, so no two arms share a
function object. "Should be" is not evidence. `aot_single_arm.py` recompiles the two load-bearing
arms in **separate processes, variant first** (so a residual bug reads as inc-looking-like-v2, the
direction that would embarrass me rather than flatter me), printing the identity of the bound symbol
before compiling:

| | `v2` (ran first) | `inc` (ran second) |
| --- | --- | --- |
| source md5 | `c0ca54a6…8dbe` | `44353b34…4540a` |
| `fn.__code__.co_filename` | `_solo_v2.py` | `_solo_inc.py` |
| `lane_iota * NUM_BUCKETS + bucket` in source | 0 | 2 |
| `pltpu.VMEM((LANES * NUM_BUCKETS + LANES,)` | 0 → 1 KiB | 1 → 16 KiB |
| ENTRY md5 | `ae9f584d2d1038333e625efa1687db33` | `ae9f584d2d1038333e625efa1687db33` — **identical** |
| full HLO md5 | `afdfcead…` | `2a15c866…` — **differ** |
| jaxpr md5 | `d4c1853b…` | `cd55ae7b…` — **differ** |
| jaxpr equations* | 591 | 871 (**−32.1%**) |
| `add` equations* | 46 | 310 (**−85.2%**) |
| VMEM refs `i32[4112]` | absent | ×32 |

\* this script's regex is not `static_compile.py`'s, so the **absolute** counts differ from the
1,013 / 1,555 and 50 / 314 in the table above. The **ratios** agree: −32.1% vs −34.9% equations,
−85.2% vs −84.1% adds. Two independent counting rules, one conclusion.

**Verdict: the cache trap did not bite.** Two source md5s, two HLO md5s, two jaxpr md5s, and the
4112-word histogram present in one arm and absent in the other — had the arms collapsed, all six
would have matched. And the ENTRY md5 being **byte-identical across two separate processes** is a
stronger form of settling-point 1 than the same-process comparison was.

⚠ **One marker I predicted wrong, stated rather than quietly dropped:** I expected
`NUM_BUCKETS // LANES` to appear 0 times in the incumbent. It appears **5** times — it is already the
file's idiom for "the 16 vectors of 16 lanes spanning 256 buckets". So that string is not an arm
discriminator and I do not use it as one. It is also a small reassurance about hunks 3 and 5: `v2`
reuses an existing idiom at its existing value rather than introducing a new constant.

### What the XLA dump directory holds that `as_text()` does not

`XLA_FLAGS=--xla_dump_to=<dir>` on a host-only AOT compile writes 19 files per module, three of which
carry information `compiled.as_text()` does not have. Free, and an independent check that the R=1
histogram really shrank.

**`*.sparse_core_specific_metadata.txt` — SparseCore sequencer bundle counts:**

| computation | sequencer | `inc` | `v2` | Δ |
| --- | --- | ---: | ---: | ---: |
| `called_computation` | SEQUENCER | 58 | 58 | 0 |
| `called_computation` | TILE_EXECUTE_CORE | 95 | 95 | 0 |
| `called_computation.1` | SEQUENCER | 60 | 60 | 0 |
| **`called_computation.1`** | **TILE_EXECUTE_CORE** | **1,168** | **919** | **−249 (−21.3%)** |

`called_computation.1` is the top-k kernel. ⚠ **This is static code size, not executed bundles** —
the loops are dynamic, so −21.3% of the instruction stream is *not* a −21.3% time prediction and I do
not use it as one in §6. What it is: an independent confirmation, from a file no other artifact in
this folder contains, that the R=1 edit shrank the emitted VLIW stream, and that the shrink is
confined to exactly one computation while the other three sequencer counts are unchanged.
`after_codegen.txt` corroborates at 46,315 → 41,470 bytes.

**`*-buffer-assignment.txt` and `*-memory-usage-report.txt` — byte-identical between the arms.**
`diff` is empty for both. HBM is 128.07 MiB in both: 40 MiB `f32[1024,10240]` in, 8 MiB
`s32[1024,2048]` out, 4 KiB `s32[1024]` lengths. XLA's own `vmem (color=1)` space is **empty** in
both arms.

That last line is a **negative result worth stating**: the 15,360 B/subcore of SparseCore tile memory
`v2` hands back is **not visible at the XLA level at all**. `spmem` appears in these dumps only as
compiler flags (`spmem=true`, `spmem_bounce_buffer=auto`); the allocation lives inside the Mosaic
custom call. So the dumps confirm the ENTRY-identity claim and simultaneously confirm that
**§7's `V2_BIGSCRATCH` arm is the only way to price the reclamation** — no static artifact can.

---

## 6. The prediction, restated as a number to be scored

I am the harness's known answer, so this section is deliberately over-specified: **two numbers, two
ρ values, and a mechanism that discriminates between them.**

### The number of record (Plan A §5, unchanged)

| | value |
| --- | ---: |
| harness baseline at ρ = 0.40005 | 189.98 µs |
| predicted | 177.79 µs |
| **Δ time %, harness** | **−6.42%** |
| **Δ time %, production** (÷ 155.82) | **−7.83%** |

Mechanism: deletes 15/16 of a dense 256-bucket histogram clear plus a 16-deep fold — a
**ρ-independent** term, which is why the prize is larger at production ρ than at the instrument's
ρ = 1. It rests on a single measured quantity, **12.195 µs** (`j-300d4690`).

### Why that is a **floor**, and what I actually predict

`j-300d4690` measured 12.195 µs on a kernel with `REFINEMENT_LEVELS = ()` (`arms/arm-R1.py:48` and
`arm-B.py:48`). **`zero_priv` and `fold_priv` run once per *executed histogram level*, not once per
kernel call** — L0 plus every unconverged refinement level — and the production kernel has three
refinement levels. So 12.195 µs is a per-level saving that the design applied once.

Two independent numbers say the sweep costs the same at every level, which is what makes the scaling
legitimate: `probes/09` puts the **L0** sweep at **18.866 µs**, and `probes/07` puts the refinement
sweep at **42.81 µs over 2.231 levels = 19.19 µs/level**. Those agree to **1.7%**.

Executed-level counts, critical-subcore aggregation (`probes/13`; the aggregation is `max` over the
32 subcores of each subcore's own row-sum, because there is no barrier):

| ρ | refinement levels | total histogram levels |
| --- | ---: | ---: |
| 1 (the instrument) | 2.03 | **3.03** |
| 0.40005 (production prefill) | 1.570 | **2.570** |

### The two numbers to score

**Score the absolute µs delta first; the percentage second.** At production ρ the denominator is
itself modelled — nothing in this exercise has ever run at ρ = 0.40005 — so the percentage moves when
the measurer's own incumbent arm lands.

| ρ | predicted Δ µs | incumbent denominator | **Δ time %** |
| --- | ---: | --- | ---: |
| **1**, b=1024, full `REFINEMENT_LEVELS` | **−36.9** (= 12.195 × 3.03) | **≈ 454.6 µs**, all measured: 327.166 L0 base + 84.79 refinement walk + 42.81 refinement sweep (`hardware_map.md` stage C) | **−8.1%** |
| **0.40005**, b=1024 | **−29.5** (= 12.195 × [0.848 + 1.570]; the 0.848 is the sheet's own L0 gating factor 16.00/18.866) | ledger-as-published **189.98 µs** harness / 155.82 production; **corrected for §9-1, ≈ 220.1 / 186.0** | **−13.4%** harness, **−15.9%** production (or −15.5% / −18.9% against the uncorrected sheet) |

### The discriminator — this is the part that makes a disagreement legible

If the R=1 saving really is L0-only, the ρ = 1 arm reads **−12.2 µs = −2.7%**, not −8.1%. The two
mechanisms are **3× apart at ρ = 1** on the same arm, in the same job, against a denominator the
measurer is measuring anyway. So:

- **≈ −8% at ρ = 1** → the saving scales with executed levels; the production number is −13% to −16%
  and Plan A's −6.42% was a floor by ~2.3×.
- **≈ −2.7% at ρ = 1** → the saving is L0-only; my scaling is wrong, Plan A's mechanism is right, and
  the production number is −6.42% / −7.83%.
- **Anything near −3.73%** would mean the measurer's incumbent arm is the L0-only kernel, not the
  production one — that is where `design.md`'s "−3.73% at ρ = 1" comes from (12.195/327.166), and it
  is a number **the measurer cannot read** against a full-refinement incumbent. If it appears, check
  the arm, not the kernel.

---

## 7. How the measurer runs me

**Map.** `variants/v02-r1-shared-hist/sparsecore_topk.py` → `src/vllm_torchtpu/kernels/deepseek_v4/
sparsecore_topk.py`, via `cdk job create --map-dir`. **In-job md5 gate: `c0ca54a66a549020cfc584e2d8d
97abe`** on the mapped path, printed before the first timing and compared in-job — an image tag is
mutable and a recipe read-back does not detect it.

**Shapes.** `--ctx 9216` (n = 10240; the bench default 132096 is the wrong shape), `b ∈ {64, 1024}`,
k = 2048, P = 1 at both. No two-stage path is exercised and none needs to be: `v2` touches only
stage-1 kernel internals.

**Checksum arm: not needed.** `v2` changes no signature, no output dtype or shape, no DMA extent and
no `out_hbm` write pattern. The optimized HLO ENTRY is byte-identical to the incumbent's (§5), so
what the DMA lands is unchanged by construction.

**⚠ The frozen bench cannot run me at production ρ.** `sctopk_2stage_bench.py`'s timing entry points
hard-set full rows — `ab()` uses `rl = np.full(b, n)` and `decompose()` likewise — and there is no
`--rho` or row-lengths flag. Ragged lengths appear only on the correctness paths (`gate()`,
`variant_equality()`, via `_row_lengths`). I am forbidden from building an arm, so I hand over the
requirement: **the ρ = 0.40005 arm has to be a `probes/08`-style driver that passes an explicit
ragged `row_lengths` into `sparsecore_topk(...)` and times it through the frozen
`glm_dsa_block_bench.measure` protocol.** `probes/08/probe.py:389-396` is the shape of it. This is
the arm `design.md:553` already says step 3 should buy first, and §6's second row cannot be scored
without it.

**Arms, minimum viable block (all in one job, interleaved, `us_median`, against a ≥3-null spread):**

| arm | source | what it answers |
| --- | --- | --- |
| `INC` | incumbent, md5 `44353b34…` | the denominator, measured rather than modelled |
| `V2` | `sparsecore_topk.py`, md5 `c0ca54a6…` | the candidate |

at **ρ = 1 and ρ = 0.40005**, b ∈ {64, 1024} — four cells. The ρ = 1 cell is the §6 discriminator and
is the cheaper of the two; run it even if the ρ arm slips.

**One rival arm my build earns, and it is not a compile-time constant of mine but an allocation.**
`probes/08`'s `arm-R1` never shrank `scratch_types`, so the 15,360 B/subcore of tile memory `v2`
hands back has **never been measured**. I have generated the isolator:

| arm | source | md5 | difference from `V2` |
| --- | --- | --- | --- |
| `V2_BIGSCRATCH` | `sparsecore_topk_bigscratch.py` | `70f9873e2941bd94601c15cea1dd9d55` | **one line**: `scratch_types` back at `LANES * NUM_BUCKETS + LANES`. §5 shows it compiles to an **identical jaxpr equation count** (1,013 / 50 adds) — the instruction stream is the same and only the allocation differs. |

`V2 − V2_BIGSCRATCH` is therefore the tile-memory term alone, with the instruction count held fixed.
I predict **0.00%** and say so under §10 as an `unmodelled` term that cannot fail; the arm exists
because at 512 KiB per subcore and a 40 KiB resident slice, 15 KiB is 2.9% of the tile, and it will
matter more at larger `slice_len` than at this one. **Drop this arm before dropping either ρ cell** —
but note it is now the *only* remaining route to that number: §5 shows the reclamation is invisible
in the XLA buffer-assignment and memory-usage dumps, which are byte-identical between the arms.

---

## 8. How the correctness runner runs me

Same `--map-dir` and the same in-job md5 gate `c0ca54a66a549020cfc584e2d8d97abe`.

```
python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

Unmodified frozen test, **all 68 ids, no `-k`**, no tolerance change, no reference change. Baseline
to beat: 68/68 PASSED (`j-62275e52`). `instruments/README.md` lists three harness prints that are
traps; ignore them as it says.

`v2` is a pure-integer change with no reassociation and no reordering of the emit, so the correct
result is **68/68 and bit-identical output** — anything else is a red, not a tolerance question.

**The five ids that exercise this change hardest**, in order, with why (densities from
`collision_census.txt`):

1. `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` — `heavy_ties` is 17 integer values, which
   `monotone_key` maps into **only 4 occupied L0 buckets**: the maximum possible intra-vector
   collision density, **8.36 M colliding lane pairs**, at the 32-wave production prefill shape. If
   `addupdate_scatter`'s conflict resolution has any count dependence, this is where it shows.
2. `test_exact_topk_full_rows[heavy_ties-p1-b64-n10240]` — the same density at the 2-wave decode
   shape, which is the *shape* half of the risk row.
3. `test_exact_topk_full_rows[strict_desc-p1-b1024-n10240]` and `[strict_desc-p1-b64-n10240]` — **the
   negative-heavy input Plan A §7 asks for.** `strict_desc` is `-np.arange(n)`, so every score is
   non-positive, `monotone_key` fires on every key *including every winner*, and there are 9.83 M
   colliding pairs across 9 buckets. These two ids are the reason the §9-2 gap is narrower than
   `design.md` §8 records.
4. `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240]` — the only P=1 test
   with ρ < 1. It includes rows with `eff_len = 0` (the `edges` list starts at 0), which is the sole
   exerciser of the `i_hi == 0` / `zg` branch beside my shrunk `priv_vmem`, and rows with
   `eff_len ≤ k` where `keep_all` fires, which is the sole exerciser of the **dynamic-zero** form of
   hunk 5, `zero_priv(jnp.where(done, 0, …))` with the bound resolving to 0.
5. Any `test_exact_topk_full_rows[random-p4-…]` — these run the refinement scatter at `:238` under a
   sparse `match` mask through the two-stage path, which no probe has ever executed under `R=1`.

**If one goes red, look at hunk 5 first** (`:224`): it is the edit Plan A did not specify, it is the
only one with a dynamic bound, and it is the only place where a wrong bound writes outside
`priv_vmem`.

---

## 9. Things I found that are not my candidate

**9-1. Ledger error: the production sheet's "V4 sweeps" row prices level 0 only, so 189.98 µs
under-counts the incumbent by ~30 µs at production ρ.** ⛔ Not a small bookkeeping point — it is the
denominator of every Plan in `design.md` §9.

- Every arm in `probes/09` carries `REFINEMENT_LEVELS = ()` (`probe.py:44` states it; `arms/*.py:48`
  confirms it in all twelve). So its `E0 − E0-swp = 18.866 µs` is the **level-0** sweep. Same for
  `probes/08`'s `B = 327.166` (`arms/arm-B.py:48`).
- `hardware_map.md:490` nonetheless labels the row **"V4 sweeps, all levels | 18.866"**, and the
  ρ = 0.40005 sheet at `:531` carries it forward as **"V4 sweeps | 16.00 | no, but gated"** — with no
  refinement-level term anywhere. The refinement sweep is a *separate*, independently measured
  **42.81 µs over 2.231 levels** (`probes/07`, `j-098b260f`, `hardware_map.md:366`), and stage C″
  never picks it up. The sheet's neighbouring refinement-`hist` row *is* carried (21.49 µs), so the
  omission is specific to the sweep.
- **Consequence:** the harness incumbent at production ρ is ≈ **220.1 µs**, not 189.98; production
  ≈ 186.0, not 155.82; and every `Δ time %` in `design.md` §9 that divides by 189.98 is inflated by
  ~16%. Rankings between plans are mostly preserved (a common denominator), but **`v2`'s own number
  is affected twice** — numerator and denominator — which is §6.
- **Route:** the measurer settles it for free. Its very first job measures an incumbent arm at
  ρ = 0.40005; that absolute is the sheet's total, and it will read ≈ 220 or ≈ 190. Hand this note to
  whoever writes `results.md`, and correct `hardware_map.md:490` (drop "all levels") either way.

**9-2. `all_negative` has no P=1 parametrization, so no shipping-shape test targets `monotone_key`
directly — but `strict_desc` covers it, and `design.md` §8's gap is narrower than recorded at P=1.**
`test_emit_pairs_is_bit_identical_to_the_incumbent` is the only test using `_scores("all_negative")`
and it is parametrized over `TWO_STAGE_SHAPES` (`:290`, `P > 1`) only — and `P = 1` is both shipping
shapes. However `strict_desc` **is** parametrized over all of `SHAPES` including `p1-b64-n10240` and
`p1-b1024-n10240`, and it is entirely non-positive, so `monotone_key` fires on every key and every
winner at P=1 already. **Route:** two parts. (i) Tell step 4 that the live gap is `all_negative` at
`P=1` specifically — a *continuous* all-negative row, where `strict_desc`'s integer spacing might
mask a bit-level slip — and that closing it means adding `SHAPES` to that test's parametrize, which
is a frozen-instrument edit and therefore a step-4 decision, not a step-3 one. (ii) Tell the
correctness runner that `strict_desc-p1-*` is the negative-heavy case Plan A §7 was asking for, so
that row is discharged rather than left open for want of `all_negative`.

**9-3. `design.md` §9 Plan A §2's edit list is incomplete, and following it literally corrupts
memory.** §1(d) above. Three edits are listed; four are required once the `v2` node's 1 KiB
allocation is applied, and the missing one (`:224`) writes 4096 words into a 272-word ref. **Route:**
correct Plan A §2 to four edits before anyone else lifts it. Any other plan that shrinks `priv_vmem`
inherits the same trap.

**9-4. A platform fact that belongs in `references/08-sparsecore.md`, and it resolves a
self-contradiction already in that file.** The file's primitive table (`:541`) says
`addupdate_scatter(ref, indices, x, *, mask)` does `ref[indices] += x` **"atomically"**, while its
`scan_count` section (`:646-649`) says "a scatter-add of ones into a counter array **silently drops
duplicates that land in the same 16-lane vector — the lanes race**". A reader building a histogram
gets opposite instructions from the same file. `j-300d4690` plus this folder's census resolve it:
**`plsc.addupdate_scatter` does resolve intra-vector address conflicts, bit-exactly, at ≥ 6.73 M
colliding lane pairs over 100% of vectors** — so the `scan_count` idiom is *not* required for a
counting histogram built with `addupdate_scatter`. What `scan_count` remains required for is
unchanged and should stay: **placement** (`base + run − 1`), and any scatter that is not an
accumulating one. `design.md:697` already flags this as a `references/` promotion owed; this is the
wording and the two line numbers to fix. **Route:** `references/08-sparsecore.md`, owned by whoever
closes the exercise.

**9-5. `arm-R1`'s 12.195 µs contains no tile-memory effect.** §1(c). It is not a defect in the probe
— it measured what it set out to — but it means the `v2` node's headline "live set 72,192 →
57,152 B, V4 falls from 16 KiB to 1 KiB" is an **unmeasured** claim riding on a measured one.
**Route:** the `V2_BIGSCRATCH` arm in §7, one job, one extra cell.

**9-6. The frozen bench has no ρ knob on any timing path.** §7. **Route:** stated there as a
requirement handed to the measurer; I did not build it, per BRIEF §2.

**9-7. Three additions to the "AOT-compile SparseCore Pallas with no TPU" technique, all of which
return a confident wrong answer rather than an error.** Earned in this build (the first two came to
me from a sibling variant agent via the coordinator; I applied and confirmed both here).

- **A two-arm AOT A/B in one process can compare an arm with itself.** `jax.jit` keys on
  `(function object, avals)`, so rebinding a module attribute between arms returns the first arm's
  executable for the second, byte-identically and silently. Loading each arm as its own module object
  avoids it; *proving* you avoided it needs one arm per process, variant first, with the source md5
  and `__code__.co_filename` printed before the compile. §5 has the worked form.
- **The negative control must be refused by the SparseCore allocator, not the MLIR layer.** The
  `o[...] = jnp.zeros_like(o)` form raises `NotImplementedError: Swap does not support storing to
  HBM` and is not a control. The working form is an 8 MiB VMEM scratch DMA'd to HBM, asserted on
  `E3000` **and** `spmem` **and** `SparseCoreAllocationFailure`.
- **`XLA_FLAGS=--xla_dump_to=<dir>` works on a host-only AOT compile** and yields
  `*.sparse_core_specific_metadata.txt` (per-computation SEQUENCER / TILE_EXECUTE_CORE /
  TILE_ACCESS_CORE **bundle counts**), `*-buffer-assignment.txt` and `*-memory-usage-report.txt` —
  none of which are in `compiled.as_text()`. Caveat that belongs with it: bundle counts are **static
  code size**, and the SparseCore tile-memory (`spmem`) allocation is **not** in the memory-usage
  report, because it lives inside the Mosaic custom call.

**Route:** all three belong in `references/08-sparsecore.md` beside the existing AOT note, alongside
9-4. `aot_single_arm.py` in this folder is the reusable form.

---

## 10. Open terms

Stated as the brief requires: a term tagged `unmodelled`, predicted at zero, or credited as a
*saving* **cannot fail**, and is recorded as `unmodelled`, not as a pass.

| term | status | why it matters |
| --- | --- | --- |
| **The 12.195 µs saving scales with executed histogram levels** (§6) | `derived`, **not measured**. Corroborated by two independent per-level sweep prices agreeing to 1.7% (18.866 L0 vs 19.19/level refinement), and by the source fact that `zero_priv`/`fold_priv` are the same code with the same static bound at every level. Still an inference. | It is the whole difference between −6.42% and −13.4% at production ρ. It is credited as a *saving*, so **it cannot fail** — the ρ = 1 discriminator in §6 is the only thing that can settle it, and until it lands this is `unmodelled`. |
| **The production denominator** (189.98 / 155.82, or 220.1 / 186.0 after §9-1) | entirely `modelled`. Nothing in this exercise has ever run at ρ = 0.40005. | Both of §6's percentages are ratios of two predictions until the measurer's incumbent arm lands. Score the **µs delta**, not the percentage. |
| **15,360 B/subcore of tile memory handed back** | predicted at **zero** effect on time. **Cannot fail.** Recorded as `unmodelled`. | Never measured — `arm-R1` did not shrink the allocation, and §5 shows the XLA dumps cannot see it either (buffer-assignment and memory-usage are byte-identical between arms). §7's `V2_BIGSCRATCH` is the only thing that converts it into something that can fail. |
| **−249 TILE_EXECUTE_CORE bundles (−21.3%)** | `measured`, but of **static code size**, not time. Deliberately **not** used in §6. | It confirms the mechanism landed and is confined to one computation. It is not a speed prediction: the bundle count is what was emitted, not what executes, and the loop trip counts are dynamic. Anyone converting −21.3% of bundles into −21.3% of time is reading it wrong. |
| **The refinement-level scatter under `R=1`** (`:238`) | `assumed` correct, by an argument (sparser mask ⇒ strictly fewer conflicts than L0, which is measured bit-exact) rather than a reading. Never executed. | A correctness term. Closed by §8 ids 1-5; open until then. |
| **`heavy_ties` under `R=1`** (4 occupied buckets, 8.36 M colliding pairs) | `assumed` — the highest conflict density in the suite and the probe never ran it. | Same. §8 id 1. |
| **The `0.848` L0 gating factor at production ρ** | `derived` from the sheet's own 16.00/18.866, and inherited with whatever error §9-1 carries. | Worth 1.8 µs of §6's production prediction; small, and it makes the number conservative. |
| **Composition with `v3` (Plan B)** | `assumed` independent, on a textual-adjacency argument. Untested. | §4 row 2. Step 4. |
| **Composition with `v5`/`v7` (Plan D)** | **known non-additive**, direction predicted (`v2` should make them *less* attractive), magnitude `unmodelled`. | §4 row 2. |
| **`code-style.md` item 15** (hardware constants hardcoded) | a blocking `❌`, inherited verbatim, identical in the control. | §3. It cannot move an A/B, and it is not fixed here. Route given. |
