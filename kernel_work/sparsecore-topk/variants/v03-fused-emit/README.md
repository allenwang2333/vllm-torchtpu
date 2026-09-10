# `v03-fused-emit` — Plan B, "fused emit, no second walk of `L`"

**Verdict up front.** The variant is built, compiles AOT for every shape the frozen suite reaches,
and is **exact** — 960/960 simulated rows return the incumbent's winner set bit for bit. But the
plan's `−20.5%` cannot happen, for a reason that is structural rather than a tuning miss, and the
honest re-price is a **loss**: **+1.85% … +6.27% (harness) / +2.27% … +7.71% (production)**, `modelled`.
The variant should still be measured — a modelled loss is not a measured loss — but it should not
be measured *first*.

Two findings are worth more than the arm:

1. **The plan's `cnt_max = 4096` "safe bound" is false**: `heavy_ties` drives `cnt` to **5526** at
   the P=1 production shape, so a variant built to the design's stated bound with
   `disable_bounds_checks=True` writes out of bounds. This variant sizes at `slice_len + LANES`, so
   overflow is impossible by construction. **The suite catches it, but by only 1.35× and on one
   score kind** — and the kind with the *highest* `cnt`, `all_negative`, never runs at P=1 at all.
   §4 row 2, §9 row 2.
2. **`v3`'s prize was computed against a pass that cannot exist.** The classification needs
   `b_star`, and `b_star` is the *output* of the level-0 histogram, so the two stores cannot ride
   that pass. They ride a full-length walk instead. That single ordering fact moves the candidate
   from rank 1 to a predicted loss. §6.
3. **The hazard the dispatch flagged is measured at zero, and the one it did not flag is confirmed.**
   The L0 `addupdate_scatter` region is bit-identical in the scheduled VLIW bundles — same blocks,
   same opcode sequences, 13 scatter instructions in both arms — so the collision nonlinearity
   cannot bite. What does bite is `store_compressed`, and the bundle dump brackets the two extra
   ones at **1.21–1.73× the pass they sit on**, which contains my model's 1.49× and excludes the
   1.0× the −20.5% assumed. Two instruments, no shared inputs. §5, §6.

---

## 1. What was built

`variants/v03-fused-emit/sparsecore_topk.py`, edited in place from the verbatim incumbent copy.

| | |
| --- | --- |
| incumbent | `44353b347c96e0eedbd742c710d4540a` (commit `b11b6eb6`) |
| variant | `7309435f82ab4b273abe22cbf8fcf502` |
| kernel scope name | `sc_topk_direct_b1024_n10240_k2048_v3fused` (`_keys` suffix on the P>1 entry point) |
| changed regions | 5 — module docstring, `_topk_body` signature, the post-L0 body, the two entry points, `_sc_topk_direct` scratch + guards |

### The structure, and the ordering that forces it

The pivot bucket `b_star` is produced by folding and prefix-summing the level-0 histogram and
searching it. **Nothing in the level-0 walk knows it.** So the fused classification cannot be an
extra pair of stores inside `hist0_body`; it has to be its own pass, which is exactly what the `v3`
version-log's phrasing ("one pass after L0") says and what `design.md` §9 Plan B §1/§2 ("as it
scatters, it also writes…", `_l0_fused`) says it isn't. I built the implementable reading.

Per row, after the level-0 histogram:

```
split_body   over L        1 load  + ALU + 3 store_compressed + 2 all_reduce      NEW
  rowbuf    <- j            mask = strict0            certain winners, emitted here
  strad_key <- key          mask = tie0               what refinement histograms
  strad_idx <- j            mask = tie0               what the final emit writes
ref_body     over cnt      x 1.570 crit levels        bound walk_chunks -> cdiv(cnt,16)   EDIT
emit_body    over cnt      2 loads + ALU + cumsum + all_reduce + 1 store           RETARGETED
```

**What is known when `split_body` emits, and what can still change it.** At the moment of the
store, `prefix_l0 = b_star - 128` and `shift = 24` are both final. Refinement only ever appends bits
*below* bit 24 (`prefix <<= width; prefix |= b2`, `shift` descends 24 → 16 → 8 → 0), so for any later
prefix `P` at shift `s`:

* `key >> 24 > prefix_l0`  ⟹ `key >> s > P`  — a certain winner stays a winner;
* `key >> 24 < prefix_l0`  ⟹ `key >> s < P`  — a certain loser stays a loser;
* `key >> 24 == prefix_l0` — undecided, and *only* these enter the straddler buffer.

So the level-0 class of an element is immutable and the single walk is final. Proved in the source
comment above `prefix_l0`, and checked exhaustively by `sim.py` (§8).

`done_l0 = keep_all | (quota == cnt)` is captured before the refinement loop because the loop
overwrites `done`; the post-loop `done` means "refinement converged", which is a different
proposition and using it would skip a needed emit.

### Ordering of output slots differs from the incumbent — this matters to the gate

The winner **set** is identical. The **slot order** is not: the incumbent emits in ascending column
index; `v3` emits all certain winners first, then the tie-broken ones. The kernel documents rows as
unordered and `_mismatch` compares score multisets, so the frozen suite is unaffected — but **a
byte-compare of `out_hbm` against the incumbent will fail on a correct `v3`.** The gate must sort
each row (or compare as a set). See §8.

Tie-break order itself is preserved: `store_compressed` appends in lane order and the split walk
runs in increasing `i`, so `strad_idx` is sorted ascending and the running `rank` ranks ties in the
same order the incumbent's single walk did.

---

## 2. The §4 Files plan, scored

| plan said | what happened | score |
| --- | --- | --- |
| `sparsecore_topk.py` holds `_l0_fused` | no `_l0_fused` exists — the pass cannot be fused into L0 (§1). It is `split_body`, a sibling pass. | ✗ **not buildable as written** |
| … the straddler scratch | built, but as **two** buffers (`strad_key_vmem`, `strad_idx_vmem`), not one. Refinement needs the key, the emit needs the index; there is no 32-bit packing of both that preserves exactness. | ✗ 2 buffers, 1 planned |
| … the two bound changes | `walk_chunks` → `cdiv(cnt,16)` on the refinement loop **and** on the emit loop, plus a `< cnt` tail lane mask the plan does not mention (the buffer tail holds the previous row's data). | ✓ + 1 unplanned |
| "the emit traversal it replaces — delete it, do not gate it" | **the emit traversal is not deleted.** `split_body` *is* that traversal, relocated: it still walks all of `L` once. What is deleted is the refinement walks over `L`. The plan's cost model deleted both. | ✗ **the prize term** |
| `~60 lines changed` | 118 changed lines (`diff` `<`/`>` count), of which ~55 are the new pass and ~35 comments/glossary. | ≈ |
| `cnt_max = 4096` "safe bound" | **false.** §4 row 2. | ✗ correctness |

Four of six rows missed, and one of the misses is the mechanism the prize was priced on.

---

## 3. `code-style.md` checklist, item by item

| # | item | v3 | note |
| --: | --- | :-: | --- |
| 1 | Dimension glossary at top | ✓ | added (`b n k P slice_len L LANES cnt`); the incumbent had none |
| 2 | Trailing `# [shape] dtype` on every ref | ✓ | new refs carry `# i32[slice_len + LANES]`, matching the incumbent's style |
| 3 | Extents from refs, not config | ~ | inherited: `b_pad = len_vmem.shape[0]` ✓, but `slice_len`/`k` still arrive as kwargs. Not changed — changing it changes the subject. |
| 4 | jaxtyping on the public wrapper | ✗ | inherited; the wrapper uses `jax.Array # f32[b, n]` comments. Not a blocking class. |
| 5 | `# Inputs. / # Outputs. / # Scratch.` sections | ✗ | inherited; Pallas's positional inputs→outputs→scratch order is documented in a comment above `_topk_kernel` instead. |
| 6 | Body order indices→shapes→sems→guard→loads→compute→masked store→epilogue | ✓ | preserved; `fill` was moved *above* `split_body` so the `-1` fill cannot overwrite emitted winners |
| 7 | Single characters only where allowed | ✓ | `i`, `j`, `w` loop/column/wave; everything new spelled out |
| 8 | Memory space in the ref name | ✓ | `strad_key_vmem`, `strad_idx_vmem` |
| 9 | Comments name the compromise | ✓ | the capacity argument, the ordering argument and the slot-order caveat are all in-source; **no microbench number is committed** (per BRIEF §7) |
| 10 | `NotImplementedError`, not `ValueError` | ✓ | both new guards raise `NotImplementedError` |
| 11 | Four-file decomposition | ✗ | inherited single file. Splitting it would make the A/B unreadable. |
| 12 | Split boundary = what a module may touch | n/a | single file |
| 13 | Frozen kw_only slots pydantic config | n/a | no config class |
| 14 | **Tested against a reference, tolerance visible at the call site** | ✓ | `tests/kernels/test_dsv4_sparsecore_topk.py`, 68 tests, `_mismatch` asserts **exactly 0.0** score difference — the tolerance is literal and at the call site. §8. |
| 15 | `Precision.HIGHEST` on every f32 contraction in `_fwd` | n/a | integer kernel, no contraction |
| 16 | **Hardware constants from `pltpu.get_tpu_info()`** | ✓ | `sc.num_lanes` is now checked against `LANES` and `sc.vmem_capacity_bytes` bounds the scratch; both raise `NotImplementedError`. The incumbent read neither. Host-side, inside the existing `get_tpu_info()` call, zero device cost. |

**Both blocking classes pass** (14 and 16). Items 4/5/11 are pre-existing incumbent shape that I
deliberately did not refactor: BRIEF §5 says build the plan, not improve the file, and a four-file
split would confound the measurement.

---

## 4. The §7 risk table, with an outcome on every row

| risk (design.md §9 Plan B §7) | outcome |
| --- | --- |
| **"the added store in L0 costs more than the walks it deletes"** — threatens *speed* | ⚠️ **HAPPENED, and worse than the row imagines.** There is no store in L0; the stores land on a full-length walk. Two extra `store_compressed` at the measured `6.675 µs/pass` cost **5.34 µs** at production ρ, against **2.74 µs** of walk saved. The trade is negative *before* any second-order term. The row's fallback — "fall back to fusing only `emit`, leaving refinement alone" — is what the variant already is, and it is the losing side. §6. <br>The row's stated signal was "L0's own arm time rises in the same job". **That signal will read clean and it will be misleading**: the scheduled bundles show L0 is bit-identical between arms (§5), so its arm time cannot rise. The cost is on a *different* pass. A de-contaminated single-region L0 arm would have exonerated the change. |
| **"`cnt` exceeds `cnt_max` → out-of-bounds write"** — threatens *correctness* | ✅ **REMOVED BY CONSTRUCTION, and the plan's bound was wrong.** `cnt` counts live elements of `L` in one bucket, so `cnt ≤ i_hi ≤ slice_len` unconditionally; the buffers are `slice_len + LANES` (the `+LANES` is `store_compressed`'s 16-wide window). No in-kernel assert is needed because no distribution can reach the bound. <br>**The plan's `4096` is falsified.** `sim.py`, now using the score generators transcribed **verbatim** from `_scores`, at ρ=1 (which is what `test_exact_topk_full_rows` sets, `row_lengths = n`):<br>`strict_desc` 1536 · `random` 3054 · **`heavy_ties` 5526** · `all_negative` 7089 · (`few_distinct`, mine, 10240).<br>**`heavy_ties` at 5526 overflows a 4096-word buffer by 1.35×, and `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` is a real collected test id** (verified by `pytest --collect-only`, not by reading). So the suite *would* have caught the plan's bound — but only by 1.35×, and only on one kind. |
| **"exactness claim rests on the vacuous gate"** — threatens *correctness* | ✅ **DISCHARGED, by argument and by simulation, not by the gate.** The straddler set `{key>>24 == b_star_key}` is a superset of everything any deeper prefix can select (§1), so the partition and the tie-break order are unchanged. `sim.py` replays both control flows on the host: **960/960 rows** set-identical to the incumbent *and* multiset-identical to a true `numpy` top-k, across `strict_desc`, `random`, `heavy_ties`, `all_negative` and a 4-distinct adversarial kind, at ρ ∈ {1.0, 0.40005}. Re-run after transcribing `_scores` verbatim: still **0 mismatches**. `monotone_key` is exercised on every element of `strict_desc` (`-np.arange`, entirely ≤ 0) and `all_negative`. |

**Which assertion would have fired** had the buffer overflowed (BRIEF's requirement). With
`disable_bounds_checks=True`, `store_compressed` past the end of `strad_key_vmem` walks into
`strad_idx_vmem` and then `len_vmem`; corrupting `len_vmem` corrupts `eff_len` for later rows in the
same subcore. `_mismatch` would return, in decreasing likelihood,
`"row i: index N at or beyond row_length"`,
`"row i: returned N valid entries, expected M"`,
`"row i: max score difference …, expected exactly 0.0"`, or `"duplicate indices"`, under
`test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` — the collected id that maximises `cnt` at
the P=1 production shape — and, at P=4,
`test_emit_pairs_is_bit_identical_to_the_incumbent[all_negative-p4-b1024-n133120]`.

---

## 5. Evidence: `static-compile`

No device time was used. Everything below is host AOT against `tpu7x:2x2x1`.

### Controls — a matched negative *and* positive pair (`vmemctl.py`)

```
POSITIVE  4 KiB   want=COMPILE  got=COMPILE  OK    6797 chars of HLO
NEGATIVE  8 MiB   want=FAIL     got=FAIL     OK
  E3000 CompileTimeSparseCoreAllocationFailure ... memory space spmem
  upper bound (33554431 words)   [allocatable: 2097151]
```

Same kernel, same `make_async_copy` VMEM→HBM form, same mesh; **the allocation size is the only
variable between the compile and the refusal.** The negative alone would have been equally
consistent with the DMA form being broken for an unrelated reason. The refusal comes from the
**SparseCore allocator**, not from an MLIR `NotImplementedError` — every green below is a real
Mosaic-SC compile.

### Arm identity — three independent proofs, so no jit-cache aliasing

The two arms are separate `importlib` modules, but that alone would not be evidence. Each arm was
**also** compiled in its own fresh process (`aot_one.py`, **variant first**), asserting source path,
source md5, and the kernel name that lands in the HLO:

| | base | v3 |
| --- | --- | --- |
| source md5 | `44353b34…4540a` | `7309435f…f502` |
| kernel name in optimized HLO | `sc_topk_direct_b1024_n10240_k2048` | `…_k2048_v3fused` |
| Mosaic `func.func` scratch operands | **6** | **8** |
| Mosaic module | 2262 lines | 2398 lines |

Different names and different arities: the arms cannot be the same executable.

### Compiles clean at every reachable shape

| shape | base | v3 |
| --- | --- | --- |
| prefill `b=1024 n=10240 k=2048` P=1 | ok | **ok** |
| the same, `emit_keys` | ok | **ok** |
| decode bucket `b=64` | ok | **ok** |
| two-stage `slice_len=33280` (`n=133120` P=4), `emit_keys` | ok | **ok** |

`slice_len=33280` is the largest slice the frozen suite reaches, and it is the one that stresses the
new scratch: 113,520 words = 454 KiB of the 512 KiB tile. It fits.

### SparseCore private VMEM — priced by the allocator, not by the memory-usage report

⛔ **`*-memory-usage-report.txt` cannot see this quantity and will silently agree with you.** Mosaic
allocates SparseCore scratch *below* XLA, so that report's `vmem` table carries XLA-level buffers
only. I checked: it is **byte-identical between my two arms**, which differ by 20,512 words of
kernel scratch. Any nonzero `vmem` total in it is an XLA-level `S(1)` staging buffer — a different
quantity. Cited as evidence it would have been a fabrication.

The one host-only instrument that *can* see it is the allocator. Sweeping `n` (P=1) to the first
refusal, per arm:

| | largest `n` that compiles | first refusal | refused by |
| --- | ---: | ---: | --- |
| base | **123,904** | 124,928 | `E3000` SparseCore allocator |
| `v3` | **40,960** | 41,984 | this variant's own `NotImplementedError` guard |

**Measured scratch ratio 123904 / 40960 = 3.025×**, against the 3.000× predicted by "one resident
word per element becomes three". And the declared model lands the base threshold to **0.23%**
(`123904 + 4112 + 272 + 2064 + 1024 = 131,376` words vs the 131,072-word tile that
`sc.vmem_capacity_bytes = 524288` implies). So the declared per-buffer table below is corroborated
by an independent measurement, rather than being a read-back of my own declaration:

| buffer | base | v3 |
| --- | ---: | ---: |
| `resident` | 10240 | 10240 |
| `priv` / `glob` / `rowbuf` | 4112 / 272 / 2064 | 4112 / 272 / 2064 |
| `strad_key` / `strad_idx` | — | **10256 / 10256** |
| `len` | 1024 | 1024 |
| **total at the production shape** | **17,712 words = 70,848 B (13.51%)** | **38,224 words = 152,896 B (29.16%)** |

Two consequences, one of them a real functional narrowing:

* The guard is calibrated to within one sweep step (1024 words) of the true capacity, and it fires
  **before** the allocator — so an over-large shape is a clean `NotImplementedError`, never a
  silent `E3000` deep in the compiler.
* ⚠️ **`v3` narrows the maximum supported `slice_len` from ~123,904 to 40,960.** `MAX_SLICE_WORDS`
  is still `65536`, so `slice_len ∈ (40960, 65536]` now raises where the incumbent compiled. Not
  reachable from production (`slice_len = 10240`) or from the suite (largest **33,280**, 18.9%
  headroom), but it is a narrowing and it should be stated rather than discovered.
  **The fix is verified and P-neutral:** lowering `MAX_SLICE_WORDS` to `40960` changes `_pick_partition`'s
  answer for **0 of 216** (n, b, k) combinations built from every `n` in the frozen test file — in
  particular `n=133120` still gives `P=4`, so `test_partition_factor_is_what_these_tests_assume`
  still passes. I did **not** apply it: it is not part of the plan, it is invisible at every measured
  shape, and it would change `P` for untested `n ∈ (81920, 131072]`. Routed in §9.

### Mosaic MLIR — the L0 histogram is provably untouched

The dispatch's specific warning was that the fused pass makes the level-0 walk do more per element,
and that `addupdate_scatter` collision behaviour is nonlinear. **It does not, and here is the proof.**
Each `scf.for` body was extracted and fingerprinted after SSA renumbering:

| loop | base md5 | v3 md5 | |
| --- | --- | --- | --- |
| `cvt` | `8d6b47d2ff` | `8d6b47d2ff` | identical |
| `zero_priv` | `6e0d7570f8` | `6e0d7570f8` | identical |
| **`hist0` (L0 `addupdate_scatter`)** | **`d826d61763`** | **`d826d61763`** | **identical** |
| `fold_priv` | `0760469177` | `0760469177` | identical |
| `zg` (glob zero) | `72dd903c15` | `72dd903c15` | identical |
| `scan_glob` | `b1f1718099` | `b1f1718099` | identical |
| `find_bucket` | `06476957c1` | `06476957c1` | identical |
| `fill` | `1c45486acf` | `1c45486acf` | identical |
| `split` (new) | — | `4117585791` (35 L) | STORE ×3, AR ×2, **no SCATTER, no SCAN** |
| refinement ×3 | 21 / 22 / 21 L | **19 / 20 / 19 L** | shorter bodies, shorter bounds |
| emit | `d9628a9b54` (36 L) | `0e11da7613` (36 L) | retargeted to the compact buffers |

Op census across the whole module:

| op | base | v3 | |
| --- | ---: | ---: | --- |
| `tpu.vector_store_idx` (`addupdate_scatter`) | **4** | **4** | **unchanged — hist0 + 3 levels** |
| `tpu.vector_store` (incl. `store_compressed`) | 16 | 19 | **+3, exactly the three new stores** |
| `tpu.all_reduce` | 5 | 7 | +2 (`split`'s two popcounts, −1 from the moved emit) |
| `tpu.scan` (`cumsum`) | 5 | 5 | unchanged |
| `scf.for` | 25 | 26 | +1, the split pass |
| `tpu.enqueue_dma` / `wait_dma` | 3 / 3 | 3 / 3 | unchanged |

**So the term the dispatch flagged cannot move.** The L0 scatter's instruction sequence, its mask,
its address expression and its bucket distribution are byte-identical; `v5`'s failure mode (delete
sweep work, grow the histogram) has no channel here. The scatter count is 4 in both, so no scatter
was added anywhere.

Two second-order MLIR observations worth recording:

* `split_body` shifts by the **constant** `%cst_7 = dense<24>`; the incumbent's emit shifts by a
  broadcast dynamic scalar `%116`. `v3` saves one `vector.broadcast` per chunk on the L walk.
* The refinement scatter bodies shrank (21→19 lines) because the `valid` mask became a single
  `< cnt` compare instead of the `(j < i_hi) & (key > KEY_NEG_INF)` pair. Per-chunk ALU is *lower*,
  not higher. But see §10 term 2: the same compaction raises the *mask density* those scatters see
  from ~35% to ~100%, and it is precisely the density that `addupdate_scatter` charges for.

### Scheduled VLIW bundles — the dispatch's L0 hazard, priced at exactly zero

`LIBTPU_INIT_ARGS=--xla_sc_dump_bundles_to=… --xla_sc_dump_llvm_ir_to=… --xla_sc_dump_mlir_to=…`
(**not** `XLA_FLAGS`; read once at libtpu load, so set before `import jax`) plus
`XLA_FLAGS=--xla_dump_to=…`. 38 dump files per arm, host-only, seconds. `bundles.py` / `bdiff.py` /
`bloc.py`. The summary count also appears in `sparse_core_specific_metadata.txt`, which is the
cheaper first look (~90 lines) and agrees: `TILE_EXECUTE_CORE_SEQUENCER` 1168 → 1370, same +202.

**Per section of the scheduled TEC program:**

| section | base | v3 | |
| --- | ---: | ---: | --- |
| `__scs_entry_jumppad` | 8 | 8 | identical |
| `__scs_overlays_trampoline_lowered` | 128 | 128 | identical |
| `called_computation.1_lowered` | 60 | 60 | identical |
| **`execute0_lowered`** | **1157** | **1359** | **+202 (+17.5%)** |
| `_tile_overlayer_lowered` | 11 | 11 | identical |

**The L0 histogram, block for block.** Splitting `execute0_lowered` into basic blocks and
fingerprinting the *opcode sequence* of each, the level-0 region is **the same seven blocks in the
same order with the same opcode sequences**:

```
base pc[96,215):  77138c93e2 74580a5979 641dc2af4c cbef99311e 1c467b04b0 143758a66c fc277543fb
v3   pc[96,215):  77138c93e2 74580a5979 641dc2af4c cbef99311e 1c467b04b0 143758a66c fc277543fb
```

and the `addupdate_scatter` instructions land in the same three blocks with the same multiplicities
(**4 + 8 + 1 = 13 in both arms**), around the same 63-bundle unrolled body. **The term the dispatch
named as most likely to eat the prize is measured at exactly zero**, by an instrument that would
have shown it: `v5`'s failure mode (delete sweep work, grow the histogram) has no channel here,
because `b_star` is the histogram's *output* and nothing about the histogram changed.

**Where the +202 actually is.** Every region is within ±6 bundles of its counterpart except one:

| region | base | v3 |
| --- | ---: | ---: |
| L0 histogram (incl. 63-bundle body) | 123 | 119 |
| **`split` pass — NEW** | **—** | **≈194** |
| refinement level ×3 | ≈209 each | ≈206 each |
| emit pass | ≈121 | ≈120 |

**The entire static growth is the one pass I added.** Nothing else moved.

**Whole-program opcode histogram** — and this is the check against "a toy op count is not the
production op count":

| opcode | base | v3 | Δ | reads as |
| --- | ---: | ---: | ---: | --- |
| `vst.msk` (`store_compressed`) | 13 | 52 | **+39** | **exactly 3 × 13** — three new stores at the same 13 instructions/site as every existing one |
| `vmpcnt.ones.xlane` (`all_reduce_population_count`) | 61 | 87 | **+26** | **exactly 2 × 13** — two new popcounts, same per-site cost |
| `vst.idx.add.s32.msk` (`addupdate_scatter`) | 40 | 44 | +4 | **L0 unchanged at 13**; the +4 is a peel-shape difference in refinement level 3 (base 2+2+4+1, v3 4+8+1) |
| `vld` | 394 | 441 | +47 | the split pass's loads |
| `sld`/`sst` against `smem` (**spill tell**) | **30** | **30** | **0** | **no new spills** |
| 37 opcodes, 385 slots | | | **0** | identical |
| total issue slots | 2809 | 3342 | +533 | |

My MLIR count said "+3 stores, +2 popcounts". The scheduler expands each site 13×, and the deltas
are *exactly* 3×13 and 2×13 — a uniform expansion with **no hidden multiplication**. The MLIR count
was not an undercount, and the check that would have caught one was run.

Fixed overhead unchanged, no spills, and the growth attributable to one named pass: this is as
clean a static attribution as the instrument can give.

---

## 6. The prediction, in both denominators

**Rate: `modelled`.** No arm of this variant has been timed. Every input below is `measured`
elsewhere; the composition is mine.

### Inputs

| symbol | value at production (ρ=0.40005, crit, b=1024) | source |
| --- | ---: | --- |
| `W` read-only walk of `L` | 8.273 µs | `probes/15` `cnt` = 20.681 @ρ=1 |
| `s` one extra `store_compressed`/pass | 2.670 µs | `probes/15` Probe B = 6.675 @ρ=1, **flat in density** |
| `Rlev` one refinement level's `L` walk | 15.204 µs | `probes/07` = 38.004 @ρ=1 |
| levels (crit) | 1.570 | `probes/13` |
| `α = cnt / i_hi` | 0.347 | `probes/13` crit `cnt` 1423 / mean `eff_len` 4096 |
| harness / production denominators | 191.76 / 155.82 µs | BRIEF §7 |

`probes/15`'s ladder closes on itself, which is why I trust it: an emit pass should be a read-only
walk plus one store, `20.681 + 6.675 = 27.356`, against `probes/03`'s measured `emit1 = 27.456`.
**0.4% apart.** Everything else in an emit chunk — the `cumsum`, the `all_reduce`, the ~10 ALU ops —
is free to the resolution of the instrument.

### The accounting

Because `store_compressed` is **flat in mask density**, it is a per-chunk instruction cost, not a
per-element transaction cost. So the whole model is "count walks and stores over `L`":

| | walks of `L` | stores on an `L`-length pass |
| --- | ---: | ---: |
| incumbent, post-L0 | `1.570` (refinement) + `1` (emit) = **2.570** | **1** |
| `v3`, post-L0 | `1` (split) + `1.570·α` (refinement) + `2α` (emit reads 2 buffers) = **2.239** | `3 + α` = **3.347** |

```
Δ = W·(2.239 − 2.570) + s·(3.347 − 1)  =  −2.74  +  +6.27  =  +3.53 µs
```

**The two extra stores cost 2.3× the walk they buy.**

### An independent check on the store price, from the bundle dump

The whole verdict turns on one claim: **a `store_compressed` is a real per-chunk cost, not free.**
That claim comes from `probes/15` (6.675 µs/pass, flat in density). The scheduled bundles test it
with no timing at all. Attributing **every** bundle of `execute0_lowered` to a region:

| region | base | v3 | Δ |
| --- | ---: | ---: | ---: |
| prologue / `cvt` | 105 | 101 | −4 |
| **L0 histogram** | **108** | **108** | **0** |
| sweeps (`fold`/`scan`/`find`) ×4 | 437 | 418 | −19 |
| **`split` — exists only in `v3`** | — | **204** | **+204** |
| refinement ×3 | 339 | 358 | +19 |
| fill + emit | 168 | 170 | +2 |
| **total** | **1157** | **1359** | **+202** |

Fully closed, no residual. **One new region accounts for +204 of the +202; everything else nets −2.**

The comparison that prices the store: the new `split` pass walks `L` exactly once, as the emit pass
does, and differs from it by two extra `store_compressed` and one extra popcount. **204 bundles
against 168** — and the 168 also contains the `fill` pass, so the true `split`-to-`emit` ratio is
*above* 1.21. Bracketing it by how much of the 168 is `fill`:

| if `fill` is … of the 168 | `emit` alone | `split`/`emit` |
| --- | ---: | ---: |
| 0 (all emit) | 168 | 1.21× |
| 40 bundles | 128 | 1.59× |
| 50 bundles | 118 | 1.73× |

**The model's `(W+3s)/(W+s) = 1.49×` sits inside that bracket.** Two instruments sharing no input —
a device timing on a different probe, and a host-only instruction schedule — agree that the ratio is
materially above 1. Had `store_compressed` been free, as the assumption that produced the −20.5%
requires, the bracket would have contained 1.0. It does not contain 1.0 under any split of the 168.

Two caveats, stated rather than buried. **Issue slots are not cycles**, and SparseCore passes are
partly memory-transaction-bound, so none of these ratios is a time ratio — they bound the compute
side and refute "free"; they do not set the µs. And **the bracket is wide** because I could not
separate `fill` from `emit` in the bundle stream with confidence; I have reported the range rather
than picking the flattering end. An earlier draft of this section quoted 1.61× from the
`store_compressed`-tagged blocks only — that was a biased subset and is withdrawn in favour of the
fully-attributed table above.

### The band

| reading | Δ µs | harness (191.76) | production (155.82) |
| --- | ---: | ---: | ---: |
| **Model B**, walk-bound refinement (most favourable to `v3`) | **+3.54** | **+1.85%** | **+2.27%** |
| Model A′, transaction-bound refinement (`R` invariant under compaction) | +12.02 | +6.27% | +7.71% |
| Model B at the cheapest measured store (3.686 µs/pass) | +0.73 | +0.38% | +0.90% |
| Model B at the dearest measured store (8.632 µs/pass) | +5.38 | +2.80% | +3.45% |
| ⚠️ `probes/09` emit **region** price (83.830 µs) instead of the per-pass ladder | +19.24 | +10.03% | +12.35% |

⚠️ **The last row should be read as a sensitivity, not as a candidate value.** `probes/18` has since
shown `probes/09` does not reproduce (§9 row 1). It is kept because it is the *most adverse*
internally-consistent reading and every reading is a loss — deleting it would narrow the band in
`v3`'s favour on the strength of a probe being unreliable, which is not a reason to prefer the other
end.

**Every internally consistent reading of the measured price sheet puts `v3` slower.** The single
reading that made it faster assumed the two stores were free because they rode the L0 histogram
pass — which `b_star`'s data dependence forbids.

### Break-even, as one number

Setting Model B's two sides equal:

```
α* = (1.570·W − 2s) / (1.570·W + 2W + s) = 0.2375
```

**`v3` wins iff the level-0 pivot bucket holds under 23.8% of the live row.** Measured: **34.7% at
prefill**, **63.6% at decode** (`probes/13` crit `cnt` 5539 / `eff_len` 8704). It is 1.46× off at
prefill and 2.7× off at decode. This replaces `probes/13`'s "break-even `cnt_eff` 8531 vs global max
6495, `v3` cannot become a loss" — that statement was computed against the impossible pass.

### What is retracted, and why

| claim | was | now |
| --- | --- | --- |
| `design.md` §9 Plan B §5 | −30.2%, then −17.51% | **superseded** — both delete the emit traversal; it is not deleted, it is relocated |
| `probes/15` "rank 1, harness −20.5%, production −25.0%" | `measured (probes/15, j-78f7f537)` | the **6.675 µs store price is `measured` and I use it unchanged**. What is superseded is the *composition*: two stores on the L0 pass, which cannot exist. |
| `probes/13` "`v3` cannot become a loss" | break-even `cnt_eff` 8531 > max 6495 | **superseded**; break-even is in `α`, and `α` is over it |
| `design.md` "4096 is the safe bound" | — | **falsified**, §4 |

---

## 7. How the measurer runs this arm

* **Subject:** `variants/v03-fused-emit/sparsecore_topk.py`, md5 `7309435f82ab4b273abe22cbf8fcf502`,
  mapped over `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`. **Gate that md5 in-job**
  with a hard `grep -q` before timing — a docker tag is mutable.
* **Arm activation is free:** the kernel names itself `…_k2048_v3fused`. Grep the job log or the
  profile scope. A run that shows `sc_topk_direct_b1024_n10240_k2048` without the suffix did not run
  this arm. Do not rely on the source md5 alone for that.
* **Instrument:** the frozen `glm_dsa_block_bench.measure`
  (`08375d7068365e1a776c58fef09cd6d0`), `us_median`, interleaved and reshuffled rounds, paired
  `REPS_LO`/`REPS_HI` slope. Not single-shot.
* **Shapes:** `n=10240, k=2048`, **`b=1024` prefill at `rho=0.40005`** and **`b=64` decode at
  `rho=0.850`**. ⛔ Do **not** leave `row_lengths` at its default — the harness defaults ragged
  lengths to max, and at ρ=1 this arm's `α` is different (and its `cnt` reaches the whole slice).
  ρ is the single largest lever on this arm's number.
* **What to compare:** whole-kernel time. ⛔ Not the region I edited. This arm's prize and its cost
  are in *different* regions — it deletes refinement-walk time and adds store time on the emit
  walk — so a single-region arm will read one and miss the other.
* **The one number that would change the verdict:** the emit pass's true price. `probes/03` says
  27.456 µs per pass; `probes/09` says 83.830 µs by region deletion — but `probes/18` has since
  shown `probes/09` does not reproduce, so treat that 3.05× as one unreliable number beside one
  reliable one, not as a live controversy to split. Measure emit cleanly (§9 row 1) rather than
  choosing a side. §10 term 1.
* **Cheapest discriminating arm, if only one job is available:** run `v3` against the incumbent at
  **prefill only**, both arms in one job, alternating C,A,C,A. A +2% loss and a −8% win are far
  apart relative to the 1.28 µs resolution floor, so a single paired prefill pass separates the
  Model-B and region-price readings.

---

## 8. How the correctness runner runs this arm

```
cd <checkout> && python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

Baseline is **68/68 PASSED** (`j-62275e52-1498-4e01-918e`). Expect 68/68.

⚠️ **Do not use a byte-compare against the incumbent's `out_hbm` as the gate.** `v3` is exact as a
*set* and deliberately reorders slots within a row (§1). A row-comparison gate must sort each row or
compare as a multiset. A gate that byte-compares will fail on a correct kernel, and "fixing" that
would mean reintroducing the traversal the variant exists to remove.

The tests that carry the load, and why:

| test id | what it catches here |
| --- | --- |
| `test_exact_topk_full_rows[heavy_ties-p1-b1024-n10240]` | **the maximum-`cnt` id at the P=1 production shape** (`cnt ≈ 5526` of 10240): 17 distinct values ⇒ maximally lopsided level-0 buckets and a large tie set straddling the boundary. Exercises the `tie_rank <= my_take` cut on the *compact* buffer, and is the row that would have destroyed a `cnt_max = 4096` buffer. |
| `test_emit_pairs_is_bit_identical_to_the_incumbent[all_negative-p4-*]` | `cnt ≈ 7089`, and the only kind whose *winners* are negative, so it is the only one where `monotone_key`'s involution does anything at emit time. ⚠️ **P>1 only** — see §9 row 2. |
| `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240]` | the `< cnt` tail lane mask on the last straddler chunk, and `eff_len`-dependent `walk_chunks` |
| `test_emit_pairs_is_bit_identical_to_the_incumbent[p4-*]` | the `emit_keys` path: `keybuf` is written from **two** places now (split and compact emit) and must stay slot-aligned with `rowbuf` at both |
| `test_partition_factor_is_what_these_tests_assume` | pins `P=4` at `n=133120`; it is why `MAX_SLICE_WORDS` was left alone and a VMEM guard added instead. (Measured since: lowering the cap to `40960` would *also* keep `P=4` here — §5 — but I left it alone anyway.) |
| `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` | `i_hi == 0` ⇒ `total = 0` ⇒ `keep_all` ⇒ `done_l0` ⇒ both new loops bound to 0 |

**Pre-hardware evidence already in hand:** `sim.py` replays both control flows in NumPy over
`{strict_desc, random, heavy_ties, all_negative, 4-distinct} × {ρ=1.0, ρ=0.40005} × 96 rows` and
reports **0 mismatches** on both set-equality against the incumbent and multiset-equality against a
true `numpy` top-k. Run it with `uv_venv/bin/python sim.py`; it needs no TPU.

---

## 9. PARKED, with routes

| # | finding | route |
| --: | --- | --- |
| 1 | **The emit pass needs a clean price. Do not try to reconcile the 3.05×.** `probes/09` `emit` = 83.830 µs against `probes/03` `emit1` = 27.456 µs, same shape, same ρ. Since my dispatch, **`probes/18` has shown `probes/09` does not reproduce** — its `B = 326.782 µs` anchor reads 453.463 µs on exact reconstruction (+38.77%) while two other jobs agree with each other, and all 12 of its arms ran at `REFINEMENT_LEVELS = ()` while one headline row is labelled "all levels". ⚠️ **This does not resolve the gap in `v3`'s favour** — correcting `B` upward makes the raw discrepancy *wider*. The right conclusion is that one of the two numbers is not a measurement, so the pair should not be averaged, interpolated, or adjudicated; emit should simply be re-measured. **Any node whose prize is "delete or shorten a walk that carries a `store_compressed`" is currently priced on a number of unknown validity.** | One arm, and it should *measure* rather than adjudicate: the incumbent with `emit_body`'s bound forced to a fraction of `walk_chunks` (0.25, 0.5, 0.75, 1.0). The **slope** in walk length is the per-pass price — no region-deletion confound, no fit, no dependence on either disputed anchor, and a live control at 1.0. ~1 job, and it re-prices several nodes, not just this one. |
| 2 | **`design.md`'s `cnt_max = 4096` is not a safe bound** — `heavy_ties` reaches 5526 at the P=1 production shape. Any node that sizes a per-bucket buffer from that sentence inherits the bug. <br>**And the suite's `cnt` coverage is thinner than it looks.** `all_negative` is both the highest-`cnt` kind at 7089 *and* the only kind whose winners are negative — the only one that makes `monotone_key`'s involution do anything at emit time — and `TWO_STAGE_SHAPES = [p for p in SHAPES if p.values[3] > 1]` means **it never runs at P=1**. So the production shape is covered for `cnt` only by `heavy_ties`, with 1.35× of margin against the design's bound, and is covered for negative winners **not at all**. A per-bucket-capacity bug slightly less bad than the design's would ship past a green 68/68. | Two things. (a) Correct the sentence in `design.md` §9 Plan B §2, then grep the other plans for a fixed per-bucket capacity. (b) Add `all_negative` to `test_exact_topk_full_rows`'s kind list — it is one word, it costs one P=1 shape's runtime, and it closes both gaps at once. ⛔ I did not make this edit: instruments are read-only in this wave (BRIEF §2). It is the harness owner's call. |
| 3 | **Deleting the `cvt` pass looks like −4.2% and is already known to lose.** Recomputing `monotone_key` in-register at each read would remove a whole walk of `L` (8.27 µs, −4.31% harness). But `hardware_map` prices the `cvt` region at **−8.20 µs** and `probes/09` at **−20.492** — *negative*, i.e. the pass pays for itself, consistent with the recorded "a pass after a DMA can have negative marginal cost; it was absorbing the transfer latency". **Closed, not open** — recorded so the next agent does not re-derive the −4.2% and spend a job on it. | none — closed |
| 4 | **Raising `NUM_BUCKETS` would put `α` under break-even, and still lose.** `α* = 0.2375` vs `α = 0.347`; `NUM_BUCKETS = 1024` would take `α` to ≈0.087. But the fixed sweep (`zero_priv` + `fold_priv`) scales with `NUM_BUCKETS`: 19.192 µs/level → ≈76.8, i.e. **+48 µs** at production against ≈8 µs bought. **Closed.** | none — closed |
| 5 | **`plsc.load_gather` cannot replace the `strad_key` store.** It would drop `v3` to two stores (the winning direction), but the SparseCore gather granule is 512 B and a `pl.Indirect` source must be 128 int32 lanes — a 16-lane gather of scattered indices from a 1-D `i32` `resident_vmem` fetches 8 KiB per chunk. | If someone wants it: it needs `resident_vmem` retiled to `(rows, 128)`, which is `v11`'s territory, not this node's. |
| 6 | **`split_body` uses a constant shift where the incumbent's emit uses a broadcast dynamic scalar.** Saves one `vector.broadcast` per chunk on the `L` walk, free, and it applies to the incumbent unchanged: the incumbent's emit could hoist `shift`/`prefix`/`my_take` broadcasts out of the loop. Below the 1.282 µs floor individually; three of them may not be. | A one-line incumbent edit; belongs to whichever node next touches `emit_body`. |
| 7 | **The variant's slot ordering differs from the incumbent's.** Not a defect, but it invalidates any byte-compare gate reused from `v2`. | Stated in §8; the harness owner should know before writing a shared gate. |
| 8 | **`v3` narrows the supported `slice_len` ceiling from ~123,904 to 40,960** (allocator-measured, §5), while `MAX_SLICE_WORDS` still says `65536`. Not reachable today, but if `v3` ships, the two numbers disagree. | Lower `MAX_SLICE_WORDS` to `40960`. **Verified P-neutral** across every `n` in the frozen test file (0 of 216 combinations change `P`). One line, no test churn. Do it *with* the ship, not before. |
| 9 | **The XLA memory-usage report is blind to Mosaic SparseCore VMEM** — empty `vmem` table, byte-identical between two arms differing by 20,512 words of scratch. Any node claiming a VMEM high-water or a spill risk from that file is claiming nothing. | Two host-only substitutes, both used here: the allocation-failure sweep (§5) for capacity, and the bundle dump's `smem` spill count (30 = 30) for spills. Neither needs a TPU. Worth putting in the area notes. |
| 10 | **Scheduled SparseCore bundles are dumpable host-only** and are sharp enough to settle a mechanism question — here, "is the L0 scatter perturbed?" answered exactly at zero, and "is `store_compressed` free?" answered no, with a bracket that contains the time model and excludes free. `LIBTPU_INIT_ARGS`, before `import jax`. | `bundles.py` / `bdiff.py` / `bloc.py` in this folder generalise: they take two bundle dumps and return per-section counts, an opcode histogram, a spill delta, and a basic-block-level localization. Reusable by any variant. |

---

## 10. Open terms

| # | term | why it is open, and what it is worth |
| --: | --- | --- |
| 1 | **The emit pass's true price at ρ=1** — `probes/03`'s 27.456 µs is the only one of the two candidates not known to come from a non-reproducing job. | Swings this arm between **+1.85%** and **+10.03%**. Every value in that range is a loss, so it does not change the verdict — it changes several *other* nodes' prizes, which is why it is still worth a job. Route in §9 row 1. |
| 2 | **Whether `addupdate_scatter` cost follows walk length or update count.** `probes/03` gives two points at one length — `scatter_dense` 116.350 and `scatter_sparse` 35.243 — and **the sparse arm's mask density is not recorded**. If it was ~15%, cost = walk floor + per-update, and compacting the refinement walk saves 8.48 µs (Model B). If it was ~30%, cost is pure per-update, compaction saves **nothing** (Model A′), and `v3` loses by 6.27%. | The gap between the two models is **8.5 µs = 4.4%** of the kernel, and it is the largest single unknown in this report. Settle it by re-running `probes/03`'s scatter arm at three recorded densities and fitting length vs count. It also decides whether *any* "compact then re-walk" candidate can work on SC. |
| 3 | **`α = cnt / i_hi` at production** is a ratio of `probes/13`'s crit `cnt` (1423) to the *mean* `eff_len` (4096), not to a crit `eff_len`. If the critical subcore also carries longer rows, `α` is smaller than 0.347 and the arm is less bad. Bounded: `α` would have to fall to 0.2375 to break even, i.e. crit `eff_len` would have to be **5990** against a mean of 4096. Implausible but unmeasured. | `probes/13`'s replay can emit crit `eff_len` at zero cost. |
| 4 | ~~**The compact emit reads two buffers, and the two loads may dual-issue**, which would bring the arm to +0.35% — a wash rather than a loss.~~ **CLOSED against `v3`, and the mechanism I guessed was wrong.** The bundle dump shows the `v3` emit body issuing **26 `vld` where base issues 13 — exactly 2× — and not one bundle in either arm issues two loads.** They do not dual-issue. They are, however, scheduled into pre-existing slack: the emit region is 170 bundles against base's 168, so the second load costs ≈0 *issue slots*. | **Half closed.** The compute side is answered and it is free; the memory side is not, because a second buffer is a second real VMEM transaction and bundle counts cannot see transactions. So `2α·W` is an over-charge on compute and possibly right on memory. Worth **−2.87 µs at most** (−1.50pp), which does not reach break-even. `/xprof-llo` on a measured run would close it. |
| 5 | **`unmodelled`, per BRIEF §8 item 10:** the `+202` TEC bundles (+17.5% of the program) may change overlay behaviour on a cold call. Bounded, not eliminated: the program fits the 16288-byte overlay slot in both arms and no `single_imem_overlay` flag changed, so it is not a *new* overlay; but I have no model for a cold-call cost and it is the one static term I cannot price. |
| 6 | Whether the `cumsum` removed from the split pass and the popcount added there net to zero. The bundle dump gives the static side: `tpu.scan` is 5 in both arms and `vmpcnt` moves by exactly +2 sites, so no *code* was lost. The dynamic side is below `probes/15`'s **1.282 µs** resolution floor and is carried as **±1.28 µs = ±0.67pp**. |
| 7 | **Refinement grew +19 bundles** (339 → 358), concentrated in level 3, whose scatter peel takes the L0 shape (4+8+1 = 13 instructions) instead of base's (2+2+4+1 = 9). Static only. If it is a peel, it is ~free; if the scheduler chose it because the `< cnt` trip count is less predictable, it is a small real cost. Not modelled; bounded by its size at well under 1pp. |
