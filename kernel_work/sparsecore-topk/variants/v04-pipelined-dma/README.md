# `v4` — pipelined, ragged score DMA (Plan C)

Parent: the incumbent `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`
(md5 `44353b347c96e0eedbd742c710d4540a`, commit `b11b6eb6`). Built against the incumbent, not
against `v3`. **No timing job of the kernel was run** — one *microbench* was (`descriptor_cost.py`,
`j-53755323`), and it is what §6 is built on. Every µs and every `Δ time %` below is `modelled`, off
measured inputs. Sign convention: `Δ time %` is `(new − old)/old` of time; negative is faster.

**This variant ships TWO rungs as two separate files, not one candidate and not one file with a
flag**, because the decomposition below shows the plan's mechanism bundles two changes worth very
different amounts at very different risk — and the measurement then proved the point: at the plan's
own chunk the bundle **loses 3.15%** while the rung inside it **wins 4.25%**.

| rung | file | md5 | bytes | descriptors | overlap | at risk from `g`? | §6 verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | `sparsecore_topk_base.py` | `44353b34…` | padded `n` | 32/subcore | none | — | control |
| **ragged blocking** | `sparsecore_topk_ragged.py` | `69c8834e…` | live `align(i_hi,16)` | **32/subcore** | none | **no** | ✅ **−4.25% harness / −5.19% production** |
| ragged + pipelined, `tc = 2048` | `sparsecore_topk_c2048.py` | `88a67892…` | live, chunked | 77/subcore | double-buffered | **yes** | ⚠ −4.64% / −5.66%, but only **0.39% ahead of ragged against a 0.72% null** — a wash |
| ~~pipelined, `tc = 1024`~~ (the plan's chunk) | **not built** | — | — | 141/subcore | — | — | ⛔ **REFUTED: +3.15% harness** |

Full md5s and the in-job gate are in §7a.

---

## 1. What you built

The score read is now a **pipelined, ragged, chunked transfer**, and the first traversal of the row
is fused onto it. The incumbent copied `slice_len = n` words — the row's *padded* KV capacity —
with a `start(); wait()` pair inside `wave_body`, then walked the landed slice twice: `cvt` to
convert to monotone keys, `hist0` to build the level-0 histogram. `v4` replaces that with a
`pltpu.emit_pipeline` over `nc = cdiv(i_hi, tc)` chunks of `tc` words, whose body converts *and*
histograms each chunk as it lands and writes the converted keys through to `resident_vmem` for the
passes that follow. Two things change at once and they are separable by construction: the grid
length is derived from `i_hi`, the row's **live** prefix, so a 40%-occupied row moves 40% of the
bytes; and `emit_pipeline`'s default double buffering overlaps chunk `c+1`'s transfer with chunk
`c`'s arithmetic. Because those two changes are worth very
different amounts, they are **two files**, not one file with a switch: `sparsecore_topk_ragged.py`
is the first alone, `sparsecore_topk_c2048.py` is both.

The `pl.when(i_hi > 0)` guard the blocking DMA sat under is **gone, not moved**: `emit_pipeline`
allocates with `pl.run_scoped`, and putting an allocation under a conditional is what
`probes/12` blocker 4 forbids. An empty row now runs a grid of length 0. That is the in-repo idiom
(`sparse_core/ragged_gather_v2.py:102`, nested pipeline at `:208`, on the production MoE path).

The **ragged rung is deliberately the incumbent's code with one expression changed** —
`ragged_len = walk_chunks * LANES` where it said `slice_len` — keeping the single descriptor, the
same destination buffer, the blocking start/wait, and the two separate `cvt` and `hist0` walks. It
is verified below to compile to *exactly* the incumbent's instruction stream. Its `diff -u` against
the incumbent is **32 +, 1 −, 3 hunks**, of which **24 lines are docstring and 5 are comment**: the
executable delta is `−1 / +4` — the `ragged_len` binding, the two-line DMA call, and the kernel-name
suffix. There is nothing else in it, which is the point.

`sparsecore_topk_c2048.py`'s `diff -u` against the incumbent: **163 +, 14 −, 14 hunks**.

| hunk | + | − | what |
| --- | ---: | ---: | --- |
| `@@ -26,9` | 15 | 0 | module docstring: the ladder, the frozen-test/tolerance pointer |
| `@@ -38,12` | 25 | 0 | dimension glossary (code-style §1) + `CHUNK_WORDS = 2048` and why it is a swept knob |
| `@@ -63,6` | 24 | 0 | `_cdiv_dyn_by`, `_pick_chunk` |
| `@@ -80,6` | 1 | 0 | `chunk_words` in `_topk_body`'s signature |
| **`@@ -160,41`** | **~76** | **21** | **the change**: the pipeline replacing DMA + `cvt` + `hist0` |
| `@@ -349…-411` | 4 | 0 | `chunk_words` through the two entry points (4 × 1 line) |
| `@@ -434,6` | 4 | 0 | `sc.num_lanes` platform check |
| `@@ -442,6` | 14 | 0 | `_pick_chunk` call + VMEM headroom check off `sc.vmem_capacity_bytes`, `buffer_count = 2` charged explicitly |
| `@@ -466,7` | 5 | 0 | arm encoded in the kernel `name=` suffix |
| `@@ -474,6` | 1 | 0 | `chunk_words` into the `functools.partial` |

⛔ **No file in this folder reads an environment variable.** The ladder was briefly built as one
file selected by `SCTOPK_V4_CHUNK` / `SCTOPK_V4_NOPIPE` / `SCTOPK_V4_RAGGED_ONLY`; that file is
gone, deliberately, so that nothing can map the wrong subject. The rung is the file and the md5 is
the gate (§7a). Splitting it was verified to be a pure refactor: **all five AOT arms that existed
before the split reproduce their ENTRY dump md5 byte-for-byte after it** (§5).

Subjects: `sparsecore_topk_base.py`, `sparsecore_topk_ragged.py`, `sparsecore_topk_c2048.py`.
Everything else in this folder is evidence, not subject: `aot.py` (+ `aot/`) the static-compile
evidence, `aot_single.py` (+ `aot2/`, `aot3/`) the one-arm-per-process re-derivation with the XLA
dump on, `mosaic.py` the jaxpr dump, `descriptor_cost.py` the microbench that produced `g`
(**run — `j-53755323`**), `checksum.py` the correctness arm §7 asks for (**not yet run**).

---

## 2. The file plan, scored

Plan C §4 gave one row.

| plan | produced | divergence |
| --- | --- | --- |
| `sparsecore_topk.py` holds `_score_pipe`, the doubled buffer, the ragged size — **~40 lines changed** | **three files, not one** (§7a): `…_base` (unchanged control), `…_ragged` (32 + / 1 −), `…_c2048` (163 + / 14 −, algorithmic hunk ~76 + / 21 −). The pipeline is inline in `wave_body`, not a named `_score_pipe` | **explained below** |
| must not hold: the padded `slice_len` path left in as a flag | ✅ it does not. There is no flag of any kind: no file in the folder reads an environment variable, and the padded path exists only as `sparsecore_topk_base.py`, which is the unmodified incumbent serving as the control | none |

Three divergences, each deliberate:

- **No `_score_pipe` helper.** The pipeline body has to close over `i_hi`, `resident_vmem`,
  `priv_vmem`, `lane_iota` and `row_c` — five of `wave_body`'s locals, four of them tracers created
  inside `lax.fori_loop`. A helper taking all five is a worse boundary than the inline form
  (code-style §10: draw the boundary at what a function may touch), and the pipeline construction
  cannot be hoisted out of the loop anyway because the `BlockSpec` index map depends on `row_c`.
- **164 lines, not 40.** The algorithmic change is the 47/17 hunk and is on plan. The other 117
  added lines are: the dimension glossary the incumbent lacks (29), `_pick_chunk` and
  `_cdiv_dyn_by` with the reasons the constraints exist (31), the docstring naming the frozen test
  and its tolerance (17), the two platform checks (17), the `chunk_words` keyword threaded through
  three signatures and three call sites (6), and the arm-in-the-name suffix (4). Removing the
  comments would land the file at ~60 added lines; the plan's estimate counted the mechanism only.
- **The fusion was not in the plan.** Plan C describes overlapping the transfer. Doing that
  requires a consumer for each chunk, and the only consumer available at chunk granularity is
  pass 1. Fusing `cvt` and `hist0` onto the pipeline body is therefore forced by the plan's own
  mechanism — but it is a **second, compounding change** (it deletes one whole VMEM read traversal
  of the row), so it gets its own decomposition arm. See §6 and §10.

---

## 3. `code-style.md` checklist

| # | item | |
| --- | --- | --- |
| 1 | Dimension glossary at the top | ✅ added (the incumbent had none): `B N K L tc nc LANES`, padded extents lettered separately |
| 2 | Trailing `# [shape] dtype` on every ref parameter | ✅ inherited and extended — `chunk_vmem` is commented `# i32[tc], pipeline-owned, one chunk` |
| 3 | Extents from the refs, not the config | ❌ — **inherited, not introduced.** `_topk_body` takes `b, n, k, slice_len, num_waves` as ints; `v4` adds `chunk_words` in the same style. Fixing it means rewriting a signature I am not the owner of, and a variant that reshapes the incumbent's plumbing cannot be A/B'd against it. Not a blocking class. |
| 4 | jaxtyping on the public wrapper | ❌ — inherited; `_sc_topk_direct` annotates `jax.Array`. Unchanged by `v4`. Not blocking. |
| 5 | Signature carries `# Inputs. / # Outputs. / # Scratch.` | ❌ — inherited; the incumbent uses per-line shape comments instead. Unchanged. Not blocking. |
| 6 | Body order: indices → shapes → semaphores → init guard → loads → compute → masked store | ✅ preserved, and `v4` *improves* it: `zero_priv` (init) now precedes the load, where it belongs, instead of sitting between the two walks |
| 7 | Single characters only for loop indices etc. | ✅ `c` is the pipeline grid index, `g` the lane group, `w` the wave; `chunk_words`/`n_blocks` spelled out |
| 8 | Memory space in the ref name | ✅ `chunk_vmem`, `resident_vmem`, `priv_vmem`, `scores_hbm` |
| 9 | Comments name the compromise | ✅ every added comment states a trade or a constraint (why grid-0 not `pl.when`; why `live` is clipped; why `tc` is swept) |
| 10 | `NotImplementedError`, not `ValueError` | ✅ both new guards raise `NotImplementedError` |
| 11 | Four-file decomposition | ❌ — inherited; this is a single-file kernel in a shared `kernels/` tree. Out of scope for a variant. |
| 12 | Boundary at what the module may touch | ✅ `_pick_chunk` is pure int arithmetic, no refs |
| 13 | Frozen pydantic config | n/a — no config class in this kernel |
| **14** | **Tested against the XLA reference, tolerance visible at the call site** | ✅ **blocking class, satisfied.** The module docstring now names `tests/kernels/test_dsv4_sparsecore_topk.py` and states the bar verbatim: the comparison is exact-selection with **no tolerance at all** (`_mismatch`: "max score difference … expected exactly 0.0"), and names `p1-b1024-n10240` as the id that exercises this change hardest. §8 gives the invocation. |
| 15 | `Precision.HIGHEST` on f32 contractions in the reference | n/a — no contraction; the reference is `jax.lax.top_k` |
| **16** | **Hardware constants from `pltpu.get_tpu_info()`, never hardcoded** | ✅ **blocking class, satisfied for the code `v4` adds.** The lane check reads `sc.num_lanes`; the double-buffer headroom check reads `sc.vmem_capacity_bytes`. ⚠ `v4` does **not** fix the incumbent's pre-existing `num_subcores = 16` / `num_sc_cores = 2` literals at `:140-141` — see §9, it has a route. |

**Neither blocking class is `❌`.** The five `❌`s are all inherited from the incumbent and untouched
by this change; fixing any of them would alter the A/B's control surface.

---

## 4. The §7 risk table, with an outcome on every row

| risk (Plan C §7) | threatens | outcome |
| --- | --- | --- |
| ⚠ **the whole prize is already taken by `cvt`** and the measured delta is ~0 | speed | **still unchecked — no timing job was run, and the signal is not statically observable.** But the *premise* is now doubtful and I would not spend the arm on it as stated: see §9, the −8.20 µs that "proves" `cvt` absorbs arrival latency was measured by *deleting* `cvt`, which also leaves the keys unconverted and so changes the L0 bucket distribution that `hist0`, the refinements and `emit` all see. That is a data-mediated confound, not an absorption measurement. The arm that discriminates is `no_pipelining=True` (§7), which holds bytes, chunks, fusion and data constant and moves only the overlap. |
| ~~`emit_pipeline` refuses a `plsc` mesh kernel~~ (closed by `probes/12`) | — | **did not happen — 7 of 7 AOT arms compiled**, including the two things `probes/12` did not test: a `plsc.parallel_loop` *inside* the pipeline body, and a `plsc.addupdate_scatter` inside that. `aot/entry-v4-c1024-b1024.txt`. |
| ⚠ **Pallas→Mosaic lowers but Mosaic→LLO is unexercised** (VMEM or register pressure only visible on hardware) | speed, and **the build** | **the BUILD half is now discharged; the SPEED half is not.** The `--xla_dump_to` run (§5) produced `after_codegen.txt` and per-sequencer bundle spans for every arm, which means the MLO pass pipeline and SparseCore code generation both ran to completion — not merely Pallas→Mosaic. The negative control proves that stage can refuse (`E3000` from the spmem allocator), so completion is a real signal. **`v4` cannot fail to build.** What remains is speed: the pipeline adds one `Ref<vmem>{i32[2048]}` = 8192 B on top of the incumbent's 70.50 KiB, i.e. ~78.50 KiB against `sc.vmem_capacity_bytes` = 512 KiB — not the exposure. **Register pressure in the fused body is**, because it now does read → `monotone_key` → write → iota → compare → shift → add → scatter in one `parallel_loop` body at `unroll=4`, where the incumbent had two shorter bodies. ⚠ The +4-bundle IMEM delta does **not** bound this: spill code inside an `unroll=4` body would show up as bundles, and 4 is too few for that, which is mildly reassuring but is not a spill count. The check is named in §7e: `/xprof-llo` the `sc_topk_direct_b1024_n10240_k2048_v4c2048` scope. ✅ **The ragged rung has no exposure on this row at all** — it has no fused body and no extra buffer; its VMEM footprint is the incumbent's to the byte. |
| the ragged size creates a sub-granule tail and the DMA slows down | speed | **did not happen, by construction — the tail cannot be sub-granule.** Every chunk `v4` transfers is exactly `tc` words, and `tc` is a multiple of `LANES = 16` = 64 B, i.e. two 32 B DMA granules (`sc.dma_granule_size_bytes`). The raggedness is expressed in the *number* of full chunks, never in a short one; the fractional part of `i_hi` is absorbed in the arithmetic (`live = clip(i_hi − c·tc, 0, tc)`), not in the transfer. The §4d T5 fallback ("round `i_hi` up to 8 words") is therefore not needed and was not taken. ⚠ The cost of *not* having a short tail is that `v4` over-transfers up to `tc − 1` words per row — priced in §6 as `f`, the bytes fraction, and it is why the chunk size is a real trade and not a free parameter. |

Two correctness rows the plan did not have, added because the dispatch named them:

| risk | threatens | outcome |
| --- | --- | --- |
| **a copy that lands the wrong bytes** — nothing in the *timing* harness checks output (`--bw` times an unchecked accumulator, `--decomp`'s `dma_only` writes a sum nobody reads) | **correctness** | **still unchecked, and it therefore blocks the variant until §7's checksum arm runs.** The frozen *pytest* file does discharge it and I name the assertion: `_mismatch` in `test_exact_topk_ragged_rows_straddle_partition_boundaries` fails on `max |score diff| != 0.0` — there is no tolerance to hide inside. `test_comparison_detects_a_wrong_index` is the frozen meta-test proving that assertion can fail. The gap the pytest job leaves is that it verifies *one* arm and the timing job times five; `checksum.py` in this folder closes exactly that gap, with its own negative control. **Instruments were not modified.** |
| **the pipeline transfers a different amount than the blocking version did** (the dispatch's ragged warning) | correctness of the *comparison*, then speed | **did not happen — checked, and it is the point of the change, not an accident.** Incumbent: `slice_len = n = 10240` words per row, unconditionally, `static in ρ`. `v4`: `ceil(i_hi/tc) · tc` words. At ρ = 0.40005 and `tc = 1024` that is ~0.450 · n. Arithmetic is held *identical*, not merely bounded: the total lane groups walked is `Σ_c cdiv(clip(i_hi − c·tc, 0, tc), 16) = cdiv(i_hi, 16) = walk_chunks`, exactly the incumbent's count, so the fused pass converts and histograms the same elements and not one more. Bytes beyond `nc·tc` in `resident_vmem` are never read: `walk_chunks · LANES = align(i_hi,16) ≤ nc·tc`, and `load_keys` masks `j < i_hi` on top. **⛔ Because `v4` changes the bytes moved, any A/B that runs at ρ = 1 measures only half of it** — see §7. |

---

## 5. `static-compile` evidence

`aot.py` → `aot/` (all arms in one process), `aot_single.py` → `aot2/` (one arm per process, XLA
dump on, env-selected arms — the pre-split evidence) and `aot_single.py` → `aot3/` (the same, after
the split, arms selected by **file**). Device pinning is `probes/17/hlo_diff.py:24-36` verbatim:
`topologies.get_topology_desc("tpu7x:2x2x1")`, `SingleDeviceSharding`, `jax.jit(...).lower().compile()`,
no TPU attached. jax 0.10.2.

**The in-process A/B was re-derived in fresh processes, and it holds.** `jax.jit` keys its cache on
`(function object, avals)` and pallas caches below that, so a multi-arm compile in one interpreter
can compare an arm against itself and return a byte-identical answer *with no error* — a sibling
variant hit exactly this. `aot.py` escapes it (each arm gets a fresh module object under its own
`sys.modules` name and a fresh lambda), but "escapes it" is not evidence, so `aot_single.py`
re-derives every arm in its own interpreter, **variant first** so a residual cache bug would show
as base-looking-like-variant rather than the reverse. Each child prints its bound `__file__` and the
**md5 of the subject file**, and asserts that the module exposes no `RAGGED_ONLY` / `NO_PIPELINING`
symbol, before compiling. Result: **all six ENTRY dumps are byte-identical to the in-process run
(md5 match, 6 of 6), and all six are distinct from each other.**

| arm | `CHUNK_WORDS` / `NO_PIPELINING` bound in the child | ENTRY md5 | vs in-process |
| --- | --- | --- | --- |
| `v4-c1024` (run **first**) | 1024 / False | `fc431810ee54` | match |
| `base` | n/a / n/a (incumbent has no such symbol) | `f6496c9d5318` | match |
| `v4-c2048` | 2048 / False | `d255f9ef6896` | match |
| `v4-c1024-nopipe` | 1024 / **True** | `0f9965d5209c` | match |
| `v4-c1024` b=64 | 1024 / False | `f9a3f1c0981d` | match |
| `base` b=64 | n/a / n/a | `6c906ce05c4d` | match |

The ragged rung was added afterwards and compiles at both shapes
(`sc_topk_direct_b1024_n10240_k2048_v4ragged`, `…_b64_…_v4ragged`). **Regression check on the
reindent that adding it required: all six ENTRY md5s above are unchanged**, so the pipelined rungs
trace exactly as they did before the ladder existed.

⛔ **The three-file split is a pure refactor, verified the same way — and this is the check that
matters most for trusting the shipped subjects.** Splitting one env-switched file into three
standalone files is exactly the kind of edit that silently drops a line. Every arm was recompiled
from the new files in fresh interpreters (`aot3/`) and compared to the pre-split dumps:

| arm | subject file (md5) | ENTRY md5, pre-split | post-split |
| --- | --- | --- | --- |
| `base`, b=1024 | `sparsecore_topk_base.py` (`44353b34…`) | `f6496c9d5318` | **identical** |
| `v4-ragged`, b=1024 | `sparsecore_topk_ragged.py` (`69c8834e…`) | `d0a483948bc3` | **identical** |
| `v4-c2048`, b=1024 | `sparsecore_topk_c2048.py` (`88a67892…`) | `d255f9ef6896` | **identical** |
| `base`, b=64 | — | `6c906ce05c4d` | **identical** |
| `v4-ragged`, b=64 | — | `0a73ef342f43` | **identical** |
| `v4-c2048`, b=64 | — | *(new arm)* | `54f54665f5b2` |

**5 of 5 pre-existing arms byte-identical, 6 of 6 post-split arms distinct**, and the TEC bundle
counts below reproduce exactly. The split changed the subject files and changed nothing the compiler
emits.

**It lowers, everywhere.** Seven positive arms, all `OK`:

| arm | b | ENTRY instrs | pallas kernel name in the HLO |
| --- | ---: | ---: | --- |
| incumbent | 1024 | 13 | `sc_topk_direct_b1024_n10240_k2048` |
| incumbent | 64 | 10 | `sc_topk_direct_b64_n10240_k2048` |
| `v4` chunk 512 / 1024 / 2048 | 1024 | 13 / 13 / 13 | `…_v4c512` / `…_v4c1024` / `…_v4c2048` |
| `v4` chunk 1024, `no_pipelining` | 1024 | 13 | `…_v4c1024_nopipe` |
| `v4` chunk 1024 | 64 | 10 | `sc_topk_direct_b64_n10240_k2048_v4c1024` |
| **`v4-ragged`** (shipped file) | 1024 / 64 | 13 / 10 | `…_v4ragged` |
| **`v4-c2048`** (shipped file) | 1024 / 64 | 13 / 10 | `…_v4c2048` |

⚠ The `c512`, `c1024` and `nopipe` rows are **pre-split evidence from `aot2/`, kept because they are
what established that the constructs lower at all.** Those arms are refuted (§6) or withdrawn
(§7a) and **have no subject file** — do not try to run them.

The arm reaches the HLO in the kernel name, so a profile scope proves which one *fired*. That is
now the **second** witness rather than the only one: since each rung is its own file, the subject's
md5 in the job log is the first (§7a). A mapped file only proves it was on the pod; the scope name
proves it executed.

**Negative control fired, and it is the right one — verified three ways in one process.** The
control has to be refused by the SparseCore *allocator*, not by the MLIR layer, or it proves
nothing:

| probe | outcome | error |
| --- | --- | --- |
| 8 MiB VMEM scratch, DMA'd to HBM — **the control** | **FAILED, as required** | `INTERNAL: Failed to run MLO pass pipeline … 'memref.alloca' op` **`E3000: CompileTimeSparseCoreAllocationFailure`**`: current allocation offset upper bound (33554431 words) exceeds the legitimate user allocatable offset upper bound (2097151…)` |
| 8 MiB VMEM scratch, written with `o[...] = jnp.zeros_like(o)` — **the wrong form** | FAILED, but **`OTHER-ERROR`** | `Swap does not support storing to HBM. Copy the data to a core-local memory space…` — the MLIR layer, reached before the allocator. A control that stops here would pass anything. |
| **4 KiB** VMEM scratch, same DMA form — **positive control** | **COMPILED** | — |

The third row is not in the brief and I added it: without it, "8 MiB fails" is also consistent with
the DMA *form* being broken for an unrelated reason. With it, the only variable between compile and
refusal is the allocation size, so `AOT OK` means the SparseCore MLO backend accepted the kernel —
not merely that the MLIR verifier did. ⚠ My first attempt at this control was the middle row, and I
caught and replaced it before any result was reported; the top row is what ran in `aot.py`.

**A dynamic-size DMA is expressible — checked before building the ragged rung, because the rung is
impossible without it.** Pallas normally requires a static size in `pl.ds`, which would have forced
the ragged arm to be chunked and destroyed the whole point of it (a chunked ragged arm is just
`no_pipelining` again). Three forms, AOT:

| form | outcome |
| --- | --- |
| dynamic **size**, `pl.ds(0, align(i_hi,16))` — what the arm needs | **COMPILES** |
| a `lax.switch` ladder of static sizes — the fallback | COMPILES (not needed) |
| dynamic **start** at an arbitrary offset | **FAILS**: `Slice offset for 32-bit 1D memrefs must be a multiple of 8` |

⚠ The third row is a platform constraint worth carrying: **1D i32 memref slice offsets must be
8-word aligned.** It makes `align(i_hi, LANES = 16)` *mandatory* rather than merely tidy — 16 is a
multiple of 8, and it is also why rounding to a lane group rather than to the exact `i_hi` is not a
wasteful choice. The incumbent satisfies it accidentally (`s = row_c * n`, `n = 10240`), as does the
pipelined rung (`row_block0 * chunk_words`, `chunk_words` a multiple of 16).

**Sequencer bundle counts, from `XLA_FLAGS=--xla_dump_to`.** These are in
`*sparse_core_specific_metadata.txt` and appear in no other artifact — `compiled.as_text()` does not
have them. They are the **static IMEM footprint** of the kernel's instruction stream on the
TILE_EXECUTE_CORE sequencer (`TPU_SEQUENCER_TYPE_SPARSE_CORE_TILE_EXECUTE_CORE_SEQUENCER`):

| arm | TEC bundles | vs base |
| --- | ---: | ---: |
| `base`, b=1024 | 1168 | — |
| **`v4-ragged`, b=1024** | **1168** | **0** |
| `v4-c1024`, b=1024 | 1172 | **+4** |
| `v4-c2048`, b=1024 | 1172 | **+4** |
| `v4-c1024-nopipe`, b=1024 | 1155 | **−13** |
| `base`, b=64 | 1166 | — |
| **`v4-ragged`, b=64** | **1166** | **0** |
| `v4-c1024`, b=64 | 1171 | **+5** |

Four readings, and the first two are the ones that matter:

- ⛔ **`v4-ragged` is `0` bundles — it is the incumbent's instruction stream, exactly.** At both
  shapes. And it goes further: the ENTRY diff against `base` is **name-only** (10 lines, all the
  kernel-name string), the `*sparse_core_specific_metadata.txt` is **byte-identical**, and so is
  `*-memory-usage-report.txt`. The only thing that changed anywhere in the compiled artifact is the
  DMA's length operand. That is the "ragged and nothing else" constraint **verified**, not asserted:
  this arm cannot be carrying a hidden second change, because there is no room in the output for
  one. It is the cleanest isolator in the ladder and the one whose result will be easiest to trust.
- **`v4` full costs 4 bundles of IMEM.** The whole pipelined, ragged, fused rewrite adds 0.34% to
  the instruction stream. Nothing here is near an overlay boundary (slot size 16288, three
  overlays), so no I-cache thrash is introduced.
- ⛔ **`c1024` and `c2048` are byte-identical at 1172 — so the grid is a genuine dynamic loop, not
  unrolled per chunk.** This was an assumption in §6 and it is now evidence: if `emit_pipeline` had
  peeled or unrolled the grid, the arm with twice the steps would carry more code, and it does not.
  It also means **the chunk sweep is free of any code-size confound**, and the move to `tc = 2048`
  that §6 forced costs zero IMEM. ⚠ It also means the reverse: a bad chunk cannot be detected
  statically, because every chunk compiles to the same 1172 bundles. `tc = 1024` is refuted by
  measurement (§6d) and by nothing here.
- **The pipelining machinery is 17 bundles** (1172 − 1155): the `copy_in`/`wait_in` split, the
  buffer-parity select and the prologue/epilogue peel, isolated exactly, because `no_pipelining`
  holds everything else constant. That is a sound statement about **static IMEM code size** and it
  stands.

  ⛔ **What does NOT stand, and it was mine: I used those 17 bundles as an upper bound on `g`'s
  bookkeeping term.** The reasoning was "at a ~1 GHz-class bundle rate, 17 bundles is order 10-20 ns,
  comfortably under the 108 ns break-even". `descriptor_cost.py` then measured
  **`g_pipe` = 248.4 ns *below* `g_raw` = 517.5 ns** (`j-53755323`) — the pipeline makes a descriptor
  **cheaper**, so the term the bound was bounding **has the opposite sign** and there was never an
  additive bookkeeping cost for 17 bundles to cap. The bound was not merely loose; it was measuring
  the wrong quantity, and it happened to point at the right verdict for the wrong reason.
  **RETIRED as a `g` proxy** (§9 carries it as a caution). The general form: *a static code-size
  bound cannot establish a term's sign, and a bound quoted as "comfortably under break-even" has
  assumed one.*

  The caveats I did attach to it were correct and are worth keeping: a bundle count is code size,
  not time; this tree publishes no SparseCore clock (`sc` exposes `num_cores`, `num_subcores`,
  `num_lanes`, `vmem_capacity_bytes`, `dma_granule_size_bytes` and nothing else); and the dominant
  term in `g` is DMA descriptor issue and sync-flag wait, which is not sequencer instruction issue
  and is invisible here. They were the reason to distrust the bound, and they were not enough — only
  the measurement was.

**What changed outside the kernel: nothing.** `aot/diff-base-vs-v4.txt` is 17 lines and **every one
of them is the kernel-name string**. ENTRY stays at 13 instructions (10 at b=64); the
`%bitcast-convert_bitcast_fusion` (16 MiB `scoped_memory_configs`) and the `ROOT %reshape.4`
(4 MiB `used_scoped_memory_configs`) are byte-identical, and so is the entire
`*-memory-usage-report.txt` and `*basic_compiler_metadata.txt`. That matters for the §8b rule: `v4`
adds and deletes no XLA-side term, so there is no unpriced op outside the kernel that could absorb
the saving. Whatever it wins or loses, it wins or loses inside `sc_topk_direct`.

**A third control, which did *not* fail, and that is the finding.** Forcing `chunk_words = 3072`
past `_pick_chunk` — a value that does **not** divide `n = 10240` — **compiles cleanly**. So the
divides-`n` rule is *mine*, not the compiler's: a future edit that hardcodes a chunk gets silent
cross-row reads, not an error. `_pick_chunk` is the only thing holding it, which is why it raises
and why it rounds *down* into the divisor set. Legal chunks at `n = 10240`, ≤ 4096:
`{16, 32, 64, 80, 128, 160, 256, 320, 512, 640, 1024, 1280, 2048, 2560}` — **4096 is not among them**
(`10240 = 2¹¹ · 5`).

**What changed inside** (`mosaic.py`, `pl.kernel(debug=True)` jaxpr — ⚠ this is the jaxpr, not
post-allocation Mosaic, so it counts *static* ops and does not show spills):

| | incumbent | `v4` c1024 | `v4` c1024 nopipe |
| --- | ---: | ---: | ---: |
| jaxpr lines | 772 | 982 | 832 |
| `dma_start` / `dma_wait` | 3 / 3 | 4 / 3 | 3 / 3 |
| `run_scoped` | 0 | 1 | 2 |
| `semaphore_mem` refs | 6 | 12 | 7 |
| `Ref<vmem>{i32[2048]}` | — | present | present |

The `i32[2048]` is the double buffer: 2 × 1024 words = **8192 B**, one `run_scoped` allocation per
wave iteration. Chunk size does not change the structure — c512, c1024 and c2048 are all 982 lines,
differing only in constants. **`no_pipelining=True` allocates the same `BufferedRef`s**
(`jax/_src/pallas/mosaic/pipeline.py:1725-1760` — it swaps `copy_in`/`wait_in` for `sync_copy` and
nothing else), so it holds identical VMEM, an identical chunk count and an identical fused body, and
isolates *only* the overlap. That is what makes it a legitimate decomposition arm rather than a
different kernel.

**Spills and VMEM high-water are still not statically observable — and I checked the one artifact
that looked like it would give them.** `*-memory-usage-report.txt` has a `Memory Space: vmem
(color=1)` section, but for this kernel its allocation table is **empty**, and `after_codegen.txt`
reports `vmem_bytes: "0"`. XLA's buffer assignment does not see a Pallas SparseCore kernel's
internal scratch at all — Mosaic allocates it below XLA — so the report is byte-identical between
`base` and `v4` and cannot price the double buffer. (It does show the HBM side: 128.07 MiB default
space, 40 MiB `f32[1024,10240]` scores, 8 MiB `s32[1024,2048]` output, and 4 B of smem.) The
double buffer is therefore evidenced only by the jaxpr's `Ref<vmem>{i32[2048]}`, and the high-water
stays **modelled**: 70.50 KiB (the incumbent's LLO high-water, `report.md` §7) + 8.00 KiB =
**78.50 KiB** of 512 KiB. `/xprof-llo` remains the only route to the real figure and to spills.
Named as an open term in §10.

---

## 6. Prediction, restated as a number to be scored

The plan predicted **−7.1% harness at the ceiling**, `modelled`, band −7.1% to 0%. Rank 4 on the
shortlist is ≤ −7.09% harness / ≤ −8.64% production, both a **ceiling**.

⚠ **This section was rewritten after `descriptor_cost.py` ran (`j-53755323`, null spread 0.079%).**
`g` was the model's one free parameter; it is now measured, and it did not merely fill a hole — **it
inverted a sign the earlier version of this section got wrong**, and it kills one of the two
pipelined rungs. What follows is the corrected model. The earlier `tc = 1024` headline of −6.38% is
**retracted**.

### 6a. The two measured inputs

| quantity | value | source |
| --- | ---: | --- |
| `X`, the incumbent's exposed score-read cost, prefill | **13.469 µs** (7.09% of the 189.98 µs harness denominator) | `measured`, `j-300d4690`, `static in ρ` |
| `X`, decode | **1.018 µs** (1.97% of 51.658 µs) | `measured`, `j-300d4690` |
| `g_raw`, per-descriptor cost of a **blocking** `start(); wait()` | **517.5 ns** | `measured`, `j-53755323` |
| `g_pipe`, per-descriptor cost **through `emit_pipeline`** | **248.4 ns** | `measured`, `j-53755323` |
| `f`, bytes fraction, and `nc`, chunks per row | per-row exact, see 6b | **`measured on host`** — new, see below |

⛔ **`g_pipe < g_raw`, by a factor of 0.480 — the pipeline *removes* 269.1 ns per descriptor rather
than adding bookkeeping on top.** The earlier version of this section, and `design.md`, both layered
an `emit_pipeline` overhead term *on top of* a raw blocking DMA and expected `g_pipe > g_raw`. That
sign was wrong. The mechanism is that a blocking `start(); wait()` pays the full descriptor latency
exposed, every time, while the pipeline overlaps issue with the previous step's arithmetic — so
`g_raw` is "descriptor cost **plus** exposed latency" and `g_pipe` is much closer to descriptor cost
alone. **Consequence for the model: there is no additive bookkeeping term. Adding one double-counts.**
The correct form is `total = bytes term + descriptors × g_arm`, with `g` taken per arm and applied
to *every* descriptor, not just the extra ones.

### 6b. `f` is no longer assumed — it is computed from the frozen distribution

The previous version took a uniform round-up, `f ≈ ρ + tc/(2n)`. That is unnecessary: the row-length
distribution the harness will actually run is `_row_lengths("random", b, n, n_p)` in the frozen
`sctopk_2stage_bench.py`, and it can be evaluated on the host for zero cluster time. It was, and it
differs from the approximation by 1.0–2.5 points at every chunk (the approximation is always
pessimistic). Both `f` and the per-row chunk count `nc` below are now **exact for the distribution
that will be measured**, and the exposed term is counted per row (an empty row has `nc = 0` and
exposes nothing) rather than as an average:

| b | rows | ρ (align-16) | zeros | full | `f`/mean `nc` at tc = 1024 | at tc = 2048 | at tc = 2560 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1024 | 1024 | **0.40004** | 205 | 204 | 0.43945 / 4.395 | 0.47949 / 2.397 | 0.49951 / 1.998 |
| 64 | 64 | **0.39126** | 13 | 12 | 0.43125 / 4.312 | 0.47187 / 2.359 | 0.49219 / 1.969 |

⛔ **The b = 64 row is a correction, and it is the most consequential thing in this section after
`g`.** The dispatch and `design.md` give the production **decode** occupancy as **ρ = 0.850**. The
frozen bench's ragged mode gives **ρ = 0.391 at b = 64** — it cycles the same edge set regardless of
`b`, so its small-batch arm is a *small-batch prefill-occupancy* run, not a decode run. The ragged
rung's saving is proportional to `1 − ρ`, so:

- what the harness will read at b = 64 is **−1.20%**, not the −0.30% the production occupancy implies;
- **the harness b = 64 number overstates the production decode saving by 4.0×** and must never be
  reported as a decode result;
- and the frozen instrument **cannot** be made to run ρ = 0.850 without editing it, which is
  forbidden. Production decode for this candidate is therefore `modelled` by construction, scaled
  from the prefill-occupancy reading by `(1 − 0.850)/(1 − 0.391)`.

This is the same class as `microbench-defaults-ragged-lengths-to-max`, inverted: there the bench was
denser than production, here it is sparser.

### 6c. The corrected ranking

Descriptors are counted **per subcore** (32 subcores; b = 1024 is 32 rows each, b = 64 is 2 each),
because that is the serial chain the wall clock sees. `dcost` is `descriptors × g_arm` differenced
against the incumbent's `32 × g_raw`.

**b = 1024, ρ = 0.400 (prefill):**

| arm | bytes | desc/subcore | gross saving | `dcost` | **net** | **Δ time % harness** | **Δ time % production** | vs ragged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 1.00000 | 32.0 | — | — | — | — | — | — |
| **ragged** | 0.40004 | **32.0** | 8.081 µs | **0** | **8.081 µs** | **−4.25%** | **−5.19%** | — |
| `tc = 512` | 0.41943 | 268.4 | 12.930 µs | +50.120 | **−37.190 µs** | **+19.58%** | +23.87% | +23.83% |
| `tc = 1024` | 0.43945 | 140.6 | 12.392 µs | +18.371 | **−5.980 µs** | **+3.15%** | +3.84% | +7.40% |
| **`tc = 2048`** | 0.47949 | 76.7 | 11.314 µs | +2.497 | **+8.818 µs** | **−4.64%** | **−5.66%** | **−0.39%** |
| `tc = 2560` | 0.49951 | 63.9 | 10.776 µs | −0.678 | +11.454 µs | −6.03% | −7.35% | −1.78% |

**b = 64, ρ = 0.391 (the harness's small-batch arm — *not* production decode):**

| arm | desc/subcore | net | Δ time % harness |
| --- | ---: | ---: | ---: |
| **ragged** | 2.0 | +0.620 µs | **−1.20%** |
| `tc = 1024` | 8.6 | −0.171 µs | +0.33% |
| **`tc = 2048`** | 4.7 | +0.719 µs | **−1.39%** |

### 6d. Three verdicts

- ⛔ **`tc = 1024` is REFUTED.** It is a **+3.15% harness loss against the incumbent** and a **+7.40%
  loss against the ragged rung**. The plan's own chunk is the worst shippable point on the ladder,
  and `tc = 512` is far worse still (+19.58%). Break-even `g_pipe` at `tc = 1024` is **148.4 ns**
  against a measured 248.4 — missed by 67%, not by a margin any calibration error covers. It is
  **not built as a subject file** and should not be run.
- ⚠ **`tc = 2048` versus the ragged rung is a WASH, and genuinely undecided — not a kill and not a
  win.** The model puts it **0.737 µs ahead of ragged, −0.39% of the harness denominator**. The null
  spread at b = 1024 is **0.72%**. ⛔ **The difference is a factor of 1.8 *below* the instrument's
  own floor, so a single A/B cannot adjudicate it.** Break-even `g_pipe` is **258.0 ns** against a
  measured **248.4 ns** — inside by **3.7%**, and the sign of the whole comparison moves with a 4%
  change in any input. Both rungs are shipped for exactly this reason.
- ✅ **The ragged rung is the result.** −4.25% harness / −5.19% production, resolvable against the
  0.72% null by a factor of 5.9, one descriptor, no `g` exposure, no extra VMEM, and an instruction
  stream that is byte-identical to the incumbent's (§5). Every quantity in `X · (1 − ρ)` is measured.
  It is what survives if the pipelined rung loses, and it is unaffected if the pipelined rung wins.

### 6e. ⚠ The model does not reconcile at the incumbent's own operating point

Stated because it bounds everything above. Charging the incumbent's own 32 descriptors at the
measured `g_raw` gives **32 × 517.5 ns = 16.560 µs**, which **exceeds `X` = 13.469 µs, the entire
measured cost of the score read it is supposed to be a component of** — by 23%. At b = 64 the same
check very nearly passes: 2 × 517.5 = 1.035 µs against `X` = 1.018 µs, an overshoot of 2%.

So `g_raw` does not transfer as a single constant: the production per-descriptor cost implied by `X`
is **421 ns at b = 1024** and **509 ns at b = 64**, while the microbench reads 517.5 ns at both.
Two candidate explanations, neither tested: the microbench has no arithmetic to absorb the transfer
latency, so its blocking descriptor is maximally exposed
(`a-pass-after-a-dma-can-have-negative-marginal-cost`); or `X`, which was measured by *deleting* the
DMA, is an underestimate because freed device time spilled onto untouched ops
(`freed-device-time-spills-onto-untouched-ops`).

Two consequences, and I would rather state them than let the tables read as tighter than they are:

1. **The 3.7% margin at `tc = 2048` is inside the transfer error.** Scaling both `g` by
   `X/(32·g_raw)` = 0.813 so the model reconciles at the incumbent — an equally defensible
   calibration — moves `tc = 2048` from +0.737 µs to within 0.01 µs of dead level with ragged.
   Under **both** calibrations it is a wash; the wash is the robust finding, the sign is not.
2. **`tc = 1024`'s refutation survives either calibration** (−11.6 µs versus ragged under the
   reconciled one, −14.1 µs under the raw one), which is why it is a verdict and not a caution.

⛔ **And this is why `tc = 2560` is not promoted despite topping the table.** Its `dcost` is
**negative** (−0.678 µs): the model claims the pipelined arm costs *less* in descriptor terms than
the incumbent's single blocking copy, purely because `g_pipe < g_raw` is applied to all 63.9
descriptors. That is not a mechanism, it is the 23% calibration gap re-appearing as free money, and
it is smaller than the gap itself. The gradient "larger `tc` is better" runs the whole width of the
legal set and must terminate — at `tc = n` the arm degenerates to the incumbent and saves nothing —
so the model has a maximum whose *location* is set entirely by terms it cannot presently resolve.
**Chasing it would be re-deriving a win from the least reliable part of the model.** `tc = 2560` is
recorded as a modelled figure and explicitly **not** recommended.

### 6f. ⛔ Say it before it is measured, separately per phase

- **`b = 1024` (prefill), the phase where this is decided.** `ragged` **−4.25% harness / −5.19%
  production**. `tc = 2048` **−4.64% / −5.66%**. The two differ by **0.39%**, against a **0.72%**
  null spread — so I predict, in advance, that **the ranking job will not separate them**, and that
  a reported separation in either direction at n = 1 is noise unless it exceeds 0.72%. Both should
  separate cleanly from the incumbent.
- **`b = 64`.** `ragged` −1.20%, `tc = 2048` −1.39%, against a **3.37%** null spread. ⛔ **Both are
  `not resolvable`. A reading here must be reported as `not resolvable`, never as "no effect", and
  never as a decode result at all** — see 6b, this arm runs at ρ = 0.391, not the production decode
  ρ = 0.850. Scaled to production decode the ragged rung is **−0.30%**, which is inside the floor by
  a factor of 11.

  ⚠ One correction to the dispatch's framing, unchanged from before: `num_waves = _cdiv(b, 32)`, so
  `b = 64` is **2** waves, not 1. The pipeline is *within* a slice, so it runs `nc ≈ 2.4` steps per
  row at either phase. **Within-slice pipelining is phase-agnostic; the cross-wave form Plan C drew
  would have been prefill-only.** The phase gating comes from the denominator and the null floor.

### 6g. The step count is the design variable, so here it is stated as one

`03-pipelining.md` gives `per_step ≈ 512 ns + bytes/3.48e12` on a TensorCore fit and warns that a
floor quoted without a step count is not a floor. `v4`'s step count is `nc`, and `nc` is the only
thing I chose: it sets the prologue (one chunk per nonempty row stays exposed), the raggedness waste
(`≤ tc/n` of `n` over-moved), the VMEM cost (`2·tc·4` B), and the descriptor count. **Those four now
have measured coefficients rather than assumed ones, and the answer they give is that the step count
wanted to go UP, not down** — every halving of `tc` from 2048 costs more in descriptors than it
recovers in prologue. The plan's `tc = 1024` was on the wrong side of that. ⚠ The measured `g_pipe`
of 248.4 ns is also, notably, **half** the TensorCore 512 ns/step figure rather than equal to it, so
the "does the TensorCore per-step cost transfer to SparseCore?" question that the previous version
of this section declined to assume away now has an answer: **not unchanged — it is about 2× too
pessimistic, and it has the wrong sign relative to a blocking copy.**


## 7. How the measurer runs me

### 7a. ⛔ Three subject FILES, three md5s — not one file with an env switch

The ladder used to be one file selected by `SCTOPK_V4_CHUNK` / `SCTOPK_V4_NOPIPE` /
`SCTOPK_V4_RAGGED_ONLY` at import. **It is not any more.** Each rung is now a separate,
self-contained file that reads no environment variable at all. An env-selected arm can be mis-set
silently by anything upstream in a job and the only evidence it was set correctly is a log line; a
file cannot be mis-set, and its md5 is gateable in-job (`docker-image-tag-is-mutable`).

| rung | file | **md5** | kernel scope name |
| --- | --- | --- | --- |
| control | `sparsecore_topk_base.py` | **`44353b347c96e0eedbd742c710d4540a`** | `sc_topk_direct_b{b}_n{n}_k{k}` |
| **ragged blocking** | `sparsecore_topk_ragged.py` | **`69c8834e130e23999e0284929230da3a`** | `…_v4ragged` |
| **ragged + pipelined, tc = 2048** | `sparsecore_topk_c2048.py` | **`88a67892e8c063b5d9fea96de72e99c4`** | `…_v4c2048` |

`sparsecore_topk_base.py` is a byte-for-byte copy of
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` at commit `b11b6eb6` — the md5s are
identical, which is the check `checksum.py` performs and prints. **Gate all three md5s in-job.**

**Map**: this folder, with each rung copied over
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` for its own process
(`cdk job create --map-dir`; no E2E image path — this is a microbench). `checksum.py` and
`descriptor_cost.py` load the three files by path directly and need no mapping at all.

⚠ There is deliberately **no `tc = 1024`, `tc = 512`, `tc = 2560` or `no_pipelining` file.** §6
refutes 1024 and 512 outright; 2560 is declined on the grounds given in §6e; `no_pipelining` was a
decomposition arm for separating "moved fewer bytes" from "hid the bytes it moved", and
`ragged` versus `c2048` now does that job directly with two arms the job is already running. If any
of them is wanted it is a one-line edit to `sparsecore_topk_c2048.py` and it must be shipped as a
**fourth file with a fourth md5**, never as a flag on one of these three.

### 7b. The arms, and the two differences that carry the verdict

| arm | subject | what it isolates |
| --- | --- | --- |
| `incumbent` | `sparsecore_topk_base.py` | control — run it at least twice so the slice can be bounded from its absolutes |
| **`v4-ragged`** | `sparsecore_topk_ragged.py` | ⛔ **run this one even if you run nothing else.** Live bytes, one descriptor, no pipeline, no extra VMEM. It is the whole result if the pipelined rung loses, and it compiles to the incumbent's exact instruction stream (§5) |
| **`v4-c2048`** | `sparsecore_topk_c2048.py` | the caller's actual mechanism: chunking + overlap + fusion |

- `v4-ragged − incumbent` = **raggedness alone**, the part that cannot lose to `g`. Predicted
  **−4.25%** harness at b = 1024, resolvable against the 0.72% null by 5.9×.
- `v4-c2048 − v4-ragged` = **overlap + fusion + chunking**, priced against the right baseline
  rather than against the incumbent. Predicted **−0.39%**, i.e. **below the null spread** — see 7d.

### 7c. Shapes and ρ

**Shapes**: `b ∈ {64, 1024}`, `--ctx 9216` (n = 10240, k = 2048, P = 1). ⚠ `--ctx 9216` is **not**
the bench default and must be passed every time.

**⛔ The ρ arm is not optional for this candidate.** The frozen `--decomp` path sets
`rl = np.full(b, n)` at `sctopk_2stage_bench.py:582` — it runs at **ρ = 1.0**. At ρ = 1 the
raggedness term is worth **exactly zero** and `v4-ragged` is a null *by construction*, so a
ρ=1-only run would score the best rung on the ladder as "no effect". Run both: **ρ = 1** isolates
the pipelining term cleanly, **ρ = 0.400** is where the production answer lives, and the difference
between them *is* the raggedness contribution. Report both absolutes, not just the ratio. If only
one ρ can be afforded, run the ragged one.

⛔ **Do not report the b = 64 arm as a decode result.** §6b: the frozen ragged distribution gives
ρ = 0.391 at b = 64, not the production decode ρ = 0.850, and the saving scales with `1 − ρ`, so
the harness reading overstates production decode by 4.0×. Both b = 64 predictions are inside the
3.37% null spread and must be reported as **`not resolvable`**, never as "no effect".

### 7d. ⛔ What to do about a difference smaller than the null spread

`v4-c2048 − v4-ragged` is predicted at **0.39% against a 0.72% null spread**. This is stated in
advance so that it is not discovered afterwards as a rationalisation:

- **A single A/B cannot adjudicate it.** Whatever sign it comes back with at n = 1 is noise.
- The honest outcomes are **"tie, both shipped"** or **replicates**. If replicates are affordable,
  they must be *separately submitted* — `replicates-submitted-together-are-one-sample`.
- ⛔ **Do not resolve the tie by promoting whichever arm happens to read faster.** If they tie,
  `v4-ragged` ships: it is strictly simpler, allocates no extra VMEM, adds no descriptors, and its
  saving depends on no measured parameter that could drift.
- The incumbent-versus-either comparison is a different question and is comfortably resolvable.

### 7e. Run order, cheapest-first

1. ~~`descriptor_cost.py`~~ — **done, `j-53755323`**, null spread 0.079%. `g_raw = 517.5 ns`,
   `g_pipe = 248.4 ns`. It is what rewrote §6, killed `tc = 1024`, and reduced `tc = 2048` to a
   wash. Re-run only if the ranking job disagrees with §6 by more than the transfer error in §6e.
2. `uv_venv/bin/python checksum.py --ctx 9216`. **The correctness arm, in the same job, before the
   timed block.** It prints the three subject md5s and checks the mapped base against the tree's
   incumbent, then runs both variant rungs at both `b` and both ρ regimes against the incumbent and
   asserts `max |score diff| == 0.0` on the sorted selected scores plus an equal live count. It
   carries its own negative control (one index swapped for the row minimum — the comparator must
   fail) and a non-vacuity guard on the distribution. Grep for `CHECKSUM: PASS` **on its own line**,
   not the exit code: a `#` comment cannot carry an `exit 1` through cdk's YAML fold. ⛔ Do not
   proceed to timing if it prints `FAIL`, and do not modify anything under `instruments/` to
   accommodate it.
3. The timed A/B, **all three arms in one job**, alternating `C, A, C, A` (priority tier confounds
   arms across jobs), incumbent appearing at least twice.
4. `/xprof-llo` the winning arm's scope. Two readings the static evidence cannot give: **spill
   count** in `c2048`'s fused pass-1 body (the register-pressure risk in §4 — `ragged` has no fused
   body and no exposure here), and whether the `ep_copy_in`/`ep_wait_in` scopes show the transfer
   actually overlapping. `trace_scopes=True` if the default scopes are not enough.

**Compile-time constants of mine that deserve a rival arm**: `chunk_words` — it is the design
variable, and §6 has now swept it in the model rather than on the pod, which is why only one value
is shipped. No other constant was introduced.


## 8. How the correctness runner runs me

Unmodified frozen test, mapped source:

```
# one process per rung; the subject is the FILE, there is no env to set
for rung in sparsecore_topk_ragged sparsecore_topk_c2048; do
  cdk job create --map-dir variants/v04-pipelined-dma/$rung.py \
                          :src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py
  md5sum src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py   # gate this
  uv_venv/bin/pytest -x -q tests/kernels/test_dsv4_sparsecore_topk.py
done
```

**Two runs, not six.** The old instruction was to repeat across four chunk sizes and a
`no_pipelining` flag; §6 refuted every chunk but 2048 and §7a removed the flag, so the shipped
surface is two files. ⛔ **Gate the mapped file's md5 in-job** — `69c8834e130e23999e0284929230da3a`
for `ragged`, `88a67892e8c063b5d9fea96de72e99c4` for `c2048`. A green pytest run tells you nothing
about *which* rung was green unless the md5 is in the log beside it; the kernel scope name
(`…_v4ragged` / `…_v4c2048`) is the second, independent witness.

**⛔ The ragged rung needs the correctness gate more than it needs the timing one, and it is the
only rung where that is true.** Its failure mode is silent: `align(i_hi, LANES)` computed one lane
group short reads fewer words than the row holds, and the tail scores simply never arrive. Nothing
crashes, no bound is violated, `resident_vmem` still holds plausible data from the previous row, and
the kernel returns a full `k` selections — just the wrong ones. It cannot be caught by the timing
harness (`instruments/README.md` is explicit that nothing frozen checks the bytes), and it cannot be
caught at ρ = 1 either, because there the arm is a no-op. **A ragged-distribution correctness run is
the only instrument that sees it.**

`checksum.py` now covers it, first in the arm list, at both `b` and both ρ regimes, and it carries a
guard against passing vacuously: it asserts up front that the `_row_lengths("random", …)`
distribution actually contains rows whose length is **not** a multiple of `LANES` (so the rounding
is exercised at all) and rows that are **partial** (so the arm is not a no-op on this data), and it
fails the run with a named reason if either count is zero. Without that, "6 arms matched" on an
accidentally lane-aligned distribution would be evidence of nothing.

**Which ids exercise the change hardest**, in order:

1. `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240-*]` — the id whose own
   comment in the frozen file says it: *"Anything that reuses a buffer across waves (a
   double-buffered or pipelined score load, say) is correct at one wave by construction and can only
   go wrong at many."* 32 waves **and** ragged rows: both of the things `v4` changes, at once.
   `_row_lengths` cycles the edges `{0, 1, 16, n_p−1, n_p, n_p+1, 2n_p, 2n_p+17, 3n_p, n−1, n}`, so
   the same launch covers `nc = 0` (empty row), `nc = 1` with `live = 1` (a single lane group in a
   1024-word chunk), and `nc = 10` with a 1023-word final chunk.
2. `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` — every row has `i_hi = 0`, so the
   grid is length 0 on every row. This is the id that covers the `pl.when` → grid-of-zero
   replacement, the single most structural thing in the diff.
3. `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b64-n10240-*]` — the decode shape.
4. `test_emit_pairs_survives_ragged_rows[p1-b1024-n10240]` — the `emit_keys` entry point also routes
   through `_topk_body`, so it must see the same `chunk_words`.

**The assertion that would fire** on a wrong-bytes copy: `_mismatch`, which reports *"max score
difference … expected exactly 0.0"* — an exact-selection bar with **no tolerance and no relative
term**. `test_comparison_detects_a_wrong_index` is the frozen meta-test that proves that assertion
can fail; it is not skipped and it is not mine.

⚠ A green pytest run on one rung is **not** evidence for the other: they are different files with
different DMA structure, and `ragged`'s failure mode (a short read) and `c2048`'s (a dropped final
chunk) are different bugs. Run both, or run `checksum.py`, which covers both in one process against
a shared reference.

---

## 9. Things I found that are not my candidate

- **`PARKED` — the `cvt` marginal (−8.20 µs) is confounded, and it is the number the whole "the
  prize is already taken" argument rests on.** It was measured by *deleting* `cvt`. That does not
  only remove a traversal: it leaves `resident_vmem` holding raw f32 bit patterns instead of
  monotone keys, so `hist0`'s L0 bucket distribution, `addupdate_scatter`'s collision density, the
  number of refinement levels that run and `emit`'s selected count all change with it. Given
  `sparsecore-addupdate-scatter-cost-is-mask-density` — a sparse scatter is 0.30× a dense one — a
  data-mediated swing of several µs is entirely plausible, and −8.20 µs is within it. **Route:** the
  clean control is to keep the traversal and change only the data — feed host-pre-converted scores
  and make the in-kernel `cvt` a read/write no-op, so `hist0` sees byte-identical input in both
  arms. That is a ~10-line change to a probe harness, no new kernel. It matters beyond `v4`: the
  same −8.20 µs is what `design.md` uses to argue `v4`'s band runs to 0%.
- ~~`carried`: make the *blocking* DMA ragged.~~ **BUILT — it is `sparsecore_topk_ragged.py`**
  (§1, §5, §6, §7), on the coordinator's instruction, because it must be separable in the pool:
  bundled with the pipeline, a large `g` would sink the whole saving including the part that was
  never at risk. ⛔ **That instruction is now vindicated by measurement, not just by argument:** at
  the plan's own `tc = 1024` the bundled arm is a **+3.15% loss** and the ragged arm alone is a
  **−4.25% win** (§6c). Had they shipped as one candidate, the plan's mechanism would have been
  scored as a 3% regression and the 4.25% win would have been thrown away with it. Modelled saving
  at ρ = 0.400: `13.469 × (1 − 0.40004)` = **8.081 µs = −4.25% harness / −5.19% production**,
  *insensitive to every unknown in §10*. ⚠ Building it turned up one thing the "~2 lines" estimate
  missed: it needs a **dynamic-size** DMA, which Pallas does not obviously permit — that had to be
  AOT-checked before the arm was worth writing, and it does compile (§5).
- ⛔ **CAUTION, and it is a correction to my own §5: the 17-bundle static bound is RETIRED as a
  proxy for `g`.** §5 measured the pipelining machinery at 17 TEC bundles (1172 − 1155) and I used
  it to argue that `emit_pipeline`'s bookkeeping is "order 10–20 ns, comfortably under break-even".
  The measurement says the bookkeeping term **has the opposite sign**: `g_pipe = 248.4 ns` is
  **less** than `g_raw = 517.5 ns`, so there is no additive overhead for 17 bundles to bound. The
  bundle count remains sound as what it literally is — static IMEM code size, and the evidence that
  the grid is a real loop rather than peeled code, which is still load-bearing in §5. It is
  **useless as a cost proxy**, and the general lesson is sharper than this kernel: a static code-size
  bound on a term cannot tell you the term's sign, and quoting it as "comfortably under break-even"
  assumed a sign that was wrong. Route: it belongs beside the AOT-bundle-count reference note in
  §9's `references/08-sparsecore.md` entry, as the caveat on that recipe.
- **Platform fact for `references/`, and it is the most transferable thing here: on SparseCore,
  `emit_pipeline` makes each DMA descriptor CHEAPER, not dearer — 248.4 ns versus 517.5 ns for a
  blocking `start(); wait()`, a factor of 0.480** (`j-53755323`, null spread 0.079%). Every model in
  this tree that reasons about pipelining, including `design.md` and the first version of this
  file's §6, layered a positive bookkeeping term on top of the raw cost. The mechanism is that a
  blocking pair pays the full descriptor latency exposed while the pipeline overlaps issue with the
  previous step's arithmetic — so the two numbers are not "cost" and "cost + overhead", they are
  "cost + exposed latency" and "cost". The practical consequence is that **the correct model is
  `bytes + descriptors × g_arm` with `g` per arm, applied to all descriptors — not to the extra
  ones.** Applying it to the extras only, which is the natural reading, over-charges `tc = 1024` by
  `32 × 269.1 ns = 8.6 µs` here. Route: `references/03-pipelining.md`, in "What a grid step costs".
- **Platform fact for `references/`: a Pallas SparseCore DMA accepts a DYNAMIC size, but its 1D i32
  slice offsets must be 8-word aligned.** Both halves are useful and neither is written down: the
  first is what makes a single-descriptor ragged transfer possible at all (without it, every ragged
  read has to become a chunked one, which is a different and more expensive change); the second is a
  silent compile failure waiting for anyone who computes an offset from a length. Route:
  `references/08-sparsecore.md`, in the DMA section. Evidence: the three-form probe in §5.
- **Platform fact for `references/`: SparseCore Pallas does not enforce that an `emit_pipeline`
  block shape tiles the underlying ref.** A block of 3072 over a flattened `[b·n]` ref with
  `n = 10240` compiles clean and reads across row boundaries at run time. `probes/12` blocker 3
  states the constraint; this adds that **nothing checks it**. Route: one line in
  `references/03-pipelining.md` beside the `BlockSpec` discussion, and it generalises past this
  kernel — any flattened-2D-into-1D pipeline has the same hole.
- **Platform fact for `references/`: `no_pipelining=True` is a *decomposition arm*, not just a debug
  switch.** `jax/_src/pallas/mosaic/pipeline.py:1725-1760` keeps the same `BufferedRef` allocations
  and the same grid and only swaps the async copies for `sync_copy`. So it holds VMEM, chunk count
  and body constant and moves exactly one variable. Route: `references/03-pipelining.md`, in the
  paragraph that currently calls it a debug switch.
- **Platform fact for `references/`, and it is worth more than my variant: SparseCore sequencer
  bundle counts are readable AOT, with no TPU.** `XLA_FLAGS=--xla_dump_to=<dir>` on a host-only
  compile writes `*sparse_core_specific_metadata.txt`, which gives per-computation bundle spans per
  sequencer (SC / TILE_EXECUTE / TILE_ACCESS) and the overlay slot layout. That is a free static
  proxy for kernel code size and for whether a construct unrolled — it is how I established that
  `emit_pipeline`'s grid is a real loop rather than peeled code, which no other artifact showed.
  Route: `references/08-sparsecore.md`, beside the AOT recipe, plus a line in the `aot-compile`
  knowledge fact. Generalises to every SparseCore kernel in the tree.
- **⚠ Platform fact, and a correction others should have: the XLA memory-usage report is BLIND to
  Pallas SparseCore VMEM.** `*-memory-usage-report.txt`'s `Memory Space: vmem` table is empty and
  `after_codegen.txt` says `vmem_bytes: "0"` for a kernel that demonstrably allocates 70+ KiB. So
  that file cannot be used to check a VMEM high-water or a spill risk, and it is identical between
  arms that differ in VMEM. Anyone told the dump gives "VMEM-vs-HBM placement" should read that as
  the *XLA-level* buffers only. Route: same reference edit, stated as the negative.
- **Inherited code-style debt with a route:** `sparsecore_topk.py:140-141` hardcodes
  `num_subcores = 16` and `num_sc_cores = 2` inside `_topk_body`, while `_sc_topk_direct` reads the
  same two values from `sc.num_cores` / `sc.num_subcores` twelve lines away to build the mesh. It is
  the blocking hardware-constants class, it is pre-existing, and no variant should fix it mid-A/B
  because it changes the control. Route: a standalone 4-line cleanup PR after step 3 closes.

---

## 10. Open terms

Everything below is `assumed`, `unmodelled` or predicted at zero. **None of these can fail**, and if
the variant is scored they must be recorded as `unmodelled` and not as passes.

| term | status | why it matters |
| --- | --- | --- |
| **`g`, the SparseCore per-DMA-descriptor cost** | ✅ **CLOSED as an open term — `measured`, `j-53755323`, null spread 0.079%: `g_raw` = 517.5 ns, `g_pipe` = 248.4 ns.** ⚠ Replaced by a narrower open term, below. | It did not fill a hole, it inverted a sign (§6a) and refuted `tc = 1024` (§6d). ✅ **The ragged rung is outside this term entirely** — one descriptor, unchanged from the incumbent, so `g` differences multiply zero. |
| **⚠ NEW: whether `g` TRANSFERS from the microbench to the production kernel** | **`unresolved`, and it is now the binding term.** §6e: charging the incumbent's own 32 descriptors at `g_raw` gives 16.560 µs against a measured `X` of 13.469 µs — the model over-predicts the whole score read by 23% at b = 1024, while very nearly reconciling (2%) at b = 64. | It is larger than the margin it has to decide. `tc = 2048` sits 3.7% inside break-even; a 23% calibration error swamps that. This is *why* §6d calls that rung a wash rather than a win, and it cannot be closed without either a non-deletion measurement of `X` or a microbench with realistic arithmetic behind the DMA. |
| **`f`, the bytes fraction, and `nc`** | ✅ **CLOSED — `measured on host`**, exactly, from the frozen `_row_lengths("random", …)` the harness will run (§6b). Zero cluster time. | It was `assumed` (uniform round-up) and the assumption was pessimistic by 1.0–2.5 points at every chunk. Closing it also surfaced the ρ = 0.391-at-b=64 correction, which is worth more than the precision. |
| **the fusion saving** (deleting `hist0`'s re-read of `resident_vmem`) | **`unmodelled`, credited at ZERO, so it cannot fail — and note the direction.** Bounded 0 to ~4 µs, entirely inside `c2048`. | It is a *second* mechanism riding in `c2048` only. ⚠ It is also the one term that could pull `c2048` clear of `ragged`: 4 µs would be 2.1% of the harness, above the null spread. Nothing separates it from the overlap now that the `no_pipelining` arm is not shipped (§7a) — if `c2048` wins by more than §6 predicts, **that is the first hypothesis, and it needs the `no_pipelining` file to test.** |
| **register pressure / spills in the fused pass-1 body** | **`unmodelled`, predicted at 0.** Not statically observable. ✅ **`ragged` has no exposure at all** — it has no fused body. | The `c2048` body went from two short `parallel_loop`s to one long one at `unroll=4`. A spill per lane group would swamp the prize. Only `/xprof-llo` sees it (§7e step 4). |
| **ρ, and the 189.98 µs / 155.82 µs denominators** | **inherited `measured + reweighting`; the denominators are not re-derived here. ⛔ But ρ at b = 64 IS corrected: the frozen bench runs 0.391, not the production decode 0.850** (§6b). | The entire raggedness term is proportional to `1 − ρ`. The b = 64 correction is a factor of **4.0** on the decode number and the frozen instrument cannot be made to run the production occupancy, so production decode is `modelled` by construction for this candidate — not measurable with the instruments available. |
| **that the pipeline hides what it is supposed to hide** | **`assumed`.** Modelled as "pass 1 is 3× the transfer, so everything but the prologue is hidden." | `probes/06` measured the *incumbent's* overlap at 0.0000 statically; nothing has measured `emit_pipeline`'s on SparseCore. The `ep_copy_in`/`ep_wait_in` scopes are the check. |
| **VMEM high-water 78.50 KiB of 512 KiB** | **`modelled`, and now known to be unrecoverable statically** — 70.50 KiB observed for the incumbent (`report.md` §7 LLO dump) + 8.00 KiB counted in the v4 jaxpr. The XLA memory-usage report is blind to Pallas SC scratch (§5, §9), so this cannot be firmed up without a pod. | Comfortable, and not the exposure. Listed so it is not mistaken for a measurement. |
| **decode** | **predicted `not resolvable`.** At the harness's own ρ = 0.391: `ragged` −1.20%, `c2048` −1.39%, against a **3.37%** null spread. Scaled to the production ρ = 0.850: `ragged` −0.30%, inside the floor by 11×. | ⛔ A b = 64 reading in that range is `not resolvable`, **never** "no effect", and **never a decode result** — see §6b. The floor is one-sided: a `g`-driven regression *is* resolvable, so this arm can produce a null or a loss but not a win. |
| **composition with `v3`, if `v3` wins** | **`unbuilt`, by instruction.** | `v3` fuses `emit` and the refinements into the L0 walk and adds a straddler buffer. ⛔ **Which of my rungs composes has now changed.** `ragged` composes with *anything*: it is one expression in the DMA's length operand, touches no pass structure, and its saving is `X · (1 − ρ)` regardless of what the rest of the kernel does. `c2048` is the one with assumptions in `v3`'s way, and two of them move: (a) "pass 1 is 3× the transfer, so hiding capacity is free" gets *stronger* — the pipeline body would enclose `cvt ⊕ v3`'s fused L0, so the hiding ceiling `min(transfer, body)` rises and the prologue stays the only exposure; (b) my absolute saving in µs is unchanged but my **percentage grows**, because `v3` shrinks the denominator — ⚠ which, given `c2048` is 0.39% ahead of `ragged` against a 0.72% floor, is the one thing that could lift that comparison out of the noise. The composed shape is my `emit_pipeline` with `v3`'s fused body substituted for `l0_block`'s inner `body`: same grid, same `live` clip, same write-through to `resident_vmem`, which `v3`'s later passes still need. Check VMEM on composition: `v3`'s straddler buffer plus my 8 KiB double buffer against the same 512 KiB, which §4d of the design says fits. |
