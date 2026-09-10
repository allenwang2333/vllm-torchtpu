# `v11` — retiled output emit

Built. Static evidence class **`static-compile`** throughout; no timing job, no pod job, no pytest
run. Sign convention: every `Δ time %` below is `(new − old)/old` of time — negative is faster.

Artifacts in this folder: `sparsecore_topk.py` (the subject, md5
`3dc249276fa68985e29f4da8e9c27fbd`), `variant.diff`, `aot_v11.py` + `aot_v11.txt`,
`shape_matrix.py` + `shape_matrix.txt`, `mosaic_dma.py` + `mosaic_dma.txt`, `hlo/`.

---

## 1. What you built

`_sc_topk_direct` now declares its Pallas output as `jax.ShapeDtypeStruct((b, k), jnp.int32)`
instead of the flat `(b*k,)`, writes each row with `out_hbm.at[row_c]` on that rank-2 HBM ref
instead of `out_hbm.at[pl.ds(row_c * k, k)]` on a flat one, and returns the kernel's result
unreshaped. Pallas pins only the minor-to-major dim order for a TPU custom call
(`jax/_src/tpu_custom_call.py:374-375`), so XLA is free to give a rank-2 i32 result the `T(8,128)`
tiling the consumer wants — and the reshape/retile that used to sit between the call and the
consumer then has nothing to do. The public signature and every downstream shape are unchanged:
`sparsecore_topk()` still returns `i32[b, k]`, and the two-stage path's `local_slots.reshape(b, p,
k_p)` / `cand_scores.reshape(b, p*k_p)` still see a `(b*p, k_p)` array, exactly as before.

`diff -u` against `probes/17/_gen_base.py` (byte-identical to the incumbent, md5
`44353b347c96e0eedbd742c710d4540a`): **77 lines, 5 hunks**.

| hunk | anchor | ± | what |
| --- | --- | --- | --- |
| 1 | `_topk_body` params `:69-70` | −2 / +2 | shape comments `i32[b * k]` → `i32[b, k]` on `out_hbm`, `keys_hbm`. Comment only |
| 2 | `_topk_body` body `:87` | −1 / +1 | same, in the `keys_hbm` prose. Comment only |
| 3 | out DMA `:313-333` | −2 / +10 | **functional**: both indexers `at[pl.ds(row_c * k, k)]` → `at[row_c]`, + 8 comment lines |
| 4 | `out_type` `:459-464` | −1 / +6 | **functional**: `(b * k,)` → `(b, k)`, + 5 comment lines |
| 5 | return `:500-504` | −3 / +2 | **functional**: drops all three `.reshape(b, k)` |

Three functional edits at four sites; the rest is comment. `py_compile` clean.

---

## 2. The file plan, scored

⚠ I have **no §9 plan section** — this node was promoted from stage E after §9 was written, so
`design.md` §5's `v11` block (lines 991-1037) is the plan and it has no §4 "Files" table. Scored
against the closest thing it states, "**Patch is four edits**: `:451`, `:317`, `:323`, `:491`":

| plan | produced | divergence |
| --- | --- | --- |
| `:451` `out_type` flat → `(b, k)` | hunk 4, at `:464` after comments | same edit, line number shifted by the comment block |
| `:317` out DMA indexer | hunk 3, `out_hbm.at[row_c]` | same |
| `:323` keys DMA indexer | hunk 3, `keys_hbm.at[row_c]` | same |
| `:491` caller-side `out.reshape(b, k)` | hunk 5 — **three** reshapes removed, not one | the plan counted the indices-only return; the `emit_keys` return at `:487-489` carries two more (`slots.reshape`, `keys.reshape`). Same edit, three sites |
| — | hunks 1, 2 | added: the ref shape comments said `i32[b * k]` and would now be lying. `code-style.md` §2 makes this mandatory, not optional |

No other divergence. The plan said "do not improve it on the way in" and nothing was improved.

---

## 3. `code-style.md` checklist

Scored twice, because the incumbent file predates the checklist: **`diff`** = does my change comply;
**`file`** = does the file comply. A `❌ (file)` that is `✅ (diff)` is inherited from the incumbent
and is identical in the baseline arm.

| # | item | diff | file |
| --- | --- | --- | --- |
| 1 | Dimension glossary at the top | ✅ n/a — I added no dimension | ❌ — module docstring only, no `# Dimensions.` block. Pre-existing |
| 2 | Trailing `# [shape] dtype` on every ref param | ✅ — hunks 1-2 exist *because* of this rule; `out_hbm` / `keys_hbm` now read `# i32[b, k]` | ✅ |
| 3 | Extents from the refs, not the config | ✅ — `b`, `k` were already the launcher's closed-over ints; I read no new shape | ✅ |
| 4 | jaxtyping on the wrapper, bare refs inside | ✅ — unchanged | ❌ — `sparsecore_topk` annotates `jax.Array`, not jaxtyping. Pre-existing |
| 5 | `# Inputs. / # Outputs. / # Scratch.` sectioning | ✅ — unchanged | ❌ — the params are grouped by a comment above `_topk_body`, not the three section markers. Pre-existing |
| 6 | Body order indices → shapes → sems → init → loads → compute → store → epilogue | ✅ — the store stayed exactly where it was, inside the same `@pl.when(out_gate)` | ✅ |
| 7 | Single chars only for indices/annotated tensors/extents | ✅ — `row_c` unchanged | ✅ |
| 8 | Memory space in the ref name where two spaces share a signature | ✅ — `out_hbm` / `rowbuf_vmem` unchanged, and hunk 1's comment now names the space *and* the rank | ✅ |
| 9 | Comments name the compromise; one-line module docstring | ✅ — the 8-line DMA comment says why `at[row_c]` and not `at[pl.ds(...)]`, and records that `at[pl.ds(row_c, 1)]` does not lower. No number is stated | ⚠ the docstring is 15 lines, not one. Pre-existing |
| 10 | `NotImplementedError` for unsupported, not `ValueError` | ✅ — untouched | ✅ — `:444` raises `NotImplementedError("SparseCore is not available")` |
| 11 | Four files `api/base/pallas_mosaic_tpu/..._kernel` | ✅ — untouched | ❌ — one file. Pre-existing, and `code-style.md` §10 "when not to use the framework at all" arguably licenses it |
| 12 | Further splits at what the module may touch | ✅ n/a | ❌ — same |
| 13 | Config is a frozen kw_only slots pydantic dataclass | ✅ n/a — I added no config | ✅ n/a — there is no config class |
| 14 | **[blocking]** Tested against the XLA `_fwd`, tolerance at the call site | ✅ — `test_dsv4_sparsecore_topk.py` compares against a NumPy exact-top-k reference and the comparison is **exact** (`_mismatch` returns a reason string; there is no tolerance to site). See §8 | ✅ |
| 15 | Every f32 contraction in `_fwd` pins `Precision.HIGHEST` | ✅ n/a — top-k performs no contraction | ✅ n/a |
| 16 | **[blocking]** Hardware constants from `pltpu.get_tpu_info()` | ✅ — I introduced none | ❌ — **`num_subcores = 16` and `num_sc_cores = 2` are hardcoded at `:94-95`** while the launcher at `:453-458` correctly reads `sc.num_cores` / `sc.num_subcores`. Pre-existing |

**Blocking classes, stated plainly.** Item 14 is `✅` both ways. Item 16 is `❌ (file)`: the kernel
body hardcodes two chip constants the same file reads properly one function up. My diff neither
introduces nor removes them, and the incumbent arm carries the identical `❌` — so this does not
discriminate between the arms, but if step 4 scores *the file* rather than *the change*, it blocks
`v11` and every other variant and the incumbent too. Route: a one-line fix (`sc = info.sparse_core`
is already in scope in the launcher and would have to be threaded in as `functools.partial` kwargs
alongside `b`, `n`, `k`) — it belongs to whoever owns the file, not to a variant, because doing it
inside one variant makes that variant's diff no longer a clean A/B.

---

## 4. The §7 risk table, with an outcome on every row

`design.md` §5's `v11` block carries no numbered §7 table either; the rows below are its named
hazards plus the one open term §10 records against this node.

| # | risk | threatens | outcome |
| --- | --- | --- | --- |
| R1 | The retiled store becomes strided — "16 runs of 512 B at stride 1024 words" — and the strided DMA costs more than the retile it deletes | perf | **still unchecked — partially priced.** The stride is now *derived from an observed layout* rather than assumed (§5), and its static footprint is **+2 SparseCore sequencer bundles**, with the TEC and TAC programs byte-identical. What remains unobservable without a TPU is the **runtime descriptor count**. Carried to §10 as `unmodelled` |
| R2 | A layout change is `exact` in arithmetic but changes the *store pattern*, so a wrong stride lands correct values at wrong offsets, silently | **correctness** | **still unchecked** — BRIEF §6 forbids me from running the frozen test, and no static artifact can rule out a mis-strided store. §8 names the assertions that fire: `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]`'s `assert (idx == -1).all()` (every landed word is a known constant, so any misplaced store is visible), and `_mismatch`'s `"index X out of range for n"` / `"duplicate indices"` / `"-1 padding is not a suffix"` in `test_exact_topk_full_rows[*-p1-b1024-n10240]`. ⛔ **Per BRIEF §8 item 4 an unchecked correctness row blocks the variant, so `v11` is blocked until the correctness runner reports.** It is unchecked for a reason the brief itself imposes, not for want of a signal |
| R3 | The 10.007 µs prize was measured on a `ROOT reshape` that does not exist in the production entry — the in-model form is an `S(1)` staging copy, a different op, and no arm had deleted *it* | perf (the prize itself) | **did not happen — the reading is §5's in-model row.** Compiling probe 14's real entry on both arms, `%reshape.1 = s32[1024,2048]{1,0:T(8,128)S(1)}` (`used_scoped` 2 MiB) **and** the `%copy-start.5`/`%copy-done.5` pair that drained it to HBM are both **absent** from the `v11` module. Buffer assignment shows why: the `cond`'s 8 MiB result buffer moves out of the VMEM pool and becomes the HBM live-out, which both branches now write in place |
| R4 | Compiling `streamindex_topk` standalone forces its ROOT layout, so the retile might be an artifact of where compilation was cut | perf | **did not happen** — closed by `probes/14` Q2 before I started, and re-confirmed here on the real entry (`hlo/inmodel-{base,v11}.txt`, md5s `1fa8e95f…` vs `ab97c4de…` — they differ, so the two arms really are two arms) |
| R5 | Feasibility was closed at exactly one shape (`b=1024, n=10240, k=2048, P=1`); the `P>1` path calls `_sc_topk_direct` twice more, once with `emit_keys=True`, and neither had ever been compiled with a 2-D `out_type` | correctness (lowering) + perf | **happened, on the perf leg.** All 7 frozen-test shape ids compile on both arms (`shape_matrix.txt`) — so the lowering leg is clear, including `emit_keys`. But at every `P>1` id `v11` trades one `reshape` for one `copy` and ENTRY *grows*: `p2-b16-n10240` 41→42, `p4-b16-n133120` 39→40, `p4-b64-n133120` 42→43, `p4-b1024-n133120` 48→52. Fallback taken: **none — the candidate is scoped to `P=1`**, which is both production shapes. See §7 and §9 |
| R6 | `v11` is subsumed by `H3`, which fuses the consumer and leaves no HBM output to retile | composition | **did not happen, and is not designed around** (per dispatch). Lines touched, so step 4 can check the conflict mechanically: `sparsecore_topk.py:69-70`, `:87`, `:313-333` (the out/keys DMA inside `@pl.when(out_gate)`), `:459-464` (`out_type`), `:500-504` (the return). `H3` must touch `out_type` and the out DMA, so the collision is exactly hunks 3-5. Composes with everything else — it is an edge move and changes no interior state |

---

## 5. `static-compile` evidence

Compiled on the host with no TPU:
`topologies.get_topology_desc("tpu7x:2x2x1").devices[0]` + `SingleDeviceSharding` +
`jax.jit(...).lower(...).compile()`, under probe 12/17's device-pinning prologue. 0.8-10.5 s per
compile.

**Negative control** (`aot_v11.py:negative_control`, probe 17's DMA form — an 8 MiB VMEM scratch
DMA'd to HBM, so the refusal must come from the allocator and not from the lowering):

```
AOT-NEGCTL-8MiB-vmem: FAILED as required=True
  'memref.alloca' op E3000: CompileTimeSparseCoreAllocationFailure: current allocation offset
  upper bound (33554431 words) exceeds the legitimate user allocatable offset upper bound
  (2097151 words) in memory space spmem when allocating 33554432 words
```

⚠ My first attempt at this control used `o[...] = jnp.zeros_like(o)` and failed with
`NotImplementedError: Swap does not support storing to HBM` — the MLIR layer refusing, not the
SparseCore allocator. It proved nothing and was replaced. Noted because it is an easy control to
build wrong.

### (a) Standalone `_sc_topk_direct`, b=1024 n=10240 k=2048

| | incumbent | `v11` |
| --- | --- | --- |
| ENTRY instructions | **13** | **12** |
| ROOT | `%reshape.4 = s32[1024,2048]{1,0:T(8,128)}`, `used_scoped` **4 MiB** | `%…call-done = s32[1024,2048]{1,0:T(8,128)}` |
| custom-call result | `s32[2097152]{0:T(1024)}` | `s32[1024,2048]{1,0:T(8,128)}` |
| output allocation (8 MiB, live-out) holds | `reshape.3` — a **second** 8 MiB buffer | the kernel's own result |
| `temp_size_in_bytes` | 84,017,024 | 84,017,024 (unchanged) |
| `generated_code_size_in_bytes` | 153,088 | 129,536 (**−23,552**) |
| SC sequencer bundles, topk `called_computation.1` | 136→**196** | 136→**198** (+2) |
| TEC / TAC overlays | — | **byte-identical** |

Peak temp does not move because the deleted 8 MiB temp shared an offset with a live 40 MiB buffer;
the saving is traffic and instructions, not high-water. No spills in either arm.

### (b) In-model — the real entry, `_streamidx_topk_jax` (probe 14's rebuild)

**This is the dispatch's central question, and the answer is: `v11` deletes the `S(1)` staging copy.
It does not move it or make it cheaper — it is gone, and so is the reshape that fed it.**

| | incumbent | `v11` |
| --- | --- | --- |
| module lines | 5348 | 5343 |
| ENTRY instructions | **220** | **218** |
| the retile | `%reshape.1 = s32[1024,2048]{1,0:T(8,128)S(1)}`, `used_scoped` **2 MiB** | **absent** |
| the staging copy | `%copy-start.5` / `%copy-done.5`, `S(1)` → HBM, 8 MiB | **absent** |
| `copy-start` count | 22 | 20 |
| `S(1)`-annotated shapes | 406 | 398 |
| `s32[2097152]` results | 3 | **0** |
| `cond` result buffer | allocation **224** (VMEM, color 1) @ 33,562,624, 8 MiB — shared by `reshape.1`, `compare_select_fusion` and the `cond.2.clone` phi | allocation **3** (HBM live-out) @ 0 — shared by `sc_topk_direct…6`, `compare_select_fusion` and the phi |
| temp allocation 222 | 151,814,400 | 151,798,016 (−16,384) |
| VMEM color-1 allocation 224 | 67,043,328 | 67,043,328 — size unchanged, but 8 MiB of the pool is no longer occupied |
| `temp_size_in_bytes` | 152,663,264 | 152,630,528 (−32,736) |
| `generated_code_size_in_bytes` | 3,883,520 | 3,860,480 (−23,040) |

Mechanism, from buffer assignment rather than from the op list: with a flat kernel result the
`cond`'s 8 MiB `T(8,128)` output buffer could not alias the call's output, so XLA placed it in VMEM,
had branch 0 materialise `reshape.1` into it and branch 1 materialise `compare_select_fusion` into
it, then DMA'd all 8 MiB out to HBM. With a 2-D result the `cond` buffer **is** the HBM live-out and
both branches write it in place.

Two consequences worth separating:

- **On branch 0 (the kernel path)**, the incumbent's post-kernel HBM traffic is 8 MiB read +
  8 MiB written = **16 MiB**; `v11`'s is zero. That is the same 16 MiB the standalone `ROOT reshape`
  moved, which is what the 10.007 µs priced — so the ceiling transfers **by bytes**, not by op
  identity, and the split into two instructions if anything makes the incumbent slightly worse than
  the ceiling, not better.
- **On branch 1 (`_fast_path`)**, the incumbent writes 8 MiB to VMEM and then DMAs it to HBM;
  `v11` writes 8 MiB to HBM once. So a saving also lands on the steps where the gate skips the
  kernel entirely. This is **outside** the prize as stated and is not claimed in §6.

⚠ Two instrument defects were found and fixed before these numbers; both would have produced a
confident wrong answer. (i) The first in-model A/B returned **byte-identical HLO** for the two arms:
`streamindex_topk.py:25` binds `from …sparsecore_topk import sparsecore_topk` at import, and
`jax.jit` caches the inner pjit **jaxpr** on `(function object, avals)`, so rebinding the attribute
let arm 2 reuse arm 1's trace. Fixed by seeding `sys.modules`, `importlib.reload`,
`jax.clear_caches()` and an identity assertion, with the arms run `v11` first so a residual cache
bug would show as base-looking-like-v11. Every in-model number above was also independently
reproduced in **two fresh single-arm processes** (§b's buffer-assignment run), which cannot have the
bug at all. (ii) The negative control above.

### (c) Every frozen-test shape, both arms (`shape_matrix.txt`)

All 14 compiles succeed — feasibility is now closed at all 7 ids, not just `b=1024`, and the
`emit_keys=True` stage-1 call lowers with a 2-D `out_type`.

| shape id | P | ENTRY | `reshape` | `copy` |
| --- | --- | --- | --- | --- |
| `p1-b64-n10240` ← **production** | 1 | 10 → **9** | 1 → **0** | 1 → 1 |
| `p1-b1024-n10240` ← **production** | 1 | 13 → **12** | 1 → **0** | 0 → 0 |
| `p2-b16-n10240` | 2 | 41 → 42 | 4 → 3 | 6 → **7** |
| `p4-b16-n133120` | 4 | 39 → 40 | 4 → 3 | 5 → **6** |
| `p4-b64-n133120` | 4 | 42 → 43 | 4 → 3 | 4 → **5** |
| `p4-b16-n133120-k512` | 4 | 39 → 40 | 4 → 3 | 5 → **6** |
| `p4-b1024-n133120` | 4 | 48 → **52** | 4 → 3 | 7 → 7 |

The `P>1` rows are a genuine adverse finding and I read the ENTRY to find out why. At
`p2-b16-n10240` the incumbent has `%reshape.31 = s32[16,4096] reshape(%copy-done.1)` folding the
flat `s32[65536]` stage-1 output into stage 2's operand for free; under `T(8,128)` the same
`(32,2048) → (16,2,2048) → (16,4096)` regrouping is **not** a bitcast, and XLA inserts a physical
`%copy.1 = s32[4,16,8,128]{3,1,2,0:T(8,128)}`. So on the two-stage path `v11` does not delete a
relayout — it **relocates** it from stage 2's output to the stage-1/stage-2 seam, where it is
larger (stage 1 emits `(b*p, k_p)`, twice the rows of the final `(b, k)` at P=2). This is BRIEF §4's
warning landing exactly as written, and it is why the candidate is scoped to `P=1`.

### (d) The strided store — Mosaic MLIR and the layout it is given

At the **Mosaic op** level the two arms are indistinguishable, which is what `probes/17` found and I
reproduce (`mosaic_dma.txt`): the enqueue is `2048xi32 vmem → 2048xi32 hbm` in both, and 3
`enqueue_dma` in both. `v11` folds the row into `memref_slice %arg4[%10, %c0] :
memref<1024x2048xi32, hbm> -> memref<1x2048xi32, hbm>` plus a `memref_squeeze` back to
`memref<2048xi32, hbm>` (`memref_squeeze` 6 → 8), where the incumbent slices a flat
`memref<2097152xi32, hbm>` directly. That MLIR is pre-`tpu-infer-memref-layout` and carries no
tiling, so it cannot settle the question by itself.

The layout that does settle it is **observed in the compiled HLO**: the custom call's result layout
is `s32[1024,2048]{1,0:T(8,128)}`. Under `T(8,128)`, one logical row of 2048 i32 lives in 16 tiles
contributing 128 words each, and consecutive tiles of a tile-row are 1024 words apart. Independently,
`jax._src.pallas.mosaic.tpu_info.infer_tiling` read **from the platform** (not hardcoded) gives
1-D i32 → `(128,)` and 2-D `(1024,2048)` i32 → `(8,128)`.

| | incumbent | `v11` |
| --- | --- | --- |
| runs per row | 1 | **16** |
| run length | 8192 B | **512 B** |
| stride | — | **4096 B** (1024 words) |
| bytes per row | 8192 | 8192 (unchanged) |
| 32 B SparseCore DMA granule | 256 granules/run | 16 granules/run — **whole, no partial granule** |
| runs across the kernel, prefill `b=1024` | 1,024 | 16,384 |
| runs across the kernel, decode `b=64` | 64 | 1,024 |

So: **yes, the retiled output store is strided**, at a granule-aligned 512 B run and a 4096 B stride.
What that costs in *descriptors* is answered in §10, not here.

---

## 6. Prediction, restated as a number to be scored

**Mechanism.** Delete, from the production entry, one `s32[2097152] → s32[1024,2048]{T(8,128)}`
retile and the 8 MiB `S(1)`→HBM staging copy that drained it: 8 MiB read + 8 MiB written on HBM per
kernel-path call, and a 2 MiB scoped VMEM reservation. Pay instead a strided output store inside the
kernel: same 8192 B/row, 16 granule-aligned 512 B runs at 4096 B stride instead of one contiguous
run, and +2 SparseCore sequencer bundles.

| | `Δ time %` |
| --- | --- |
| harness denominator, **189.98 µs** | ≤ **−5.27 %** |
| production denominator, **155.82 µs** | ≤ **−6.42 %** |

Both are **ceilings** (`≤`), and both are the design's own number: 10.007 µs `measured (j-9b520877)`
divided by each denominator, less the strided-store cost, which is `unmodelled`.

**What the static evidence now changes about that ceiling.** It does not raise or lower it, but it
substantially firms the *denominator of the claim*:

- The dispatch's caveat — "10.007 µs was measured on the ROOT reshape, and no arm has yet deleted
  the staging copy" — is discharged. The in-model form is a different pair of ops moving the **same
  16 MiB of HBM traffic**, and `v11` deletes both. The price transfers by bytes.
- The build diverges from the plan in one way that *widens* the prize and which I am deliberately
  **not** folding into the number: the fast-path branch also loses an 8 MiB VMEM round trip and its
  DMA. It is real, it is visible in buffer assignment, and it has never been priced — so it is a
  §10 open term credited at **zero**, not a bigger prediction.
- The build diverges in one way that *narrows the scope*: the prediction applies at **`P=1` only**.
  At `P>1` the relayout is relocated rather than deleted (§5c), so the honest prediction there is
  **a regression of unquantified size**, not a win.

Scoped restatement, to be scored per shape rather than pooled:

| bench arm at `--ctx 9216` | P | predicted |
| --- | --- | --- |
| `b=1024` | 1 | ≤ −5.27 % harness / ≤ −6.42 % production. This is *the* number |
| `b=64` | 1 | same mechanism, 1/16 the bytes (512 KiB output). Absolute saving ≈ 0.63 µs `modelled`; the **relative** saving is smaller than at `b=1024` because per-row kernel cost roughly triples at small batch |
| `b=16` | **2** | **predicted to regress.** Not part of the prize and must not be pooled with the other two |

---

## 7. How the measurer runs me

**The swap.** One file, mapped over the installed kernel:

```
--map-dir kernel_work/sparsecore-topk/variants/v11-retiled-output-emit/sparsecore_topk.py
        → src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py
```

In-job md5 gate (BRIEF §7 — the docker tag is mutable): `v11` arm must read
`3dc249276fa68985e29f4da8e9c27fbd`; control arm must read `44353b347c96e0eedbd742c710d4540a`.

**Arm activation is fully covered by that md5**, and this is worth stating because it is unusual:
`v11` has **no env var, no flag and no runtime switch** — the whole variant is the file. There is
nothing that can be "set but not fire". If a profile is captured anyway, the positive tell is the
kernel's HLO result shape, `s32[1024,2048]{1,0:T(8,128)}` in place of `s32[2097152]{0:T(1024)}`, and
the disappearance of the post-call `reshape`/`copy` pair. ⚠ The Pallas scope name is
`sc_topk_direct_b{b}_n{n}_k{k}[_keys]` in **both** arms — I deliberately did not add a `_2d` suffix,
because renaming the kernel would break op-name matching between the arms in exactly the profile
that would be used to check the arm. Match on shape, not on name.

**Arms and shapes.**

```
python sctopk_2stage_bench.py --decomp --ctx 9216 --batches 16 64 1024 --rounds 9
python sctopk_2stage_bench.py --passes --ctx 9216 --batches 1024      --rounds 9
```

`--ctx 9216` every time (it is not the default). Read `us_median`, not `us`.

⛔ **Do not pool `b=16` with `b=64` and `b=1024`.** At `--ctx 9216`, `b=16` resolves to `P=2` and
`b∈{64,1024}` to `P=1`, and `v11` is predicted to move them in **opposite directions** (§5c, §6).
Pooling them would average a win against a loss and report a null. If only one batch can be
afforded, run **`b=1024`** — it is the production prefill bucket, the shape the prize was priced at,
and the one with 32 waves.

**Checksum arm: not needed, provided the correctness job runs the same md5.** The change does alter
what the DMA lands, so a wrong stride would be silent in a timing harness — but the frozen unit test
is a *full-content* comparison against a NumPy exact-top-k reference over every one of the
1024 × 2048 output words, which is strictly stronger than a checksum. Gate the same md5 in both
jobs and the coverage is complete. Do **not** add an arm to the frozen bench.

**Rival arms for a compile-time constant of mine:** none. `v11` introduces no tunable — the only
"knob" is the rank of `out_type`, which is binary and is the variant itself.

---

## 8. How the correctness runner runs me

```bash
cd <checkout> && python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

Unmodified frozen test, md5 `b3241859e471fc87fa6e7a50e4ffec4b`, with the same `--map-dir` and the
same in-job md5 gate as §7. 68 tests; baseline **68/68 PASSED** (`j-62275e52-1498-4e01-918e`).
If sharded with `-k`, check the union with `--collect-only -q`, not by eye.

**Which of the 68 exercise this change hardest**, and what each would catch:

| id | why it is the hard one |
| --- | --- |
| `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` | **the single most discriminating test.** `write_empty` fills every one of 1024×2048 output words with `-1`, and the assertion is `assert (idx == -1).all()`. Every landed word is a known constant, so a store that lands 512 B runs at the wrong stride shows up as a non-`-1` no matter which offsets it hit |
| `test_exact_topk_full_rows[strict_desc\|random\|heavy_ties-p1-b1024-n10240]` | the production shape, `P=1`, 32 waves, every row written by the edited DMA. Catches a mis-strided store as `_mismatch`'s `"index X out of range for n"`, `"duplicate indices"` or `"returned N valid entries, expected M"` |
| `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240]` | rows of length 0, 1, 16 and `k` mixed in one batch — the `write_empty` and partially-filled row stores, and `"-1 padding is not a suffix"` |
| `test_emit_pairs_is_bit_identical_to_the_incumbent[random\|heavy_ties\|all_negative-p4-*]` | the **only** tests that execute `keys_hbm.at[row_c]`, my second edited indexer. Asserts `n_diff == 0`. ⚠ note `emit_keys` is reachable only on `TWO_STAGE_SHAPES`, so the keys DMA is never exercised at `P=1` — that is the incumbent's structure, not something my change introduced |
| `test_exact_topk_full_rows[*-p2-b16-n10240]`, `[*-p4-b1024-n133120]` | the four `P>1` ids are the ones whose ENTRY grew (§5c). They check that the relocated relayout is *correct*; they say nothing about its cost |
| `test_exact_topk_full_rows[*-p1-b64-n10240]` | the decode production shape, 2 waves — the low-wave end of the same store |

A green run on all 68 discharges risk R2 and unblocks the variant. A red run on the first row above
is the signature of a mis-strided store specifically.

---

## 9. Findings that are not my candidate

1. **`design.md` §5 `v11` Note has an arithmetic slip: "the same 40 MiB of movement".** The retiled
   buffer is `s32[1024,2048]` = **8 MiB**, and the design's own rate check agrees —
   10.007 µs ÷ 1.251 µs/MiB = 8.0 MiB. 40 MiB is the `f32[1024,10240]` **input**. The prize itself is
   unaffected (it was computed from the rate, not from the 40). **Route:** correct the sentence in
   `design.md` §5 at step 4; nothing else cites it.
2. **`v11` is a `P=1`-only win; at `P>1` it relocates the relayout to the stage-1/stage-2 seam and
   makes it bigger** (§5c). Both production shapes are `P=1`, so this does not sink the candidate —
   but it is a live hazard for any *other* variant that changes `_pick_partition`, or for any future
   config that pushes a production shape to `P>1`. **Route:** a one-line warning on the `v11` node in
   `design.md` §5, and a note on whichever node owns `_pick_partition`.
3. **The output-side saving is not confined to steps that run the kernel.** Deleting the `S(1)`
   staging also removes an 8 MiB VMEM round trip from `_fast_path` (branch 1), i.e. from the ~1/4 of
   prefill rows and every step where `max(seq_lens) > 2048` sends the gate the other way. Never
   priced. **Route:** the measurer gets it for free in any e2e number; if a microbench number is
   wanted, it is a `static-compile`-sized question and belongs to whoever next opens
   `streamindex_topk.py:858-864`. Credited at **zero** in §6 and carried in §10.
4. **Platform fact for `references/04-layout-precision.md`:** *a Pallas TPU custom call's output
   rank is a layout lever.* Pallas pins only minor-to-major dim order
   (`jax/_src/tpu_custom_call.py:374-375`); declaring `out_type` at the consumer's **rank** lets XLA
   hand the call the consumer's **tiling**, and the relayout then never exists. This is a genuine
   member of that file's "Arranging so the relayout never exists" family — and finding 2 above is
   equally a member of its "five that did not survive" family, since the same move at `P>1` merely
   relocated the copy. **Route:** propose both to `references/04-layout-precision.md` at step 4.
5. **Platform fact, method:** `XLA_FLAGS=--xla_dump_to=<dir>` on a host-only AOT compile yields
   `*-buffer-assignment.txt`, `*-memory-usage-report.txt` and
   `*sparse_core_specific_metadata.txt` — which is where the VMEM-vs-HBM placement of an `S(1)`
   value and the **SparseCore sequencer bundle counts** are legible, neither of which is in
   `compiled.as_text()`. This is how §5(a)'s "+2 bundles" and §5(b)'s allocation table were read, at
   zero cluster cost. **Route:** `references/08-sparsecore.md`, beside the existing AOT-compile note.
6. **Operational:** the shared `/tmp/libtpu_lockfile` serialises host AOT compiles across agents;
   two of my runs died with `ABORTED: Internal error when accessing libtpu multi-process lockfile`
   while sibling variant agents were compiling. `sudo rm` would break their run. **Route:** a
   retry-with-backoff loop (as in this folder's scripts) belongs in the BRIEF's AOT-compile row.

---

## 10. Open terms

| term | status | what it rests on |
| --- | --- | --- |
| **Strided-store descriptor count** | ⛔ **`unmodelled`** | The store *is* strided — 16 granule-aligned 512 B runs at 4096 B stride replacing one contiguous 8192 B run, 1,024 → 16,384 runs at prefill `b=1024` and 64 → 1,024 at decode `b=64` (§5d). What I **cannot** produce statically is whether the SparseCore DMA engine issues **one strided descriptor or sixteen**. The only observed evidence bearing on it is that the topk kernel's SC sequencer program grows by exactly **+2 VLIW bundles** with the TEC and TAC programs byte-identical — consistent with one enqueue whose descriptor gained a stride field, and inconsistent with Mosaic having expanded the store into a 16-iteration loop, which would cost far more than 2 bundles. That is suggestive, not decisive. **Per BRIEF §8 item 10 this comes back as `unmodelled`, not as a pass.** It is the term that could eat the entire prize |
| **Correctness under the new store pattern** | **unchecked** | R2. Blocks the variant until the correctness runner reports. §8 names the assertions |
| **The `_fast_path` (branch 1) saving** | **credited at zero** | Real and visible in buffer assignment (§5b), never priced, and deliberately excluded from §6's number. Cannot fail; will be recorded as `unmodelled` |
| **`P>1` regression size** | **`unmodelled`** | §5c establishes the mechanism and the direction from ENTRY op counts, not the magnitude. The `b=16` bench arm will measure it — as a **separate** number (§7) |
| **The prize's own denominator** | **inherited, `modelled`** | BRIEF §7: every design percentage has a `modelled` denominator, every probe ran at ρ=1, and nothing has run at the production ρ=0.40005. The 10.007 µs is `measured (j-9b520877)`; the two percentages built on it are not |
| **Composition with `H3`** | **noted, not designed around** | `H3` subsumes `v11`. Lines touched are listed in R6 so the collision is checkable mechanically |
