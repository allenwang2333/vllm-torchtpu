# `v12` (= U2) — `f32` kernel input, reinterpret in-register

Sign convention, binding throughout: every percentage is `(new − old)/old` of **time**. Negative is
faster. A signed percentage appears only in a column headed `Δ time %`.

Subject: `variants/v12-f32-input/sparsecore_topk.py`, md5 **`408ef72e9afc935c928f2803764878cc`**,
edited from the incumbent copy `44353b347c96e0eedbd742c710d4540a` (commit `b11b6eb6`).
Live tree unchanged (`src/…/sparsecore_topk.py` still `44353b34…`). All three instruments unchanged
(`b3241859…`, `bdd978e6…`, `08375d70…`).

**Headline: the one open term is closed at zero, with evidence, not credited.** The seven in-body
bitcasts emit **no machine instruction at all** — the scheduled SparseCore VLIW bundle listing of
the variant is byte-identical to the incumbent's except for a two-register allocation swap. The
40 MiB buffer is genuinely deleted — not renamed, not re-staged as a copy, and not relocated into
VMEM (`vmem` is unchanged to the byte). Every headline number is re-derived in a fresh single-arm
process, so none of it can be a `jax.jit` cache hit.

---

## 1. What I built

The caller used to hand the kernel `int32` by materialising a whole second copy of the score array:
`sparsecore_topk.py:438` did `words = jax.lax.bitcast_convert_type(scores, jnp.int32)` and `:486`
passed `words.reshape(-1)`. XLA planned that as a `kLoop` fusion with its own disjoint 40 MiB temp
buffer, so the entire score array was read and rewritten before the kernel started. `v12` deletes
it: the kernel now takes `scores` as `f32` and does the reinterpret in-register.

It is three edits plus a tag, exactly as `design.md` §5 predicted, because **the DMA is
dtype-checked rather than byte-oriented** — `tpu.enqueue_dma` rejects a source/target element-type
mismatch, so the resident VMEM slice has to move to `f32` too. Its *contents* are int32 monotone
keys throughout; the `f32` type is a declaration, never a value interpretation. Nothing in the
kernel does float arithmetic.

Placement of the reinterpret relative to `monotone_key` (both are bit-level, and the wrong order is
silent): the chain is `load f32 → bitcast to i32 → monotone_key → bitcast to f32 → store`, and on
the read side `load f32 → bitcast to i32 → use as key`. `monotone_key` therefore sees exactly the
same `int32` it saw in the incumbent, and the involution at emit time (`monotone_key(key)` into
`keybuf_vmem`) is untouched.

`diff -u` against the live incumbent — **7 hunks, +36 / −10 lines**:

| hunk | lines | what |
| --- | --- | --- |
| `@@ -26,6 +26,12` | +6 | module docstring: what the `f32` type does and does not mean |
| `@@ -64,11 +70,11` | ±2 | `_topk_body` shape comments: `scores_hbm` `f32[b*n]`, `resident_vmem` `f32[slice_len]  holds int32 monotone-key BITS` |
| `@@ -168,14 +174,27` | +13 −2 | `cvt` and `load_keys`: the two bitcast sites, with the compromise named |
| `@@ -435,7 +454,10` | +4 −1 | **`:438` deleted** — `words = bitcast_convert_type(...)` gone |
| `@@ -452,7 +474,10` | +4 −1 | **`:455`** resident scratch `jnp.int32 → jnp.float32` |
| `@@ -465,8 +490,10` | +3 −1 | kernel-name tag `_f32in` |
| `@@ -483,7 +510,7` | ±1 | **`:486`** `words.reshape(-1)` → `scores.reshape(-1)` |

The `_f32in` tag makes the kernel name `sc_topk_direct_b1024_n10240_k2048_f32in`, so a profile scope
name proves *this body executed* rather than merely that the arm was offered. Nothing in the frozen
test or the microbench keys off the kernel-name string (checked by grep). Raw diff: `/tmp` is not
durable, so the file itself is the record — `diff -u` it against
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`.

## 2. The file plan, scored

`v12` was promoted from stage E **after** §9 was written, so it has no Plan §4 "Files" table. Its §5
block states the plan as a line: *"three edits — `:438` (`words = scores`), `:455`
(scratch `int32 → float32`), and in-register `bitcast_convert_type` at `:172-173` and `:178`."*

| plan row | produced | divergence |
| --- | --- | --- |
| `sparsecore_topk.py` — `:438` delete the bitcast | done, 1 line deleted + 4 comment | none |
| `sparsecore_topk.py` — `:455` scratch dtype | done, 1 line changed + 3 comment | none |
| `sparsecore_topk.py` — `:172-173`, `:178` in-register bitcasts | done, at `:184-188` and `:196-197` after the docstring grew | line numbers only |
| *(not in the plan)* `:486` call site | **added** — `words.reshape(-1)` → `scores.reshape(-1)` | the plan's "`:438` (`words = scores`)" implies it; spelling it as a deletion rather than a rebind means the call site moves too. Same edit, one line either way. |
| *(not in the plan)* `:490` kernel-name tag | **added** | `references/` fact: an env var or a mapped file only proves the arm was *offered*; the scope name proves this body ran. Costs nothing and no instrument keys off it. |
| *(not in the plan)* shape comments at `:70-77` | **added** | `code-style.md` §2 requires the trailing `# [shape] dtype` comment to be true, and two of them stopped being true. |

No new file. No shared helper. Nothing outside this folder.

## 3. `code-style.md` checklist

Scored against the **whole file**, with each row marked whether `v12`'s diff caused it. The file
opts out of the four-file decomposition, which §10 permits explicitly ("say so in the module
docstring and leave").

| item | | note |
| --- | --- | --- |
| Dimension glossary at the top | ❌ — inherited | the file has a prose docstring, not a `# Dimensions.` block. `b`, `n`, `k`, `P`, `slice_len`, `LANES`, `walk_chunks` are never named in one place. Not introduced by `v12`; my diff adds no new undocumented dimension. |
| Trailing `# [shape] dtype` on every ref parameter | ✅ | present on all 11 refs of `_topk_body`, and **`v12` updated the two it invalidated** — `scores_hbm` to `f32[b * n]` and `resident_vmem` to `f32[slice_len]` with an explicit "holds int32 monotone-key BITS". Says whole-array vs block. |
| Extents derived from the refs, not the config; no bare `x.shape[2]` | ✅ | `b, n = scores.shape` at `:449`; body extents come from the `functools.partial` bindings, not a config object. |
| jaxtyping on the public wrapper; bare refs inside | ❌ — inherited | `sparsecore_topk` at `:544` annotates `jax.Array`, not `Float[Array, "B N"]`. Refs inside *are* bare (correct half). Untouched by `v12`. |
| `# Inputs. / # Outputs. / # Scratch.` sections in the signature | ❌ — inherited | `_topk_body`'s 11 refs are one unsectioned list. The shape comments carry it instead. Untouched by `v12`. |
| Body order: indices → shapes → semaphores → init → loads → compute → store → epilogue | ✅ | `wave_body` follows it; `v12`'s edits sit inside step 5/6 and do not reorder anything. |
| Single characters only for loop indices / annotated tensors / uppercase extents | ✅ | `i`, `j`, `w` are loop indices; `s`, `sel`, `key`, `bits` are words or annotated. `v12` introduces `bits`, a word. |
| Memory space in the ref name where two spaces share a signature | ✅ | `scores_hbm` / `resident_vmem` / `priv_vmem` / `out_hbm` throughout. No `_ptr`. |
| Comments name the compromise, not the operation | ✅ | all four comment blocks `v12` adds name a compromise: why the type is `f32` (the DMA matches element types), what the bitcasts replace (a 40 MiB HBM buffer), why the store/load is a pure bit round trip, why the kernel names itself. |
| One-line module docstring | ❌ — inherited | it is a 20-line docstring. `v12` added 6 more lines to it, so **`v12` made this one worse**, deliberately: the "this f32 buffer holds int32 bits" invariant is the single most dangerous thing a future reader can get wrong here, and §8's "three times the comment density in a kernel body" is the rule I weighted higher. |
| Unsupported features raise `NotImplementedError` | ✅ | `:455` `NotImplementedError("SparseCore is not available")`; the four `ValueError`s at `:569-576` are genuine caller-argument errors. Correct split. |
| Four files `api/base/pallas_mosaic_tpu/…_kernel` | ❌ — inherited | one 590-line file. §10's "opting out is a legitimate outcome" applies, but the docstring does not *say* it opted out. Untouched by `v12`. |
| Further split by what the module may touch | n/a | single file. |
| Config is a frozen kw_only slots pydantic dataclass | n/a | there is no `Config`; the knobs are module constants. |
| **BLOCKING — tested against the XLA `_fwd`, tolerance visible at the call site** | ✅ | the frozen `test_dsv4_sparsecore_topk.py` tests against `_reference_scores`, a NumPy exact sort in the same file, and the tolerance is **written at the call site and is exactly zero**: `_mismatch` line 133-136, `# Exact: this is a selection, so a correct kernel returns the same score values bit for bit. No relative tolerance.` → `if diff != 0.0`. There is no tolerance to relax, and `v12` did not touch it. |
| Every f32 contraction in the reference pins `Precision.HIGHEST` | n/a | there is no contraction anywhere in this op — it is a selection. No MXU, no matmul, no precision knob. |
| **BLOCKING — hardware constants from `pltpu.get_tpu_info()`** | ❌ — inherited | the *launcher* reads the platform (`:452 info = pltpu.get_tpu_info()`, `:468-469 sc.num_cores` / `sc.num_subcores`), but the *body* hardcodes `num_subcores = 16` and `num_sc_cores = 2` at `:100-101`, plus `LANES = 16` and `MAX_SLICE_WORDS = 64 * 1024` at module scope. |

**The blocking `❌`, stated plainly rather than shipped with a note.** By the letter of BRIEF §8
item 3, the hardcoded `num_subcores = 16` / `num_sc_cores = 2` / `MAX_SLICE_WORDS` block this
variant. Three things the caller needs in order to decide, because I will not decide it silently:

1. It is **inherited, not introduced.** `v12`'s +36 lines add no hardware constant of any kind, and
   the same `❌` is true of the incumbent and therefore of every variant in this wave equally. It
   cannot discriminate between arms.
2. It is **contradicted inside the same file** — the launcher already reads `sc.num_cores` and
   `sc.num_subcores` from `get_tpu_info()` and passes them to the mesh, while the body recomputes
   the same two numbers as literals. That is the exact drift §13 is about, and it is a real latent
   bug, not a formatting complaint.
3. **Route:** it is a non-Pallas cleanup on the incumbent, worth one commit and no measurement
   (`num_subcores = sc.num_subcores` threaded through the `functools.partial` that already carries
   `b`, `n`, `k`). It belongs to whoever lands the winning arm, not to `v12`. See §9.

## 4. Risk table, with an outcome on every row

`v12` has no §9 plan and therefore no §7 risk table. This one is constructed from the §5 block's
three named hazards plus the two the build surfaced.

| risk | threatens | outcome |
| --- | --- | --- |
| The two in-body `tpu.bitcast`s are not free — a real op, an extra VREG, or a spill, in the inner loop | speed | **did not happen — the reading, in full, is §5 below.** There are **seven**, not two, and they replicate to **35** LLVM instructions under `unroll=4`; all of them are in `plsc.parallel_loop` bodies, i.e. the innermost loops. And **all 35 are eliminated at instruction selection**: the variant's scheduled SC bundle listing is 1364 bundles against the incumbent's 1364, with an **identical opcode histogram** and a whole-file diff of two lines that swap which scalar register receives two `sadd.s32` immediates *in the same bundle*. No spill (identical load/store opcode counts), no extra VREG (identical register-allocation outcome bar the swap), no bundle. **Price: 0, evidence class `static-compile`.** |
| The 40 MiB buffer is not actually deleted — renamed, replaced by a copy, or re-materialised as a staging op (the failure `probes/14` Q2 found on `v11`) | speed | **did not happen.** ENTRY: the incumbent's `%bitcast-convert_bitcast_fusion` (`kind=kLoop`, `used_scoped 16777216`) is replaced by `%bitcast.3 = f32[128,80,8,128]{3,2,1,0:T(8,128)} bitcast(%s.1)` with **no `backend_config` at all**. `fusion(` in ENTRY 1 → 0; **`copy(`/`copy-start(` in ENTRY 0 → 0** — no staging copy appeared. Buffer assignment: the value `<11 bitcast-convert_bitcast_fusion @0> (size=41943040)` is **absent** from the variant; the variant's 4 `copy` values are the same 4 the incumbent has. The memory-usage report closes the last hole in this row — the temp arena goes 80 → 48 MiB because the incumbent held **two** 40 MiB buffers and the variant holds one, and **`vmem` is unchanged to the byte**, so the buffer was deleted rather than relocated into another memory space. Re-derived in a fresh single-arm process. Numbers in §5. |
| `monotone_key` writes an arbitrary int32 pattern **through an `f32`-typed VMEM store** — NaN payloads and denormals may not survive the round trip | **correctness** | **did not happen — and the static reading is the load-bearing evidence, not the unit test.** ⚠ Read the qualification below the table: the frozen test is the *weaker* of the two gates here. |
| The score DMA lands different bytes once its target is `f32`-typed | **correctness** | **did not happen.** The HBM layouts are identical instruction for instruction — `{3,2,1,0:T(8,128)}` in, `{3,1,2,0:T(8,128)}` out of the data-format call, `{0:T(1024)}` flat into the kernel — with only the element-type *name* changing (`hlo-entry-diff.txt`). Inside the SC program the transfer is `sc_tpu.enqueue_transfer` f32→f32 against `memref<10240xf32, #tpu.tiled<(128),[1]>, tile_spmem>`, and the bundles that enqueue it are byte-identical to the incumbent's, immediates included. Nothing to checksum: see §7. |
| The two-stage (`P>1`) path stops lowering, or regresses | **correctness** / speed | **happened, partly — and it is priced.** All 7 frozen-test shapes lower on both arms (`aot_two_stage.txt`), so correctness is not at risk. But at `P>1` the variant **adds back** one real op the incumbent got for free: stage 1's `bitcast_convert_type(keys, f32)` at `:516` used to cancel against stage 2's input bitcast, and now has nothing to cancel with, so it materialises as `%bitcast_convert_type.2 = f32[65536] bitcast-convert(%mpmd_map.1)`. ENTRY grows by exactly +1 on every `P>1` shape; `temp_size_in_bytes` is **1572000 in both arms**, so no buffer was added. **Cost ≈ 0.17 µs per `P>1` call, `modelled`** (65536 × 4 B read+write ÷ 2.97e12 B/s). Net at `P=2` the variant is still ahead: it deletes a 640 KiB dense fusion and adds a 256 KiB one. **Fallback taken: none, and none needed — `P = 1` at both shipping shapes, so this term is exactly zero in production.** Route in §9. |

**The correctness row, discharged properly, and where the frozen test is weaker than the compile.**

Which assertion in the frozen test would fire: `_mismatch`'s exact-score branch,
`test_dsv4_sparsecore_topk.py:133-136` —
`# Exact: this is a selection … No relative tolerance.` → `if diff != 0.0: return f"max score
difference {diff!r}, expected exactly 0.0"` — surfaced by
`assert _mismatch(scores, _run(...), k, row_lengths) is None` at `:235`
(`test_exact_topk_full_rows`), `:258`, `:270`, `:286` and `:373`. Any key corruption that changes
*which* elements win changes the returned score multiset and fires it. On the `P>1` ids the emitted
keys additionally *drive* stage 2's selection, so `test_emit_pairs_is_bit_identical_to_the_incumbent`
line 345, `assert n_diff == 0`, fires as well — and its `all_negative` arm exists precisely because
`monotone_key` is the identity on non-negative floats, so it is the one arm where the involution is
load-bearing.

⚠ **But I checked the inputs, and the frozen test would probably stay green under the specific
corruption this row is about, so I am not resting the row on it.** The `f32` view of a monotone key
is a NaN only when the original score's exponent field is `0x00` (i.e. `-0.0` or a negative
denormal), and a denormal only when the original is `±inf`. Across the five `_scores` kinds:
`random` and `heavy_ties` produce neither (a NaN key needs `|x| < 1.2e-38`); `all_negative` has
magnitudes ≥ 1.0, all normal; `all_neg_inf` gives `KEY_NEG_INF = 0x807FFFFF`, a negative denormal —
but flushing it to `0x80000000` makes it *more* negative, so `key > KEY_NEG_INF` still rejects it
and the test still passes. `strict_desc` **does** put `-0.0` at column 0 of every row, key
`0xFFFFFFFF`, a quiet NaN — but it is the row maximum by a wide margin, so canonicalising it to
`0xFFC00000` or `0x7FC00000` leaves it the maximum and the selection unchanged. A green run is
therefore weak evidence for this row.

What actually discharges it is stronger than any test: **the variant's emitted instruction stream is
identical to the incumbent's**. The incumbent loads, XORs and stores into an `int32`-typed VMEM
buffer; the variant emits *the same opcodes, the same count, the same bundles* against an
`f32`-typed one. There is no float-typed load or store instruction on this subcore and no
canonicalisation opcode anywhere in either listing — the dtype has ceased to exist by the time
instructions do. Identical machine code on identical input bytes is bit-identical output by
construction, which is the strongest form the "bit-identical row gate" `design.md` says `v12` owes
can take, and it holds for *all* inputs rather than the five the test samples.

Residual, and it is a request not a claim: the static argument assumes the compile the correctness
runner gets is the compile I read. That is what the in-job md5 gate on
`408ef72e9afc935c928f2803764878cc` is for (§7, §8). I am also asking for one cheap runtime
confirmation in §8 that the frozen test cannot express, because it compares two arms *within* one
build and this needs a comparison *across* builds.

## 5. `static-compile` evidence

All of it on a host with **no TPU**, from `jax.jit(...).lower(...).compile()` against
`topologies.get_topology_desc("tpu7x:2x2x1")`, at the shipping prefill shape
`b=1024, n=10240, k=2048` (`P=1`, `--ctx 9216`). Zero cluster time, zero jobs. Scripts and raw
output in this folder: `aot_check.py` → `aot_check.txt`, `buffers_and_llo.py` → `buffers.txt`,
`sc_bundles.py` → `sc_bundles.txt`, `aot_two_stage.py` → `aot_two_stage.txt`,
`verify_fresh_procs.py` → `verify_fresh_procs.txt`; HLO in
`hlo-entry-{incumbent,variant}.txt` / `hlo-entry-diff.txt` / `p2-entry-diff.txt` /
`fresh-entry-{incumbent,variant}.txt` / `fresh-entry-diff.txt`, Mosaic in
`mosaic-{incumbent,variant}.mlir`, compiler dumps in `xdump-*/` and `scdump-*/`.
Every compile is wrapped in retry-with-backoff on `/tmp/libtpu_lockfile`, which serialises host AOT
compiles across all agents on this box; it is never removed, since that would break a sibling's
in-flight compile.

**Negative control — it fires, and it is the allocator that refuses it.** A control only controls if
the **SparseCore allocator** rejects it; an MLIR-verifier rejection (the well-known trap here is
`o[...] = jnp.zeros_like(o)` → `NotImplementedError: Swap does not support storing to HBM`) proves
nothing about whether the SC backend ran. Mine is the 8 MiB-VMEM-scratch form, DMA'd to HBM so it
cannot be dead-code-eliminated, and the **untruncated** text (`verify_fresh_procs.txt`) is:

```
INTERNAL: Failed to run MLO pass pipeline.aot_negctl_oversized_vmem.6:49:1: error:
'memref.alloca' op E3000: CompileTimeSparseCoreAllocationFailure: current allocation
offset upper bound (33554431 words) exceeds the legitimate user allocatable offset
upper bound (2097151 words) in memory space spmem when allocating 33554432 words
...
%36 = "memref.alloca"() : () -> memref<16384x128xi32, #sc_tpu.memory_space<tile_spmem>>
main.1:1:1: error: NOTE: Failed in pass AllocationAssignment
```

It names **`memory space spmem`**, the offending op is a `tile_spmem` `memref.alloca`, and the pass
that failed is **`AllocationAssignment`** — the allocator, not the verifier. (My earlier record
truncated this at 260 characters, which stopped one character before `spmem`; the check was right
and the *evidence* for it was not legible. Now it is.) So `.compile()` really does drive the
SparseCore Mosaic/MLO backend, and every "compiles ok" below means something.

**It lowers.** Both arms compile at `f32[1024,10240] → s32[1024,2048]`, and all 7 frozen-test shapes
compile on both arms (`aot_two_stage.txt`).

**Every headline number below was re-derived in a fresh single-arm process.** `jax.jit` caches on
`(function object, avals)`, so an A/B that rebinds a module attribute can compare an arm against
itself and return byte-identical HLO — a clean-looking null that is really a cache hit.
`aot_check.py` and `aot_two_stage.py` compile both arms in **one** process, so they are exposed to
this in principle. `verify_fresh_procs.py` re-runs the `P=1` numbers with **exactly one arm per
process**, **variant first** (so any residual leak would show as *base looking like variant* — the
base losing its 40 MiB buffer, which is loud — rather than the silent reverse), asserting inside
each child that exactly one `sparsecore_topk.py` module is resident and printing the source md5 it
actually loaded (`408ef72e…` / `44353b34…`). **Every number reproduced exactly**, and the arms
differ, so no cache collision occurred. `verify_fresh_procs.txt`, `fresh-entry-*.txt`.

**ENTRY, incumbent → variant, `P=1` production shape**

Counts are of the **opcode** (`fusion(`, `bitcast(`, …), matched with the trailing paren, in the
ENTRY body only:

| | incumbent | variant |
| --- | ---: | ---: |
| ENTRY instructions | **13** | **13** |
| `fusion(` | **1** | **0** |
| `bitcast(` | 1 | **2** |
| `bitcast-convert(` | **0** | **0** |
| `copy(` / `copy-start(` | **0** | **0** |
| `custom-call` | 4 | 4 |

13 → 13 is a 1-for-1 replacement, and that is the honest reading: the `kLoop` fusion becomes a
`bitcast`, an instruction that carries **no `backend_config`** — the same signature the genuinely
free flatten beside it carries, and the tell `probes/11` used. `copy` staying at 0 is the specific
check `probes/14` Q2's `v11` finding demanded, and the failure mode did **not** recur here.

⚠ **A correction to my own earlier count, because it is the exact error class this file is supposed
to catch.** I first reported "`bitcast-convert` in ENTRY 1 → 0". That was a *text* match, and the
text it matched was the fusion's **name**, `%bitcast-convert_bitcast_fusion` — the incumbent ENTRY
contains **no `bitcast-convert` opcode at all**, because the op lives inside `%fused_computation`,
one level down. Grepping the name and reporting it as an opcode count is counting text, not
occurrences. The corrected row is `fusion(` 1 → 0, which is the same finding stated in terms of
something that is actually in ENTRY, and the conclusion is unchanged. The same loose pattern
(`re.findall("bitcast-convert", entry)`) is what produced the "bcvt" column in `aot_two_stage.txt`,
so **read that column as "occurrences of the text, fusion names included", not as an opcode count** —
its qualitative content (the variant has strictly fewer) survives, its absolute values are not
opcode counts. The `P>1` regression in §4 is **not** affected: that one was read off a real
instruction line, `%bitcast_convert_type.2 = f32[65536]{0:T(1024)} bitcast-convert(%mpmd_map.1)`.

**Buffer assignment (`--xla_dump_to`, `*buffer-assignment.txt`) — the 40 MiB buffer is gone**

| | incumbent | variant | Δ |
| --- | ---: | ---: | ---: |
| allocations | 8 | 8 | 0 |
| **sum of allocation sizes** | 167,841,800 | **134,287,368** | **−33,554,432** (−32 MiB) |
| **peak `preallocated-temp`** | 83,951,616 | **50,397,184** | **−33,554,432** (−32 MiB) |
| `memory_analysis().temp_size_in_bytes` | 84,017,024 | 50,462,592 | −33,554,432 |
| `argument_size` / `output_size` | 41,947,136 / 8,388,608 | unchanged | 0 |
| values matching `bitcast-convert` | **1** (`size=41943040`) | **0** | −1 |
| values matching `copy` | 4 | 4 | 0 |

The distinction worth keeping straight: the *value* deleted is 41,943,040 B (40 MiB), but the
*arena peak* falls by 32 MiB because the allocator repacks —
`83,951,616 = 2 × 41,943,040 + 65,536` becomes `50,397,184 = 41,943,040 + 65,536 + 8,388,608`. The
variant's one 40 MiB `copy` value (`<0 copy.4 @0> f32[128,80,8,128]{3,1,2,0}`) is the pre-existing
`sparse-core-data-format` output moving into the freed slot, not a new op — its count is 4 in both
arms. Corroborated from inside the SC program: `sc_tpu.allocate_at_offset` goes
`memref<83951616xi8, hbm>` → `memref<50397184xi8, hbm>`.

`generated_code_size_in_bytes` goes 153,088 → 115,712, but that is dominated by the deleted
TensorCore fusion's codegen and does **not** isolate the SC kernel — do not read it as a kernel-size
result.

**Memory-usage report — the saving is HBM, and VMEM does not pay for it.** `--xla_dump_to` writes
two files beyond the allocation table, and `*-memory-usage-report.txt` breaks the totals down **by
memory space**, which the rollups above cannot do. It settles a question the buffer-assignment table
leaves open: whether the 32 MiB came out of HBM *into* VMEM.

| memory space | incumbent | variant | Δ |
| --- | ---: | ---: | ---: |
| `default` (HBM) | 134,287,360 (128.07 MiB) | **100,732,928 (96.07 MiB)** | **−32.00 MiB** |
| **`vmem`** | 33,554,432 (32.00 MiB) | **33,554,432 (32.00 MiB)** | **0** |
| `smem` | 4 B | 4 B | 0 |
| `color_7` / `color_8` | 0 B / 4 B | 0 B / 4 B | 0 |

**VMEM is unchanged to the byte.** The resident scratch changed dtype `i32 → f32` at the same width,
and this is the artifact that proves the width claim rather than asserting it. Nothing was displaced
into VMEM, and no VMEM pressure was added that could cost occupancy later.

It also gives the *mechanism* for the "40 MiB deleted, 32 MiB peak drop" arithmetic, which I
previously justified only by arithmetic. The temp arena, entry by entry:

| | incumbent | variant |
| --- | --- | --- |
| slot @42008576 | 40.00 MiB, 1 value, `s32[128,80,8,128]` | 8.00 MiB, 1 value, `s32[2097152]` |
| slot @65536 | 40.00 MiB, **2 values**, `s32[128,80,8,128]` + `s32[2097152]` | 40.00 MiB, 1 value, `f32[128,80,8,128]` |
| arena total | **80.00 MiB** | **48.00 MiB** |

So the incumbent held **two** 40 MiB buffers, with the 8 MiB output *co-resident* in the second
slot. The variant deletes one outright; the 8 MiB output can no longer share a slot with it and
takes its own. 40 − 8 = **32 MiB**, which is the drop, and the deleted buffer really is a whole
40 MiB. The 40 MiB `f32[1024,10240]` in the second arena is the **argument** and is present in both.

**SparseCore metadata — the bundle counts, from a second and independent artifact.**
`*sparse_core_specific_metadata.txt` records each SC computation's overlay bundle ranges. It was
produced by a different part of the compiler from the `--xla_sc_dump_bundles_to` listing, so it is
an independent check on the central "the kernel is unchanged" claim rather than a restatement:

| computation | sequencer | incumbent | variant |
| --- | --- | ---: | ---: |
| `called_computation.1` (**the topk kernel**) | TEC (vector) | 0–**1157**, 1157–1168 | 0–**1157**, 1157–1168 |
| `called_computation.1` | SCS (scalar) | 136–196 (**60**) | 136–196 (**60**) |
| `called_computation` (data-format) | TEC | 0–**84** | 0–**84** |
| `called_computation` (data-format) | SCS | 0–**58** | 0–**60** |

`overlay_slot_size` is 8144 for the kernel in both arms and `single_imem_overlay` is false in both,
so instruction-memory pressure is unchanged too. The kernel's bundle counts are identical on both
sequencers, and the only delta in the whole module is the **+2 SCS bundles in the data-format
program** — the same +2 the bundle listing showed, and (see below) a harness artifact absent from
production.

**Pricing the in-body bitcasts — the open term, closed at zero**

`--xla_sc_dump_bundles_to`, `--xla_sc_dump_mlir_to` and `--xla_sc_dump_llvm_ir_to` are
`LIBTPU_INIT_ARGS` (XLA rejects them by name in `XLA_FLAGS`), read once at libtpu load, so they must
be set before `import jax`. They work offline. Following one bitcast down the stack:

| stage | incumbent | variant |
| --- | ---: | ---: |
| Mosaic MLIR (`mosaic-*.mlir`) | 0 `tpu.bitcast` | **7** `tpu.bitcast`, all `vector<16xf32> ↔ vector<16xi32>` |
| last Mosaic stage (`*post-emit-tec.txt`) | 915 lines, 0 | 922 lines (+7), 7 `arith.bitcast` |
| SC LLVM IR (`*_unoptimized.ll`) | 0 `bitcast` | **35** `bitcast <16 x float> ↔ <16 x i32>` |
| **scheduled VLIW bundles** | **1364** | **1364** |

The seven sites are 2 in `cvt` (load `f32→i32`, store `i32→f32`) and 5 at `load_keys` call sites
(`hist0_body`, the three statically-unrolled `REFINEMENT_LEVELS` `ref_body`s, and `emit_body`). All
seven are inside `plsc.parallel_loop` bodies — the innermost loops, executing once per 16-element
chunk per pass — and `unroll=4` replicates them to 35 LLVM instructions. `probes/17`'s "two
`tpu.bitcast`" was measured on a toy kernel and undercounts the real one by 3.5×.

Per-section bundle counts are equal everywhere, including the vector body:

| section | incumbent | variant |
| --- | ---: | ---: |
| `__scs_entry_jumppad` | 8 | 8 |
| `__scs_overlays_trampoline_lowered` | 128 | 128 |
| `called_computation.1_lowered` (scs) | 60 | 60 |
| **`execute0_lowered` (tec / vector body)** | **1157** | **1157** |
| `_tile_overlayer_lowered` | 11 | 11 |
| **total** | **1364** | **1364** |

Opcode histogram: **identical**, every opcode, every count — which also rules out spill code, since
a spill is a load and a store. Whole-file diff after normalising the `_f32in` name tag: **two
lines**, swapping which of `s4`/`s5` receives each of two `sadd.s32` immediates within one bundle.
No float, convert or bitcast opcode appears anywhere in either listing.

**So: 7 source bitcasts → 35 LLVM instructions → 0 machine instructions.** On the SC vector subcore
an `f32` VREG and an `i32` VREG are the same 16-lane register, and an equal-width bitcast is a pure
type relabel that the backend folds at instruction selection. **0 bundles, 0 ns, `static-compile`.**

**One residual the bundle dump surfaced, in a different program.** The `sparse-core-data-format`
call — a *separate* SC program present in both arms — grows its scalar sequencer prologue by 2
bundles (`called_computation_lowered` 58 → 60: one extra `rddreg [dreg:$0x2]`, one extra `sld`, one
extra `shalt`, one fewer `sadd.s32`). Its vector body is byte-identical at 84 bundles, so this is
one-shot launch setup, ~1–2 ns, not per-element. ⚠ And it is a **harness artifact of my AOT
wrapper, not production**: my standalone `jit` takes `f32[1024,10240]{1,0}` at the entry, so XLA
inserts a data-format retile that stage E's U1 re-derivation found does **not** exist in the real
production module (0 `sparse-core-data-format-call`, because the producer already delivers
`f32[1024,80,128]{2,1,0}`). Present in both arms either way, so it cannot bias the A/B.

**`v11` and `v12` still compose**, re-confirmed here rather than assumed: the variant's ENTRY still
carries `ROOT %reshape.4 … used_scoped_memory_configs 4194304`, which is exactly `v11`'s target.

## 6. Prediction, restated as a number to be scored

The build did not diverge from the plan, and the one term that could have reduced the prize prices
at zero, so the design's number stands unreduced.

| denominator | baseline | predicted | `Δ time %` |
| --- | ---: | ---: | ---: |
| harness (`instruments/sctopk_2stage_bench.py`, `--ctx 9216`) | 189.98 µs | 163.73 µs | **−13.82%** |
| production (× 1.219) | 155.82 µs | 129.57 µs | **−16.8%** |

**Mechanism:** deletion of one 26.251 µs dense elementwise pass over the score array — 40 MiB read
plus 40 MiB written, in a `kLoop` fusion with its own disjoint temp buffer, replaced by a
layout-preserving `bitcast` that XLA emits with no `backend_config`. It is a pure edge move: no pass
inside the kernel is added, removed or reordered, which is now not an argument but a reading — the
kernel's 1364 bundles are the same 1364 bundles.

Both figures are **`modelled`**, as BRIEF §7 requires me to say: the 26.251 µs is `measured` on the
harness's op, but 189.98 → 163.73 is arithmetic, no arm of `v12` has run, and every design
percentage in this file rests on a `modelled` denominator.

## 7. How the measurer runs me

- **Map** `variants/v12-f32-input/sparsecore_topk.py` → `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`.
  One file; nothing else in this folder ships. Every other `.py` here is static analysis and must
  **not** be mapped.
- **Gate in-job**: `md5sum src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py` must print
  `408ef72e9afc935c928f2803764878cc`, and the control arm must print `44353b347c96e0eedbd742c710d4540a`.
  A docker tag is mutable and a recipe read-back does not detect a re-push.
- **Arm names**: control `v1-incumbent`, treatment `v12-f32in`. Independent activation check that
  does not depend on the mapping having worked: the kernel scope name in a profile is
  `sc_topk_direct_b{b}_n{n}_k{k}_f32in` on the treatment and `sc_topk_direct_b{b}_n{n}_k{k}` on the
  control. Present in the job log too, via the HLO op name.
- **Invocation**, frozen instrument, `--ctx 9216` every time, read `us_median` not `us`:
  ```
  python sctopk_2stage_bench.py --decomp --ctx 9216 --batches 16 64 1024 --rounds 9
  python sctopk_2stage_bench.py --passes --ctx 9216 --batches 1024      --rounds 9
  ```
  `b ∈ {64, 1024}` are the shipping shapes (decode 2 waves, prefill 32 waves); `b=16` only for the
  `P>1` cross-check.
- ⚠ **`v12`'s prize is invisible to `--decomp` and `--passes`.** Both time the *kernel*, and `v12`
  changes nothing inside the kernel — its 1364 bundles are unchanged. The deleted op is a
  **TensorCore fusion outside the kernel call**, so the arm that scores `v12` must time the
  **whole `sparsecore_topk(...)` call**, or the module, not `stage1`/`full`. If the harness's `full`
  arm times only the `pl.kernel` invocation, `v12` will read as a flat null and that null will be an
  instrument artifact, not a result. This is the single thing most likely to throw the number away.
  Per BRIEF §2 the arm goes in a copy under `variants/`, never in `instruments/`; I have not written
  it because I do not own a timing job — flagging the requirement is the instruction I was given.
- **Checksum arm: not needed.** BRIEF/instruments warn that nothing in the frozen bench catches a
  copy that lands the wrong bytes. `v12` does not change what the DMA lands: same byte count, same
  HBM layout instruction-for-instruction, same enqueue bundles with the same immediates (§5). It
  adds no chunking and no buffering.
- **Rival arms for a compile-time constant of mine: none.** `v12` has no tunable. And a rival
  *implementation* that existed on paper is now retired: `probes/17` named a 2-D-reshaped scratch
  with `ref.bitcast(jnp.int32)` as the fallback "if the two bitcasts ever price". They price at
  **zero**, so that fallback would buy nothing and cost a structural change. Do not build it.
- **ρ:** every probe ran at ρ=1 and nothing has run at production ρ=0.40005. See §10 — the ρ
  reweighting is expected to move `v12`'s scored percentage *more* negative, not less.

## 8. How the correctness runner runs me

```bash
cd <checkout>   # with variants/v12-f32-input/sparsecore_topk.py mapped over
                # src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py
python -m pytest tests/kernels/test_dsv4_sparsecore_topk.py -v --timeout=3600
```

Unmodified frozen test at `b3241859e471fc87fa6e7a50e4ffec4b`, mapped into `tests/kernels/` so it
imports the installed package. **68 collected, baseline 68/68 PASSED** (`j-62275e52-1498-4e01-918e`);
`v12` must match, with no `-k` narrowing beyond a fan-out whose shards union to exactly the 68
`--collect-only -q` ids. Same in-job md5 gate on the subject as §7.

Ids that exercise `v12` hardest, in order:

1. `test_exact_topk_full_rows[strict_desc-p1-b1024-n10240]` — the shipping prefill shape, 32 waves,
   and the only kind that puts a NaN-payload monotone key (`-0.0` → `0xFFFFFFFF`) into the
   `f32`-typed resident buffer.
2. `test_exact_topk_full_rows[{strict_desc,random,heavy_ties}-p1-b64-n10240]` — the shipping decode
   shape, 2 waves.
3. `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` and
   `test_exact_topk_ragged_rows_straddle_partition_boundaries[p1-b1024-n10240]` — the negative
   *denormal* key (`KEY_NEG_INF = 0x807FFFFF`) round-tripping through the `f32` store, and the
   ragged `i_hi` bound that decides how much of the buffer `cvt` converted.
4. `test_emit_pairs_is_bit_identical_to_the_incumbent[all_negative-p4-b1024-n133120]` and the rest of
   the `all_negative` arm — the only ids where the involution is load-bearing (measured in the
   test's own docstring: mutating `monotone_key` to the identity left the file green otherwise), and
   the ids that exercise the `P>1` regression priced in §4.

**One thing I am asking for that the frozen test cannot express, and why.** The test compares two
arms *within* one build; `v12`'s claim is bit-identity *across* builds. Ten lines, out of band, in
the runner's own scratch — not an edit to any instrument:

```
for kind in strict_desc random heavy_ties all_negative:
    scores = _scores(kind, 1024, 10240, rng);  row_lengths = full(1024, 10240)
    a = incumbent_build.sparsecore_topk(scores, 2048, row_lengths)
    b = v12_build.sparsecore_topk(scores, 2048, row_lengths)
    assert (a != b).sum() == 0
```

`_mismatch` compares score *multisets*, so it cannot see a tie-break change; this can. It is the
bit-identical row gate `design.md` §5 says `v12` owes, it costs one extra job-second, and it is the
runtime counterpart of the bundle-identity argument in §4. If it is not run, say so — do not let
68/68 stand in for it.

## 9. Found, not my candidate — each with a route

1. **`--xla_sc_dump_bundles_to` works offline, from an AOT compile, on a host with no TPU.** This is
   new capability, not a `v12` fact: the full SparseCore backend dump —
   `--xla_sc_dump_bundles_to` (scheduled VLIW bundles, scs *and* tec sections),
   `--xla_sc_dump_llvm_ir_to`, `--xla_sc_dump_mlir_to`, plus `--xla_jf_dump_to` /
   `--xla_mosaic_dump_to` and `XLA_FLAGS=--xla_dump_to` for buffer assignment — all land from
   `jax.jit(f).lower(...).compile()` with no accelerator. 19 dump files per arm in seconds. The
   catch that cost me a run: they are **`LIBTPU_INIT_ARGS`**, read once at libtpu load, so they must
   be set before `import jax`; XLA hard-rejects them by name in `XLA_FLAGS`. And the `jf`/LLO bundle
   files cover **TensorCore** programs only — the SC kernel appears only under the `sc_` flags.
   **This is what turned "the bitcasts are `unmodelled`" into "the bitcasts are 0", so it is worth a
   bundle-count A/B on any variant in this wave that claims a bundle-level effect.**
   **Route:** `references/` — it extends `probes/17`'s `static-compile` from "does it lower" to
   "how many bundles", and `xprof-llo`'s SKILL.md currently documents only the TC path. Owner: the
   main thread, to fold into the skill reference root.
   ⭑ Two additions worth the same route, because `compiled.as_text()` has neither and both answered
   a question for me that the rollups could not. `XLA_FLAGS=--xla_dump_to` also writes
   **`*-memory-usage-report.txt`** — totals **per memory space**, which is the only artifact that
   distinguishes "saved 32 MiB of HBM" from "moved 32 MiB into VMEM", and which broke my temp arena
   down slot by slot — and **`*sparse_core_specific_metadata.txt`**, which carries per-computation
   SCS/TEC **overlay bundle ranges** and `overlay_slot_size`. The second is an independent
   corroboration of `--xla_sc_dump_bundles_to` from a different part of the compiler, and it is
   about 90 lines rather than 19 files, so it is the cheaper instrument to reach for first when the
   question is only "did the kernel's bundle count move?".
2. **`v12` costs ≈ 0.17 µs on the `P>1` path (`modelled`), and there is a clean fix I did not
   build.** Stage 1 emits int32 keys, `:516` bitcasts them to `f32` for the public contract, and
   stage 2 used to bitcast straight back — a pair XLA cancelled for free. `v12` breaks the
   cancellation. Declaring `keys_hbm` as `f32` (the symmetric output-edge move) deletes both
   bitcasts and is the same `exact` reinterpret argument. **Route:** it is a *different edge* and
   BRIEF §5 says a move that changes what the variant is becomes its own node — offer it as
   **`v12b`, output-edge reinterpret**, prize 0 in production (`P=1` at both shipping shapes) and
   ≈ −0.17 µs on `P>1`. Low priority precisely because its production prize is zero; the reason to
   have it on the sheet is that it removes the one row in §4 marked `happened`.
3. **`num_subcores = 16` / `num_sc_cores = 2` hardcoded at `:100-101`, in a file whose launcher
   already reads both from `pltpu.get_tpu_info()` at `:452, :468-469`.** A blocking `code-style.md`
   §13 `❌`, inherited by every variant, and a real latent drift rather than a formatting point —
   the two sources will disagree the first time either changes. `MAX_SLICE_WORDS = 64 * 1024` is the
   same class against `info.sparse_core` tile memory. **Route:** one non-Pallas commit on the
   incumbent, no measurement needed (thread `sc.num_subcores` through the `functools.partial` that
   already carries `b`, `n`, `k`), landed by whoever lands the winning arm. It must not be folded
   into a variant's diff, because it would then contaminate that variant's A/B.
4. **`probes/17`'s named fallback is retired, and that is a result.** "Ref-level `.bitcast` would
   avoid the two per-load bitcasts … a reshaped scratch is a fallback **if the two bitcasts ever
   price**." They price at zero on emitted code, so the fallback buys nothing. **Route:** a one-line
   strike-through in `design.md` §5's `v12` block, main thread — it saves the next agent from
   building a structural change for a zero prize.
5. **`probes/17` undercounted the bitcasts 3.5×** — it reported 2 `tpu.bitcast` from a toy kernel;
   the real kernel emits 7, replicating to 35 LLVM instructions under `unroll=4`. Harmless here
   because the true count still prices at zero, but the *method* generalises badly: a toy-kernel op
   count is not the production op count. **Route:** same `design.md` §5 correction as (4).

## 10. Open terms

Stated as BRIEF §8 item 10 requires: a term tagged `unmodelled`, predicted at zero, or credited as a
saving **cannot fail**, and is recorded as `unmodelled` rather than as a pass.

| term | status | why it is where it is |
| --- | --- | --- |
| price of the in-body bitcasts | ✅ **closed at 0, `static-compile`** — *not* `unmodelled` | This was the node's one open term and it is discharged by a measurement of emitted code, not by an argument that it could only shrink the prize: 1364 bundles vs 1364, identical opcode histogram, two-line register-allocation diff. It is a zero I read, not a zero I assumed. |
| bit-preservation of the `f32`-typed VMEM store/load | **`static-compile`, with a named runtime confirmation still owed** | Discharged by instruction-stream identity (§4), which covers all inputs. The frozen test is the *weaker* gate and I have said why. The cross-build bit-identity check in §8 is the confirmation; until it runs, the runtime leg is **`assumed`**. |
| the **−13.82% / −16.8%** prize itself | **`modelled`** | 26.251 µs is `measured` on the harness's op (`probes/11`), and `probes/14` Q1 closed the production-path question on buffer assignment. But no arm of `v12` has ever executed, and the arithmetic from 189.98 µs is a ratio of predictions. |
| ρ reweighting | ⛔ **`unmodelled`, and it moves the number** | Every probe ran at ρ=1; production is ρ=0.40005 prefill. The deleted `bitcast-convert` is **dense and ρ-independent** — it rewrites all 40 MiB regardless of `row_lengths` — while the kernel's `cvt`/`hist`/refinement/`emit` walks are all bounded by `i_hi = min(slice_len, eff_len)` and *do* scale with ρ. So at production ρ the numerator holds and the denominator shrinks, and `v12`'s share should go **more** negative than −16.8%. I am not restating the prediction on that basis: the direction is arithmetic but the magnitude depends on what fraction of the 155.82 µs is ρ-scaled, which nothing has measured. The measurer buys this arm in the first job. |
| the `P>1` stage-2 `bitcast-convert` | **`modelled` at ≈ 0.17 µs, exactly 0 in production** | 65536 × 4 B read+write ÷ 2.97 TB/s. `temp_size_in_bytes` is unchanged at 1,572,000 in both arms, so it added no buffer — but 0.17 µs is a division, not a measurement, and BRIEF §4's rule says a region-local number is not a result. `P=1` at both shipping shapes makes it moot for the prize; it is not moot for the 54 `P>1` correctness ids, which is why it is here and not dropped. |
| the measurer's arm actually spanning the deleted op | ⛔ **`unmodelled`, and it is a way to lose the whole number** | `v12` changes nothing inside the kernel. If the timing arm brackets the `pl.kernel` call rather than the `sparsecore_topk(...)` call, it will measure a flat null and the null will be an artifact. Flagged in §7; I cannot discharge it because I do not own a timing job. |
| composition with `v3` / `v10` / `v2` / `v4` | **`assumed`** | `v12` is an edge move and its 1364 bundles are the incumbent's, so it should be additive with any interior change — but the only composition actually re-checked in this build is with `v11` (`ROOT %reshape.4` with `used_scoped 4194304` survives in the variant ENTRY). The rest is inference from where the edit sits. |
| the compile I read being the compile that runs | **`assumed`, mitigated** | A docker tag is mutable. The in-job md5 gate on `408ef72e9afc935c928f2803764878cc` (§7, §8) is the mitigation; without it this whole document describes a file that may not be the one on the pod. |
| `jax.jit` cache collision faking a null | ✅ **closed** | Every `P=1` headline number re-derived with one arm per process, variant first, one resident module asserted, source md5 printed from inside the child. All reproduced; the arms differ. `verify_fresh_procs.py`. |
| VMEM paying for the HBM saving | ✅ **closed** | `vmem` space is 33,554,432 B in **both** arms, unchanged to the byte (memory-usage report). |
| the `aot_two_stage.txt` "bcvt" column | ⚠ **known-loose, and it does not carry weight** | It is a text match that counts fusion *names* too, so its absolute values are not opcode counts. It is used in this document only for "does it lower" (unaffected) and for a strictly-fewer comparison. The `P>1` cost in §4 rests on a real instruction line, not on this column. Re-running it with an opcode regex costs one compile round and nobody's conclusion turns on it. |
