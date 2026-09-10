# `final/` — the composed shipping candidate

One file, one diff, against the incumbent
`src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`
(md5 `44353b347c96e0eedbd742c710d4540a`, 623 lines).

| | |
| --- | --- |
| candidate | `final/sparsecore_topk.py`, md5 `109a12a572116c00c795ee170256db33`, 36 599 B |
| diff | `final/variant.diff`, md5 `fca4c3ac608d611ce84ca6e2484fbadb`, 368 `+` / 88 `−` |
| scope name | `sc_topk_direct_b{b}_n{n}_k{k}[_keys]_c{chunk_words}_final` |
| production scope | `sc_topk_direct_b1024_n10240_k2048_c2048_final` |

**Sign convention.** Every percentage below is `(new − old)/old` of **time**;
negative is faster. Every number is tagged `measured`, `static-compile`,
`modelled` or `unmodelled`.

**The composed speed-up is not in this file.** Composition is measured, not
summed, and the three arms' `µs` are deliberately *not* added anywhere here.
The composed number comes from the measurer's probe `24-final`, against this
md5, in one job. See `../final_report.md`.

---

## 1. What was composed, and what was not

Three arms, all `measured` at prefill `b=1024`, `ρ=0.40005`, production
arrangement, against an incumbent of 275.213 µs whole-call / 240.659 µs
inside-kernel (floors 0.040 / 0.433 µs).

| arm | source | md5 | whole Δµs | inside Δµs | in? |
| --- | --- | --- | ---: | ---: | --- |
| `v2` R=1 shared histogram | `variants/v02-r1-shared-hist/sparsecore_topk.py` | `3c041d95…` | −37.05 | −37.40 | ✅ all of it |
| `v4-c2048` ragged + pipelined | `variants/v04-pipelined-dma/sparsecore_topk_c2048.py` | `88a67892…` | −11.59 | −11.19 | ✅ |
| `v12` f32 input | `variants/v12-f32-input/sparsecore_topk.py` | `408ef72e…` | −25.15 | −0.08 | ✅ entry glue only |

Excluded, with the reason each was excluded: `v11` retiled output emit
(−9.45 whole / −0.08 inside) — buys the *same* ~25 µs entry copy as `v12` and
buys less of it, so it is superseded, not additive; `V2_BIGSCRATCH`
(−36.85 / −36.91) — inside the floor against `v2` and the sign inverts between
estimators; `v10-a2` (+121.60) — arithmetic regression; `v3` fused emit — fails
correctness 46/68.

`c2048` composed cleanly, so the `v4-ragged` fallback
(`69c8834e…`, −7.09 / −6.17) was **not** needed. Nothing about `v4`'s
contribution was dropped.

---

## 2. Three-way merge: every line two arms both touch

Method: `diff -u incumbent <arm>` for each arm, then intersect the sets of
incumbent line numbers each diff *removes*, and the insertion points each diff
*adds at*. Four regions collide.

### 2a. Line-level intersections found mechanically

| region | incumbent line(s) | arms | resolved how |
| --- | --- | --- | --- |
| A | `185` `zero_priv(NUM_BUCKETS)` | `v2` ∩ `v4` | see below |
| B | `172–173` `bits = resident_vmem[…]` / write-back | `v4` ∩ `v12` | see below |
| C | `468–469` scope-name comment + `suffix = …` | `v2` ∩ `v12` (`v4` rewrites `470`) | see below |
| D | insert point `470` (`out = pl.kernel(`), and `29` (module docstring) | all three | additive, no conflict |

Per-arm removal sets, for the record:
`v2` removes `{72, 132–135, 185, 191, 224, 238, 456, 468, 469}`;
`v4-c2048` removes `{163, 167–176, 183–188, 196}`;
`v12` removes `{67, 71, 172, 173, 178, 438, 455, 468, 469, 486}`.

### 2b. The intersection **undercounts**, and this is the important one

`diff` aligned incumbent line `191`

```
addr = lane_iota * NUM_BUCKETS + bucket
```

as *context* in the `v4` diff, because `v4` relocates the level-0 histogram
body into `l0_block` rather than deleting it. `v2` **edits** that exact line
(`addr = bucket`). So the two arms collide on a line the mechanical
intersection reports as clean. **Region A′** is therefore a real fifth
collision and is why the merge was done by reading both diffs rather than by
running a merge tool. A three-way `git merge-file` would have taken `v4`'s
relocated copy verbatim and silently reverted `v2`'s R=1 change on the
level-0 histogram — the single largest arm in the composition — while leaving
`v2`'s R=1 change on the *refinement* levels in place, which is not merely
slower but **out of bounds**: `priv_vmem` is `NUM_BUCKETS + LANES` words under
`v2`, and `lane_iota * NUM_BUCKETS + bucket` addresses up to
`LANES * NUM_BUCKETS`.

### 2c. Each resolution, and why it is the incumbent's semantics

**A — `zero_priv` bound and placement.** `v2` shrinks the bound
(`NUM_BUCKETS` → `NUM_BUCKETS // LANES`) because R=1 leaves one histogram
where there were sixteen. `v4` *moves* the call out of the
`@pl.when(i_hi > 0)` guard, because `emit_pipeline` allocates with
`pl.run_scoped` and an allocation may not sit under a `pl.when`; the guard
becomes a value, `jnp.where(i_hi > 0, NUM_BUCKETS, 0)`. The two edits are on
different axes, so both apply:

```python
hist_chunks = NUM_BUCKETS // lanes            # 16
zero_priv(jnp.where(i_hi > 0, hist_chunks, 0))
```

`final/sparsecore_topk.py:336`. Clearing 16 lane-chunks × 16 lanes = 256 words
is exactly the incumbent's 256-bucket clear; an empty row still clears nothing.
Semantics unchanged.

**A′ — the scatter address inside the relocated histogram.** `v2`'s
`addr = bucket` is applied inside `v4`'s `l0_block`
(`final/sparsecore_topk.py:365-371`). `plsc.addupdate_scatter` resolves
intra-vector address conflicts in hardware, so sixteen lanes may target one
shared histogram; the `lane_iota * NUM_BUCKETS` stride existed only to keep
them apart. The counts summed are identical, so the histogram — and therefore
every bucket boundary the selection depends on — is the incumbent's.
Corroborated statically: `vst.idx.add.s32.msk` is **40 in both arms**
(`static-compile`) — `v2` changes the scatter *address*, not the number of
scatters.

**B — the monotone-key conversion.** `v4` deletes the standalone `cvt` pass
outright: the conversion is fused into the pipeline body, which sees each
chunk once. `v12` rewrites the same two lines to bitcast an f32-typed buffer.
Resolution: `v4`'s deletion wins for the *location*, `v12`'s typing wins for
the *dtype*, and both land inside `l0_block`
(`final/sparsecore_topk.py:349-358`):

```python
key = monotone_key(
    jax.lax.bitcast_convert_type(chunk_vmem[pl.ds(off, lanes)], jnp.int32))
resident_vmem[pl.ds(base + off, lanes)] = (
    jax.lax.bitcast_convert_type(key, jnp.float32))
```

`resident_vmem` is `f32[slice_len]` holding int32 key bits. Nothing in the
kernel does float arithmetic on it; the only reason it is f32-typed is that
`tpu.enqueue_dma` matches element types rather than reinterpreting bytes.
`load_keys` bitcasts back on the read side
(`final/sparsecore_topk.py:298-299`). The round trip is bit-exact, so the keys
the refinement levels see are the incumbent's.

**C — the scope name.** Three arms, three tags, and they disagree on
placement: `v2` **prepends** `_v2r1` *before* `_keys`, `v4` and `v12`
**append**. Resolution is a single tag that carries the composition, not three
stacked tags:

```python
suffix = ("_keys" if emit_keys else "") + f"_c{chunk_words}_final"
```

`final/sparsecore_topk.py:716`. The chunk is kept in the name because it is the
one composed parameter that is resolved at trace time, so a profile can tell
`c2048` from a fallback without reading the source. Because `v2` prepends and
the others append, an `endswith` activation check is not safe here — the
correctness harness uses a **substring** test, and `arm_check_final.py`
additionally requires the `_c\d+_final` tail on **every** name a shape lowers,
not `any`.

**D — additive insertions.** All three arms add lines immediately before
`out = pl.kernel(` (line 470) and inside the module docstring (line 29). These
are new statements and new prose; nothing is overwritten.

### 2d. Are `v2` and `v4` disjoint? — no, and here is the exact extent

They are **not** disjoint. They collide at A (`zero_priv`) and A′
(`addr`), both inside the level-0 histogram, which is the region `v2` exists to
change and the region `v4` relocates. Everywhere else they are on different
axes: `v2` touches `fold_priv`, the refinement-level `zero_priv`, and the
`priv_vmem` scratch shape; `v4` touches the score read, the pipeline grid, and
the launcher's chunk selection. The composition is therefore **not** a
guarantee that their measured `µs` add, and no such sum is claimed anywhere.

---

## 3. Hardcoded hardware constants (`§4` cleanup)

Bit-identical refactor, not an optimization. Verified below.

| incumbent | was | now | source |
| --- | --- | --- | --- |
| `:41` `LANES = 16` | module constant, used everywhere | **kept as a constant**, and every *use* inside `_topk_body` reads a `lanes` kwarg threaded from `sc.num_lanes` | `pltpu.get_tpu_info().sparse_core.num_lanes` |
| `:45` `MAX_SLICE_WORDS = 64*1024` | literal | `_max_slice_words(sc) = sc.vmem_capacity_bytes // 2 // 4` | `sc.vmem_capacity_bytes` |
| `:94` `num_subcores = 16` | literal inside the kernel body | `num_subcores` kwarg | `sc.num_subcores` |
| `:95` `num_sc_cores = 2` | literal inside the kernel body | `num_sc_cores` kwarg | `sc.num_cores` |
| `_pick_partition`'s implicit `32` | literal `32` in the body | `total_subcores` kwarg | `sc.num_cores * sc.num_subcores` |

**`LANES` cannot be derived and stay a constant, and the module binding stays.**
Two reasons no accessor removes:

1. it is a static vector *shape* — `jnp.arange(lanes)`, `pl.ds(…, lanes)`,
   and `_cdiv_dyn`'s shift amount — so it must be a Python `int` at trace time;
2. the frozen test imports it at module scope
   (`instruments/test_dsv4_sparsecore_topk.py:54`), before any TPU is consulted.

So the constant stays *and* `_sc_topk_direct` refuses to lower when the
platform disagrees:

```python
if lanes != LANES:
    raise NotImplementedError(
        f"this kernel walks {LANES} lanes at a time; this SparseCore has {lanes}")
```

The constant and the hardware can no longer silently diverge. This is the
`§4` "if a constant has no `get_tpu_info()` source, leave it hardcoded and say
which" case — except that here the source exists and is used as a *check*
rather than as the value.

`MAX_SLICE_WORDS` stays as a module constant too, because `_pick_partition`
is called with three positional arguments by the frozen suite
(`:226/:281/:321`) and by `probes/22-correctness/xgate.py:105,107`, which have
no `sc`. It is now `_pick_partition`'s **default**, and `sparsecore_topk`
passes the derived value. On tpu7x the two agree exactly:
512 KiB / 2 / 4 B = 65 536 words = `64 * 1024`.

### Bit-identity of the cleanup — `static-compile`, four controls

Run by `final/aot_final.py` (host AOT, no TPU; one arm per process, because
`jax.jit` caches on `(function object, avals)`).

| control | what it does | result |
| --- | --- | --- |
| **H1** | `final` vs `lit` — `lit` is `final` with every `sc.*` read textually substituted back to its tpu7x literal | normalized **jaxpr byte-identical**, `f3c1d15baed0f1732b7bea93d6176e5e`. HLO differs on exactly 2 lines, both inside the base64 `custom_call_config.body` that embeds the source path |
| **PC1** | patch `sc.num_subcores` → 8 | jaxpr moves in exactly 26 lines: `mul k 32`→`mul k 16`, `mul bbt 16`→`mul bbt 8`, `length=32`→`length=64`. The threading is **live**, not shadowed by a literal |
| **PC2** | `_max_slice_words(256 KiB)` = 32768 | moves `P` at `n=65536` (1→2) and `n=133120` (4→8); **no move at production `n=10240`** |
| **PC3** | patch `sc.num_lanes` → 8 | raises `NotImplementedError` as designed |

H1 alone would be a positive result with no way to fail; PC1 is what proves the
constants are actually being *read*. Without PC1, a refactor that computed
`sc.num_subcores` and then ignored it would pass H1 perfectly.

Note the full-HLO md5 is **not** stable across runs (`final` produced
`09ccb49f` then `6b321344`), so it cannot serve as the identity test. The
normalized jaxpr is.

### Separate line item, deliberately not folded in

`v3` verified that `MAX_SLICE_WORDS` 65536 → 40960 is **P-neutral** — 0 of 216
shapes change partition factor. That is a `static-compile` fact about a
*different* constant value; this cleanup ships 65536 and does not touch it.

---

## 4. VMEM and spills — `static-compile`

⚠ **XLA's memory-usage report is blind to Pallas/Mosaic SparseCore VMEM.** It
reports `vmem_bytes: "0"` and is byte-identical between arms that differ. It
was not used. Two independent sources were:

1. Mosaic's own allocation list from the optimized MLIR
   (`llvm_tpu.allocate.tilespmem`, with its offsets);
2. a census of the kernel's jaxpr ref signature plus nested-only refs (the
   `emit_pipeline` buffer is nested, not in the signature).

They agree exactly on the **delta**, which is the quantity at issue.

At the production shape `b=1024, n=10240, k=2048`, by declared scratch shape:

| scratch | incumbent | composed | Δ B |
| --- | ---: | ---: | ---: |
| resident slice | `i32[10240]` 40 960 B | `f32[10240]` 40 960 B | 0 |
| lane-private → shared histogram | `i32[4112]` 16 448 B | `i32[272]` 1 088 B | **−15 360** |
| folded histogram | `i32[272]` 1 088 B | `i32[272]` 1 088 B | 0 |
| row buffer (`k + lanes`) | `i32[2064]` 8 256 B | `i32[2064]` 8 256 B | 0 |
| length vector | `i32[1024]` 4 096 B | `i32[1024]` 4 096 B | 0 |
| pipeline double buffer (`2 × tc`) | — | `f32[4096]` 16 384 B | **+16 384** |
| **total** | **70 848 B (69.19 KiB)** | **71 872 B (70.19 KiB)** | **+1 024 B** |

And by Mosaic's own `tilespmem` allocation offsets, read out of the optimized
MLIR (the allocations are packed, so consecutive offsets give the padded
sizes):

| | incumbent | composed |
| --- | --- | --- |
| offsets (words) | 0, 10240, 14464, 14848, 17024 | 0, 10240, 10624, 11008, 13184, 14208 |
| padded sizes | 10240, 4224, 384, 2176, 1024 | 10240, 384, 384, 2176, 1024, 4096 |
| high-water | 18 048 words = **72 192 B** | 18 304 words = **73 216 B** |

**Δ = 256 words = +1 024 B — the two methods agree on the delta exactly.** They
differ on the absolute (each carries the same 1 344 B of Mosaic tiling padding,
which the source census does not model). 73 216 B is **14.0 %** of the 512 KiB
subcore tile memory.

`v2`'s 15 360 B saving very nearly pays for `v4`'s 16 384 B pipeline buffer
outright — which is the specific composition risk `§3.2` names, and it does not
fire. The composed kernel also adds one extra `sflag` allocation (the
pipeline's DMA semaphores; offset 5 vs the incumbent's single one at offset 1).

**Spills: zero in both arms.** `'spill'` string hits = 0 in the incumbent's
bundle dump and 0 in the composed one.

### Priced behaviourally, by allocation failure

`final/vmemctl.py`, paired controls, both required:

- **POSITIVE** 4 KiB scratch, same DMA form → **compiles**. (Without this a
  negative control proves only that the harness can produce a compile error.)
- **NEGATIVE** 8 MiB scratch → **fails**
  `E3000: CompileTimeSparseCoreAllocationFailure … memory space spmem`,
  user upper bound 33 554 431 words.

Production-reachable footprint, modelled from the same allocation model and
checked against the allocator: production `n=10240` → 71 872 B (13.7 % of cap);
the worst slice `_pick_partition` can produce is `n=65536`, `b=1024` →
293 056 B = 286.19 KiB (55.9 %). **The Python guard cannot fire at any shape
`_pick_partition` produces.** It is an early, legible error, not a safety
property.

### ⚠ Two negative results about the sweep, recorded so they are not re-derived

1. **Bisecting on `n` is invalid.** A fine sweep appeared to show the composed
   candidate compiling at a *larger* `n` than the incumbent (124 672 vs
   124 160). `final/_guard_model.py` shows the guard verdict flips **12 times**
   over `n ∈ [124000, 125400)`, because `_pick_chunk` returns the largest
   divisor of the slice length ≤ 2048 and the pipeline buffer is therefore
   **non-monotone in `n`**. The "final compiles larger" reading is an artifact
   and is **not** reported as a result.
2. **The two arms overflow different memory spaces.** The incumbent's first
   refusal is in `spmem`; the composed candidate's is in `tile_spmem`
   (131 071-word / 512 KiB bound). Their thresholds are not comparable at all.

The guard is *permissive* relative to the allocator by −528 words at extreme
`n`, because it prices source allocas and not Mosaic's tiling padding. Only
reachable outside `_pick_partition`'s range.

---

## 5. `v12`'s call-site win — **confirmed in situ**, `static-compile`

`§3.3` requires confirming that the fusion `v12` deletes is one the
*production* caller actually pays, and forbids quietly keeping the −25.15 if it
cannot be.

**The production caller is
`src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py:831`**, and it hands
`sparsecore_topk` a plain **f32** array:

```python
scores_to_reduce = scores
if scores_to_reduce.dtype != jnp.float32:
    scores_to_reduce = scores_to_reduce.astype(jnp.float32)
...
topk_idxs = sparsecore_topk(scores_to_reduce, k, row_lengths=eff_row_lengths,
                            write_empty_rows=True)
```

So the entry glue is on the production path by construction. What that glue
costs is visible in the **optimized HLO ENTRY** at the production shape
(`final/hlo-inc.txt` vs `final/hlo-final.txt`):

incumbent —
```
%bitcast-convert_bitcast_fusion = s32[128,80,8,128]{3,2,1,0} fusion(%s.1), kind=kLoop
```
composed —
```
%bitcast.3 = f32[128,80,8,128]{3,2,1,0} bitcast(%s.1)
```

A real `kLoop` fusion over `f32[1024,10240]` becomes a free layout
reinterpretation. Both arms keep the `sparse-core-data-format-call` retile
(`{3,2,1,0}` → `{3,1,2,0}`); `v12` does not touch it, it is only now f32-typed
instead of s32-typed, at identical bytes. Nothing else in the ENTRY moves.

**Corroboration, `modelled`:** the deleted fusion reads and writes
1024 × 10240 × 4 B = 41.94 MB each way = 83.89 MB. At the tpu7x TC HBM figure
this exercise measured (2138 GB/s SparseCore read = 63.5 % of TC peak ⇒
≈3367 GB/s), that is **≈24.9 µs** — against `v12`'s `measured` −25.15 µs, a
1 % agreement. Mechanism and magnitude both check out.

**Residual uncertainty, stated rather than buried.** The AOT lowering feeds the
kernel from a *parameter*; production feeds it from a Pallas custom-call output
via a reshape and a conditional pad. The confirmation transfers because a
custom-call output is a fusion barrier exactly as a parameter is, so XLA has no
producer to fuse the bitcast-convert into in either case. That is an argument,
not a measurement of the full path — but the arm's own −25.15 µs *was* measured
whole-call, so the contribution is `measured`, and only this in-situ
attribution is `static-compile`.

---

## 6. `/xprof-llo` reading — `static-compile`, code size only

⛔ A bundle count measures **issue** cost and is blind to **stall** cost. In
this same kernel family `addupdate_scatter` is 5.2× a lane popcount in *time*
and 0.82× in *bundles*, and two builders have already read a bundle delta as a
cost proxy, once with the sign inverted. **Nothing below is a time.**

`final/bundle_stats.py <final-dump> <inc-dump>` (note the arg order):

| | incumbent | composed | Δ |
| --- | ---: | ---: | ---: |
| TEC bundles | 1 168 | 941 | **−227 (−19.43 % code size)** |
| SCS bundles | 196 | 196 | 0 |
| TEC nop-only | 10 (0.9 %) | 4 (0.4 %) | −6 |
| spill hits | 0 | 0 | 0 |

TEC counts independently confirmed against
`*sparse_core_specific_metadata.txt` (≈90 lines, the cheaper first read of the
19 dump files).

Movers, TEC opcode histogram: `vld` −253, `vadd.s32` −240, `vor.u32` −29,
`sand.u32` +23, `sor.u32` +16, `snop` +7, `stream.linear.gather` 2→3.
The `vld`/`vadd.s32` collapse is `v2`: the 16-deep `fold_priv` accumulate
(sixteen loads and fifteen adds per lane group) becomes a copy.

Invariants that had to hold, and did:

| primitive | incumbent | composed | why it must not move |
| --- | ---: | ---: | --- |
| `vst.idx.add.s32.msk` | 40 | 40 | `v2` changes the scatter *address*, not the scatter *count* |
| `vmpcnt.ones.xlane` | 61 | 61 | the counting pass is untouched |
| `vadd.scan.msk.s32` | 29 | 29 | the refinement scan is untouched |

Mosaic MLIR: the composed `memref.alloca` list is §4's table; the pipeline adds
one `sflag_tile 2×i32` for its DMA semaphores. No new memory space appears.

---

## 7. Correctness — job and gates

Submitted by this agent, tagged `agent-jobs,final-composer`.

| | |
| --- | --- |
| job | `j-49ee915f-6906-43dc-ba3f` |
| image | `…/vllm-torchtpu-hning_google_com:fa63d6df` |
| stage | `probes/25-final-correctness/stage`, mapped to `/root/p25` |
| `RUN_CMD` md5 (local, no trailing newline) | `b39a92ee465aaed141f25f1d985b8f8e`, 899 chars |
| recipe read-back | YAML-unfolded, **md5 identical** to local |

⚠ **A docker image tag is mutable** — `fa63d6df` was re-pushed mid-exercise in
this repo — so all three subject md5s are gated **in-job**, twice: once in
`RUN_CMD` before anything runs, and again inside each shard.

Two shards, one chip each, via `tpu-fanout.sh`:

- `composed` — the candidate. `arm_check.py --expect-suffix _final` **plus**
  `arm_check_final.py`, then the **frozen 68 unmodified** (no `-k`), then
  `xgate.py`.
- `control` — the pinned **incumbent** through the byte-identical harness, with
  `arm_check.py --expect-suffix NONE`. Without it, a red on the other shard
  cannot be told from a harness or image finding.

`arm_check_final.py` exists because `arm_check.py` matches with
`any(nm.endswith(suffix))`. Two of the seven suite shapes lower **two** kernels
(stage 1 and the `_keys` stage 2), so an `any` is green when only one of them is
the new body; and `endswith` has already failed spuriously in this exercise
because `v2` prepends its tag. `arm_check_final.py` requires **every** name over
**all seven** shapes to contain `_final` and to match `_c\d+_final$`, and
requires the incumbent read through the identical path to carry `_final` on
**none**.

The frozen 68 alone is not sufficient: nothing in it compares a variant to the
incumbent *file*. `xgate.py` does — 3 shapes × 5 score kinds × {full, ragged}
= 30 cells, exact-selection plus bit-identity against the pinned incumbent, and
it carries its own negative control (corrupt one index; both verdicts must
reject it).

**Expectation: bit-identical to the incumbent.** Every change above is either a
relocation, a re-typing that round-trips, or an addressing change on a
conflict-resolving scatter. A bit-identity failure would be a **real finding**,
not a gate bug.

### RESULT — GREEN. `Succeeded`, `tpu-fanout: 2/2 shards succeeded`

| gate | `composed` | `control` (incumbent) |
| --- | --- | --- |
| in-job md5 gates | `MAP_GATE_OK`, `SHARD_GATE_OK` | `SHARD_GATE_OK` |
| `arm_check.py` | `ARM_CHECK_OK` (`_final`) | `ARM_CHECK_OK` (bare, no suffix) |
| `arm_check_final.py` | `ARM_CHECK_FINAL_OK`, 7/7 shapes | n/a |
| frozen 68 | **68 passed**, 0 fail / 0 err / 0 skip, 24.45 s | **68 passed**, 25.62 s |
| xgate cells | 30 | 30 |
| xgate exact top-k | 30 / 30 | 30 / 30 |
| xgate **bit-identical to incumbent** | **30 / 30** | 30 / 30 (self-check) |
| xgate negative control | **`NEGATIVE_CONTROL_OK` — went RED** | `NEGATIVE_CONTROL_OK` |

**The candidate is bit-identical to the incumbent on every cell.** That was the
expectation, and it also closes every value-mediated mechanism in advance of
the measurer's job: whatever the composed timing is, it cannot be explained by
the kernel selecting different columns.

**The negative control did go RED** — corrupting one index to a real but losing
column produced `bit_equal=False` and the multiset reason
`max score difference np.float32(4.8411946), expected exactly 0.0`. Without
that line the 30/30 would be a tautology.

Arm activation held on **all seven** shapes, including the five that lower two
kernels each — both the stage-1 and the `_keys` stage-2 scope names carry
`_c<chunk>_final`, and the incumbent read through the identical path carries it
on none. Full table and artifacts in
`../probes/25-final-correctness/README.md` and its `outputs/`.

---

## 8. Known limitation found while composing — `_pick_chunk`'s small-chunk cliff

Inherited from `v4-c2048`, **not created by the composition**.

`_pick_chunk` returns the largest divisor of the slice length ≤ `CHUNK_WORDS`.
Over the 49 608 reachable `(b, n, k)` combinations surveyed by
`final/_chunk_survey.py`, the chunk ranges 16…2048 and **6 208 (12.5 %) resolve
to < 128 words** — e.g. `slice_len = 2096 = 16 × 131` with 131 prime → chunk
16 → 131 pipeline grid steps of 16 words each. That is a per-descriptor cost
regime nothing has measured.

No checked-in production shape reaches it, and none of the seven frozen-suite
shapes does (they resolve to 2048 / 1280 / 1664 / 2048), so the submitted
correctness job does **not** cover the regime.

**Decision: the file is not changed.** Adding a fallback selector would
introduce a code path no job has executed, and this md5 is already gated in a
submitted job and pre-registered for the measurer's probe `24-final`. Routed as
a follow-up in `../final_report.md`.

---

## 9. Files here

| file | what |
| --- | --- |
| `sparsecore_topk.py` | **the candidate** |
| `variant.diff` | `diff -u` incumbent → candidate |
| `aot_final.py` | host AOT, no TPU. Arms `inc`/`final`/`lit`/`pc1`/`pc3`, `--sweep` over the 7 suite shapes, `SCTOPK_DUMP=<dir>` for SC dumps |
| `bundle_stats.py` | `*_bundles.txt` → totals, nop density, spill hits, opcode histogram, row-walking blocks |
| `vmemctl.py` | paired 4 KiB / 8 MiB allocation-failure controls; arms `base`/`final`/`final-noguard` |
| `hlo-*.txt`, `jaxpr-*.txt` | the artifacts §3 and §5 read |
| `_guard_model.py`, `_chunk_survey.py`, `_vmemctl_*.py`, `_solo_*.py`, `_vm*.py` | scratch analysis and per-process arm files (leading underscore) |

Offline SC dump flags are `LIBTPU_INIT_ARGS`, set **before** `import jax` — XLA
rejects them in `XLA_FLAGS`.

**Nothing is committed.** The tree is intentionally dirty.
