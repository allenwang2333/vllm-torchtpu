# Probe 06 — `llo-incumbent`: open the incumbent kernel from the inside

**Question.** Six, ranked. The first is the caller's premise: **is the score DMA actually exposed —
is there any compute/DMA overlap at all today?** Then per-unit occupancy, what
`plsc.addupdate_scatter` lowers to, bundle/NOP/spill density, VMEM high-water, and the Mosaic MLIR
for the wave body.

**Why it needed hardware.** Q1 gates whether the caller's change has anything to hide, and Q5 gates
whether a second slice buffer fits. Neither is answerable from source: the source shows
`cp.start(); cp.wait()`, but that says nothing about what the compiler scheduled between them.

## Config

Two jobs, one 4-chip tpu7x pod each, one arm per chip. Image `fa63d6df` with a pristine
`git archive b11b6eb6 src` mapped over it; **pod-side md5 `44353b347c96e0eedbd742c710d4540a`**
verified in-job. Pod-side shape line: `b=1024 n=10240 k=2048 -> P=1 slice_len=10240 num_waves=32`.

| job | status | produced |
| --- | --- | --- |
| `j-5726664a-06d0-40f9-88cf` | 4/4 Succeeded | `.xplane.pb` traces, `jf` + Mosaic (TensorCore) dumps |
| `j-f65d952c-e64c-40e4-ade8` | **"Failed" by design** — shard 1 is the bogus-flag control and libtpu exits 1 on an unknown flag; shards 0/2/3 rc=0 | SparseCore bundle dumps, SC MLIR/LLVM IR, high-water protobuf |

`probe.py` is the driver (the checked-in `sctopk_xprof_capture.py` could not be reused: it hard-fails
at `not shape["two_stage"]`, i.e. exactly at `P=1`, and sets no dump-side flags at all).

**Two capture findings that are themselves results.** First, the documented `--xla_jf_dump_to` chain
**contains no SparseCore program** — the kernel is an `async_execution_thread="sparsecore"`
`call-start`/`call-done` with `offload: OFFLOAD_KERNEL`, and the 781-bundle `jf` dump is the
TensorCore *launcher* (78.73% `<no hlo>`). Reaching the SC code needs an undocumented flag family
found in libtpu 0.0.44.1's own DebugOptions name table: `--xla_sc_dump_bundles_to`,
`--xla_sc_dump_mlir_to`, `--xla_sc_dump_llvm_ir_to`, `--xla_sc_high_water_mark_protobuf_file`,
`--xla_mosaic_enable_llo_source_annotations`. Second, shard 1 deliberately passes a misspelled
flag; libtpu prints `Unknown command line flag` and exits 1. **So an accepted flag is proven
recognised, and an empty dump dir means "nothing to write", not "flag silently ignored".**

## Answers

Everything below is `[static]` — read out of the compiler's dumps — except the per-TEC durations,
which are `[measured, contended]`: 4 shards shared a host and HBM, so no duration here is a
benchmark number.

### Q1 — exposed? **Yes.** Three transfers in the program, identified by size:

| bundle | transfer | size | wait | gap |
| --- | --- | --- | --- | --- |
| `0xf` | gather | 1024 w = `len_vmem` | `0x10` | 0 |
| `0x16` | scatter | 2048 w = `rowbuf` | `0x17` | 0 |
| **`0x28`** | **gather, 10240 w = `slice_len`** | **the score DMA** | **`0x2d`** | **4 bundles** |

The four bundles are `sand.u32`, `seq.s32`, and **two literal `snop`s**. The compiler had slack
across a 40 KiB HBM read and filled half of it with nothing.

**Structurally, overlap is impossible today**: the module has 3 `enqueue_transfer` / 3
`transfer_wait` and exactly **one `resident` alloca (`10240xi32`)**. A second buffer is required,
which is exactly what the caller's change adds.

> ⚠ **The agent's own citation for this was wrong** — it quoted `0xf`/`0x10` as the score gather and
> concluded "adjacent bundles". That is the 1024-word lengths DMA. Verified by hand against
> `bundles-incumbent-tec.txt`; the conclusion survived, the evidence changed. **Identify a transfer
> by its size against a known buffer extent, never by position or neighbours.**

### Q3 — `plsc.addupdate_scatter` → **one instruction, `vst.idx.add.s32.msk`.** 40 in the program.

Per 64 elements (unroll-4 × 16 lanes), `hist0` issues **4** and a refinement pass issues **4**. **The
mask does not remove the instruction.** So mask density is a *memory-side* cost, which is why probe
03 sees 35.243 vs 116.350 µs from bundle-identical code.

**The loops are not issue-bound, and the discriminator needs no clock.** Refinement is 14 bundles
per 64 elements against `hist0`'s 9 — ratio **1.556**. Measured, the two are within 3%. Off by 1.6×
*in the wrong direction*: refinement issues 56% more instructions and runs slightly faster. **The
pace-setter is the `vst.idx.add.s32.msk` read-modify-write into tilespmem.**

Marginal static cost of one scatter pass, from the isolated probe arms: `cvt_p1` 7 bundles →
`scatter_p1` 10 → `scatter_p2` 10+10. **+3 bundles per 64 elements per pass, and stacking passes
adds no issue penalty.**

### Q4 — 1168 bundles, 2544 slots, mean 2.178/bundle — but **4.00–4.67 in every hot loop**

**Zero pure-NOP bundles in any hot loop** (program-wide 3.94%, all in scalar setup/teardown). Spills
are 110 slots / 1168 bundles, and **the only hot loop containing any is emit** (`.LBB2_105`, 9
spill/fill slots in 26 bundles). `cvt`, all four scatter loops and the histogram reductions are
spill-free. **"Schedule it better" and "reduce register pressure" are both dead ends** — except in
emit, which is 6.1% of the kernel.

### Q5 — VMEM high-water **18,048 words = 70.50 KiB per subcore**, confirmed twice

`sc_hwm.pb` field-4 varint `80 8d 01` = 18048; and independently from base addresses in the bundle
dump — resident `0x0000`, priv `0x2800`, glob `~0x3810`, rowbuf `0x3A00`, len_vmem `0x4280`, DMA
scratch `0x4680`. Top = `0x4680` = 18,048 w, of which 336 w is inter-buffer padding.

**A second `10240xi32` slice buffer → 110.50 KiB, ~21.6% of 512 KiB. The double buffer fits with
room to spare.** (The §1 diagram's 180.5 KiB is the `n=133120` shape, where the slice alone is 130
KiB — not a conflict.) **Open**: `kTileSpmemCapacity` is a compiled constant, not a string, so the
512 KiB *denominator* could not be confirmed compiler-side. It would have to be wrong by 2× to
matter.

### Q2 — measured per-unit occupancy is **not obtainable for this kernel**, and no substitute was faked

The only route to real counters is GFC periodic sampling, whose verified index table is entirely
**TensorCore** (`count_mxu_busy`, `count_vector_alu_instruction`, …). This kernel runs wholly on
SparseCore; the TensorCore line has 78 events all of duration 0 ns. No SC equivalent exists in that
table. What is available instead: `sc_topk_direct_….call-done` is **83.79%** of the XLA-op line and
`OFFLOAD_KERNEL` is **86.29%** of the SparseCore offload line, so the block really is the kernel.

> ⚠ **All 16 `TEC n` lines report 356,434–356,451 ns — a 17 ns spread that reads as perfect subcore
> balance. It is an artifact.** The Mosaic dump has `sc_tpu.sfence` (904) and `sc_tpu.barrier` (907)
> at the end of the body, so every subcore's scope ends at the same instant by construction. The
> source has no `subcore_barrier` — the lowering emits one. This both **hides** the 1.28× decode
> imbalance from the trace and **confirms** that the cost is `max` over subcores rather than the
> mean.

### Q6 — `ref_body` → 19 MLIR ops; `hist0_body` → 15

The 4-op difference is the prefix match, and it is exactly the 15-vs-19 that appears downstream as
9-vs-14 bundles. Four loop-invariant recomputes survive into the body (`lane_iota*256` and two
broadcasts). Hoisting them removes ~3 of 14 bundles, and since the loop is **not** issue-bound the
expected saving is near zero — **not worth a job**, recorded so nobody spends one.

## Decided

1. **The caller's premise is confirmed and their change targets the right object** — one buffer, no
   overlap possible, and the compiler idling across the transfer. Report §6 upgraded from `modelled`
   to `static`. The −7.18%/−2.02% ceiling is unchanged; only its basis is now firm.
   ⛔ SUPERSEDED by `probes/08` (`j-300d4690`): the ceiling is **−4.12% / −1.97%**. Q1's structural finding is untouched — the DMA
   really is exposed — but its *size* was a bandwidth model, and the measured exposed cost is 0.40×
   it. Q1's own 4-bundle slack reading is the clue: two of those four bundles are `snop`, so the
   compiler had little to hide and hid little.
2. **The double buffer fits** — 21.6% of VMEM. This removes the one structural objection to it.
3. **✅ The cost model in report §5d is contradicted — and this probe was right.** A refinement pass
   has a *sparse* mask and should therefore cost ~35 µs by probe 03's own price sheet, but the fit
   assigned it 114.3 µs — 3.2× off, ~150 µs misattributed.
   **`probes/07-refinement-fixed-vs-walk` (`j-098b260f-a9f7-4fe5-937c`) settled it in this probe's
   favour**: a level is **57.196 µs**, of which the `L` walk is **38.004 µs = 1.1876 µs/wave**
   against probe 03's `scatter_sparse` at 1.1013 µs/wave — **7.8% apart**. The misattributed ~150 µs
   was in the *base*, not the level: 43 µs of fixed `zero_priv`/`fold_priv` sweep (the explanation
   this probe favoured, right in kind) and ~81 µs of per-wave cost still unattributed. Candidate 1's
   prize fell from 51.4% to **18.7%**.
   **The transferable part**: a machine-code reading overturned a one-parameter fit that had
   validated out of sample to −0.14%. It could do that because the fit's validation ran along wave
   count — the axis every confounded term scales on — so it could not fail.
4. Emit is the only place register pressure exists; schedule quality and LICM are dead ends.

## Bonus, same-job control

Shard 3 captured `incumbent_keys` at the same shape in the same job: `TEC 0` 372,062.8 ns against
`s0`'s 356,434.2 ns = **+4.38%** for one extra `2064xi32` alloca, one extra DMA pair and two extra
`vst`. Both arms shared the contention, so the ratio is readable even though neither absolute is.
