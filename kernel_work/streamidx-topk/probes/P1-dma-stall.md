# P1 — how much of stage A's decode kernel is DMA-semaphore wait?

**Verdict: NOT DMA-bound.** Measured `count_sync_wait` = **16.65 % mean / 5.51 % median** of
TensorCore cycles. The brief's thresholds were ≥ 50 % → DMA-bound-and-done, ≤ 15 % → compute-bound.
16.65 % falls between them, one and a half points above the compute-bound line, with a median far
below it. The ≥ 50 % branch is decisively refuted; the challenge to `report.md` §4's headroom claim
is **not** sustained, and §4's headroom question stays open rather than being closed with "no".

Because the compute-bound branch is the live one, the same capture supplies the number the brief
asked for in that case: the **VPU issue rate, previously `assumed`, is now measured at 50.97 %**.

Everything in the "measured" table below is sampled hardware. Everything in the "static" table is
compile-time analysis and is labelled as such.

- job: **`j-e107f728-6fdc-47d8-98f9`** (`JobStatus` at time of reading: `Running` — shard 0's
  capture steps all returned `rc=0`; the only step still outstanding was `pack`, which tars the
  dumps and produces no data)
- shape: nightly decode, 64 seqs × 1 token, ctx 8704, `num_positions` 10240, page size 1024,
  `bq_sz=1`, `bkv_sz=2048`, `seq_batch_size=4`, grid 16, out `(64,80,128)` f32 = 2.62 MB
- kernel arm: `StreamIdxTC-d-bq_1-bkvp_2` — the incumbent's own kernel body, launched through
  `_stage_a.py` (`streamindex_topk.py` md5 `1e221f10cbf3573aa0d2ae78c3463532`)
- reference arm: `StreamIdxTC-d-bq_1-bkvp_1` — same bytes, same compute, **half the DMA granule and
  twice as many transfers**. `/xprof-llo` will not accept a single kernel's absolute numbers, and
  this pair is also the crude DMA discriminator area 3 asked for.

---

## 1. Measured: sampled GFC hardware counters

Periodic counter sampling over the profiler gRPC service, `interval_us: 20`, 2 s window,
66,053 samples, 1.3206 s covered. Kernel arm only.

**The wrap check passes.** The GFC counters are 16 bit; a reading is a residue rather than a rate
once `max_raw_count` reaches 65535. At `interval_us: 20` the largest raw count over all thirteen
counters is **38,521** — comfortably clear. The control capture at `interval_us: 1000` has eleven
of thirteen counters at ≥ 47,424 and five at ≥ 65,473, i.e. **wrapped**, and its percentages moved
by 10–40× against the same workload. Only the 20 µs capture is quoted.

| GFC counter | mean % | p50 % | max % | max_raw (65535 = wrap) |
| --- | ---: | ---: | ---: | ---: |
| **`count_sync_wait` (105)** | **16.65** | **5.51** | 101.28 | 38,486 |
| `count_mxu_busy_0` (56) | 95.49 | 94.80 | 101.37 | 38,521 |
| `count_mxu_busy_1` (57) | 0.12 | 0.14 | 0.15 | 56 |
| `count_mxu_busy_2` (58) | 3.74 | 4.31 | 4.64 | 1,764 |
| `count_vector_alu_instruction_0` (31) | 55.18 | 62.91 | 65.88 | 25,031 |
| `count_vector_alu_instruction_1` (32) | 52.70 | 60.09 | 63.04 | 23,955 |
| `count_vector_alu_instruction_2` (33) | 49.49 | 56.43 | 59.36 | 22,554 |
| `count_vector_alu_instruction_3` (34) | 46.50 | 53.02 | 55.75 | 21,185 |
| `count_vld_instruction_0` (37) | 48.27 | 55.10 | 57.92 | 22,008 |
| `count_vld_instruction_1` (38) | 33.58 | 38.36 | 40.36 | 15,336 |
| `count_vst_instruction` (35) | 23.57 | 26.77 | 28.09 | 10,672 |
| `count_scalar_alu_instruction_0` (10) | 4.00 | 4.22 | 52.11 | 19,800 |
| `count_scalar_alu_instruction_1` (11) | 2.34 | 2.46 | 2.87 | 1,090 |

Roll-up: **Vector ALU 50.97 %**, Vector Load 40.92 %, MXU 33.12 % (mean of three counters — see
§1.2), Vector Store 23.57 %, **Sync Wait 16.65 %**, Scalar ALU 3.17 %.

Interval-diff, kernel arm, mean %:

| counter | `interval_us: 20` | `interval_us: 1000` | iv1000 max_raw |
| --- | ---: | ---: | ---: |
| `count_sync_wait` | 16.65 | 1.47 | 65,473 ⚠ wrapped |
| `count_mxu_busy_0` | 95.49 | 2.34 | 64,221 ⚠ |
| `count_vector_alu_instruction_0` | 55.18 | 1.67 | 65,528 ⚠ |
| `count_vst_instruction` | 23.57 | 0.92 | 65,474 ⚠ |

### 1.1 What 16.65 % is a fraction of

The sampled window is the workload process running the stage-A launch in a tight loop. From the
trace-side capture of the same arm (§3) the device is **98.6 % busy** in that window and
**96.34 %** of the `XLA Ops` line is the kernel itself, so the sampled percentages are within a few
points of being per-kernel. Correcting for the ~5 % non-kernel remainder puts sync wait at roughly
**17.5 % of kernel cycles** — still nowhere near 50 %.

The distribution matters as much as the mean: **p50 = 5.51 %, max = 101.28 %**. The kernel is not
uniformly ~17 % stalled; it is almost never stalled, punctuated by short windows that are fully
stalled. That is the signature of a per-`bkv`-block fetch wait at the head of each iteration, not
of a kernel starved for bandwidth throughout.

### 1.2 `count_mxu_busy_0` = 95.5 %, and why it does not contradict `MXU static = 4.61 %`

These two numbers count different things and both are correct:

- **static 4.61 %** is *issue slots* — 32 `matmul` instructions in ~2,779 bundles.
- **measured 95.5 %** is *cycles the MXU is busy*. One `matmul` occupies the array for as many
  cycles as its operand is deep, so 32 instructions can keep it busy essentially all the time.

Taken together they say something the brief did not anticipate: **the MXU is busy almost every
cycle while doing very little arithmetic.** Only `mxu_busy_0` is loaded (95.5 %); `mxu_busy_1` and
`_2` are 0.12 % and 3.74 %.

*Modelled, with the arithmetic shown and the assumptions named.* Per decode call the kernel scores
64 seqs × 9216 padded positions × 32 heads × 128 dims = **2.416 × 10⁹ MACs**. At the measured
110.38 µs (P4, direct) that is 2.19 × 10¹³ MAC/s. Assuming a 256 × 256 MXU at 1.9 GHz (the clock
the counter sampling pins, see the wrap arithmetic), two MXUs, and 2 f8e4m3 MACs per cell per
cycle, peak is 4.98 × 10¹⁴ MAC/s → **≈ 4.4 % of peak**. The stationary operand explains it: q is
32 heads × 128 dims, so the array is loaded 128 deep of 256 and 32 wide of 256 = **6.25 % of its
cells**, which caps the achievable fraction at 6.25 % before any other loss. The MXU-generation
constants are assumptions; the 2.416 × 10⁹ MAC count and the 110.38 µs are not.

**Design read:** the largest single lever this capture points at is not DMA and not the VPU — it is
that stage A's matmul presents a 128 × 32 tile to a 256 × 256 array. Packing more heads or more
sequences into the stationary operand is the change that would move `count_mxu_busy_0`'s *useful*
fraction. This is a lead, not a measurement.

---

## 2. Static: LLO compile-time analysis, kernel vs reference

**All of §2 is static compile-time analysis — issue slots and allocation sizes, not measured
time.** Both arms captured with `--xla_jf_dump_to` / `--xla_jf_dump_llo_text=true` /
`--xla_mosaic_dump_to`; 12,671 jf files + 21 mosaic files per arm, 778 MB and 743 MB.

| field | kernel `bkvp_2` | reference `bkvp_1` |
| --- | ---: | ---: |
| **fully idle bundles** | **15.5 %** | 23.6 % |
| **overlap (MXU + mem)** | **6.3 %** | 4.7 % |
| **mem-stall, MXU idle** | **54.9 %** | 42.8 % |
| **MXU matmuls** | **32** (mxu0+mxu1, f8e4m3, ~256-lane staging) | 16 |
| **VMEM** | **4.17 MiB** — 87.0 % headroom of the 32 MiB scoped default, 96.7 % of 128 MiB HW | 2.14 MiB — 93.3 % / 98.3 % |
| MXU static occupancy | 4.61 % | 3.54 % |
| VALU static occupancy | 62.08 % | 47.73 % |
| VLOAD / VSTORE / SALU | 38.18 / 18.08 / 13.35 % | 29.02 / 12.58 / 17.53 % |
| **spills + fills** | **996 + 999 over 2,779 bundles = 0.72/bundle** | 446 + 446 over 1,808 = 0.49/bundle |
| SALU bubble | 15.5 % | 20.6 % |
| compiler-inserted `tpu.relayout` | 37 | 45 |
| tool verdict | VPU-serialized (VALU dominates, MXU starved) | VPU-serialized — *identical label* |

Per `bkv` page the two arms are the same kernel: `bkvp_2` does 2 pages and 32 matmuls per loop
body, `bkvp_1` does 1 page and 16 — 16 matmuls per page either way, as intended.

Two internal inconsistencies in the tool worth recording rather than smoothing over: the spill
block reports 2,779 bundles while the `totals` block reports 2,347 (4 empty), and both arms get the
**same verdict line** despite differing on every field above — the behaviour `/xprof-llo` warns
about. The raw fields are the data.

**Spills answer the brief's side question: yes, there are spills — 1,995 spill+fill instructions,
0.72 per bundle — and the VMEM working set is 4.17 MiB, i.e. 3.3 % of the 128 MiB the hardware has.**
The kernel is spilling registers while leaving 96.7 % of VMEM untouched.

### 2.1 The biggest static stall block is not DMA — instruction-level evidence

`get_llo_fit_summary` names the top stall run as bundles **`0x7ac`–`0xada`, 815 bundles (29 % of
2,779), classified `mem_stall_mxu_idle`**. Dumping it (`get_llo_bundles --address_range=0x7ac-0x7e0`,
53 bundles returned, not truncated) shows what is actually there:

```
0x7ac  vst [vmem:[#allocation229_spill]] ;; vmul.f32 ;; vrot.slane %…,1 ;; vadd.f32 ;; vadd.f32
0x7ae  vadd.f32 ;; vadd.f32 ;; vrot.slane %…,4 ;; vadd.f32 ;; vpop.f32.mrb[126].mxu1 ;; vpop.f32.mrb[127].mxu0
0x7af  vrot.slane %…,4 ×3 ;; vadd.f32 ;; vst [vmem:[#allocation332_spill]] ;; vst [vmem:[#allocation219_spill]] ;; vpop.f32.mrb[127].mxu1
```

That is a **cross-lane rotate-and-add reduction tree** (`vrot.slane` by 1, 2, 4 …) draining matmul
results out of the MXU result buffers (`vpop.f32.mrb[*]`), interleaved with **register-spill
stores**. There is not a semaphore wait or a DMA in the range. The tool's `mem_stall_mxu_idle`
label is counting spill traffic as memory traffic, which is exactly why the label reads as a DMA
problem and is not one.

---

## 3. Trace-side: is the work in the kernel, and what does halving the DMA granule cost?

Capture gate (`xprof-dumper inspect`): `Tensor Core` line present with 24,150 events (kernel) /
15,350 (reference) — the trace flags took. `XLA TraceMe` present too.

| | kernel `bkvp_2` | reference `bkvp_1` |
| --- | ---: | ---: |
| kernel share of the `XLA Ops` line | **96.34 %** | 96.99 % |
| `overlayer-overhead` per call | 127.20 µs × 50 | 155.44 µs × 50 |
| `XLA TraceMe` `bnd,bmd->bnm` regions per call | 80 × 399.9 ns = 32.0 µs (25.2 % of the call) | 144 × 259.1 ns = 37.3 µs (24.0 %) |

The region counts confirm the loop structure directly: 80 = 16 grid steps × 5 `bkv` blocks
(`cdiv(9216, 2048)`), 144 = 16 × 9 (`cdiv(9216, 1024)`).

**These are LLO-instrumented and contended (four shards, one chip each, shared host and HBM), so
they are not timings.** As a *ratio* against its own reference, however: **halving the DMA granule
and doubling the number of transfers costs 22.2 % more time** (155.44 / 127.20). A kernel sitting
on the HBM bandwidth wall would barely notice — the bytes are identical. A 22 % penalty says the
kernel is sensitive to per-transfer overhead, not to bandwidth, which is the same conclusion §1
reaches from `count_sync_wait` and §2.1 reaches from the bundles.

---

## 4. What this settles

- **Settled: stage A's decode kernel is not DMA-stall-bound.** 16.65 % measured sync wait against a
  50 % bar. The challenge to `report.md` §4 fails; §4's headroom stays on the table.
- **Settled: the VPU issue rate is 50.97 %** (per-slot 55.18 / 52.70 / 49.49 / 46.50), replacing the
  roofline's largest `assumed` input with a measured one.
- **Settled: there are register spills** — 0.72 spill+fill per bundle, 1,995 instructions — against a
  VMEM working set of 4.17 MiB that leaves 96.7 % of the hardware's VMEM unused. Spilling with that
  much VMEM idle is a lowering problem, not a capacity one.
- **Opened: the MXU is ~95 % busy at a modelled ≈ 4 % of peak**, because the stationary operand fills
  6.25 % of the array. This was not on the probe list and is now the largest lever P1 can see.
- **Not settled: what the remaining ~75 % of the kernel outside the `bnd,bmd->bnm` regions is doing.**
  §2.1 identifies the largest block as a rotate-reduce tree with spills; attributing the rest needs a
  `get_llo_schedule_analysis` pass that was not run.

## 5. Tools that returned nothing, and whether that is expected

- `get_llo_utilization` on the **`interval_us: 1000`** capture returned numbers, but they are wrap
  residues (§1) and are not quoted as rates. Expected — documented behaviour of a 16-bit counter.
- `get_kernel_stage_breakdown` was not run; the `XLA TraceMe` line is populated here
  (`bnd,bmd->bnm`), so it would not have been empty.
- `check_kernel_profiling`, `get_overview`, `get_kpi_metrics`, `get_roofline_model`,
  `get_smart_suggestions`, `get_device_wall_report` were deliberately not run — dead or false-green
  on v7x + libtpu 0.0.44.1.
