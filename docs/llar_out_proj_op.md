# Batch O, arm 1 — two-shot out-proj all-reduce behind a token-switched opaque op

Branch `gnovack/k3-llar-op`. The op, the layer class, the gate, and the tests
are the deliverable; this note records what they measured. **The arm is a net
regression as wired and stays behind `TPU_LOW_LATENCY_ALLREDUCE=0`.**

## What was built

`vllm_torchtpu::llar_out_proj_reduce_7168` — one opaque custom op, called
unconditionally by `LowLatencyRowParallelLinear` (the K3 MLA and KDA o_proj),
switching internally on the concrete token count: the two-shot Pallas kernel
at or below the wired ceiling, the stock `dist.all_reduce` above it. Unlike
`gnovack/llar-k3-integration`, there is no forward-context lookup and no
single-compile-size restriction: the workspaces are fixed-size op arguments
sliced per shape inside the traced JAX body, so every compile bucket presents
the same graph interface. The server compiled and served the full
`COMPILE_SIZES=4,8,16,32,8192` ladder with the op live — the thing the
previous branch structurally could not do.

## Threshold: verified, T<=8 stands

`scripts/k3/llar_out_proj_check.sbatch` (job 375, 32 devices, bf16, SC-offload
threshold at the production 64 MiB), K3's four decode payloads as the kernel
sees them:

| payload | T | XLA us | two-shot us | ratio |
|---|---|---|---|---|
| 56 KiB | 4 | 19.71 | 14.09 | 1.40x |
| 112 KiB | 8 | 22.06 | 18.32 | 1.20x |
| 224 KiB | 16 | 23.59 | 26.59 | **0.89x — kernel loses** |
| 448 KiB | 32 | 26.91 | 45.44 | 0.59x |

Batch N's "ties at 224 KiB" does not reproduce on this slice: T=16 is a loss,
so the shipped ceiling (T<=8, i.e. the 128 KiB gate) is right. Verification
against psum passed in bf16 (scaled tolerance) and f32 (exact).

## The sweep: kernel fast, integration slow

Same slice (ici-2x2x4-b), same tree, dummy weights, fp8 KV, MAX_NUM_SEQS=32,
COMPILE_SIZES=4,8,16,32,8192, mnbt 8192, mml 10240; only the flag differs.
Pre-registered prediction was c=1 -3.6%, c=4 -3.2%, c=8 -1.9%, c=16/32 flat.

| c | control TPOT ms | arm-1 TPOT ms | arm-1 vs control | predicted |
|---|---|---|---|---|
| 1 | 15.45 | 16.80 | **+8.7%** | -3.6% |
| 2 | 15.64 | 17.00 | **+8.7%** | -3.3% |
| 4 | 15.92 | 17.32 | **+8.8%** | -3.2% |
| 8 | 17.73 | 18.03 | **+1.7%** | -1.9% |
| 16 | 19.45 | 19.52 | +0.4% | flat |
| 32 | 24.86 | 25.97 | +4.5% | flat |

(Job 288's table is not comparable: it predates the d60a797a bucket-8/16/32
decode tuning — its c=8/16/32 are 22.52/31.70/51.25 against *both* of these
arms' ~18/19.5/25. The control above is the honest baseline.)

## Why: the o_proj AR was nearly free, and the boundary is not

Decode trace at the T=4 bucket, ~31 steps, rank0 (xprof hlo_stats):

- Arm-1 runs the kernel 93x/step at **14.76 us** each (vs 19.7 us for the
  standalone XLA AR) — device-side this *saves* ~0.45 ms/step.
- But in the control, the o_proj AR does not exist as a line item: XLA's
  collective combiner fuses it with a second [T,7168] reduction into a
  **two-way all-reduce at 19.28 us for the pair** — the o_proj member is
  ~free. Separately, each layer has a four-way (7168+3584+7168+3584) fused AR
  at ~28 us in *both* arms.
- So the kernel's 1.4x on an isolated 56 KiB payload buys nothing the combiner
  wasn't already giving, and each of the 93 opaque crossings per step costs
  host-side dispatch (graph-split/eager fallback, per-call kernel-key
  hashing, workspace writeback copies): net +1.35 ms/step at c=1 after the
  ~0.45 ms device-side saving, i.e. ~19 us of host overhead per crossing.
- At c=32 the fallback (clone + dist.all_reduce inside the op) keeps the
  o_proj AR out of the fused pair, costing ~+4.5% without any kernel
  involvement. c=16 landing flat says the fusion matters less at 224 KiB.

This is the same trap the llar design doc recorded for the latent-MoE pair
("splits what the combiner was fusing, and the lost fusion costs more than
the faster kernel gains: +5.5% device time"), now measured for the o_proj:
**on current k3-dev the per-layer collectives are fused, so any opaque-op
reroute of one member starts from behind.** Note the Batch D trace the arm
was sized from (187 unfused collectives/step at ~26 us) does not describe
this tree — with hierarchical EP + intra-chip latent TP the ARs fuse.

## Fallback equivalence

Above the threshold the op runs `dist.all_reduce` on a clone of the input
over the same device group — the same collective the stock
RowParallelLinear forward issues, plus a value-preserving copy. The kernel
branch's numerics are float-reassociation-equivalent (bf16 partial sums in a
different order), verified against psum in bf16 and exactly in f32 by the
check job. c=16, where every o_proj takes the fallback, lands within 0.4% of
the control end-to-end.

## Consequence for arm 2

A fused out-proj+AR kernel with overlap faces the same fusion-membership
arithmetic: it replaces a reduction the combiner currently amortizes to ~0.
The ~14.5 us/host-crossing overhead of the opaque op also has to go (93 eager
crossings per step) before any kernel-side win is visible end to end.

## Reproduce

```bash
# microbench + numerics (30 min, 4 nodes)
sbatch --partition=ici-2x2x4-b scripts/k3/llar_out_proj_check.sbatch

# arm / control servers, then sweep each
TPU_LOW_LATENCY_ALLREDUCE={1,0} TPU_MOE_HIERARCHICAL_EP=1 \
TPU_LATENT_PROJ_INTRA_CHIP_TP=1 MAX_NUM_SEQS=32 \
COMPILE_SIZES=4,8,16,32,8192 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=10240 \
PROFILE_DIR=$PWD/results/<arm>/traces KIMI_K3_LOG_ROOT=$PWD/results/<arm>-log \
sbatch --partition=ici-2x2x4-b --job-name=<arm> \
  scripts/k3/slurm_launch_k3_tp32_tp.sbatch --load-format dummy --kv-cache-dtype fp8

ENDPOINT=$(cat results/<arm>-log/endpoint.txt) CONCURRENCIES="1 2 4 8 16 32" \
ARM=<arm> OUTPUT_DIR=results/<arm>/sweep scripts/k3/concurrency_sweep.sh
```

Jobs: 375 (check), 376 (arm-1), 377 (control). Raw data:
`results/llar-op-check-375/`, `results/llar-arm1/`, `results/llar-control/`.

---

# Batch R2 — the opaque-op boundary, made free

Branch `gnovack/k3-llar-op` (continuation). Batch O's arm regressed +1.33
ms/step *on device* — not host dispatch, not lost overlap (both refuted by
the trace): the JAX body wrapper around the kernel was generating per-layer
glue. The op-level trace inventory at the T=4 bucket (93 layers):

| glue op (per layer)        | us   | mechanism |
|----------------------------|------|-----------|
| 2 slices of the max-size workspace | 2.6 | `rs[..., :cols]` |
| 2 dynamic-update-slice fusions     | 7.9 | `.at[].set` writeback |
| reshape relayouts in+out   | 4.6  | `[T,7168] -> [32,T*224]` is NOT a bitcast (the two shapes tile differently) |
| reduce as a squeeze        | 1.0  | XLA lowers `[32,1,896]->[32,896]` (size-1 middle dim) as a *reduce* |

Plus ~0.5 ms/step of slowdowns in unrelated fusions (layout contagion from
the custom-call boundary).

## What did NOT work (measured, in order)

1. **Write-only workspaces.** Making the kernel mutate persistent buffers
   through its input refs fails: XLA *stages* every custom-call operand into
   fresh VMEM/SMEM buffers per call — a step counter written through the
   input ref never reaches the persistent buffer (chain-HLO and a step
   readback test both confirm). Donation/aliasing (`buffer_donor` →
   `donated_indices`) is plumbed end-to-end but cannot help: the graph holds
   the workspaces live across all 93 calls.
2. **Stateless single slot + full barrier per call.** Correct under every
   staging regime, but the serial barrier costs +3.5 us/call (17.1 us vs the
   flat kernel's 13.5 at the same chunking — measured; scalar-div addressing
   and workspace provenance were both bisected out as causes).

## What works: static slots

The call index of each layer is a *compile-time constant* of the graph (93
unrolled callsites), so the slot rotation needs no runtime state at all:
`slot = layer_ordinal % 2`, with the first layer of every execution taking a
barrier-first variant to cover the step boundary. Consecutive calls
alternate slots, so the overlapped-barrier protocol (barrier after the
phase-1 sends, hiding its ~2.9 us) is exactly the original one's. The kernel
carries no step counter; workspaces are two-slot per-call scratch, never
returned, never written back.

Two integration landmines are encoded in the wiring: torch_tpu's custom-op
bridge drops trailing **bool** args (and `jax_placeholders` drops trailing
statics generally), so `slot`/`barrier_first` are ints and the three used
variants — (0, first), (0, -), (1, -) — are separate inner ops with the
statics baked into closures; and each inner op is keyed to its bucket by
exactly-sized per-bucket workspaces (no runtime slicing anywhere).

## Measured (job 414, 32 devices, bf16, SC-offload at production 64 MiB)

Per-op latency, chained (slope), us:

| payload        | XLA   | flat two-shot | natural (serial barrier) | **static slots** |
|----------------|-------|---------------|--------------------------|------------------|
| [4,7168] 56KiB | 19.65 | 14.09*        | 17.21                    | **12.40 (1.58x)** |
| [8,7168] 112KiB| 21.97 | 18.32*        | 22.40                    | **17.24 (1.27x)** |
| [16,7168] 224KiB| 23.60| 26.59*        | 32.09                    | 25.50 (0.93x) |
| [32,7168] 448KiB| 26.91| 46.03         | 52.61                    | 44.62 (0.60x) |

(*flat numbers at the kernel's equivalent flat views [32,896]/[32,1792]/...
from jobs 375/405/412.) The true crossover at the real shapes: the kernel
wins at T<=8, breaks even at T=16 — the shipped T<=8 threshold stands, and
this time it was measured at the shapes the model actually reduces.

**Chunk-misalignment hypothesis (task 3): refuted.** The flat kernel at
224/448/896-element chunks (1.75/3.5/7 lane tiles) costs 12.60/13.22/14.02 us
— time scales gently with bytes, no alignment penalty; and the kernel never
sees misaligned chunks anyway, because the wrapper chunks along the flat
row-major index: at width 7168 over 32 devices every wired bucket lands on
whole 128-lane tiles by construction (w = 224*T, T a multiple of 4).

## Server arm (job 424) — the boundary is cheaper, but not free

Same server config as the control (dummy weights, fp8 KV, COMPILE_SIZES
4,8,16,32,8192, mnbt 8192, mml 10240, MAX_NUM_SEQS=32, hierarchical EP +
intra-chip latent TP). Canary text is bit-identical to the control's (dummy
weights are deterministic). Device delta is the step-normalized per-category
self-time diff of the c=1 decode traces (job 377 control):

| c | control TPOT ms | R2 TPOT ms | delta |
|---|------|------|-------|
| 1 | 15.45 | 15.79 | +2.1% |
| 4 | 15.92 | 16.27 | +2.2% |
| 8 | 17.73 | 18.03 | +1.7% |
| 16 | 19.45 | 19.44 | -0.0% |
| 32 | 24.86 | 24.68 | -0.7% |

Device: **+0.313 ms/step** (arm-1 was +1.329). Decomposition:

| mechanism | ms/step | evidence |
|---|---|---|
| kernel custom-calls, 93/step @ 15.78 us | +1.59 | vs 12.40 us in the bench chain; the barrier absorbs per-layer cross-rank skew the isolated bench never sees |
| all-reduce category (o_proj AR removed, pair split) | -2.04 | -93 ARs/step, leftover member standalone |
| [7168,384] weight-staging transpose copies | +0.31 | 129/step @ 2.42 us vs control's 62/step @ ~0: the op's (2,128)-tiled 4-D output layout shifts XLA's global layout assignment for the MoE weights — layout contagion, not workspace glue |
| existing loop/conv fusions slower, same op counts | +0.34 | same contagion |
| output squeeze as reduce, 93/step @ 1.09 us | +0.10 | XLA lowers the size-1-middle-dim squeeze as a reduce; a Mosaic-level view is refused ("minormost dimension unchanged" / tiled-slice alignment), so this one is structural at this op shape |

The Batch-O glue itself — slices, dynamic-update-slice writebacks, writeback
copies — is GONE from the trace (0 ops). What remains is the kernel's own
in-server price plus layout contagion from the custom-call boundary. Net:
the arm is still a small regression at c<=8 and stays behind
`TPU_LOW_LATENCY_ALLREDUCE=0`.

**Conclusion for the workstream.** The kernel is real (1.58x at [4,7168],
1.27x at [8,7168], measured at the shapes the model uses). The boundary can
be made *glue-free* but not *layout-neutral*: with XLA owning layout
assignment around opaque custom-calls, a boundary whose operand/result
layouts differ from the collective it replaces taxes its neighbours. The
remaining +0.3 ms/step is below arm-1's by 4x and fully attributed, but it
is not <= -0.2, so the arm does not ship. The fix that would matter is a
custom-call result layout that matches XLA's own all-reduce output
({1,0:T(4,128)} on VMEM); that requires XLA-side layout control the current
torch_tpu custom-op path does not expose.

---

# Batch W — packing-dimension layout pin: layout pins, kernel can't consume it

Job 448 (4-node ici-2x2x4-d, phase-1 dumps only; server never ran). The
ragged-paged-attention packing-dimension trick (Jiang et al. sec. 3.1.1):
the op's XLA-visible view gains a size-4 packing dim in the second-minor
position, `[T, per_row, 1, w]` -> `[T, per_row//4, 4, w]`, and the kernel
unpicks it with a preprocessing `ref.reshape` back to the working view.

**The pin works at the XLA level — measured result layout (after layout
assignment, `module_0041.jit_chain_body.after_optimizations.txt`):**

```
bf16[4,2,4,896]{3,2,1,0:T(4,128)(2,1)S(1)} custom-call   (T=4, x20)
bf16[8,1,4,1792]{3,2,1,0:T(4,128)(2,1)S(1)} custom-call  (T=8, x20)
```

i.e. exactly XLA's all-reduce tiling T(4,128) on VMEM, vs the committed
op's `bf16[4,8,1,896]{3,2,1,0:T(2,128)(2,1)S(1)}` (job 414). The
contagion terms would in principle have died.

**But the packed buffer is unconsumable by the kernel.** All four dtype/T
combos fail Mosaic compilation at the phase-1 DMA:

```
tpu.memref_reshape : memref<4x2x4x896xbf16, #tpu.tiled<(4,128)(2,1),[14,7,7,1]>>
                  -> memref<4x8x1x896xbf16, #tpu.tiled<(4,128)(2,1),[56,7,7,1]>>
"Expected the 2nd minor dimension is aligned to the tile"
```

The packed ref is DENSE (strides [14,7,7,1]: 4 chunks share one tile-row
group, one sublane each); the working view is PADDED ([56,7,7,1]: one chunk
per group, sublane 0). A view reshape cannot convert dense<->padded, and the
three Mosaic rules this workstream has now measured close every other
arrangement:

1. extent-1 dynamic slice of the tiled (sublane) dim — refused
   ("tiled-slice alignment"), which rules out indexing the packed ref
   directly: the fastest-varying chunk bits always land on the sublane dim,
   because the 896-element chunk is 1.75 lane tiles and the packing splits
   the width *between* chunks;
2. dense->padded view merge — refused ("2nd minor dimension aligned to the
   tile", this job);
3. 2-D <-> 4-D view reshape — refused ("minormost dimension unchanged",
   job 440).

Making the packing dim interior to the chunk instead ([T, 8, 4, 224]) keeps
the tile but breaks lane alignment (minor 224) and reintroduces the
relayout at the output squeeze. The only unconsumed route is a data-moving
in-kernel unpack/repack (~2 extra payload round-trips per call, est.
+0.1-0.2 ms/step against the ~-0.34 the pin would have bought — at best
marginal against the -0.2 bar, and a different arm than the one measured).

**Conclusion: the two-shot AR op is shelved for good on this stack.** The
result layout CAN be pinned to match XLA's all-reduce (that is the finding
of record), but no Mosaic-legal consumption of the pinned buffer exists at
the K3 chunk geometry, so the +0.313 ms/step of R2 cannot be converted.
Branch state: the packed-view attempt is the uncommitted diff in this
worktree (tests still 35/35; it fails only at Mosaic compile time).

---

# Packed-layout probe — transport rules, and the fused o_proj+AR kernel

After Batch W shelved the layout-pin route, the question moved to fusing
the o_proj matmul and the two-shot exchange into one custom call over the
PACKED payload (`[28, 2, 128]` tiles at T=1, the layout the dot's
accumulator naturally fills). All of this lives in
`scripts/k3/packed_layout_probe.py`; one variant per fresh job (a second
collective workload in one allocation hangs).

## The transport rules (each one measured, one job per arm)

The exchange deadlocks at execution time — codegen is always fine — unless
**every rank receives exactly as many remote copies as every other rank,
per phase**. The bisect, each variant changing one thing off the working
offset-rem pattern (every rank receives 31):

| job | variant | change vs working pattern | result |
|-----|---------|---------------------------|--------|
| 494 | statsrc | static python-int src tile index | PASS |
| 495 | rem29   | dsts rem-29: ranks 29-31 receive ZERO | HANG |
| 496 | selfsend| + a 32nd copy to the sender itself | PASS, numerics ok |
| 497 | pipedup | tile-major ORDER, but uniform receives (tiles 0-3 duplicated to ranks 28-31, self-copies to a trash slot) | PASS, numerics ok |
| 503 | uneven1 | + one extra copy per sender to rank 0 (63 vs 31 receives) | HANG |

Send-side counts were uniform (31) in both hanging bisects, so the
constraint is on RECEIVES. Leading mechanism hypothesis: the transport
provisions receive-side resources symmetrically across the mesh axis, and
any asymmetry (starved or overrun) back-pressures the senders into
deadlock; not confirmed at the hardware level (a diagnostic-coordinator
stall harvest is the next tool if this ever needs the silicon answer).
Corollaries: an earlier note blaming `lax.select` in the DMA device_id was
wrong (both select and arithmetic dst hang only in zero-receiver patterns),
self-sends are legal, and static src-slice lowering is fine.

## Compute/DMA overlap: it works, at coarse granularity only

Mosaic PRESERVES source-order interleaving of remote-copy starts with
matmul bundles in the generated LLO — verified on the static schedule
(`xprof get_llo_debug_string`, module `tpu_custom_call.2`): pipedup is a
strict `M,store,D,D` per-tile alternation, pipehalf is `Mx14, Dx18, Mx14,
Dx14`. So the lower-level control the source asks for is real. Device self
time of the fused kernel (entry-barriered, traced, 32 devices):

| job | variant | device us/call |
|-----|---------|----------------|
| 462 | sequential (full dot, then 31 sends) | 17.32 |
| 501 | slabdup (28 slab dots, then the 32 tile-ordered sends) | 17.35 |
| 499 | pipedup (send tile t right after slab t) | 20.03 |
| 502 | pipehalf (send tiles 0-13, then compute 14-27, then send) | **16.89** |

Slab decomposition itself is free (slabdup == sequential; the full dot
lowers to the same 28 tile `vmatmul.mubr`s anyway). Per-tile interleaving
costs +2.68 us ~= 96 ns per tile boundary — each copy-start stalls on the
just-stored slab's VMEM visibility, and at 512 B per tile there is too
little DMA to hide behind one 256-column slab. One coarse boundary instead
of 27 keeps the win and drops the stalls: pipehalf is the fastest RS
variant measured, -0.43 us vs sequential.

## The fused o_proj+AR kernel, cleaned up for integration

`src/vllm_torchtpu/kernels/collectives/fused_out_proj_all_reduce.py` is the
production form of the probe's `fullar_nb` (barrier-free, one REGULAR-
semaphore signal round per phase instead of full barriers: 28.01 us with 3
barriers in job 488 -> 21.68 us in job 491): offset-rem phase-1 sends,
owner reduce, uniform-trash phase-2 broadcast, signal rounds gating reads.
Cross-call safety is R2's static protocol: `slot` alternates per callsite,
`barrier_first` on the first callsite of every execution; the arr1 wait in
call k proves every peer completed call k-1, which with 2 slots gives
exactly the one-call slack the rotation needs. T=1 only
(`tiles = width//256 <= num_devices`).

Torch side: `src/vllm_torchtpu/distributed/fused_out_proj_all_reduce.py`,
one opaque op `fused_out_proj_ar_<width>_<k_shard>(x, w, rs, ag, slot,
barrier_first)` on the R2 sharded_jax_op/_TpMesh bridge, gated by
`TPU_FUSED_OUT_PROJ_ALLREDUCE` (default off), T!=1 falls back to matmul +
ring all-reduce inside the op. CPU tests:
`tests/distributed/test_fused_out_proj_all_reduce.py` (7 tests: routing,
fake shapes, workspace shapes, kernel validation, flag-off). On-device
numerics/slot-rotation check: `scripts/k3/fused_out_proj_ar_check.sbatch`.
Job 505 (32 devices): 8 alternating-slot calls x 2 executions (the second
run exercises the execution-boundary barrier_first), every call within
1.7e-2 of the on-device fp32 reference, PASS on all 4 hosts. The
after_optimizations dump shows the JAX body is a bare custom call per
callsite -- result `bf16[28,2,128]{2,1,0:T(2,128)(2,1)S(1)}` on VMEM, one
hoisted w staging copy shared by all 8 calls, and only the small packed->
[1,7168] reshape per call (the two layouts tile differently; a real but
tiny relayout). No slice/pad/DUS glue.

Known integration gaps (deliberately not done): K3's o_proj weights are
compressed-tensors quantized — the kernel wants bf16 `[width, k_shard]`
shards, so wiring needs in-kernel dequant or a bf16 weight copy; the w
HBM->VMEM staging (~4.6 us/call) is still outside the kernel; and the win
vs XLA's matmul + combiner-fused AR (Batch O: the AR is ~free when fused)
is unproven — the fused kernel at 21.68 us probe time is not obviously
ahead before those overheads are attacked.

---

# VMEM chaining and w prefetch (probe: scripts/k3/vmem_chain_probe.py)

Question: can the o_proj weight staging be prefetched -- e.g. issued before
the attention kernel so w is already in VMEM when o_proj runs? Single-device
per rank, no collectives; all variants numerics-checked (jobs 508-515).

**Finding 0: the staging is a transpose, not a copy.** w arrives in HBM as
`{0,1:T(8,128)(2,1)}` (dim-0 minor) while the kernel's VMEM operand is
`{1,0:T(8,128)(2,1)S(1)}` -- the ~4.55 us staging copy is a dimension-order
transpose, which is also why XLA emits it as a blocking `copy` while the
small same-layout rs/ag workspace operands get async copy-start/copy-done
pairs. This answers the earlier bandwidth question: the low ~0.7 TB/s was
the retile, not the 3.2 MiB size -- the pure same-layout DMA runs ~2.06 us
(~1.55 TB/s, right for the size).

**Finding 1: VMEM out_spec chaining works.** A pallas kernel's VMEM
out-spec result feeds the next kernel's VMEM in-spec directly -- no HBM
roundtrip, no retile -- as long as the tilings match (they did:
`{1,0:T(8,128)(2,1)S(1)}` on both sides, read off the after_optimizations
dump). But chaining does not make staging cheaper by itself: the stage
kernel's own HBM operand still pays the transposing copy.

**Finding 2: pin the HBM layout, get a pure async DMA.** device_put with
`jax.experimental.layout.Layout(major_to_minor=(0,1), tiling=((8,128),(2,1)))`
(careful: XLA layout strings are minor-to-major, so the kernel's {1,0} is
major_to_minor=(0,1) -- pinning (1,0) is a no-op that matches the layout w
already had, measured) makes the w parameter `{1,0:T(8,128)(2,1)}` and the
staging becomes a same-layout copy-start/copy-done: ~2.06 us, async, and
hoistable. Two gotchas measured along the way: the pin must be on an
ARGUMENT (closure-captured arrays become XLA constants whose layouts XLA
re-assigns), and jaxprs DCE a discarded stand-in result (return it).

**Finding 3: the scheduler prefetches only along the dependency shape.**
With an INDEPENDENT intermediate kernel C (traced before or after, pallas
custom call or plain XLA fusion), XLA sinks C below the consumer and never
hoists the w staging above it -- the staging stays adjacent to its
consumer. But with the PRODUCTION shape -- C (attention) feeds the o_proj
consumer -- the latency hider hoists the async copy-start to before C and
the transfer hides completely: measured copy-start at t=-40.2 us, 40.1 us
tanh fusion runs, copy-done at t=0 with the consumer (job 515,
prelaid_dep). So "prefetch w during attention" works for free via the
layout pin alone -- no custom prefetch kernel needed -- precisely because
attention feeds o_proj.

Production note: the pin must survive the torch->jax weight handoff; in
JAX-land it is one device_put at load time per weight.

## Follow-up: why doesn't the hoist happen e2e, and what are the explicit levers?

Probe matrix now complete (jobs 515/521/529): with a same-layout async
staging copy, XLA's latency hider hoists the copy-start above the covering
compute whenever the consumer DEPENDS on that compute -- for XLA fusions
(515) and pallas custom calls (521) alike. With an INDEPENDENT intermediate
kernel the staging stays adjacent to its consumer (the scheduler runs the
consumer's chain first). The `xla_tpu_cmem_min/max_*overlap*_ratio` and
`max_outstanding_prefetches` flags (accepted via LIBTPU_INIT_ARGS, not
XLA_FLAGS) do NOT change the placement (job 528). A two-layer probe
degenerated: XLA fused both stand-ins into one fusion and hoisted both
stagings above it (529) -- small modules behave too well to reproduce the
e2e failure, so the e2e blocker needs the real module's dump.

What the e2e trace already shows (job 409 decode profile): NO async
copy-start exists for any [*, 224] o_proj-shaped weight at all -- the only
weight-ish async staging is one cross_program_prefetch of a [192,7168]
param. So in the e2e arm the o_proj w staging is either a blocking
(transposing) copy or streamed inside the dot fusion -- i.e. the layout pin
has not reached the torch weight-load path, and without the async form
there is nothing to hoist. If a different e2e config does show an async
copy-start placed late, the remaining suspects are scoped-VMEM pressure and
scheduler distance in the 93-layer module; that needs the
after_optimizations dump of the real step module to resolve.

Explicit levers, in increasing order of machinery (all verified to exist in
libtpu 0.0.44.1/0.0.46):

1. `xla.experimental.layout.Layout` pin on the weight buffer (measured
   here: staging becomes a pure async DMA, 4.55 -> 2.06 us, hoistable).
2. Cross-program prefetch: mark the weight parameter `S(1)` in
   entry_computation_layout (libtpu flags xla_enable_cross_program_prefetch,
   xla_msa_max_cross_program_prefetches, xla_{g,v}f_vmem_enable_cross_program
   _prefetch family). MSA hoists a copy-start to schedule time -1 and keeps
   the buffer in VMEM ACROSS steps. VMEM capacity caps this at a handful of
   3.2 MiB weights -- not all 93 layers.
   (libtpu also has a custom_call_prefetch pass producing async custom-call
   start/done pairs, but its user annotation spelling is internal.)
3. Scheduler-proof: in-kernel prefetch -- a long kernel (attention) issues
   the async HBM->VMEM DMA for the next layer's w itself and waits at its
   own end. pallas->pallas VMEM chaining is proven glueless (job 511/512),
   but the buffer must stay inside one XLA module: opaque-op (torch)
   boundaries force HBM residency, so this requires attention and o_proj in
   the same exported fn.

## In-kernel prefetch probe (jobs 530-534)

The pattern the name suggests -- the attention kernel starts an async
HBM->VMEM DMA of the next layer's w at entry and waits at exit, o_proj
consumes the VMEM buffer -- compiles and is numerically exact (rel err 0,
jobs 530/534), and the chaining is glueless (`_lambda_.2#1` VMEM result
feeds the o_proj custom call directly, no HBM roundtrip). The in-kernel
copy fully overlaps the producer's compute: the 48-pass stand-in grew
8.55 -> 8.80 us while carrying a 3.2 MiB copy (+0.25 us exposed).

BUT: the in-kernel DMA turned out to be VMEM->VMEM, not HBM->VMEM. XLA
stages every whole-tensor custom-call operand to VMEM itself, for both
pl.ANY and pltpu.HBM BlockSpec memory spaces (read off the after_
optimizations dumps): the kernel's w operand arrives as a staged S(1)
buffer. So the grid-free form cannot bypass XLA's staging; the HBM->VMEM
transfer stays XLA-managed (async + hoistable when the layout is pinned,
but scheduler-placed). Raw HBM refs appear to require the gridded/blocked
BlockSpec form; tested separately (job 536).

Also measured: unpinned w's staging splits into async pure DMA ({0,1} ->
{0,1}S(1)) + a separate VMEM retile to {1,0} (2.14 us); pinning removes
the retile.

## In-kernel prefetch, continued (jobs 534-537): how HBM operands actually bind

Deeper decode of the job-534/539/540 artifacts (Mosaic body + buffer
assignment + traces) resolved the operand-staging question -- and corrected
an earlier version of this section, which wrongly claimed the staging was
dead. The accurate model:

- **The kernel's ref type and the bound buffer are different things.** A
  DMA-only operand gets an `#tpu.memory_space<hbm>`-typed ref in the Mosaic
  body, but the buffer the kernel ACTUALLY receives is whatever XLA's
  operand contract produces -- job 539 proves the binding follows the HLO
  operand, not the ref type: with an HBM-colored operand and an unpinned
  ({0,1}) source, libtpu retiled HBM->VMEM->HBM and the kernel read the
  RETILED HBM buffer (numerics exact; reading the {0,1} original with the
  {1,0}-expecting ref would have failed). So an "HBM ref" is a DMA
  descriptor role, not a promise about physical placement.
- **Operand colors come from the argument avals, not the BlockSpec.**
  Without a mark (job 534), the operand is VMEM-colored and libtpu stages
  (async DMA + retile, ~4.2 us/call for 3.2 MiB, kernel then DMAs
  VMEM->VMEM). The ONLY way to a raw-HBM operand is
  pltpu.with_memory_space_constraint(w, pltpu.HBM) + a matching physical
  layout (job 540: zero staging; kernel's DMA is the sole HBM->VMEM
  transfer).
- **Kernel exit does NOT drain outstanding DMAs.** attn_with_prefetch
  without the pf.wait() produces garbage (rel err 0.987, job 537) -- the
  consumer raced the in-flight DMA. So the prefetch's overlap window is
  bounded by the producing kernel's own duration; fire-and-forget across
  kernels is impossible. (In-kernel double-buffered prefetch-into-next-
  call is likewise out: the DMA can't outlive its kernel.)
- **Gridded HBM BlockSpecs are rejected** unless the block is the whole
  array with a trivial index map ("The Pallas TPU lowering currently
  supports in memory space ANY only blocks having the same block shape as
  the array shape and a trivial index_map", jax 0.11.1) -- so per-block
  manual HBM reads are not available; per-block streaming is what
  VMEM-target BlockSpecs (Mosaic auto-pipelining) are for.

Net for the production question (prefetch layer L+1's o_proj w under layer
L's attention): the explicit in-kernel pattern works and is waste-free when
the operand is HBM-marked AND layout-pinned (job 540); the scheduler-managed
form (pin + dependency shape) achieves the same overlap for free when XLA
cooperates (jobs 515/521). The e2e choice between them hinges on why the
real step module doesn't hoist (needs its after_optimizations dump), or on
whether the weights can be pinned through the torch load path.

## The operand-staging question, resolved (jobs 534/539/540)

Following the layers down (full draft bug report:
docs/pallas_hbm_operand_staging.md):

- Pallas emits the custom call with the raw HBM parameter as the operand --
  no staging (before_optimizations). The vestigial staging is inserted by
  LIBTPU's tpu_custom_call operand handling, driven by the operand
  memory-space colors.
- The colors come from the input ARGUMENT avals' memory_space
  (_resolve_memory_spaces in pallas_call_registration.py), NOT from the
  BlockSpec. BlockSpec(memory_space=pltpu.HBM) sets the kernel-side ref
  type only.
- The working knob: pltpu.with_memory_space_constraint(w, pltpu.HBM) on the
  argument. Unpinned: libtpu keeps the operand HBM-resident but retiles via
  an HBM->VMEM->HBM roundtrip to meet the {1,0} layout constraint. Pinned
  (Layout(major_to_minor=(0,1), tiling=((8,128),(2,1)))): ZERO staging --
  the custom call takes the raw HBM param and the kernel owns the
  HBM->VMEM DMA entirely. Numerics exact everywhere.
- So the explicit in-kernel prefetch pattern is fully viable TODAY: pin the
  weight layout at load time, mark the operand HBM, DMA at kernel entry,
  wait before exit, chain the VMEM result to the consumer.

---

# Batch X — fused o_proj+AR with KDA-kernel weight prefetch (T=1 e2e)

Wires the measured pieces into the K3 model for T=1 decode, behind
TPU_FUSED_OUT_PROJ_ALLREDUCE (default off; own XLA cache tag
`-fused-oproj`):

- `kernels/kimi_k3/decode_kda.py::decode_kda_wprefetch` -- the decode kernel
  plus an o_proj weight prefetch: async HBM->VMEM DMA of the (HBM-marked,
  layout-pinned) weight at kernel entry, waited at exit, VMEM output `wv`.
  Numerics bitwise-identical to decode_kda and wv == w_o (job 542).
- `layers/vllm/custom_ops/kda_attention_op.py` -- opt-in `w_prefetch=True`
  op variant (15th operand in, `wv` 4th out; separate op name
  `..._wpf_...`; production op byte-identical). pallas.jax_op requires the
  return annotation -- a MISSING one fails `_verify_signature`
  (`Signature.empty is not None`), measured e2e in job 545's boot failure.
- `models/vllm/kimi_k3/attention.py` -- KimiDeltaAttention arms the fused op
  (runner: initialize_fused_oproj + ordinal in modules() order, mirroring
  initialize_llar), pins the weight layout via torch_tpu LayoutContext
  (minor_to_major=[1,0], tiles=[[8,128],[2,1]]), and routes T=1 through
  `fused_out_proj_ar` on (x=[1,384], wv); other payloads fall back inside
  the op. Requires VLLM_TORCHTPU_KDA_FUSED=0 (the fused KDA core takes no
  extra operands) and COMPILE_SIZES to include bucket 1 -- the launcher's
  default 2,16,512 pads c=1 decode to T=2, where the op deliberately falls
  back.
- Chain check (job 544, 32 devices, real K3 KDA shape [1,384]/[7168,384]):
  8 alternating-slot calls x 2 executions, rel err <= 2.6e-2 vs fp32
  reference; dump shows the fused op's w operand as a bare
  get-tuple-element of the producer custom call's result -- no staging
  copies anywhere on the w path.

## E2E at T=1 (jobs 550 control / 558+563 arm): works, and measured slower

Arm: TPU_FUSED_OUT_PROJ_ALLREDUCE=1 VLLM_TORCHTPU_KDA_FUSED=0
COMPILE_SIZES=1,512 (bucket 1 is REQUIRED -- the launcher default
2,16,512 pads c=1 decode to T=2, where the op deliberately falls back).
Both arms ran the unfused KDA path; control had the flag off.

- Boot/arming: "Initialized the fused o_proj+all-reduce op for 69 KDA
  layers" on all 32 workers; MLA layers unchanged.
- Correctness: byte-identical canary output vs control (dummy weights, so
  identical weights -> identical text is the meaningful check).
- Speed: TPOT c=1 18.38 ms vs 16.45 ms control (+1.93 ms/step, +12%).
  Per the xprof op stats the arm's fused kernel costs ~21.6 us/call x 69
  KDA layers (~1.5 ms/step) while the stock path's matmul+ring-AR is
  combiner-fused to nearly free (Batch O's finding), and the remaining
  ring ARs slowed 18.0 -> 22.6 us/op. So the fused path loses by roughly
  its own device time: eliminating the w staging bought nothing e2e because
  the stock dot fusion already streams w (the load is hidden in the matmul
  the way the job-515 dependency-shape hoist hides it, and neither
  approach beats free).
- Failure modes found and fixed during bring-up: pallas.jax_op requires a
  return annotation (Signature.empty != None; job 545 boot crash); the
  torch_tpu RowParallelLinear weight is [k_shard, width] -- contracting dim
  FIRST (job 558 warmup crash on my [width, k_shard] assumption); the
  kernel's phase-1 arrival wait must be UNCONDITIONAL -- a
  fired-but-unwatched semaphore at kernel exit is a hard runtime fault
  (tpu_program_termination_validation, job 558) that
  disable_semaphore_checks does NOT cover (it only silences the Mosaic
  compile-time check).

To make this arm win, the fused kernel's ~21.6 us has to come down toward
the RS-only floor (~11-17 us in the probe): the signal rounds and SALU
address math are the measured fat (docs' packed-probe section), and the
pipehalf half-payload overlap applies to the dot->send boundary. The
prefetch half of the story is only worth the KDA-kernel surface area once
the kernel itself wins.

## Fused-core variant (job 569): the prefetch rides the production KDA kernel

The prefetch was ported to the fused GDN v3 core (VLLM_TORCHTPU_KDA_FUSED=1,
the production default): `outer_kernel_wpf` in kernels/gdn/v3/wrapper.py
wraps `outer_kernel` with the DMA-start/pipeline/DMA-wait sandwich (the
kernel body itself is untouched), `fused_conv1d_gdn(..., w_o_prefetch=...)`
adds the operand on the first (batched-mode) pallas call, and the op
builder's `w_prefetch` flag now works on both KDA paths (the wpf op name
gets a `_wpf` suffix because vLLM's compile cache does not key on it).
On-device: 14/14 GDN v3 KDA tests pass, including a new wpf test (attention
outputs unchanged, wv bitwise-equal).

E2E at T=1 (COMPILE_SIZES=1,512, dummy weights; canary text byte-identical
to both controls):

| arm | TPOT c=1 | KDA kernel | fused op |
|-----|----------|-----------|----------|
| 377 production control (fused KDA) | 15.45 ms | 9.0 us | -- |
| 550 control (unfused KDA) | 16.45 ms | 7.3 us | -- |
| 563 arm, unfused KDA | 18.38 ms | 27.8 us (!) | ~21.6 us |
| 569 arm, fused KDA | 17.07 ms | 10.3 us (+14%) | ~20.7 us |

So the "3x slower KDA kernel" was the UNFUSED path's decode_kda carrying
the prefetch (7.3 -> 27.8 us); on the fused core the same prefetch costs
+1.3 us/layer -- the pipeline's double-buffered DMA machinery absorbs it.
The unfused kernel's pathology is plausibly DMA-engine contention (its
state-pool gathers already saturate the engine) and/or VMEM pressure from
the 5.5 MB wv buffer against vmem_limit_bytes; not investigated further
since the fused core is the production path.

The remaining regression vs 377 (+1.62 ms) is again dominated by the fused
op's own ~20.7 us x 69 layers ~= 1.43 ms/step -- the kernel's cost, not
the prefetch or the plumbing. All roads point to the same next step: the
fused kernel has to get from ~21 us down toward the RS-only floor
(~11-17 us) before any e2e win is possible; attack list per the probe
measurements: signal rounds, SALU address math, pipehalf-style dot/send
overlap, and the T=1 protocol specialization (31-copy fan-out at 32
devices is uniform-trash heavy; 28 owners x 4 ownerless).

## Fused-kernel cost attack (jobs 570-581): nosig protocol wins, but the bar moved

Microbench: `scripts/k3/fused_oproj_bench.py` (32 devices, K3 KDA T=1 shape
x[1,384] x w[384,7168], 8-call alternating-slot chain + per-op traces;
BENCH_VARIANTS selects). Device time per call, measured:

| variant | us/call |
|---------|---------|
| XLA dot + ring psum (the bar) | 22.6 |
| baseline (shipped, signal rounds) | 21.4 |
| oneshot (all-gather partials) | 31.4 -- 27x egress loses |
| directout / pipehalf / treereduce | 21.4 / 21.1 / 21.4 (noise) |
| **nosig (no signal rounds)** | **15.6** |

Phase isolation: dot+store 0.9 us; entry barrier 6.0 us; phase-1 RS ~13 us;
phase-2 AG ~7.4 us. The two signal rounds are ~5.6 us of it -- and they are
removable: the sender-side copy.wait()s plus the slot rotation suffice
(the barrier-free Lamport-style protocol of
two_shot_all_reduce(steady_barrier=False), with its documented
rank-divergence caveat). `steady_signals=False` is now the shipped default;
a hard floor worth remembering: kernel exit does NOT drain DMAs, and a
fired-but-unwatched REGULAR semaphore at exit is a HAL-level fault that
disable_semaphore_checks does not cover (job 558) -- the nosig form has
neither hazard because it never fires the sems.

E2E (job 581, nosig + fused-core prefetch, T=1): TPOT 16.66 ms (vs 17.07
with signal rounds, 15.45 production control). The bench-to-e2e transfer
is exact: -5.5 us x 69 = -0.38 ms measured, -0.41 ms predicted. Canary
byte-identical to control. But note what this says: even beating XLA's
isolated dot+AR (15.6 vs 22.6 us) the arm still loses 1.2 ms/step e2e,
because in the live pipeline the ring AR's apparent per-op cost is mostly
skew absorption that any synchronized custom op re-pays, and the stock
matmul's weight streaming is already hidden. The interesting question
shifts from "is the kernel faster than the XLA AR" to "what does the XLA
AR's skew-absorption actually buy the step".

## Prefetch placement A/B (jobs 581 vs 582): free either way on the fused core

`outer_kernel_wpf_late` issues the w_o DMA AFTER the pipeline's own
transfers instead of at entry (VLLM_TORCHTPU_KDA_WPF_LATE=1; the op name
carries `_wpflate`). TPOT c=1: 16.70 vs 16.66 ms early -- identical within
noise, canary byte-identical. So at T=1 on the fused core the prefetch's
HBM read does not measurably contend with the state-pool transfers, and
the +1.3 us/layer it costs (10.3 vs 9.0 us kernel time) is the DMA engine
occupancy, not contention. Keep the early placement (it has the larger
overlap window at bigger decode batches).

## Arm summary (all T=1 decode, COMPILE_SIZES=1,512, dummy weights)

| arm | TPOT c=1 | note |
|-----|----------|------|
| 377 production control | 15.45 ms | fused KDA, stock o_proj |
| 550 unfused control | 16.45 ms | unfused KDA costs +1.0 ms by itself |
| 563 unfused arm | 18.38 ms | prefetch breaks the small decode kernel |
| 569 fused arm, signal rounds | 17.07 ms | first working fused-core arm |
| 581 fused arm, nosig | 16.66 ms | nosig transfers exactly (-0.41 ms) |
| 582 = 581 + wpf_late | 16.70 ms | placement is free |

The residual +1.2 ms vs 377 is the fused op's device time (~15.6 us x 69)
vs a stock path whose marginal cost is ~0 (matmul streams w; the ring AR's
apparent cost is skew absorption any synchronized op re-pays). The
prefetch machinery works and is cheap; the blocker is the fused op's
device floor vs the combiner-fused status quo. Next lever if pursued:
cutting the op's two exchange phases to one at this payload (oneshot lost
on egress: 31 us), or folding the op boundary away entirely.

## The combined-layer microbenchmarks + the wv memory-space saga (jobs 583-591)

Iteration harness (all ~90s on a 4-node slice):
`scripts/k3/fused_oproj_bench.py` (fused-op variants, 8-call chain, per-op
traces; probe venv), `scripts/k3/kda_wpf_llo_bench.py` (single-kernel KDA
variants stock/wpf/wpf_late/wpf_pinned, LLO traceable), and
`scripts/k3/kda_fused_chain_check.py` (the REAL fused KDA core with the
prefetch + the real fused op under shard_map, in-body psum reference per
call, slot rotation x 2 executions).

The KDA-kernel slowdown, resolved:
- The "~3x slower" was the UNFUSED decode_kda carrying the prefetch:
  7.3 -> 27.8 us. On the FUSED core it is +1.3 us (9.0 -> 10.3).
- The in-kernel DMA cost is layout-driven: from an untiled/mismatched HBM
  source the 5.5 MB transfer runs at ~0.4 TB/s (~13.7 us); with the pinned
  layout it is a linear ~3.5 us that the pipeline mostly hides (the +1.3).
- BUT the deeper finding: with the HBM mark on w_o, the wv OUTPUT loses
  its VMEM color (the marked input + the state aliasing poison the custom
  call's output colors -- measured in the dumps) and lands in HBM, so the
  prefetch self-defeats: kernel DMAs HBM->VMEM, XLA evicts wv VMEM->HBM,
  the consumer restages HBM->VMEM. With the mark DROPPED
  (VLLM_TORCHTPU_GDN_WPF_HBM_MARK=0, jobs 590/591) wv stays VMEM and
  several fused-op callsites consume it as a bare get-tuple-element of the
  KDA call -- no copies -- while w_o is staged by XLA as an async
  copy-start (same-layout when pinned, and hoistable: w_o is a step-module
  parameter with no dependencies, so the latency hider can place it as
  early as it likes).

So the recipe that actually realizes the prefetch today: pin the weight
layout at load, do NOT HBM-mark the operand, and let XLA's async staging
be the prefetch (its placement over the preceding layer's work is the
job-515/521 hoisting question); the in-kernel DMA+wait in the kernel only
helps if the color clobbering is fixed (pallas/libtpu bug report:
docs/pallas_hbm_operand_staging.md).

## The corrected e2e attribution (jobs 592/593) -- read this, not the earlier one

The earlier "+1.2 ms regression" attribution was wrong: it compared the arm
against the PRODUCTION control (377), which pads c=1 decode to T=2
(COMPILE_SIZES=2,16,512). The arm needs bucket 1 (the fused op is T=1-only
by construction), and the T=1 bucket is +1.05 ms/step SLOWER than T=2 on
its own (measured: job 592's fused-KDA, bucket-1, flag-off control is
16.50 ms vs 377's 15.45). The interesting question of why padding to T=1
underperforms T=2 per step is open (tiling/alignment effects at the
bucket's kernels, or per-step fixed overheads amortized worse).

Against the MATCHED control (592: fused KDA core, bucket 1, flag off), the
full arm (593: nosig fused op + fused-core prefetch, mark=0) is
16.65 vs 16.50 ms -- the mechanism is within ~1% of break-even at T=1:

| config | TPOT c=1 |
|--------|----------|
| 377 production (buckets 2,16,512) | 15.45 ms |
| 592 matched control (bucket 1, flag off) | 16.50 ms |
| 581 arm2 (nosig, mark=1) | 16.66 ms |
| 582 arm2 + wpf_late | 16.70 ms |
| 593 arm3 (nosig, mark=0) | 16.65 ms |
| 596 arm4 (arm3 + natural [1,7168] op result) | 16.24 ms |
| 784 arm5 (arm4 + g_proj layout pin) | 16.09 ms |
| 831 arm6 (arm5 + AttentionResidual tail fusion) | 15.86 ms |

Per-category (592 -> 593, same profiled window): all-reduce 114.3 -> 104.1
ms (-2001 o_proj ARs, the KDA layers), custom-call 54.2 -> 50.3, data
formatting 0 -> +16.1 (attribution corrected later: it is the g_proj
weight-staging relayout, killed by the layout pin in arm5 -- see "The
'data formatting' term" below), conv fusion -2.7, loop fusion -12.9, IDLE
-398. Canary text byte-identical to control on every arm.

So: the user's challenge was right -- the control's ring ARs are NOT free
(they cost real device time), and the arm's device work genuinely dropped;
the remaining ~1% gap is the fused op's 15.6 us vs the stock matmul's ~4
us per KDA layer, plus the w_o staging copies, minus the AR removal. The
nosig protocol + the chain are correct and cheap; the bucket economics and
the per-layer staging placement are the remaining levers.

## The "data formatting" term (copy.823/copy.898): Batch W's contagion, not our data path

Tracked down in the arm3/arm4 traces: the copy is
`bf16[7168,384]{1,0:T(8,128)(2,1)S(1)} -> {0,1:T(16,128)(2,1)S(1)}`.
(Earlier versions of this note said MoE weight-shard, then q/k projection
-- both wrong. The definitive trace, from the full HLO module text of both
arms plus the ENTRY parameter block: all 69 copies are produced by
`bitcast(custom-call.NNN)` where the custom-call is `ConcatBitcast`
reassembling the two async HBM->VMEM staging slices of ONE standalone
parameter (s8[4194304] + s8[1310720] = exactly 7168*384 bf16), and ALL 69
feed `bitcast -> bf16[3,128,7168]{2,1,0}` -> a kOutput `fusion.*` whose
ROOT is `convolution(...) window={size=3 pad=2_2 rhs_reversal=1}
dim_labels=fb0_0oi->b0f` (a conv-formulated matmul contracting over 7168)
with the rmsnorm/rsqrt epilogue. The copied parameter sits in the per-KDA-
layer param block between fused_qkvb [7168,1155], f_a [7168,128] and f_b
[128,384]: it is `g_proj.weight`, the full-rank gate projection
([hidden=7168 -> 3 heads x 128 per rank]; 96 heads / 32-way TP). q/k/v
come from the merged [7168,1155] parameter, which is already {0,1} and
stages clean; w_o is [384,7168]. NOT MoE and NOT the wv/o_proj path.) It
is the Batch W contagion term ("[7168,384] weight-staging transpose
copies, +0.31 ms/step"): an exotic-layout value crossing our custom-call
boundary shifts XLA's global layout assignment and the staging transpose
materializes.

Where it appears is chaotic evidence of the mechanism: 569 (signal-rounds
kernel) had NONE; 581/593 (nosig kernel) had +16.1 ms/window; 596 (nosig +
natural [1,7168] op result) has +14.2. And TPOT anticorrelates with its
presence (569 worst at 17.07 ms; 596 best at 16.24) -- the copies are
absorbed device work, OFF the critical path. The natural-output change is
still worth keeping (it removes the 4-D packed custom-call result, the
most exotic value), but the remaining exotic inputs are the rs/ag
workspaces ([2,32,2,128]/[2,29,2,128] 4-D T(2,128)(2,1), VMEM-staged per
call) and the wv buffer.

### Measured size

From the rank-0 hlo_stats of the 593/596 decode traces
(`results/foa-arm{3,4}/hlo-decode_only.json`; both windows are exactly 31
decode steps, 2139 KDA-kernel occurrences / 69 layers):

| | arm3 (593) | arm4 (596, natural output) | arm5 (784, +g_proj pin) |
|---|---|---|---|
| data-formatting category, window total | 26.2 ms | 23.7 ms | 7.97 ms |
| extra vs matched control (592) | +16.1 ms | +14.2 ms | ~0 |
| **extra per step** | **~520 us** | **~458 us** | **~0** |
| control's own baseline | ~305 us/step | ~305 us/step | ~305 us/step |

The extra is exactly 69 relayout copies per step -- one per KDA layer --
at ~7.3 us each (15.14 ms / 31 steps in arm4). The natural-output change
cut it ~12%; the g_proj layout pin (path B below) removed ALL of it
(arm5: zero such copies, and the category total dropped below the
control's own baseline because the baseline figure included unrelated
noise). In arm3/arm4 all of it landed in device IDLE bubbles (IDLE
dropped -398 ms in the same window) -- off the critical path at c=1, but
its removal still improved TPOT 16.24 -> 16.09 ms.

### Paths to remove it

**(A) 3-D-ify the rs/ag workspaces -- NOT NEEDED.** Path B (below) removed
the entire term, so the workspaces' exotic layouts turned out not to carry
the contagion (or at least not to matter once g_proj is pinned). Kept here
for the record. The hypothesis was: exotic layouts crossing the
custom-call boundary perturb global layout assignment, and after the
natural-output change the workspaces ([2,32,2,128]/[2,29,2,128] 4-D
T(2,128)(2,1)) are the most exotic values left. Re-express them as
[2,32,256]/[2,29,256] 3-D (same physical bytes) -- or as flat buffers
with a boring declared layout, reinterpreted inside the kernel. Cost: the
kernel's DMA payload packing is built on [2,128] tile units, so the
indexing changes, and the uniform-receive protocol constraint (exactly
uniform per-rank receives or the exchange deadlocks, jobs 495/503) must be
re-validated. The flat-reinterpret variant also risks the Mosaic
3-D-shape/ref-view refusals that blocked Batch R2.

**(B) Pin the g_proj weight's layout -- DONE, works (job 784,
foa-arm5-gpin).** The copied tensor is `g_proj.weight`, a standalone XLA
parameter (bf16[7168,384] per KDA layer), so it is directly pinnable --
the `ConcatBitcast` just reassembles the async staging slices of that
parameter and takes whatever layout the parameter has. Implementation:
`pin_kda_gate_weight_layout` in `runner/utils.py` (called from
`load_model` under `TPU_FUSED_OUT_PROJ_ALLREDUCE`) re-materializes each
KDA `g_proj.weight` via a host round trip inside
`LayoutContext(TpuLayout(minor_to_major=[0,1], tiles=[[16,128],[2,1]]))`
-- LayoutContext hooks host->device transfers only -- and registers
(bf16, [7168,384]) -> [0,1] in `_PINNED_ENTRY_LAYOUTS` so the compiled
entry matches. No shape collision: w_o is [384,7168] and q/k/v are merged
into [7168,1155]. Result on the decode trace: the 69 relayout copies are
GONE (2139 -> 0 occurrences, -15.14 ms/window), nothing new grew anywhere
in the category, data formatting fell 23.7 -> 7.97 ms/window (763 -> 257
us/step, below the matched control's ~305 baseline), canary
byte-identical, TPOT 16.09 ms (vs arm4 16.24, matched control 16.50) --
the fastest fused-op arm. The "anticorrelation" caveat from the arm3/arm4
analysis did not survive contact: removing the off-critical-path copies
still bought ~0.15 ms/step, presumably via reduced HBM bandwidth
contention.

**(C) Accept it -- MOOT, the term is gone.** Superseded by (B). The
original justification: off the critical path, absorbed in IDLE,
anticorrelates with TPOT across arms. Even so, removal bought ~0.15
ms/step of TPOT (16.24 -> 16.09), so "free" was overstating it.

## Fusing the post-AR epilogue (residual add + RMSNorm) into the op -- job 787

What sits between the fused op's AR result and the next layer's
projections in the compiled graph (arm5 trace): one multiply_reduce
kLoop fusion (residual add + f32 sumsq, ~1.6 us) plus a scalar
mul/add/rsqrt chain; the gamma scale is already inlined into the
projection fusions, which consume (hidden, rsqrt) -- so the op only needs
to output those two to be a drop-in replacement.

Microbench (fused_oproj_bench.py, job 787, device us/call from rank0
traces; wall-clock is dispatch-bound at ~50-70 and only relative):

| variant | device us/call |
|---------|----------------|
| baseline (shipped nosig op) | 15.6-15.8 |
| baselinexlaep (op + XLA epilogue, today's model path) | op 15.6 + epilogue ~2.0 (multiply_reduce 1.79 + scalar chain) |
| tailfuse (epilogue at assembly) | 16.3-16.6 |
| tailowner (residual add moved into the owner-reduce) | 16.3-16.6 |

Numerics: all variants pass (max rel err 0.0148, same as baseline).

Readings:

- The in-kernel epilogue costs +0.6-0.8 us/call vs +2.0 us/call for the
  XLA path -- a ~1.3 us/call device-time win, plus two op boundaries off
  the serial chain (the e2e-relevant part at T=1; bench wall-clock
  71.5 -> 70.6 understates it because bench dispatch dominates).
- tailowner == tailfuse: moving the residual add into the owner-reduce is
  not measurable at this size -- the [2,128] add is noise either way.
- The "overlap the norm partials with the phase-2 sends" variant is NOT
  worth building: the epilogue is only ~0.7 us in-kernel, and any
  transport for the 28 f32 partials (31 extra tiny sends, or a widened
  payload with bit-split f32-in-bf16) costs more than that. Per-tile
  arrival processing is blocked by arrival-order opacity under nosig.
  Overlap does not pay at this payload size; the straight tail fusion is
  the right design.

### What the ops after the multiply_reduce fusion are (K3 AttentionResidual)

K3 sets attn_res_block_size=12: every decoder layer runs TWO
AttentionResidual modules (self_attention_res before input_layernorm,
mlp_res after attention; models/vllm/kimi_k3/layers.py:85). Reading
fused_computation.3622 (the multiply_reduce_fusion body that consumes the
fused op's output): the fusion computes hidden = AR_out + residual, then
TWO reduces over it -- sumsq(hidden) (the "sum"+"mean" the trace shows)
and score_dot = sum(hidden * w_folded) where w_folded = norm.weight *
proj.weight (AttentionResidual's folded [7168] scorer) -- plus the scalar
chain (x(1/7168), +eps, rsqrt -> inv_rms) and mul = score_dot * inv_rms
(the "mul"). The "2 softmax ops" are maximum() scalars (XLA names them
softmax.*): the max-subtract of the tiny softmax over the [1, K+1] block
scores that weights the per-block residual snapshots against the current
prefix. The [1,2] pad/pack and broadcast_divide fusions are the rest of
that softmax.

Measured size of the whole small-op region (arm5 decode trace, per step
over 93 layers): multiply_reduce 729 us + score pad/pack 86 +
broadcast_divide 136 + bitcast glue 113 ~= 1.06 ms/step of device busy
time, PLUS the per-op boundary latency that matters more at T=1 (~10 such
ops per layer).

Fusibility: sumsq, score_dot, and the residual add all depend ONLY on
(AR_out, residual, folded_weight) -- all three ride the op's tail. The
block-side stats (block_residuals) do not depend on the AR and can overlap
it; the score softmax + weighted sum must stay in XLA. Boundary condition
to respect at integration: XLA currently reassociates the residual adds
ACROSS the layer boundary (the fusion consuming our AR also consumes the
previous layer's MoE/shared-expert ARs); folding the add into the opaque
op removes that freedom -- value-identical per element, but bf16 add order
changes, so the canary must be re-verified.

Job 790 bench (adds the score-dot; BENCH_VARIANTS=baseline,
baselinexlaep2,tailfuse,tailscore; device us/call, rank0 traces):

| variant | device us/call |
|---------|----------------|
| baseline | 15.63 |
| baselinexlaep2 (XLA epilogue incl. score-dot) | 15.63 + 2.03 (multiply_reduce) + scalar chain |
| tailfuse (resid+sumsq+rsqrt in-kernel) | 16.38 |
| tailscore (+ score-dot and score*inv in-kernel) | 16.39 |

The second reduce is FREE (16.39 vs 16.38 -- idle VPU cycles during the
epilogue). Net: +0.76 us/call in-kernel replaces >=2.0 us of XLA fusions
plus their scalar chain and ~3 op boundaries per KDA layer. (The stray
0.9 us `reduce` in the tail-variant traces is a bench artifact -- XLA
lowering the callsite's resid reshape -- not the design.)

Integration plan (next): the shipped op gains two replicated inputs --
the residual ([1,width]) and the mlp_res folded scorer weight (f32
[width]) -- and three outputs: hidden = AR + residual [1,width], the
inv_rms rsqrt scalar, and the folded AttentionResidual prefix score; the
model's `residual + hidden` and the mlp_res prefix-branch reduces are
bypassed on the fused path. The post_attention_layernorm itself stays in
XLA (it normalizes the mlp_res weighted-sum OUTPUT, which depends on the
block scores). Canary risk: the sumsq/score reduction orders differ from
XLA's (per-tile in-kernel vs XLA's tree) and the residual-add
reassociation freedom is removed -- f32/bf16-ulp-level differences; arm
canaries have tolerated the bf16 reduce-order change, so likely fine,
verify on device.

### Integration (arm6) + the score-check false alarm (jobs 791-802)

Integrated: the shipped op takes (x, w, resid, fw, rs, ag) and returns
(hidden, inv_rms, score); the model layer (KimiDecoderLayer) passes the
prefix_sum residual and mlp_res.folded_weight in, skips its own residual
add, and calls mlp_res.forward_with_prefix_stats (blocks stats + softmax +
weighted sum stay in XLA). Block-boundary layers (prefix_sum is None)
pass a zeros residual, reproducing `prefix_sum = attn_out` exactly (bf16
x + 0 == x). The op's T!=1 fallback computes the same triple in torch.

Jobs 791-794 chased a deterministic ~4.3x-wrong score at barrier_first=1
calls: hidden and inv_rms exact, score wrong, and ONLY on call 0. Read-fw-
early (794), collective_id 8 vs 0 (792), and an output-slot swap (796) all
changed nothing; a bare sum(fw) probe kernel with the entry barrier was
clean (795). Job 800's mode bisect then showed mode=0 failing with AND
without the barrier -- i.e. not barrier-related at all. Root cause: call 0
is the only unperturbed input (x + 0), and its score-dot reference happens
to land near zero (want 0.093 with ~85-magnitude terms); the dot is
cancellation-heavy, so bf16-level noise in hidden gives ~0.3 ABSOLUTE
error -> rel err 3.31. The other calls' wants are ~50-90, so they pass.
No kernel bug. Lesson: relative-error checks on cancellation-heavy scalars
need an absolute floor sized to the dot's magnitude scale; both
fused_out_proj_ar_check.py and fused_oproj_bench.py now floor the score
tolerance at 1% of sqrt(sum((hid*fw)^2)).

arm6 e2e (job 831, canary byte-identical): TPOT 16.09 -> 15.86 ms (-0.23
ms/step). Device time barely moves (multiply_reduce -27 us/step, exactly
one fusion per KDA layer removed: 467 -> 398/step; fused op unchanged at
16.16 us/call -- the in-kernel epilogue hides under e2e contention); the
win is the dependency-chain shortening, as predicted. g_proj relayouts
still zero. Two integration bugs worth remembering: torch's infer_schema
rejects a bare `-> tuple` return annotation on the sharded_jax_op fn
(subscripted tuple[...] is fine), and the no-metadata warmup path must
honor the 3-tuple contract when the layer is armed. Remaining gap to
production (15.45) is the T=1-vs-T=2 bucket cost.
