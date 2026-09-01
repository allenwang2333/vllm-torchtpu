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
