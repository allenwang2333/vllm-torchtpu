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
