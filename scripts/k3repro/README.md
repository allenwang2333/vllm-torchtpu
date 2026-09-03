# Reproducing the Kimi-K3 TP32 serving numbers

`repro_k3_tp32.sh` is self-contained: the serve command, the environment, the
benchmark sweep and the summary. It does not use Buildkite or the in-repo
benchmark harness, so it can be run on any 32-device v7x pod, including a
slurm cluster.

## Run it

On the Ray **head** host of an already-formed 32-device cluster, inside the
serving container:

    bash scripts/k3repro/repro_k3_tp32.sh

Overridable: `MODEL_URI`, `CONCURRENCIES`, `REPS`, `ISL`, `OSL`, `OUT`, `PORT`.

First boot streams the 1561 GB checkpoint and takes roughly 25 minutes. The
script prints the root cause and exits if the server dies.

## What it measured

`[Kimi-K3 full checkpoint, TP32+EP32, v7x-32 = 16 chips x 2 cores over 4
hosts, real weights, ISL/OSL 8192/1024, greedy]`, three repeats per cell
agreeing to under 0.1%:

| conc | median TTFT | median TPOT | output tok/s |
|-----:|------------:|------------:|-------------:|
|    1 |   6053.3 ms |    27.98 ms |         29.5 |
|    2 |   9064.1 ms |    33.52 ms |         47.2 |
|    4 |  15126.1 ms |    35.51 ms |         62.8 |

Source: build 360 on `geyuhao/k3-perf-clean` @ `4273c1695`.

## Two things to know before comparing

**This is a colleague's configuration, not our fastest.** It runs at their
`--block-size 16`; our own A/B puts 256 at −42.6% TTFT. Our best measured
configuration on the same rig is **TTFT 1435.9 ms / TPOT 17.785 ms**, which
additionally uses page 256, a reachable MLA tuned row, async scheduling and a
decode token bucket of 1.

**`TPU_LATENT_PROJ_INTRA_CHIP_TP` is deliberately not set.** It requires
PR #448's code, which is not on main. Their configuration sets it, so this
reproduction is their config minus one lever.

## Corrected from build 360

Build 360 sent **6 prompts at every concurrency**, so its c>=8 cells were
starved and returned identical, meaningless numbers. This script sets
`num_prompts = 4 x concurrency` (minimum 8). Only the c=1, 2 and 4 rows above
come from valid cells.

## Why the cache is cleared

`LIBTPU_INIT_ARGS` is consumed after the HLO is emitted, so torch_tpu's
tier-2 cache in `/dev/shm` is not keyed on it. A warm cache silently reuses an
executable built with different flags, and the flags become a no-op. The
script removes it before serving. A round-3 A/B of four XLA flags had to be
withdrawn for exactly this reason.
