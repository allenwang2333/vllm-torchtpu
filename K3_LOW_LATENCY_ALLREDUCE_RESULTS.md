# Low-latency all-reduce kernels in the K3 forward pass

Results for `TPU_LOW_LATENCY_ALLREDUCE`, branch
`gnovack/low-latency-allreduce-k3` (the low-latency kernels of `9d7a638` plus
the integration in `9dcf7ca`).

The kernels in `kernels/collectives/low_latency_all_reduce.py` beat XLA's ring
all-reduce by 1.1-2.5x at decode payloads in JAX microbenchmarks
(`docs/low_latency_allreduce.md`). This is what happens when they are put on
K3's actual critical path.

## Why this was worth trying

All-reduce is a large, well-measured share of K3 decode. From the
latent-projection traces (`results/k3-hier-latent-v4`, rank 0, decode-only):
**79.80 ms of 257.06 ms device time, 31%**, and at `compile_sizes=[1]`
72.97 ms of 232.20 ms, still 31%. A per-collective saving lands directly on
that.

The doc's own recommendation was to confirm section 4 first -- torchtpu
appeared to add ~139 us per collective, which would dwarf a 2x win on a 9 us
op. That is now answered, twice over, and it was a microbenchmark artifact:
the K3 traces show ~34 us per TP32 all-reduce, and the integration benchmark
below measures the kernels at their JAX-level speed through the torch path.

## Getting the kernels to run at all

This was most of the work, and none of it was in the kernels. Four things had
to line up, each discovered by hitting it:

| what | why it fails | resolution |
|---|---|---|
| A mesh spanning peers | vLLM gives each worker a **single-device** JAX mesh; the kernels need `shard_map` over all 32 | `jax.devices()` exposes the whole slice inside a worker even though `jax.local_devices()` is one core, so a mesh is built over the group's TPU global device ids. **Rank order is not device-id order** (rank 2 sits on device 4), so the map is gathered |
| Skipping `shard_map` entirely | JAX refuses: `Cannot use communication in pallas_call without shard_map` | not available; the mesh form is mandatory |
| Operand partition specs | `P("tp")` makes the module entry the *global* shape and torch_tpu then refuses the rank's local tensor: `invalid refinement ... tensor<32x7168> -> tensor<1x7168>` | declare the operands **replicated**, so global == local; `shard_map` still hands the body each rank's own buffer |
| `has_side_effects=True` | XLA's partitioner rejects a side-effecting custom call inside the manual region a shard_map lowers to: `Side-effect HLO must have sharding` | isolated to exactly this flag by a three-variant probe; the kernels now take it as a parameter and the integration passes `False` |

Turning off `has_side_effects` on a *collective* deserves the suspicion it
attracts. It is safe for these kernels because the three properties the flag
protects are already guaranteed by other means: the output is consumed so
nothing can be dead-code eliminated, each call site gets its own
`collective_id` so two sites cannot be CSE'd together, and the aliased
`ws`/`step` chain orders successive calls that share a workspace. It would
not be safe for a stateless collective, and the kernel docstring says so.

A dead end worth recording: the PCP streaming attention kernel looks like a
precedent for a cross-device Pallas kernel under `shard_map`, and is not one.
Its pallas_calls are pure compute; the communication around them is ordinary
JAX collectives. Nothing in this repo had run a side-effecting Pallas
collective through torch_tpu before.

## Correctness

`scripts/debug/test_low_latency_all_reduce.py`, 8 devices, driving the
dispatcher exactly as `TpuDeviceCommunicator.all_reduce` does. Each shape runs
a **six-round chain** whose input depends on the previous output, with a
per-round twist, so a dropped, stale or duplicated round shows up as a value
mismatch instead of hiding inside an idempotent mean -- the workspace parity
is the part of the design most exposed to going wrong.

| dtype | result |
|---|---|
| float32 | **exact**: `max|diff| = 0.00000` on every shape, both kernels |
| bfloat16 | 0.0113-0.0150 relative, ~3 ulps -- summation order, confirmed by the float32 run |

Every shape also routed exactly as the gate specifies, including the ones that
must fall through to XLA.

## Latency through the torch path

Same two-point slope methodology as the microbenchmarks, but through
`torch.compile(backend="tpu")` and the dispatcher:

| payload | XLA | one-shot | speedup | JAX-level (doc) |
|--------:|----:|---------:|--------:|----------------:|
| 2 KiB | 7.87 us | **3.94 us** | **2.00x** | 2.13x |
| 4 KiB | 9.70 us | **4.94 us** | **1.96x** | 2.22x |

The win survives the bridge essentially intact. **This is the result that
retires section 4's open question**: if torchtpu really added ~139 us per
collective, a 4 us kernel and a 9 us kernel would be indistinguishable here.
They are not.

## two-shot is off by default

two-shot is *correct* -- exact against `funcol` in float32 across every shape
and chain length -- but not yet *reliable*. A 256 KiB reduction that had
passed twice took the core down on the third run with
`RuntimeUnexpectedCoreHalt`. That nondeterminism has the same shape as the
semaphore-overlap fragility in `docs/low_latency_allreduce.md` section 3a and
has not been root-caused for this call path.

It costs little to leave off here: K3 decode at TP=32 reduces 7-28 KiB, which
is one-shot's band. Set `TPU_LOW_LATENCY_ALLREDUCE_TWO_SHOT=1` to include it.

## End-to-end on K3, TP=32

**It does not translate. Decode is 5.8% slower at concurrency 1, not faster.**
Jobs 88 (treatment) and 63 (control), same flags but for
`TPU_LOW_LATENCY_ALLREDUCE`, both through `arm_check.sh`.

| conc | control TPOT | treatment TPOT | delta | control tok/s | treatment tok/s |
|-----:|-------------:|---------------:|------:|--------------:|----------------:|
| 1    | 16.80 ms | **17.77 ms** | **+5.8%** | 58.9  | 55.4  |
| 2    | 17.95 ms | 18.86 ms | +5.1% | 110.3 | 105.1 |
| 4    | 23.82 ms | 24.00 ms | +0.8% | 166.0 | 161.9 |
| 8    | 26.99 ms | 26.96 ms | -0.1% | 292.1 | 292.6 |

Correctness is not the issue: GSM8K 100/100 on both arms, and the canary text
is byte-identical between them.

Coverage was as designed -- `routed: one_shot=4 two_shot=0 xla=17`, with
`(2,3584)` at 14 KiB on the kernel and `(2,7168)` at 28 KiB correctly left on
XLA.

### Why, from the trace

`diff_arms.py`, rank 0, decode-only, identical 16-step windows:

| category | control | treatment | delta |
|---|--------:|----------:|------:|
| all-reduce   | 79.03 ms | 60.29 ms | **-18.74 ms (-23.7%)** |
| custom-call  | 32.33 ms | 70.84 ms | **+38.51 ms (+119.1%)** |
| **total device self time** | **259.92 ms** | **275.61 ms** | **+15.69 ms (+6.0%)** |

+6.0% device against +5.8% TPOT: wall clock and hardware counters agree, so
this is the mechanism and not something off-device.

The kernel really did displace XLA all-reduce work -- that category fell by a
quarter. It lost anyway, for two separate reasons, and the second is the
interesting one.

**1. The kernel costs more in production than on the bench.** The +38.51 ms of
custom-call over 1472 occurrences is **~26 us per one-shot call**, against
15.14 us for the same 16 KiB payload at 32 devices in the microbenchmark, and
against XLA's ~20 us here. Production collectives carry roughly 10 us of
per-collective overhead that the bench does not, and the Pallas kernel pays it
too -- so a kernel that wins on a 17 us baseline can lose on a 27 us one. The
microbenchmark's ratio is not transportable.

**2. XLA had already fused two collectives into one, and the kernel broke it.**
Read the per-occurrence rows:

| arm | collective | ms | occ | us/occ |
|---|---|---:|---:|---:|
| control   | all-reduce `bf16[2,3584], bf16[2,7168]` | 50.00 | 1472 | **34.0** |
| control   | all-reduce `bf16[2,7168]` | 29.03 | 1520 | 19.1 |
| treatment | all-reduce `bf16[2,7168]` | 60.29 | **2992** | 20.1 |

The control reduces the 3584-wide and 7168-wide payloads **in a single
collective** costing 34.0 us for both. Pulling the 3584 half out into a kernel
leaves the 7168 half as its own XLA all-reduce -- the occurrence count rises by
exactly 1472, from 1520 to 2992 -- so the same work now costs 20.1 us of XLA
plus ~26 us of kernel, about 46 us where it used to cost 34 us. That is
+12 us x 1472 = +18 ms, which is the regression.

This is the part no op-level benchmark could have predicted: the kernel was
measured against a *standalone* all-reduce, but the all-reduce it actually
replaces was not standalone. Displacing one member of a fused collective costs
the fusion.

### What this implies for the follow-ups

- **`compile_sizes=[1]`** would improve the kernel's own numbers (7 KiB is
  1.65x on the bench rather than 1.14x) but does nothing about the fusion,
  which is the larger term. Not worth running on its own.
- **Wiring both widths to kernels** now looks more attractive than it did, not
  less: if both halves of the fused pair go to kernels, nothing is left behind
  to need a separate XLA collective. It needs a second workspace pair, and it
  has to beat 34.0 us for the pair, which at ~26 us per kernel call is not
  obviously achievable.
- **Padding-awareness** (reducing only the valid rows, which XLA cannot do
  because its collective shape is static) remains the one idea here with a
  structural rather than constant-factor advantage, and it is the only one that
  would also shrink the 7168-wide half.

### A methodology cost worth recording

Seven K3 bring-ups were spent on this, six of them on integration bugs rather
than measurement. Five were failures that a faithful local reproduction would
have caught in ten minutes; the TP8 harness used plain `torch.compile`, while
every failure lived in vLLM's own stack. `scripts/k3/slurm_kimi_linear_tp8_hier_ep.sbatch`
with `ARMS=hier-b2` now serves as that reproduction -- Kimi-Linear-48B at TP=8,
same dynamo/AOT/piecewise path, ~10 minutes.

The sixth failure was not a code bug at all: `VLLM_XLA_CACHE_PATH` is one
persistent tree shared by every run, so the treatment loaded an executable the
*control* had compiled and died with `execution supplied 2135 arguments but
compiled program expected 2133` -- off by exactly the two workspace tensors.
A persistent executable cache is a correctness hazard for A/B work, not just a
speed optimization; the launcher now tags the path when the flag is on.

## Reproducing

```bash
# treatment
TPU_MOE_HIERARCHICAL_EP=1 TPU_LATENT_PROJ_INTRA_CHIP_TP=1 \
TPU_LOW_LATENCY_ALLREDUCE=1 \
  PROFILE_DIR=$PWD/results/<name>/traces KIMI_K3_LOG_ROOT=$PWD/results/<log> \
  sbatch --job-name=<name> scripts/k3/slurm_launch_k3_tp32_tp.sbatch
# control: same line with TPU_LOW_LATENCY_ALLREDUCE=0

ENDPOINT=<url> ARM=<name> OUT=results/<name> scripts/k3/arm_check.sh
python scripts/k3/diff_arms.py results/<control> results/<treatment>
```

The sbatch pins the flag with a default and echoes back **what the dispatcher
actually did** -- how many shapes went to each kernel, read out of the server's
own log -- not what was requested. The dispatcher falls back to XLA silently
whenever a shape sits outside the measured crossover or an op fails to build,
so "the flag was set" is not evidence that a single all-reduce changed.

The 8-device correctness and latency harness:

```bash
sbatch scripts/debug/slurm_test_ll_allreduce_tp8.sbatch --dtype float32 --skip-timing
sbatch scripts/debug/slurm_test_ll_allreduce_tp8.sbatch --rows 1,2,4,8,16
```
