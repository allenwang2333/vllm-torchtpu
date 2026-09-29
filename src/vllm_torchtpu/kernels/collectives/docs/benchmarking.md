# Collective validation and benchmarking

Measure the kernel on the current repository runtime and actual target devices.
Record JAX/JAXlib/libtpu, Torch/TorchTPU versions, source identities, compiler flags,
physical inventory, logical rank order and the adapter being measured. Standalone
JAX and a Torch bridge are separate measurement cohorts.

## Entrypoints

Run the repository's [benchmark runner](../../../../../scripts/collectives/two_shot_benchmark.py)
from the repository root. The launcher must supply the matching distributed
rendezvous and worker environment. For an initialized allocation, the JAX cohort
uses this command on every participating host:

```bash
python scripts/collectives/two_shot_benchmark.py run \
    --framework jax --distributed --output /path/to/run
```

The Torch cohort uses `--framework torch` in the actual Torch distributed worker
context. Keep its native/candidate wrappers and rank mapping identical to serving.
`--smoke` and `--batches` runs are diagnostics, not complete qualification runs.
Two complete independent result files feed `qualify`; its report preserves every
accepted and rejected exact case. Use `--manifest` only for the separate Torch
automatic-dispatch check after qualification.

## Correctness

Validate every addressable output shard on every process against independent
NumPy FP64 references. Inputs must vary with rank, element and invocation so stale
buffers and wrong destinations are observable. Include exact cancellation data,
random data, both dtypes and padding boundaries. Check equality across replicas.

The CPU tests cover padding, invalid arguments, VMEM limits and offline TPU
lowering, including the distinct public/private compiler effect flags. Configure four CPU devices before starting the process to exercise the
lowering checks. The TPU tests require distributed JAX initialization before
pytest starts and the same test list on every worker of a 32-device slice.
Both compiler modes cover repeated changing inputs, odd/even invocation counts,
scratch reuse, logical-rank permutation, delayed host entry and interleaved native
collectives. The launcher owns rendezvous,
timeouts and whole-group cleanup.

## Timing

Compare native and custom implementations using the same source base, runtime,
input and compiler flags. Compile both arms and validate correctness before
measurement. Record compilation and first-call costs separately from steady state.

Use five warmups and 25 interleaved paired trials of 20- and 60-operation dependent
chains. A full-output dependency and completion wait keep every collective
observable. Alternate native/custom and low/high ordering. Aggregate the slowest
process outside the timing interval and report `(T60-T20)/40`. Retain the
per-process raw durations, signed slopes and p10/median/p90 statistics. Record a
dependency-only control without subtracting it from collective measurements.

Audit optimized HLO from the actual timed executable. Verify the expected custom
call/native collective counts and reject folded-away communication or retracing.
For the private uniform SPMD path, compare the optimized collective call/dependency
graph across every rank and require genuine previous-output dependency edges;
matching counts alone is insufficient. Audit the actual adapter used by serving,
not only the public standalone JAX entry.
Missing backend placement annotations do not identify a TensorCore or SparseCore
execution engine.

## Selection and reporting

Use two independent launches with the same declared protocol. An exact custom
case must pass correctness and compiler audits, reduce median latency by more
than 5%, save more than 0.5 microseconds, and place its p90 below native p10 in
both launches. Uncertain, unsupported and unmeasured cases retain native.
Validate published automatic selections in a separate execution without using
that run to choose winners. No broad shape range follows from nearby exact wins.

Retain losses, rejected cases, startup failures and negative slope estimates.
If timing noise requires longer chains, declare and rerun the complete affected
cohort; preserve the earlier results. Save large HLO dumps and raw runs in artifact
storage. Include compact results, reproducible commands, all source/runtime
identities and limitations in the PR report. Historical measurements do not
qualify this port. End-to-end serving and model accuracy require separate checks.
