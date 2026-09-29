# Collective architecture

The Pallas kernel receives a shard-local array and static communication metadata.
It owns padding, remote transfers, synchronization, reduction and scratch. The
Torch adapter owns rank-to-device mapping, output placement and mutation semantics.
Model-specific payload packing and parallel layout remain in the model.

## Explicit kernel API

`two_shot_all_reduce(x, *, axis_name, axis_size, collective_id)` executes inside a
flat, single-axis `jax.shard_map`. The array shape is concrete and uniform across
participants. Supported group sizes are 2, 4, 8, 16 and 32; inputs are BF16 or FP32.
The caller must use the bound axis size and a collective ID in 0–15. Callers must route empty
arrays and one-rank operations to the native identity path.

`two_shot_unsupported_reason(shape, dtype, *, axis_size, collective_id)` checks
local contracts without initializing a device runtime. It does not certify the
physical topology, compiler/runtime support or relative performance. The kernel
raises for an unsupported explicit request. Automatic dispatch belongs above it.

## Private uniform SPMD entry

`_two_shot_all_reduce_uniform_spmd` shares the public entry's validation, memory
budget and communication protocol, but emits a compiler-optimizable custom call
(`has_side_effects=False`). It is reserved for the K3 adapter's audited uniform
SPMD program. The public API continues to emit `has_side_effects=True`.

The private caller must establish that all participants have the same live
collective program after optimization, with genuine previous-output dependencies
between in-graph invocations. Separate executions must synchronize before the
next begins. Independent or overlapping invocations are unsupported, including
across different collective IDs and native collectives. Matching Python source,
shapes and runtime versions alone does not establish this contract; qualification
compares the optimized collective call/dependency graphs across all ranks.

The result depends on remote operands, so this is not a generally pure local
operation. Rank-asymmetric elimination or common-subexpression merging can remove
one participant's communication and deadlock the others. Under the uniform
program contract, whole-group elimination or merging preserves the functional SUM
because remote writes target invocation-local scratch/output and transfers drain
before return. This compiler mode does not remove entry barriers, DMA waits or
semaphore checks. It permits the native Torch importer's repeated partitioning to
handle this constrained call without a side-effect sharding annotation.

## Serving eligibility

A serving adapter must verify the actual hardware inventory, flat group mapping,
supported runtime, shape/dtype and scratch budget before calling Pallas. A valid
input is not necessarily faster with a custom collective. Automatic selection
additionally requires a measured entry for that exact source, execution context
and input. Missing or mismatched evidence retains native execution.

The initial K3 callsite is BF16 `[B, 7168]` after KDA projection, with full-world
TP32, DP/PP/PCP/DCP all one and sequence parallelism disabled. The opt-in flag is
`TPU_K3_TWO_SHOT_ALL_REDUCE=1`; the shipped qualification manifest is empty.
Admission also requires `async_scheduling=False`, `max_num_seqs=1`, no speculative
decoding and no overlapping microbatches. The pinned Torch bridge does not
establish device-completion ordering between independent executions. Async
prefill steps can lack a previous-token dependency, and multiple requests can
create independently enqueued runner chunks even with synchronous scheduling.
The single-request synchronous path waits for sampled output before its next
execution. These guards preserve the private entry's synchronization contract
without changing the runner. Canonical asynchronous serving therefore retains
native collectives and cannot establish a Two Shot serving gain in this revision.

The initial runtime is the repository's JAX/JAXlib 0.10.2 and libtpu 0.0.47 stack.
Qualification must be performed on that stack with the actual adapter and compiler
flags. A changed stack, bridge, source or ownership mapping requires validation;
historical standalone JAX results cannot authorize a Torch path.

## Synchronization ownership

All ranks must enter the same collective sequence. An ID can be reused by ordered
invocations after the preceding invocation drains its transfers. Concurrent
protocols are unsupported. Independent sequences use different caller-allocated
IDs and must execute sequentially. The kernel has no
persistent buffers or semaphore state, and does not recover a failed worker group.
On a hardware failure or timeout, terminate the whole test or benchmark group.

Kernel implementations and their tests stay in this repository. Large traces,
compiler dumps and experiment runs stay outside the source tree. Keep source
provenance and compact reproducibility records with the benchmark results.

## Adding an algorithm

Add each protocol in its own kernel module with an explicit shard-local API.
Keep its shape/dtype validation, memory bound and scratch allocation beside the
kernel. Document numerical rounding and every barrier, transfer and completion
condition before connecting a serving callsite. Collective IDs are a caller-owned
resource: the current K3 full-world protocol reserves 1, distinct from fused MoE's
0. Do not share persistent workspace or assume independent protocols may overlap.

Reuse the distributed Torch bridge and keep model-specific dispatch in the model's
adapter. Preserve rank-local partial inputs explicitly and provide a Tensor fake
implementation that follows Torch symbolic shapes. Add adversarial numerical and
lifecycle tests, actual multi-process bridge and cache tests, and optimized HLO
execution checks before collecting paired timings. Give a new algorithm its own
exact qualification identity and admission evidence; do not inherit Two Shot's
measured cases. Keep unqualified shapes on the original native path and retain
losing measurements in campaign artifacts.
