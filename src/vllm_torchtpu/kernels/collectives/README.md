# TPU collective kernels

This directory contains JAX/Pallas communication kernels used by vLLM TorchTPU.
The kernels operate on arrays and communication groups. Model layouts, packing,
Torch mutation semantics and serving dispatch belong to their callers.

## Two-shot all-reduce

`two_shot.two_shot_all_reduce` computes a SUM all-reduce over a flat, single-axis
JAX mesh. Its input is local to one shard, and its output has the same shape and
dtype. BF16 inputs accumulate in FP32 and round once at the output. Scratch is
invocation-local VMEM; callers do not allocate or retain workspace. The public
entry remains compiler-effectful. A [private uniform SPMD entry](docs/architecture.md#private-uniform-spmd-entry)
is reserved for adapters that establish its stricter compiled-program contract.

```python
import jax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.two_shot import two_shot_all_reduce


def local_step(x):
    y = two_shot_all_reduce(
        x[0], axis_name="tp", axis_size=mesh.size, collective_id=3
    )
    return y[None]


# x has global shape [mesh.size, *local_shape].
run = jax.jit(
    jax.shard_map(
        local_step, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False
    )
)
```

Create `mesh` after distributed JAX initialization. Every participant must use
matching shapes, dtypes, collective IDs and execution order. Repeated ordered
calls can reuse an ID. Independent call sequences require distinct IDs, and
overlapping execution is unsupported. The example preserves a leading logical-rank
dimension so each replica is observable. See the [complete runnable example](docs/two-shot.md#complete-example)
for mesh construction, distributed initialization and input creation.

The explicit kernel validates shape, dtype, group size and its conservative
32 MiB VMEM budget. Hardware/runtime eligibility and performance selection are
separate caller responsibilities. Unsupported or unqualified serving requests
retain the native communicator. The kernel module itself performs no backend
discovery. Importing it through `vllm_torchtpu` follows the package's normal
environment initialization, including its compiler flag defaults.

## Documentation

- [Architecture and integration boundary](docs/architecture.md)
- [Two-shot protocol and numerical contract](docs/two-shot.md)
- [Validation and benchmark protocol](docs/benchmarking.md)

The protocol is adapted from Apache-2.0 vllm-torchtpu PR #567, commit
`6da6ed314155e5fa2533bfdea4d68b7d3c7f230b`, through the `tpu-collectives`
implementation at commit `5c75cc95dbd69c27eff07dec8316fb863c5ab371`.
Historical measurements of those implementations do not qualify a new kernel
revision or runtime.
