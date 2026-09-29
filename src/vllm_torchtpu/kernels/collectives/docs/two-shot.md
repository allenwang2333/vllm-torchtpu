# Two-shot all-reduce

Each rank divides its flattened local input into one chunk per logical rank.
Chunks are padded to 128-element vector lanes. Rank `r` receives chunk `r` from
every peer, sums these contributions in FP32, then broadcasts the completed chunk.
The output joins the completed chunks in logical-rank order and removes padding.

## Protocol

1. Signal and consume an all-peer entry barrier.
2. Start one remote DMA per peer, sending that peer's owner chunk.
3. Drain all reduce-scatter transfers, then reduce the received contributions
   using an FP32 addition tree and round to the output dtype.
4. Start one remote DMA per peer to distribute the completed owner chunk.
5. Drain all all-gather transfers before returning.

A rank cannot complete reduce-scatter without data from every peer. Every peer
sends only after consuming its entry barrier. Consequently every current barrier
wait has completed before any rank can signal the next invocation's barrier.
This argument requires matching ordered groups and does not permit concurrent
protocols to share an ID. Semaphore checks remain enabled in both the public
effectful entry and the private uniform SPMD entry; their communication protocol
and scratch lifetime are identical. The private entry additionally requires the
[compiled-program contract](architecture.md#private-uniform-spmd-entry).

Two phases do not mean two network messages. A group of size `P` performs
`2(P-1)` remote data transfers and `P-1` barrier signals per rank. Per-rank data
volume is approximately `2(P-1)/P` times the padded local payload. Direct peer
traffic can traverse multiple physical links; topology and contention affect
performance.

## Memory and numerics

Inputs and final outputs are BF16 or FP32. Every reduction contribution is
converted to FP32 before addition. BF16 rounds once when writing the completed
owner chunk; the all-gather performs no additional arithmetic. Native XLA may use
a different reduction order and accumulation precision, so bitwise agreement
with native reduction is not required. Replicas produced by one custom call must
agree bitwise with one another.

All scratch buffers are invocation-local VMEM. The conservative 32 MiB admission
budget includes padded input/output, receive storage and FP32 reduction headroom.
For small arrays, padding is a material fraction of transport and allocation.
Large arrays that exceed the budget are rejected before Pallas execution.

For finite values without overflow or underflow, correctness uses an independent
FP64 host sum and an elementwise bound. Let `a = gamma(P-1) * sum(abs(x))`, with
`gamma(k) = k*u / (1-k*u)` and `u = 2^-24`. The allowed error is
`a + u_out * (abs(reference) + a)`, where `u_out` is `2^-8` for BF16 and `2^-24`
for FP32. Exactly representable cancellation fixtures use zero tolerance.
This is an operator contract, not a full-model accuracy certificate.

## Complete example

Save this as `two_shot_example.py` in an environment with vLLM TorchTPU and its
matching JAX/libtpu runtime. On a single host with 2, 4, 8, 16 or 32 visible TPU
devices, run `python two_shot_example.py`. For a distributed slice, run the same
file once on every worker with its unique zero-based process ID:

```bash
python two_shot_example.py --coordinator COORDINATOR_IP:12345 \
  --num-processes 4 --process-id WORKER_ID
```

The coordinator endpoint must be reachable from every worker. Initialize the
slice's usual TPU topology environment before starting these processes. This
example uses the runtime's device order directly; it does not qualify a topology
or choose the serving dispatch policy.

```python
import argparse

import jax
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.two_shot import two_shot_all_reduce

parser = argparse.ArgumentParser()
parser.add_argument("--coordinator")
parser.add_argument("--num-processes", type=int, default=1)
parser.add_argument("--process-id", type=int, default=0)
args = parser.parse_args()
if args.num_processes > 1:
    if not args.coordinator:
        parser.error("--coordinator is required for multiple workers")
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )

# Discover devices only after distributed initialization.
devices = jax.devices()
assert all(device.platform == "tpu" for device in devices)
mesh = Mesh(np.asarray(devices), ("tp",))
assert mesh.size in (2, 4, 8, 16, 32)
local_shape = (1, 7168)
sharding = NamedSharding(mesh, P("tp"))
x = jax.make_array_from_callback(
    (mesh.size,) + local_shape,
    sharding,
    lambda index: np.full(
        (1,) + local_shape, index[0].start + 1, dtype=ml_dtypes.bfloat16
    ),
)


def local_step(local_x):
    output = two_shot_all_reduce(
        local_x[0], axis_name="tp", axis_size=mesh.size, collective_id=3
    )
    return output[None]


run = jax.jit(
    jax.shard_map(
        local_step, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False
    )
)
output = run(x)
jax.block_until_ready(output)
expected = mesh.size * (mesh.size + 1) // 2
for shard in output.addressable_shards:
    np.testing.assert_array_equal(np.asarray(shard.data), expected)
print(f"process {jax.process_index()}: all local replicas equal {expected}")
```
