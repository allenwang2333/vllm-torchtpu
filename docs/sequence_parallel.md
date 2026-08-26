# Model-level sequence parallelism

vLLM TPU exposes model-level sequence parallelism through vLLM's existing
`--additional-config` argument:

```bash
vllm serve MODEL \
  --tensor-parallel-size 8 \
  --additional-config '{"sequence_parallel_size": 4}' \
  --compilation-config '{"compile_sizes":[4,1024,2048,4096]}'
```

`sequence_parallel_size=1` is the default and preserves the ordinary serving
path. The setting participates in vLLM's configuration hash, so compiled
artifacts from different layouts do not share a cache key. If the installed
vLLM exposes a native `ParallelConfig.sequence_parallel_size`, the TPU backend
accepts that field as well.

When SP is enabled, every compile size must be divisible by
`sequence_parallel_size`. Keeping one layout across all buckets, including
decode, gives each compiled trace the same model subgraph boundaries.

This setting does not reuse prefill context parallelism. PCP expands the process
world and partitions context/KV-cache work. Model-level SP instead factors an
existing ordered TP group into a logical mesh:

```text
TP = sequence_parallel_size x lane_size
tp_rank = lane_rank * sequence_parallel_size + sequence_rank
```

The common TPU implementation owns configuration resolution, rank validation,
group construction, token sharding, and the lane/sequence collectives. It does
not switch arbitrary models into SP. A model adapter defines the tensor-layout
boundaries required by its attention, residual, recurrent-state, and
feed-forward implementations.

## Current support

The first adapter targets Kimi-K3 execution:

| Property | Supported value |
| --- | --- |
| Architectures | `KimiLinearForCausalLM`, `KimiK3ForConditionalGeneration` |
| Sequence parallel size | A divisor of TP; SP4 is currently validated |
| Validated TP sizes | 8 and 32 with SP4 |
| Logical layouts | SP4 x lane2 and SP4 x lane8 |
| DP / PP | DP1 / PP1 |
| SP token buckets | Every configured TPU compile bucket; each must be divisible by SP |
| Decode | Uses the same SP layout as prefill |
| Feed-forward / MoE | Full-token all-gather followed by local reshard |

Unsupported models and topologies fail during configuration instead of silently
running without SP.

For Kimi-K3, attention consumes the full ordered token tensor, then its partial
TP result is reduced along the lane axis and reduce-scattered along the sequence
axis. Norms and AttentionResidual state stay sequence-local. The layout remains
the same for every compile bucket so TPU traces have stable subgraph boundaries.
The current feed-forward blocks consume full ordered tokens, so the adapter
all-gathers at that boundary and returns to sequence shards afterward.

## Adapting another model

Most models require modeling changes, even when they can reuse the common group
plan and collective helpers. An adapter normally needs to:

1. choose safe input and output shard boundaries;
2. disable or replace the attention projection's default TP all-reduce;
3. keep residuals, norms, and any recurrent state sequence-local;
4. define dense and MoE feed-forward transitions;
5. restore the tensor layout expected by logits, pooling, or later stages; and
6. register the architecture in the TPU support validation after correctness
   and compile-shape tests pass.

Kernel changes are not automatically required. They are needed only when an
attention, recurrent, or MoE kernel assumes a full token dimension, performs a
collective internally, or cannot compile the adapter's local shapes. Optimized
native sharded kernels can then replace full-token gathers without changing
the user-facing configuration.
