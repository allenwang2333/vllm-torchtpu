# Kimi sequence parallelism on TorchTPU

This note records the design, integration issues, fixes, and validation for
sequence parallelism (SP) in the TPU implementations of Kimi-Linear and
Kimi-K3. Results were collected on one `tpu7x-8` host with vLLM `0.25.1`.

## Is this automatic in XLA?

No. XLA schedules and lowers the operations it is given, including explicit
collectives, but it does not infer this model-level token distribution in the
current TorchTPU MPMD runtime. Without the changes described here, every TP
rank receives the full token sequence after the attention output all-reduce.
The MoE then dispatches duplicated tokens. The sharding, the placement of
all-gather/reduce-scatter, and the choice of TP versus EP process groups must
all be explicit in the model and communicator.

SP is enabled only when all of the following hold:

- pipeline parallel size is 1;
- expert parallelism is enabled;
- tensor parallel size is greater than 1; and
- data parallel size is greater than 1.

This follows vLLM's NVIDIA Kimi-K3 implementation. On an eight-chip host the
validation topology is TP4 x DP2 with EP8. TP8 x DP1 does **not** activate this
path.

## Design and fixes

At model entry, the padded token dimension is split into contiguous TP shards.
Each decoder layer then performs:

1. local normalization on the token shard;
2. TP all-gather and removal of SP padding before attention;
3. attention with its output projection TP reduction disabled;
4. TP reduce-scatter of the attention partials; and
5. local dense-MLP or EP-MoE computation on the resulting token shard.

The final hidden states are TP-all-gathered and trimmed before logits. Dense
and shared MLP weights are replicated (`disable_tp=True`) while SP is active,
matching the upstream implementation. MoE dispatch and combine use the full EP
group for SP; the existing DP group remains in use for ordinary DP MoE.

Kimi-K3 latent MoE needs special care. In the existing non-SP TPU path, TP
partials are summed before latent RMS normalization. Under SP, vLLM's runner
performs EP combine/reduce-scatter before `KimiRoutedOutputTransform`, so that
transform already receives the complete latent vector for its local token
shard. The extra TP all-reduce and division must therefore be skipped only on
the SP path. The routed normalization and replicated up-projection still run
after the combine, and the local shared-expert result is then added.

TorchTPU cannot export the generic communicator's
`torch.empty(symbolic_shape)` allocation. TPU-specific SP collectives now use
PyTorch functional collectives, which preserve the symbolic result shape and
lower through the distributed backend. They are exposed as
`custom_all_gather` and `custom_reduce_scatter`, so ordinary logits and non-SP
collectives retain their established implementation.

The functional final gather carries symbolic layout metadata out of the
compiled backbone. Since TorchTPU rejects SymInts at the input of the
separately compiled logits graph, the runner copies that result into a small,
reusable, concrete-shape activation buffer for each scheduler token bucket.
This is not a checkpoint-weight allocation and does not change the intentional
use of `torch.empty` during weight loading.

## Integration issues found

- The TPU layer-registration hook was a no-op. Consequently vLLM selected its
  generic Kimi model instead of the TPU model. The hook now calls
  `register_models()`.
- `grouped_topk` was absent from the TPU dynamic-compile unwrapping list. It is
  now registered, allowing the TPU Kimi routed-expert path to compile.
- Hybrid KV page sizing detected MLA layers but calculated page sizes only for
  ordinary `Attention`, eventually popping from an empty set. MLA now uses
  `PallasMLAttentionBackend` for the calculation and the code asserts that at
  least one attention layer was found.
- The first compile-safe collective attempt replaced all TPU gathers. That
  affected the separately compiled logits head and was narrowed to SP-only
  custom collectives.
- With `max_num_seqs=16`, the existing DP-idle path can synthesize a
  Pallas-capped 13-request dummy batch although only request-count buckets 8
  and 16 are precompiled. This causes an unrelated dynamic-shape failure. The
  validation uses `max_num_seqs=8`; the 13-request precompile gap remains a
  separate runner issue.
- A baseline worktree initially loaded an SP executable from the shared TPU
  compilation cache. The executable expected SP-replicated MLP weights and
  failed against the baseline's TP-sharded weights with a 4x parameter-size
  mismatch. Model-source changes are therefore not fully represented in the
  current cache key. Baseline measurements use isolated `VLLM_CACHE_ROOT` and
  `VLLM_XLA_CACHE_PATH` directories; no existing cache was deleted.
- `mgoin/Kimi-K3-pruned75` uses float W4A16/MXFP4 MoE weights. Current `main`
  rejects that quantization before model execution, including with dummy
  weights. K3 validation therefore uses the existing MXFP4 implementation from
  PR #220 in an isolated combined validation worktree. The SP commits do not
  copy or modify that implementation, and PR #220 itself was not changed.
- vLLM's generic sequence chunk pads a two-dimensional `[tokens, hidden]`
  tensor. Kimi-K3 carries `[tokens, 1, hidden]`; using the generic padding
  tuple grew the singleton axis and failed Dynamo shape propagation. The K3
  path temporarily removes that structural singleton axis around token
  chunking and reduce-scatter padding, then restores it afterward.
- K3 allocates an empty attention-residual stack with shape
  `[tokens, 0, hidden]`. It was initially allocated before token sharding, so
  the first residual append tried to concatenate a full-token stack with a
  sharded-token state. Creating it after sharding then exposed TorchTPU's
  inability to export `new_empty` with the chunk op's symbolic token count.
  It is now a zero-width view of the sharded hidden states, which needs no
  allocation and preserves the correct token count.
- The full 93-layer K3 checkpoint cannot fit the TP4 x DP2 SP topology on this
  eight-chip host: replicated dense/shared MLP weights plus two DP replicas
  exhaust HBM while MXFP4 weights are repacked. This is not a zero-fill
  regression; checkpoint parameters continue to use `torch.empty`. Real K3
  remains a TP8 x DP1 deployment on this host, where model-level SP is not
  activated.
- Automatic compact-hybrid KV sizing derives its attention-block count from
  each rank's measured free HBM. During the real Kimi-Linear TP4 x DP2 load,
  those measurements differed by ten blocks while the multiprocess executor's
  consistency tolerance is four, so startup stopped at its assertion. The
  compact-hybrid calculator currently ignores `--kv-cache-memory-bytes`; an
  explicit equal `--num-gpu-blocks-override` avoids this nondeterministic
  validation failure. This is an existing KV-sizing synchronization issue,
  not an SP collective or model-correctness failure.

## Validation configuration

The compiled dummy-weight smoke and benchmark command used:

```text
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --load-format dummy --tensor-parallel-size 4 \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --enable-expert-parallel --attention-backend CUSTOM \
  --max-model-len 256 --max-num-batched-tokens 16 \
  --max-num-seqs 8 --compilation-config '{"compile_sizes":[16]}'
```

The end-to-end smoke request used a 13-token prompt and generated 16 tokens.
It completed without recompilation or collective errors. Dummy output is not a
quality signal because the weights are random.

Kimi-K3 was also compiled in the combined MXFP4 validation worktree as a
four-layer model (three KDA layers followed by one MLA layer), with TP4 x DP2,
EP8, a 16-token compile bucket, and dummy weights. All eight ranks compiled,
both API servers reached `Application startup complete`, and a completion
request consumed 8 prompt and 8 generated tokens with HTTP 200. This exercises
the K3-only `[tokens, 1, hidden]` sharding, latent routed-output transform,
shared expert, MLA transition, final gather, and separately compiled logits
path end to end. The layer reduction is needed only to fit the diagnostic SP
topology; the full topology's HBM limitation is described above.

## Benchmark results

### Earlier TP8 utilization profiling

The compiled dummy-weight TP8 profiling run used explicit 30-second phases.
The ordinary request loop delivered 62.1 output tok/s for decode batch 1 and
400.5 output tok/s for decode batch 8. A continuously fed loop raised those
figures to 160.4 and 556.7 output tok/s, respectively. This shows a substantial
host/request-feeding bubble at low concurrency, even though the server log
confirmed that vLLM asynchronous scheduling was enabled automatically.

In the continuously fed run, the sampled duty cycle averaged 44.9% for batch 1
and 48.1% for batch 8, with a maximum of 49.8%. On this runtime the exported
duty-cycle counter has an effective 50% ceiling; 49.8% is therefore its
saturated value, not evidence that another 50 percentage points can be reached
by a scheduler flag. The metric measures time that the TensorCore execution
pipeline is busy, rather than MXU arithmetic alone. The separate TensorCore
utilization samples averaged only 7.1% for batch 1 and 4.2% for batch 8. Taken
together, those counters say that the execution pipeline is occupied but most
cycles are not dense MXU work: vector/scalar work, KDA/state updates, MoE
routing, memory movement, and collectives are material bottlenecks.

To approach the effective duty-cycle ceiling, keep a continuous request queue,
use enough concurrent sequences to fill the configured token buckets, retain
compiled mode, and leave async scheduling enabled. Literal 100% in this
counter is not a meaningful target on this setup, and sequence parallelism is
not inferred by XLA as a way to reach it.

### Sequence-parallel comparison

Kimi-Linear dummy-weight requests used a 14-token arithmetic prompt and 32
generated tokens, with three measured rounds after warmup. Both variants used
fresh, isolated compilation-cache directories.

| Variant | Concurrency | Median wall time | Mean request latency | Prompt tok/s | Output tok/s | Total tok/s |
|:---|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 0.753 s | 0.752 s | 18.6 | 42.5 | 61.1 |
| SP | 1 | 0.771 s | 0.771 s | 18.2 | 41.5 | 59.7 |
| Baseline | 8 | 0.569 s | 0.506 s | 197.0 | 450.2 | 647.1 |
| SP | 8 | 0.595 s | 0.531 s | 188.1 | 430.0 | 618.1 |

At this deliberately small 16-token bucket, SP is 2.4% slower in total
throughput at concurrency 1 and 4.5% slower at concurrency 8. The added
all-gather/reduce-scatter overhead is not amortized here. SP's purpose is to
remove duplicated token work in the EP MoE and improve larger-batch scaling
and memory behavior; a speedup should not be inferred from this microbenchmark.

### Real-weight GSM8K sanity check

The full Kimi-Linear checkpoint was served in compiled mode with the active SP
topology (TP4 x DP2, EP8), `max_model_len=512`, and an explicit
`num_gpu_blocks_override=1000`. Ten deterministic examples from the official
GSM8K test split produced the following final values:

| Problem | Reference | Generated | Correct |
|:---|---:|---:|:---:|
| Duck-egg sales | 18 | 18 | yes |
| Robe fabric | 3 | 3 | yes |
| House flipping | 70,000 | 70,000 | yes |
| Weekly sprints | 540 | 540 | yes |
| Chicken feed | 20 | 20 | yes |
| Discounted glasses | 64 | 64 | yes |
| Three sheep flocks | 260 | 260 | yes |
| Weekly overtime | 460 | 460 | yes |
| Program downloads | 366 | 366 | yes |
| Bakery purchase | 694 | 694 | yes |

Result: **10/10 numerically correct**. The responses also contained coherent
intermediate arithmetic. Some used Markdown bold headings rather than the
requested `#### number` delimiter, so correctness was checked from the final
value rather than treating output-format variation as a model error. Two
different test examples were excluded because their otherwise sensible
reasoning hit the deliberately small 256-token response cap before emitting a
final answer; they were not counted as passes.

### Full real K3 validation blocker

The full real K3 checkpoint was also loaded at TP8 x DP1 in compiled mode,
because it does not fit the SP topology on this eight-chip host. Weight loading
and MXFP4 repacking completed (442.52 GiB of checkpoint data), all eight ranks
finished warmup compilation, and the API server reached
`Application startup complete`. A cold load took about 780 seconds; a second
run using the warm caches took about 292 seconds for loading plus 378 seconds
for engine initialization, of which 208 seconds was compilation.

Generation did not complete. An eight-request attempt and a separate
single-request attempt both remained at zero generated tokens until the
multiprocess executor's five-minute timeout. The isolated request had completed
its 16-token prefill and scheduled four more tokens, but no worker response was
returned; the terminal error was `RPC call to sample_tokens timed out`. The
server then returned HTTP 500 and shut down. The checkpoint has no usable chat
template, so these tests correctly used the raw completions endpoint; the
failure occurred after request admission rather than during prompt rendering.

This prevents reporting a ten-prompt real-K3 quality result. It is not evidence
against the new SP path: TP8 x DP1 does not enable SP, and the same timeout was
reproduced with one request, independently of concurrency. The evidence scopes
the remaining issue to the existing real-K3 TP8 execution/sampling path. The
real Kimi-Linear 10/10 check above validates full-checkpoint SP quality, while
the reduced TP4 x DP2 K3 smoke validates K3-specific SP execution, including
latent MoE, through an HTTP completion.
