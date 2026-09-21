# Output-tiled fused MoE proof of concept

This experimental FP8 path starts from original V2's 3+3 weight buffers.
The original `fused_moe/v2` and the existing adaptive kernel are unchanged.
Routing, activation quantization, transport and combine retain V2's semantics.

The later fixed-slot prototype is documented in
[`streaming_fused_moe`](../streaming_fused_moe/README.md).

## Schedule

Each W1/W2 buffer holds an output-column slice, with the contraction dimension
kept complete. For each visited expert:

1. Prefetch the first two W1/W2 slices. W2 loads overlap W1 computation.
2. Keep one paired gate/up slice of W1 resident while processing all the
   expert's rows in fixed token tiles (128 by default). Apply the activation and asynchronously
   store BF16 intermediate columns in HBM.
3. Repeat for the remaining W1 slices. Prefetch two slices ahead through the
   three-slot ring, and double-buffer activation loads and intermediate stores.
4. For each W2 output slice, stream complete intermediate rows, compute their
   global row scale, quantize to FP8 and run the down projection. Asynchronously
   store BF16 output columns, with the intermediate and weight scales applied.
5. Stream complete output rows, compute their global wire scale, and use V2's
   transport and combine.

Partial matrix sums are not stored: the output slices are concatenated.
The first implementation resets the weight ring at each expert. With N=2,
only two of the three allocated slots are needed; N=4 exercises ring reuse.
FP8 per-channel scale tables remain resident as in original V2.

## Memory

`weight_slices=N` reduces raw FP8 weight scratch from
`3 * (H * 2I + I * H)` to `3 * (H * 2I + I * H) / N` bytes. For Qwen3.5,
N=2 reduces it from 36 MiB to 18 MiB.

VMEM holds fixed token tiles, regardless of the actual number of routed rows.
HBM temporary slabs are reused between experts. Their capacity is based on the
worst case (every input token selects one expert), plus destination padding
and a final tile, rather than on average expert load. At 32K input tokens,
Qwen3.5's two BF16 temporary slabs together use 321.25 MiB per rank.

An optional `vmem_budget_bytes` sets a compiler VMEM limit capped by the device's
normal budget. The preflight estimate includes V2's full tile body and the new
scratch; it is not the compiler's measured peak. The current Qwen3.5 N=2 path
passes complete numerical validation with a 32 MiB compiler budget.

## Scope and limitations

- FP8 weights with per-channel scales, without biases. NVFP4 is not implemented.
- The slice count is explicit; automatic tile selection is future work.
- Both output widths must be multiples of 256. The 128-column experiment did
  not preserve baseline results and is rejected.
- Full contraction dimensions remain resident within each weight slice.
  This does not yet provide a fixed memory bound for arbitrary hidden sizes.
- W1 rereads the input for every slice. W2 rereads and requantizes the
  intermediate for every slice. HBM traffic and phase transitions can make
  this slower than V2; reduced VMEM is not evidence of equal performance.
- An attempted cross-expert prefetch optimization produced incorrect outputs
  under reduced compiler budgets and was removed. Its root cause is unresolved.
  The validated schedule below retains only the within-expert slice ring.

## Validation

Device tests compare complete outputs with original V2 for N=2 and N=4,
and concentrated routing with 32K rows per selected expert and six idle ranks.
The production-shape benchmark uses Qwen3.5 geometry, EP8, and 4K input tokens
per rank, with deterministic synthetic weights rather than a checkpoint.

```bash
TPU_SKIP_MDS_QUERY=true JAX_PLATFORMS=tpu PYTHONPATH=src \
  python scripts/benchmark_adaptive_fused_moe.py \
  --model /path/to/Qwen3.5-397B-A17B-FP8 \
  --output /tmp/tiled-moe-new-run \
  --implementation tiled --weight-slices 2 --vmem-budget-mib 32 \
  --tokens-per-rank 4096 --warmup 10 --iterations 50
```

## Qwen3.5 initial result

At EP8, 4K tokens per rank, one warmed-up round of 50 synchronized calls:

| Metric | Original V2 | Tiled N=2, 32 MiB budget |
| --- | ---: | ---: |
| Weight scratch | 36 MiB | 18 MiB |
| Source-level VMEM estimate | 44.39 MiB | 30.14 MiB |
| Layer median | 3.151 ms | 4.348 ms |
| Layer p95 | 3.183 ms | 4.370 ms |

The complete 32K-token output is bit-identical to the saved V2 baseline. The
sampled dense reference also passes the original error thresholds. Latency
increased by about 38%; this is a memory-reduction POC, not a performance-neutral
replacement. Timings are synchronized host wall time, and must not be compared
directly with the earlier 1.85 ms device-only fused-kernel time.

## Token tile height: 128 versus 256

The benchmark accepts `--capacity 128` (default) or `--capacity 256`.
The 256-row compute tile requires a scale-transport layout change: physical
HBM scale-mirror rows stay 128 elements wide, with two physical rows per
logical compute tile. DMA offsets and semaphore byte counts use that factor;
the public mirror view retains its original logical shape. Original V2 is
unchanged.

A same-code comparison at N=2, EP8, 4K input tokens per rank, default compiler
budget, 10 warmups and 50 timed calls gave:

| Metric | 128 tokens per tile | 256 tokens per tile |
| --- | ---: | ---: |
| Layer median | 4.336 ms | 3.930 ms |
| Layer p95 | 4.369 ms | 3.967 ms |
| Source-level VMEM estimate | 30.14 MiB | 40.77 MiB |
| Complete output versus original V2 | Bit-identical | Bit-identical |

Median latency improved by 9.37%, with an additional 10.625 MiB in the VMEM
estimate. Both runs passed the sampled dense reference check. The 256-row
configuration exceeds the existing 32 MiB preflight budget; use the default
budget for this comparison. The larger tile does not change weight-slice sizes.
The measurements alone do not separate MXU utilization, loop overhead and
transport savings.
