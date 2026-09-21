# Streaming fused MoE prototype

Token tiles stay in VMEM through the expert FFN. Gate and up share two weight
buffers; down uses two independent buffers. Each slot targets 2 MiB of physical
weight storage, including packed FP4 storage where applicable: 8 MiB total for
the tested model geometries. Per-channel FP8 or block FP4 scales travel with
each weight slice instead of keeping every expert's scales resident.

## Schedule and numerical semantics

1. Gather a token tile and load its first gate slice. Prime the first down slice.
2. Load the matching up slice while computing gate. Load the next gate slice
   while computing up and SiLU(gate) times up. Keep each BF16 result in VMEM.
3. Reduce the complete intermediate row to obtain its FP8 scale, then quantize.
4. Stream down slices across output columns. Each slice contracts the complete
   intermediate axis. Keep the complete BF16 output in VMEM.
5. During the last down slice, prefetch the next tile/expert's input and gate.
6. Quantize with a full output-row scale and launch the original V2 output
   transport. Wait before reusing an output slot.

There is no activation or partial-sum HBM workspace between gate/up and down.
The original V2 routing, transport and combine remain in use. Output transport
may use an HBM contribution slab; that is separate from FFN intermediates.

Gate/up DMA tiles split the output axis; they retain the complete contraction
axis. FP4 matmuls accumulate scaled contraction blocks in the same order as the
reference. SiLU and row quantization occur only after the corresponding sums
are complete. FP8 weight scales and activation scales follow the original
arithmetic order. Columns are never independently quantized with different
row scales.

## Controls

- `weight_tile_bytes`: physical bytes per weight slot, default 2 MiB.
- `capacity`: resident token rows, one of 128, 256, 512, 1024.
- `output_buffers`: one or two output staging slots, default two.
- `gate_compute_rows`: FP4 gate/up computation row bound, default 256. Weights
  stay resident while all row blocks use them. FP8 computes the whole tile.
- `down_compute_cols`: computation width within a resident down DMA tile;
  zero means the whole DMA width. This does not change DMA size.
- `adaptive_rows`: optionally choose a smaller computation height for short
  token tiles. Capacity still determines allocated scratch.

For Qwen3.5 FP8 (H=4096, I=1024), 2 MiB gate/up slots hold [4096, 512],
and down slots hold [1024, 2048]. For Qwen3.8 NVFP4 (H=8192, I=2048),
packed slots hold logical [8192, 512] and [2048, 2048], respectively.
NVFP4 checkpoints must first use the existing block16-to-runtime-block64
conversion; this module consumes the runtime FP4 layout.

## Validation and limits

The prototype accepts FP8 and FP4 weights without biases. Tests compare full
outputs bit-for-bit with original V2 or adaptive V2, including concentrated
routing, empty ranks, short token inputs, FP4 block scales, and buffer reuse.
Production-shape benchmarks use EP8 and 4096 tokens per rank on one TPU7x host.
They use synthetic operands, not checkpoint evaluation or a serving workload.

Fixed weight scratch does not bound total VMEM independently of H, I or token
capacity. Full BF16 intermediate/output rows and compiler temporaries still
consume memory. The static scratch log excludes compiler temporaries and
register spills. Qwen3.8 with 1024 resident rows requires smaller computation
blocks to compile; a full-width implementation exceeded the hardware budget.

Reduced custom `vmem_budget_bytes` values are rejected: an earlier iteration
returned incorrect production-shape outputs under a 48 MiB compiler limit,
although the default hardware budget passed. Its root cause is unresolved.
The default-budget production-shape checks do not validate reduced budgets.

This is a standalone research prototype with explicit tuning parameters, not
a serving integration or an automatic shape-independent memory planner.

## Benchmark results

See the [measurement report](../../../../../docs/benchmarks/fixed_vmem_moe.md)
for Qwen3.5/Qwen3.8 timings, effective utilization, memory limits, and the
external FP4-to-FP8 experiments. Channel-wise FP4 quality remains unresolved
and requires real-checkpoint validation; those experiments are not part of
the serving path.
