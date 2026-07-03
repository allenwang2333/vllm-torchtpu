# Kernel Arena Run — qwen35_residual_rmsnorm_qkv_fp8

- Run id: `20260702-231727-6c49f36e`
- Backend: `vertex`
- Generations: 10
- Candidates: 31
- Passed numerics: 28
- Best geomean speedup: **1.03x**
- Best peak utilization: 0.0%
- Wallclock: 898.7s

Winner promoted to runs/<task>/<run_id>/winner.py

## Top candidates

- `9e850f676f` gen4 persona=None role=optimizer speedup=1.03x peak%=0.0
- `5aa722331a` gen0 persona=minimalist role=author speedup=1.02x peak%=0.0
- `d00d57aa12` gen4 persona=None role=optimizer speedup=1.02x peak%=0.0
- `136c2161d5` gen6 persona=None role=optimizer speedup=1.02x peak%=0.0

## Insights pool (final)

1. Upcasting fp8/int8 weights to fp32 before dot_general silently disables the low-precision MXU path and doubles-to-quadruples VMEM traffic; feed native low-precision operands with preferred_element_type=jnp.float32 instead.
2. For small-M decode-shaped matmuls (M<=256), the kernel is latency/memory bound, not compute bound; collapsing the OUT grid into a single block removes pipeline overhead and repeated scratch reloads.
3. Fused rmsnorm+fp8-qkv kernels must persist the quantized activation (and its scale) in scratch across the N-tiling grid dimension; recomputing per-token quant for each weight tile is pure waste.
4. Casting fp8/int8 weights to fp32 before dot_general silently drops you off the MXU low-precision path — always feed bf16 (or the native quantized type) operands with preferred_element_type=fp32.
5. Small-M (bs<=128) fused-matmul kernels are setup/latency bound, not FLOP bound; reducing per-tile redundant reductions dominates over raw matmul tuning.
6. For fused rmsnorm+fp8-qkv kernels, per-token quant must be computed once per row-block in the prologue and cached in scratch — recomputing it per output-tile is a hidden O(OUT/BN) tax.
7. Casting fp8/int8 weights to fp32 before dot_general silently discards the entire bandwidth/MXU benefit of quantization; the dot must consume the low-precision dtype directly with fp32 accumulation.
8. Small-batch QKV projections are weight-memory-bound: the winning move is reducing weight bytes (native fp8) and reusing activations across weight tiles, not increasing arithmetic intensity via extra float ops.
9. In fused rmsnorm+quant+matmul kernels tiled over the output (N) dimension, everything before the matmul (norm, residual, per-token quant) is invariant across N-tiles and MUST be computed once in a pl.when(program_id_N==0) prologue and cached in scratch, else you pay it N/BN times.
10. Upcasting quantized operands to fp32 before dot_general silently disables the fp8/int8 MXU path and yields ~1.0x baseline; keep operands in their low-precision dtype and only set preferred_element_type=fp32 for the accumulator.
11. A ~1.02x speedup with 'unknown/0% peak' profiler readings usually means the kernel lowered to essentially the reference computation — check for hidden fp32 upcasts and redundant recompute across grid columns first.
12. Fused residual+RMSNorm+fp8-quant+matmul kernels are limited by the elementwise prologue and activation HBM re-reads when M is small; the fp8 matmul itself is cheap, so splitting prologue (M-only grid) from matmul (2D grid) usually wins.
13. With a `pl.when(n==0)` prologue, x/residual still occupy double-buffered VMEM on every n iteration — the guard saves compute but not HBM streaming; only a separate kernel or grid restructure removes that traffic.
14. Small-M fp8 matmuls (BM=64) underfill the MXU systolic array; padding M to 128 or the geomean-driving smallest shape is often the highest-leverage tuning knob.
15. Fused residual+RMSNorm+FP8-quant+matmul kernels have a producer-consumer split: the norm/quant is per-M-row (cheap, bandwidth-bound) while the matmul is per-(M,N) tile (compute-bound). Stashing quantized activation in VMEM scratch under pl.when(n==0) is correct but serializes the N loop via 'arbitrary' semantics.
16. For small M (bs64), the M-dimension MXU tile is undersized; prefer a single large N tile (BN=OUT) to keep the MXU busy rather than looping over many small N blocks.
17. Dequant of fp8 matmul needs both per-token activation scale (BM,1) and per-channel weight scale (1,BN); fuse them into one broadcasted multiply to save an elementwise pass over (BM,BN).
18. When a kernel has a per-row prologue (norm/quant) gated by pl.when(n_idx==0), an n-dimension grid still re-streams the row inputs (x/residual) every n-step unless the index_map makes them n-invariant AND the pipeline recognizes reuse — collapsing the n-loop into the kernel body is often cheaper for memory-bound small-M shapes.
19. fp8 MXU matmuls with tiny M (batch<=64) are M-lane-starved; prefer BM=M (single tile) so the norm/quant runs once and the MXU sees the largest possible M contraction feed.
20. For fused residual+rmsnorm+quantized-matmul, storing the quantized activation to a persistent VMEM scratch and reusing it across output tiles is the right structure; the remaining win is minimizing redundant HBM reloads of the un-quantized inputs.