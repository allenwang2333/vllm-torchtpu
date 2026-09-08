# Draft bug report: pallas TPU HBM operands — staging hop cannot be avoided from the BlockSpec alone

**Title**: pallas TPU: BlockSpec(memory_space=pltpu.HBM) does not reach the
custom call's operand colors — DMA-only operands are always staged through
VMEM unless the *argument* is marked with with_memory_space_constraint

## Summary

For a pallas TPU kernel operand that the kernel only *DMAs* from (never
vector-reads), there is no way via `BlockSpec(memory_space=...)` to keep the
operand in HBM: pallas derives the tpu_custom_call's operand memory-space
colors from the input *argument avals*, not from the BlockSpec, so libtpu
stages the operand into VMEM regardless, and the kernel's in-HBM DMA plan
silently becomes an extra VMEM hop (plus a layout retile when layouts
differ). The Mosaic body types the ref `#tpu.memory_space<hbm>`, but the
buffer the kernel actually receives is whatever XLA staged.

## Environment

- jax 0.11.1 / libtpu 0.0.46 (probe venv); API surface also present on
  jax 0.10.2 / libtpu 0.0.44.1 (production stack).
- TPU, single-device repro (ici-2x2x4 slice).

## Reproduction

`scripts/k3/vmem_chain_probe.py` in this worktree, variants `inkernel`,
`wmark`, `wmark_pre` (jobs 534, 539, 540). Shape of it:

```python
def kernel(xb_ref, w_ref, h_ref, wv_ref, sem):
    pf = pltpu.make_async_copy(src_ref=w_ref, dst_ref=wv_ref, sem=sem)
    pf.start()
    ...  # compute on xb_ref
    pf.wait()

pl.pallas_call(
    kernel,
    out_shape=(jax.ShapeDtypeStruct(xb.shape, xb.dtype),
               jax.ShapeDtypeStruct(w.shape, w.dtype)),
    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM),
              pl.BlockSpec(memory_space=pltpu.HBM)],   # w: intent is "raw HBM"
    out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
    scratch_shapes=[pltpu.SemaphoreType.DMA],
)(xb, w)   # w: bf16[7168, 224]
```

## Evidence (all from on-device jobs with --xla_dump_to)

1. **Pallas emits clean HLO.** In `before_optimizations`, the custom call
   takes the raw HBM parameter: `custom-call(%xb.1, %w.1)` — no staging.
2. **libtpu inserts staging per the operand colors**, and the colors come
   from the input avals (`_resolve_memory_spaces` in
   `jax/_src/pallas/mosaic/pallas_call_registration.py` reads
   `aval.memory_space` of the call *arguments*; BlockSpec.memory_space only
   sets the kernel-side ref type):
   - no mark (job 534): operand colors VMEM → `copy-start`+`copy-done`
     (HBM→VMEM async DMA) + `{0,1}→{1,0}` retile in VMEM, ~2.1+2.1 us per
     call; the kernel's DMA then reads the staged VMEM buffer (numerics
     prove the binding follows the HLO operand: the kernel's ref expects
     `{1,0}`, the original param is `{0,1}`, and the result is exact).
   - `with_memory_space_constraint(w, pltpu.HBM)` but unpinned layout (job
     539): operand color flips to HBM (`input_memory_space_colors` operand 1
     color 0), but the `{1,0}` layout constraint forces an
     HBM→VMEM→HBM roundtrip retile — worse than no mark.
   - mark + layout pinned to the constraint (`jax.experimental.layout
     .Layout(major_to_minor=(0,1), tiling=((8,128),(2,1)))`, job 540): the
     custom call takes the raw HBM parameter with **zero staging copies**;
     the kernel's DMA is the only HBM→VMEM transfer. Numerics exact.
3. So the net effect of the gap: anyone who writes
   `BlockSpec(memory_space=pltpu.HBM)` intending "the kernel will DMA from
   HBM itself" silently pays an extra HBM→VMEM staging (plus retile) per
   call, and the intent is only expressible through the unrelated-looking
   argument-marking API plus a matching physical layout.

## Asks

1. Propagate `BlockSpec(memory_space=...)` for input operands into the
   custom call's operand memory-space colors (or warn when a BlockSpec
   requests HBM but the operand gets staged to VMEM anyway).
2. When the operand color is HBM and the layout constraint does not match
   the buffer, prefer retiling **in HBM** or relaxing the constraint over
   the HBM→VMEM→HBM roundtrip.
3. Document `pltpu.with_memory_space_constraint` as the way to keep a
   pallas_call operand in HBM (its docstring currently only says it
   "constrains the inputs of pallas_calls").

## Related measured limits (same probe)

- A kernel must wait its own DMAs before exit: an unwatched in-kernel DMA
  races the consumer (rel err ~1; kernel exit does not drain DMAs).
- Gridded BlockSpecs with memory_space=HBM are rejected unless the block is
  the whole array with a trivial index map ("...supports in memory space ANY
  only blocks having the same block shape as the array shape and a trivial
  index_map", jax 0.11.1) — no per-block manual HBM reads.
