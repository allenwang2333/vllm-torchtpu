# Probe 17 — U2 (f32 kernel input) and `v11` (2-D tiled output): feasibility

Settles the two `unmodelled` feasibility terms in `stage-e-boundary.md:48` (U2) and `:55` / the
`v11` block (`:152-174`), which `§6 item 2` makes mandatory. **Evidence class: `static-compile`
throughout. No TPU job was run and no µs figure is produced here** — the prices (26.251 µs, 10.007 µs)
are already `measured (j-9b520877)` elsewhere; this probe only says whether the changes are reachable.

## Verdicts

### U2 — **REACHABLE**, but the *scratch* dtype must change too

`hlo/diff-base-vs-u2.txt`. Real `_sc_topk_direct` (`sparsecore_topk.py:419`) at production dims
`f32[1024,10240] → i32[1024,2048]`, AOT-compiled for tpu7x, post-optimization ENTRY:

```
-  %bitcast-convert_bitcast_fusion = s32[128,80,8,128]{3,2,1,0:T(8,128)} fusion(%s.1), kind=kLoop,
     calls=%fused_computation, ... "output_window_bounds":["64","16","1","1"] ...
+  %bitcast.3 = f32[128,80,8,128]{3,2,1,0:T(8,128)} bitcast(%s.1)
```

The `kLoop` **fusion is gone**, replaced by a free `bitcast` with **no `backend_config` at all**.
This re-derives `stage-e-boundary.md:77-80` independently: the base fusion carries
`used_scoped_memory_configs` **16 MiB @ memory_space 1** and the u2 replacement carries none
(per-instruction table in `hlo_diff.txt`). So the op really was planned work, and declaring the
input `f32` really deletes it. ⚠ The 26.251 µs price stays a *separate* measured claim — this probe
does not re-price it.

**But the DMA is dtype-checked, not byte-oriented.** `compile_check.txt` variant `U2a` (f32 HBM ref →
i32 VMEM scratch, the minimal edit at `sparsecore_topk.py:169`):

> `'tpu.enqueue_dma' op DMA source and target element type mismatch`

So the change is **three** edits, not one: `:438` (`words = scores`), `:455` (resident scratch
`jnp.int32 → jnp.float32`), and an in-register `jax.lax.bitcast_convert_type` at every read/write of
`resident_vmem` — `:172-173` (`cvt`/`monotone_key`) and `:178` (`load_keys`). That form (`U2b`)
lowers, and its Mosaic body contains **2 `tpu.bitcast` ops** with `%arg2: memref<10485760xf32, hbm>`
and the scratch `memref<10240xf32>` (vs 0 bitcasts / all-i32 for the incumbent control `C0`).
`lax.bitcast_convert_type_p` is registered for `[*tpu_core.CoreType]`, i.e. including
`SC_VECTOR_SUBCORE` (`jax/_src/pallas/mosaic/lowering.py:4505-4520`).

Ref-level `.bitcast` — which would avoid the two per-load `tpu.bitcast`s — is **unavailable on a 1-D
ref**: `IndexError: list index out of range` at `jax/_src/state/utils.py:98`, which does
`shape[-2] = shape[-2] * x_bitwidth // y_bitwidth` unconditionally even at equal bitwidth. It **does**
work on a 2-D ref (`aot_controls.txt`, `refbitcast-2D-f32-scratch: AOT COMPILED ok`), so reshaping
`resident_vmem` to `(slice_len // LANES, LANES)` is an option if the two bitcasts ever price. Noted,
not recommended — it is a larger structural change.

⚠ **Residual risk the correctness gate must close, not a blocker.** `monotone_key`
(`sparsecore_topk.py:116-118`) writes an arbitrary int32 bit pattern *through an f32-typed VMEM
store*. Bit-preservation across an f32 store/load of a NaN payload is assumed, not proven here.
`v11` already owes a bit-identical row gate; U2 owes the same one.

### `v11` — **REACHABLE**. The `ROOT reshape` disappears

`hlo/diff-base-vs-v11.txt`, same AOT compile:

```
-  %sc_topk_direct_….call-start = ((s32[10485760]{0:T(1024)}, …), s32[2097152]{0:T(1024)}) call-start(…)
-  %sc_topk_direct_….call-done  = s32[2097152]{0:T(1024)} call-done(…)
-  ROOT %reshape.4 = s32[1024,2048]{1,0:T(8,128)} reshape(%sc_topk_direct_….call-done)
+  %sc_topk_direct_….call-start = ((s32[10485760]{0:T(1024)}, …), s32[1024,2048]{1,0:T(8,128)}) call-start(…)
+  ROOT %sc_topk_direct_….call-done = s32[1024,2048]{1,0:T(8,128)} call-done(…)
```

ENTRY goes 13 → 12 instructions. The base `ROOT %reshape.4` carries `used_scoped_memory_configs`
**4 MiB @ ms1** — re-deriving `stage-e-boundary.md:54`'s `HLO :1480` — and in `v11` there is nothing
left to carry it. XLA hands the Pallas custom call the `{1,0:T(8,128)}` buffer directly. Pallas pins
only the minor-to-major dim order for a TPU custom call
(`jax/_src/tpu_custom_call.py:374-375`), leaving the tiling to XLA, and XLA chose the consumer's.

The patch is four edits: `:451` (`(b*k,) → (b,k)`), `:317` and `:323` (`out_hbm.at[pl.ds(row_c*k,k)]
→ out_hbm.at[row_c]`), `:491` (drop `out.reshape(b, k)`). ⚠ The indexer form matters —
`o.at[pl.ds(row,1)]` gives `'tpu.enqueue_dma' op DMA source and target shape mismatch`
(a `(1,k)` vs `(k,)` rank mismatch, `compile_check.txt` `V11b`); `o.at[row]` and `o.at[row, pl.ds(0,k)]`
both lower.

**What the out DMA becomes** (`dma_shape.txt`, Mosaic dump, incumbent vs 2-D):

| | incumbent | `v11` |
| --- | --- | --- |
| HBM memref | `memref<2097152xi32, hbm>` | `memref<1024x2048xi32, hbm>` |
| slice | `memref_slice %arg2[%0] → memref<2048xi32>` | `memref_slice %arg2[%arg1,%c0] → memref<1x2048xi32>` then `memref_squeeze → memref<2048xi32>` |
| `tpu.enqueue_dma` | `2048xi32 vmem → 2048xi32 hbm` | **identical** |

⚠ **Correction to `stage-e-boundary.md:55` / the `v11` `transfer` field.** Stage E models the move as
"one 8 KiB contiguous run per row → **16 runs of 512 B at stride 1024 words**". At the *Mosaic op*
level that is not visible: the enqueue is the same `2048xi32 → 2048xi32` copy in both arms, with the
row index folded into a `memref_slice`. The 16×512 B strided descriptor is implied only by the
memref's *tiling*, which `tpu-infer-memref-layout` assigns (mirrored in
`jax/_src/pallas/mosaic/tpu_info.py:565-580`: 2-D i32 → `Tiling.COMPACT` `(8,128)`), matching XLA's
`T(8,128)`. So Stage E's byte-level model is consistent with what lowers, but **it is a model, not
something this probe observed**; 512 B = 128 int32 = the SC granule remains a modelled claim. The
strided-DMA cost stays `unmodelled` — and it is now the only `unmodelled` term left in `v11`.

⚠ `v11`'s prize is still conditional on `stage-e-boundary.md:118-135` (does the retile survive
in-model?). Nothing here touches that bet. This probe only removes the feasibility term.

**U2 and `v11` are independent and compose**: the `u2` ENTRY still has the `ROOT %reshape.4`
(4 MiB), and the `v11` ENTRY still has the `%bitcast-convert_bitcast_fusion` (16 MiB).

## Did the negative controls fail as they should? **Yes, both.**

| control | result |
| --- | --- |
| `jnp.sin` on SC (probe 12's, `compile_check.py:157-180`) | **FAILS** — `NotImplementedError: Unimplemented primitive in Pallas TPU lowering for sc_vector_subcore: sin` |
| 8 MiB VMEM scratch, AOT (`aot_controls.py:36`) | **FAILS** — `E3000: CompileTimeSparseCoreAllocationFailure: … 33554432 words … upper bound (2097151 words) in memory space spmem` |
| `C0` incumbent (i32 in, i32 scratch) | LOWERS — the harness itself is sound |

The second control is the one that matters for the HLO diffs. `.compile()` runs against a
`topologies.get_topology_desc(topology_name="tpu7x:2x2x1")` device with **no TPU attached**; the
control proves that path really invokes the **SparseCore Mosaic/MLO backend** (the error is a
SparseCore `spmem` allocator error), so "AOT COMPILED ok" means the backend accepted the kernel, not
just that the MLIR verifier did.

## Files

`compile_check.py`/`.txt` — 8 toy SC variants at production dims, extends
`probes/12/compile_check.py`'s device-pinning prologue verbatim (`:30-39`).
`hlo_diff.py`/`.txt` + `hlo/` — AOT compile of the **real** kernel, three ways; `_gen_{base,u2,v11}.py`
are the generated patched copies (string patches on a copy; `src/…/sparsecore_topk.py` is untouched).
`dma_shape.py`/`.txt` — Mosaic out-DMA dump. `aot_controls.py`/`.txt` — AOT negative control +
2-D `ref.bitcast`.

Repro: `uv_venv/bin/python <script>` from this directory. jax 0.10.2, libtpu 0.0.44.1, no device.
