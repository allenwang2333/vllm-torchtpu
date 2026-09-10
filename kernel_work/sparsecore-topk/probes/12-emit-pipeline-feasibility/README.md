# Probe 12 — can `pltpu.emit_pipeline` be used inside the SparseCore top-k kernel?

**VERDICT: `emit_pipeline` is VIABLE on SparseCore. Not a contingency, not a maybe — it is already
shipping in this repo inside a `plsc.VectorSubcoreMesh` kernel on the production MoE path, and a
toy SC kernel using it lowers cleanly to Mosaic here on CPU.** The recorded fallback (hand-rolled
`BufferedRef` + explicit semaphores) is not needed. Close the OPEN risk.

Two things the design has *not* recorded do change the v4 plan; see "Blockers not in the design".
⚠ Blocker 1 as originally written was wrong at prefill and has been corrected in place by the main
thread, with the re-derivation shown; the correction is a phase split, not a retraction of the verdict.

Evidence classes below are tagged **[read]**, **[compiled]**, **[inferred]**.

---

## Q1 — `emit_pipeline`: existence, path, signature

**[read]** Public export: `jax.experimental.pallas.tpu.emit_pipeline`
(`uv_venv/lib/python3.12/site-packages/jax/experimental/pallas/tpu.py:40`), jax/jaxlib 0.10.2.

Definition: `jax/_src/pallas/mosaic/pipeline.py:1835`

```python
def emit_pipeline(body, *, grid: tuple[int | jax.Array, ...], in_specs=(), out_specs=(),
                  tiling: Tiling | None = None, core_axis: tuple[int, ...] | int | None = None,
                  core_axis_name: tuple[str, ...] | str | None = None,
                  dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None,
                  trace_scopes: bool = True, no_pipelining: bool = False,
                  _explicit_indices: bool = False)
```

`pipeline.py:1854` reads `config.use_emit_pipeline_primitive`, which is **`False`** by default
**[compiled]** — so the live path is `_emit_pipeline` at `pipeline.py:1579`, whose docstring
(`:1594-1621`) says: *"the same semantics as pallas_call but is meant to be called inside
pallas_call for nesting grids"*, `grid` = "a pallas grid definition", `in_specs`/`out_specs` =
"input/output pallas block specs". `pipeline.py:1623` requires each grid entry to be a Python `int`
or a `jax.Array` — **dynamic (traced) grid bounds are explicitly allowed**; only *static*
non-positive entries are rejected (`pipeline.py:1849`). Block shapes must be static.

## Q2 — usable inside a SparseCore kernel body? Yes.

- **Primitives emitted.** `_emit_pipeline` builds `BufferedRef`s (`pipeline.py:535-634`) that
  allocate a VMEM window plus `SemaphoreType.DMA((buffer_count,))` (`:623`, `:628`; default
  `buffer_count=2`, i.e. double buffering, `:636`) and copy with `tpu_primitives.make_async_copy`
  (`pipeline.py:866, 887, 902, 921`) — the *same* call the incumbent kernel already uses at
  `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py:109`.
- **Lowering rules exist for SparseCore.** `dma_start_p` → `sc_lowering.py:428`, `dma_wait_p` →
  `sc_lowering.py:533`. `sc_lowering.py:62` registers its rules into the *shared* registry
  (`lowering.py:632`, keyed by `CoreType`), so `run_scoped`, `semaphore_signal/wait`, `cond`,
  `while`, `scan` are all present for `SC_VECTOR_SUBCORE` too **[compiled]** — enumerated by
  diffing `lowering_rules[CoreType.TC]` (127 rules) against `lowering_rules[SC_VECTOR_SUBCORE]`
  (67). The 60 TC-only rules are transcendentals, `dot_general`, RNG, reductions — **nothing
  `emit_pipeline` emits**.
- **No TensorCore-only memory-space assumption.** `pipeline.py:605` has an explicit SparseCore
  branch: for a 1-D block shape it flattens the window to `(buffer_count * block,)` *unless*
  `tiling is Tiling.SPARSE_CORE` (`tpu_info.py:567`). There is no `(sublane, lane)` requirement on
  the 1-D path. The repo's shipped SC pipelines pass no `tiling=` at all.
- **Enclosing grid.** With `core_axis`/`core_axis_name` left `None`, `_emit_pipeline` never calls
  `program_id`/`num_programs` on the *outer* kernel (those appear only under `core_axis`,
  `pipeline.py:1444-1445`). `emit_pipeline` supplies its *own* grid, so `pl.program_id(0)` inside
  the body refers to the pipeline. A `lax.fori_loop` in a `plsc.parallel_loop`-style mesh kernel is
  therefore irrelevant to it — there is no outer Pallas grid to be compatible with.

## Q3 — existing use of pipelining inside a SparseCore Pallas kernel: yes, several, shipped

**[read]** In this checkout, all inside `plsc.VectorSubcoreMesh` kernels:

| file:line | construct |
| --- | --- |
| `src/vllm_torchtpu/kernels/sparse_core/ragged_gather_v2.py:182` and `:208` | **nested** `pltpu.emit_pipeline`, outer grid `(num_blocks,)` *dynamic*, inner `(num_row_subchunks, num_cols)`; mesh at `:265` |
| `src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_v2/kernel.py:101` and `:311` | `emit_pipeline` whose `in_specs` index maps read a **persistent VMEM scratch** (`refs.scratch.sorted_by_validity_vmem`) |
| `src/vllm_torchtpu/kernels/sparse_core/ragged_gather.py:80`, `ragged_gather_reduce.py:139`, `ragged_gather_reduce_v3.py:112,146`, `ragged_scatter.py:248` | `functools.partial(pltpu.emit_pipeline, ...)` |

`ragged_gather_v2` is on the production MoE path (`src/vllm_torchtpu/layers/common/fused_moe_gmm.py:238`)
and has TPU tests (`tests/kernels/ragged_gather_v2_test.py:53`). `ragged_gather_v2.py:208` also
proves a **dynamic, possibly-zero** grid works: `num_blocks` is `jnp.where(end <= start, 0, ...)`
at `:102`.

None of these use `BufferedRef` directly, `emit_pipeline_with_allocations`, or manual
two-buffer/two-semaphore code. `emit_pipeline` is the idiom.

## Q4 — actual compile **[compiled]**

Script: `compile_check.py`. Verbatim output: `compile_check.txt`. No TPU on this host
(`jax.devices() == [CpuDevice(id=0)]`; the `tpu` backend fails with
`NOT_FOUND: TPU initialization failed: No jellyfish device found`).

Two device queries had to be stubbed, both pure lookups, neither part of the lowering rules:
`tpu_info.get_tpu_info()` pinned to `get_tpu_info_for_chip(ChipVersion.TPU_7X, 1)` (a table entry
jax itself ships, `tpu_info.py:526`), and `tpu_info.is_tpu_device()` forced `True` to lift the
device gate at `jax/_src/pallas/mosaic/pallas_call_registration.py:618`
(`ValueError: SparseCore kernels are only supported on TPU, but the current device is cpu.`).
Lowering was then forced with `jax.jit(f).trace(x).lower(lowering_platforms=("tpu",))`.

| variant | shape | result |
| --- | --- | --- |
| **C** control | incumbent: one synchronous `make_async_copy` + `plsc.parallel_loop` | **lower: OK** (4048 chars) |
| **A** | `emit_pipeline` HBM→HBM, grid `(8,)`, 1-D `BlockSpec((512,), lambda i: (i,))` | **lower: OK** (10194 chars) |
| **B** | **the v4 shape**: `emit_pipeline` HBM → persistent `pltpu.VMEM((4096,))` scratch (body writes `resident_vmem[pl.ds(i*BLOCK, BLOCK)]`), then one manual DMA out | **lower: OK** (8746 chars) |
| **D** negative control | same as C but with `jnp.sin` in the loop body | **LOWER FAILED**, as intended |

Variant D's verbatim traceback tail — this is what proves A/B/C actually reached the SparseCore
lowering rules rather than being short-circuited:

```
  File ".../jax/_src/pallas/mosaic/sc_primitives.py", line 804, in _parallel_loop_lowering_rule
    carry_out = tc_lowering.jaxpr_subcomp(
  File ".../jax/_src/pallas/mosaic/lowering.py", line 1839, in jaxpr_subcomp
    raise NotImplementedError(
NotImplementedError: Unimplemented primitive in Pallas TPU lowering for sc_vector_subcore: sin.
```

Caveat **[inferred]**: this exercises Pallas→Mosaic MLIR generation and the Mosaic verifier. It does
**not** exercise the Mosaic→LLO backend compile or execution. Nothing here can rule out a
register/VMEM-pressure failure at LLO time; only a TPU job can.

---

## Blockers the design has NOT recorded

1. ⚠ **Corrected by the main thread — the wave count splits by phase, and the original claim here
   ("`_sc_topk_direct` almost always sees `b_eff <= 32` → one wave") is true at decode and FALSE at
   prefill.** `_pick_partition`'s `max_p = 32 // b if b < 32 else 1` (`sparsecore_topk.py:504`) caps
   **`P`**, not the wave count; the two are different quantities and were conflated. Re-derived from
   the source:

   | phase | `b` into `sparsecore_topk` | `P` | `b_eff` into `_sc_topk_direct` | `num_waves` (`:443`) |
   | --- | ---: | ---: | ---: | ---: |
   | **prefill** | 1024 | **1** (`p_min = cdiv(10240, 65536) = 1`; `b >= 32` so `max_p = 1`) | 1024 | **32** |
   | **decode** | 16 (the token bucket, floor 16) | **2** (`max_p = 32//16 = 2`; `5120 >= k` and `>= LANES`) | 32 | **1** |

   So the consequence is a **phase split**, not a blocker:

   - **At prefill — 32 waves — the caller's cross-wave `emit_pipeline` is exactly the right shape**,
     and this probe clears it. Prefill is also the phase the whole `design.md` denominator
     (189.98 µs) is measured in, so `v4`'s recorded prize is not affected by this row.
   - **At decode — one wave — cross-wave pipelining buys literally zero.** The `fori_loop` at `:329`
     has a single iteration and there is no next wave's DMA to overlap. Winning at decode needs the
     **within-slice** form — variant B's shape: `grid=(n // chunk,)`, body converts to monotone keys
     and does the L0 histogram scatter straight off the pipeline buffer while writing into
     `resident_vmem`, overlapping the HBM read with the pass that today only starts after `dma(...)`
     returns at `:169`.

   The within-slice form subsumes the cross-wave one and is the better build for that reason, not
   because the cross-wave one is dead.
2. **Only the *first* pass over the slice can be pipelined.** **[read]** The algorithm is
   multi-pass — `cvt` (`:171`), `hist0_body` (`:187`), the three `REFINEMENT_LEVELS`, then
   `emit_body` (`:283`) all re-read `resident_vmem`. Passes 2..N read VMEM, not HBM, so
   `emit_pipeline` has nothing to hide there. The upper bound on v4 is "the score DMA disappears
   behind pass 1", not "the score DMA disappears".
3. **Static block shape is fine, but the *index map* constrains `chunk`.** **[read]** `slice_len`
   (`:442`) is static (`_align_to(n, LANES)`, and `n % LANES == 0` is enforced at `:546`, so
   `slice_len == n`), so the "varies per wave" worry does not apply — the per-wave dynamic quantity
   is `i_hi` (`:164`), the *useful* length, and today's DMA already copies the full static
   `slice_len` regardless. But a `BlockSpec` index map returns **block** indices, and the kernel
   flattens scores to 1-D (`:486`, `words.reshape(-1)`), so the source offset `row_c * n` must be a
   multiple of `chunk`. Either require `chunk | n` (all production `n` are powers of two in
   2048..256K, so any power-of-two chunk works) or pass the scores as 2-D `(b, n)` with a
   `(1, chunk)` block.
4. **The `pl.when(i_hi > 0)` gate around the DMA (`:166`) has to become a dynamic grid.**
   **[inferred, with precedent]** Wrapping `emit_pipeline` in `pl.when` puts its `run_scoped`
   allocations under a conditional. The idiom already in the repo is instead to pass a grid that
   evaluates to `0` — `ragged_gather_v2.py:102,208`.
5. **VMEM cost.** **[read]** Double buffering adds `2 * chunk * 4` bytes on top of the existing
   `resident_vmem` (up to `MAX_SLICE_WORDS = 64K` words = 256 KB, `sparsecore_topk.py:45`) against a
   512 KB budget (`SparseCoreInfo.vmem_capacity_bytes = 524288`). Comfortable for `chunk <= 8K`
   words, but it is a real subtraction from the headroom `_pick_partition` is sizing against.

## Not needed

The hand-rolled `BufferedRef` / explicit-semaphore fallback. Nothing in the repo does it, and
`emit_pipeline` is what the four shipped SparseCore kernels use.
