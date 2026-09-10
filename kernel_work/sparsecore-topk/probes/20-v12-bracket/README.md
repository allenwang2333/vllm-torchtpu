# probe 20 — is `v12`'s deleted op inside the program `P3.prefillX` times?

Task 2a. **No pod time.** AOT compile only (`aot_entry.py`), following the "AOT-compile SparseCore
Pallas with no TPU" path: optimized HLO in seconds for zero cluster time.

## Answer

**INSIDE. At every shape and at both entry ranks.**

`probes/18`'s `P3.*` arms time exactly

```python
jax.jit(lambda s, l: sparsecore_topk(s.reshape(b, n), k=k, row_lengths=l))   # s : f32[b, n/128, 128]
```

I recompiled that program verbatim against the pinned subject (`44353b34…`) and against `v12`
(`408ef72e…`), and read the optimized ENTRY back:

| cell | incumbent priced ops | `v12` priced ops | temp bytes incumbent → v12 | Δ temp |
| --- | --- | --- | --- | --- |
| **3-D entry, b=1024** (`P3.prefillX`) | `bitcast-convert`, 66712 cyc, 16 MiB scoped | **none** | 50,397,120 → 8,454,080 | **−41,943,040 = exactly 40 MiB = b·n·4** |
| 2-D entry, b=1024 (`P2.*`) | `fusion` `kind=kLoop`, 66712 cyc, 16 MiB scoped | **none** | 84,017,024 → 50,462,592 | −33,554,432 |
| **3-D entry, b=64** (`D3.decodeX`) | `bitcast-convert`, 5318 cyc, 5 MiB scoped | **none** | 3,211,200 → 589,760 | **−2,621,440 = exactly 2.5 MiB = b·n·4** |

The `Δ temp` column is an independent confirmation from `compiled.memory_analysis()`, not from
counting instructions: the deleted buffer is exactly the score array, to the byte, at both batch
sizes.

**Opcodes, not substrings — and the correction cuts both ways.** In the 2-D dumps `v12` shipped,
the deleted instruction is `%bitcast-convert_bitcast_fusion`, whose **opcode is `fusion`**
(`kind=kLoop`); there is **no `bitcast-convert` opcode anywhere in either fresh 2-D ENTRY**, so the
builder's "`bitcast-convert` 1 → 0" is false as stated and the coordinator's flag is correct. But at
the **3-D production entry the opcode genuinely is `bitcast-convert`, and it genuinely goes 1 → 0** —
the builder's sentence is wrong for the layout it dumped and right for the layout it did not. Full
opcode histograms are in `aot_entry.json`; the ENTRY texts are in `entry-*.txt`.

Materialization was tested separately from opcode identity: an instruction counts as *priced* only
if its `backend_config` carries a `window_config` with `estimated_cycles > 0`. `%bitcast.1`
(f32[1024,80,128] → f32[10485760]) has none — it is a free reinterpretation, which is precisely why
`v12`'s replacement costs nothing. The incumbent's `%bitcast_convert_type.3` has 66712 cycles and
16 MiB of scoped memory: a real TensorCore pass over the whole score buffer.

## What this means for the numbers

**1. `v12`'s 26.251 µs is quotable against `probes/18`'s measured production denominators.**

| denominator | value | 26.251 µs reads as |
| --- | --- | --- |
| `P3.prefillX` (ρ = 0.40005, exact, production 3-D entry) | 275.185 µs | **−9.54%** |
| gate-corrected per-invocation (ρ = 0.50005) | 291.075 µs | **−9.02%** |
| `P3.rho1` (what every prior probe ran) | 425.366 µs | −6.17% |
| the sheet's modelled 155.82 µs | 155.82 µs | −16.85% ← nearly double, do not use |

**2. The ρ fork validates itself here.** `v12`'s deletion is dense over the full `b × n` buffer and
**ρ-independent** — `row_lengths` never enters it. So it takes the ×1.55 branch of probe 18's fork,
and it does: −9.54% / −6.17% = **1.546**, against the predicted 425.366/275.185 = 1.5457. The two
were derived independently.

**3. Decode is where the price is not yet measured.** The 3-D b=64 cell deletes 2.5 MiB read +
2.5 MiB rewrite. Two estimates: by the cost model's own cycle ratio, 5318/66712 × 26.251 =
**2.09 µs**; at TC HBM peak (~3367 GB/s implied by the 2138 GB/s SparseCore read figure being 63.5%
of it), 5.24 MB / 3367 GB/s = **1.56 µs**. Against `D3.decodeX` = 49.171 µs that is **−3.2% to
−4.3%**. Both are MODELLED; the ranking job measures it. The same roofline check on prefill gives
83.9 MB / 3367 GB/s = **24.9 µs** against `v12`'s 26.251 µs — 5% apart, so the prefill price is
roofline-consistent and is not a cost-model artifact.

**4. The relayout term is a different op and remains additive.** The 3-D ENTRY contains **no**
`sparse-core-data-format-call`; the 2-D ENTRY contains one (`OFFLOAD_DATA_FORMATTING`, an extra
call-start/call-done pair and 33.5 MiB of temp). That call is what probe 18 measured as
`P2 − P3 = 35.576 µs`, and it is disjoint from the `bitcast-convert` that `v12` deletes. Both
statements from probe 18 survive: `T_production = T_harness − 35.563 µs`, and `v12`'s prize is
inside `T_production`.

## Consequences for the ranking harness

- **The production 3-D whole-call bracket scores `v12` correctly.** `v12` does *not* need the 2-D
  harness entry to be visible — it needed a bracket outside the `pl.kernel` invocation, and the
  `P3.*` whole-call bracket already is one. An inside-kernel bracket (`--decomp` / `--passes`) reads
  it as a flat null.
- **Activation tell confirmed at production rank.** The 3-D ENTRY names the SparseCore call
  `%sc_topk_direct_b1024_n10240_k2048_f32in.3.cloned.1.call-start`. The `_f32in` suffix is present in
  the 3-D program, so the scope-name check works without falling back to result-shape matching.
- **A second, free activation tell:** `temp_size_in_bytes` differs by exactly `b·n·4`. That is
  readable from `compiled.memory_analysis()` with no profile at all.

## PROBING.md five fields

**probe** — recompile probe 18's exact `P3` jitted program against both sources and diff the
optimized ENTRY by *opcode*, with materialization judged by `window_config.estimated_cycles > 0`
rather than by name. Cross-check every deletion against `memory_analysis().temp_size_in_bytes`.

**config** — `uv_venv/bin/python aot_entry.py`, no TPU: `jax._src.pallas.mosaic.tpu_info` monkeypatched
to TPU7x and `topologies.get_topology_desc("tpu7x:2x2x1")` for the device. jax 0.10.2. Both source
files md5-gated in-process before import (`MD5_GATE_OK`), and `resident sparsecore_topk modules`
asserted to be the two distinct arm modules, so no import shadowing.

**job** — none. Zero cluster time.

**answer** — INSIDE, at 3-D b=1024, 2-D b=1024 and 3-D b=64; Δtemp exactly `b·n·4` at both batch
sizes.

**decided** — `v12` is scored against 275.185 µs (**−9.54%**) or the gate-corrected 291.075 µs
(**−9.02%**), *not* against 155.82 µs; its decode price is MODELLED at 1.56–2.09 µs (−3.2% to −4.3%)
and must be measured. The `v12`-shaped hole in the harness is closed: the whole-call bracket at the
production 3-D entry is sufficient.

⚠ **Not settled here.** `v12` is a *bit* change, not a value change, but AOT HLO says nothing about
numerical equivalence in the kernel — the `f32`-typed VMEM slice holding int32 monotone-key bits is a
correctness question for the fan-out, not for this probe.
