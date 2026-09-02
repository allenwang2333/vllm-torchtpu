# P0 — does `donate_argnums=(2,)` elide `args[2].copy_(new_cache)`?

**VERDICT: ELIDED.** No full KV-cache copy exists anywhere in the executed program. The 21 indexer
K caches are entry-computation parameters that are **input/output aliased**, and the only
instruction that writes them is a **16-row in-place scatter costing 3.02 µs each** — not a
189 MiB copy costing ~67-134 µs each.

Everything below is read from **job `j-a71a648c-3064-471a-a131`, rank 0** (AgentX,
`max_model_len=262144`, page 256, TP16/DP1, padded batch 16, decode, 36 module executions × 2
cores). **Every µs in this document is AgentX.** Instruction counts and structure carry to the
nightly config; µs do not.

Artifacts used (both already on this machine, no job was run):

| artifact | path |
| --- | --- |
| xplane | `/tmp/prof_a71/rank0_j-a71a648c-3064-471a-a131-ray-node-0-0.xplane.pb` |
| `xprof-dump` JSON (has `tools.hlo_stats`, 16554/16554 rows, untruncated) | `/tmp/prof_a71/rank0_hlo.json` |
| main-module `HloProto` | `/tmp/prof_a71/tt_jit_layernorm_L62C0_fused_add_rms_norm_mul(17643488860325271500).hlo_proto.pb` |
| rendered HLO text (48 MB, produced for this probe) | `/tmp/p0_hlo.txt` |

The HLO text was produced by stripping the `HloProto` wrapper (field 1 = `hlo_module`, 16 794 853 B;
field 3 = `buffer_assignment`, 3 064 122 B) and feeding the `HloModuleProto` to
`jaxlib.xla_client.XlaComputation(...).as_hlo_text()`. Reproducible; the module bytes are cached at
`/tmp/p0_hlo_module.pb`.

---

## 1. The four checks the probe asked for

### (1) Inside the HLO — no copy of the cache shape exists

`measured (j-a71a648c rank 0, /tmp/p0_hlo.txt, whole-module opcode scan on the literal
"3027,64,4,256")`. Every instruction in the module — all computations, not just `ENTRY` — whose
shape is the indexer cache:

```
custom-call        63     (the StreamIdxTC / topk pallas calls taking it as an operand)
parameter          42     (21 in ENTRY + 21 inside the fused computations)
scatter            21     (inside fused_computation.348, .352, .355, ...)
get-tuple-element  21
fusion             21     (the 21 `scatter_fusion*` in ENTRY)
copy / copy-start / copy-done / bitcast :  0
```

**Zero `copy`, `copy-start`, `copy-done` or `bitcast` of the cache shape in the whole module.**

### (2) Input/output aliasing — declared, for all 21, `may-alias`

`measured (/tmp/p0_hlo.txt line 1, module header)`. The module header carries
`input_output_alias={...}` with **177 entries**. Filtering `ENTRY main.1741_spmd`'s 2070 parameters
by shape gives exactly 21 with `u8[3027,64,4,256]{3,2,1,0:T(4,128)(4,1)}` — the indexer caches
(§6.1 of `report.md` established 21 indexer layers independently) — and **all 21 are aliased**:

```
param   22 (param.242)  -> out{0}    may-alias
param   50 (param.287)  -> out{4}
param   74 (param.311)  -> out{7}
param  173 (param.410)  -> out{16}
param  279 ... 1975 (param.2212) -> out{169}      [21 in total, all may-alias]
```

The other 156 aliased params are the MLA caches (`u8[3027,64,4,128]` ×78 and
`u8[3027,256,4,128]` ×78). Same mechanism, same result.

The `ROOT tuple` of `ENTRY` (line 146626, 179 operands) feeds those aliased outputs directly from
the scatter fusions — `out{0} <- scatter_fusion`, `out{4} <- scatter_fusion.3`,
`out{169} <- scatter_fusion.168`. There is no intervening copy, and XLA's copy-insertion pass did
**not** un-donate: had the buffer been unavailable, a `copy` of the operand would sit in front of
each scatter, and check (1) shows there is none.

The scatter fusion itself, verbatim (`/tmp/p0_hlo.txt` line 125257, and the same text in the
`hlo_stats` `HLO op text` column):

```
%scatter_fusion = u8[3027,64,4,256]{3,2,1,0:T(4,128)(4,1)}
    fusion(u8[3027,64,4,256]{3,2,1,0:T(4,128)(4,1)} %param.242,
           s32[16]{0:T(128)S(1)}   %get-tuple-element.2879,
           u8[16,256]{1,0:T(8,128)(4,1)S(1)} %pad.159),
    kind=kCustom, calls=%fused_computation.348
```

Framework op name: `custom_kernel/custom_kernel/jit(_streamidx_topk_jax)/scatter:` — i.e. the
`cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)` at
`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py:103-104`. The operands say it
outright: **16 scatter indices (`s32[16]`) and 16 padded rows (`u8[16,256]` = 4096 B)** written
into a 189 MiB buffer.

### (2b) The measurement — the arithmetic that makes this airtight

`measured (j-a71a648c rank 0, /tmp/prof_a71/rank0_hlo.json, tools.hlo_stats, columns
"Avg. self time (us)" and "HBM BW (GiB/s)")`. 21 rows selected by
`HLO op name startswith "scatter_fusion"` **and** `"3027,64,4,256" in HLO op text`; all 21 are
TensorCore and all 21 carry the `jit(_streamidx_topk_jax)/scatter:` framework scope.

| | value |
| --- | ---: |
| rows | 21 |
| `#Occurrences` each | 72 (36 steps × 2 cores) or 70 |
| **avg self time, per instruction** | **3.024 µs** (range 2.867 – 3.121; the one non-cache-shaped `scatter_fusion` row at 1.849 µs is a different instruction) |
| **sum over 21, per core-step** | **63.50 µs** |
| mean `HBM BW (GiB/s)` | 2.525 |

Now the discriminator. `modelled`:

- Cache size: `3027 × 64 × 4 × 256 = 198,377,472 B = 189.19 MiB`.
- A real copy moves read+write: `396,754,944 B`.
- The profile's own `peak_hbm_bw_gigabytes_per_second` for this core is **3686.16 GB/s**
  (`/tmp/prof_a71/rank0_hlo.json`, `structure.planes[0].stats`). A 396.75 MB copy therefore cannot
  finish in less than `396,754,944 / 3.68616e12 = 107.6 µs`; even a write-only interpretation
  needs 53.8 µs. At the brief's 2.97e12 B/s it is 133.6 µs (or 66.8 µs one-way — the brief's
  ≈67 µs).
- Measured: **3.024 µs**. To move 396.75 MB in 3.024 µs would need **131 TB/s, 36× above this
  chip's peak HBM bandwidth.** It did not happen.
- What *did* move, from the measured BW: `3.024e-6 s × 2.525 GiB/s × 1.0737e9 B/GiB = 8197 B`.
  Predicted for a 16-row in-place scatter: `16 × 256 B × 2 (read+write) = 8192 B`. **Agreement to
  0.06%.**

Two independent lines — the static instruction inventory and the measured time against a hardware
bound — give the same answer.

### (3) Above the HLO — nowhere for a framework copy to hide

`measured (/tmp/prof_a71/rank0_hlo.json, structure.planes)`. The device planes carry only
`XLA Modules`, `XLA Ops`, `Async XLA Ops`, `XLA TraceMe`, `Scalar Unit`, `TC Overlay`,
`Pallas Primitives`, `_counters_`, and the SparseCore lines. **There is no memcpy / D2D transfer
line**; on this capture every device-side byte movement is an HLO op, and check (1) already swept
all of them.

The whole decode step is **one** XLA module (`17643488860325271500`, 2070 parameters, 179 outputs,
78 layers). The KV caches enter it as parameters and leave as aliased outputs, so there is no
boundary at which a torch-level buffer copy could be inserted. `hlo_stats` does contain seven other
program ids (`as_strided` ×2, `compile_mlir_as_strided`, `uniform_` ×2, `token_scatter`,
`sample_from_logits_argmax`) totalling 189 rows; **none of them contains the string `3027`** and
their largest ops are the sampler's `bf16[8,9680]` fusion and all-gather. No KV-cache traffic
outside the main module.

### (4) From source — the mechanism, corroborating

`read from source`:

- `mla_attention_op.py:126-128` — `pallas.jax_op(op_name, _streamidx_topk_jax, donate_argnums=(2,))`.
- `torch_tpu/_internal/pallas/pallas.py:898-906` — `jax_op` forwards `donate_argnums` to
  `custom_jax_kernel`.
- `pallas.py:804-819` — which puts it on `jax.jit(..., donate_argnums=..., keep_unused=True)` and
  onto the `JaxCallable`.
- `pallas.py:684-690` — `JaxCallable.__call__` passes `donate_argnums=self.donate_argnums` into
  `tpu_torch_pallas.call_custom_kernel(...)`.
- `pallas.py:692-693` — the *deprecated* `input_output_aliases` path is the one that does an
  explicit `tensor_args[in_idx].copy_(results[out_idx])`. This op does **not** use that path
  (`input_output_aliases` is `None`, so `self.input_output_aliases == {}` — `pallas.py:536-537`),
  so that loop is empty.
- `pallas.py:908-914` — `mutates_args = ()`, with the comment "*If we mark donated args as mutable
  then Dynamo will copy back the original value of the donated arg which then defeats the purpose
  of donating*". The torch custom op is therefore opaque and non-mutating, which is exactly why the
  wrapper at `mla_attention_op.py:138-141` has to do the `copy_` itself.
- `pallas.py:867-870` — the docstring tells the user to do precisely this: "*Consider using a
  wrapper function and `copy_` or `set_` to overwrite these invalid tensors to prevent accidental
  reuse.*"

So the source-level reading is: the `copy_` at line 140 is the sanctioned idiom, and in the traced
graph it becomes an input mutation on a graph input, which torch_tpu lowers into the
`input_output_alias` entry we measured. The value written is the scatter fusion, and XLA assigns
it the parameter's own buffer.

**One thing this artifact cannot separate**: whether the elision is *caused by* `donate_argnums`
or whether XLA's copy-insertion would have reached the same in-place assignment without it. That
needs an A/B compile with the flag removed, which is a hardware job and out of scope here. What the
artifact does settle — and what the probe actually asked operationally — is that **no full KV-cache
copy is executed, per layer or otherwise.**

---

## 2. Cost accounting

**Cost if NOT elided (the hypothesis being tested), `modelled`:** 21 × ~107.6 µs (r+w at this
chip's peak) = 2.26 ms/core-step, or 21 × 66.8 µs = 1.40 ms/core-step at the brief's one-way
figure.

**Actual cost, `measured`:** 21 × 3.024 µs = **63.50 µs/core-step**. The hypothesis overstates it
by 22–36×.

**Share of the ~964 µs/step/core residue: 6.6%** (63.50 / 964). Real, small, and it is *not* a
copy — it is the KV insertion the op has to do anyway. Worth one note for the designer: the 21
`scatter_fusion` instructions sit in `ENTRY`, **outside** the `conditional`, so the KV write runs
unconditionally even when `STREAMIDX_ENABLE_EARLY_EXIT` takes the cheap branch.

---

## 3. What is actually in the residue — this is the part that matters

`measured (j-a71a648c rank 0, tools.hlo_stats, "Avg. self time (us)" summed over distinct
instructions = per core-step; `#Occurrences` is 72 or 70 = 36 steps × 2 cores)`.

Reconstructing the briefer's bucket as `conditional` + framework scope `custom_kernel/custom_kernel/cond:`
+ framework scope `custom_kernel/custom_kernel/bitcast_convert_type:` gives **1254.63 µs/core-step**
against the reported 1.230 ms — the same bucket to within membership rounding. Its contents:

| item | µs/core-step (AgentX) | n | avg | what it is |
| --- | ---: | ---: | ---: | --- |
| **`copy f32[2,8,2048,128]` `{3,2,1,0}S(1)` → `{3,1,2,0}` HBM** | **496.16** | 21 | 23.63 | **the single largest plumbing item** |
| **`bitcast-convert s32[16,2048,128]` ← `f32[16,2048,128] %StreamIdxTC-m-…`** | **255.27** | 21 | 12.16 | a *semantically free* reinterpret, fully materialized |
| `conditional.*` self time | 178.08 | 21 | 8.48 | the early-exit `conditional` instruction itself |
| `async-done s8[4194304]` ← `s8[16777216]` (`slice-start`/`slice-done`) | 134.71 | 84 | 1.60 | 4 async slices per indexer layer |
| loop-fusion `bitcast_convert_type` (`s32[131072]`, `u8[16,512]`, `u8[16,64]`) | 81.81 | 177 | 0.46 | 44.41 of it is 21× `s32[131072]`; the rest is per-MLA-layer |
| `copy s32[2,64,8,128]` ×42, `copy s32[2,16,8,128]` ×21, `copy s32[1,16,4,128]` ×21 | 70.89 | 84 | 0.84 | smaller layout copies on the `cond` boundary |
| `loop fusion s32[16,8192]` and other `cond:` fusions | ~25.9 | 84 | 0.31 | |
| everything else under `cond:` (copy-start/done pairs, `custom-call s32[]`, tokens) | ~11.8 | ~750 | <0.01 | |

**The top three account for 929.5 µs — 96% of the ~964 µs residue.** Adding the 63.50 µs KV
scatter brings it to 993 µs, i.e. the residue is now fully named.

### The two items worth the designer's attention

Both are `measured`; the *identification* of the buffer is `modelled` from dimension arithmetic and
should be confirmed against the kernel source before acting on it.

1. **496.16 µs — a transposing 16 MiB copy of the score buffer, once per indexer layer.**
   `%copy.7018 = f32[2,8,2048,128]{3,1,2,0:T(8,128)} copy(f32[2,8,2048,128]{3,2,1,0:T(8,128)S(1)} %bitcast.6833)`
   — 21 of these, 23.627 µs each, dead flat (23.627–23.630). The shape is
   `2×8 = 16` sequences by `2048×128 = 262144 = max_model_len`, so this is the **full
   `f32[16,262144]` score buffer** §6.1 already identified, being moved out of `S(1)` and having
   dims 1 and 2 transposed on the way. `modelled`: 16,777,216 B, read+write 33.55 MB in 23.627 µs
   = 1.42 TB/s = 39% of the 3.686 TB/s peak — consistent with a strided/transposing copy.
   **352 MiB/core-step of pure data movement that computes nothing.**

2. **255.27 µs — a materialized bit-reinterpretation of the same buffer.**
   `%bitcast_convert_type.156 = s32[16,2048,128]{2,1,0:T(8,128)} bitcast-convert(f32[16,2048,128]{2,1,0:T(8,128)} %StreamIdxTC-m-bq_128-bkvp_2.42)`
   — 21 of these, 12.156 µs each. Same 16 MiB, same layout `{2,1,0}` on both sides, **same tiling**.
   This is `f32`→`s32` on identical bytes and should be free; XLA has materialized it as a full
   copy (33.55 MB r+w in 12.156 µs = 2.76 TB/s = 75% of peak). This looks like the most directly
   removable 255 µs in the bucket — it is `StreamIdxTC-m`'s output being re-typed for the top-k
   stage, and having the kernel emit `s32` (or the consumer accept `f32`) would delete it outright.

`STREAMIDX_ENABLE_EARLY_EXIT`'s conditional costs 178.08 µs of `conditional` self time plus the
70.89 + 134.71 µs of boundary copies and slices it forces — ~384 µs/core-step of overhead for a
guard that, at these context lengths, never takes the cheap branch. That is a third candidate.

**AgentX caveat, and it bites hardest here.** Items 1 and 2 are both linear in the score-buffer
width, which is `max_model_len` = 262144 on this job and **10240** on the nightly baseline of
record — 25.6× narrower. Naively rescaled these become ~19.4 µs and ~10.0 µs per core-step, i.e.
they would nearly vanish. Do not carry the 496 µs / 255 µs figures into a nightly-config decision;
carry the *structure* (one full-width transposing copy and one materialized bitcast per indexer
layer) and re-measure. §6.3's deconfounding run is the precedent: the `max_model_len` effect was
real but much smaller than the confounded comparison suggested.

---

## 4. Bookkeeping discrepancy the designer should know about

The second briefer's **265.7 µs** attributed to `lt`, `reduce_sum`, `select_n`, `dynamic_slice`,
`gather`, `slice`, `concatenate` reproduces exactly — I measure **268.91 µs** for the framework
scopes `custom_kernel/custom_kernel/cond/branch_0_fun/*` excluding the two `pallas_call` kernels:

```
  98.56 us n= 42  .../cond/branch_0_fun/slice:
  89.75 us n=105  .../cond/branch_0_fun/gather:
  40.21 us n= 42  .../cond/branch_0_fun/concatenate:
  24.35 us n= 21  .../cond/branch_0_fun/lt:
  13.45 us n= 21  .../cond/branch_0_fun/dynamic_slice:
   2.47 us n= 21  .../cond/branch_0_fun/reduce_sum:
   0.12 us n= 21  .../cond/branch_0_fun/select_n:
 268.91 us total
```

But those scopes are `.../cond/branch_0_fun/...`, **not** `.../cond:`, and they are not
`conditional` or `bitcast_convert` either. If the 1.230 ms bucket is what I reconstructed above
(1254.63 µs), then **the 265.7 µs was never inside it**, and "1230 − 265.7 = 964 unattributed" is
subtracting a disjoint set. The right statement is: the bucket is ~1255 µs and is now ~99%
itemized in §3, and the branch-0 metadata ops are a *separate* ~269 µs sitting inside the indexer's
conditional branch. The two together are ~1524 µs/core-step of non-kernel indexer overhead at
AgentX. This does not change any conclusion in §3 — the same three items dominate either way — but
someone summing buckets later will double-count or lose 269 µs if it is not flagged.

---

## 5. Provenance summary

| claim | tag |
| --- | --- |
| 21 indexer caches, all `may-alias`ed, param→output map | measured — j-a71a648c r0, HLO module header + `ENTRY main.1741_spmd` parameter list, `/tmp/p0_hlo.txt` |
| zero `copy`/`copy-start`/`copy-done`/`bitcast` of `u8[3027,64,4,256]` module-wide | measured — same file, opcode scan |
| scatter fusion operands `s32[16]` + `u8[16,256]` | measured — `hlo_stats` `HLO op text`, and `/tmp/p0_hlo.txt:125257` |
| 3.024 µs avg / 63.50 µs per core-step / 2.525 GiB/s | measured — `rank0_hlo.json` `tools.hlo_stats` |
| 107.6 µs floor for a 396.75 MB copy; 131 TB/s needed at 3.024 µs; 8197 B vs 8192 B | modelled — arithmetic shown in §1(2b), peak BW from the profile's own `device_info` |
| no device memcpy line; no other program touches the cache | measured — `rank0_hlo.json` `structure.planes`, and program-id partition of `hlo_stats` |
| `donate_argnums` plumbing through `pallas.py` | read from source — `torch_tpu/_internal/pallas/pallas.py:536, 684-693, 804-819, 898-914`; `mla_attention_op.py:103-104, 126-128, 138-141` |
| residue table in §3 | measured — `hlo_stats`, grouped by `Framework op name` + `HLO op category` + output shape |
| "`f32[2,8,2048,128]` is the `f32[16,262144]` score buffer" | modelled — `2×8=16`, `2048×128=262144=max_model_len`; consistent with §6.1's `gather_offload_async_done f32[16,262144]`. Not independently confirmed against kernel source. |
| whether removing `donate_argnums` would reintroduce a copy | **not determined** — needs an A/B compile |
