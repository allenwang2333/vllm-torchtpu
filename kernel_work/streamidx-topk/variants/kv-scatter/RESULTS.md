# The indexer KV-cache write: killing the `T(4,128)` → `T(8,128)` round trip

**Status: route (a) passes the HLO pass condition — the two `reshape`s become `bitcast`s — and is
worth −513.2 µs per prefill KV write end to end, at a cost of +26.2 µs per decode write.**

Getting there took three attempts at the one hard part. Making the cache 128-minor moves the
problem into the `StreamIdxTC` kernel's block load, which must then separate value sub-rows from
scale sub-rows. The obvious lowering (`index`) costs more at decode than the whole win
(+1053.7 µs); the obvious fix (`stride`) does not compile. The third (`split`) changes *where the
caller puts the bytes* so both reads are contiguous, and reduces the decode cost 54× to +19.4 µs.
Sections 6 and 6.1 are that search; section 6.2 is the three-way comparison.

Everything below was measured on one 4-chip `tpu7x` pod through `run-command-in-cdk`. No
end-to-end model benchmark was run and nothing here belongs in the experiment log.

---

## 0. Terms, and how to read every number in this file

Read this section before any table. Several of the columns are easy to misread as something they
are not.

### The finding this work is about

`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py:104` writes the indexer KV cache:

```python
cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)
```

`cache_kv` is `u8[415, 256, 4, 256]` — 108.8 MB — and XLA gives it layout
`{3,2,1,0:T(4,128)(4,1)}`. The scatter wants a 2-D operand, and `u8[424960,256]` is
`T(8,128)(4,1)`: a *different tile height*. So the flatten is not free. At prefill it compiles to
a three-op triad — relayout in, scatter, relayout out — with no consumer for either relayout. It
is a round trip.

### Vocabulary

| Term | Meaning |
| --- | --- |
| **triad** | The three ops `reshape → scatter_fusion → reshape` that the `.at[].set()` compiles to at prefill. Named for the production profile's finding; used here as a measurement bucket (below). |
| **relayout** | A `reshape` that XLA cannot satisfy by reinterpreting bytes, so it materialises a new buffer. Visible in the emitted HLO as an op named `reshape` carrying a non-zero `scoped_memory_configs` size. |
| **bitcast** | A `reshape` that XLA *can* satisfy by reinterpreting bytes. Free: no data movement, `scoped_memory_configs` size `0`. **Turning the two `reshape`s into `bitcast`s is this exercise's pass condition.** |
| **minor dim** | The last (fastest-varying) dimension. `[415,256,4,256]` is "256-minor"; `[415,256,8,128]` is "128-minor". |
| **`T(a,b)(c,1)`** | The TPU tiled layout. `(a,b)` is the tile in the two minor dims; `(c,1)` is sub-byte/sub-word packing on the sublane axis. For `u8`, `T(8,128)(4,1)` packs four logical rows into one physical sublane. The tile height `a` is chosen from the second-minor dim, which is why `4` vs `8` differs between the two cache shapes. |
| **`kv_packing`** | 4. How many tokens share one `[…, kv_packing, ·]` group. GLM-5.2 indexer geometry. |
| **`ROWS_PER_TOKEN`** | 2. In the 128-minor layout, a token's 256 bytes are two 128-byte rows instead of one 256-byte row. |
| **sub-row / row** | Index into the third axis of the cache, i.e. within one `(page, slot//kv_packing)` group. The base cache has 4 such rows of 256 B; the v128 cache has 8 of 128 B. |

### The arms

Six. Three run the **whole** operation (caller prologue + scatter + `StreamIdxTC` + SparseCore
top-k); three run the **scatter only**, isolating the triad from everything else.

| Arm | Cache shape | Kernel | What it is |
| --- | --- | --- | --- |
| `base` | `[415,256,4,256]` | production | **The incumbent.** Reproduces production exactly. |
| `v128` | `[415,256,8,128]` | v128 | **Route (a).** Cache *allocated* 128-minor; kernel reads it 128-minor. |
| `v128rt` | `[415,256,4,256]` | v128 | Route (a) **as a round trip**: the production-shaped buffer is reshaped to 128-minor for the write and back afterwards. The control that answers "is this a pure view?". |
| `scat256` | `[415,256,4,256]` | — | `base`'s scatter, alone. |
| `scat128` | `[415,256,8,128]` | — | `v128`'s scatter, alone. |
| `scat128rt` | `[415,256,4,256]` | — | `v128rt`'s scatter, alone. |

`v128rt` / `scat128rt` exist for one reason: if the 4-D→4-D reshape were free, the round-trip arm
would cost the same as `v128`. It does not. That is section 4's evidence.

Every 128-minor arm additionally has a **sub-row placement**, set by the `KVS_DEINTERLEAVE`
environment variable and shared by the caller, the kernel and the input builder:

| Placement | Where a token's two sub-rows go | Byte order vs production |
| --- | --- | --- |
| `index`, `stride` | **interleaved** — rows `2k` and `2k+1` | identical; a plain reshape |
| `split` | **grouped** — value rows `0..kv_packing-1`, scale rows `kv_packing..2*kv_packing-1` | a permutation |

Sections 2-5 are the interleaved placement. Section 6.1 is `split`. Where a table does not say,
it is interleaved.

### The configs

| Config | Tokens written | Sequences | Context | Stands for |
| --- | --- | --- | --- | --- |
| `pre1024` | 1024 | 1 | 9216 | **Prefill.** The production shape the finding was measured at. |
| `dec64` | 64 (padded) | 64 | 8704 | **Decode.** The regression risk: at decode the production triad has *no* reshapes at all. |

### The metric

**`hlo_stats` device self time, TensorCore rows only, in microseconds.** Column `Avg. self time
(us)` from the xprof `hlo_stats` table, summed per bucket.

- **Never a wall clock.** Four shards share the pod, so wall clock is contended. `bench_kvs_trace.py`
  emits it under the key `wall_us_per_call_CONTENDED_NOT_A_RESULT` and it is not reported here.
- **No host dispatch floor is subtracted**, anywhere.
- **SparseCore is a separate plane** and is reported in its own column, never added to a
  TensorCore total.
- Every arm is gated on two checks, reported as `checks: ok`:
  `|sum(buckets) − TensorCore total| < 1e-6` (the buckets partition the total, nothing double- or
  un-counted) and `rows_kept == rows_total` (no row silently dropped).

### The buckets

Each arm's TensorCore self time is partitioned into exactly these, disjoint and exhaustive:

| Column | Contents |
| --- | --- |
| **full µs** | The whole arm: every TensorCore row. The sum of the four columns that follow. |
| **stage A µs** | The `StreamIdxTC` Pallas kernel — the consumer that reads the cache. **This is where a 128-minor cache costs something.** |
| **stage B µs** | The rest of the fused indexer work, unaffected by any arm here (it is flat to 0.15 µs across all arms, which is itself a check that the arms differ only where intended). |
| **triad µs** | Every op matching `reshape|scatter|^copy|bitcast|transpose`. **This is the bucket route (a) targets.** It is carved out of `residual`, so it is wider than the three named triad ops — sub-microsecond `copy-start`/`copy-done` pairs land here too. Section 3 gives the per-op breakdown so the three ops that matter are visible on their own. |
| **residual µs** | Everything else on the TensorCore. |
| **SC µs** | SparseCore plane, reported and never summed into the above. |

`Δ` columns are against the arm named in the **vs** column: full arms against `base`, scatter-only
arms against `scat256`. Negative is faster.

---

## 1. What ran

| | |
| --- | --- |
| Checkout | `/home/annyan_google_com/workspaces/copy1/vllm-torchtpu`, branch `annyan/streamidx-topk-opt` |
| Image | `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-hning_google_com:ec4bcecf784d0bc3aeb8be37554431e3b5eef70b` |
| `~/cloud-devkit` | `c4e1d4e`, dirty (1 untracked file: `recipes/experimental/gdn-comparison.yaml`) |

| Job | Status | Shards | What it was for |
| --- | --- | --- | --- |
| `j-e4b8eeb2-6272-4387-8aea` | **deleted before running** | — | Wedged in `PodInitializing` ~50 min on a node with a broken `cdk-outputs` CSI mount. Deleted (my own job) and resubmitted. No data. |
| `j-8ddefa9d-a60d-412b-b4c9` | **Succeeded** | 4/4, all rc=0 | **Every number in sections 2-5.** |
| `j-53493769-f689-413f-bc7d` | **Failed** | 0/4, all rc=1 | The `KVS_DEINTERLEAVE=stride` lowering. One root cause, quoted in section 9.2. No usable data. |
| `j-21eeaf78-9de7-47c7-bb93` | **Failed** | 3/4, `correct` rc=1 | First `KVS_DEINTERLEAVE=split` attempt. Failed on a harness gap, not the kernel; **its timings are void** and are not reported. Section 9.3. |
| `j-3391639d-1e5b-4c3d-84a8` | **Succeeded** | 4/4, all rc=0 | **Every number in section 6.1.** The `split` lowering with the harness fixed. |

### The stale-image guard

The image ships a **different** `streamindex_topk.py` and **no** `sparsecore_topk.py` at all. Every
job maps the checkout's `src/` over the image's and every probe refuses to run on an md5 mismatch.
Read back on-pod in `j-8ddefa9d`:

```
1e221f10cbf3573aa0d2ae78c3463532  /root/.../kernels/deepseek_v4/streamindex_topk.py
8bc9e25b8701a53a97704564b9fbc658  /root/.../kernels/deepseek_v4/sparsecore_topk.py
a305fdd8451c6429cbcf67abb83ed311  /root/.../layers/vllm/custom_ops/mla_attention_op.py
EXPECT streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532
EXPECT sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658
[kvs-hlo] MD5_READBACK_OK=True
```

### The caller is the real one

`kvs_op.py` delegates to `variants/option-b-runner/runner_op.py`, which carries a verbatim copy of
the production `_streamidx_topk_jax` prologue. Checked mechanically each run:

```
[kvs-hlo] FIDELITY real_caller=/root/.../custom_ops/mla_attention_op.py
[kvs-hlo] FIDELITY prologue_identical=True prologue_lines=23
```

`cache_kv` is donated (`donate_argnums=(2,)`) in every timed arm, as production donates it.

---

## 2. The pass condition: does `reshape` become `bitcast`?

**Yes, at prefill, for the arms whose cache is *allocated* 128-minor.** Quoted verbatim from the
emitted optimized HLO (`hlo/<config>__<arm>.hlo.txt`, from
`jax.jit(fn).lower(*args).compile().as_text()`).

### `scat256` — the incumbent. Two real relayouts.

```
%reshape.234 = u8[424960,256]{1,0:T(8,128)(4,1)} reshape(%cache_kv.1), backend_config={... "scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"33554432"}] ...}
ROOT %scatter.1 = u8[424960,256]{1,0:T(8,128)(4,1)} scatter(%param_0.9, %transpose.33, %transpose.34), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%region_2.14
ROOT %reshape.236 = u8[415,256,4,256]{3,2,1,0:T(4,128)(4,1)} reshape(%scatter_fusion), metadata={op_name="jit(_streamidx_topk_jax_scatter_only)/scatter" ...}
```

Note the layouts on the two sides of `%reshape.234`: `T(4,128)(4,1)` in, `T(8,128)(4,1)` out. That
tile-height change is the whole cost. `scoped_memory_configs` size `33554432` is the scratch it
needs — a bitcast asks for `0`.

### `scat128` — route (a). Both relayouts are gone.

```
%bitcast = u8[849920,128]{1,0:T(8,128)(4,1)} bitcast(%cache_kv.1), backend_config={"flag_configs":[],"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"0"}],"used_scoped_memory_configs":[]}
ROOT %scatter.1 = u8[849920,128]{1,0:T(8,128)(4,1)} scatter(%param_0.9, %transpose.34, %transpose.35), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%region_2.14
ROOT %bitcast.71 = u8[415,256,8,128]{3,2,1,0:T(8,128)(4,1)} bitcast(%scatter_fusion), metadata={op_name="jit(_streamidx_topk_jax)/scatter" ...}
```

Both ops are now `bitcast`, both with `scoped_memory_configs` size `0`. The 4-D cache and its 2-D
scatter view are both `T(8,128)(4,1)` — same tile — so the flatten reinterprets bytes.
**This is the pass condition, met.**

### `scat128rt` — the round trip. Still relayouts.

```
%reshape.173 = u8[849920,128]{1,0:T(8,128)(4,1)} reshape(%cache_kv.1), ... "size":"33554432" ...
ROOT %reshape.13 = u8[415,256,4,256]{3,2,1,0:T(4,128)(4,1)} reshape(%scatter_fusion), metadata={op_name="jit(_streamidx_topk_jax)/reshape" ...}
```

Buying the 128-minor *view* of a 256-minor *buffer* buys nothing: the relayout simply moves to the
view boundary.

### Relayout counts, all twelve (config, arm) pairs

Count of ops in the emitted HLO with `opcode: reshape` acting on a **108,789,760-element** buffer,
i.e. on the cache itself. (Free `bitcast`s and smaller unrelated reshapes are excluded; where the
count is 0 for a 128-minor arm at prefill there are exactly 2 cache `bitcast`s in their place.)

| config | `base` | `v128` | `v128rt` | `scat256` | `scat128` | `scat128rt` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre1024` | 2 | **0** | 2 | 2 | **0** | 2 |
| `dec64` | **0** | **0** | 2 | **0** | **0** | 2 |

At decode the incumbent already has none — matching the production profile, where the decode
scatter is a bare `scatter_fusion` at 8.87 µs. There is nothing for route (a) to win at decode,
only something to lose. And `v128rt` *introduces* two relayouts at decode where the incumbent has
none.

---

## 3. Prefill and decode

`hlo_stats` device self time, µs, TensorCore only. Every arm `checks: ok`.

### 3.1 Summary, both configs

| config | arm | vs | full µs | Δfull | stage A µs | ΔA | stage B µs | triad µs | Δtriad | residual µs | SC µs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre1024 | `base` | `base` | 1731.763 | +0.000 | 247.774 | +0.000 | 431.831 | 801.195 | +0.000 | 250.962 | 430.565 |
| pre1024 | `v128` | `base` | 1332.031 | **−399.732** | 352.648 | **+104.874** | 431.721 | 297.055 | **−504.141** | 250.607 | 430.797 |
| pre1024 | `v128rt` | `base` | 1905.264 | +173.501 | 247.507 | −0.267 | 431.806 | 973.689 | +172.493 | 252.262 | 430.778 |
| pre1024 | `scat256` | `scat256` | 919.475 | +0.000 | — | — | — | 757.761 | +0.000 | 161.714 | — |
| pre1024 | `scat128` | `scat256` | 418.917 | **−500.558** | — | — | — | 256.543 | **−501.218** | 162.373 | — |
| pre1024 | `scat128rt` | `scat256` | 1093.885 | +174.410 | — | — | — | 931.553 | +173.792 | 162.331 | — |
| dec64 | `base` | `base` | 182.844 | +0.000 | 111.766 | +0.000 | 32.808 | 15.411 | +0.000 | 22.859 | 31.036 |
| dec64 | `v128` | `base` | 1243.185 | **+1060.341** | 1165.505 | **+1053.739** | 32.687 | 22.556 | +7.145 | 22.437 | 31.148 |
| dec64 | `v128rt` | `base` | 875.232 | +692.387 | 111.717 | −0.049 | 32.700 | 701.498 | +686.087 | 29.317 | 31.131 |
| dec64 | `scat256` | `scat256` | 16.678 | +0.000 | — | — | — | 9.589 | +0.000 | 7.089 | — |
| dec64 | `scat128` | `scat256` | 24.026 | +7.348 | — | — | — | 16.827 | +7.237 | 7.199 | — |
| dec64 | `scat128rt` | `scat256` | 702.134 | +685.456 | — | — | — | 694.949 | +685.360 | 7.185 | — |

Three things to take from this table.

1. **The scatter-side win is real and it is large.** `scat128` vs `scat256` at prefill:
   **−500.6 µs**, isolated from the kernel entirely.
2. **The kernel-side cost is real too, and at decode it dominates.** `v128` stage A is +104.9 µs at
   prefill and **+1053.7 µs** at decode — a 10.4× blow-up of the `StreamIdxTC` kernel. End to end
   `v128` is −399.7 µs at prefill and **+1060.3 µs at decode**. As measured, route (a) is a
   prefill win and a much larger decode loss.
3. **`residual` and `stage B` are flat** across all arms at each config (≤1.6 µs and ≤0.15 µs of
   spread). The arms differ where they were meant to and nowhere else.

### 3.2 The three triad ops, on their own

The `triad` bucket is wider than the three ops of the finding; here are those three, per arm, so
the comparison is like-for-like with the production profile.

**Prefill (`pre1024`, 1024 rows written):**

| arm | forward reshape | scatter | backward reshape | three-op total |
| --- | ---: | ---: | ---: | ---: |
| production profile (`j-9b23bb32`, rank 8) | 349.03 | 135.11 | 290.43 | **774.57** |
| `base` | `reshape.304` 333.427 | `scatter_fusion` 133.741 | `reshape.306` 290.605 | **757.77** |
| `scat256` | `reshape.234` 332.939 | `scatter_fusion` 133.554 | `reshape.236` 289.042 | **755.54** |
| `scat128` | — (bitcast, free) | `scatter_fusion` 253.287 | — (bitcast, free) | **253.29** |
| `v128` | — (bitcast, free) | `scatter_fusion` 253.083 | — (bitcast, free) | **253.08** |
| `scat128rt` | `reshape.173` 344.618 | `scatter_fusion` 250.980 | `reshape.13` 332.592 | **928.19** |

The `base`/`scat256` reproduction agrees with the production profile to **2.5%** on the three-op
total (755.5-757.8 µs here vs 774.6 µs there), which is the evidence that this microbenchmark is
measuring the thing the profile found.

**Note the scatter itself gets slower**: 133.6 µs → 253.3 µs, **+119.7 µs**. Twice as many rows
(2048 × 128 B instead of 1024 × 256 B) for the same bytes, and the scatter cost is per row. So the
`scat128` win is not "the reshapes were free money" — it is 621.98 µs of relayout removed minus
119.73 µs of extra scatter = 502.25 µs, which is the −500.6 µs in the summary table (the ~1.7 µs
difference is the sub-microsecond `copy`/`bitcast` chaff also in the bucket).

**Decode (`dec64`, 64 padded tokens):**

| arm | forward reshape | scatter | backward reshape | three-op total |
| --- | ---: | ---: | ---: | ---: |
| production profile (`j-9b23bb32`, rank 8) | — | 8.87 | — | **8.87** |
| `scat256` | — | `scatter_fusion` 8.729 | — | **8.73** |
| `scat128` | — | `scatter_fusion` 15.520 | — | **15.52** |
| `scat128rt` | `reshape.7` 345.007 | `scatter_fusion` 15.415 | `reshape.13` 333.205 | **693.63** |

`scat256` reproduces the production decode number to **1.6%** (8.73 vs 8.87 µs). At decode route
(a) has no relayout to remove and only the doubled row count to pay: **+6.8 µs on the scatter**.
That is small. The decode problem is not the scatter, it is stage A.

---

## 4. Is route (a) a pure view change? No.

The brief asked whether `[415,256,4,256]` and `[415,256,8,128]` are the same bytes in the same
order, and said to verify rather than assume. Both halves were checked, and they give **different
answers**, which is the crux of this section.

### 4.1 The byte order is identical — verified, not assumed

`probe_kvs_hlo.py --mode layout`, three independent checks:

```
LAYOUT index_map_a_k_d_equals_b_2k_plus_d_div_128 = True
LAYOUT device_reshape_matches_numpy = True
LAYOUT device_reshape_round_trips  = True
```

The first is elementwise over the full array: `a[p,s,k,d] == b[p,s,2k+d//128,d%128]` for every
index. The logical byte order really is the same. Additionally, `build_inputs` constructs the v128
cache as a `.reshape()` of the *same* base array, so the two arms provably start byte-identical —
confirmed at both configs by `inputs_identical: 0/108789760 differing bytes`.

### 4.2 But the *physical layout* is not, and that is what costs

| Shape | Layout XLA assigns |
| --- | --- |
| `u8[415,256,4,256]` | `T(4,128)(4,1)` |
| `u8[415,256,8,128]` | `T(8,128)(4,1)` |
| `u8[424960,256]` | `T(8,128)` |
| `u8[849920,128]` | `T(8,128)` |

The tile height comes from the second-minor dim: `4` for the base cache, `8` for the v128 cache.
So `[415,256,4,256]` → `[415,256,8,128]` changes the tiling, and the standalone reshape between
them, compiled on its own at production size, is a **real relayout**:

```
%v.1 = u8[415,256,4,256]{3,2,1,0:T(4,128)(4,1)} parameter(0)
ROOT %reshape.1 = u8[415,256,8,128]{3,2,1,0:T(8,128)(4,1)} reshape(%v.1), ... "scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"33554432"}] ...
```

**Conclusion: route (a) is a pure view change *of the scatter*, but not a free reinterpretation of
an existing buffer.** The win requires the cache to be **allocated** `[415,256,8,128]` by vLLM's
cache engine. It cannot be bolted on at the call site. `v128rt` is the arm that tries exactly that,
and it is the worst arm in the table at both configs (+173.5 µs prefill, +692.4 µs decode) — the
cost does not disappear, it relocates and duplicates.

---

## 5. Correctness (the `index` lowering)

`probe_kvs_correct.py`, both configs, non-donating (so buffers survive for comparison). This
section is `j-8ddefa9d`, i.e. the interleaved (`index`) placement; the same checks for `split` are
in section 6.1 and also pass with zero failures.

```
SUMMARY failed_checks=0 []
```

| Check | What it proves | `pre1024` | `dec64` |
| --- | --- | --- | --- |
| `inputs_identical` | The two arms start from the same bytes | `array_equal=True`, 0/108789760 differ | same |
| `bytes/scat128` | The 128-minor scatter lands bytes in the same places, read back through the **original** `[415,256,4,256]` view | `np.array_equal=True` | `True` |
| `bytes/scat128rt` | Same, for the round-trip form | `np.array_equal=True` | `True` |
| `crosscheck/bytes` | A `v128`-written cache, reshaped back, is byte-identical to a `base`-written one | `True` | `True` |
| `crosscheck/v128_written_then_prod_kernel` | The **production** `StreamIdxTC` reads the same top-k out of a `v128`-written cache | `array_equal=True`, `[1024,2048]` | `True`, `[64,2048]` |
| `kernel/v128` | The whole v128 arm (128-minor cache **and** 128-minor kernel) returns the same top-k, and leaves the same cache bytes | top-k and cache both `array_equal=True` | both `True` |
| `kernel/v128rt` | Same for the round-trip arm | both `True` | both `True` |

Top-k indices are compared **exactly**, not as sets — a per-row set comparison is reported
alongside so a pure tie-break difference would be distinguishable from a real one. It did not
arise: exact equality held everywhere.

This is the strong form of the correctness question the brief asked: bytes land in the same
places, *and* the production kernel reads the same values out of a `v128`-written cache, *and* the
v128 kernel reads the same values out too.

---

## 6. The kernel-side problem, and the two lowerings that do not solve it

Route (a)'s cost is entirely in `load_bkv` in `StreamIdxTC`. The reason is layout, not arithmetic.

In the base cache a token's row is 256 bytes: the 128 fp8 value bytes are lanes 0-127 and the
scale byte is lane 128. Both are **lane** slices of one row — cheap. In the v128 cache the value
bytes are sub-row `2k` and the scale is lane 0 of sub-row `2k+1`. Separating them is a
**stride-2 sub-row read**, and under `T(8,128)(4,1)` four logical rows are packed into one physical
sublane, so rows {0,2,4,6} straddle both packed sublanes of the group. That is the expensive shape.

Three lowerings were written, selected by the `KVS_DEINTERLEAVE` env var, all in
`streamindex_topk_v128.py::load_bkv`:

| Lowering | Idea | Outcome |
| --- | --- | --- |
| `index` | Load the 8-row block, pick sub-rows in registers | **Compiles, measured, too slow.** +104.9 µs prefill / +1053.7 µs decode in stage A. This is the number in section 3. |
| `stride` | Ask the VMEM ref for the strided sub-rows directly (`pl.Slice(0, kv_packing, 2)`) | **Does not compile.** Mosaic has no strided load for 8-bit data. Verbatim in section 9.2. |
| `split` | Change *where the caller puts the bytes* — all value sub-rows in the first `kv_packing` rows of the group, all scale sub-rows in the last — so each read is a contiguous sub-row run and lands on exactly one packed sublane | **Compiles, measured, works.** +6.8 µs prefill / +19.4 µs decode. Section 6.1. |

`split` is not a view change; it **permutes the cache byte order**. `kvs_common.base_to_v128` and
`kvs_common.v128_to_base` are placement-aware and exact inverses, so every byte-equality check
below stays valid under it rather than having to be skipped — which matters, because the one thing
that went wrong with `split` was a placement the harness applied inconsistently (section 9.3).

### 6.1 `split` works, and it is what makes route (a) landable

Measured in `j-3391639d-1e5b-4c3d-84a8` (Succeeded, 4/4 shards rc=0). Same metric, same buckets,
same gates — every arm `checks: ok`.

| config | arm | vs | full µs | Δfull | stage A µs | ΔA | stage B µs | triad µs | Δtriad | residual µs | SC µs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre1024 | `base` | `base` | 1730.767 | +0.000 | 247.798 | +0.000 | 431.491 | 800.475 | +0.000 | 251.003 | 430.467 |
| pre1024 | `v128` | `base` | 1217.568 | **−513.199** | 254.623 | **+6.824** | 431.689 | 280.833 | −519.642 | 250.423 | 430.346 |
| dec64 | `base` | `base` | 182.484 | +0.000 | 111.550 | +0.000 | 32.758 | 15.365 | +0.000 | 22.811 | 31.010 |
| dec64 | `v128` | `base` | 208.708 | **+26.224** | 130.990 | **+19.440** | 32.582 | 21.909 | +6.544 | 23.228 | 31.127 |

**The de-interleave cost collapses.** Stage A goes from `index`'s +104.9 µs to **+6.8 µs** at
prefill, and from **+1053.7 µs to +19.4 µs** at decode — a 54× reduction in the decode regression.
That is the packed-sublane argument confirmed: reading rows 0-3 and rows 4-7 of a
`T(8,128)(4,1)` group is two whole-sublane reads, where rows {0,2,4,6} is a shuffle.

**The pass condition still holds.** Under `split`, `pre1024__scat128` and `pre1024__v128` have
**0** cache `reshape`s and **2** cache `bitcast`s each; `pre1024__base` and `pre1024__scat256`
have 2 cache `reshape`s and 0 bitcasts. Changing where the caller puts the bytes did not cost the
bitcast.

Per-op, prefill: `base` = `reshape.304` 333.346 + `scatter_fusion` 133.348 + `reshape.306` 290.314
(**757.01 µs**); `v128` = `scatter_fusion` **237.915 µs** and no reshapes. The split scatter is
also slightly cheaper than the interleaved one (237.9 vs 253.1 µs) — its updates are two
contiguous runs rather than an interleave. Decode: `base` `scatter_fusion` 8.772 µs, `v128`
15.276 µs.

**Correctness: `SUMMARY failed_checks=0 []`.** Because `base_to_v128` / `v128_to_base` are exact
inverses, the byte-equality checks stay valid under the permuted placement rather than having to
be skipped. At both configs: `bytes/scat128` `array_equal=True` (0/108789760 differ),
`crosscheck/bytes` `True`, `crosscheck/v128_written_then_prod_kernel` top-k `array_equal=True`,
`kernel/v128` top-k and cache both `True`.

The round-trip arms are **dropped** under `split`, and recorded as dropped
(`round_trip_arms_meaningful: false`): they reach the 128-minor form by a *reshape*, and under
`split` that form is a permutation, so they would write this step's tokens in one placement into a
buffer whose history is in the other. Incoherent by construction, not a result.

### 6.2 Where the three lowerings leave route (a)

| | prefill Δfull | prefill ΔstageA | decode Δfull | decode ΔstageA |
| --- | ---: | ---: | ---: | ---: |
| `index` | −399.732 | +104.874 | **+1060.341** | +1053.739 |
| `stride` | — does not compile — | | | |
| `split` | **−513.199** | +6.824 | **+26.224** | +19.440 |

`split` is the only one of the three that is a net win at prefill *and* a small enough decode cost
to be worth arguing about. **It is not free at decode**: +26.2 µs per write, ×22 writes = **+0.58 ms
per decode step**. This exercise did not measure a decode step's TC-busy time, so what fraction
that is remains unknown and must be measured before anything is landed on it. Most of it is stage
A (+19.4 µs), i.e. the `StreamIdxTC` kernel is still 17.4% slower at decode reading a 128-minor
cache; the scatter's own share is only +6.5 µs.

**`split` is not a view change of any kind.** It permutes the cache byte order, so it is a change
to the *storage format* of the indexer KV cache, not merely to its shape. Everything that reads or
writes that cache has to agree. That raises the landing cost above route (a)-with-`index` — but
route (a)-with-`index` is not landable at all, so the comparison is moot.

---

## 7. Route (b): doing the write inside the Pallas kernel — feasibility

Analysis only; not built. The brief permits that.

Route (b) replaces `cache_kv.at[...].set(k_packed)` with a write performed inside a Pallas kernel,
so no XLA-level scatter — and therefore no 2-D flatten and no relayout — exists at all. It is
feasible and it is strictly more general than route (a), but it is a substantially larger change
and it does not obviously beat route (a) on the numbers measured here.

**What makes it feasible.** The write is a pure scatter of `n_tok` rows into an HBM buffer at
indices the caller already has as arrays (`page`, `slot`). Pallas on TPU supports exactly this
shape through `PrefetchScalarGridSpec`: put `page` and `slot // kv_packing` in the scalar-prefetch
operands, give `cache_kv` an HBM `BlockSpec` whose `index_map` reads those scalars, and DMA one
token's row per grid step from a VMEM-resident `k_packed`. The indexer already runs a
`PrefetchScalarGridSpec` kernel over this same cache with this same page table, so the machinery
and the index arithmetic are both already present in `streamindex_topk.py`. The write is also
naturally fusable into that kernel rather than being a second one, which is the version worth
building: it would remove the separate `scatter_fusion` op too, not just the reshapes.

**What makes it awkward.** Three things. First, the grid: one step per token means 1024 grid steps
at prefill with a 256-byte DMA each, which is far below the DMA size where a TPU is efficient;
the tokens are not contiguous in the cache, so they cannot be batched into larger DMAs without
first sorting by page, and consecutive tokens of one sequence *are* contiguous within a page, so a
run-length pass could recover most of that — at the cost of a caller-side sort and a dynamic grid.
Second, donation and aliasing: the cache must be input-output aliased through the `pallas_call`
for the write to be in-place, which is expressible but interacts with the fact that the same
buffer is an input to the read side of the fused kernel — a read-write hazard the current split
into "scatter then read" avoids for free. Third, and most important, it does **not** by itself fix
the thing that actually blocks route (a): if the cache stays `[415,256,4,256]`, a Pallas write
avoids the relayout, but if the cache moves to `[415,256,8,128]` to help anything else, the
`load_bkv` de-interleave cost of section 6 is still there. Route (b) and the kernel-side problem
are independent.

**Where it would land on the numbers.** Route (b) against the incumbent targets the full 757.8 µs
prefill triad rather than route (a)'s 500.6 µs, because it removes the 133.6 µs scatter as well and
does not pay route (a)'s +119.7 µs row-count penalty. Its ceiling is therefore roughly 1.5× route
(a)'s — but only if the small-DMA grid does not eat the difference, which is precisely the thing
this analysis cannot predict and a microbenchmark would settle in one job. It is the better
*eventual* answer and the worse *next* step: route (a) reaches a measured −513.2 µs per prefill
write with a five-line kernel edit and no new kernel at all, and — unlike route (b) — it is
already measured rather than modelled.

One thing route (b) *would* fix that route (a) does not: route (a) with `split` changes the
cache's storage format, and every reader has to be updated. A Pallas write against the unchanged
`[415,256,4,256]` cache changes nothing anyone else sees. If step 4 of section 8 turns out to be
expensive, that is the argument for reopening route (b).

---

## 8. What this is worth, and what it would take to land

**Worth, measured:** route (a) with the `split` placement nets **513.2 µs per prefill KV write**
end to end (`v128` vs `base`, `hlo_stats` device self time) and costs **26.2 µs per decode write**.
Isolating the scatter, the interleaved placement nets 500.6 µs per prefill write — 622.0 µs of
relayout removed, minus 119.7 µs of extra scatter from writing twice as many, half-width rows.

The production profile counts 22 such writes per prefill step, so the ceiling is **≈11.3 ms per
prefill step** (22 × 513.2 µs), against an 863.24 ms TC-busy prefill step: **≈1.31%**. The decode
cost is **≈+0.58 ms per decode step** (22 × 26.2 µs) against a decode step whose TC-busy time this
exercise did not measure.

Three cautions, none of which this work removes. **40.4% of the prefill step is already collective
time**, so part of an 11.3 ms saving may be absorbed rather than realised. The production
per-write figures rest on **one** prefill step per capture (two captures, agreeing to 0.03%). And
the 1.31% is a device-time ceiling measured on an isolated microbenchmark at the production shape
— it is not an end-to-end throughput prediction and must not be quoted as one.

**What it would take to land**, in dependency order:

1. **`src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`** — `load_bkv`, taking the
   `split` form of section 6.1. This is the substantive kernel change and the one this exercise
   spent three attempts on; the working version is in `streamindex_topk_v128.py` under
   `DEINTERLEAVE == "split"`, five lines.
2. **The cache allocation** — wherever vLLM's cache engine sizes the indexer KV cache, so it is
   created `[415,256,8,128]`. Section 4 is the evidence that this cannot be a call-site reshape.
   In this repo that is the indexer cache-shape logic reached from the MLA attention backend's
   `get_kv_cache_shape` path; the exact file was not chased down as part of this exercise and
   should be confirmed rather than assumed.
3. **`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py:104`** — the scatter itself,
   replaced by the `split` branch of `_v128_scatter` in `kvs_op.py`.
4. **Every other reader or writer of that cache.** Under `split` this is a change to the cache's
   *storage format*, not just its shape, so this step is larger than it would be for a pure shape
   change and is the main landing risk. Not surveyed here; `mla_attention_op.py` and
   `streamindex_topk.py` are the two this exercise touched. In particular anything that
   saves, loads, transfers or prefix-caches this buffer has to agree on the new placement.
5. **Measure a decode step.** The +0.58 ms/step decode cost is real and its significance is
   currently unknown. It should be sized before the change is landed, not after.

**Nothing under `src/` was modified.** All work is in
`kernel_work/streamidx-topk/variants/kv-scatter/`.

---

## 9. Everything that failed, verbatim

### 9.1 `j-e4b8eeb2-6272-4387-8aea` — wedged in `PodInitializing`, deleted

Scheduled onto node `gke-tpu-d716ab26-7jh8` and looped in `PodInitializing` for ~50 minutes. From
`cdk job desc`:

```
failed to stat ".../volumes/kubernetes.io~csi/cdk-outputs/mount": transport endpoint is not connected
MountVolume.SetUp failed for volume "cdk-outputs" : ... the sidecar container terminated due to Error, exit code: 255
```

A broken CSI mount on that node, not a fault of the job. Deleted (my own job, per Rule 5) and
resubmitted; the replacement landed on `gke-tpu-d716ab26-6804` and ran clean. No data lost.

### 9.2 `j-53493769-f689-413f-bc7d` — the `stride` de-interleave does not compile

`JobStatus=Failed`, `tpu-fanout: 0/4 shards succeeded`, all four shards rc=1, all with the same
root cause:

```
MosaicError: INTERNAL: Mosaic failed to compile TPU kernel with MLIR module name `<no name>`
HLO name `StreamIdxTC-d-bq_1-bkvp_2.2`: not implemented: Strided load with non 32-bit data

The MLIR operation involved:
  %223 = "tpu.strided_load"(%arg12, %222, %24, %24, %24, %24)
    <{strides = array<i32: 1, 1, 1, 2, 1>}>
    : (memref<2x4x512x8x128xi8, #tpu.tiled<(8,128)(4,1),[2048,512,1,1,1]>,
       #tpu.memory_space<vmem>>, index, index, index, index, index)
    -> vector<1x1x512x4x128xi8>
```

`tpu.strided_load` is not implemented for 8-bit element types on this libtpu/jax. Note the
`strides = [1,1,1,2,1]` — the `2` is the stride-2 sub-row read. The lowering is unavailable, not
slow; there is no number to report for it.

### 9.3 `j-21eeaf78-9de7-47c7-bb93` — the first `split` attempt, void on a harness gap

`JobStatus=Failed`, 3/4 shards succeeded; the `correct` shard returned rc=1 with every top-k row
differing at both configs:

```
[kvs-correct] TOPK pre1024/kernel/v128: np.array_equal=False {'array_equal': False, 'shape': [1024, 2048], 'differing_rows': 1024, 'total_rows': 1024, 'rows_with_identical_set': 0, 'first_differing_row': 0}
[kvs-correct] TOPK dec64/kernel/v128: np.array_equal=False {'array_equal': False, 'shape': [64, 2048], 'differing_rows': 64, 'total_rows': 64, 'rows_with_identical_set': 0, 'first_differing_row': 0}
[kvs-correct] SUMMARY failed_checks=4 ['pre1024/kernel/v128', 'pre1024/kernel/v128rt', 'dec64/kernel/v128', 'dec64/kernel/v128rt']
```

**Not a kernel bug.** `build_inputs` produced the v128 cache with `cache_kv.reshape(V128_CACHE_SHAPE)`
— the *interleaved* placement. The scatter writes only this step's 64 or 1024 tokens; the other
~8700 tokens per sequence are pre-existing KV history that the kernel also reads, and under
`split` the kernel expects that history in the split placement too. So the arm was a correct
kernel reading a cache it could not interpret. Fixed by `base_to_v128` / `v128_to_base` in
`kvs_common.py`, which are placement-aware and exact inverses.

**Its timings are void and are not reported anywhere in this file**, for a reason worth recording:
they *looked* plausible (prefill Δfull −377.8 µs, decode Δfull +33.0 µs) but stage B rose from
431.6 to 568.6 µs and the SparseCore plane from 430.4 to 567.7 µs — buckets the arm cannot touch.
Garbage scores change how much work `sparsecore_topk` and the kernel's early exit do, so the whole
measurement was data-dependent. After the fix, stage B is 431.7 vs 431.5 and SparseCore 430.3 vs
430.5 — flat, as they are for every other arm. **A timing run whose correctness check failed is
not a slightly-wrong number; it is not a number.** The tell was a bucket moving that had no
mechanism to move.

### 9.4 Tooling notes, for the next session

- `cdk job log --no-color` — `unknown flag: --no-color`. `cdk job recipe` does accept it.
- `cdk queue` does not exist (`unknown command "queue"`). Use `cdk usage -o json` and `cdk job desc`.
- `pkill -f <pattern>` inside a Bash tool call will match the tool's **own** command string and kill
  the shell (exit 144). Verify with `ps -ef | grep -c "[p]attern"` instead.
- Two concurrent trace shards writing one `trace_meta.json` clobber each other; `kvs_shard.sh`
  writes to `$OUT/trace/s$TPU_FANOUT_SHARD`.
- `jax.profiler.trace` SIGSEGVs on the image's pinned jax 0.10.2 against libtpu 0.0.44.1
  (`PLUGIN_Profiler_Api size: expected 80, got 104`). The `trace` step builds a side venv with
  jax/jaxlib 0.11.0; the `hlo` and `correct` steps, which need no profiler, stay on 0.10.2.

---

## 10. Files

| File | What it is |
| --- | --- |
| `kvs_common.py` | Geometry at production cache size, the two configs, the arm/layout tables, md5 gating, input construction, and the placement-aware `base_to_v128` / `v128_to_base` pair (exact inverses — section 9.3 is what happens when they are not used). |
| `kvs_op.py` | The `base` and `v128` callers. Delegates the prologue to `option-b-runner/runner_op.py` (fidelity-checked against the real `mla_attention_op.py`). Holds `_v128_scatter`. |
| `streamindex_topk_v128.py` | Copy of the production kernel with five marked edits for a 128-minor cache. `KVS_DEINTERLEAVE` selects the `load_bkv` lowering. |
| `probe_kvs_hlo.py` | `--mode layout` (section 4.1) and `--mode hlo` (sections 2 and 4.2). Compiler only; no timing. |
| `probe_kvs_correct.py` | Section 5. |
| `bench_kvs_trace.py` | One `*.xplane.pb` per (config, arm). |
| `read_kvs_hlo_stats.py` | Off-pod reader: `hlo_stats` → the bucketed table of section 3. |
| `kvs_shard.sh` | The fan-out shard entry point: `hlo`, `correct`, `trace`. |

**Static compile-time analysis** (`--mode hlo`, `--mode layout`) and **contended fan-out wall
clock** are both present in the artifacts and neither is a benchmark result. Nothing in this file
goes to `/log-experiment`.
