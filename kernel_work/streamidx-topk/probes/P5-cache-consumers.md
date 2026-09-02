# P5 — Does the indexer K cache have one consumer or two?

**Question.** The indexer K cache is allocated at `lkv_dim = align_to(129, 128) = 256` but only 129
bytes of every 256 are meaningful (128 fp8 key bytes + 1 e8m0 scale byte). Whether that padding is
reclaimable — by repacking the cache to 129, or by splitting it into a 128-byte array plus a
separate scale array the way MLA already splits nope/rope — depends entirely on **who else touches
that buffer**. If the main MLA attention path or `sc_dsa_gather` reads the same allocation, the
layout is a contract between two subsystems and cannot be changed from the indexer side alone.

Area 4's **D1 falsifier** (`briefs/area4-layout-precision.md` §D1) is the check. It has two routes.
Both are addressed below, and they get different answers.

---

## VERDICT

**`D1_VERDICT PASS`** — via **Route 0**, cross-checked against source.

The indexer K cache is a **separate allocation from every MLA-side KV buffer**, and its complete
set of touchers is **three instructions, all inside the indexer path**: two readers (the two shape
specializations of stage A) and one writer (the cache-update scatter). Neither `MLA-*` nor
`sc_dsa_gather` names it.

**Route 1 was NOT run, and should not be: as written it cannot discriminate.** See
[§4](#4-route-1-is-not-a-falsifier-do-not-run-it).

**Design decision settled:** repacking the indexer K cache is a change confined to
`streamindex_topk.py` and its one writer. It requires no MLA-side change, no `sc_dsa_gather`
change, and no coordination with the split-cache layout. The 127 bytes/row of padding are
structurally reclaimable.

**Design decision NOT settled:** whether reclaiming them *pays*. P5 says the padding is nobody
else's; **P6** says whether a narrow read is even charged less. If P6 says the DMA is charged for
the granule it lands in, then the only route to the saving is repacking the cache in HBM, which is
a much larger change than narrowing a slice.

---

## 1. Evidence: Route 0 — operand extraction from a real profile

Source: xprof profile of job **`j-a71a648c-3064-471a-a131`**, sessions `2026_09_02_10_53_40`
(primary, batch geometry 16) and `2026_09_02_10_50_26` (cross-check, batch geometry 1024/64), read
with `xprof-dumper dump <pb> --tools hlo_stats --top-n 20000 --trace-sample 0 --max-stat-len 20000`.

> **Extraction note, worth carrying forward.** `xprof-dumper dump` defaults to
> `--max-stat-len 512`, which truncates the `HLO op text` column *mid-operand-list* — the u8 KV
> operand sits at position 10 of `StreamIdxTC` and was silently being cut off. `--top-n` also
> defaults low (200 of 16554 rows). The flags above are required for any operand-level question.
> Both dumps ran clean: `tools.hlo_stats.status == "ok"`, `rows_kept == rows_total`.

### 1.1 Three distinct uint8 4-D KV buffers exist

`measured (j-a71a648c, session 10_53_40)` — complete inventory of 4-D u8 shapes:

| buffer | shape | bytes | who names it |
| --- | --- | --- | --- |
| **indexer K cache** | `u8[3027,64,4,256]` | 198,377,472 | `StreamIdxTC-d-bq_1-bkvp_2`, `StreamIdxTC-m-bq_128-bkvp_2`, `conditional.N`, `scatter_fusion` |
| MLA nope cache | `u8[3027,256,4,128]` | 396,754,944 | `scatter_fusion` only |
| MLA rope cache | `u8[3027,64,4,128]` | 99,188,736 | `scatter_fusion` only |

`u8[3027,64,4,256]` is the **only** shape in the whole profile with trailing geometry `[..., 4, 256]`,
and it is mentioned in 105 rows total, all four names above.

### 1.2 The MLA path reads different buffers, end to end

`measured (j-a71a648c, session 10_53_40)` — the flat forms and the chain:

```
u8[3027,256,4,128] = 396,754,944 B = u8[3099648,128]   ─┐
u8[3027,64,4,128]  =  99,188,736 B = u8[774912,128]    ─┴─> sc_dsa_gather operands
                                                            │
sc_dsa_gather outputs: u8[131072,128] (16,777,216 B) ───────┘
                       u8[ 32768,128] ( 4,194,304 B)
                             ║ byte-identical to
MLA-p_2048-bz_16-gcz_16 operands: u8[16,8192,128] (16,777,216 B)
                                  u8[16,2048,128] ( 4,194,304 B)
```

Verbatim, the `sc_dsa_gather` instruction the D1 criterion asks about:

```
%sc_dsa_gather.N.cloned.N.call-start = ((u8[3099648,128]{1,0:T(8,128)(4,1)}, u8[774912,128]{1,0:T(8,128)(4,1)}, s32[32768]{0:T(1024)}, token[]), (u8[131072,128]{1,0:T(8,128)(4,1)}, u8[32768,128]{1,0:T(8,128)(4,1)})) async-start(u8[3099648,128]{1,0:T(8,128)(4,1)} %bitcast.N, u8[774912,128]{1,0:T(8,128)(4,1)} %bitcast.N, s32[32768]{0:T(1024)} %bitcast.N, token[] %prepare_done_sc_dsa_gather.N.cloned.N.call-start), async_execution_thread="sparsecore", calls=%async_computation.N
```

**Against the stated PASS criterion** — "`sc_dsa_gather` takes two uint8 cache operands of
different minor widths, neither with `[..., 4, 256]`":

- two uint8 cache operands ✅ (`u8[3099648,128]`, `u8[774912,128]`)
- neither with `[..., 4, 256]` ✅
- "different **minor** widths" ❌ — **both are 128.** They differ in the *major* dim (3099648 vs
  774912), a 4:1 ratio, which is the nope:rope size ratio. The criterion's wording anticipated the
  split showing up as two different minor widths; in the lowered HLO the split shows up as two
  different lengths at the same 128-lane minor. This is the criterion being imprecisely worded, not
  the hypothesis failing — the substance (two separate uint8 caches, both 128-wide, neither the
  indexer's) is exactly what "split nope/rope cache" predicts. Recorded here rather than glossed.

Verbatim, the MLA kernel:

```
%MLA-p_2048-bz_16-gcz_16.N = bf16[16,4,640]{2,1,0:T(4,128)(2,1)} custom-call(s32[]{:T(128)S(6)} %select_n.N, s32[16]{0:T(128)} %get-tuple-element.N, s32[2]{0:T(128)} %pad_add_fusion.N, s32[2]{0:T(128)} %broadcast.N, bf16[16,4,640]{2,1,0:T(4,128)(2,1)} %fusion.N, u8[16,8192,128]{2,1,0:T(8,128)(4,1)} %bitcast.N, u8[16,2048,128]{2,1,0:T(8,128)(4,1)} %bitcast.N), custom_call_target="tpu_custom_call", ...
```

Source `attention.py:1564:0`. **No `[..., 4, 256]` operand.**

Cross-check in session `10_50_26` (different shapes) reaches the same conclusion and adds a link the
other session lacks: there the indexer buffer appears in both flat and 4-D form joined by *real,
timed* `reshape` instructions, `u8[774912,256] <-> u8[3027,64,4,256]` (both 198,377,472 B), and the
flat form's mention set is `reshape` / `scatter_fusion` and nothing else. MLA there is
`MLA-p_2048-bz_16-gcz_64` on `u8[64,8192,128]` / `u8[64,2048,128]`.

### 1.3 The complete toucher set for `u8[3027,64,4,256]`

| instruction | role | fw op name / source |
| --- | --- | --- |
| `StreamIdxTC-d-bq_1-bkvp_2` | **reader** — stage A, decode specialization | `.../StreamIdxTC-d-bq_1-bkvp_2/pallas_call:` — `mla_attention_op.py:140:0` |
| `StreamIdxTC-m-bq_128-bkvp_2` | **reader** — stage A, mixed/prefill specialization | same, `-m-` variant |
| `conditional.N` | pass-through — the `cond` wrapper that selects between the two branches above; carries the cache in its operand tuple | `branch_computations={%region_968.N_spmd, %region_972.N_spmd}` |
| `scatter_fusion` (`u8[3027,64,4,256]` variant) | **writer** — cache update | `custom_kernel/custom_kernel/jit(_streamidx_topk_jax)/scatter:` — `mla_attention_op.py:104` / `:140` |

Verbatim, the writer:
```
%scatter_fusion.N = u8[3027,64,4,256]{3,2,1,0:T(4,128)(4,1)} fusion(u8[3027,64,4,256]{3,2,1,0:T(4,128)(4,1)} %param.N, s32[16]{0:T(128)S(1)} %copy-done.N, u8[16,256]{1,0:T(8,128)(4,1)S(1)} %pad.N), kind=kCustom, calls=%fused_computation.N
```
Note its third operand is `u8[16,256]` — the update slice is itself padded to 256 (`%pad.N`), so the
writer is padding on the way in. `pad.N` producing `u8[16,256]` from `u8[16,129]` appears in the
small-shape inventory.

`conditional.N` is not a fourth consumer; it is the `jax.lax.cond` that dispatches to `-d-` or `-m-`
and therefore has to name the cache in its operand tuple.

**Two readers, one writer, all three inside the indexer path.** No MLA instruction, no
`sc_dsa_gather`, no `sc_topk_*` (which are pure s32).

---

## 2. Cross-check: the same conclusion, derived from source

The profile above is the **AgentX** deployment (`max_model_len=262144`, batch 16 / 1024), **not the
nightly baseline of record**. The transfer of a shape-dependent number from AgentX to nightly would
be `assumed` and I do not make one. But the *structural* claim — which buffers exist and who is
wired to them — comes from `initialize_kv_cache`, which does not depend on the operating point.
That derivation, independent of the profile:

1. `vllm/model_executor/models/deepseek_v2.py:1091` — `self.is_v32 = hasattr(config, "index_topk")`,
   true for GLM-5.2 / DSA. Line 1159 passes `is_sparse=self.is_v32`.
2. `vllm/model_executor/layers/mla.py:93,123` → `mla_attention.py:554` — every MLA attention layer
   ends up with `self.use_sparse = True`.
3. `src/vllm_torchtpu/runner/tpu_runner.py`, the `MLAAttentionSpec` branch:
   ```python
   if getattr(attn_module, "use_sparse", False):
       # Split (nope, rope) cache in uint8 dtype (expected by `dsa_gather`).
       specs = (PallasMLAttentionBackend.get_sparse_kv_cache_specs(...))
       attn_module.mla_kv_spec = specs
       kv_caches[layer_name] = tuple(
           torch.zeros(spec.shape, dtype=torch.uint8).to(self.device)
           for spec in specs)
       continue
   kv_cache_shape = PallasMLAttentionBackend.get_kv_cache_shape(...)
   tpu_kv_cache = torch.zeros(kv_cache_shape, dtype=dtype).to(self.device)
   kv_caches[layer_name] = tpu_kv_cache
   ```
   So all 78 MLA layers take the **tuple-of-two-uint8** path — matching `u8[3027,256,4,128]` +
   `u8[3027,64,4,128]` in §1.1 — and `continue` past the single-tensor path.
4. `DeepseekV32IndexerCache.get_kv_cache_spec` (`deepseek_v2.py:620-640`) returns a plain
   `MLAAttentionSpec`, and that class has **no `use_sparse` attribute**. `getattr(..., False)` is
   therefore `False` and the indexer cache falls to the **single 4-D tensor** path — matching
   `u8[3027,64,4,256]`.
5. Allocation is a separate `torch.zeros` per `layer_name`. The only aliasing mechanism in the
   runner, `bundle_view_by_name`, lives in the non-MLA `AttentionSpec` branch and is never reached
   from here.

Profile and source agree. That agreement is what makes the AgentX profile usable for a structural
question at the nightly operating point.

---

## 3. What is actually reclaimable, and what number goes with it

`modelled` — arithmetic shown, from the shapes in §1.1:

| | allocated (256) | useful (129) | padding |
| --- | --- | --- | --- |
| AgentX, per indexer layer, `u8[3027,64,4,256]` | 3027·64·4·256 = 198,377,472 B | 3027·64·4·129 = 99,963,648 B | **98,413,824 B = 98.4 MB** |
| Nightly microbench shape, `u8[576,256,4,256]` | 576·256·4·256 = 150,994,944 B | 576·256·4·129 = 76,087,296 B | **74,907,648 B = 74.9 MB** |

> **Do not conflate these with the brief's "83.2 MB".** That figure is from `report.md` §4 and is
> **read traffic per decode call** (167.8 MB at `lkv_dim=256` vs 84.5 MB at 129; 167.8 − 84.5 = 83.3),
> not storage. The two are different quantities with similar magnitudes, which is exactly how they
> get swapped. P5 establishes that *both* are the indexer's alone to reclaim; it establishes nothing
> about whether narrowing the read realizes the traffic saving. That is P6.

The useful fraction is 129/256 = 0.5039, so a perfect repack would roughly halve both figures.

The precedent for how to do it is already in the tree: MLA splits its cache into nope (128-wide) and
rope (128-wide) rather than padding one array to 256. The indexer could do the same — a 128-wide
fp8 key array plus a separate 1-byte-per-position scale array — and §1.3 says nothing outside
`streamindex_topk.py` and its one `scatter_fusion` writer would have to change.

---

## 4. Route 1 is not a falsifier. Do not run it.

`briefs/area4-layout-precision.md` §D1's Route 1 inserts a `_area4_d1_check(kv_caches,
static_forward_context)` snippet at the end of `TPUModelRunner.initialize_kv_cache`, folded into a
nightly benchmark run. Two reasons it was not run, in order of importance.

### 4.1 It emits a vacuous PASS — the exact failure mode the brief warned about

The snippet classifies entries into `indexer`, `mla_split` and `mla_single`, and its final `else`
prints

```
D1_VERDICT PASS distinct-buffers
```

whenever `indexer` is non-empty and `mla_single` is empty. **That includes the case where the MLA
side matched nothing at all** — `mla_split` empty *and* `mla_single` empty, i.e. the classification
silently failed to recognize any MLA layer. In that state the check has compared the indexer cache
against nothing and reports green.

This is not hypothetical: the classification keys on container type and shape, and a rename, a
`get_sparse_kv_cache_specs` signature change, or a layer-name pattern drift all produce exactly it.
The adjacent `D1_COUNTS ... mla_split=0` line does reveal it — but `D1_COUNTS` is not the output
contract. The contract is the `D1_VERDICT` line, and that line is green in a state where the check
proved nothing. **It would return PASS whether or not the hypothesis is true**, for at least one
reachable input.

Two secondary weaknesses, recorded for whoever repairs it:

- The `shared`-storage pointer comparison only runs when **both** sides hold single tensors. On this
  model the MLA side holds a *tuple*, so the pointer check never executes — a tuple element sharing
  storage with the indexer tensor would pass unnoticed. The one arm that could have detected true
  aliasing is dead on the only configuration that matters.
- The brief itself documents that `id()` and `data_ptr()` are unreliable under torch_xla (all-zero
  `data_ptr()` must map to INCONCLUSIVE, never to pass/fail). So even a repaired pointer check is a
  weak instrument here, and container-type-plus-shape remains the primary discriminator — which is
  precisely what Route 0 does directly off the compiler's own operand lists, without a code
  injection.

**Minimum repair, if someone wants Route 1 anyway:** make the verdict line conditional on
`len(mla_split) + len(mla_single) > 0`, and emit `D1_VERDICT INCONCLUSIVE mla-side-empty` otherwise.

### 4.2 It is not affordable in this budget

Route 1 only runs folded into a nightly end-to-end benchmark, 2.5-4 h. Route 0 answered the same
question off a profile that already exists.

---

## 5. Scope limits — what this does NOT establish

- **"Not shared with MLA attention" is narrower than "one consumer."** §1.3 enumerates *three*
  touchers of the indexer cache and one of them is a writer. Any repack must change the writer
  (`scatter_fusion`, and the `%pad.N` producing `u8[16,256]` feeding it) in lockstep with the two
  readers. The padding is nobody else's; it is not unowned.
- **Aliasing at the *allocation* level cannot be excluded from the profile alone.** The MLA and
  `sc_dsa_gather` u8 operands are `%bitcast.N`, and zero-cost `bitcast` instructions do not appear
  in `hlo_stats`, so they cannot be chained back to a named producer. The byte-size equalities in
  §1.2 are arithmetic coincidence-checks, not aliasing evidence. What closes this gap is §2's source
  derivation — separate `torch.zeros` per `layer_name` — not the profile.
- **No HLO proto or module text exists in `j-a71a648c`'s output** (`gsutil ls -r` confirms only
  `.xplane.pb`, `.pt.trace.json.gz`, `vllm.log` and benchmark sidecars), so the full def-use graph
  could not be walked. The mention-set argument is the strongest form available from this artifact.
- **The profile is AgentX, not nightly.** Every shape in §1 carries AgentX geometry. Only the
  structural claim transfers, and it transfers because §2 derives it independently of shape.

---

## Provenance of every number above

| number | tag |
| --- | --- |
| all HLO shapes, operand lists, instruction texts, mention counts | `measured (j-a71a648c-3064-471a-a131`, sessions `2026_09_02_10_53_40` and `2026_09_02_10_50_26`, xprof `hlo_stats`) — **AgentX deployment**, batch 16 / 1024 |
| padding capacity figures in §3 | `modelled` — arithmetic shown inline |
| 167.8 / 84.5 / 83.3 MB decode read traffic | from `report.md` §4, itself `modelled` at the nightly decode shape |
| 129 = 128 fp8 + 1 e8m0 scale; `lkv_dim = align_to(129,128) = 256` | source, `instruments/bench_streamidx_topk.py` + `streamindex_topk.py` |
| the `use_sparse` wiring in §2 | source read, not measured |
