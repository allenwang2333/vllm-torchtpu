# DeepSeek-V4 on TPU: architecture overview

This document explains how the DeepSeek-V4 (DSv4) support that landed in `main`
between 2026-08-14 and 2026-08-19 fits together: which files own which concern,
how a forward pass flows through them, and — the part that is easy to lose —
how the four KV caches are laid out on top of each other.

Everything described here is in-tree on `main` as of commit `f204f84`.

---

## 1. How it got here

Nine PRs, in order. Each one is a useful bisect point.

| PR | Commit | What it added |
| --- | --- | --- |
| #444 | `f4d965a` | Compressor custom op + compressor state cache (`VllmCompressorStateCache`). State and compressed KV overlay one tensor, each addressed by its own metadata. |
| #445 | `8d64c58` | Indexer custom op over the `streamindex_topk` kernel. Top-k, compression ratio, head dim and softmax scale are baked into the registered op name. |
| #459 | `a4b795c` | One JAX op serving all three attention flavours (SWA / CSA / HCA), SWA cache donated for in-place update, ops memoized by real page geometry with a quarantined "profiling" slot. |
| #462 | `8a85e15` | First working end-to-end path: runtime monkey-patching of the vendor model, CUDA stream/event stubs, `examples/run_dsv4_inference.py`. |
| #464 | `46a6a94` | `torch.compile` enabled, plus three vLLM patches that unblock it (see §9). |
| #500 | `c5eac89` | **The big one.** Replaced monkey-patching with a native model package registered in `ModelRegistry`. Fixed the SWA overlay host selection (a 1.32× KV over-allocation bug). Retired the custom requantizer for the standard blockwise FP8 loader. |
| #521 | `7a2a0ce` | Deleted the now-dead patch/stub/requant files. |
| #527 | `7fd13fc` | Loader fixes (bare shard names, raw E8M0 scale bytes, `attn_sink` TP slicing), DP all-gather of `input_ids` for hash MoE, `torch.cuda.Event` stub restored. |
| #525 | `f204f84` | Ported `compress_and_store/` and `core_attention/` kernels from tpu-inference. **Test-only today** — see §11. |

The net shape of the change: DSv4 is a *native* model in this repo
(`ModelRegistry.register_model("DeepseekV4ForCausalLM", ...)`), subclassing
vLLM's `vllm.models.deepseek_v4` layer classes only where the structure is
shared, and replacing every CUDA-shaped piece with a Pallas/JAX op.

---

## 2. File map

```
src/vllm_torchtpu/
├── models/vllm/
│   ├── __init__.py                       # register_models() → ModelRegistry
│   └── deepseek_v4/
│       ├── model.py                      # DeepseekV4ForCausalLM / Model / DecoderLayer, load_weights
│       ├── attention.py                  # VllmDeepseekV4MLAAttention — the layer
│       ├── moe.py                        # DeepseekV4MoE — gate + FusedMoEFactory
│       └── layers.py                     # DeepseekV4MLP, mhc_collapse_head
├── layers/vllm/custom_ops/deepseek_v4/
│   ├── deepseek_v4_attention_op.py       # _attention_jax, SWA cache spec, packed head size
│   ├── deepseek_v4_compressor.py         # _compressor_jax, VllmCompressorStateCache
│   ├── deepseek_v4_indexer.py            # _indexer_jax, VllmDeepseekV4IndexerCache
│   └── deepseek_v4_attention.py          # pre-#500 op module — DEAD, no importers
├── layers/vllm/quantization/deepseek_v4_fp8.py   # "deepseek_v4_fp8" quant config
├── kernels/deepseek_v4/
│   ├── mla.py, mla_swa.py                # ← LIVE: what the model calls
│   ├── compressor.py, compress_norm_rope.py
│   ├── streamindex_topk.py
│   ├── core_attention/                   # ← ported from tpu-inference, test-only
│   └── compress_and_store/               # ← ported from tpu-inference, test-only
├── runner/tpu_runner.py                  # KV cache allocation + overlay (§5), kernel pre-build
└── platforms/tpu_platform.py             # DSv4 block-size override
examples/run_dsv4_inference.py            # offline runner
```

---

## 3. The residual stream is not a vector

DSv4 carries **`hc_mult` parallel residual streams** ("MHC" — the multi-head
coupled residual). This is the first thing to internalise, because it changes
the shape of everything.

`DeepseekV4Model.forward` embeds tokens, then immediately:

```python
hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
# [num_tokens, hidden] → [num_tokens, hc_mult, hidden]
```

Every decoder layer is `tensor-in, tensor-out` at that rank-3 shape. Inside a
layer, each of the two sublayers (attention, FFN) is wrapped:

```
post, comb, layer_input = mhc_pre_layer(residual=x, fn=hc_*_fn, ...)   # mix streams → one vector
x = norm(layer_input); x = sublayer(x)                                  # ordinary sublayer
x = mhc_post_layer(x, residual, post, comb)                             # scatter back to streams
```

`mhc_pre_layer` / `mhc_post_layer` come from vLLM
(`vllm.model_executor.kernels.mhc.torch`). The mixing uses a Sinkhorn
normalisation (`hc_sinkhorn_iters`, `hc_eps`) over per-layer parameters
`hc_{attn,ffn}_{fn,base,scale}`.

At the end of the backbone, `mhc_collapse_head` (in `layers.py`) folds the
`hc_mult` streams back into one vector before `norm` and the LM head. That one
is a pure-PyTorch reimplementation — upstream only ships Triton/TileLang GPU
kernels for `hc_head`.

Two consequences worth remembering:

- Pipeline-parallel intermediate tensors are `[batch, hc_mult, hidden]`
  (`make_empty_intermediate_tensors`).
- `_mtp_hidden_buffer` is sized `hc_mult * hidden_size` — it holds the
  uncollapsed stream. It is allocated and exposed via
  `get_mtp_target_hidden_states()`, but nothing writes it yet, and `mtp.`
  weights are skipped at load. **MTP is a stub.**

---

## 4. Three attention flavours, one op

Every layer's `VllmDeepseekV4MLAAttention` has a `compress_ratio` that selects
its behaviour:

| `compress_ratio` | Flavour | Reads | Extra operand |
| --- | --- | --- | --- |
| `<= 1` | **SWA** — sliding window only | SWA cache | — |
| `== 4` | **CSA** — top-k sparse | SWA cache + compressed KV | `topk_indices` from the indexer |
| `> 4` (128 in practice) | **HCA** — long range | SWA cache + compressed KV | `kv_lens_to_attend = (positions+1) // compress_ratio` |

All three go through **one** registered Pallas op, `_attention_jax` in
`deepseek_v4_attention_op.py`, which internally:

1. Runs `mla_sliding_window_ragged_paged_attention` (writes new KV into the SWA
   cache, returns output plus running `l`/`m` softmax statistics).
2. If `swa_only`, returns there.
3. Otherwise runs `mla_ragged_paged_attention` over the compressed KV cache,
   **continuing** the SWA softmax accumulation via `swa_accumulation`, `swa_l`,
   `swa_m`. That is why the SWA kernel is called with
   `unnormalized_output=True` for non-SWA layers — the two attentions are one
   fused softmax, not two results averaged.

### Op identity and the profiling trap

Ops are memoized in `_attn_op_cache` under a name that encodes the geometry:

```
pallas::deepseek_v4_attention_{swa|csa|hca}_v2_p{logical_page_size}[_aliased][_prof]
```

The `_prof` suffix matters. vLLM's profiling forward pass runs **before**
`initialize_kv_cache`, when `kv_cache` is a `numel == 0` placeholder. An op
built then has the wrong page geometry baked in, so it goes into a quarantined
memo slot (`_attn_op_prof_instance`) and can never serve real traffic. Once
real caches exist, `_build_attn_op` runs again and the result is stored under
`_attn_op_instance`. The `[ATTN_BUILD]` log line records which geometry each op
was built with — that log is the first thing to check when attention output is
wrong but not NaN.

`_aliased` records `two_caches_same_buffer`: when the SWA and compressed-KV
caches are literally the same tensor (see §5), the JAX function forwards the
SWA kernel's *updated* buffer into the main kernel so it sees tokens written
this step.

The SWA cache is passed with `donate_argnums=(2,)` so XLA updates it in place
rather than allocating a second full-size copy; the Python side copies the
returned buffer back into the original tensor.

---

## 5. Four caches, one buffer — the overlay

This is the subtlest part of the whole feature, and it lives in
`runner/tpu_runner.py`, not in the model.

DSv4 declares **four** kinds of KV cache:

| Cache | Spec | `head_size` (bytes) | Page holds |
| --- | --- | --- | --- |
| Compressed KV (main) | `MLAAttentionSpec` | `align_to(kv_lora_rank + 2·qk_rope_head_dim + kv_lora_rank/64, 128)` = **640** | `block_size / compress_ratio` compressed rows |
| SWA | `SlidingWindowMLASpec` | same 640 | `min(block_size/4, window_size)` rows |
| Indexer K | `MLAAttentionSpec` | `align_to(129, 128)` = **256** | — |
| Compressor state | `SlidingWindowMLASpec` | `align_to(state_dim·4, 128)` | `block_size/compress_ratio/4/2/coff` rows |

All four declare `dtype=torch.uint8`. That is deliberate: `uint8` is the escape
hatch that stops `kv_cache_spec_normalizer` from retyping the spec to the
model's KV dtype and truncating the packed record. The runner exempts these
layers from normalisation entirely via `is_cache_for_ds_v4()`.

### Why the runner has to rebuild the layout

vLLM's group allocator lays these out as **one packed slab** and hands out page
indices assuming the caches share it. On GPU that works because you can take a
byte view into a slab. **TPU arrays are strongly typed and cannot be byte views
into a slab**, so `_initialize_ds_v4_kv_cache` reconstructs the same aliasing
from the specs:

- every `MLAAttentionSpec` layer gets its own array, shaped by its own spec;
- every **compressor state cache** is aliased onto its own layer's
  compressed-KV array (`_plan_ds_v4_auxiliary_overlay` /
  `_alias_ds_v4_auxiliary_caches`). The compressor writes both records through
  that single tensor;
- every **SWA cache** is aliased onto a *distinct* compressed-KV array
  (`_plan_ds_v4_swa_overlay`), distinct because all layers in one cache group
  share a block table. Any SWA cache with no host left gets its own array.

Overlays are **planned before allocation**, not allocated-then-aliased —
otherwise every array in the loop is live simultaneously and counts against
peak HBM. That, plus the per-group host selection fixed in #500, is what
removed the 1.32× KV over-allocation and unblocked `gpu_memory_utilization`
0.75.

The guards in `_alias_ds_v4_auxiliary_caches` are load-bearing: if a layout
change makes the state record no longer fit in the host page, it raises there
rather than letting `pack_state_cache` write past the page.

### Block-size arithmetic

`tpu_platform.update_block_size_for_backend` overrides the default for DSv4:
the backend's hardcoded 256 does not fit the packed latent record, so it takes
`PallasMLAttentionBackend.get_page_size()` = **1024**.

At `block_size = 1024`:

- CSA compressed page = 1024/4 = **256 rows**
- HCA compressed page = 1024/128 = **8 rows**
- SWA page = `min(1024/4, window_size)`
- HCA compressor state page = 8/4/2/1 = **1 row**

1024 is not a comfortable default — it is the *minimum*. `_state_block_size`
raises below 1024, because an HCA page at `block_size=512` is 4 rows and the
state block size floors to zero. If you lower the block size, DSv4 fails at KV
cache init with an explicit error, by design.

---

## 6. The compressor

`VllmDeepseekCompressor` (`deepseek_v4_compressor.py`) writes compressed KV
rows and owns the state cache. Called once per layer from `attention_impl`.

The subtlety it exists to handle: **state and compressed KV overlay one
tensor**, at different granularity, with two independent block tables. So the
JAX function computes two separate slot mappings:

```python
# state rows, addressed by the state cache's own metadata
slot_mapping    = state_block_tables[state_block_num] * state_block_size + state_offset
# compressed KV rows, addressed by k_cache's metadata
kv_slot_mapping = k_block_tables[k_block_num]         * k_block_size     + k_offset
```

Each uses **its own metadata's `input_positions` and block tables**. Borrowing
the other layer's would scatter rows into the wrong pages — the `forward`
method raises a `KeyError` naming the missing prefix rather than silently
falling back. The `% max_num_*_blocks_per_req` guards against a stray absolute
position aliasing into the next request.

Other notes:

- Two kernels: `compressor_forward` for the main compressor (`head_dim == 512`)
  and `compressor_forward_indexer` for the indexer's compressor
  (`head_dim == 128`).
- The cache is donated (`donate_argnums=(12,)` — the index counts torch_tpu's
  *filtered* tensor args) and copied back.
- vLLM allocates the tensor as `kv_cache_dtype`, but the kernel writes raw
  bytes, so the compressor takes a `.view(torch.uint8)` — assigning uint8 into
  an fp8 array would convert rather than bitcast. `.view()` shares storage, so
  the writes still land.
- `use_fp4_indexer_cache` raises `NotImplementedError`: the MXFP4 record layout
  has no TPU kernel, and our kernel always emits FP8/UE8M0.

---

## 7. The indexer

`VllmDeepseekV4Indexer` ranks compressed KV rows via the `streamindex_topk`
Pallas kernel and returns `topk_indices` for CSA layers.

Flow per layer: project `q` → RoPE → run the indexer's own compressor → quantise
`q` to `float8_e4m3fn` in JAX (with a `nan_to_num` on the reciprocal scale so an
all-zero row quantises to 0 rather than NaN) → fold the softmax scale,
`head_dim**-0.5` and the per-row q-scale into `indexer_weights` → top-k.

The registered op name bakes in `k`, compression ratio, head dim and softmax
scale:

```
pallas::deepseek_v4_indexer_k{topk}_c{ratio}_d{head_dim}_s{scale}
```

A constant name would hand the first layer's compiled op to every other indexer
layer — this is the same class of bug as the `compile_prefix` issue in §9.

---

## 8. MoE and hash routing

`DeepseekV4MoE` wraps a `GateLinear` plus `FusedMoEFactory`. The routing mode
depends on the layer index:

- **Layers `< config.num_hash_layers`**: *static hash routing*. `gate.tid2eid`
  is a `[vocab_size, num_experts_per_tok]` table; a token's experts are looked
  up by its **token id**, not by gating logits. The gate output is still used
  for the weights (`scores.gather(-1, topk_ids)`). Implementation:
  `_hash_moe_select` in `layers/vllm/moe_routing.py`.
- **Later layers**: ordinary scored routing, `scoring_func="sqrtsoftplus"`,
  optionally with `noaux_tc` correction bias.

### The DP gotcha

Hash layers route on `input_ids`, but the MoE runner routes over the
**DP-gathered global batch**. Rank-local ids would be misaligned with the rows
the runner is routing. `_dp_gather_hash_moe_input_ids` (added in #527) does an
`all_gather(dim=0)` — the same gather hidden states get — so rows line up. It
only runs when the PP stage actually contains hash layers.

`DeepseekV4MoE.forward` raises if `tid2eid` is set but `input_ids` is `None`.

---

## 9. torch.compile

`support_torch_compile` is applied to `DeepseekV4Model` (the whole backbone —
#500 moved it there from per-layer, to prevent AOT parameter collisions on the
dynamic hash layers). Three vLLM patches in `src/vllm_torchtpu/__init__.py`
make the compiled path work:

1. **`_patch_vllm_compile_prefix_isolation`** — vLLM's shared
   `compile_prefix=""` gives every instance of a decorated class one on-disk
   cache directory. That is only safe when instances close over
   identically-shaped buffers; the DSv4 KV cache overlay breaks it, and one
   layer's executable gets reused by another (XLA e0102). The patch keys the
   prefix on `self.prefix`. It is a no-op for shape-uniform models.
2. **`_patch_vllm_reset_compile_wrapper`** — makes `reset_compile_wrapper`
   recurse into per-layer wrappers instead of only checking the object it was
   handed and its `.model`.
3. **`_patch_vllm_piecewise_backend`** — when `sym_shape_indices` is empty
   (all arguments are tensors, no `SymInt`), fall back to tensor dim 0 to pick
   the compiled token-count bucket, instead of asserting that only one bucket
   exists.

Plus `sys.setrecursionlimit(50000)` in `DeepseekV4ForCausalLM.__init__` —
tracing 61 layers at long context exceeds CPython's 1000-frame limit.

`compilation/tpu_compiler.py` lists
`models/vllm/deepseek_v4/attention.py` among the files whose source is hashed
into the compile cache key.

The runner pre-builds all three op families before serving
(`_build_attn_op` / `_build_compressor_op` / `_build_indexer_op` are walked
over `self.model.modules()` in `tpu_runner.py`), popping the memoized instance
first so the profiling-stage op is discarded.

---

## 10. Quantization and weight loading

**Quant config** — `VllmDeepseekV4Fp8Config` registers `"deepseek_v4_fp8"` (and
`"tpu-deepseek_v4_fp8"`). Linear layers always take `VllmFp8LinearMethodTPU`
(blockwise FP8). MoE dispatches on the checkpoint's `expert_dtype`:
`"fp4"` → `VllmDeepseekV4Mxfp4MoEMethod`, otherwise `VllmFp8MoEMethodTPU`.
`is_scale_e8m0` additionally sniffs `hf_config.quantization_config["scale_fmt"]
== "ue8m0"`.

**Weight names** — `_make_deepseek_v4_weights_mapper` rewrites checkpoint names
into vLLM layer names (`layers.` → `model.layers.`, `head.weight` →
`lm_head.weight`, `.ffn.gate.bias` → `.e_score_correction_bias`, `.scale` →
`.weight_scale_inv`, with an extra fp4-only rule for expert scales).

**`DeepseekV4Model.load_weights`** handles four cases:

- *stacked* — including the bare shard names `w1`/`w3` → `gate_up_proj`, which
  is what makes the shared-expert `ffn.shared_experts.w1/w3` tensors land
  correctly (restored in #527), plus `attn.wq_a`/`attn.wkv` → `fused_wqa_wkv`
  and `compressor.wkv`/`wgate` → `fused_wkv_wgate`;
- *routed experts* — via `fused_moe_make_expert_params_mapping`, computed once
  outside the loop. **E8M0 scales are `.view(torch.uint8)`, not `.copy_()`**:
  the checkpoint stores `float8_e8m0fnu` but the MoE param is `uint8`, and a
  numeric conversion would destroy the raw exponent bytes (2⁻⁷ → 0);
- *`attn_sink`* — stored per **global** head, so each rank narrows to its own
  `[n_local_head * tp_rank : n_local_head * (tp_rank+1)]` slice;
- *plain* — `default_weight_loader`.

`DeepseekV4ForCausalLM.load_weights` also patches
`safetensors.torch._TYPES["F8_E8M0"] = torch.uint8` so safetensors will hand
back the raw bytes at all, and skips `mtp.` weights.

---

## 11. Two kernel trees

`f204f84` (#525) ported `compress_and_store/` and `core_attention/` from
tpu-inference. **Nothing in `src/` imports them** — the only importers are
`tests/kernels/deepseek_v4/*`. The live path is still:

- `kernels/deepseek_v4/mla.py` — main attention, handles both CSA
  (`topk_indices`) and HCA (`kv_lens_to_attend`) in one kernel;
- `kernels/deepseek_v4/mla_swa.py` — sliding window;
- `kernels/deepseek_v4/compressor.py` + `compress_norm_rope.py`;
- `kernels/deepseek_v4/streamindex_topk.py`.

The ported tree is a genuinely divergent snapshot, not a copy — it splits CSA
into `sparse_mla.py` + `csa_gather.py`, and its shared helpers differ (e.g. the
in-tree `cdiv` is traceable with a symbolic denominator, the ported one asserts
`b != 0`). Reconciling the two, or deciding which survives, is open work.

---

## 12. Running it

```bash
export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
python examples/run_dsv4_inference.py
```

Knobs the example reads (all env vars):

| Var | Default | Note |
| --- | --- | --- |
| `DSV4_MODEL_PATH` | `deepseek-ai/DeepSeek-V4-Flash` | |
| `VLLM_TP_SIZE` / `VLLM_EP_SIZE` | 8 / 8 | v7x-8 |
| `DSV4_MAX_MODEL_LEN` | 64 | |
| `DSV4_MAX_NUM_BATCHED_TOKENS` | = max_model_len | keeps long prompts chunked, as the production benchmark does |
| `GPU_MEMORY_UTILIZATION` | 0.40 | 0.75 works post-#500 |
| `DSV4_ENABLE_PREFIX_CACHING` | `0` | **off by default** — it changes block allocation, which interacts with the SWA/compressed-KV overlay |
| `DSV4_ASYNC_SCHEDULING` | auto | set `0` to disable |
| `VLLM_ENFORCE_EAGER` | `0` | |

Fixed in the example: `quantization="deepseek_v4_fp8"`,
`kv_cache_dtype="fp8_e4m3"`, `trust_remote_code=True`.

The runner also has a TPU preflight that waits for `/dev/accel*` locks to clear
and sweeps stale `/tmp/ipc*`, `/tmp/psm*`, `/tmp/vllm*` handles.

The #500 hardware gate (v7x-8, TP=8, EP=8, compiled) was: tracebacks=0,
sweeps=13, raises=0, needle exact, generations byte-identical to baseline.

---

## 13. Known gaps

- **MTP is a stub.** `_mtp_hidden_buffer` is allocated and exposed but never
  written; `mtp.` weights are skipped at load.
- **The two kernel trees have not been reconciled** (§11).
- **No DSv4 job in `.buildkite/`.** Unit tests run in the general suite; there
  is no e2e or perf pipeline entry. Correctness has been gated by hand on
  hardware.
- **Prefix caching is untested** with the overlay, hence off by default.
- **`BATCH_AXIS = None`** in all three op modules — the mesh has no batch axis
  yet. It gets a name when the DP-attention work lands.
- **`_unified_kv_layout` / `_is_ds_v4` are per-model flags** in the runner;
  the standing TODO is to replace both with per-layout allocation dispatch
  around a shared tail.
- **Dead code:** `layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py`
  (818 lines) has no importers in `src/` or `tests/`. It is the pre-#500 op
  module — it still imports `vllm.models.deepseek_v4.amd.model`, the ROCm
  variant #500 removed the dependency on. `7a2a0ce` swept the other three
  pre-#500 files; this one survived.
