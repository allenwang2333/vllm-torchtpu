# GLM-5.3-Flash on TPU

Design notes for the `glm5_next` port: what the model is, which parts of it
are reused, which are new, and what has been measured.

File:line citations in this port — in the code and on these pages — are as-of
the revisions this tree is pinned to: vLLM `v0.27.0`, per
[`../vllm-kv-cache-api-pin.md`](../vllm-kv-cache-api-pin.md), and the
`glm5_next` transformers overlay described under "Running the parity suites".
Line numbers rot; the symbol names around them are the durable part.

## The model

Structurally DeepSeek-V4 — multi-hyper-connection residual manifold,
sparse-MLA-shaped full attention, sigmoid/`noaux_tc` MoE — with 34 of its 45
attention layers replaced by Kimi Delta Attention, in a 3:1 schedule. The
shipped FP8 checkpoint is 328 GB and runs at TP=8.

| | |
| --- | --- |
| layers | 45 |
| attention | 34 KDA (linear) + 11 DSA (MLA) |
| `qk_rope_head_dim` | 0 — NoPE |
| `kv_lora_rank` | 512 |
| `index_kpool` / `index_topk` | 4 / 2048 |
| MoE | 288 routed experts + 1 shared, `topk_method="noaux_tc"` |

Both halves already existed in this repo, so most of the port is composition:
KDA layers reuse `models/vllm/kimi_k3/attention.py:KimiDeltaAttention`
verbatim, and the mHC machinery is DeepSeek-V4's. The parts that are not
composition are below, ordered by how badly each fails when it is wrong.

## Layer classification: two conventions with the same key name

`layer_types` is a per-layer list and is 0-indexed by construction.
`linear_attn_config["kda_layers"]` also exists in GLM's config, is also
0-indexed — and the identically-named Kimi-Linear field counts from 1. Reading
GLM's list with Kimi's rule returns a plausible count, builds every layer, and
mis-sizes the KV cache for the wrong set of layers with no error.
`layer_types` therefore wins wherever it is present; see
`vllm_torchtpu/__init__.py:_patch_vllm_kimi_kda_layer_counts` and
`tests/models/vllm/test_kda_layer_counts.py`.

## NoPE full attention, and the zero rope slab

`qk_rope_head_dim == 0`: position information reaches the DSA layers through
the KDA recurrence and the short convolutions only.

The MLA v2 Pallas kernel cannot take `r_dim = 0`. It is parametric in
multiples of 128, but zero is structurally impossible rather than merely
untested: it allocates `pltpu.VMEM((..., r_dim))` scratch that Mosaic cannot
size at zero lanes, its prologue reshapes by `(2, -1, r_dim)`, several KV DMAs
would become zero-byte copies sharing a semaphore with a live one, and it
contracts a matmul over `d == 0`.

So the layer synthesises a slab of zeros instead. That is numerically exact,
not an approximation: `q_pe · k_pe` is identically 0, so the rotary term drops
out of the logits, and the value path never reads those columns. The cost is a
wider KV cache (512 → 640 in the minor dim, +25%) and ~25% more QK MXU work;
the PV matmul that dominates decode is untouched. Teaching the kernel
`r_dim = 0` means editing at least five sites inside a 2600-line Pallas kernel
where the failure mode is a DMA-semaphore hang rather than an error.

## Submodules are named exactly as the checkpoint names them

`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm` — not
anything tidier. `quantization_config.modules_to_not_convert` is a list of
*checkpoint* paths and `is_layer_skipped` matches it against *module* prefixes
exactly, so renaming `self_attn` to `attn` would silently stop all 34 bf16 KDA
layers from being recognised as unquantized and hand them to the FP8 method.

## MoE: two options DeepSeek-V4's layer did not need

* `apply_routed_scale_to_output=False`. GLM has `routed_scaling_factor` 2.5
  and a *separate* shared expert, so scaling the combined output would scale
  the shared expert too. The factor belongs on the router's top-k weights,
  which is what transformers' `DeepseekV3TopkRouter` does.
* `use_grouped_topk=True`. GLM is `topk_method="noaux_tc"`, so expert
  *selection* uses `sigmoid(logits) + e_score_correction_bias` while the
  weights come from the unbiased scores. vLLM applies that bias only inside
  the grouped-top-k router; without this the bias is ignored and the wrong
  experts run, with no error.

## The mHC collapse

GLM's final manifold collapse is a mean over hyper-connection streams where
DeepSeek-V4 takes the first — `layers.py:mhc_collapse_head_mean`, pinned by
`tests/models/vllm/test_glm5_next_collapse_head.py`.

## The dense-MLA context bound

The DSA layers run as dense causal MLA, and that is *exact* only up to
`index_topk` (2048) tokens: a query at position `p`
has `floor((p+1)/index_kpool)` candidate pools against a budget of
`index_topk // index_kpool`, so the pooled top-k would have selected
everything and the union is exactly `[0, p]`. Past that the reference attends
sparsely and a dense build does not, and the output is fluent and wrong.

`model._check_context_within_dense_mla_bound` raises at construction, before
the checkpoint loads. `VLLM_TORCHTPU_GLM5_ALLOW_LONG_CONTEXT=1` downgrades it
to a capitalised warning, which is right for measuring throughput and never
right for judging output.

## Running the parity suites for real

Several test suites are green-by-skip without their inputs. To run them:

1. **transformers with `glm5_next`.** `glm5_next` is new enough that the
   installed transformers may not carry it. Point
   `GLM5_NEXT_TRANSFORMERS_SRC` at a directory holding
   `transformers/models/glm5_next/` and it goes on `PYTHONPATH` ahead of the
   installed package; it stops being necessary the day `glm5_next` ships in a
   transformers release. That one variable is the only spelling — the scripts
   and the suites both read it. Affected: `scripts/glm5_next/`.
2. **The tiny fixture.** `scripts/glm5_next/build_tiny_fixture.py` writes a
   tiny random GLM-5.3-Flash checkpoint; `GLM5_NEXT_TINY` names where.
3. **The oracles.** `scripts/glm5_next/dump_reference.py` captures the fp32
   HF reference that `test_glm5_next_kda_parity.py` diffs against.

`VLLM_TORCHTPU_CPU_STUB=1` is required for the whole rung; see
`tests/stubs/torch_tpu.py`.

## Evidence

CPU, reproducible from this tree:

| claim | where |
| --- | --- |
| KDA layers match the fp32 HF reference to rel 5e-7 (all 6 KDA layers of the tiny fixture) | `tests/kernels/test_glm5_next_kda_parity.py` |

On device, and **not reproducible from this tree** — these were run through an
out-of-tree job recipe and their probe scripts are not part of the patch:

| claim | configuration |
| --- | --- |
| whole model agrees with the fp32 oracle per layer: worst rel-med 2.138e-02, top-1 0.9792, logit cosine 0.999763 | tiny fixture, 1 chip |
| 189/189 parameters loaded, 0 missing, 0 unmapped | tiny fixture |
| 328 GB FP8 checkpoint loads from 62 shards and generates coherent greedy text | real checkpoint, TP=8 |
