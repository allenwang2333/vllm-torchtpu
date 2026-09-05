# GLM-5.3-Flash MTP on TPU

The draft head is one decoder block. Almost all of the difficulty is on the
*target* side, because 34 of the target's 45 attention layers are KDA and a
linear-attention layer cannot roll a rejected draft back the way an attention
layer can.

See [`glm5-3-flash.md`](glm5-3-flash.md) for the model as a whole.

## Why the target side needs a kernel change

Verifying `num_speculative_tokens` drafted tokens means running the target
over a window and then keeping only the accepted prefix. For attention layers
that is free: the KV cache is addressed by position, so rejecting a token is
just not reading it again.

A KDA layer has no per-position record. It has one recurrent state, which is a
summary of everything before it, so "as of the last accepted token" is a
different state, and there is no way to compute it from the post-window state.
Recomputing means replaying the prefix.

So the kernel keeps a checkpoint per window position.
`compute_kda.recurrent_kda_per_seq` loops the tile and stacks the post-token
state of the last `window_size` positions; rollback is then *selecting* the
right checkpoint, addressed by a per-slot read offset. At `window_size == 1`
the stack is the single final state and the result is bit-identical to before,
which is why this is inert for a non-speculative run.

Two consequences worth stating:

* **`chunked_kda` deliberately does not do this.** The chunked solve only ever
  produces an end-of-chunk state, exactly as its GDN counterpart does. It runs
  only in PER_SEQ mode, where the window is 1; a verify window is routed to
  `recurrent_kda` instead.
* **The mask had to become per position, not per sequence.** The old
  `real_sizes > 0` mask is correct when a tile is one token and wrong when it
  is a verify window whose positions are separate tokens: a partially-accepted
  window has `real_sizes < window_size`, and padded positions stayed live and
  kept advancing the recurrence past the last real token. Nothing about the
  shapes changes, so the symptom would have been a state that silently drifts
  by however many draft tokens were rejected. The mask is applied to the
  *activated* log-gate rather than to the raw gate input, because
  `activate_gate(0)` is not 0 — masking the input would still decay a state
  that is supposed to be left untouched.

## Why the state shape is not widened by `num_spec`

vLLM's `MambaStateShapeCalculator.kda_state_shape` asks for
`conv_kernel_size - 1 + num_spec` columns of conv cache, so a GPU verify step
can roll back through a sliding window of conv inputs. On TPU the rollback is
checkpoint selection, with a full conv *and* recurrent state written per
window position into `num_spec + 1` consecutive slots. The extra columns would
never be read — which is why the GDN op, whose vLLM base class does apply that
widening, slices them straight back off before the kernel. Not asking for them
is the same thing without the slice.

The part of the spec sizing that *does* apply needs nothing extra:
`MambaBase.get_kv_cache_spec` adds `num_speculative_blocks` to the spec, and
that is what reserves the slots the kernel writes.

## The draft head itself

Checkpoint layer 45: a decoder block plus the standard MTP wrapper (`enorm`,
`hnorm`, `eh_proj`, `shared_head.norm`). Three things differ from every
sibling MTP head in vLLM, each silent when wrong:

* **Layer 45 has no hyper-connection tensors.** Layers 0..44 each ship
  `hc_attn_{base,fn,scale}` and `hc_ffn_{base,fn,scale}`; layer 45 ships none,
  so the block runs a plain pre-norm residual rather than the mHC seam.
* **`shared_head.head.weight` does not exist.** With
  `tie_word_embeddings: false` the draft's LM head is a copy of the top-level
  `lm_head.weight`, so it is built at prefix `lm_head` — which is also the
  entry the FP8 ignore list carries. Naming it
  `model.layers.45.shared_head.head` would quantize a bf16 head.
* **`num_hidden_layers` must not be zeroed.** The sibling `glm_ocr_mtp` and
  `glm4_moe_lite_mtp` config branches set it to 0.
  `Glm5NextMultiTokenPredictor` derives the MTP layer index from
  `text_config.num_hidden_layers`, so zeroing it renames the `ModuleDict` key
  from `"45"` to `"0"`, every `model.layers.45.*` tensor lands nowhere, and
  the draft runs on random weights at ~0 acceptance with no error.

Layer 45 also carries its own complete seven-tensor sparse indexer block. When
the target stack goes sparse, so does the draft's block — by construction,
because it is built by the same class.

## `glm5_next_spec_patch.py`

vLLM does not know `glm5_next` has an MTP head. Teaching it is two hunks in
`vllm/config/speculative.py`: add `"glm5_next_mtp"` to `MTPModelTypes`, and
add a `Glm5Next` branch to `SpeculativeConfig.hf_config_override`. That belongs
upstream; `glm5_next_spec_patch.py` applies it at runtime because the TPU
image ships its own vLLM under `site-packages`, so editing a local vLLM
checkout never reaches the pod.

It is a module beside the `glm5_next` package rather than inside it, because
`register_models()` imports it on every model's boot path and importing a
submodule runs its parent package first — see the module docstring and
`tests/models/vllm/test_glm5_next_boot_isolation.py`.

Two properties it depends on:

* `MTPModelTypes` is resolved through the module global at *call* time, so
  rebinding it works — but only before the config is built, and the failure
  otherwise is `NotImplementedError: Unsupported speculative method: 'mtp'`,
  which names neither GLM nor the Literal.
* The override callable is handed to the draft `ModelConfig` as its
  `hf_overrides` and pickled into the spawned EngineCore process, so it has to
  be a module-level function in an installed package rather than a closure.
  Unpickling it re-imports this module, which re-applies the patch in the
  child; `register_models` applies it in the parent.

## Operational requirement: `SPEC_WARMUP=0`

MTP produces correct output under `enforce_eager` and wrong output compiled,
unless the speculative warm-up is disabled. `SPEC_WARMUP=0` is a requirement,
not a tuning knob.

## Evidence

CPU: `tests/kernels/test_kda_spec_checkpoints.py` runs the recurrence as plain
JAX against an independent reference, including the partially-accepted case;
`tests/models/vllm/test_kda_spec_plumbing.py` covers the operand plumbing;
`tests/models/vllm/test_glm5_next_mtp.py` covers the name mapping, which is
where an MTP port fails silently.

On device, and not reproducible from this tree: on the real checkpoint at
TP=8, MTP reached **75% acceptance** and produced tokens identical to a
no-spec run, both eager and compiled with `SPEC_WARMUP=0`.
