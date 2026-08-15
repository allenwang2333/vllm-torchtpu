#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline compiled inference for a real Kimi K3 checkpoint, trimmed small.

Pointed at an actual Kimi K3 checkpoint rather than a synthetic
``load_format="dummy"`` config (contrast ``offline_eager_kimi.py``), because
K3's MoE experts are mxfp4-pack-quantized (compressed-tensors), and dummy
loading cannot drive that path:

- ``VllmCompressedTensorsW4ANMxfp4MoEMethod.process_weights_after_loading``
  reads ``layer.w13_weight_packed._cpu_scratch``, an attribute only ever
  attached by the real CPU weight-loader hook
  (``compressed_tensors_moe/utils.py``). Dummy loading skips that hook
  entirely, so the attribute never exists and loading fails with
  ``AttributeError: 'Parameter' object has no attribute '_cpu_scratch'``.
- The native (non-compressed-tensors) mxfp4 path requires
  ``moe_config.has_bias=True``, which ``KimiMoE`` never sets, so that path
  is not an option either.

This checkpoint's ``text_config`` is what makes it "Kimi K3 style" rather
than plain Kimi Linear: mxfp4-pack-quantized routed experts, an
``attn_res_block_size`` (K3's block-softmax attention residual, see
``KimiDecoderLayer`` / ``AttentionResidual``), a latent-MoE
``routed_expert_hidden_size``, and the ``situ`` activation.

Trimming to keep it cheap to run. The real checkpoint is 8 layers / 896
experts / ~118 GiB on disk, which needs ``tensor_parallel_size>=2`` just for
memory. ``--num-layers``/``--num-experts`` (default 2 / 8) shrink the
*constructed* model via a ``hf_overrides`` callable, so the real run needs
far less TPU memory and far fewer MoE kernel builds -- small enough that
``tensor_parallel_size=1`` fits again. Layers beyond ``--num-layers`` and
experts beyond ``--num-experts`` are skipped in ``AutoWeightsLoader`` (patched
in, see ``_patch_layer_and_expert_skip``) so trimming doesn't crash on
now-unexpected checkpoint keys.

This does NOT reduce disk I/O: vLLM's safetensors loader reads every tensor
out of every shard file it opens regardless of which layers/experts the
trimmed model keeps (its only pre-read skip hook is for expert-parallel
sharding, not for a trimmed model). Every run still pays close to the full
checkpoint's read time; only TPU-side memory and compute drop.

The layer forced into the full-attention (MLA) role -- the last of
``--num-layers`` -- is a real KDA layer in the checkpoint whenever
``--num-layers`` isn't a multiple of the original layout's one MLA layer (4).
Its self_attn weights are therefore skipped too and left randomly
initialized; everything else (MoE experts kept, embeddings, norms, the
attn-residual blocks) is the checkpoint's real data. Generated text is
consequently not meaningful -- this is for exercising control flow and real
mxfp4/attn_res code paths cheaply, not for checking generation quality.

Usage::

    python scripts/debug/offline_eager_kimi_k3.py

    # More of the real model.
    python scripts/debug/offline_eager_kimi_k3.py --num-layers 4 --num-experts 32
"""

from __future__ import annotations

import argparse
import os
import re

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ.setdefault("TPU_LAUNCH_BLOCKING", "1")

# os.environ["LIBTPU_INIT_ARGS"] = (
#     f"--xla_jf_dump_to=/mnt/lustre/home/george/ws-torchtpu/vllm-torchtpu/artifacts/kimi_mini_dump "
#     f"--xla_jf_dump_hlo_text=true "
#     f"--xla_jf_dump_llo_text=true "
#     f"--xla_jf_dump_llo_html=false "
#     f"--xla_jf_dump_llo_static_gaps=true "
#     f"--xla_jf_emit_annotations=true "
#     f"--xla_jf_debug_level=2 "
# )

DEFAULT_MODEL = (
    "/mnt/lustre/hf-models/hub/models--Inferact--Kimi-K3-dummy-9L/"
    "snapshots/4f528f3ba9265eeb79f048d8591a1b9039036a2e")

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _patch_layer_and_expert_skip(num_layers: int,
                                 extra_skip_prefixes: list[str]) -> None:
    """Make ``KimiLinearForCausalLM.load_weights`` tolerate a trimmed model.

    Two separate holes to plug, because ``load_weights`` matches expert
    weights against ``expert_params_mapping`` *before* any of them reach
    ``AutoWeightsLoader``: that loop does an unconditional ``params_dict[name]``
    for any weight name containing ``experts.<id>.`` for a kept expert id,
    with no layer check, so an excess layer's expert 0..7 (num_experts=8's
    kept range) crashes with ``KeyError`` there instead of ever being skipped.

    - Excess layers (index >= ``num_layers``) are filtered out of the raw
      ``weights`` iterator before ``load_weights`` runs at all, so they never
      reach that loop regardless of shape.
    - Excess experts *within a kept layer* (id >= ``--num-experts``) and the
      role-swapped layer's self_attn (see ``_make_hf_overrides``) don't hit
      the expert-mapping loop, so extending ``AutoWeightsLoader``'s
      ``skip_prefixes`` is enough for those. Rebinding the name it looks up on
      the model module works because Python resolves ``AutoWeightsLoader``
      there at call time, not at the original import.
    """
    from vllm.model_executor.models.utils import AutoWeightsLoader

    import vllm_torchtpu.models.vllm.kimi_k3.model as kimi_model

    class _TrimmedAutoWeightsLoader(AutoWeightsLoader):

        def __init__(self, module, *, skip_prefixes=None, **kwargs):
            super().__init__(
                module,
                skip_prefixes=[*(skip_prefixes or []), *extra_skip_prefixes],
                **kwargs)

    kimi_model.AutoWeightsLoader = _TrimmedAutoWeightsLoader

    original_load_weights = kimi_model.KimiLinearForCausalLM.load_weights

    def load_weights(self, weights):

        def kept_layers_only():
            for name, tensor in weights:
                match = _LAYER_IDX_RE.search(name)
                if match and int(match.group(1)) >= num_layers:
                    continue
                yield name, tensor

        return original_load_weights(self, kept_layers_only())

    kimi_model.KimiLinearForCausalLM.load_weights = load_weights


def _make_hf_overrides(num_layers: int, num_experts: int):
    """Shrink ``text_config`` to ``num_layers``/``num_experts`` in place.

    Passed as vLLM's ``hf_overrides``, which also calls this on a throwaway
    ``PretrainedConfig`` probe (no ``text_config``) before the real one --
    that call must be a no-op, hence the early return.
    """

    def hf_overrides(config):
        if not hasattr(config, "text_config"):
            return config
        text_config = config.text_config

        original_num_experts = text_config.num_experts
        first_k_dense_replace = text_config.first_k_dense_replace
        moe_layer_freq = text_config.moe_layer_freq
        linear_attn_config = text_config.linear_attn_config
        original_kda_layers = set(linear_attn_config["kda_layers"])

        # Same convention as offline_eager_kimi.py's synthetic config: KDA on
        # every layer but the last, which is forced to full attention (MLA) --
        # a KDA-only model has no full-attention layer for
        # `_find_non_ssm_backend` to return.
        new_kda_layers = list(range(1, num_layers))
        linear_attn_config["kda_layers"] = new_kda_layers
        linear_attn_config["full_attn_layers"] = [num_layers]
        text_config.num_hidden_layers = num_layers

        text_config.num_experts = num_experts
        text_config.num_experts_per_token = min(
            text_config.num_experts_per_token, num_experts)

        skip_prefixes = []
        for layer_idx in range(num_layers):
            is_moe = (layer_idx >= first_k_dense_replace
                      and layer_idx % moe_layer_freq == 0)
            if is_moe:
                skip_prefixes.append(
                    f"model.layers.{layer_idx}.block_sparse_moe.gate.")
                skip_prefixes.extend(
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{eid}."
                    for eid in range(num_experts, original_num_experts))

            new_is_kda = (layer_idx + 1) in new_kda_layers
            original_is_kda = (layer_idx + 1) in original_kda_layers
            if new_is_kda != original_is_kda:
                # This layer's role changed -- its real self_attn weights
                # don't match the module now built for it. Leave it randomly
                # initialized rather than risk a shape-coincidence silently
                # loading the wrong tensor into it.
                skip_prefixes.append(f"model.layers.{layer_idx}.self_attn.")

        _patch_layer_and_expert_skip(num_layers, skip_prefixes)
        return config

    return hf_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument(
        "--compile-size",
        type=int,
        default=16,
        help="Only token-count vLLM compiles a graph for; every prefill/"
        "decode step is padded up to this many tokens.")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig

    llm = LLM(
        model=args.model,
        trust_remote_code=True,  # checkpoint supplies its own config/tokenizer
        hf_overrides=_make_hf_overrides(args.num_layers, args.num_experts),
        tensor_parallel_size=args.tensor_parallel_size,
        compilation_config=CompilationConfig(
            compile_sizes=[args.compile_size]),
        max_num_batched_tokens=args.compile_size,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        max_num_seqs=1,
    )

    text_config = llm.model_config.hf_text_config
    quantization_config = getattr(text_config, "quantization_config",
                                  None) or {}
    print(
        "K3 config in effect: "
        f"num_hidden_layers={text_config.num_hidden_layers} "
        f"quant_method={quantization_config.get('quant_method')} "
        f"num_experts={text_config.num_experts} "
        f"attn_res_block_size={getattr(text_config, 'attn_res_block_size', None)} "
        f"hidden_act={text_config.hidden_act}",
        flush=True,
    )

    prompts = [
        "The capital of France is",
        "In machine learning, a transformer is",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )

    for prompt, out in zip(prompts, outputs):
        print(f"prompt={prompt!r} -> {out.outputs[0].text!r}")
    print("RUN_OK")


if __name__ == "__main__":
    main()
