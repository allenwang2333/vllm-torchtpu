# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash (``glm5_next``) for TPU: the text stack.

Structurally this is DeepSeek-V4 -- multi-hyper-connection residual manifold,
sparse-MLA-shaped full attention, sigmoid/noaux_tc MoE -- with 34 of its 45
attention layers replaced by Kimi Delta Attention. Both halves already exist in
this repo, so this module is mostly composition:

* KDA layers reuse ``models/vllm/kimi_k3/attention.py:KimiDeltaAttention``
  unchanged. It reads exactly the keys GLM's ``linear_attn_config`` provides
  (``head_dim``, ``num_heads``, ``short_conv_kernel_size``,
  ``gate_lower_bound``), defaults to the low-rank gate GLM uses, and its
  ``forward(positions, hidden_states)`` signature already matches the MLA
  layer's, so the layer-type choice is confined to ``__init__``.
* Full-attention layers use ``attention.py:Glm5NextMLAAttention``.
* The mHC machinery is DeepSeek-V4's, except the final collapse -- see
  ``layers.py:mhc_collapse_head_mean``.

The full-attention layers normally carry a kpool sparse indexer. Without
one they run as dense causal MLA, which is exact only up to ``index_topk``
tokens of context; :func:`_check_context_within_dense_mla_bound` refuses to
build past that rather than generating fluent nonsense.

Design notes: ``docs/models/glm5-3-flash.md``.
"""

import os
import re

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (PPMissingLayer, make_layers,
                                              maybe_prefix)

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (  # noqa: E501
    get_mhc_ops, get_mhc_post_op)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP
from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE
from vllm_torchtpu.models.vllm.glm5_next.attention import Glm5NextMLAAttention
from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    assert_indexer_weights_loaded, indexer_parameter_names)
from vllm_torchtpu.models.vllm.glm5_next.layers import mhc_collapse_head_mean
from vllm_torchtpu.models.vllm.kimi_k3.attention import (KimiDeltaAttention,
                                                         kda_state_dtype)

logger = init_logger(__name__)

_LINEAR_ATTENTION = "linear_attention"


def text_config_of(config):
    """GLM ships a multimodal outer config; the text half is what we build."""
    return getattr(config, "text_config", config)


def as_text_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """A view of ``vllm_config`` whose ``hf_config`` is the text config.

    Reused components -- ``DeepseekV4MoE``, ``DeepseekV4MLP``,
    ``Glm5NextMLAAttention`` -- read ``vllm_config.model_config.hf_config``
    directly and expect flat fields like ``hidden_size``. GLM's top-level
    config is the multimodal ``Glm5NextConfig``, which has none of them, so
    hand every submodule the text half rather than teaching each one to unwrap.
    """
    hf_config = vllm_config.model_config.hf_config
    text_config = text_config_of(hf_config)
    if text_config is hf_config:
        return vllm_config
    # `text_config` carries no `architectures`, and vLLM re-validates the
    # derived ModelConfig -- an empty list trips `architectures[0]` while it
    # probes for a Transformers-backend fallback. Carry the name across.
    return vllm_config.with_hf_config(text_config,
                                      architectures=["Glm5NextForCausalLM"])


def is_linear_attention_layer(text_config, layer_idx: int) -> bool:
    """Whether layer ``layer_idx`` is KDA rather than full attention.

    ``layer_types`` is authoritative and 0-indexed. GLM also ships
    ``linear_attn_config["kda_layers"]``, which is *also* 0-indexed -- unlike
    the identically-named Kimi-Linear field, which counts from 1. Prefer
    ``layer_types`` so that ambiguity never has to be resolved here; see
    ``vllm_torchtpu/__init__.py:_patch_vllm_kimi_kda_layer_counts``.
    """
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return layer_types[layer_idx] == _LINEAR_ATTENTION

    linear_attn_config = getattr(text_config, "linear_attn_config", None) or {}
    if "kda_layers" in linear_attn_config:
        return layer_idx in set(linear_attn_config["kda_layers"])

    raise ValueError(
        "glm5_next config declares neither layer_types nor "
        "linear_attn_config['kda_layers']; cannot tell KDA layers from "
        "full-attention ones.")


#: Set to "1" to run past the dense-MLA bound anyway. Only sane for measuring
#: performance, never for judging output.
_ALLOW_UNSOUND_CONTEXT = "VLLM_TORCHTPU_GLM5_ALLOW_LONG_CONTEXT"


def _check_context_within_dense_mla_bound(vllm_config,
                                          text_config,
                                          *,
                                          indexer_enabled: bool
                                          | None = None) -> None:
    """Refuse a context length where this build is silently wrong.

    The bound belongs to the *dense substitution*, not to the model. Running
    the full-attention layers as dense causal MLA is exact only while the
    indexer would have selected every causal token, which holds when
    ``max_model_len <= index_topk``: a query at position ``p`` has
    ``floor((p+1)/index_kpool)`` candidate pools against a budget of
    ``index_topk // index_kpool``. Once the prompt outgrows that, the
    reference model attends sparsely and a dense build does not.

    With the kpool indexer built, longer contexts are *legal* and the bound
    does not apply. With it off -- the kill switch, an unsupported config, a
    fallback -- they are still silently wrong, and that is the case this
    function exists for. Deleting it along with the cap would remove the only
    thing standing between "the indexer got turned off for a baseline
    measurement" and "the baseline measurement is fluent nonsense".

    Args:
        indexer_enabled: ``None`` asks the same resolver the attention layers
            will use, so the guard and the layers cannot disagree about the
            env. :func:`assert_indexer_enablement_uniform` re-checks that
            against the layers actually built.

    The failure is silent -- plausible tokens, wrong attention -- so it is
    worth an exception rather than a warning. Raising at construction also
    catches it before a 328 GB checkpoint load.
    """
    if indexer_enabled is None:
        from vllm_torchtpu.models.vllm.glm5_next.attention import \
            kpool_indexer_enabled
        indexer_enabled = kpool_indexer_enabled()

    index_topk = getattr(text_config, "index_topk", None)
    max_len = getattr(vllm_config.model_config, "max_model_len", None)
    if not index_topk or not max_len or max_len <= index_topk:
        return
    if indexer_enabled:
        # Legal, and the point of the whole exercise. Logged rather than
        # silent: this is the one line that says the long-context run is
        # relying on the indexer being correct rather than on the identity.
        logger.info(
            "glm5_next: max_model_len=%d exceeds index_topk=%d; the kpool "
            "indexer is built, so the full-attention layers select sparsely "
            "and this is sound. (Dense-MLA builds are refused past %d.)",
            max_len, index_topk, index_topk)
        return
    if os.environ.get(_ALLOW_UNSOUND_CONTEXT) == "1":
        logger.warning(
            "glm5_next: max_model_len=%d exceeds index_topk=%d with the kpool "
            "indexer OFF, so the full-attention layers will attend densely "
            "where the reference attends sparsely. Output beyond %d tokens is "
            "WRONG. Continuing because %s=1.", max_len, index_topk, index_topk,
            _ALLOW_UNSOUND_CONTEXT)
        return
    from vllm_torchtpu.models.vllm.glm5_next.attention import KPOOL_INDEXER_ENV
    raise ValueError(
        f"glm5_next: max_model_len={max_len} exceeds index_topk={index_topk} "
        "and the kpool indexer is NOT built, so the full-attention layers run "
        f"as dense causal MLA -- exact only at or below {index_topk} tokens, "
        "and fluent-but-wrong above it. Either build the indexer (unset "
        f"{KPOOL_INDEXER_ENV}, or set it to 1), lower max_model_len to "
        f"{index_topk}, or set {_ALLOW_UNSOUND_CONTEXT}=1 if you are "
        "measuring performance and do not care that the output is wrong.")


def collect_indexer_enablement(module) -> dict[str, bool]:
    """``{prefix: indexer_enabled}`` for every built ``Glm5NextMLAAttention``.

    Walks the module tree rather than the config so it reports what exists,
    not what was intended.
    """
    return {
        (getattr(sub, "prefix", None) or name): bool(sub.indexer_enabled)
        for name, sub in module.named_modules()
        if isinstance(sub, Glm5NextMLAAttention)
    }


def assert_indexer_enablement_uniform(per_layer: dict[str, bool],
                                      assumed: bool) -> None:
    """Every DSA layer agrees with itself, and with the context-bound guard.

    Two failure shapes, both silent, both cheap to exclude:

    * a **mixed** stack -- some layers sparse, some dense -- reads as a model
      that retrieves on 6 of its 11 full-attention layers. Nothing raises;
      the KV caches are per-layer, the shapes all work, and the only symptom
      is worse-than-expected long-context accuracy.
    * a stack that disagrees with :func:`_check_context_within_dense_mla_bound`
      means the bound was evaluated against a different answer than the one
      the layers got, which is exactly how a dense build slips past a guard
      that was written to catch it.
    """
    if not per_layer:
        return
    values = set(per_layer.values())
    if len(values) > 1:
        sparse = sorted(n for n, v in per_layer.items() if v)
        dense = sorted(n for n, v in per_layer.items() if not v)
        raise ValueError(
            "glm5_next: the full-attention stack is mixed -- "
            f"{len(sparse)} layer(s) with a kpool indexer and {len(dense)} "
            f"without. sparse={sparse[:4]}, dense={dense[:4]}. Every DSA "
            "layer must make the same choice; a partial stack retrieves on "
            "some layers and not others, with no error and no shape "
            "mismatch.")
    actual = values.pop()
    if actual != assumed:
        raise ValueError(
            "glm5_next: the dense-MLA context bound was evaluated assuming "
            f"indexer_enabled={assumed}, but the {len(per_layer)} DSA layers "
            f"were built with indexer_enabled={actual}. The guard and the "
            "model disagree, so whichever of the two is right, the other is "
            "not protecting anything.")


def _is_sparse_mlp_layer(text_config, layer_idx: int) -> bool:
    mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
    if mlp_layer_types:
        return mlp_layer_types[layer_idx] == "sparse"
    # Fall back to the DeepSeek-style dense prefix count.
    return layer_idx >= getattr(text_config, "first_k_dense_replace", 3)


class Glm5NextDecoderLayer(nn.Module):
    """One decoder layer: mHC seam, KDA *or* MLA, then dense MLP or MoE."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        text_config = text_config_of(vllm_config.model_config.hf_config)
        # Everything below builds against the text config; see
        # `as_text_vllm_config`.
        vllm_config = as_text_vllm_config(vllm_config)

        self.prefix = prefix
        match = re.search(r"layers\.(\d+)", prefix or "")
        self.layer_idx = int(match.group(1)) if match else -1

        self.hidden_size = text_config.hidden_size
        self.rms_norm_eps = text_config.rms_norm_eps

        self.is_linear_attention = is_linear_attention_layer(
            text_config, self.layer_idx)
        if self.is_linear_attention:
            # Reused verbatim from Kimi-K3: same recurrence, same operand
            # contract, and it reads GLM's `linear_attn_config` as-is.
            self.self_attn = KimiDeltaAttention(text_config,
                                                vllm_config,
                                                prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = Glm5NextMLAAttention(vllm_config,
                                                  prefix=f"{prefix}.self_attn")

        if _is_sparse_mlp_layer(text_config, self.layer_idx):
            self.mlp = DeepseekV4MoE(
                vllm_config,
                prefix=f"{prefix}.mlp",
                # GLM has routed_scaling_factor 2.5 and a *separate* shared
                # expert. Scaling the combined output would scale the shared
                # expert too; put the factor on the router's top-k weights
                # instead, which is what transformers' DeepseekV3TopkRouter
                # does. See DeepseekV4MoE.__init__.
                apply_routed_scale_to_output=False,
                # GLM is topk_method="noaux_tc", so expert *selection* uses
                # sigmoid(logits) + e_score_correction_bias while the weights
                # come from the unbiased scores. vLLM only applies that bias
                # inside the grouped-top-k router, so without this the bias is
                # ignored and the wrong experts run.
                use_grouped_topk=True,
            )
        else:
            self.mlp = DeepseekV4MLP(
                hidden_size=self.hidden_size,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=getattr(text_config, "swiglu_limit", None),
            )

        self.input_layernorm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size,
                                                self.rms_norm_eps)

        self.hc_mult = text_config.hc_mult
        self.hc_sinkhorn_iters = text_config.hc_sinkhorn_iters
        self.hc_eps = text_config.hc_eps
        # Not a config field on either model; DeepSeek-V4 upstream hardcodes 2.0
        # and GLM subclasses its hyper-connection unchanged.
        self.hc_post_alpha = getattr(text_config, "hc_post_alpha", 2.0)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        for name, shape in (
            ("hc_attn_fn", (mix_hc, hc_dim)),
            ("hc_ffn_fn", (mix_hc, hc_dim)),
            ("hc_attn_base", (mix_hc, )),
            ("hc_ffn_base", (mix_hc, )),
            ("hc_attn_scale", (3, )),
            ("hc_ffn_scale", (3, )),
        ):
            setattr(
                self, name,
                nn.Parameter(torch.empty(*shape, dtype=torch.float32),
                             requires_grad=False))

    @property
    def mhc_ops(self):
        """This layer's mHC ops, built once outside any compiled region."""
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            ops = get_mhc_ops(self.rms_norm_eps, self.hc_eps, self.hc_eps,
                              self.hc_post_alpha, self.hc_sinkhorn_iters)
            object.__setattr__(self, "_mhc_ops_instance", ops)
        return ops

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ):
        """Each sublayer's post is deferred into the next pre, as in DSv4.

        The caller carries ``(residual, post_mix, res_mix)`` -- the recombine
        the previous sublayer still owes -- and closes the chain after the last
        layer.
        """
        if residual is None:
            residual = x
            post_mix, res_mix, x = self.mhc_ops.pre(x, self.hc_attn_fn,
                                                    self.hc_attn_scale,
                                                    self.hc_attn_base)
        else:
            residual, post_mix, res_mix, x = self.mhc_ops.fused(
                x, residual, post_mix, res_mix, self.hc_attn_fn,
                self.hc_attn_scale, self.hc_attn_base)

        x = self.input_layernorm(x)
        x = self.self_attn(positions, x)

        residual, post_mix, res_mix, x = self.mhc_ops.fused(
            x, residual, post_mix, res_mix, self.hc_ffn_fn, self.hc_ffn_scale,
            self.hc_ffn_base)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x, residual, post_mix, res_mix


class Glm5NextModel(nn.Module):
    """Embedding, the hybrid layer stack, the mHC collapse, and the norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = text_config_of(config)
        self.config = text_config

        self.hc_mult = text_config.hc_mult
        self.hidden_size = text_config.hidden_size
        self.hc_dim = self.hc_mult * self.hidden_size
        self.rms_norm_eps = text_config.rms_norm_eps

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(text_config.hidden_size,
                                eps=text_config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    @property
    def mhc_post_op(self):
        op = self.__dict__.get("_mhc_post_op_instance")
        if op is None:
            op = get_mhc_post_op()
            object.__setattr__(self, "_mhc_post_op_instance", op)
        return op

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (inputs_embeds if inputs_embeds is not None else
                         self.embed_input_ids(input_ids))

        # Lift one vector into `hc_mult` identical residual streams.
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, post_mix, res_mix, residual)

        # Settle the post the last layer deferred, then collapse the manifold.
        hidden_states = self.mhc_post_op(hidden_states, residual, post_mix,
                                         res_mix)
        hidden_states = mhc_collapse_head_mean(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Glm5NextForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Text-only GLM-5.3-Flash."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = text_config_of(config)
        self.config = text_config
        self.vllm_config = vllm_config

        # 45 layers plus the mHC seam blows CPython's default frame limit while
        # tracing ahead-of-time.
        import sys
        sys.setrecursionlimit(50000)

        from vllm_torchtpu.models.vllm.glm5_next.attention import \
            kpool_indexer_enabled
        indexer_enabled = kpool_indexer_enabled()
        _check_context_within_dense_mla_bound(vllm_config,
                                              text_config,
                                              indexer_enabled=indexer_enabled)

        self.model = Glm5NextModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))

        # The bound above was evaluated against a *prediction* of how the
        # layers would be built; this is the layers themselves.
        assert_indexer_enablement_uniform(
            collect_indexer_enablement(self.model), indexer_enabled)

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(text_config.vocab_size)

        # Whether any attention layer actually carries kpool indexer
        # parameters. Read once here rather than per weight name: it is a walk
        # over ~1e4 parameter paths and `_should_skip` runs per checkpoint
        # tensor. See `_should_skip`.
        self._has_indexer_params = bool(indexer_parameter_names(self))

    # --- hybrid (Mamba-style) cache declarations -------------------------
    #
    # These three must agree with what `KimiDeltaAttention` reports per layer:
    # vLLM sizes the pool from the class methods and allocates from the
    # layer's, so a disagreement is a silent mis-size.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return kda_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = text_config_of(vllm_config.model_config.hf_config)
        kda = text_config.linear_attn_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_heads = int(kda["num_heads"]) // tp_size
        head_dim = int(kda["head_dim"])
        conv_size = int(kda["short_conv_kernel_size"])
        return (
            (conv_size - 1, 3, num_heads, head_dim),
            (num_heads, head_dim, head_dim),
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        # No `vllm_config`: vLLM calls this bare, unlike the dtype and shape
        # hooks above (`interfaces.py:982`).
        from vllm.model_executor.layers.mamba.mamba_utils import \
            MambaStateCopyFuncCalculator
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    # --- weight loading ---------------------------------------------------

    #: Checkpoint prefix rewrites.
    #:
    #: Only the multimodal wrapper prefix needs stripping. Submodules are
    #: deliberately named exactly as the checkpoint names them -- ``self_attn``,
    #: ``mlp``, ``input_layernorm``, ``post_attention_layernorm`` -- rather
    #: than something tidier, because ``modules_to_not_convert`` is a list of
    #: *checkpoint* paths and ``is_layer_skipped`` matches it against *module*
    #: prefixes, exactly. Renaming ``self_attn`` to ``attn`` would silently
    #: stop all 34 bf16 KDA layers from being recognised as unquantized and
    #: they would be handed to the FP8 method.
    _NAME_REWRITES = (("model.language_model.", "model."), )

    #: Fusions this model performs that the checkpoint does not, so
    #: ``is_layer_skipped`` can expand a fused prefix into the shards named in
    #: ``modules_to_not_convert``. Every shard of a fusion must share a
    #: precision or vLLM raises; that holds here -- the KDA fusions are wholly
    #: unquantized, and ``q_a_proj`` / ``kv_a_proj_with_mqa`` are both FP8.
    packed_modules_mapping = {
        "fused_qkvb_proj": ["q_proj", "k_proj", "v_proj", "b_proj"],
        "fused_fa_ga_proj": ["f_a_proj", "g_a_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    #: ``(fused_param, checkpoint_param, shard_id)``. The KDA layer keeps
    #: Kimi's fusions -- q/k/v/b into one projection and the two low-rank gate
    #: stems into another -- and the MLA layer fuses the q and kv down-
    #: projections the way DeepSeek-V4 does.
    _STACKED = (
        ("self_attn.fused_qkvb_proj", "self_attn.q_proj", 0),
        ("self_attn.fused_qkvb_proj", "self_attn.k_proj", 1),
        ("self_attn.fused_qkvb_proj", "self_attn.v_proj", 2),
        ("self_attn.fused_qkvb_proj", "self_attn.b_proj", 3),
        ("self_attn.fused_fa_ga_proj", "self_attn.f_a_proj", 0),
        ("self_attn.fused_fa_ga_proj", "self_attn.g_a_proj", 1),
        ("self_attn.fused_qkv_a_proj", "self_attn.q_a_proj", 0),
        ("self_attn.fused_qkv_a_proj", "self_attn.kv_a_proj_with_mqa", 1),
        ("mlp.gate_up_proj", "mlp.gate_proj", 0),
        ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ("mlp.shared_experts.gate_up_proj", "mlp.shared_experts.gate_proj", 0),
        ("mlp.shared_experts.gate_up_proj", "mlp.shared_experts.up_proj", 1),
    )

    def _rewrite(self, name: str) -> str:
        for src, dst in self._NAME_REWRITES:
            name = name.replace(src, dst)
        return name

    def _should_skip(self, name: str) -> bool:
        """Weights intentionally not built for functional bringup."""
        if name.startswith("model.visual.") or ".visual." in name:
            return True  # vision tower
        # The MTP layer sits one past the decoder stack (index 45 of 45). A
        # speculative draft head is a *separate* vLLM model with its own
        # load_weights pass, so this model has no parameter to put its 1760
        # tensors in; without the skip they would all become unmapped
        # fall-throughs.
        #
        # This must come *before* the indexer check below: layer 45 ships its
        # own seven indexer tensors, and they belong to the draft head, not
        # here. Ordering it the other way makes 7 of the draft head's tensors
        # look like the target's, where they find no parameter and vanish.
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) >= self.config.num_hidden_layers:
            return True
        # The checkpoint's sparse-indexer tensors, seven per DSA layer.
        # Skipped only while nothing in the tree can hold them -- i.e. when
        # the indexer is switched off and these layers run as dense MLA.
        # Keying on the built module tree rather than on a constant means the
        # skip cannot outlive the reason for it; letting the names fall
        # through `params_dict.get(name) is None` instead would have the same
        # effect with no record of why. `load_weights` then asserts every
        # built indexer parameter really was filled.
        if ".indexer." in name:
            return not self._has_indexer_params
        return False

    def load_weights(self, weights):
        from vllm.model_executor.layers.fused_moe import \
            fused_moe_make_expert_params_mapping
        from vllm.model_executor.model_loader.weight_utils import \
            default_weight_loader

        params_dict = dict(self.named_parameters())
        # GLM names its expert projections gate/up/down, unlike DeepSeek-V4's
        # w1/w3/w2.
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if self._should_skip(name):
                continue
            name = self._rewrite(name)

            for fused, ckpt, shard_id in self._STACKED:
                if ckpt not in name or ".experts." in name:
                    continue
                mapped = name.replace(ckpt, fused)
                param = params_dict.get(mapped)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in (
                        expert_mapping):
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    param = params_dict.get(mapped)
                    if param is None:
                        continue
                    param.weight_loader(param,
                                        loaded_weight,
                                        mapped,
                                        shard_id=shard_id,
                                        expert_id=expert_id)
                    loaded.add(mapped)
                    break
                else:
                    param = params_dict.get(name)
                    if param is None:
                        continue
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded.add(name)

        assert_indexer_weights_loaded(self, loaded)
        return loaded
