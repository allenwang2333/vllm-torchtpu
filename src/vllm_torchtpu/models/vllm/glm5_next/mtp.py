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
"""GLM-5.3-Flash multi-token-prediction (MTP) draft head for TPU.

The checkpoint carries one nextn layer at index 45, one past the 45-layer
decoder stack. ``transformers`` 5.16.1 does not model it -- it drops the
tensors with ``_keys_to_ignore_on_load_unexpected = [r"layers\\.45\\.", ...]``
(``modeling_glm5_next.py:1359``) -- so the weight names are the only
specification, and three of them contradict what a copy of
``glm4_moe_mtp.py`` would build:

* Layer 45's ``self_attn`` is **full attention** (DSA/MLA), not KDA. It has
  ``q_a_proj`` / ``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_b_proj`` /
  ``o_proj`` plus a complete ``indexer.*`` block, and not one of the fourteen
  KDA markers (``q_conv1d``, ``A_log``, ``dt_bias``, ``f_a_proj``, ...). Note
  that neither ``layer_types`` (length 45) nor
  ``linear_attn_config["kda_layers"]`` covers index 45, so a lookup there
  would read index 44 -- ``"linear_attention"`` -- and silently build a KDA
  draft head whose weights do not exist.
* Layer 45 has **no hyper-connection tensors**. Every layer 0..44 ships
  ``hc_attn_{base,fn,scale}`` and ``hc_ffn_{base,fn,scale}``; layer 45 ships
  none, so the block runs a plain pre-norm residual rather than the mHC seam
  in ``model.py:Glm5NextDecoderLayer``.
* ``shared_head.head.weight`` does not exist -- only ``shared_head.norm``.
  With ``tie_word_embeddings: false`` the draft's LM head is a copy of the
  top-level ``lm_head.weight``.

The draft head itself keeps no linear-attention state -- it is one full-
attention block -- but the *target* stack does, and 34 of its 45 layers are
KDA. Rolling back a rejected draft therefore depends on the KDA kernel keeping
one recurrent-state checkpoint per verify-window position; see
``kernels/gdn/v3/compute_kda.py`` and ``docs/models/glm5-3-flash-mtp.md``.
"""

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE
from vllm_torchtpu.models.vllm.glm5_next.attention import Glm5NextMLAAttention
from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    assert_indexer_weights_loaded, indexer_parameter_names)
from vllm_torchtpu.models.vllm.glm5_next.model import (
    Glm5NextForCausalLM, _check_context_within_dense_mla_bound,
    as_text_vllm_config, assert_indexer_enablement_uniform,
    collect_indexer_enablement, text_config_of)

logger = init_logger(__name__)

#: Names directly under ``model.layers.<mtp>.`` that belong to the MTP wrapper
#: rather than to the decoder block. Everything else on that layer is block
#: state and gets ``.mtp_block`` spliced in; see ``_rewrite_layer_name``.
_MTP_WRAPPER_NAMES = ("enorm", "hnorm", "eh_proj", "shared_head")


class Glm5NextSharedHead(nn.Module):
    """The checkpoint's ``shared_head``: a norm, and nothing else.

    Upstream's ``glm4_moe_mtp.py:SharedHead`` also owns a ``ParallelLMHead``
    and its loader redirects ``lm_head.weight`` into it. GLM-5.3-Flash ships
    no ``layers.45.shared_head.head.weight``, so the projection lives on the
    model instead (``Glm5NextMTP.lm_head``) where it can keep the ``lm_head``
    quantization prefix the FP8 ignore list actually names.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm5NextMTPBlock(nn.Module):
    """Layer 45's decoder block: dense MLA plus sparse MoE, plain residual.

    Deliberately not ``Glm5NextDecoderLayer``. That class allocates six mHC
    parameters per layer and threads a deferred-post residual manifold through
    its caller; layer 45 has no ``hc_*`` tensors, so instantiating it here
    would either leave six parameters uninitialised or fail the load, and the
    manifold would have no weights to mix with.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        text_config = text_config_of(vllm_config.model_config.hf_config)
        vllm_config = as_text_vllm_config(vllm_config)

        hidden_size = text_config.hidden_size
        eps = text_config.rms_norm_eps

        # Full attention, from the tensor signature -- see the module
        # docstring. Built by the *same* class as layers 3..43, so it cannot
        # end up configured differently from the target stack it drafts for.
        self.self_attn = Glm5NextMLAAttention(vllm_config,
                                              prefix=f"{prefix}.self_attn")

        # Sparse, with all 288 routed experts and the shared one.
        # `mlp_layer_types` has length 45 and `first_k_dense_replace` only
        # describes the leading dense prefix, so neither can be consulted for
        # this layer; the checkpoint's `mlp.experts.{0..287}` settles it.
        self.mlp = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.mlp",
            apply_routed_scale_to_output=False,
            use_grouped_topk=True,
        )

        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    """One nextn layer: the embed/hidden merge, the block, the head norm."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        text_config = text_config_of(vllm_config.model_config.hf_config)
        hidden_size = text_config.hidden_size
        eps = text_config.rms_norm_eps

        self.enorm = RMSNorm(hidden_size, eps=eps)
        self.hnorm = RMSNorm(hidden_size, eps=eps)
        # `eh_proj.weight` is BF16 [4096, 8192] with no `weight_scale_inv`, and
        # `model.layers.45.eh_proj` is in `modules_to_not_convert`. Routing it
        # through a vLLM linear (rather than `nn.Linear`, as upstream does)
        # keeps it on the same TPU linear method as the rest of the port and
        # lets `is_layer_skipped` see the checkpoint-shaped prefix; the
        # exemption is then checked rather than assumed.
        self.eh_proj = ReplicatedLinear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = Glm5NextSharedHead(hidden_size, eps)
        # The block's prefix is the *layer's*, not `{prefix}.mtp_block`: the
        # FP8 ignore list names `model.layers.45.self_attn.kv_b_proj` and
        # friends, and `is_layer_skipped` matches those against the prefix a
        # module was constructed with. The `mtp_block` segment exists only in
        # the Python attribute path, and the loader splices it in.
        self.mtp_block = Glm5NextMTPBlock(vllm_config, prefix=prefix)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Position 0 has no predecessor to condition on; every GLM MTP in vLLM
        # zeroes its embedding rather than feeding a token the draft would be
        # predicting from nothing.
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0.0, inputs_embeds)

        # Embeds first. Settled against the checkpoint, not by convention:
        # `eh_proj.weight[:, 4096:]` is ~1.78*I (cos with the identity 0.796,
        # diagonal 86x the off-diagonal rms) while the first half has no
        # diagonal structure at all (cos 0.018). A residual pass-through can
        # only be the previous hidden state, so it is the SECOND block.
        # Reversing this is silently wrong -- the shapes still match, and the
        # only symptom is a collapsed acceptance rate.
        hidden_states = self.eh_proj(
            torch.cat([
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_states)
            ],
                      dim=-1))[0]

        return self.mtp_block(positions, hidden_states)


class Glm5NextMultiTokenPredictor(nn.Module):
    """The nextn layers, keyed by their absolute checkpoint layer index."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        text_config = text_config_of(vllm_config.model_config.hf_config)
        self.config = text_config

        self.mtp_start_layer_idx = text_config.num_hidden_layers
        self.num_mtp_layers = getattr(text_config, "num_nextn_predict_layers",
                                      0)
        if not self.num_mtp_layers:
            raise ValueError(
                "glm5_next MTP was requested but the config declares "
                f"num_nextn_predict_layers={self.num_mtp_layers}. There is no "
                "draft layer in the checkpoint to load.")

        # String keys, so the weight name's layer index addresses the module
        # directly and a checkpoint that numbered its nextn layers differently
        # would miss rather than silently land on the wrong one.
        self.layers = nn.ModuleDict({
            str(idx):
            Glm5NextMultiTokenPredictorLayer(vllm_config,
                                             prefix=f"{prefix}.layers.{idx}")
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })

        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def _layer(self, spec_step_idx: int) -> Glm5NextMultiTokenPredictorLayer:
        step = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + step)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self._layer(spec_step_idx)(positions, previous_hidden_states,
                                          inputs_embeds)


class Glm5NextMTP(nn.Module):
    """GLM-5.3-Flash's draft head, as a standalone vLLM model."""

    #: Same fusions as the target, so ``is_layer_skipped`` can expand a fused
    #: prefix back into the checkpoint shards named in
    #: ``modules_to_not_convert``. Shared with the target rather than copied:
    #: the two must agree or layer 45 is classified differently from layers
    #: 3..43 built from the same tensors.
    packed_modules_mapping = Glm5NextForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        text_config = text_config_of(vllm_config.model_config.hf_config)
        self.config = text_config
        self.vllm_config = vllm_config

        # The draft's MLA layer is built exactly like the target's -- same
        # class -- so the same soundness bound applies to it, and for the same
        # reason: dense MLA without an indexer is exact only at or below
        # `index_topk`.
        from vllm_torchtpu.models.vllm.glm5_next.attention import \
            kpool_indexer_enabled
        indexer_enabled = kpool_indexer_enabled()
        _check_context_within_dense_mla_bound(vllm_config,
                                              text_config,
                                              indexer_enabled=indexer_enabled)

        self.model = Glm5NextMultiTokenPredictor(vllm_config=vllm_config,
                                                 prefix=maybe_prefix(
                                                     prefix, "model"))

        # Same guard as `Glm5NextForCausalLM.__init__`, and needed here for a
        # sharper reason: this module reads the kill switch twice by
        # independent routes -- once above for the bound, once inside the
        # layer it builds -- and nothing else ties the two answers together.
        # A draft head that went dense while the target went sparse is a
        # collapsed acceptance rate and no error.
        assert_indexer_enablement_uniform(
            collect_indexer_enablement(self.model), indexer_enabled)

        # Prefix `lm_head`, not `model.layers.45.shared_head.head`: the
        # checkpoint has no such tensor, this head is loaded from the
        # top-level `lm_head.weight`, and `lm_head` is the entry the FP8
        # ignore list carries. Naming it anything else quantizes a bf16 head.
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(text_config.vocab_size)

        # See `Glm5NextForCausalLM.__init__` and `_rewrite_layer_name`.
        self._has_indexer_params = bool(indexer_parameter_names(self))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states, inputs_embeds,
                          spec_step_idx)

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor:
        layer = self.model._layer(spec_step_idx)
        return self.logits_processor(self.lm_head,
                                     layer.shared_head(hidden_states))

    # --- weight loading ---------------------------------------------------

    def _rewrite_layer_name(self, name: str) -> str | None:
        """Map a checkpoint name onto this model, or ``None`` to drop it.

        The draft is handed the whole checkpoint, so most of it is somebody
        else's: layers 0..44, the target's final norm, and the vision tower
        all belong to ``Glm5NextForCausalLM``.
        """
        for src, dst in Glm5NextForCausalLM._NAME_REWRITES:
            name = name.replace(src, dst)

        if name == "lm_head.weight":
            return name
        if name == "model.embed_tokens.weight":
            return name
        # `getattr`, because `test_glm5_next_mtp.py` drives this mapper
        # against a stand-in that carries only the layer index -- constructing
        # a real `Glm5NextMTP` needs a distributed group and a 288-expert MoE.
        if ".indexer." in name and not getattr(self, "_has_indexer_params",
                                               False):
            # Dropped only while the draft head has no indexer to put them
            # in. Making this skip unconditional would be a trap: the draft
            # head would stay dense after the target went sparse, and the only
            # symptom would be a collapsed acceptance rate. Layer 45 carries
            # its own complete seven-tensor indexer block in the checkpoint
            # (shard 2, all BF16), so once an indexer is built the names must
            # reach it -- and `load_weights` asserts they did.
            return None

        prefix = f"model.layers.{self.model.mtp_start_layer_idx}."
        if not name.startswith(prefix):
            return None

        tail = name[len(prefix):]
        if tail.startswith(_MTP_WRAPPER_NAMES):
            return name
        return f"{prefix}mtp_block.{tail}"

    def load_weights(self, weights) -> set[str]:
        from vllm.model_executor.layers.fused_moe import \
            fused_moe_make_expert_params_mapping
        from vllm.model_executor.model_loader.weight_utils import \
            default_weight_loader

        params_dict = dict(self.named_parameters())
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            mapped_name = self._rewrite_layer_name(name)
            if mapped_name is None:
                continue
            name = mapped_name

            for fused, ckpt, shard_id in Glm5NextForCausalLM._STACKED:
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
