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
"""DeepSeek-V4 for TPU.

Registered for ``DeepseekV4ForCausalLM`` from
:func:`vllm_torchtpu.models.vllm.register_models`, so vLLM resolves this
implementation instead of its own.

This is a standalone definition of the model: it builds every submodule itself
and inherits no upstream DeepSeek-V4 class. It is adapted from
``tpu_inference/models/vllm/experimental/deepseek_v4.py``, which is in turn a
device-neutral fork of vLLM's AMD implementation. Where this file departs from
that reference, the divergence is called out at the site with a ``TPU:``
comment; the recurring reasons are

* vLLM API drift -- the reference targets a vLLM whose MoE entry point is
  ``FusedMoE``; on 0.27 it is ``FusedMoEFactory``;
* graph mode -- torch.compile is applied per decoder layer here, whereas
  tpu-inference traces the whole model through JAX;
* op coverage -- tpu-inference registers TPU ``forward_oot`` implementations
  for the MHC custom ops; this stack has none, so ops without a working
  ``forward_native`` are inlined instead.
"""

import re
import sys
import typing
from collections.abc import Callable, Iterable, Iterator
from itertools import islice

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (get_dp_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import (SiluAndMul,
                                                   SiluAndMulWithClamp)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory, GateLinear, fused_moe_make_expert_params_mapping)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, WeightsMapper,
                                              extract_layer_index,
                                              is_pp_missing_parameter,
                                              make_layers, maybe_prefix)
from vllm.model_executor.offloader import NoopOffloader, set_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# TPU: tracing 43 decoder layers nests deeper than CPython's default 1000-frame
# limit. Raised at import, which the lazy registry entry only triggers for a
# DeepSeek-V4 checkpoint. The only other TPU shim this model needs is scoped to
# the code that needs it: the CUDA-event mock in the attention constructor
# (`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py`).
sys.setrecursionlimit(50000)


class DeepseekV4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor",
                                             1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        # TPU: `FusedMoE` on vLLM 0.27 is `FusedMoEFactory`, a function that
        # returns a `MoERunner`. `apply_routed_scale_to_output` is deliberately
        # left at its default (False), which folds `routed_scaling_factor` into
        # the routing weights instead of scaling the combined output: that is
        # what the TPU routing kernels in `layers/vllm/moe_routing.py` assume,
        # and it is the behaviour the MMLU parity run was measured against.
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

        # Upstream vLLM stores the hash routing table on the fused-topk router
        # (`FusedTopKBiasRouter._hash_indices_table`). The TPU MoE apply path is
        # only handed the `RoutedExperts` layer, so mirror the table onto it
        # here so it can be read at forward time (`moe_routing.route` looks it
        # up as `layer.hash_indices_table`).
        #
        # TPU: `object.__setattr__` rather than plain assignment -- the table is
        # an `nn.Parameter`, and `nn.Module.__setattr__` would register it a
        # second time under `experts.routed_experts.hash_indices_table`, so it
        # would be sharded twice and reported as a missing checkpoint key.
        if self.gate.tid2eid is not None:
            object.__setattr__(self.experts.routed_experts,
                               "hash_indices_table", self.gate.tid2eid)

    def forward(self,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError(
                "DeepSeek V4 hash MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the MoERunner class
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        return final_hidden_states.view(org_shape)


# TPU: `support_torch_compile` per decoder layer. tpu-inference needs no
# equivalent because it traces the whole model through JAX; here one graph over
# all 43 layers does not trace, so each layer is compiled on its own.
@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "positions": 0,
        "residual": 0,
        "post_mix": 0,
        "res_mix": 0,
        "input_ids": 0,
    })
class DeepseekV4DecoderLayer(nn.Module):

    # TPU: keyword-only on purpose. `support_torch_compile` wraps `__init__`
    # with a positional-argument type check that zips the passed args against
    # the signature's parameters *including* `self`, so every positional
    # argument is validated against the annotation of the one before it.
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ):
        super().__init__()

        # Registers both torch.ops.vllm.mhc_pre and mhc_post.
        from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreOp

        # TPU: the Pallas sparse-MLA/SWA attention, constructed directly.
        # tpu-inference substitutes the class symbol on the upstream module
        # instead, because its decoder layer is the upstream one.
        from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_attention import \
            VllmDeepseekV4MLAAttention

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        # Distinct per layer, so each gets its own compile cache directory.
        self.prefix = prefix

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = VllmDeepseekV4MLAAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            # TPU: no streams to overlap the attention input GEMMs on.
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    # TPU: only the unfused path exists here. Its fused counterpart is built on
    # `MHCFusedPostPreOp`, which has no `forward_native` and no TPU
    # `forward_oot` on this stack; upstream gates it on tilelang, which is
    # never available on TPU, so the fused path was permanently dead code.
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None]:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale,
                                    self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale,
                                    self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None


def hc_head_fp32(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """HC head reduction, forced to fp32.

    TPU: the reference calls ``HCHeadOp``, relying on the TPU ``forward_oot``
    that tpu-inference registers for it. This stack registers none, and
    ``HCHeadOp.forward_native`` raises ``NotImplementedError``, so the
    reduction is written out here instead.

    ``hc_fn``/``hc_scale``/``hc_base`` are stored as float32, so the projection
    has to run in float32; casting them down to the activation dtype first is a
    dtype mismatch, and doing the reduction in bf16 loses accuracy on a sum
    over ``hc_mult`` streams.
    """
    residual_flat = hidden_states.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(),
                     dim=-2).to(hidden_states.dtype)


# TPU: the head is its own compiled module. The reference runs it inline at the
# end of the backbone, which is fine when the whole model is one JAX trace;
# here only the decoder layers are compiled, so an inline head would fall back
# to eager op-by-op execution.
@support_torch_compile(dynamic_arg_dims={"hidden_states": 0})
class DeepseekV4Head(nn.Module):
    """Final HC head reduction + norm."""

    def __init__(
        self,
        *,
        hc_head_fn: nn.Parameter,
        hc_head_scale: nn.Parameter,
        hc_head_base: nn.Parameter,
        norm: nn.Module,
        rms_norm_eps: float,
        hc_eps: float,
    ):
        super().__init__()
        # One head per model, so a literal is a stable compile cache key.
        self.prefix = "model.head"
        self.hc_head_fn = hc_head_fn
        self.hc_head_scale = hc_head_scale
        self.hc_head_base = hc_head_base
        self.norm = norm
        self.rms_norm_eps = rms_norm_eps
        self.hc_eps = hc_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hc_head_fp32(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.norm(hidden_states)


class DeepseekV4Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # DeepSeek-V4's config selects vLLM's weight-prefetch offloader, which
        # make_layers (below) uses to wrap every decoder layer: it allocates
        # torch.cuda.Event stream-sync events and installs CUDA-stream forward
        # hooks. None of that works on TPU (there is no CUDA) -- it is a GPU-HBM
        # offload optimization with no TPU analog. Force the no-op offloader
        # before make_layers so the layers are built and run unwrapped.
        set_offloader(NoopOffloader())

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = None

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        # TPU: allocated on the device explicitly -- the default device here is
        # CPU, and the indexer kernel needs it resident.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.head = DeepseekV4Head(
            hc_head_fn=self.hc_head_fn,
            hc_head_scale=self.hc_head_scale,
            hc_head_base=self.hc_head_base,
            norm=self.norm,
            rms_norm_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
        )

        # MTP/Eagle draft models are not supported on this path yet;
        # `get_mtp_target_hidden_states` reads this and tolerates None.
        self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _dp_gather_hash_moe_input_ids(self,
                                      input_ids: torch.Tensor) -> torch.Tensor:
        """All-gather ``input_ids`` over the DP group for hash-MoE routing.

        TPU: the first ``num_hash_layers`` MoE layers route on token ids --
        ``hash_indices_table[input_ids]`` -- rather than on gate logits (see
        ``moe_routing._hash_moe_select``). Every other routing scheme derives
        its expert ids from ``router_logits``, which the MoE runner dispatches
        alongside ``hidden_states``; ``input_ids`` is a separate tensor that
        upstream threads through that dispatch untouched.

        With DP > 1 the runner all-gathers ``hidden_states``/``router_logits``
        over the DP group *before* routing, runs the experts on all
        ``dp_size * T`` rows, then reduce-scatters back. Routing therefore
        happens on the global batch, and every rank needs every token's id: the
        ``topk_ids.reshape`` in ``_hash_moe_select`` targets the score row count
        and raises on the mismatch otherwise.

        Gathering here rather than inside the runner is safe because
        ``input_ids`` reaches the experts unmodified, so this is the same tensor
        the runner would have gathered -- just gathered once per forward pass
        instead of once per hash layer. It has to use the same
        ``dp_group.all_gather(dim=0)`` the communicator uses for
        ``hidden_states``, or rows would pair with the wrong ids.
        """
        dp_group = get_dp_group()
        if dp_group.world_size == 1:
            return input_ids
        return dp_group.all_gather(input_ids, dim=0)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) -- V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the head collapses it.
        return IntermediateTensors({
            "hidden_states":
            torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(
                1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # The embedding above indexes the *local* ids; the hash-MoE layers need
        # them gathered across DP. Only the layers this rank holds matter, and
        # hash routing is confined to the first `num_hash_layers`.
        moe_input_ids = input_ids
        if input_ids is not None and self.start_layer < self.config.num_hash_layers:
            moe_input_ids = self._dp_gather_hash_moe_input_ids(input_ids)

        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                moe_input_ids,
                post_mix,
                res_mix,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return self.head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 -> 0),
                    # destroying the raw exponent bytes.
                    if ("weight_scale" in name
                            and loaded_weight.dtype == torch.float8_e8m0fnu):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    name_mapped = None
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = (
                            mapping)
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        if name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(Callable[..., bool],
                                                    param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    if name_mapped is not None:
                        loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    narrow_weight = loaded_weight[
                        head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


def _layer_index(name: str) -> int | None:
    if not name.startswith("layers."):
        return None
    try:
        return int(name.split(".")[1])
    except (ValueError, IndexError):
        return None


def _prepare_deepseek_v4_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    num_hidden_layers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Filter the raw checkpoint down to what the model actually builds.

    This runs *before* :class:`WeightsMapper` renames anything, and only trims:
    layers past ``num_hidden_layers`` are dropped so ``--hf-overrides
    '{"num_hidden_layers": N}'`` works against a full checkpoint, as are the
    MTP blocks this path does not build.

    Nothing else needs doing here. Quantization stays with the linear methods:
    the fp8 attention and shared-expert projections keep their checkpoint
    128x128 block scales all the way to ``VllmFp8LinearMethodTPU``, which
    rebases them to the runtime per-output-channel layout in
    ``process_weights_after_loading`` -- the same path every other fp8 model on
    this backend takes. Module names need no rewriting either: DeepSeek-V4
    checkpoints are already laid out the way this model builds itself
    (``layers.N.attn.wq_a``, ``layers.N.ffn.experts.E.w1``,
    ``layers.N.hc_attn_fn``), so the mapper below covers the rest.
    """
    for name, loaded_weight in weights:
        # The MTP blocks are not built on this path; `AutoWeightsLoader` would
        # drop them anyway, but skipping here avoids materializing weights
        # nothing will read.
        if name.startswith("mtp."):
            continue
        layer_idx = _layer_index(name)
        if layer_idx is not None and layer_idx >= num_hidden_layers:
            continue
        yield name, loaded_weight


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # shared experts use Fp8LinearMethod's block scales, which
        # register as ``weight_scale_inv``.
        #
        # TPU: vLLM 0.27's mapper has a third rule here, rewriting a bare
        # ``.weight_scale`` to ``.weight_scale_inv`` for AMD-Quark
        # checkpoints. It stays out: DeepSeek's own checkpoints ship every
        # scale as ``.scale``, so the two rules below already cover them, and
        # the third would only fire on names this checkpoint never contains.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"):
            r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
                expert_dtype)

        self.model = self.model_cls(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # TPU: scales are registered while the quant methods create weights,
        # which runs after every linear layer's __init__, so their TP status is
        # still unset at this point.
        for _name, module in self.model.named_modules():
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()

        # TPU: drop the MTP blocks and anything past the end of a model
        # truncated with `--hf-overrides num_hidden_layers`.
        prepared = _prepare_deepseek_v4_weights(
            weights,
            num_hidden_layers=self.config.num_hidden_layers,
        )

        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(prepared,
                                            mapper=self.hf_to_vllm_mapper)

        # TPU: ranks finish loading at different times and the next thing they
        # do is collective; without this the fastest rank can enter it while a
        # slower one is still writing weights.
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("Aligning workers at barrier after load_weights.")
            dist.barrier()

        from torch_tpu._internal import sync
        sync.synchronize(wait=True)
        import gc
        gc.collect()

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
