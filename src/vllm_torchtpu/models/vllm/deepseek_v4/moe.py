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
"""DeepSeek-V4 mixture-of-experts layer."""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.models.utils import extract_layer_index

from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP


class DeepseekV4MoE(nn.Module):
    """DeepSeek-V4 MoE layer with dynamic hash routing and fused routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        apply_routed_scale_to_output: bool = True,
        use_grouped_topk: bool = False,
    ):
        """
        Args:
          use_grouped_topk: route through vLLM's grouped-top-k router. This is
            not only about expert *groups*: ``e_score_correction_bias`` is
            consumed **exclusively** inside
            ``fused_moe/router/grouped_topk_router.py``, so with the default
            plain router the bias is accepted and then silently ignored, and
            top-k runs on the raw sigmoid scores. Any ``noaux_tc`` model needs
            this True or it selects the wrong experts. Left False by default so
            DeepSeek-V4's behaviour is unchanged.
          apply_routed_scale_to_output: when True the runner multiplies the
            *combined* MoE output by ``routed_scaling_factor``. vLLM only
            compensates a shared expert for that when the shared expert is
            fused into the expert slots (it sets the fused slot's weight to
            ``1/routed_scaling_factor``, see
            ``fused_moe/layer.py:305-317``). This layer passes a **separate**
            shared-expert module, so with a scaling factor other than 1.0 the
            shared expert would be scaled too. DeepSeek-V4 defaults the factor
            to 1.0, which makes the choice a no-op; models that set a real
            factor and keep a separate shared expert -- GLM-5.3-Flash uses 2.5
            -- must pass False, which puts the scale on the router's top-k
            weights instead and leaves the shared expert alone.
        """
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor",
                                             1.0)
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
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
        # `num_hash_layers` is DeepSeek-V4-specific. Other models reusing this
        # MoE (GLM-5.3-Flash) have no hash-routed layers at all, and absent
        # means zero of them.
        num_hash_layers = getattr(config, "num_hash_layers", 0)
        is_hash_moe = extract_layer_index(prefix) < num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
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
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
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
            use_grouped_topk=use_grouped_topk,
            num_expert_group=(getattr(config, "n_group", None)
                              if use_grouped_topk else None),
            topk_group=(getattr(config, "topk_group", None)
                        if use_grouped_topk else None),
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
        )

        # The monolithic TPU MoE path never consults the router: MoERunner
        # calls `routed_experts.forward_monolithic`, which hands the
        # `RoutedExperts` layer straight to `quant_method.apply_monolithic`,
        # and `moe_routing.route` looks the table up as
        # `layer.hash_indices_table`. Passing `hash_indices_table` to
        # `FusedMoEFactory` above only reaches the router object, so mirror it
        # onto the experts layer.
        if self.gate.tid2eid is not None:
            object.__setattr__(self.experts.routed_experts,
                               "hash_indices_table", self.gate.tid2eid)

    def forward(self,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass executing router gate and fused expert computation."""
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError(
                "DeepSeek V4 hash MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
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
