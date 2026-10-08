"""V4.1 model bindings for the shared DeepSeek V4 TPU quantization path."""

from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.models.deepseek_v4.nvidia import model as v4_upstream
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE

from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.upstream_bindings import device_hooks
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP


@register_quantization_config("tpu-deepseek_v41_fp8")
class VllmDeepseekV41Fp8Config(VllmDeepseekV4Fp8Config):
    """Use V4 weight preparation, activation quantization and MoE dispatch."""

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"


class DeepseekV41MoE(DeepseekV4MoE):
    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("V4.1 TPU MoE does not support EPLB")
        if vllm_config.kernel_config.moe_backend in v4_upstream.MEGA_MOE_BACKENDS:
            raise NotImplementedError("V4.1 TPU requires the native V4 MoE backend")
        factory = v4_upstream.FusedMoEFactory

        def v4_factory(**options):
            # Match the native TPU V4 model's scaling and shared-expert combine.
            options["apply_routed_scale_to_output"] = True
            # Run self.gate in _forward_fused_moe so non-SP EP layers (CED=1
            # decoder_stage and DSpark draft layers) can slice gate input to
            # [T // ep_size, H] before MoERunner._forward_impl executes.
            options["gate"] = None
            return factory(**options)

        with device_hooks(
            v4_upstream,
            DeepseekV4MLP=DeepseekV4MLP,
            FusedMoEFactory=v4_factory,
        ):
            super().__init__(vllm_config, *args, **kwargs)

    def _maybe_slice_gate_input(self, hidden_states):
        if self.use_sequence_parallel:
            return hidden_states
        experts = getattr(self, "experts", None)
        moe_config = getattr(experts, "moe_config", None)
        if getattr(moe_config, "pcp_size", 1) > 1 or not getattr(
            getattr(moe_config, "moe_parallel_config", None), "use_ep", False
        ):
            return hidden_states
        routed_experts = getattr(experts, "routed_experts", None)
        quant_method = getattr(routed_experts, "quant_method", None)
        if (
            not getattr(quant_method, "is_monolithic", False)
            or getattr(quant_method, "supports_internal_mk", False)
            or getattr(routed_experts, "custom_routing_function", None) is not None
            or moe_routing._SIMULATION_STRATEGY is not None
        ):
            return hidden_states
        slice_group = (
            get_tp_group() if getattr(moe_config, "dp_size", 1) > 1 else get_ep_group()
        )
        group_size = slice_group.world_size
        num_tokens = hidden_states.shape[0]
        if group_size <= 1 or num_tokens % group_size != 0:
            return hidden_states
        chunk = num_tokens // group_size
        rank = slice_group.rank_in_group
        return hidden_states[rank * chunk : (rank + 1) * chunk]

    def _forward_fused_moe(self, hidden_states, input_ids=None):
        org_shape = hidden_states.shape
        router_logits, _ = self.gate(self._maybe_slice_gate_input(hidden_states))
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        return final_hidden_states.view(org_shape)

    def forward(self, hidden_states, input_ids=None):
        if (
            self.use_sequence_parallel
            and input_ids is not None
            and self.gate.tid2eid is not None
            and self.experts.do_naive_dispatch_combine
        ):
            # Hash routing indexes its table by token id, so the ids must line
            # up row-for-row with the hidden states the experts see. Under SP
            # the runner gathers those over the EP group (TpuDeviceCommunicator)
            # but passes input_ids through as this rank's shard.
            input_ids = get_ep_group().all_gather(input_ids, dim=0)
        return super().forward(hidden_states, input_ids)
