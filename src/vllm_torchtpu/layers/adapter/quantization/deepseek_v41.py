"""V4.1 model bindings for the shared DeepSeek V4 TPU quantization path."""

import types
from collections.abc import Callable

import torch
import torch.nn.functional as F
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

# Marks a layer whose gate-slice plan has not been resolved yet; ``None`` is a
# valid cached plan meaning "never slice".
_UNRESOLVED = object()
_BASE_MOE_FORWARD = DeepseekV4MoE.forward
_COMPILED_TPU_GATE_KERNELS: dict[
    tuple[tuple[int, ...], tuple[int, int] | None, torch.dtype, torch.dtype | None],
    Callable,
] = {}


def _build_tpu_gate_kernel(
    x_shape: tuple[int, ...],
    bounds: tuple[int, int] | None,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype | None,
):
    key = (x_shape, bounds, w_dtype, out_dtype)
    fn = _COMPILED_TPU_GATE_KERNELS.get(key)
    if fn is not None:
        return fn

    def _gate_impl(hidden_states, weight):
        gate_in = (
            hidden_states[bounds[0] : bounds[1]]
            if bounds is not None
            else hidden_states
        )
        if out_dtype is not None and gate_in.dtype != w_dtype:
            gate_in = gate_in.to(w_dtype)
        logits = F.linear(gate_in, weight)
        if out_dtype is not None and logits.dtype != out_dtype:
            logits = logits.to(out_dtype)
        return logits

    name = f"_gate_impl_{len(_COMPILED_TPU_GATE_KERNELS)}"
    specialized = types.FunctionType(
        _gate_impl.__code__.replace(co_name=name),
        _gate_impl.__globals__,
        name,
        _gate_impl.__defaults__,
        _gate_impl.__closure__,
    )
    fn = torch.compile(specialized, backend="tpu", dynamic=False)
    _COMPILED_TPU_GATE_KERNELS[key] = fn
    return fn


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

    def _resolve_gate_slice_plan(self):
        """Return ``(group_size, rank)`` for gate slicing, or ``None``.

        Everything checked here is fixed once the layer and the process
        groups are built, so the answer is cached per layer by the caller.
        """
        experts = getattr(self, "experts", None)
        moe_config = getattr(experts, "moe_config", None)
        if getattr(moe_config, "pcp_size", 1) > 1 or not getattr(
            getattr(moe_config, "moe_parallel_config", None), "use_ep", False
        ):
            return None
        routed_experts = getattr(experts, "routed_experts", None)
        quant_method = getattr(routed_experts, "quant_method", None)
        if (
            not getattr(quant_method, "is_monolithic", False)
            or getattr(quant_method, "supports_internal_mk", False)
            or getattr(routed_experts, "custom_routing_function", None) is not None
        ):
            return None
        slice_group = (
            get_tp_group() if getattr(moe_config, "dp_size", 1) > 1 else get_ep_group()
        )
        group_size = slice_group.world_size
        if group_size <= 1:
            return None
        return group_size, slice_group.rank_in_group

    def _gate_slice_bounds(self, num_tokens: int):
        state = self.__dict__
        # use_sequence_parallel and the simulation strategy can be toggled at
        # runtime (tests, simulation harness), so they stay per-call checks.
        if (
            state.get("use_sequence_parallel", False)
            or moe_routing._SIMULATION_STRATEGY is not None
        ):
            return None
        # Bypass nn.Module.__getattr__/__setattr__: the plan is a plain cache.
        plan = state.get("_gate_slice_plan", _UNRESOLVED)
        if plan is _UNRESOLVED:
            plan = state["_gate_slice_plan"] = self._resolve_gate_slice_plan()
        if plan is None:
            return None
        group_size, rank = plan
        if num_tokens % group_size != 0:
            return None
        chunk = num_tokens // group_size
        start = rank * chunk
        return start, start + chunk

    def _maybe_slice_gate_input(self, hidden_states):
        bounds = self._gate_slice_bounds(hidden_states.shape[0])
        if bounds is None:
            return hidden_states
        return hidden_states[bounds[0] : bounds[1]]

    def _eval_gate_logits(self, gate, hidden_states):
        state = self.__dict__
        cached = state.get("_gate_linear_fast", _UNRESOLVED)
        if cached is _UNRESOLVED:
            weight = getattr(gate, "weight", None)
            if (
                weight is not None
                and getattr(gate, "bias", None) is None
                and getattr(gate, "quant_method", None) is not None
                and not getattr(gate, "allow_dsv3_router_gemm", False)
                and not getattr(gate, "allow_fp32_router_gemm", False)
                and not getattr(gate, "allow_cublas_router_gemm", False)
            ):
                cached = (weight, weight.dtype, getattr(gate, "out_dtype", None))
            else:
                cached = None
            state["_gate_linear_fast"] = cached
        if cached is None:
            router_logits, _ = gate(self._maybe_slice_gate_input(hidden_states))
            return router_logits
        weight, w_dtype, out_dtype = cached
        bounds = self._gate_slice_bounds(hidden_states.shape[0])
        if hidden_states.device.type == "tpu":
            kernel = _build_tpu_gate_kernel(
                hidden_states.shape, bounds, w_dtype, out_dtype
            )
            return kernel(hidden_states, weight)
        gate_in = (
            hidden_states[bounds[0] : bounds[1]]
            if bounds is not None
            else hidden_states
        )
        if out_dtype is not None and gate_in.dtype != w_dtype:
            gate_in = gate_in.to(w_dtype)
        router_logits = F.linear(gate_in, weight)
        if out_dtype is not None and router_logits.dtype != out_dtype:
            router_logits = router_logits.to(out_dtype)
        return router_logits

    def _forward_fused_moe(self, hidden_states, input_ids=None):
        org_shape = hidden_states.shape
        state = self.__dict__
        modules = state.get("_modules")
        gate = modules["gate"] if modules and "gate" in modules else state["gate"]
        experts = (
            modules["experts"] if modules and "experts" in modules else state["experts"]
        )
        router_logits = self._eval_gate_logits(gate, hidden_states)
        final_hidden_states = experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        if final_hidden_states.shape != org_shape:
            return final_hidden_states.view(org_shape)
        return final_hidden_states

    def forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
        state = self.__dict__
        modules = state.get("_modules")
        gate = modules["gate"] if modules and "gate" in modules else state["gate"]
        experts = (
            modules["experts"] if modules and "experts" in modules else state["experts"]
        )
        if (
            state.get("use_sequence_parallel", False)
            and input_ids is not None
            and gate.tid2eid is not None
            and experts.do_naive_dispatch_combine
        ):
            # Hash routing indexes its table by token id, so the ids must line
            # up row-for-row with the hidden states the experts see. Under SP
            # the runner gathers those over the EP group (TpuDeviceCommunicator)
            # but passes input_ids through as this rank's shard.
            input_ids = get_ep_group().all_gather(input_ids, dim=0)
        if DeepseekV4MoE.forward is _BASE_MOE_FORWARD and not state.get(
            "use_mega_moe", False
        ):
            if input_ids is None:
                if gate.tid2eid is not None:
                    raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
                if getattr(gate, "bias_vl", None) is not None:
                    raise ValueError(
                        "DeepSeek V4 vision MoE routing requires input_ids."
                    )
            return self._forward_fused_moe(hidden_states, input_ids)
        if mega_gate_metadata is None:
            return super().forward(hidden_states, input_ids)
        try:
            return super().forward(hidden_states, input_ids, mega_gate_metadata)
        except TypeError:
            return super().forward(hidden_states, input_ids)
