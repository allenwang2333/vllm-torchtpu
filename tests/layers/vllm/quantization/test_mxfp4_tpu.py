import os
import tempfile
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig

from vllm_torchtpu.layers.vllm.quantization.mxfp4 import \
    VllmDeepseekV4Mxfp4MoEMethod


def _build_layer_and_method(device):
    num_experts = 8
    top_k = 2
    hidden_size = 512
    intermediate_size = 2048

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig, RoutingMethodType)
    moe_parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
        pcp_size=1,
        pcp_rank=0,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.bfloat16,
        has_bias=True,
    )

    dummy_vllm_config = SimpleNamespace(
        model_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        lora_config=None,
        device_config=SimpleNamespace(device=device),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            enable_expert_parallel=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
            expert_placement_strategy="linear",
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            static_all_moe_layers=[],
            custom_ops=["all"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
            max_cudagraph_capture_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, ),
    )

    from vllm.config.vllm import set_current_vllm_config
    with set_current_vllm_config(dummy_vllm_config):
        layer = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            activation="silu",
            has_bias=True,
        )
        if hasattr(layer, "routed_experts"):
            layer = layer.routed_experts

        layer.layer_idx = 3
        layer.renormalize = True
        layer.device = device

        # Bypass compile wiring; use dummy top-k router for unit test.
        def _dummy_routing_fn(hidden_states, gating_output, topk, renormalize):
            topk_weights, topk_ids = torch.topk(gating_output, topk, dim=-1)
            topk_weights = torch.softmax(topk_weights.float(), dim=-1)
            return topk_weights.to(hidden_states.dtype), topk_ids.to(
                torch.int32)

        layer.custom_routing_function = _dummy_routing_fn

        method = VllmDeepseekV4Mxfp4MoEMethod(moe_config)
        if "quant_method" in layer._modules:
            del layer._modules["quant_method"]
        layer.quant_method = method
        method.create_weights(
            layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )

        # Populate staging buffers as left by _dsv4_weight_loader.
        layer.w1_temp.copy_(
            torch.randint(0, 256, layer.w1_temp.shape, dtype=torch.uint8))
        layer.w3_temp.copy_(
            torch.randint(0, 256, layer.w3_temp.shape, dtype=torch.uint8))
        layer.w2_temp.copy_(
            torch.randint(0, 256, layer.w2_temp.shape, dtype=torch.uint8))
        # Reassign CPU-created Parameters onto target TPU device.
        layer.w13_weight = torch.nn.Parameter(torch.randint(
            0, 256, layer.w13_weight.shape, dtype=torch.uint8).to(device),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randint(
            0, 256, layer.w2_weight.shape, dtype=torch.uint8).to(device),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.randint(120,
                          135,
                          layer.w13_weight_scale.shape,
                          dtype=torch.uint8).to(device),
            requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.randint(120,
                          135,
                          layer.w2_weight_scale.shape,
                          dtype=torch.uint8).to(device),
            requires_grad=False)
        layer.w13_bias = torch.nn.Parameter(torch.randn(
            layer.w13_bias.shape, dtype=torch.bfloat16).to(device),
                                            requires_grad=False)
        layer.w2_bias = torch.nn.Parameter(torch.randn(
            layer.w2_bias.shape, dtype=torch.bfloat16).to(device),
                                           requires_grad=False)

        return layer, method


class TestMxfp4MoETPU:

    def _run_and_check_forward(self,
                               layer,
                               method,
                               device,
                               hidden_size=128,
                               num_experts=8,
                               num_tokens=16):
        x = torch.randn(num_tokens,
                        hidden_size,
                        dtype=torch.bfloat16,
                        device=device)
        router_logits = torch.randn(num_tokens,
                                    num_experts,
                                    dtype=torch.bfloat16,
                                    device=device)
        out = method.apply_monolithic(layer, x, router_logits)
        assert out.shape == (num_tokens, hidden_size)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any().item(), "Output contains NaNs!"
        assert not torch.isinf(out).any().item(), "Output contains Infs!"

    def test_mxfp4_moe_forward_tpu(self, device):
        """Test uncacheable path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["VLLM_FP4_CACHE_DIR"] = temp_dir
            try:
                layer, method = _build_layer_and_method(device)
                method.process_weights_after_loading(layer)

                assert layer.w13_weight.dtype == torch.uint8
                assert layer.w2_weight.dtype == torch.uint8

                cached_files = os.listdir(temp_dir)
                assert cached_files, (
                    "process_weights_after_loading should have written the "
                    "on-disk FP4 cache.")

                self._run_and_check_forward(layer, method, device)
            finally:
                os.environ.pop("VLLM_FP4_CACHE_DIR", None)

    def test_mxfp4_moe_forward_tpu_cache_hit(self, device):
        """Test cache hit path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["VLLM_FP4_CACHE_DIR"] = temp_dir
            try:
                # First pass populates the cache.
                layer, method = _build_layer_and_method(device)
                method.process_weights_after_loading(layer)

                # Second pass hits the cache.
                layer2, method2 = _build_layer_and_method(device)
                method2.process_weights_after_loading(layer2)

                assert layer2.w13_weight.shape == layer.w13_weight.shape
                assert layer2.w2_weight.shape == layer.w2_weight.shape

                self._run_and_check_forward(layer2, method2, device)
            finally:
                os.environ.pop("VLLM_FP4_CACHE_DIR", None)

    def test_mxfp4_cache_path_ep_ranks(self, device):
        """Verify distinct cache paths are generated for different EP ranks without collision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["VLLM_FP4_CACHE_DIR"] = temp_dir
            try:
                layer, method = _build_layer_and_method(device)
                paths = set()
                for ep_rank in range(8):
                    layer.moe_config.moe_parallel_config.use_ep = True
                    layer.moe_config.moe_parallel_config.ep_rank = ep_rank
                    layer.moe_config.moe_parallel_config.tp_rank = 0

                    _, cache_path = method._get_cache_path(layer)
                    assert f"rank_{ep_rank}_" in os.path.basename(cache_path)
                    paths.add(cache_path)

                assert len(paths) == 8, (
                    "All 8 EP ranks must generate distinct cache file paths.")
            finally:
                os.environ.pop("VLLM_FP4_CACHE_DIR", None)

    def test_mxfp4_cache_path_dist_rank(self, device, monkeypatch):
        """Verify distributed rank takes precedence when torch.distributed is initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["VLLM_FP4_CACHE_DIR"] = temp_dir
            try:
                import torch.distributed as dist
                monkeypatch.setattr(dist, "is_initialized", lambda: True)
                monkeypatch.setattr(dist, "get_rank", lambda: 5)

                layer, method = _build_layer_and_method(device)
                _, cache_path = method._get_cache_path(layer)
                assert "rank_5_" in os.path.basename(cache_path)
            finally:
                os.environ.pop("VLLM_FP4_CACHE_DIR", None)
