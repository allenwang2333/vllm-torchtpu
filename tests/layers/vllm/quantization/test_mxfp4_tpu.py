import torch
import pytest
from types import SimpleNamespace
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod

class TestMxfp4MoETPU:
    def test_mxfp4_moe_forward_tpu(self, device):
        # 1. Config
        num_experts = 8
        top_k = 2
        hidden_size = 128
        intermediate_size = 256
        num_tokens = 16
        
        # 2. Instantiate FusedMoE Parallel Config
        from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
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
        
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        moe_config = FusedMoEConfig(
            num_experts=num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            num_local_experts=num_experts,
            num_logical_experts=num_experts,
            activation=MoEActivation.SILU,
            device=device,
            routing_method=RoutingMethodType.DeepseekV4,
            moe_parallel_config=moe_parallel_config,
            in_dtype=torch.bfloat16,
        )
        
        # Define a minimal offline mock for VllmConfig
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
                custom_ops=[],
            ),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
            ),
        )
        
        # Instantiate inside the thread-local config context to satisfy CustomOp/layer registration checks
        from vllm.config.vllm import set_current_vllm_config
        with set_current_vllm_config(dummy_vllm_config):
            layer = FusedMoE(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                params_dtype=torch.bfloat16,
                renormalize=True,
                tp_size=1,
                ep_size=1,
                dp_size=1,
                pcp_size=1,
                activation="silu",
            )
            
        layer.layer_idx = 3
        layer._tpu_activation_str = "silu"
        layer.renormalize = True
        
        # Assign mock weights to w13 and w2 (simulating BF16 model weights on TPU)
        layer.w13_weight = torch.nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        )
        
        # 3. Instantiate MXFP4 quant method
        method = VllmMxfp4MoEMethod(moe_config)
        layer.quant_method = method
        
        # 4. Run weight processing (simulating weight loader)
        method.process_weights_after_loading(layer)
        
        # Verify weight attributes are populated as native FP4 packed tensors
        assert hasattr(layer, "w13_packed")
        assert hasattr(layer, "w2_packed")
        assert layer.w13_packed.dtype == torch.uint8
        assert layer.w2_packed.dtype == torch.uint8
        
        # 5. Execute forward pass (apply_monolithic) on TPU
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
        
        # Trigger the monolithic forward pass
        out = method.apply_monolithic(layer, x, router_logits)
        
        # 6. Verify outputs
        assert out.shape == (num_tokens, hidden_size)
        assert out.dtype == torch.bfloat16
        # Assert no NaNs or Infs
        assert not torch.isnan(out).any().item(), "Output contains NaNs!"
        assert not torch.isinf(out).any().item(), "Output contains Infs!"
        
        print("TPU Forward Output Mean:", out.mean().item())
        print("TPU Forward Output Max:", out.max().item())
        print("TPU Forward Output Min:", out.min().item())
