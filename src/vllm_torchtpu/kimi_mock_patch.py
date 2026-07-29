import torch

def patch_dummy_ops() -> None:
    print("[kimi_mock_patch] Registering dummy C++ ops in PyTorch...")
    dummy_ops = {
        "_C::rotary_embedding": "(Tensor! positions) -> ()",
        "_C::static_scaled_fp8_quant": "(Tensor! input) -> ()",
        "_C::dynamic_scaled_fp8_quant": "(Tensor! input) -> ()",
        "_C::dynamic_per_token_scaled_fp8_quant": "(Tensor! input) -> ()",
        "_C::silu_and_mul": "(Tensor! input) -> ()",
        "_C::silu_and_mul_quant": "(Tensor! input) -> ()",
        "_C::silu_and_mul_nvfp4_quant": "(Tensor! input) -> ()",
        "_C::rms_norm": "(Tensor! input) -> ()",
        "_C::fused_add_rms_norm": "(Tensor! input) -> ()",
        "_C::per_token_group_fp8_quant": "(Tensor! input) -> ()",
        "_C::scaled_fp4_quant": "(Tensor! input) -> ()",
        "_C::rms_norm_static_fp8_quant": "(Tensor! input) -> ()",
        "_C::fused_add_rms_norm_static_fp8_quant": "(Tensor! input) -> ()",
        "_C::rms_norm_dynamic_per_token_quant": "(Tensor! input) -> ()",
        "_C::rms_norm_per_block_quant": "(Tensor! input) -> ()",
        "_C::fused_qk_norm_rope": "(Tensor! input) -> ()",
        "_C::silu_and_mul_per_block_quant": "(Tensor! input) -> ()",
        "_C::cutlass_scaled_mm": "(Tensor! input) -> ()",
    }
    for op_name, sig in dummy_ops.items():
        try:
            torch.library.define(op_name, sig)
            torch.library.impl(op_name, "default", lambda *args, **kwargs: None)
        except Exception as e:
            pass
        try:
            torch.library.define(op_name + ".out", "(Tensor! input, *, Tensor(a!) out) -> Tensor(a!)")
            torch.library.impl(op_name, "out", lambda *args, **kwargs: None)
        except Exception as e:
            pass


def patch_kimi_k3() -> None:
    patch_dummy_ops()

    try:
        import vllm.v1.attention.backends.fa_utils as fa_utils
        if not hasattr(fa_utils, "flash_attn_varlen_func"):
            fa_utils.flash_attn_varlen_func = lambda *args, **kwargs: None
            fa_utils.compile_flash_attn_varlen_func_from_specs = None
            fa_utils.get_scheduler_metadata = lambda *args, **kwargs: None
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM, KimiLinearModel
        from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

        
        orig_load_weights = KimiLinearForCausalLM.load_weights
        def patched_clm_load_weights(self, weights):
            print("RUNNING PATCHED CLM LOAD_WEIGHTS!")
            loader = AutoWeightsLoader(
                self,
                skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
            )
            mapper = WeightsMapper(orig_to_new_prefix={"language_model.": ""})
            valid_keys = set(dict(self.named_parameters()).keys())
            
            def safe_weights():
                for name, tensor in weights:
                    mapped_name_info = mapper._map_name(name)
                    
                    if mapped_name_info is not None:
                        mapped_name = mapped_name_info
                        if mapped_name not in valid_keys:
                            # MoE routed_experts dynamically map names, so keep anything with experts.
                            # Also keep lm_head since it might bypass valid_keys if tied? No lm_head is in valid_keys anyway.
                            if "experts.routed_experts" not in mapped_name and "experts.shared_experts" not in mapped_name:
                                print(f"Skipping dummy checkpoint parameter not in model: {mapped_name}")
                                continue

                    yield (name, tensor)

            loaded = loader.load_weights(safe_weights(), mapper=mapper)
            self.model.finalize_mega_moe_weights()
            return loaded
        KimiLinearForCausalLM.load_weights = patched_clm_load_weights
    except Exception as e:
        print("PATCH FAILED WITH:", e)
        
    try:
        # Overwrite all occurrences of torch.cuda.Event and Stream with mocks so the arch doesn't crash on TPU compilation.
        if not torch.cuda.is_available():
            torch.cuda.Event = lambda *args, **kwargs: None
            torch.cuda.Stream = lambda *args, **kwargs: None
    except Exception:
        pass

        
    try:
        from vllm.models.kimi_k3.nvidia import kda
        
        kda.causal_conv1d_fn = lambda x, weight: torch.zeros_like(x.transpose(0, 1))

        orig_kda_forward = kda.KimiK3DeltaAttention.forward
        def mock_kda_forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, **kwargs):
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            attn_metadata_raw = forward_context.attn_metadata
            
            # We need batch size from metadata
            if attn_metadata_raw is not None:
                from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
                # It might be a dict, or it might be raw metadata directly.
                if isinstance(attn_metadata_raw, dict):
                    attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
                    if attn_metadata_narrowed is not None:
                        batch_size = getattr(attn_metadata_narrowed, "num_prefills", 1)
                    else:
                        batch_size = 1
                else:
                    batch_size = getattr(attn_metadata_raw, "num_prefills", 1)
            else:
                batch_size = 1

            if not hasattr(self, 'kv_cache') or self.kv_cache is None or self.kv_cache[0] is None:
                try:
                    width_size = self.conv1d.weight.shape[-1] - 1
                except Exception:
                    width_size = 3
                    
                dim_size = self.local_num_heads * self.head_dim
                conv_shape = (batch_size, dim_size, width_size)  # conv1d uses (B, D, W)
                
                conv_state = torch.zeros(conv_shape, dtype=hidden_states.dtype, device=hidden_states.device)
                recurrent_state = torch.zeros((batch_size, self.local_num_heads, self.head_dim, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
                
                self.kv_cache = (conv_state, recurrent_state)
            # Ensure we return a zero output and skip the actual missing kernel calls!
            return torch.zeros_like(hidden_states)

        kda.KimiK3DeltaAttention.forward = mock_kda_forward
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        from vllm.models.kimi_k3.nvidia import model
        model.attn_res = lambda prefix, *args, **kwargs: torch.zeros_like(prefix)
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        from vllm.models.kimi_k3.nvidia import mla
        
        class MockPrefillBackend:
            def __init__(self, *args, **kwargs): pass
            def forward(self, *args, **kwargs): pass
            
        class MockDecodeBackend:
            def __init__(self, *args, **kwargs): pass
            def forward(self, *args, **kwargs): pass

        mla.get_mla_prefill_backend = lambda *args, **kwargs: MockPrefillBackend
        mla.get_mla_decode_backend = lambda *args, **kwargs: MockDecodeBackend
        
        def mock_mla_forward(self, positions, hidden_states, **kwargs):
            return torch.zeros_like(hidden_states)

        mla.MultiHeadLatentAttention.forward = mock_mla_forward
    except Exception as e:
        print("PATCH FAILED WITH:", e)


    try:
        import vllm_torchtpu.layers.vllm.moe_routing as moe_routing
        def mock_select_experts(hidden_states, router_logits, topk, *args, **kwargs):
            return torch.ones((hidden_states.shape[0], topk), dtype=hidden_states.dtype, device=hidden_states.device), \
                   torch.zeros((hidden_states.shape[0], topk), dtype=torch.int32, device=hidden_states.device)

        moe_routing.select_experts = mock_select_experts
    except Exception as e:
        print("PATCH FAILED WITH:", e)


