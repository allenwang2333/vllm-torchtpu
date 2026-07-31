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
            torch.library.impl(op_name, "default", lambda *a, **kw: None)
        except Exception:
            pass
        try:
            torch.library.define(
                op_name + ".out",
                "(Tensor! input, *, Tensor(a!) out) -> Tensor(a!)",
            )
            torch.library.impl(op_name, "out", lambda *a, **kw: None)
        except Exception:
            pass


def patch_kimi_k3() -> None:
    patch_dummy_ops()

    try:
        import vllm.v1.attention.backends.fa_utils as fa_utils

        if not hasattr(fa_utils, "flash_attn_varlen_func"):
            fa_utils.flash_attn_varlen_func = lambda *a, **kw: None
            fa_utils.compile_flash_attn_varlen_func_from_specs = None
            fa_utils.get_scheduler_metadata = lambda *a, **kw: None
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM
        from vllm.model_executor.models.utils import (
            AutoWeightsLoader,
            WeightsMapper,
        )

        def patched_clm_load_weights(self, weights):
            print("RUNNING PATCHED CLM LOAD_WEIGHTS!")
            skip = ["lm_head."] if self.config.tie_word_embeddings else None
            loader = AutoWeightsLoader(self, skip_prefixes=skip)
            mapper = WeightsMapper(
                orig_to_new_prefix={"language_model.": ""}
            )
            valid_keys = set(dict(self.named_parameters()).keys())

            def safe_weights():
                for name, tensor in weights:
                    mapped_name_info = mapper._map_name(name)
                    if mapped_name_info is not None:
                        mapped_name = mapped_name_info
                        if mapped_name not in valid_keys:
                            is_routed = "experts.routed_experts" in mapped_name
                            is_shared = "experts.shared_experts" in mapped_name
                            if not (is_routed or is_shared):
                                print("Skipping dummy checkpoint parameter "
                                      f"not in model: {mapped_name}")
                                continue

                    yield (name, tensor)

            loaded = loader.load_weights(safe_weights(), mapper=mapper)
            self.model.finalize_mega_moe_weights()
            return loaded

        KimiLinearForCausalLM.load_weights = patched_clm_load_weights
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        if not torch.cuda.is_available():
            torch.cuda.Event = lambda *a, **kw: None
            torch.cuda.Stream = lambda *a, **kw: None
    except Exception:
        pass

    try:
        from vllm.models.kimi_k3.nvidia import model

        # Elementwise residual addition for bring-up
        model.attn_res = lambda prefix, res=None, *a, **kw: (
            prefix + res if res is not None else prefix
        )
    except Exception as e:
        print("PATCH FAILED WITH:", e)

    try:
        from vllm.models.kimi_k3.nvidia import mla

        class MockPrefillBackend:

            def __init__(self, *a, **kw):
                pass

            def forward(self, *a, **kw):
                pass

        class MockDecodeBackend:

            def __init__(self, *a, **kw):
                pass

            def forward(self, *a, **kw):
                pass

        mla.get_mla_prefill_backend = lambda *a, **kw: MockPrefillBackend
        mla.get_mla_decode_backend = lambda *a, **kw: MockDecodeBackend

        def mock_mla_forward(self, positions, hidden_states, **kwargs):
            return torch.zeros_like(hidden_states)

        mla.MultiHeadLatentAttention.forward = mock_mla_forward
    except Exception as e:
        print("PATCH FAILED WITH:", e)


    try:
        import vllm_torchtpu.layers.vllm.moe_routing as moe_routing

        def mock_select_experts(hidden_states, router_logits, topk, *a, **kw):
            weight = torch.ones(
                (hidden_states.shape[0], topk),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            indices = torch.zeros(
                (hidden_states.shape[0], topk),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            return weight, indices

        moe_routing.select_experts = mock_select_experts
    except Exception as e:
        print("PATCH FAILED WITH:", e)
