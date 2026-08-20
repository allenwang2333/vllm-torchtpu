from vllm.model_executor.models.config import VerifyAndUpdateConfig

_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")
_KIMI_K3_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
                  "KimiK3ForConditionalGeneration")
_DSV4_MODEL = ("vllm_torchtpu.models.vllm.deepseek_v4:"
               "DeepseekV4ForCausalLM")


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration",
                                 _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config

    if hasattr(ModelRegistry, "is_text_generation_model"):
        orig_fn = ModelRegistry.is_text_generation_model
        if not getattr(orig_fn, "_tpu_kimi_k3_patch", False):
            def patched_is_text_gen(model_arch: str, *args, **kwargs) -> bool:
                if model_arch in ("KimiK3ForConditionalGeneration",
                                  "KimiLinearForCausalLM"):
                    return True
                return orig_fn(model_arch, *args, **kwargs)

            patched_is_text_gen._tpu_kimi_k3_patch = True
            ModelRegistry.is_text_generation_model = patched_is_text_gen

