from vllm.model_executor.models.config import VerifyAndUpdateConfig

_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")
_KIMI_K3_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
                  "KimiK3ForConditionalGeneration")
_DSV4_MODEL = ("vllm_torchtpu.models.vllm.deepseek_v4:"
               "DeepseekV4ForCausalLM")
_GLM5_NEXT_MODEL = ("vllm_torchtpu.models.vllm.glm5_next:"
                    "Glm5NextForCausalLM")
# Named by submodule path and deliberately NOT re-exported from the glm5_next
# package `__init__`: reaching the multimodal class pulls vLLM's glm4_1v and
# therefore torchvision, which the text-only path must not have to pay for.
_GLM5_NEXT_MM_MODEL = ("vllm_torchtpu.models.vllm.glm5_next.multimodal:"
                       "Glm5NextForConditionalGeneration")


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration",
                                 _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    # GLM-5.3-Flash. The checkpoint's architecture string is the multimodal
    # `...ForConditionalGeneration`, so that name resolves to the multimodal
    # class. `Glm5NextForCausalLM` stays registered under its own name so the
    # text-only stack can still be selected with an
    # `hf_overrides={"architectures": ["Glm5NextForCausalLM"]}` swap -- which
    # is also the only way to load this model on a host without torchvision,
    # since vLLM inspects a registered architecture by importing it in a
    # subprocess.
    ModelRegistry.register_model("Glm5NextForCausalLM", _GLM5_NEXT_MODEL)
    ModelRegistry.register_model("Glm5NextForConditionalGeneration",
                                 _GLM5_NEXT_MM_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
