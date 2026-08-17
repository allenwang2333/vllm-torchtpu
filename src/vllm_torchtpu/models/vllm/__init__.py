from vllm.model_executor.models.config import VerifyAndUpdateConfig

# Architecture -> "module:ClassName" for the TPU implementations that replace
# vLLM's built-in ones. The lazy string form keeps the model module (and the
# device-specific shims it installs on import) out of the process unless that
# architecture is actually instantiated.
_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")
_KIMI_K3_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
                  "KimiK3ForConditionalGeneration")
_DEEPSEEK_V4_MODEL = ("vllm_torchtpu.models.vllm.deepseek_v4:"
                      "DeepseekV4ForCausalLM")


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration",
                                 _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DEEPSEEK_V4_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
