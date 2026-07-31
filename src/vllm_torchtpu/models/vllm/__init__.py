_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
