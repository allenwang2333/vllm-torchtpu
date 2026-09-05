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
_GLM5_NEXT_MTP = ("vllm_torchtpu.models.vllm.glm5_next:"
                  "Glm5NextMTP")


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
    # MTP draft head. vLLM's speculative config rewrites the draft's
    # `architectures` before the registry is consulted, so the name it asks
    # for is the one spec_patch writes there (`Glm5NextMTPModel`, following
    # the Glm4Moe convention); the class name is registered too so an
    # explicit --speculative-config also resolves.
    ModelRegistry.register_model("Glm5NextMTPModel", _GLM5_NEXT_MTP)
    ModelRegistry.register_model("Glm5NextMTP", _GLM5_NEXT_MTP)
    # Teach an unpatched vLLM that glm5_next has an MTP draft head. Must run
    # before a SpeculativeConfig is built, which platform init precedes;
    # idempotent, and importing the module applies it again in the spawned
    # EngineCore child.
    #
    # `glm5_next_spec_patch`, NOT `glm5_next.spec_patch`, and the distinction
    # is the point: this line is on every model's boot path, and importing a
    # submodule runs its parent package first. The package spelling would
    # execute eleven GLM modules plus the DeepSeek-V4 kernels in every Qwen
    # and DeepSeek-V4 process, and would make an import error anywhere in
    # GLM's tree their boot failure too. `tpu_runner.py`'s
    # `is_compressed_indexer_cache` documents that hazard and answers it
    # without importing anything; this has to agree with it.
    from vllm_torchtpu.models.vllm import glm5_next_spec_patch
    glm5_next_spec_patch.install()
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
