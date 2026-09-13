# ruff: noqa
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

_tpu_info_logged = False


def _log_tpu_info() -> None:
    global _tpu_info_logged
    if _tpu_info_logged:
        return
    _tpu_info_logged = True

    from vllm_torchtpu import tpu_info as ti
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
    try:
        logger.info(
            "TPU info: node_name=%s | tpu_type=%s | worker_id=%s | "
            "num_chips=%s | num_cores_per_chip=%s", ti.get_node_name(),
            ti.get_tpu_type(), ti.get_node_worker_id(), ti.get_num_chips(),
            ti.get_num_cores_per_chip())
    except Exception as error:
        logger.error(
            "Error occurred while logging TPU info: %s. "
            "Are you running on CPU?", error)


def _register_qwen4_exp_hf_config() -> None:
    import importlib.util
    import os
    import sys
    import types
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3_next.configuration_qwen3_next import (
        Qwen3NextConfig,
    )

    if "vllm.transformers_utils.configs.qwen3_next" not in sys.modules:
        q3n_mod = types.ModuleType("vllm.transformers_utils.configs.qwen3_next")
        q3n_mod.Qwen3NextConfig = Qwen3NextConfig
        sys.modules["vllm.transformers_utils.configs.qwen3_next"] = q3n_mod

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec and vllm_spec.origin:
        vllm_dir = os.path.dirname(vllm_spec.origin)
        models_dir = os.path.join(vllm_dir, "models")
        qwen4_dir = os.path.join(models_dir, "qwen4_exp")
        cfg_path = os.path.join(qwen4_dir, "config.py")
        if os.path.exists(cfg_path):
            if "vllm.models" not in sys.modules:
                models_mod = types.ModuleType("vllm.models")
                models_mod.__path__ = [models_dir]
                sys.modules["vllm.models"] = models_mod
            if "vllm.models.qwen4_exp" not in sys.modules:
                qwen4_mod = types.ModuleType("vllm.models.qwen4_exp")
                qwen4_mod.__path__ = [qwen4_dir]
                sys.modules["vllm.models.qwen4_exp"] = qwen4_mod
            if "vllm.models.qwen4_exp.config" in sys.modules:
                mod = sys.modules["vllm.models.qwen4_exp.config"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "vllm.models.qwen4_exp.config", cfg_path
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["vllm.models.qwen4_exp.config"] = mod
                spec.loader.exec_module(mod)
            CONFIG_MAPPING.register("qwen4_exp", mod.Qwen4ExpConfig, exist_ok=True)
            CONFIG_MAPPING.register(
                "qwen4_exp_text", mod.Qwen4ExpTextConfig, exist_ok=True
            )

    try:
        import vllm.v1.attention.backends.utils as _attn_utils

        if not hasattr(_attn_utils, "get_kv_cache_layout"):
            def get_kv_cache_layout() -> str:
                return os.environ.get("VLLM_KV_CACHE_LAYOUT", "NHD")

            get_kv_cache_layout.cache_clear = lambda: None
            _attn_utils.get_kv_cache_layout = get_kv_cache_layout
    except Exception:
        pass


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    _log_tpu_info()
    _register_qwen4_exp_hf_config()
    return f"{TpuPlatform.__module__}.{TpuPlatform.__qualname__}"


