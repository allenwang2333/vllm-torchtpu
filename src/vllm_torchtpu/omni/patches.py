# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for vLLM-Omni on Cloud TPU.

Enables vLLM-Omni models (like Qwen3-Omni) to run natively on Google Cloud TPU
without modifying upstream vLLM-Omni source code.
"""

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OMNI_PATCHES_APPLIED = False


def patch_qwen3_omni_pipeline() -> None:
    """Patch Qwen3-Omni pipeline topology and sampling constraints."""
    try:
        from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
            QWEN3_OMNI_PIPELINE,
        )

        stages = QWEN3_OMNI_PIPELINE.stages
        # Stage 1: Talker — ensure 2150 (Qwen3-Omni Codec EOS) is in stop_token_ids
        if len(stages) > 1:
            talker_stage = stages[1]
            constraints = getattr(talker_stage, "sampling_constraints", {})
            stop_tokens = list(constraints.get("stop_token_ids", []))
            for tok in (2150, 4198, 151645):
                if tok not in stop_tokens:
                    stop_tokens.append(tok)
            object.__setattr__(
                talker_stage,
                "sampling_constraints",
                {**constraints, "stop_token_ids": stop_tokens, "detokenize": False},
            )
            logger.info("[TPU Omni Patch] Updated Talker stop_token_ids: %s", stop_tokens)

        # Stage 2: Code2Wav — constrain max_tokens to 1 to prevent vocoder looping
        if len(stages) > 2:
            code2wav_stage = stages[2]
            constraints = getattr(code2wav_stage, "sampling_constraints", {})
            object.__setattr__(
                code2wav_stage,
                "sampling_constraints",
                {**constraints, "detokenize": True, "max_tokens": 1},
            )
            logger.info("[TPU Omni Patch] Constrained Code2Wav max_tokens=1")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_qwen3_omni_pipeline skipped: %s", e)


def patch_deploy_config_loader() -> None:
    """Patch deploy config loader to inject TPU platform stages and preserve float32."""
    try:
        import vllm_omni.config.stage_config as sc

        if getattr(sc, "_tpu_patched", False):
            return

        orig_load_deploy_config = sc.load_deploy_config

        def patched_load_deploy_config(path_or_dict: Any, *args, **kwargs):
            deploy = orig_load_deploy_config(path_or_dict, *args, **kwargs)
            if hasattr(deploy, "platforms") and isinstance(deploy.platforms, dict):
                # Inject TPU platform configuration if missing
                if "tpu" not in deploy.platforms:
                    deploy.platforms["tpu"] = {
                        "stages": [
                            {"stage_id": 0, "tensor_parallel_size": 2, "devices": "0,1"},
                            {"stage_id": 1, "tensor_parallel_size": 1, "devices": "2"},
                            {
                                "stage_id": 2,
                                "tensor_parallel_size": 1,
                                "devices": "3",
                                "dtype": "float32",
                            },
                        ]
                    }
                    logger.info("[TPU Omni Patch] Injected TPU stage configuration into deploy config")
                elif "stages" in deploy.platforms["tpu"]:
                    # Ensure stage 2 uses float32 to prevent TPU MXU reduced-precision noise
                    for stg in deploy.platforms["tpu"]["stages"]:
                        if stg.get("stage_id") == 2 and "dtype" not in stg:
                            stg["dtype"] = "float32"
                            logger.info("[TPU Omni Patch] Injected dtype: float32 for Code2Wav stage")
            return deploy

        sc.load_deploy_config = patched_load_deploy_config

        # Prevent global CLI dtype='auto' from clobbering per-stage float32
        orig_build_overrides = sc.build_stage_runtime_overrides

        def patched_build_stage_runtime_overrides(stage_id: int, cli_overrides: dict[str, Any], *args, **kwargs):
            filtered = dict(cli_overrides)
            # If CLI passed dtype='auto', remove it so stage-specific dtype wins
            if filtered.get("dtype") == "auto":
                filtered.pop("dtype", None)
            return orig_build_overrides(stage_id, filtered, *args, **kwargs)

        sc.build_stage_runtime_overrides = patched_build_stage_runtime_overrides
        sc._tpu_patched = True
        logger.info("[TPU Omni Patch] Patched load_deploy_config and build_stage_runtime_overrides")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_deploy_config_loader skipped: %s", e)


def patch_qwen3_omni_models() -> None:
    """Patch Qwen3-Omni model classes for TPU execution."""
    # 1. Code2Wav streaming chunk seam stitching
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
            Qwen3OmniMoeCode2Wav,
        )

        if not getattr(Qwen3OmniMoeCode2Wav, "_tpu_patched", False):
            orig_streaming = Qwen3OmniMoeCode2Wav.chunked_decode_streaming

            def patched_chunked_decode_streaming(
                self, codes, left_context_size=None, seq_token_counts=None
            ):
                if seq_token_counts is None and codes is not None:
                    num_q = getattr(getattr(self, "config", None), "num_quantizers", 16)
                    seq_token_counts = [codes.shape[-1] * num_q] * codes.shape[0]
                if left_context_size is None and codes is not None:
                    left_context_size = [0] * codes.shape[0]
                return orig_streaming(
                    self,
                    codes,
                    left_context_size=left_context_size,
                    seq_token_counts=seq_token_counts,
                )

            Qwen3OmniMoeCode2Wav.chunked_decode_streaming = patched_chunked_decode_streaming
            Qwen3OmniMoeCode2Wav._tpu_patched = True
            logger.info("[TPU Omni Patch] Patched Qwen3OmniMoeCode2Wav.chunked_decode_streaming")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch Qwen3OmniMoeCode2Wav skipped: %s", e)

    # 2. Qwen3OmniMoe generate_audio seq_token_counts
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
            Qwen3OmniMoeForConditionalGeneration,
        )

        if not getattr(Qwen3OmniMoeForConditionalGeneration, "_tpu_patched", False):
            orig_generate_audio = Qwen3OmniMoeForConditionalGeneration.generate_audio

            def patched_generate_audio(
                self, codes, left_context_size=None, seq_token_counts=None, **kwargs
            ):
                if seq_token_counts is None and codes is not None:
                    seq_token_counts = [codes.shape[-1] * 16] * codes.shape[0]
                return orig_generate_audio(
                    self,
                    codes,
                    left_context_size=left_context_size,
                    seq_token_counts=seq_token_counts,
                    **kwargs,
                )

            Qwen3OmniMoeForConditionalGeneration.generate_audio = patched_generate_audio
            Qwen3OmniMoeForConditionalGeneration._tpu_patched = True
            logger.info("[TPU Omni Patch] Patched Qwen3OmniMoeForConditionalGeneration.generate_audio")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch Qwen3OmniMoeForConditionalGeneration skipped: %s", e)


def patch_qwen3_code_predictor() -> None:
    """Patch Qwen3 code predictor layers for TPU and CPU delegation."""
    try:
        import vllm_omni.model_executor.models.common.qwen3_code_predictor as cp

        if getattr(cp, "_tpu_patched", False):
            return

        import torch

        # Patch _RMSNorm with forward_tpu
        if hasattr(cp, "_RMSNorm"):
            cp._RMSNorm.forward_tpu = cp._RMSNorm.forward_native

        # Patch _RotaryEmbedding with forward_tpu
        if hasattr(cp, "_RotaryEmbedding"):
            cp._RotaryEmbedding.forward_tpu = getattr(
                cp._RotaryEmbedding, "_lookup", cp._RotaryEmbedding.forward_native
            )

        # Patch CodePredictorWrapper.predict_codes for CPU execution when invoked on TPU
        if hasattr(cp, "CodePredictorWrapper"):
            orig_predict = cp.CodePredictorWrapper.predict_codes

            def patched_predict_codes(
                self,
                layer0_code,
                layer0_embed,
                last_talker_hidden,
                do_sample=False,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                generator=None,
                generators=None,
            ):
                device = layer0_code.device
                if device.type == "tpu":
                    try:
                        torch.set_num_threads(min(4, torch.get_num_threads()))
                    except Exception:
                        pass
                    if next(self.parameters()).device.type != "cpu":
                        self.to("cpu")
                    orig_device = device
                    res = self.forward(
                        layer0_code=layer0_code.to("cpu"),
                        layer0_embed=layer0_embed.to("cpu"),
                        last_talker_hidden=last_talker_hidden.to("cpu"),
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        generators=generators,
                    )
                    if isinstance(res, tuple):
                        return tuple(
                            r.to(device=orig_device) if isinstance(r, torch.Tensor) else r
                            for r in res
                        )
                    return res.to(device=orig_device)
                return orig_predict(
                    self,
                    layer0_code,
                    layer0_embed,
                    last_talker_hidden,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    generators=generators,
                )

            cp.CodePredictorWrapper.predict_codes = patched_predict_codes

        cp._tpu_patched = True
        logger.info("[TPU Omni Patch] Patched qwen3_code_predictor for TPU")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_qwen3_code_predictor skipped: %s", e)


def apply_omni_patches() -> None:
    """Apply all vLLM-Omni TPU runtime patches."""
    global _OMNI_PATCHES_APPLIED
    if _OMNI_PATCHES_APPLIED:
        return
    _OMNI_PATCHES_APPLIED = True
    logger.info("[TPU Omni Patch] Applying vLLM-Omni TPU runtime patches...")
    patch_qwen3_omni_pipeline()
    patch_deploy_config_loader()
    patch_qwen3_omni_models()
    patch_qwen3_code_predictor()
    logger.info("[TPU Omni Patch] All vLLM-Omni TPU runtime patches applied successfully.")
