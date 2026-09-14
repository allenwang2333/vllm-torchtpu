# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for vLLM-Omni on Cloud TPU.

Enables vLLM-Omni models (like Qwen3-Omni) to run natively on Google Cloud TPU
without modifying upstream vLLM-Omni source code.
"""

from typing import Any
import sys
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OMNI_PATCHES_APPLIED = False


def patch_omni_platform_interface() -> None:
    """Equip OmniPlatform with is_tpu check."""
    try:
        import vllm_omni.platforms.interface as pi
        if not hasattr(pi.OmniPlatform, "is_tpu"):
            pi.OmniPlatform.is_tpu = (
                lambda self: getattr(self, "device_name", "") == "tpu"
                or getattr(self, "device_type", "") == "tpu"
                or getattr(self, "is_tpu_platform", False)
            )
            logger.info("[TPU Omni Patch] Patched OmniPlatform.is_tpu")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_omni_platform_interface skipped: %s", e)


def patch_orchestrator_timeouts() -> None:
    """Increase default timeouts to 1800s to accommodate TPU JIT compilation."""
    try:
        import vllm_omni.config.omni_config as oc
        if hasattr(oc, "VllmOmniOrchestratorConfig"):
            for fld in ("stage_init_timeout", "init_timeout"):
                if fld in getattr(oc.VllmOmniOrchestratorConfig, "model_fields", {}):
                    oc.VllmOmniOrchestratorConfig.model_fields[fld].default = 1800

        import vllm_omni.engine.arg_utils as au
        if hasattr(au, "OrchestratorArgs"):
            au.OrchestratorArgs.stage_init_timeout = 1800
            au.OrchestratorArgs.init_timeout = 1800

        import vllm_omni.entrypoints.omni_base as ob
        if not getattr(ob.OmniBase, "_tpu_timeout_patched", False):
            orig_omni_base_init = ob.OmniBase.__init__

            def patched_omni_base_init(self, *args, **kwargs):
                kwargs.setdefault("init_timeout", 1800)
                kwargs.setdefault("stage_init_timeout", 1800)
                return orig_omni_base_init(self, *args, **kwargs)

            ob.OmniBase.__init__ = patched_omni_base_init
            ob.OmniBase._tpu_timeout_patched = True
        logger.info("[TPU Omni Patch] Patched orchestrator timeouts to 1800s")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_orchestrator_timeouts skipped: %s", e)


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
    import torch

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

    # 2. Qwen3OmniMoeForConditionalGeneration methods: generate_audio, talker_preprocess, compute_logits, make_omni_output
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
            Qwen3OmniMoeForConditionalGeneration,
        )

        if not getattr(Qwen3OmniMoeForConditionalGeneration, "_tpu_patched", False):
            # generate_audio
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

            # talker_preprocess
            orig_talker_preprocess = Qwen3OmniMoeForConditionalGeneration.talker_preprocess

            def patched_talker_preprocess(self, input_ids, input_embeds, payload):
                if input_ids is None and input_embeds is not None:
                    span_len = input_embeds.shape[0]
                    input_ids = torch.zeros((span_len,), dtype=torch.long, device=input_embeds.device)
                ids, embeds, update_dict = orig_talker_preprocess(self, input_ids, input_embeds, payload)
                # Ensure code_predictor_codes is on CPU to avoid allocating large unpadded tensors on TPU HBM
                if isinstance(update_dict, dict) and "codes" in update_dict:
                    if isinstance(update_dict["codes"], dict) and "audio" in update_dict["codes"]:
                        if isinstance(update_dict["codes"]["audio"], torch.Tensor):
                            update_dict["codes"]["audio"] = update_dict["codes"]["audio"].cpu()
                if isinstance(update_dict, dict) and "meta" in update_dict:
                    if update_dict["meta"].get("num_processed_tokens") is None:
                        update_dict["meta"]["num_processed_tokens"] = 1
                return ids, embeds, update_dict

            Qwen3OmniMoeForConditionalGeneration.talker_preprocess = patched_talker_preprocess

            # compute_logits
            orig_compute_logits = Qwen3OmniMoeForConditionalGeneration.compute_logits

            def patched_compute_logits(self, hidden_states, sampling_metadata=None):
                if getattr(self, "model_stage", None) == "code2wav":
                    if hasattr(hidden_states, "text_hidden_states"):
                        return hidden_states.text_hidden_states
                    return hidden_states
                try:
                    return orig_compute_logits(self, hidden_states, sampling_metadata)
                except AttributeError as e:
                    if "compute_logits" in str(e):
                        if hasattr(hidden_states, "text_hidden_states"):
                            hidden_states = hidden_states.text_hidden_states
                        return hidden_states
                    raise

            Qwen3OmniMoeForConditionalGeneration.compute_logits = patched_compute_logits

            # make_omni_output
            orig_make_omni_output = Qwen3OmniMoeForConditionalGeneration.make_omni_output

            def patched_make_omni_output(self, *args, **kwargs):
                output = orig_make_omni_output(self, *args, **kwargs)
                if hasattr(output, "multimodal_outputs") and isinstance(output.multimodal_outputs, dict):
                    mm = output.multimodal_outputs
                    if "audio" not in mm:
                        for k in ("model_outputs", "waveform", "audio_waveform"):
                            if k in mm and isinstance(mm[k], (list, tuple)) and len(mm[k]) > 0:
                                mm["audio"] = [t.flatten() if isinstance(t, torch.Tensor) else t for t in mm[k]]
                                break
                    if "waveform" not in mm and "audio" in mm:
                        mm["waveform"] = mm["audio"]
                return output

            Qwen3OmniMoeForConditionalGeneration.make_omni_output = patched_make_omni_output
            Qwen3OmniMoeForConditionalGeneration._tpu_patched = True
            logger.info("[TPU Omni Patch] Patched Qwen3OmniMoeForConditionalGeneration methods")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch Qwen3OmniMoeForConditionalGeneration skipped: %s", e)

    # 3. Qwen3Thinker embed_input_ids and AudioEncoder
    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeAudioEncoder,
        )
        import torch.nn.functional as F

        if not getattr(Qwen3OmniMoeThinkerForConditionalGeneration, "_tpu_patched", False):
            orig_embed = Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids

            def patched_embed_input_ids(self, input_ids, multimodal_embeddings=None, is_multimodal=None):
                if multimodal_embeddings:
                    text_cfg = getattr(getattr(self, "config", None), "text_config", None)
                    hidden_size = getattr(text_cfg, "hidden_size", None)
                    if hidden_size is not None:
                        multimodal_embeddings = [
                            e for e in multimodal_embeddings
                            if getattr(e, "shape", None) and e.shape[-1] == hidden_size
                        ]
                return orig_embed(self, input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal)

            Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids = patched_embed_input_ids
            Qwen3OmniMoeThinkerForConditionalGeneration._tpu_patched = True
            logger.info("[TPU Omni Patch] Patched Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids")

        if not getattr(Qwen3OmniMoeAudioEncoder, "_tpu_patched", False):
            orig_forward_encoder = Qwen3OmniMoeAudioEncoder._forward_encoder

            def patched_forward_encoder(self, input_features, chunk_lengths, *args, **kwargs):
                if hasattr(chunk_lengths, "sum") and hasattr(input_features, "shape"):
                    total_needed = int(chunk_lengths.sum().item())
                    if input_features.shape[1] < total_needed:
                        input_features = F.pad(input_features, (0, total_needed - input_features.shape[1]), value=0)
                    elif input_features.shape[1] > total_needed:
                        input_features = input_features[:, :total_needed]
                return orig_forward_encoder(self, input_features, chunk_lengths, *args, **kwargs)

            Qwen3OmniMoeAudioEncoder._forward_encoder = patched_forward_encoder
            Qwen3OmniMoeAudioEncoder._tpu_patched = True
            logger.info("[TPU Omni Patch] Patched Qwen3OmniMoeAudioEncoder._forward_encoder")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch Qwen3Thinker skipped: %s", e)


def patch_stage_input_processors() -> None:
    """Patch stage input processors for TPU tensors and async handoffs."""
    try:
        import vllm_omni.model_executor.stage_input_processors.qwen3_omni as sip
        import torch

        if getattr(sip, "_tpu_patched", False):
            return

        # thinker2talker_async_chunk
        orig_t2t = getattr(sip, "thinker2talker_async_chunk", None)
        if orig_t2t:
            def patched_t2t(request, transfer_manager, chunk_id=0, *args, **kwargs):
                return orig_t2t(request, transfer_manager, chunk_id=chunk_id, *args, **kwargs)
            sip.thinker2talker_async_chunk = patched_t2t

        # talker2code2wav_async_chunk
        orig_t2c_async = getattr(sip, "talker2code2wav_async_chunk", None)
        if orig_t2c_async:
            def patched_t2c_async(request, transfer_manager, chunk_id=0, *args, **kwargs):
                payload = orig_t2c_async(request, transfer_manager, chunk_id=chunk_id, *args, **kwargs)
                # Ensure codes tensor is 2D, not 1D flattened
                if payload and hasattr(payload, "codes") and isinstance(payload.codes, dict):
                    for k, v in payload.codes.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 1:
                            num_q = 16
                            if v.shape[0] % num_q == 0:
                                payload.codes[k] = v.view(-1, num_q).transpose(0, 1)
                return payload
            sip.talker2code2wav_async_chunk = patched_t2c_async

        # talker2code2wav_full_payload
        orig_t2c_full = getattr(sip, "talker2code2wav_full_payload", None)
        if orig_t2c_full:
            def patched_t2c_full(request, transfer_manager, *args, **kwargs):
                payload = orig_t2c_full(request, transfer_manager, *args, **kwargs)
                if payload and hasattr(payload, "codes") and isinstance(payload.codes, dict):
                    for k, v in payload.codes.items():
                        if isinstance(v, list):
                            payload.codes[k] = torch.tensor(v, dtype=torch.long)
                        elif isinstance(v, torch.Tensor) and v.ndim == 1:
                            num_q = 16
                            if v.shape[0] % num_q == 0:
                                payload.codes[k] = v.view(-1, num_q).transpose(0, 1)
                return payload
            sip.talker2code2wav_full_payload = patched_t2c_full

        sip._tpu_patched = True
        logger.info("[TPU Omni Patch] Patched stage_input_processors.qwen3_omni")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_stage_input_processors skipped: %s", e)


def patch_qwen3_code_predictor() -> None:
    """Patch Qwen3 code predictor layers for TPU and CPU delegation."""
    try:
        import vllm_omni.model_executor.models.common.qwen3_code_predictor as cp

        if getattr(cp, "_tpu_patched", False):
            return

        import torch
        from contextlib import nullcontext

        # Patch _RMSNorm with forward_tpu
        if hasattr(cp, "_RMSNorm"):
            cp._RMSNorm.forward_tpu = cp._RMSNorm.forward_native

        # Patch _RotaryEmbedding with forward_tpu
        if hasattr(cp, "_RotaryEmbedding"):
            cp._RotaryEmbedding.forward_tpu = getattr(
                cp._RotaryEmbedding, "_lookup", cp._RotaryEmbedding.forward_native
            )

        # Patch CodePredictorBaseModel.forward to avoid unsupported torch.amp.autocast on TPU
        if hasattr(cp, "CodePredictorBaseModel"):
            orig_base_forward = cp.CodePredictorBaseModel.forward

            def patched_base_forward(self, inputs_embeds, position_ids):
                input_dtype = inputs_embeds.dtype
                use_fp32 = input_dtype == torch.float16 and inputs_embeds.device.type not in ("cpu", "tpu")
                if use_fp32:
                    inputs_embeds = inputs_embeds.float()
                    ctx = torch.amp.autocast(inputs_embeds.device.type, dtype=torch.float32)
                else:
                    ctx = nullcontext()
                with ctx:
                    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
                    hidden_states = inputs_embeds
                    for layer in self.layers:
                        hidden_states = layer(hidden_states, position_embeddings)
                    return hidden_states

            cp.CodePredictorBaseModel.forward = patched_base_forward

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


def patch_compat_fallbacks() -> None:
    """Inject compatibility fallbacks for missing vLLM v1 symbols."""
    try:
        import vllm.v1.outputs as v1_out
        if not hasattr(v1_out, "RoutedExpertsLists"):
            from typing import Any
            v1_out.RoutedExpertsLists = Any
    except Exception:
        pass

    try:
        import vllm.v1.metrics.stats as v1_stats
        if not hasattr(v1_stats, "PrefillStats"):
            from dataclasses import dataclass
            @dataclass
            class PrefillStats:
                pass
            v1_stats.PrefillStats = PrefillStats
    except Exception:
        pass


def apply_omni_patches() -> None:
    """Apply all vLLM-Omni TPU runtime patches."""
    global _OMNI_PATCHES_APPLIED
    if _OMNI_PATCHES_APPLIED:
        return
    _OMNI_PATCHES_APPLIED = True
    logger.info("[TPU Omni Patch] Applying comprehensive vLLM-Omni TPU runtime patches...")
    patch_compat_fallbacks()
    patch_omni_platform_interface()
    patch_orchestrator_timeouts()
    patch_qwen3_omni_pipeline()
    patch_deploy_config_loader()
    patch_qwen3_omni_models()
    patch_stage_input_processors()
    patch_qwen3_code_predictor()
    logger.info("[TPU Omni Patch] All vLLM-Omni TPU runtime patches applied successfully.")
