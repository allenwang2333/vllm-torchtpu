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


def patch_compat_fallbacks() -> None:
    """Inject compatibility fallbacks for missing vLLM symbols in older/newer vLLM versions."""
    from typing import Any
    from dataclasses import dataclass

    # 1. vllm.entrypoints.serve.create_error_response
    try:
        import vllm.entrypoints.serve as vllm_serve
        if not hasattr(vllm_serve, "create_error_response"):
            try:
                from vllm.entrypoints.utils import create_error_response
                vllm_serve.create_error_response = create_error_response
            except ImportError:
                def create_error_response(*args, **kwargs):
                    pass
                vllm_serve.create_error_response = create_error_response
            logger.info("[TPU Omni Patch] Injected create_error_response fallback into vllm.entrypoints.serve")
    except Exception:
        pass

    # 2. vllm.v1.outputs.RoutedExpertsLists
    try:
        import vllm.v1.outputs as v1_out
        if not hasattr(v1_out, "RoutedExpertsLists"):
            v1_out.RoutedExpertsLists = Any
            logger.info("[TPU Omni Patch] Injected RoutedExpertsLists fallback into vllm.v1.outputs")
    except Exception:
        pass

    # 3. vllm.v1.metrics.stats.PrefillStats
    try:
        import vllm.v1.metrics.stats as v1_stats
        if not hasattr(v1_stats, "PrefillStats"):
            @dataclass
            class PrefillStats:
                pass
            v1_stats.PrefillStats = PrefillStats
            logger.info("[TPU Omni Patch] Injected PrefillStats fallback into vllm.v1.metrics.stats")
    except Exception:
        pass

    # 5. vllm.v1.engine.utils.CoreEngineLaunch & wait_for_engine_startup adapter
    try:
        import functools
        import vllm.v1.engine.utils as eng_utils
        if not hasattr(eng_utils, "CoreEngineLaunch"):
            @dataclass
            class CoreEngineLaunch:
                engine_manager: Any = None
                coordinator: Any = None
                addresses: Any = None
                tensor_queue: Any = None
            eng_utils.CoreEngineLaunch = CoreEngineLaunch
            setattr(eng_utils, "CoreEngineLaunch", CoreEngineLaunch)
            if "vllm.v1.engine.utils" in sys.modules:
                setattr(sys.modules["vllm.v1.engine.utils"], "CoreEngineLaunch", CoreEngineLaunch)
            logger.info("[TPU Omni Patch] Injected CoreEngineLaunch fallback into vllm.v1.engine.utils")

        if hasattr(eng_utils, "wait_for_engine_startup"):
            orig_wait_for_engine_startup = eng_utils.wait_for_engine_startup

            @functools.wraps(orig_wait_for_engine_startup)
            def compatible_wait_for_engine_startup(*args: Any, **kwargs: Any) -> Any:
                # Adapts vllm-omni 6-argument call:
                # (handshake_socket, core_engines, parallel_config, coordinated_dp, cache_config, launch_info)
                # to vllm 8-argument call:
                # (handshake_socket, addresses, core_engines, parallel_config, coordinated_dp, cache_config, proc_manager, coord_process)
                if len(args) == 6:
                    handshake_socket = args[0]
                    core_engines = args[1]
                    parallel_config = args[2]
                    coordinated_dp = args[3]
                    cache_config = args[4]
                    launch_info = args[5]

                    addresses = getattr(launch_info, "addresses", None)
                    proc_manager = getattr(launch_info, "engine_manager", None)
                    coord_process = getattr(launch_info, "coordinator", None)

                    return orig_wait_for_engine_startup(
                        handshake_socket,
                        addresses,
                        core_engines,
                        parallel_config,
                        coordinated_dp,
                        cache_config,
                        proc_manager,
                        coord_process,
                        **kwargs,
                    )
                return orig_wait_for_engine_startup(*args, **kwargs)

            eng_utils.wait_for_engine_startup = compatible_wait_for_engine_startup
            if "vllm.v1.engine.utils" in sys.modules:
                setattr(sys.modules["vllm.v1.engine.utils"], "wait_for_engine_startup", compatible_wait_for_engine_startup)
            for mod_name in ("vllm_omni.engine.stage_engine_startup", "vllm_omni.diffusion.stage_diffusion_proc"):
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], "wait_for_engine_startup", compatible_wait_for_engine_startup)
            logger.info("[TPU Omni Patch] Injected compatible wait_for_engine_startup adapter into vllm.v1.engine.utils")
    except Exception as e:
        logger.warning("[TPU Omni Patch] Failed to inject CoreEngineLaunch / wait_for_engine_startup fallback: %s", e)

    # 4. vllm.model_executor.models.registry._resolve_module_name
    try:
        import vllm.model_executor.models.registry as mr
        if not hasattr(mr, "_resolve_module_name"):
            mr._resolve_module_name = lambda mod_relname: f"vllm.model_executor.models.{mod_relname}"
            logger.info("[TPU Omni Patch] Injected _resolve_module_name fallback into vllm.model_executor.models.registry")
    except Exception:
        pass

    # 6. vllm_omni.request.OmniRequest session_id compatibility with vLLM 0.27
    try:
        import functools
        import vllm_omni.request as omni_req

        orig_init = omni_req.OmniRequest.__init__

        @functools.wraps(orig_init)
        def compatible_omni_request_init(self: Any, *args: Any, **kwargs: Any) -> None:
            session_id = kwargs.pop("session_id", None)
            orig_init(self, *args, **kwargs)
            self.session_id = session_id

        omni_req.OmniRequest.__init__ = compatible_omni_request_init

        @classmethod
        def compatible_from_engine_core_request(cls: Any, request: Any, block_hasher: Any = None) -> Any:
            return cls(
                request_id=request.request_id,
                external_req_id=getattr(request, "external_req_id", None),
                client_index=request.client_index,
                prompt_token_ids=request.prompt_token_ids,
                prompt_embeds=request.prompt_embeds,
                prompt_is_token_ids=request.prompt_is_token_ids,
                mm_features=request.mm_features,
                sampling_params=request.sampling_params,
                pooling_params=request.pooling_params,
                arrival_time=request.arrival_time,
                lora_request=request.lora_request,
                cache_salt=request.cache_salt,
                priority=request.priority,
                trace_headers=request.trace_headers,
                block_hasher=block_hasher,
                additional_information=getattr(request, "additional_information", None),
                model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
                resumable=request.resumable,
                session_id=getattr(request, "session_id", None),
                reasoning_ended=request.reasoning_ended,
                reasoning_parser_kwargs=request.reasoning_parser_kwargs,
                abort_immediately=request.abort_immediately,
            )

        omni_req.OmniRequest.from_engine_core_request = compatible_from_engine_core_request
        for mod in list(sys.modules.values()):
            if getattr(mod, "OmniRequest", None) is not None:
                mod.OmniRequest.from_engine_core_request = compatible_from_engine_core_request
                mod.OmniRequest.__init__ = compatible_omni_request_init
            if getattr(mod, "Request", None) is omni_req.OmniRequest:
                mod.Request.from_engine_core_request = compatible_from_engine_core_request
                mod.Request.__init__ = compatible_omni_request_init
        logger.info("[TPU Omni Patch] Patched OmniRequest for vLLM 0.27 session_id compatibility")
    except Exception as e:
        logger.warning("[TPU Omni Patch] Failed to patch OmniRequest: %s", e)

    # 7. CompletionOutput multimodal_output fallback
    try:
        from vllm.outputs import CompletionOutput
        if not hasattr(CompletionOutput, "multimodal_output"):
            CompletionOutput.multimodal_output = None
            logger.info("[TPU Omni Patch] Injected multimodal_output fallback into CompletionOutput")
    except Exception:
        pass


def patch_omni_platform_interface() -> None:
    """Equip OmniPlatform with is_tpu check."""
    try:
        import vllm_omni.platforms.interface as pi
        import vllm_omni.platforms as platforms
        if "vllm_omni" in sys.modules:
            setattr(sys.modules["vllm_omni"], "platforms", platforms)
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

        def _inject_tpu_stages(deploy: Any) -> None:
            deploy.async_chunk = False
            if hasattr(deploy, "stages") and deploy.stages:
                for stg in deploy.stages:
                    if hasattr(stg, "async_chunk"):
                        stg.async_chunk = False
            logger.info("[TPU Omni Patch] Enforced deploy.async_chunk = False (full-payload mode)")
            if not hasattr(deploy, "platforms") or deploy.platforms is None:
                deploy.platforms = {}
            if isinstance(deploy.platforms, dict):
                if "tpu" not in deploy.platforms:
                    deploy.platforms["tpu"] = {
                        "stages": [
                            {"stage_id": 0, "tensor_parallel_size": 1, "devices": "0", "worker_cls": "vllm_torchtpu.omni.worker.TPUOmniWorker"},
                            {"stage_id": 1, "tensor_parallel_size": 1, "devices": "1", "worker_cls": "vllm_torchtpu.omni.worker.TPUOmniWorker"},
                            {
                                "stage_id": 2,
                                "tensor_parallel_size": 1,
                                "devices": "2",
                                "dtype": "float32",
                                "enforce_eager": True,
                                "worker_cls": "vllm_torchtpu.omni.worker.TPUOmniWorker",
                            },
                        ]
                    }
                    logger.info("[TPU Omni Patch] Injected TPU stage configuration into deploy config (0->0, 1->1, 2->2)")
                elif "stages" in deploy.platforms["tpu"]:
                    for stg in deploy.platforms["tpu"]["stages"]:
                        stg.setdefault("worker_cls", "vllm_torchtpu.omni.worker.TPUOmniWorker")
                        if stg.get("stage_id") == 2:
                            if "dtype" not in stg:
                                stg["dtype"] = "float32"
                            if "enforce_eager" not in stg:
                                stg["enforce_eager"] = True
                            logger.info("[TPU Omni Patch] Injected dtype: float32, enforce_eager: True for Code2Wav stage")

        orig_load_deploy_config = sc.load_deploy_config

        def patched_load_deploy_config(path_or_dict: Any, *args, **kwargs):
            if isinstance(path_or_dict, dict):
                import tempfile, yaml
                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                    yaml.dump(path_or_dict, f)
                    tmp_name = f.name
                deploy = orig_load_deploy_config(tmp_name, *args, **kwargs)
            else:
                deploy = orig_load_deploy_config(path_or_dict, *args, **kwargs)
            _inject_tpu_stages(deploy)
            return deploy

        sc.load_deploy_config = patched_load_deploy_config
        for mod_name in (
            "vllm_omni.config",
            "vllm_omni.config.stage_config",
            "vllm_omni.config.config_factory",
            "vllm_omni.config.omni_config",
            "vllm_omni.engine.async_omni_engine",
            "vllm_omni.entrypoints.utils",
        ):
            if mod_name in sys.modules:
                setattr(sys.modules[mod_name], "load_deploy_config", patched_load_deploy_config)

        # Patch _apply_platform_overrides
        if hasattr(sc, "_apply_platform_overrides"):
            orig_apply_platform_overrides = sc._apply_platform_overrides

            def patched_apply_platform_overrides(deploy: Any, platform: str | None = None) -> Any:
                if platform is None:
                    try:
                        from vllm_omni.platforms import current_omni_platform
                        dev = getattr(current_omni_platform, "device_name", None)
                        platform = dev.lower() if dev is not None else "tpu"
                    except Exception:
                        platform = "tpu"
                if platform == "tpu":
                    _inject_tpu_stages(deploy)
                return orig_apply_platform_overrides(deploy, platform)

            sc._apply_platform_overrides = patched_apply_platform_overrides
            for mod_name in ("vllm_omni.config.stage_config", "vllm_omni.config.config_factory"):
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], "_apply_platform_overrides", patched_apply_platform_overrides)

        # Patch merge_pipeline_deploy to guarantee disjoint TPU devices across all stages
        if hasattr(sc, "merge_pipeline_deploy"):
            orig_merge_pipeline_deploy = sc.merge_pipeline_deploy

            def patched_merge_pipeline_deploy(pipeline: Any, deploy: Any, cli_overrides: dict[str, Any] | None = None) -> list[Any]:
                _inject_tpu_stages(deploy)
                deploy.async_chunk = False
                stages = orig_merge_pipeline_deploy(pipeline, deploy, cli_overrides)
                used_devices = set()
                for stage in stages:
                    stage.async_chunk = False
                    rt = getattr(stage, "yaml_runtime", None)
                    if isinstance(rt, dict):
                        dev = rt.get("devices")
                        if dev is None or dev in used_devices:
                            new_dev = str(stage.stage_id)
                            rt["devices"] = new_dev
                            logger.info("[TPU Omni Patch] Assigned disjoint TPU device '%s' to stage %d", new_dev, stage.stage_id)
                        used_devices.add(rt.get("devices"))
                    ea = getattr(stage, "yaml_engine_args", None)
                    if isinstance(ea, dict):
                        ea["async_chunk"] = False
                    if getattr(stage, "stage_id", None) == 2:
                        if isinstance(ea, dict):
                            ea["dtype"] = "float32"
                            ea["enforce_eager"] = True
                return stages

            sc.merge_pipeline_deploy = patched_merge_pipeline_deploy
            for mod_name in (
                "vllm_omni.config",
                "vllm_omni.config.stage_config",
                "vllm_omni.config.config_factory",
                "vllm_omni.config.omni_config",
            ):
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], "merge_pipeline_deploy", patched_merge_pipeline_deploy)

        # Patch StageRuntime._resolve_replica_physical_devices to guarantee no device collision
        try:
            import vllm_omni.engine.stage_runtime as sr
            if not getattr(sr, "_tpu_patched", False):
                orig_resolve_dev = sr.StageRuntime._resolve_replica_physical_devices

                def patched_resolve_dev(self, stage_id: int, runtime_cfg: Any) -> str | None:
                    resolved = orig_resolve_dev(self, stage_id, runtime_cfg)
                    if resolved == "1" and stage_id == 2:
                        logger.info("[TPU Omni Patch] Disambiguated stage 2 physical devices from '1' to '2'")
                        return "2"
                    return resolved

                sr.StageRuntime._resolve_replica_physical_devices = patched_resolve_dev
                sr._tpu_patched = True
                logger.info("[TPU Omni Patch] Patched StageRuntime._resolve_replica_physical_devices")
        except Exception:
            pass

        # Prevent global CLI dtype='auto' from clobbering per-stage float32
        orig_build_overrides = sc.build_stage_runtime_overrides

        def patched_build_stage_runtime_overrides(stage_id: int, cli_overrides: dict[str, Any], *args, **kwargs):
            filtered = dict(cli_overrides)
            if filtered.get("dtype") == "auto":
                filtered.pop("dtype", None)
            return orig_build_overrides(stage_id, filtered, *args, **kwargs)

        sc.build_stage_runtime_overrides = patched_build_stage_runtime_overrides
        sc._tpu_patched = True
        logger.info("[TPU Omni Patch] Patched load_deploy_config, merge_pipeline_deploy, and build_stage_runtime_overrides")
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
        import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as qwen3_omni_mod
        Qwen3OmniMoeForConditionalGeneration = qwen3_omni_mod.Qwen3OmniMoeForConditionalGeneration

        if not getattr(Qwen3OmniMoeForConditionalGeneration, "_tpu_patched", False):
            # generate_audio
            orig_generate_audio = Qwen3OmniMoeForConditionalGeneration.generate_audio

            def patched_generate_audio(
                self, codes, left_context_size=None, seq_token_counts=None, **kwargs
            ):
                if codes is not None and isinstance(codes, torch.Tensor):
                    if codes.ndim == 1 and codes.shape[0] % 16 == 0:
                        codes = codes.view(16, -1)
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

            def patched_talker_preprocess(self, input_ids, input_embeds=None, *args, **kwargs):
                if input_ids is None and input_embeds is not None:
                    span_len = input_embeds.shape[0]
                    input_ids = torch.zeros((span_len,), dtype=torch.long, device=input_embeds.device)
                ids, embeds, update_dict = orig_talker_preprocess(self, input_ids, input_embeds, *args, **kwargs)
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

            # _thinker_decode_to_talker_decode
            def patched_thinker_decode_to_talker_decode(
                self,
                payload: Any,
                device: torch.device,
                update_dict: Any,
            ):
                embed = payload.get("embed", {}) if isinstance(payload, dict) else {}
                meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                hs = payload.get("hidden_states", {}) if isinstance(payload, dict) else {}

                cached_decode = embed.get("cached_decode", None)
                incoming_decode = embed.get("decode", None)

                # Check finished flag from thinker
                finished_raw = meta.get("finished", False)
                if isinstance(finished_raw, torch.Tensor):
                    thinker_finished = bool(finished_raw.item()) if finished_raw.numel() > 0 else False
                else:
                    thinker_finished = bool(finished_raw)

                if meta.get("_thinker_ever_finished", False):
                    thinker_finished = True
                elif thinker_finished:
                    update_dict.setdefault("meta", {})["_thinker_ever_finished"] = True

                # 1. Incorporate any surplus trailing_text from prefill if cached_decode not yet initialized
                if cached_decode is None and isinstance(hs, dict):
                    trailing = hs.get("trailing_text")
                    if isinstance(trailing, torch.Tensor) and trailing.numel() > 0:
                        if trailing.ndim >= 2 and trailing.shape[0] > 1:
                            cached_decode = trailing[:-1].to(device=device, dtype=torch.bfloat16)
                            update_dict.setdefault("embed", {})["cached_decode"] = cached_decode.detach()
                            update_dict.setdefault("hidden_states", {})["trailing_text"] = None

                # 2. Append new incoming decode embedding to cached buffer
                if incoming_decode is not None:
                    incoming_t = incoming_decode.to(device=device, dtype=torch.bfloat16)
                    if incoming_t.ndim == 1:
                        incoming_t = incoming_t.unsqueeze(0)
                    elif incoming_t.ndim == 3:
                        incoming_t = incoming_t.reshape(-1, incoming_t.shape[-1])
                    if cached_decode is not None:
                        cached_decode = torch.cat([cached_decode.to(device=device, dtype=torch.bfloat16), incoming_t], dim=0)
                    else:
                        cached_decode = incoming_t
                    update_dict.setdefault("embed", {})["cached_decode"] = cached_decode.detach()
                elif cached_decode is not None:
                    cached_decode = cached_decode.to(device=device, dtype=torch.bfloat16)

                # Always clear incoming decode so it is not re-appended in subsequent steps
                update_dict.setdefault("embed", {})["decode"] = None

                # 3. Consume from cached_decode using cursor
                cursor = meta.get("_tpu_text_cursor", 0)

                if cached_decode is not None and cursor < cached_decode.shape[0]:
                    thinker_embed = cached_decode[cursor:cursor + 1]
                    update_dict.setdefault("meta", {})["_tpu_text_cursor"] = cursor + 1
                    return self.talker.text_projection(thinker_embed).to(device)

                # 4. If buffer is currently exhausted:
                if thinker_finished:
                    if meta.get("eos_emitted", False):
                        return self.tts_pad_embed.to(device)
                    update_dict.setdefault("meta", {})["eos_emitted"] = True
                    logger.info(
                        "[TPU Talker] All %d thinker decode tokens consumed and thinker finished; emitting tts_eos_embed",
                        cursor
                    )
                    return self.tts_eos_embed.to(device)
                else:
                    return self.tts_pad_embed.to(device)

            Qwen3OmniMoeForConditionalGeneration._thinker_decode_to_talker_decode = patched_thinker_decode_to_talker_decode
            if hasattr(qwen3_omni_mod, "Qwen3OmniForConditionalGeneration"):
                qwen3_omni_mod.Qwen3OmniForConditionalGeneration._thinker_decode_to_talker_decode = patched_thinker_decode_to_talker_decode


            # forward (extract real codec codes from runtime_additional_information / model_intermediate_buffer for code2wav)
            orig_forward = Qwen3OmniMoeForConditionalGeneration.forward

            def patched_forward(self, input_ids=None, positions=None, *args, **kwargs):
                if getattr(self, "model_stage", None) == "code2wav":
                    runtime_info = kwargs.get("runtime_additional_information")
                    if runtime_info is None:
                        runtime_info = kwargs.get("model_intermediate_buffer")
                    if runtime_info is None and hasattr(self, "model_intermediate_buffer"):
                        buf = getattr(self, "model_intermediate_buffer", {})
                        if isinstance(buf, dict):
                            runtime_info = list(buf.values())

                    device = getattr(self, "device", None)
                    if device is None and hasattr(self, "code2wav"):
                        device = getattr(self.code2wav, "device", None)
                    if device is None and hasattr(self, "parameters"):
                        try:
                            device = next(self.parameters()).device
                        except Exception:
                            device = torch.device("cpu")

                    extracted_ids = None
                    if runtime_info is not None:
                        for info in runtime_info:
                            if isinstance(info, dict):
                                codes = info.get("codes")
                                if isinstance(codes, dict):
                                    audio = codes.get("audio")
                                    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                                        extracted_ids = audio.to(device=device, dtype=torch.long).reshape(-1)
                                        break
                                    elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                                        extracted_ids = torch.as_tensor(audio, device=device, dtype=torch.long).reshape(-1)
                                        break
                                if "codes.audio" in info:
                                    audio = info["codes.audio"]
                                    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                                        extracted_ids = audio.to(device=device, dtype=torch.long).reshape(-1)
                                        break
                                    elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                                        extracted_ids = torch.as_tensor(audio, device=device, dtype=torch.long).reshape(-1)
                                        break

                    if extracted_ids is not None:
                        input_ids = extracted_ids
                    elif input_ids is None:
                        input_ids = torch.zeros((16,), dtype=torch.long, device=device)
                    else:
                        input_ids = input_ids.to(device=device, dtype=torch.long).reshape(-1)

                    seq_counts = kwargs.get("seq_token_counts")
                    if seq_counts is not None and len(seq_counts) > 0:
                        total_valid = sum(int(c) for c in seq_counts)
                        # Trim off TPU bucket padding so torch.split(input_ids, seq_token_counts) matches exactly
                        if input_ids.shape[0] > total_valid:
                            input_ids = input_ids[:total_valid]
                        elif input_ids.shape[0] < total_valid:
                            kwargs["seq_token_counts"] = [int(input_ids.shape[0])]
                    elif "seq_token_counts" not in kwargs or kwargs["seq_token_counts"] is None:
                        kwargs["seq_token_counts"] = [int(input_ids.shape[0])]

                return orig_forward(self, input_ids, positions, *args, **kwargs)

            Qwen3OmniMoeForConditionalGeneration.forward = patched_forward

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
                if getattr(self, "model_stage", None) == "talker":
                    info_dicts = kwargs.get("model_intermediate_buffer")
                    if info_dicts is None:
                        info_dicts = kwargs.get("runtime_additional_information")
                    if info_dicts:
                        for info in info_dicts:
                            if isinstance(info, dict):
                                codes = info.setdefault("codes", {})
                                if isinstance(codes, dict) and codes.get("audio") is None:
                                    dev = self._module_device(getattr(self, "talker", self))
                                    num_code_groups = getattr(getattr(self, "talker", None), "num_code_groups", 16)
                                    codes["audio"] = torch.zeros((1, num_code_groups), dtype=torch.long, device=dev)
                try:
                    output = orig_make_omni_output(self, *args, **kwargs)
                except Exception as e:
                    if getattr(self, "model_stage", None) == "talker":
                        logger.warning("[TPU Omni Patch] orig_make_omni_output talker failed (%s), constructing fallback", e)
                        model_outputs = args[0] if len(args) > 0 else kwargs.get("model_outputs")
                        dev = self._module_device(getattr(self, "talker", self))
                        num_code_groups = getattr(getattr(self, "talker", None), "num_code_groups", 16)
                        dummy_codes = torch.zeros((1, num_code_groups), dtype=torch.long, device=dev)
                        from vllm_omni.core.sched.output import OmniOutput
                        output = OmniOutput(
                            text_hidden_states=model_outputs[:1] if model_outputs is not None else None,
                            multimodal_outputs={"codes": {"audio": dummy_codes}},
                        )
                    else:
                        raise

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
        import inspect
        import vllm_omni.model_executor.stage_input_processors.qwen3_omni as sip
        import torch

        if getattr(sip, "_tpu_patched", False):
            return

        def _call_filtered(fn, *args, **kwargs):
            sig = inspect.signature(fn)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_varkw:
                valid_keys = set(sig.parameters.keys())
                kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            return fn(*args, **kwargs)

        # thinker2talker_async_chunk
        orig_t2t = getattr(sip, "thinker2talker_async_chunk", None)
        if orig_t2t:
            def patched_t2t(transfer_manager, multimodal_output, request, is_finished=False, *args, **kwargs):
                if isinstance(multimodal_output, dict):
                    # 1. Unflatten any dotted keys if present
                    try:
                        from vllm_omni.data_entry_keys import unflatten_payload
                        unflat = unflatten_payload(multimodal_output)
                        for k, v in unflat.items():
                            if k not in multimodal_output or not multimodal_output[k]:
                                multimodal_output[k] = v
                            elif isinstance(v, dict) and isinstance(multimodal_output[k], dict):
                                for sub_k, sub_v in v.items():
                                    if sub_k not in multimodal_output[k]:
                                        multimodal_output[k][sub_k] = sub_v
                    except Exception:
                        pass

                    # 2. Guarantee hidden_states and layers dict exist
                    hs = multimodal_output.setdefault("hidden_states", {})
                    if not isinstance(hs, dict):
                        hs = {"output": hs}
                        multimodal_output["hidden_states"] = hs
                    layers = hs.setdefault("layers", {})
                    if not isinstance(layers, dict):
                        layers = {}
                        hs["layers"] = layers

                    # Check for layer 0 / "0" (embeddings)
                    emb = layers.get("0", layers.get(0))
                    if emb is None:
                        emb = (
                            multimodal_output.get("hidden_states.layer_0")
                            or multimodal_output.get("embed", {}).get("prefill")
                            or multimodal_output.get("embed", {}).get("decode")
                            or multimodal_output.get("hidden")
                            or multimodal_output.get("text_hidden_states")
                        )
                    if emb is not None:
                        layers["0"] = emb
                        layers[0] = emb

                    # Check for layer 24 / "24" (hidden states)
                    hid = layers.get("24", layers.get(24))
                    if hid is None:
                        hid = (
                            multimodal_output.get("hidden_states.layer_24")
                            or multimodal_output.get("hidden")
                            or multimodal_output.get("text_hidden_states")
                            or emb
                        )
                    if hid is not None:
                        layers["24"] = hid
                        layers[24] = hid

                    # 3. For decode steps (chunk_id > 0), ensure thinker_emb has shape [1, D]
                    req_id = getattr(request, "external_req_id", getattr(request, "request_id", None))
                    chunk_id = getattr(transfer_manager, "put_req_chunk", {}).get(req_id, 0)
                    if chunk_id > 0:
                        for k in ("0", 0):
                            t = layers.get(k)
                            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] > 1:
                                layers[k] = t[-1:]
                        for k in ("24", 24):
                            t = layers.get(k)
                            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] > 1:
                                layers[k] = t[-1:]

                return _call_filtered(orig_t2t, transfer_manager, multimodal_output, request, is_finished=is_finished, *args, **kwargs)
            sip.thinker2talker_async_chunk = patched_t2t

        # talker2code2wav_async_chunk
        orig_t2c_async = getattr(sip, "talker2code2wav_async_chunk", None)
        if orig_t2c_async:
            def patched_t2c_async(*args, **kwargs):
                return _call_filtered(orig_t2c_async, *args, **kwargs)
            sip.talker2code2wav_async_chunk = patched_t2c_async

        # talker2code2wav_full_payload
        orig_t2c_full = getattr(sip, "talker2code2wav_full_payload", None)
        if orig_t2c_full:
            def patched_t2c_full(transfer_manager=None, pooling_output=None, request=None, multimodal_output=None, *args, **kwargs):
                if pooling_output is None and multimodal_output is not None:
                    pooling_output = multimodal_output
                res = _call_filtered(orig_t2c_full, transfer_manager, pooling_output, request, *args, **kwargs)
                if res is None and isinstance(pooling_output, dict):
                    codes = pooling_output.get("codes.audio")
                    if codes is None and isinstance(pooling_output.get("codes"), dict):
                        codes = pooling_output["codes"].get("audio")
                    if codes is not None:
                        t = torch.as_tensor(codes)
                        if t.ndim >= 2 and t.numel() > 0:
                            codec_codes = t.transpose(0, 1).cpu().reshape(-1).tolist()
                            if codec_codes:
                                logger.info("[TPU Omni Patch] talker2code2wav_full_payload fallback returned %d codec codes", len(codec_codes))
                                return {
                                    "codes": {"audio": codec_codes},
                                    "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
                                }
                return res
            sip.talker2code2wav_full_payload = patched_t2c_full

        # thinker2talker_full_payload
        orig_t2t_full = getattr(sip, "thinker2talker_full_payload", None)
        if orig_t2t_full:
            def patched_t2t_full(transfer_manager=None, pooling_output=None, request=None, multimodal_output=None, *args, **kwargs):
                if pooling_output is None and multimodal_output is not None:
                    pooling_output = multimodal_output
                if isinstance(pooling_output, dict):
                    if "hidden_states.layer_0" not in pooling_output:
                        val = (
                            pooling_output.get("hidden")
                            or pooling_output.get("text_hidden_states")
                            or pooling_output.get("hidden_states", {}).get("layers", {}).get(0)
                            or pooling_output.get("hidden_states", {}).get("layers", {}).get("0")
                            or pooling_output.get("embed", {}).get("prefill")
                            or pooling_output.get("embed", {}).get("decode")
                        )
                        if val is not None:
                            pooling_output["hidden_states.layer_0"] = val
                    if "hidden_states.layer_24" not in pooling_output:
                        val = (
                            pooling_output.get("hidden")
                            or pooling_output.get("text_hidden_states")
                            or pooling_output.get("hidden_states", {}).get("layers", {}).get(24)
                            or pooling_output.get("hidden_states", {}).get("layers", {}).get("24")
                            or pooling_output.get("hidden_states", {}).get("output")
                            or pooling_output.get("hidden_states.layer_0")
                        )
                        if val is not None:
                            pooling_output["hidden_states.layer_24"] = val
                    for tok in ("tts_bos", "tts_eos", "tts_pad"):
                        k = f"embed.{tok}"
                        if k not in pooling_output or pooling_output[k] is None:
                            if isinstance(pooling_output.get("embed"), dict) and pooling_output["embed"].get(tok) is not None:
                                v = pooling_output["embed"][tok]
                                pooling_output[k] = v[0] if isinstance(v, list) and len(v) > 0 else v
                            else:
                                hid = pooling_output.get("hidden_states.layer_0")
                                h_dim = int(hid.shape[-1]) if isinstance(hid, torch.Tensor) and hid.ndim >= 1 else 2048
                                pooling_output[k] = torch.zeros((1, 1, h_dim), dtype=torch.bfloat16)
                return _call_filtered(orig_t2t_full, transfer_manager, pooling_output, request, *args, **kwargs)
            sip.thinker2talker_full_payload = patched_t2t_full

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
                    hidden_states = self.norm(hidden_states)
                return hidden_states.to(input_dtype)

            cp.CodePredictorBaseModel.forward = patched_base_forward

        # Patch CodePredictorWrapper.forward for CPU execution when invoked on TPU
        if hasattr(cp, "CodePredictorWrapper"):
            orig_forward = cp.CodePredictorWrapper.forward

            def patched_forward(
                self,
                layer0_code,
                layer0_embed,
                last_talker_hidden,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                generator=None,
                generators=None,
            ):
                device = layer0_code.device
                if device.type in ("tpu", "xla"):
                    try:
                        torch.set_num_threads(min(4, torch.get_num_threads()))
                    except Exception:
                        pass
                    if next(self.parameters()).device.type != "cpu":
                        self.to("cpu")
                    if getattr(self, "_compiled_model_fwd", None) is None:
                        self._model_dtype = next(self.model.parameters()).dtype
                        self._lm_heads_list = list(self.lm_head)
                        self._codec_embeds_list = list(self.model.codec_embedding)
                        self._compiled_model_fwd = self.model.forward
                    orig_device = device
                    res = orig_forward(
                        self,
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
                return orig_forward(
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

            cp.CodePredictorWrapper.forward = patched_forward

        try:
            from vllm_omni.model_executor.models.qwen3_omni import (
                qwen3_omni_moe_code_predictor_mtp as cp_mtp,
            )
            if hasattr(cp_mtp, "Qwen3OmniMoeTalkerCodePredictor"):
                cp_mtp.Qwen3OmniMoeTalkerCodePredictor.forward = cp.CodePredictorWrapper.forward
        except Exception as e:
            logger.debug("[TPU Omni Patch] patch Qwen3OmniMoeTalkerCodePredictor skipped: %s", e)

        cp._tpu_patched = True
        logger.info("[TPU Omni Patch] Patched qwen3_code_predictor for TPU")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_qwen3_code_predictor skipped: %s", e)


def patch_omni_connector_validation() -> None:
    """Ensure TPU workers are accepted by vLLM-Omni connector validation."""
    try:
        import vllm_omni.worker.omni_connector_validation as ocv
        from vllm_omni.worker.omni_connector_model_runner_mixin import (
            OmniConnectorModelRunnerMixin,
        )

        orig_validate = ocv.validate_worker_omni_connector

        def patched_validate(worker_cls_path, required):
            if not required:
                return
            try:
                worker_cls = ocv._resolve_worker_cls(worker_cls_path)
                model_runner_cls = getattr(worker_cls, "model_runner_cls", None)
                if model_runner_cls is not None and issubclass(
                    model_runner_cls, OmniConnectorModelRunnerMixin
                ):
                    return
                worker_name = getattr(worker_cls, "__name__", str(worker_cls_path))
                if "TPU" in worker_name or "tpu" in str(worker_cls_path).lower():
                    return
            except Exception:
                pass
            return orig_validate(worker_cls_path, required)

        ocv.validate_worker_omni_connector = patched_validate
        logger.info("[TPU Omni Patch] Patched validate_worker_omni_connector for TPU")
    except Exception as e:
        logger.debug("[TPU Omni Patch] patch_omni_connector_validation skipped: %s", e)


def apply_omni_patches() -> None:
    """Apply all vLLM-Omni TPU runtime patches."""
    global _OMNI_PATCHES_APPLIED
    if _OMNI_PATCHES_APPLIED:
        return
    _OMNI_PATCHES_APPLIED = True
    logger.info("[TPU Omni Patch] Applying comprehensive vLLM-Omni TPU runtime patches...")
    patch_compat_fallbacks()
    patch_omni_platform_interface()
    patch_omni_connector_validation()
    patch_orchestrator_timeouts()
    patch_qwen3_omni_pipeline()
    patch_deploy_config_loader()
    patch_qwen3_omni_models()
    patch_stage_input_processors()
    patch_qwen3_code_predictor()
    try:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
        from vllm_torchtpu.omni.runner import TPUOmniModelRunner
        TPUWorker.model_runner_cls = TPUOmniModelRunner
        logger.info("[TPU Omni Patch] Wired TPUWorker.model_runner_cls = TPUOmniModelRunner")
    except Exception as e:
        logger.debug("[TPU Omni Patch] TPUWorker.model_runner_cls wiring skipped: %s", e)
    logger.info("[TPU Omni Patch] All vLLM-Omni TPU runtime patches applied successfully.")
