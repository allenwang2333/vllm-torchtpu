# SPDX-License-Identifier: Apache-2.0
"""Model-specific patches and TPU compilation optimizations for Gemma-4 multimodal/audio."""

import functools
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_AUDIO_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_AUDIO_FRAME_BUCKETS = (800, 1600, 3000)


def _get_padded_bucket(val: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if val <= b:
            return b
    return ((val + 63) // 64) * 64


def _is_gemma4_model(model_config: Optional["ModelConfig"]) -> bool:
    if model_config is None:
        return True
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        if "gemma4" in model_type:
            return True
        archs = getattr(hf_config, "architectures", []) or []
        if any("gemma4" in str(a).lower() for a in archs):
            return True
    model_name = str(getattr(model_config, "model", "")).lower()
    return "gemma-4" in model_name or "gemma4" in model_name


def maybe_patch_gemma4(
        model_config: Optional["ModelConfig"] = None) -> bool | None:
    if not _is_gemma4_model(model_config):
        return False

    try:
        from vllm.model_executor.models import gemma4_mm
    except ImportError:
        return False

    model_cls = getattr(gemma4_mm, "Gemma4ForConditionalGeneration", None)
    if model_cls is None:
        return False

    if getattr(model_cls, "_tpu_gemma4_patched", False):
        return None

    # 1. Canonical multimodal placeholder tokens (<|image|>, <|audio|>, <|video|>)
    @classmethod
    def _patched_get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|image|>"
        if modality == "audio":
            return "<|audio|>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")

    model_cls.get_placeholder_str = _patched_get_placeholder_str

    # 2. Static-bucketed compiled audio_tower + embed_audio and zero-sync prefix slicing
    orig_process_audio_input = model_cls._process_audio_input
    batch_audio_features_fn = gemma4_mm.batch_audio_features

    @functools.wraps(orig_process_audio_input)
    def _patched_process_audio_input(
        self,
        audio_input: "gemma4_mm.Gemma4AudioInputs",
    ) -> list[torch.Tensor]:
        assert self.audio_tower is not None
        assert self.embed_audio is not None

        raw_features = audio_input["input_features_padded"]
        raw_masks = audio_input["input_features_mask"]

        # Compute valid output soft-token counts on CPU before moving to TPU.
        # Gemma4AudioSubSampleConvProjection applies mask[:, ::2] twice (layer0, layer1),
        # so the output mask is deterministically mask[:, ::4].
        valid_lens: list[int] | None = None
        if isinstance(raw_masks, list):
            flat_valid_lens: list[int] = []
            for m in raw_masks:
                m_cpu = m.detach().cpu() if m.device.type != "cpu" else m
                if m_cpu.ndim == 1:
                    flat_valid_lens.append(int(m_cpu[::4].sum().item()))
                else:
                    for row in m_cpu.reshape(-1, m_cpu.shape[-1]):
                        flat_valid_lens.append(int(row[::4].sum().item()))
            valid_lens = flat_valid_lens
        elif isinstance(raw_masks, torch.Tensor):
            m_cpu = raw_masks.detach().cpu() if raw_masks.device.type != "cpu" else raw_masks
            if m_cpu.ndim == 1:
                valid_lens = [int(m_cpu[::4].sum().item())]
            else:
                valid_lens = [
                    int(row[::4].sum().item())
                    for row in m_cpu.reshape(-1, m_cpu.shape[-1])
                ]

        input_features, input_features_mask = batch_audio_features_fn(
            raw_features,
            raw_masks,
        )

        batch_size, seq_frames, _ = input_features.shape
        if valid_lens is None or len(valid_lens) != batch_size:
            masks_cpu = input_features_mask.detach().cpu()
            valid_lens = [int(row[::4].sum().item()) for row in masks_cpu]

        target_b = _get_padded_bucket(batch_size, _AUDIO_BATCH_BUCKETS)
        target_t = _get_padded_bucket(seq_frames, _AUDIO_FRAME_BUCKETS)

        pad_b = target_b - batch_size
        pad_t = target_t - seq_frames
        if pad_b > 0 or pad_t > 0:
            input_features = F.pad(input_features, (0, 0, 0, pad_t, 0, pad_b),
                                   value=0.0)
            input_features_mask = F.pad(input_features_mask,
                                        (0, pad_t, 0, pad_b),
                                        value=False)

        compiled_fn = getattr(self, "_tpu_compiled_audio_fn", None)
        if compiled_fn is None:
            audio_tower = self.audio_tower
            embed_audio = self.embed_audio

            def _audio_tower_and_embed(
                feats: torch.Tensor,
                mask: torch.Tensor,
            ) -> torch.Tensor:
                out = audio_tower(feats, mask)
                enc = out[0] if isinstance(out, tuple) else out.last_hidden_state
                return embed_audio(inputs_embeds=enc)

            compiled_fn = torch.compile(
                _audio_tower_and_embed,
                backend="tpu",
                dynamic=False,
            )
            self._tpu_compiled_audio_fn = compiled_fn

        audio_features = compiled_fn(input_features, input_features_mask)

        # Slice each audio item's valid prefix without boolean mask indexing or D2H sync
        return [
            audio_features[i, :valid_lens[i]]
            for i in range(batch_size)
        ]

    model_cls._process_audio_input = _patched_process_audio_input

    # 3. Eliminate 27MB in-place dynamic-update-slice .copy_() into self.per_layer_embeddings
    #    and compile PLE + token embedding lookup per static token bucket.
    orig_embed_input_ids = model_cls.embed_input_ids
    orig_forward = model_cls.forward

    @functools.wraps(orig_embed_input_ids)
    def _patched_embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.per_layer_embeddings is not None:
            if is_multimodal is not None and multimodal_embeddings:
                ple_input_ids = torch.where(
                    is_multimodal.to(input_ids.device, non_blocking=True),
                    torch.zeros_like(input_ids),
                    input_ids,
                )
            else:
                ple_input_ids = input_ids

            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                ple_input_ids
            )
            if per_layer_inputs is not None:
                self._latest_per_layer_inputs = per_layer_inputs.reshape(
                    -1,
                    self.config.text_config.num_hidden_layers,
                    self.config.text_config.hidden_size_per_layer_input,
                )
            else:
                self._latest_per_layer_inputs = None

        if not multimodal_embeddings or is_multimodal is None:
            return super(model_cls, self).embed_input_ids(input_ids)

        return super(model_cls, self).embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    @functools.wraps(orig_forward)
    def _patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        latest_ple = getattr(self, "_latest_per_layer_inputs", None)
        if (latest_ple is not None and inputs_embeds is not None
                and latest_ple.shape[0] == inputs_embeds.shape[0]):
            per_layer_inputs = latest_ple
            self._latest_per_layer_inputs = None
        else:
            per_layer_inputs = (
                self.per_layer_embeddings[:inputs_embeds.shape[0]]
                if self.per_layer_embeddings is not None
                and inputs_embeds is not None else None)

        self._clear_mm_prefix_for_full_attn_layers()

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            per_layer_inputs=per_layer_inputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    model_cls.embed_input_ids = _patched_embed_input_ids
    model_cls.forward = _patched_forward
    model_cls._tpu_gemma4_patched = True
    logger.info(
        "Applied TPU patch: Gemma4 static-bucketed compiled audio tower, zero-sync slicing, and direct PLE forward."
    )
    return None
