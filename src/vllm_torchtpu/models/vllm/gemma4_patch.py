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
_AUDIO_FRAME_BUCKETS = (544, 808, 1600, 3000)


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


def _patch_gemma4_audio_attention_bf16() -> None:
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    attn_cls = getattr(modeling_gemma4, "Gemma4AudioAttention", None)
    if attn_cls is None or getattr(attn_cls, "_tpu_bf16_attn_patched", False):
        return

    def _bf16_audio_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        compute_dtype = hidden_states.dtype

        q_scale = (self.q_scale * F.softplus(self.per_dim_scale)).to(
            dtype=compute_dtype)
        k_scale = float(self.k_scale)

        query_states = self.q_proj(hidden_states).view(hidden_shape) * q_scale
        key_states = self.k_proj(hidden_states).view(hidden_shape) * k_scale
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(
            -1, self.num_heads, self.head_dim).to(dtype=compute_dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1,
                                       self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks,
                                      self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac.float() + matrix_bd.float()
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(),
                self.config.attention_invalid_logits_value,
            )

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(
            attn_output.to(dtype=self.post.linear.weight.dtype))

        return attn_output, attn_weights

    attn_cls.forward = _bf16_audio_attn_forward
    attn_cls._tpu_bf16_attn_patched = True
    logger.info("Applied TPU patch: Gemma4AudioAttention native BF16 MXU matmuls.")


def maybe_patch_gemma4(
        model_config: Optional["ModelConfig"] = None) -> bool | None:
    if not _is_gemma4_model(model_config):
        return False

    _patch_gemma4_audio_attention_bf16()

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
        t_out = audio_features.shape[1]

        # When valid_lens matches t_out (e.g., 808 frames -> 202 soft tokens),
        # unbind rows without creating variable-length sub-slices.
        if all(v == t_out for v in valid_lens):
            return list(audio_features[:batch_size].unbind(0))

        return [
            audio_features[i, :valid_lens[i]]
            for i in range(batch_size)
        ]

    model_cls._process_audio_input = _patched_process_audio_input

    # 3. Compile fused PLE + token embedding + multimodal gather/merge into static XLA graphs
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
        if self.per_layer_embeddings is None:
            if not multimodal_embeddings or is_multimodal is None:
                return super(model_cls, self).embed_input_ids(input_ids)
            return super(model_cls, self).embed_input_ids(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        num_layers = self.config.text_config.num_hidden_layers
        ple_dim = self.config.text_config.hidden_size_per_layer_input

        if not multimodal_embeddings or is_multimodal is None:
            text_embed_fn = getattr(self, "_tpu_compiled_text_embed_fn", None)
            if text_embed_fn is None:
                lm_model = self.language_model.model
                lm = self.language_model

                def _text_embed_and_ple(
                    ids: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    ple = lm_model.get_per_layer_inputs(ids).reshape(
                        -1, num_layers, ple_dim)
                    txt = lm.embed_input_ids(ids)
                    return txt, ple

                text_embed_fn = torch.compile(
                    _text_embed_and_ple,
                    backend="tpu",
                    dynamic=False,
                )
                self._tpu_compiled_text_embed_fn = text_embed_fn

            inputs_embeds, per_layer_inputs = text_embed_fn(input_ids)
            self._latest_per_layer_inputs = per_layer_inputs
            return inputs_embeds

        mm_embed_fn = getattr(self, "_tpu_compiled_mm_embed_fn", None)
        if mm_embed_fn is None:
            lm_model = self.language_model.model
            lm = self.language_model

            def _mm_embed_and_ple(
                ids: torch.Tensor,
                mm_flat: torch.Tensor,
                is_mm: torch.Tensor,
                gather_idx: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                ple_ids = torch.where(is_mm, torch.zeros_like(ids), ids)
                ple = lm_model.get_per_layer_inputs(ple_ids).reshape(
                    -1, num_layers, ple_dim)
                txt = lm.embed_input_ids(ids)
                mm_cast = mm_flat.to(dtype=txt.dtype)
                gathered = mm_cast.index_select(0, gather_idx)
                merged = torch.where(is_mm.unsqueeze(-1), gathered, txt)
                return merged, ple

            mm_embed_fn = torch.compile(
                _mm_embed_and_ple,
                backend="tpu",
                dynamic=False,
            )
            self._tpu_compiled_mm_embed_fn = mm_embed_fn

        mm_flat = (
            multimodal_embeddings[0]
            if isinstance(multimodal_embeddings, (list, tuple))
            and len(multimodal_embeddings) == 1
            else torch.cat(list(multimodal_embeddings), dim=0)
        )
        gather_idx = getattr(is_multimodal, "_tpu_mm_gather_idx", None)
        if gather_idx is None:
            is_mm_dev = is_multimodal.to(input_ids.device, non_blocking=True)
            gather_idx = torch.clamp(
                torch.cumsum(is_mm_dev, dim=0, dtype=torch.int32) - 1,
                0,
                max(0, mm_flat.shape[0] - 1),
            )
        else:
            is_mm_dev = is_multimodal

        inputs_embeds, per_layer_inputs = mm_embed_fn(
            input_ids,
            mm_flat,
            is_mm_dev,
            gather_idx,
        )
        self._latest_per_layer_inputs = per_layer_inputs
        return inputs_embeds

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
        "Applied TPU patch: Gemma4 BF16 audio attention, 808-frame audio buckets, and compiled fused PLE/MM embedder."
    )
    return None

