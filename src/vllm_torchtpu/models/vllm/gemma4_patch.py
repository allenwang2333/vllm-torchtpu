# SPDX-License-Identifier: Apache-2.0
"""Model-specific patches and TPU compilation optimizations for Gemma-4 multimodal/audio."""

import functools
from typing import TYPE_CHECKING, Optional

import numpy as np
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


def _patch_gemma4_audio_feature_extractor() -> None:
    try:
        import numpy as np
        from transformers.feature_extraction_utils import BatchFeature
        from transformers.models.gemma4 import feature_extraction_gemma4
    except ImportError:
        return

    fe_cls = getattr(feature_extraction_gemma4, "Gemma4AudioFeatureExtractor",
                     None)
    if fe_cls is None or getattr(fe_cls, "_tpu_fast_stft_patched", False):
        return

    orig_call = fe_cls.__call__

    @functools.wraps(orig_call)
    def _fast_fe_call(
        self,
        raw_speech,
        padding="longest",
        max_length: int | None = 480_000,
        truncation: bool = True,
        pad_to_multiple_of: int | None = 128,
        return_tensors=None,
        return_attention_mask: bool | None = True,
        **kwargs,
    ):
        # Fallback to original if dither is enabled
        if getattr(self, "dither", 0.0) > 0.0:
            return orig_call(
                self,
                raw_speech,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )

        window_t = getattr(self, "_tpu_window_f32", None)
        if window_t is None:
            window_t = torch.from_numpy(self.window.astype(np.float32))
            self._tpu_window_f32 = window_t
            self._tpu_win0_scale = float(window_t[0].item() *
                                         (1.0 - self.preemphasis))
            self._tpu_mel_filters_f32 = torch.from_numpy(
                self.mel_filters.astype(np.float32))
            self._tpu_per_bin_mean_f32 = (
                torch.from_numpy(
                    self.per_bin_mean.astype(np.float32)).squeeze(0)
                if self.per_bin_mean is not None else None)
            self._tpu_per_bin_stddev_f32 = (
                torch.from_numpy(
                    self.per_bin_stddev.astype(np.float32)).squeeze(0)
                if self.per_bin_stddev is not None else None)

        if isinstance(raw_speech, np.ndarray):
            speeches = ([raw_speech] if raw_speech.ndim == 1 else
                        [np.asarray(s, dtype=np.float32).reshape(-1)
                         for s in raw_speech])
        elif isinstance(raw_speech, (list, tuple)):
            if len(raw_speech) > 0 and isinstance(
                    raw_speech[0], (np.ndarray, list, tuple)):
                speeches = [
                    np.asarray(s, dtype=np.float32).reshape(-1)
                    for s in raw_speech
                ]
            else:
                speeches = [np.asarray(raw_speech, dtype=np.float32).reshape(-1)]
        else:
            speeches = [np.asarray(raw_speech, dtype=np.float32).reshape(-1)]

        if truncation and max_length is not None:
            speeches = [s[:max_length] for s in speeches]

        max_len = max(len(s) for s in speeches)
        if pad_to_multiple_of:
            rem = max_len % pad_to_multiple_of
            if rem != 0:
                max_len += pad_to_multiple_of - rem

        pad_left = self.frame_length // 2
        frame_size = self.frame_length + 1
        hop = self.hop_length
        pre = float(self.preemphasis)
        mel_floor = float(self.mel_floor)

        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)

        batch_size = len(speeches)
        w = torch.zeros((batch_size, pad_left + max_len), dtype=torch.float32)
        valid_lens = []
        for i, s in enumerate(speeches):
            n_s = len(s)
            valid_lens.append(n_s)
            w[i, pad_left:pad_left + n_s] = torch.from_numpy(
                np.ascontiguousarray(s, dtype=np.float32))

        if self.input_scale_factor != 1.0:
            w.mul_(float(self.input_scale_factor))

        num_mel_frames = (w.shape[1] - frame_size) // hop + 1
        if pre > 0.0 and self.preemphasis_htk_flavor:
            w_pre = torch.empty_like(w)
            w_pre[:, 0] = w[:, 0]
            torch.sub(w[:, 1:], w[:, :-1], alpha=pre, out=w_pre[:, 1:])
            uf_pre = w_pre[:, :(num_mel_frames - 1) * hop +
                           self.frame_length].unfold(-1, self.frame_length,
                                                     hop)
            frames = uf_pre * window_t
            win0_scale = self._tpu_win0_scale
            if win0_scale != 0.0:
                uf_orig0 = w[:, :(num_mel_frames - 1) * hop + 1:hop]
                frames[..., 0] = uf_orig0 * win0_scale
            else:
                frames[..., 0] = 0.0
        elif pre > 0.0:
            uf = w.unfold(-1, frame_size, hop)
            frames = (uf[..., 1:] - pre * uf[..., :-1]) * window_t
        else:
            uf = w.unfold(-1, frame_size, hop)
            frames = uf[..., :-1] * window_t

        stft = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        mag = torch.abs(stft)
        mel = torch.matmul(mag, self._tpu_mel_filters_f32)
        log_mel = torch.log(mel + mel_floor)
        if self._tpu_per_bin_mean_f32 is not None:
            log_mel.sub_(self._tpu_per_bin_mean_f32)
        if self._tpu_per_bin_stddev_f32 is not None:
            log_mel.div_(self._tpu_per_bin_stddev_f32)

        masks = torch.zeros((batch_size, num_mel_frames), dtype=torch.bool)
        for i, n_s in enumerate(valid_lens):
            vf = max(0, (pad_left + n_s - frame_size) // hop + 1)
            masks[i, :vf] = True
            if vf < num_mel_frames:
                log_mel[i, vf:] = 0.0

        if return_tensors == "pt":
            return BatchFeature({
                "input_features": [log_mel[i] for i in range(batch_size)],
                "input_features_mask": [masks[i] for i in range(batch_size)],
            })
        return BatchFeature(
            {
                "input_features":
                [log_mel[i].numpy() for i in range(batch_size)],
                "input_features_mask":
                [masks[i].numpy() for i in range(batch_size)],
            },
            tensor_type=return_tensors,
        )

    fe_cls.__call__ = _fast_fe_call
    fe_cls._tpu_fast_stft_patched = True
    logger.info(
        "Applied TPU patch: Gemma4AudioFeatureExtractor vectorized PyTorch float32 STFT."
    )


def _patch_fast_wav_audio_loader() -> None:
    try:
        from vllm.multimodal.media.audio import AudioMediaIO
    except Exception:
        return

    if getattr(AudioMediaIO, "_tpu_fast_wav_patched", False):
        return

    orig_load_bytes = AudioMediaIO.load_bytes

    @functools.wraps(orig_load_bytes)
    def _fast_load_bytes(self, data: bytes) -> tuple[np.ndarray, float]:
        # Fast zero-copy parser for standard uncompressed 16-bit PCM WAV (RIFF/WAVE)
        if (len(data) > 44 and data[:4] == b"RIFF"
                and data[8:16] == b"WAVEfmt "):
            import struct
            fmt_size, audio_fmt, num_channels, sample_rate, _, _, bits_per_sample = (
                struct.unpack_from("<IHHIIHH", data, 16))
            if audio_fmt == 1 and bits_per_sample == 16 and num_channels == 1:
                pos = 20 + fmt_size
                while pos + 8 <= len(data):
                    chunk_id = data[pos:pos + 4]
                    chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
                    pos += 8
                    if chunk_id == b"data":
                        end = min(len(data), pos + chunk_size)
                        n_samples = (end - pos) // 2
                        pcm = np.frombuffer(
                            data, dtype="<i2", count=n_samples, offset=pos)
                        wav_f32 = pcm.astype(np.float32) * (1.0 / 32768.0)
                        resample_sr = getattr(self, "resample_sr", None) or (
                            getattr(self, "kwargs", {}) or {}).get("sr")
                        if resample_sr is not None and int(
                                resample_sr) != sample_rate:
                            return orig_load_bytes(self, data)
                        return wav_f32, float(sample_rate)
                    pos += chunk_size + (chunk_size & 1)
        return orig_load_bytes(self, data)

    AudioMediaIO.load_bytes = _fast_load_bytes
    AudioMediaIO._tpu_fast_wav_patched = True
    logger.info("Applied TPU patch: AudioMediaIO zero-copy PCM_16 WAV loader.")


def maybe_patch_gemma4(
        model_config: Optional["ModelConfig"] = None) -> bool | None:
    _patch_fast_wav_audio_loader()
    _patch_gemma4_audio_feature_extractor()
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
            ) -> tuple[torch.Tensor, torch.Tensor]:
                out = audio_tower(feats, mask)
                enc = out[0] if isinstance(out, tuple) else out.last_hidden_state
                emb = embed_audio(inputs_embeds=enc)
                flat = emb.reshape(-1, emb.shape[-1])
                target_rows = feats.shape[0] * 256
                if flat.shape[0] < target_rows:
                    flat = F.pad(flat, (0, 0, 0, target_rows - flat.shape[0]))
                return emb, flat

            compiled_fn = torch.compile(
                _audio_tower_and_embed,
                backend="tpu",
                dynamic=False,
            )
            self._tpu_compiled_audio_fn = compiled_fn

        audio_features, parent_flat = compiled_fn(
            input_features, input_features_mask)
        t_out = int(audio_features.shape[1])

        if all(v == t_out for v in valid_lens):
            rows = list(audio_features[:batch_size].unbind(0))
        else:
            rows = [
                audio_features[i, :valid_lens[i]]
                for i in range(batch_size)
            ]
        for i, r in enumerate(rows):
            r._tpu_parent_flat = parent_flat
            r._tpu_parent_offset = i * t_out
        return rows

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
        n_tok = int(input_ids.shape[0])
        if mm_flat.shape[0] > n_tok:
            mm_flat = mm_flat[:n_tok]
        elif mm_flat.shape[0] < n_tok:
            mm_flat = F.pad(mm_flat, (0, 0, 0, n_tok - mm_flat.shape[0]))
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

