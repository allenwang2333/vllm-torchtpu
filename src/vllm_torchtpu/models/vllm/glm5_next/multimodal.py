# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal plumbing for GLM-5.3-Flash: preprocessing and the outer model.

The tower itself lives in ``vision.py``; this module is everything around it
-- how big an image is allowed to get, what the placeholder run looks like,
and how the encoder output is merged into the text embeddings on TPU.

Most of it is GLM-4V's, inherited. The three places it is not:

1. **Image and video sizing** (``Glm5NextProcessingInfo``). GLM-5.3-Flash
   budgets in tokens, not pixels, and ships no ``preprocessor_config.json``
   at all. See ``mm_sizing.py``.
2. **Video frame sampling**. GLM-5.3-Flash samples at a fixed 2 fps up to 2048
   frames; none of the three branches ``_construct_video_placeholder``
   dispatches to does that.
3. **No M-RoPE.** ``Glm5NextModel.__init__`` does ``del self.rope_deltas`` and
   feeds the decoder plain 1-D positions, so ``SupportsMRoPE`` -- which
   ``Glm4vForConditionalGeneration`` carries -- is deliberately absent here.
   Adding it would hand the text stack a ``(3, seq_len)`` position tensor it
   does not understand.

Module layout is load-bearing for FP8. The checkpoint's
``modules_to_not_convert`` names the text stack ``model.layers.*`` and the
tower ``visual.*``, and ``is_layer_skipped`` matches those against the
``prefix`` each ``LinearBase`` was constructed with, exactly. So the language
model is built at prefix ``""`` (giving ``model.layers.*``, not
``language_model.model.layers.*``) and the tower at ``visual``. Getting this
wrong does not raise: it silently hands bf16 weights to the FP8 method.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4_1v import (Glm4vDummyInputsBuilder,
                                                Glm4vMultiModalProcessor,
                                                Glm4vProcessingInfo)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   MultiModalEmbeddings,
                                                   SupportsMultiModal)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import _flatten_embeddings, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize

from vllm_torchtpu.models.vllm.glm5_next.mm_sizing import (
    DEFAULT_MAX_IMAGE_TOKENS, DEFAULT_MAX_VIDEO_TOKENS,
    DEFAULT_MIN_IMAGE_TOKENS, pixels_per_token, vision_info)
from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM
from vllm_torchtpu.utils import synchronize_tensors


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """GLM-4V's processing info with the pixel/token unit change corrected.

    ``_get_image_max_pixels`` and ``_get_video_max_pixels`` keep their
    inherited *pixel* semantics -- callers in the base class divide by
    ``pixels_per_token`` and would break otherwise -- but the number now comes
    from the processor's token budget rather than from a ``longest_edge`` that
    does not exist.
    """

    def _pixels_per_token(self) -> int:
        vision_config = self.get_hf_config().vision_config
        return pixels_per_token(
            patch_size=vision_config.patch_size,
            merge_size=vision_config.spatial_merge_size,
            temporal_patch_size=vision_config.temporal_patch_size,
        )

    def _get_min_image_tokens(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("min_image_tokens")) is not None:
            return int(override)
        return int(
            getattr(self.get_image_processor(), "min_image_tokens",
                    DEFAULT_MIN_IMAGE_TOKENS))

    def _get_image_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_image_tokens")) is not None:
            max_tokens = int(override)
        else:
            max_tokens = int(
                getattr(self.get_image_processor(), "max_image_tokens",
                        DEFAULT_MAX_IMAGE_TOKENS))
        return max_tokens * self._pixels_per_token()

    def _get_video_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_video_tokens")) is not None:
            max_tokens = int(override)
        else:
            # Same attribute name as the image processor's, 30x the value.
            max_tokens = int(
                getattr(self.get_video_processor(), "max_image_tokens",
                        DEFAULT_MAX_VIDEO_TOKENS))
        return max_tokens * self._pixels_per_token()

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 16,
        do_resize: bool = True,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_TOKENS * 2 * 28 * 28,
    ) -> tuple[ImageSize, int]:
        vision_config = self.get_hf_config().vision_config
        resized_width, resized_height, num_vision_tokens = vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            patch_size=vision_config.patch_size,
            merge_size=vision_config.spatial_merge_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            max_tokens=max_image_pixels // self._pixels_per_token(),
            min_tokens=self._get_min_image_tokens(),
            do_resize=do_resize,
        )
        return (ImageSize(width=resized_width,
                          height=resized_height), num_vision_tokens)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        result: dict[str, int] = {}

        if mm_counts.get("image", 0) > 0:
            result["image"] = self.get_max_image_tokens()

        if mm_counts.get("video", 0) > 0:
            max_vision_tokens = (self._get_video_max_pixels() //
                                 self._pixels_per_token())

            vision_config = self.get_hf_config().vision_config
            max_grid_t = (self._get_max_frames() //
                          vision_config.temporal_patch_size)

            # Each frame contributes <|begin_of_image|>, <|end_of_image|> and
            # its "N.N seconds" stamp on top of its vision tokens; the whole
            # run is wrapped in <|begin_of_video|>/<|end_of_video|>.
            tokenizer = self.get_tokenizer()
            max_ts_tokens = max(
                len(
                    tokenizer.encode(f"{t:.1f} seconds",
                                     add_special_tokens=False))
                for t in range(min(max_grid_t, 300)))

            result["video"] = (max_vision_tokens + max_grid_t *
                               (2 + max_ts_tokens) + 2)

        return result

    def _get_max_frames(self) -> int:
        return int(getattr(self.get_video_processor(), "max_frames", 2048))

    def _get_video_second_idx_glm46v(
        self,
        metadata: dict[str, Any],
        total_frames: int,
    ) -> list[int]:
        """``Glm5NextVideoProcessor.sample_frames``, mirrored.

        This overrides the *GLM-4.6V* hook because that is the branch
        ``_construct_video_placeholder`` dispatches to for us: GLM-5.3-Flash's
        sub-processors are named ``Glm5Next*``, so ``_is_glmga_model`` says no
        and the processor is not a ``Glm4vProcessor``. The inherited body
        picks frames at a duration-dependent fps (3 / 1 / 0.5); GLM-5.3-Flash
        is a flat 2 fps with no duration cap. A wrong frame count here means a
        wrong number of timestamp tokens, which shows up as a placeholder
        length mismatch rather than a wrong answer -- but only for videos long
        enough to cross one of the inherited thresholds.
        """
        video_processor = self.get_video_processor()

        video_fps = metadata["fps"]
        meta_frames = metadata.get("total_num_frames", total_frames)
        max_frame_idx = meta_frames - 1
        duration = metadata.get("duration",
                                round(max_frame_idx / video_fps) + 1)

        if not metadata.get("do_sample_frames", True):
            frame_indices = metadata["frames_indices"]
        else:
            target_fps = getattr(video_processor, "fps", 2)
            max_frames = self._get_max_frames()
            # `max_duration` is 0 for GLM-5.3-Flash, i.e. disabled, and the
            # break below is against the *uncapped* duration either way.
            max_duration = getattr(video_processor, "max_duration", 0)
            max_seconds = int(duration)
            effective = (duration if max_duration <= 0 else min(
                duration, max_duration))

            extract_t = min(int(effective * target_fps), max_frames)
            duration_per_frame = 1 / video_fps
            timestamps = [i * duration_per_frame for i in range(meta_frames)]

            if meta_frames < extract_t:
                frame_indices = np.linspace(0,
                                            meta_frames - 1,
                                            extract_t,
                                            dtype=int).tolist()
            else:
                frame_indices = []
                current_second = 0.0
                inv_fps = 1 / target_fps
                for frame_index in range(meta_frames):
                    if timestamps[frame_index] >= current_second:
                        current_second += inv_fps
                        frame_indices.append(frame_index)
                        if current_second >= max_seconds:
                            break

            if len(frame_indices) < extract_t:
                if len(frame_indices) == 0:
                    start, end = 0, max(meta_frames - 1, 0)
                else:
                    start, end = frame_indices[0], frame_indices[-1]
                frame_indices = np.linspace(start, end, extract_t,
                                            dtype=int).tolist()
            elif len(frame_indices) > extract_t:
                frame_indices = np.linspace(0,
                                            meta_frames - 1,
                                            extract_t,
                                            dtype=int).tolist()

        seen: set[int] = set()
        uniq: list[int] = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)
        if len(uniq) & 1:
            uniq.append(uniq[-1])

        # One stamp per temporal patch, i.e. per pair of sampled frames.
        return [int(idx / video_fps) for idx in uniq][::2]


class Glm5NextDummyInputsBuilder(Glm4vDummyInputsBuilder):
    """Unchanged from GLM-4V.

    ``get_dummy_text`` decodes the placeholders straight out of
    ``hf_config.{image,video}_token_id``, which GLM-5.3-Flash's config
    provides under exactly those names (154854 / 154855), and
    ``get_dummy_mm_data`` gets its sizes from the corrected processing info.
    """


class Glm5NextMultiModalProcessor(Glm4vMultiModalProcessor):
    """Unchanged from GLM-4V.

    Worth stating why, since the prompt-update targets are silent if wrong:
    ``_get_prompt_updates`` targets ``hf_processor.image_token`` for images
    and the literal id triple for the ``<|begin_of_video|><|video|>``
    ``<|end_of_video|>`` run for videos. Against GLM-5.3-Flash's tokenizer
    those are 154854 and
    ``[154832, 154855, 154833]``, and ``Glm5NextProcessor.__init__`` sets
    ``image_token``/``video_token`` to ``<|image|>``/``<|video|>``. The
    replacement text is GLM-4.6V's shape too -- per frame
    ``<|begin_of_image|>`` + image tokens + ``<|end_of_image|>`` + a
    ``"{:.1f} seconds"`` stamp -- which is exactly
    ``Glm5NextProcessor.replace_frame_token_id``.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm5NextDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(nn.Module, SupportsMultiModal,
                                       HasInnerState, IsHybrid):
    """GLM-5.3-Flash with its vision tower."""

    #: Consumed by ``configure_quant_config`` before any module is built, so
    #: ``is_layer_skipped`` can expand a fused prefix into the shard names the
    #: checkpoint's ignore list actually contains. The same dict covers the
    #: tower: its two fusions (block MLP and merger) are both ``gate_up_proj``,
    #: and ``attn.qkv`` is already fused in the checkpoint and listed under
    #: that name, so it must *not* appear here.
    packed_modules_mapping = Glm5NextForCausalLM.packed_modules_mapping

    _TOWER_PREFIX = "model.visual."

    #: ``(checkpoint_leaf, fused_leaf, shard_id)`` for the tower.
    _TOWER_STACKED = (
        ("gate_proj", "gate_up_proj", 0),
        ("up_proj", "gate_up_proj", 1),
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("image"):
            return "<|begin_of_image|><|image|><|end_of_image|>"
        if modality.startswith("video"):
            return "<|begin_of_video|><|video|><|end_of_video|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Glm5NextForConditionalGeneration requires a "
                             "multimodal config")
        if multimodal_config.mm_encoder_tp_mode == "data":
            # The tower would be *built* data-parallel by
            # `is_vit_use_data_parallel()` while this forward still runs it
            # rank-local, which is wrong rather than slow.
            raise ValueError(
                "glm5_next: mm_encoder_tp_mode='data' is not implemented for "
                "the vision tower; use the default tensor-parallel mode.")

        self.config = config
        self.multimodal_config = multimodal_config

        # Imported here, not at module scope: the tower reaches vLLM's
        # `glm_ocr`, which reaches `torchvision`. Keeping the import inside
        # `__init__` lets the preprocessing classes above be imported (and
        # tested) on a host without working torchvision ops.
        from vllm_torchtpu.models.vllm.glm5_next.vision import \
            Glm5NextVisionTransformer

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                quant_config=self._tower_quant_config(vllm_config.quant_config,
                                                      "visual"),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            # `prefix`, not `maybe_prefix(prefix, "language_model")`. See the
            # module docstring: the FP8 ignore list spells the text stack
            # `model.layers.*`, and nesting it would stop every one of the 34
            # bf16 KDA layers being recognised as unquantized.
            self.language_model = Glm5NextForCausalLM(vllm_config=vllm_config,
                                                      prefix=prefix)

    #: Tower layers whose skip status decides whether the tower is quantized
    #: at all. One per distinct construction site, because they are named
    #: independently in the checkpoint's ignore list and a partial list is
    #: exactly the case this must not silently accept.
    _TOWER_QUANT_WITNESSES = (
        "blocks.0.attn.qkv",
        "blocks.0.attn.proj",
        "blocks.0.mlp.gate_up_proj",
        "blocks.0.mlp.down_proj",
        "merger.proj",
        "merger.gate_up_proj",
        "merger.down_proj",
    )

    @classmethod
    def _tower_quant_config(cls, quant_config, tower_prefix: str):
        """``None``, once it is established that the tower is really bf16.

        Handing the tower the checkpoint's FP8 config and trusting
        ``is_layer_skipped`` to neutralise it looks equivalent and is not.
        ``glm_ocr.py:127`` builds the fused projection at prefix
        ``.qkv_proj`` whenever *any* quant config is present -- a rename that
        matches GLM-4.5V-FP8's ignore list. GLM-5.3-Flash's ignore list spells
        it ``visual.blocks.N.attn.qkv``, so the rename makes the skip miss and
        all 24 blocks build their qkv as an ``Fp8LinearMethod`` over bf16
        weights that ship no scales.

        Nothing about that fails loudly. The module attribute is still
        ``qkv``, so ``blocks.N.attn.qkv.weight`` still loads and a loader
        check still reports every tower tensor filled; the tower just returns
        non-finite values. Measured on the real checkpoint at TP=8: the tower
        emitted a (256, 4096) bf16 output that was entirely non-finite, after
        which the sampler emitted one unused token id forever and the reply
        detokenized to the empty string -- while text-only prompts of 18, 135
        and 343 tokens all answered correctly.

        So the tower is built with no quant config, which is what "every
        ``visual.*`` module is in ``modules_to_not_convert``" actually means,
        and is immune to how any of them are spelled. The premise is checked
        rather than assumed: if a future checkpoint really does quantize part
        of the tower, this raises instead of quietly ignoring it.
        """
        if quant_config is None:
            return None

        from vllm.model_executor.layers.quantization.utils.quant_utils import \
            is_layer_skipped

        ignored = list(getattr(quant_config, "ignored_layers", None) or [])
        fused = getattr(quant_config, "packed_modules_mapping", None) or {}
        kwargs: dict[str, object] = {"fused_mapping": fused}
        # `match_mode` is newer than the vLLM this image ships; passing it
        # unconditionally is a TypeError at engine start, and hard-coding
        # "exact" would be wrong for a config that asks for something else.
        import inspect
        if "match_mode" in inspect.signature(is_layer_skipped).parameters:
            kwargs["match_mode"] = getattr(quant_config,
                                           "ignored_layers_match_mode",
                                           "exact")
        quantized = [
            name for name in cls._TOWER_QUANT_WITNESSES
            if not is_layer_skipped(prefix=f"{tower_prefix}.{name}",
                                    ignored_layers=ignored,
                                    **kwargs)
        ]
        if quantized:
            raise ValueError(
                f"glm5_next: the vision tower is built unquantized, but this "
                f"checkpoint's {type(quant_config).__name__} does not exempt "
                f"{quantized} under prefix {tower_prefix!r}. Building it in "
                "bf16 anyway would silently ignore real quantization; add the "
                "quantized path deliberately instead.")
        return None

    # --- hybrid (Mamba-style) cache declarations -------------------------
    #
    # vLLM asks the *registered architecture* for these, so they cannot just
    # live on the text model. `Glm5NextForCausalLM` already unwraps
    # `text_config` itself, so the multimodal config passes straight through.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    # --- vision -----------------------------------------------------------

    def _grid_thw(self, value: object, name: str) -> torch.Tensor:
        if isinstance(value, list):
            value = torch.cat([torch.as_tensor(v) for v in value], dim=0)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value)}")
        return value.reshape(-1, 3)

    def _encode(self, pixel_values: object, grid_thw: torch.Tensor,
                name: str) -> tuple[torch.Tensor, ...]:
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [torch.as_tensor(p) for p in pixel_values], dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got "
                            f"{type(pixel_values)}")

        embeds = self.visual(pixel_values.type(self.visual.dtype),
                             grid_thw=grid_thw)

        # One embedding per merged patch; split back into one entry per item
        # because vLLM matches embeddings to placeholder runs positionally.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return tuple(embeds.split(sizes))

    def embed_multimodal(self,
                         **kwargs: object) -> MultiModalEmbeddings | None:
        embeddings: tuple[torch.Tensor, ...] = ()
        # Iterate over the kwargs in arrival order so that a prompt mixing
        # images and videos keeps the order the placeholders appear in.
        for key in list(kwargs):
            if key == "pixel_values":
                embeddings += self._encode(
                    kwargs["pixel_values"],
                    self._grid_thw(kwargs.get("image_grid_thw"),
                                   "image_grid_thw"), "pixel_values")
            elif key == "pixel_values_videos":
                embeddings += self._encode(
                    kwargs["pixel_values_videos"],
                    self._grid_thw(kwargs.get("video_grid_thw"),
                                   "video_grid_thw"), "pixel_values_videos")
            elif key == "image_embeds":
                embeddings += self._split_precomputed(
                    kwargs["image_embeds"],
                    self._grid_thw(kwargs.get("image_grid_thw"),
                                   "image_grid_thw"))
            elif key == "video_embeds":
                embeddings += self._split_precomputed(
                    kwargs["video_embeds"],
                    self._grid_thw(kwargs.get("video_grid_thw"),
                                   "video_grid_thw"))
        return embeddings or None

    def _split_precomputed(self, embeds: object,
                           grid_thw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not isinstance(embeds, torch.Tensor):
            raise TypeError(f"embeds must be a tensor, got {type(embeds)}")
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return tuple(embeds.type(self.visual.dtype).split(sizes))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter the encoder output into the text embeddings, TPU-safely.

        vLLM's ``_merge_multimodal_embeddings`` indexes with a boolean mask,
        which is a data-dependent shape and forces a host round-trip under the
        lazy TPU backend. The cumsum/``index_select``/``where`` form below is
        statically shaped, and it is the same one Kimi-K3 uses on this backend.
        """
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            # The placeholder rows keep the <|image|> token embedding.
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError(
                "is_multimodal is required when merging vision embeddings")

        mm_embeds = _flatten_embeddings(multimodal_embeddings).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        mask_cpu = is_multimodal.detach().reshape(-1).to(device="cpu",
                                                         dtype=torch.bool)
        expected = int(mask_cpu.sum().item())
        actual = mm_embeds.shape[0]
        if actual != expected:
            raise ValueError(f"Attempted to assign {actual} multimodal tokens "
                             f"to {expected} placeholders")
        if actual == 0:
            return inputs_embeds

        media_indices_cpu = mask_cpu.to(torch.int64).cumsum(0).sub_(1)
        media_indices_cpu.clamp_(min=0)
        media_indices = media_indices_cpu.to(device=inputs_embeds.device)
        aligned_media = mm_embeds.index_select(0, media_indices)
        mask = is_multimodal.reshape(-1, 1).to(device=inputs_embeds.device,
                                               dtype=torch.bool)
        merged = torch.where(mask, aligned_media, inputs_embeds)
        if merged.device.type != "tpu":
            return merged

        materialized = torch.empty_like(merged).copy_(merged)
        synchronize_tensors(materialized)
        return materialized

    # --- text -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        return self.language_model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor,
                       **kwargs: object) -> torch.Tensor:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.model",
            connector="visual.merger.",
            tower_model="visual.",
        )

    # --- weight loading ---------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Split ``model.visual.*`` off the stream; the rest is the text model.

        Streamed rather than partitioned into lists: the real checkpoint is
        328 GB and the loader hands weights over one at a time.
        """
        tower_params = dict(self.visual.named_parameters())
        loaded: set[str] = set()

        def _text_weights():
            for name, weight in weights:
                if name.startswith(self._TOWER_PREFIX):
                    self._load_tower_weight(name[len(self._TOWER_PREFIX):],
                                            weight, tower_params, loaded)
                    continue
                yield name, weight

        for name in self.language_model.load_weights(_text_weights()):
            loaded.add(f"language_model.{name}")
        return loaded

    # A classmethod, not an instance method: it touches no instance state, and
    # `scripts/glm5_next/vision_probe.py` loads a bare tower with it so that
    # the on-device check exercises this mapping rather than a copy of it.
    @classmethod
    def _load_tower_weight(cls, name: str, weight: torch.Tensor,
                           tower_params: dict[str, torch.nn.Parameter],
                           loaded: set[str]) -> None:
        for ckpt_leaf, fused_leaf, shard_id in cls._TOWER_STACKED:
            head, sep, tail = name.rpartition(f".{ckpt_leaf}.")
            if not sep:
                continue
            mapped = f"{head}.{fused_leaf}.{tail}"
            param = tower_params[mapped]
            param.weight_loader(param, weight, shard_id)
            loaded.add(f"visual.{mapped}")
            return

        param = tower_params[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, weight)
        loaded.add(f"visual.{name}")
