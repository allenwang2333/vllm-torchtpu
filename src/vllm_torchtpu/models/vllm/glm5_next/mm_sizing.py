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
"""How large a GLM-5.3-Flash image or video gets, in patches and in tokens.

Split out of ``multimodal.py`` deliberately: this is the one piece of the
vision plumbing that can be checked without a device *or* a working
torchvision, and it is checked in
``tests/models/vllm/test_glm5_next_mm_sizing.py`` against the transformers
source text. Nothing here may import torch or vLLM.

The reason it cannot simply be inherited from ``Glm4vProcessingInfo`` is a
units change. GLM-4V's processor budgets in **pixels** and stores that budget
in ``size["longest_edge"]``; ``Glm4vProcessingInfo._get_image_max_pixels``
reads it straight out of ``preprocessor_config.json``. GLM-5.3-Flash's
processor is Glmga-derived and budgets in **tokens**
(``min_image_tokens=16``, ``max_image_tokens=8000``), keeping
``size = {"longest_edge": 1}`` as a placeholder that the resize path never
reads (``image_processing_glm5_next.py:127,140-168``). Inheriting unchanged
therefore computes a one-pixel image, and every downstream size is garbage.

Worse, the GLM-5.3-Flash repo ships **no** ``preprocessor_config.json`` at all
-- the image and video processor dicts live under ``processor_config.json``
-- so the inherited reader does not merely return the wrong number, it raises.
"""

import math

#: ``Glm5NextImageProcessor``/``Glm5NextVideoProcessor`` class defaults, which
#: are also what ``processor_config.json`` sets. Used only where the real HF
#: processor object is unavailable.
DEFAULT_MIN_IMAGE_TOKENS = 16
DEFAULT_MAX_IMAGE_TOKENS = 8000
DEFAULT_MAX_VIDEO_TOKENS = 240000


def pixels_per_token(*, patch_size: int, merge_size: int,
                     temporal_patch_size: int) -> int:
    """Pixels consumed by one LLM vision token: 2 * 28 * 28 = 1568 for GLM.

    This is the exchange rate between the two units above, and it is exact in
    both directions -- one token is exactly one ``merge_size``-square of
    ``patch_size``-squares across ``temporal_patch_size`` frames.
    """
    return temporal_patch_size * (patch_size * merge_size)**2


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = DEFAULT_MIN_IMAGE_TOKENS,
    max_pixels: int = DEFAULT_MAX_IMAGE_TOKENS,
) -> tuple[int, int]:
    """``transformers`` ``glm5_next.smart_resize``, mirrored line for line.

    Kept as a transcription rather than a tidier reimplementation because the
    canvas it returns has to agree with the HF processor's *exactly*: vLLM
    sizes the placeholder run from this and the encoder emits one embedding
    per patch of the real thing, so a one-patch disagreement is a hard length
    mismatch at merge time.

    Note the parameter names are HF's and are misleading: ``min_pixels`` and
    ``max_pixels`` arrive in **tokens** and are scaled to pixels on the first
    two lines.
    """
    pixels_per_tok = temporal_factor * factor**2
    min_pixels *= pixels_per_tok
    max_pixels *= pixels_per_tok

    def align(value, factor):
        return math.ceil(value / factor) * factor

    def fit_within_budget(aligned_frames):
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. At least "
                f"{minimum_pixels} pixels are required for one aligned patch.")

        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)

            pixel_budget = aligned_frames * candidate_height * candidate_width
            if pixel_budget <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(temporal_factor,
                         round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)

    return aligned_height, aligned_width


def vision_info(
    *,
    image_width: int,
    image_height: int,
    num_frames: int,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    max_tokens: int,
    min_tokens: int = DEFAULT_MIN_IMAGE_TOKENS,
    do_resize: bool = True,
) -> tuple[int, int, int]:
    """``(resized_width, resized_height, num_vision_tokens)``.

    ``num_frames`` is the pre-padding frame count; 1 for a still image.

    The resize sees ``max(num_frames, temporal_patch_size)`` rather than
    ``num_frames``. That is not a rounding convenience: HF's image path calls
    ``smart_resize(num_frames=temporal_factor)`` unconditionally
    (``image_processing_glm5_next.py:155``), and while ``aligned_frames`` is
    clamped to ``temporal_factor`` anyway, the minimum-size rescue branch
    divides by the *raw* ``num_frames``. Passing 1 there would upscale a
    below-floor image by sqrt(2) more than HF does.
    """
    factor = patch_size * merge_size
    if do_resize:
        resized_height, resized_width = smart_resize(
            num_frames=max(num_frames, temporal_patch_size),
            height=image_height,
            width=image_width,
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_pixels=min_tokens,
            max_pixels=max_tokens,
        )
    else:
        resized_height, resized_width = image_height, image_width

    # Both processors pad the frame axis up to a whole number of temporal
    # patches before patchifying (`video_processing_glm5_next.py:patchify`),
    # and a still image is one temporal patch.
    padded_num_frames = num_frames + (-num_frames % temporal_patch_size)
    grid_t = max(padded_num_frames // temporal_patch_size, 1)
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    num_vision_tokens = grid_t * grid_h * grid_w // (merge_size**2)
    return resized_width, resized_height, num_vision_tokens
