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
"""GLM-5.3-Flash image/video sizing, checked against the transformers source.

The port's ``smart_resize`` has to agree with HF's *exactly*, not
approximately: vLLM reserves one placeholder token per merged patch of the
canvas this returns, while the encoder emits one embedding per patch of the
canvas the HF processor actually produced. One patch of disagreement is a
merge-time length error, and off-by-a-row is the likely failure mode because
the two implementations differ in shape (HF binary-searches the content
height; the Qwen-style resize GLM-4V inherits scales by a square root).

So the reference here is HF's own function, lifted out of the installed
transformers by AST rather than imported -- importing the module pulls
torchvision, whose compiled ops do not load in the CPU dev venv.
"""

import ast
import math
import os
from pathlib import Path

import pytest

from vllm_torchtpu.models.vllm.glm5_next.mm_sizing import (smart_resize,
                                                           vision_info)

#: The differential tests below need transformers' own glm5_next image
#: processor to diff against. Resolved from the installed package when it has
#: one, and otherwise from a source overlay named by
#: ``GLM5_NEXT_TRANSFORMERS_SRC`` (the directory holding ``transformers/
#: models/glm5_next/``). Skipped, with the variable named, when neither is
#: present.
_TRANSFORMERS_SRC_ENV = "GLM5_NEXT_TRANSFORMERS_SRC"


def _hf_image_processor_source() -> Path | None:
    try:
        import transformers.models.glm5_next.image_processing_glm5_next as m
        return Path(m.__file__)
    except ImportError:
        pass
    overlay = os.environ.get(_TRANSFORMERS_SRC_ENV)
    if not overlay:
        return None
    path = (Path(overlay) / "transformers" / "models" / "glm5_next" /
            "image_processing_glm5_next.py")
    return path if path.exists() else None


# GLM-5.3-Flash's vision_config plus processor_config.json.
_PATCH, _MERGE, _TEMPORAL = 14, 2, 2
_FACTOR = _PATCH * _MERGE
_MAX_IMAGE_TOKENS = 8000
_MIN_IMAGE_TOKENS = 16


@pytest.fixture(scope="module")
def hf_smart_resize():
    """transformers' own ``smart_resize``, extracted without importing it.

    The module it lives in pulls torchvision at import time, which is exactly
    what the CPU rung cannot have. Parsing out the one function keeps the
    differential test available anyway.
    """
    src = _hf_image_processor_source()
    if src is None:
        pytest.skip("transformers has no glm5_next image processor; point "
                    f"${_TRANSFORMERS_SRC_ENV} at a source tree that does")
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "smart_resize")
    ns: dict = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    return ns["smart_resize"]


_SIZES = [
    (28, 28),
    (32, 24),
    (224, 224),
    (448, 448),
    (1024, 768),
    (1920, 1080),
    (4096, 4096),
    (8000, 137),  # extreme aspect ratio: the binary search's hard case
    (7, 5),  # below the minimum-token floor
    (1, 1),
]


@pytest.mark.parametrize("height,width", _SIZES)
def test_image_resize_matches_hf(hf_smart_resize, height, width):
    kwargs = dict(num_frames=_TEMPORAL,
                  height=height,
                  width=width,
                  temporal_factor=_TEMPORAL,
                  factor=_FACTOR,
                  min_pixels=_MIN_IMAGE_TOKENS,
                  max_pixels=_MAX_IMAGE_TOKENS)
    assert smart_resize(**kwargs) == hf_smart_resize(**kwargs)


@pytest.mark.parametrize("num_frames", [2, 4, 16, 64, 300])
@pytest.mark.parametrize("height,width", [(1080, 1920), (480, 640)])
def test_video_resize_matches_hf(hf_smart_resize, num_frames, height, width):
    kwargs = dict(num_frames=num_frames,
                  height=height,
                  width=width,
                  temporal_factor=_TEMPORAL,
                  factor=_FACTOR,
                  min_pixels=_MIN_IMAGE_TOKENS,
                  max_pixels=240000)
    assert smart_resize(**kwargs) == hf_smart_resize(**kwargs)


def test_the_placeholder_longest_edge_would_have_been_one_pixel():
    """Why ``_get_image_max_pixels`` cannot be inherited.

    ``Glm4vProcessingInfo`` feeds ``size["longest_edge"]`` to ``smart_resize``
    as a pixel budget. GLM-5.3-Flash's processor leaves that key at the
    literal placeholder ``1``, and in *these* units a budget of 1 is one token
    -- not a whole image, and 8000x smaller than the real cap.
    """
    placeholder = smart_resize(num_frames=_TEMPORAL,
                               height=1024,
                               width=768,
                               temporal_factor=_TEMPORAL,
                               factor=_FACTOR,
                               max_pixels=1)
    assert placeholder == (_FACTOR, _FACTOR)

    real = smart_resize(num_frames=_TEMPORAL,
                        height=1024,
                        width=768,
                        temporal_factor=_TEMPORAL,
                        factor=_FACTOR,
                        max_pixels=_MAX_IMAGE_TOKENS)
    assert real == (1036, 784)  # 1024 -> 1036, 768 -> 784, both aligned to 28


def test_image_token_count_fills_the_budget_without_exceeding_it():
    """The size vLLM profiles with. Under budget, and only just."""
    width, height, tokens = vision_info(image_width=9999999,
                                        image_height=9999999,
                                        num_frames=1,
                                        patch_size=_PATCH,
                                        merge_size=_MERGE,
                                        temporal_patch_size=_TEMPORAL,
                                        max_tokens=_MAX_IMAGE_TOKENS)
    assert width % _FACTOR == 0 and height % _FACTOR == 0
    # A square cannot hit 8000 exactly: 89 merged patches a side is 7921 and
    # 90 is 8100. The budget is a cap, not a target.
    assert (width, height, tokens) == (2492, 2492, 7921)
    assert tokens <= _MAX_IMAGE_TOKENS
    one_step_bigger = ((width + _FACTOR) // _PATCH)**2 // _MERGE**2
    assert one_step_bigger > _MAX_IMAGE_TOKENS


def test_small_image_is_not_upscaled_past_its_own_grid():
    """A 448x448 image is already aligned and well inside budget."""
    width, height, tokens = vision_info(image_width=448,
                                        image_height=448,
                                        num_frames=1,
                                        patch_size=_PATCH,
                                        merge_size=_MERGE,
                                        temporal_patch_size=_TEMPORAL,
                                        max_tokens=_MAX_IMAGE_TOKENS)
    assert (width, height) == (448, 448)
    # 448/14 = 32 patches a side, merged 2x2 -> 16x16 = 256 tokens.
    assert tokens == 256


def test_a_still_image_is_one_temporal_patch():
    """``num_frames=1`` must not halve the grid or double it."""
    for num_frames in (1, 2):
        _, _, tokens = vision_info(image_width=448,
                                   image_height=448,
                                   num_frames=num_frames,
                                   patch_size=_PATCH,
                                   merge_size=_MERGE,
                                   temporal_patch_size=_TEMPORAL,
                                   max_tokens=_MAX_IMAGE_TOKENS)
        assert tokens == 256


@pytest.mark.parametrize("height,width", [(7, 5), (1, 1), (28, 28)])
def test_below_floor_image_uses_the_hf_frame_count(hf_smart_resize, height,
                                                   width):
    """``num_frames=1`` must still resize as if there were two frames.

    Only the minimum-size rescue branch reads the raw frame count, so this is
    the one regime where the difference shows -- and it is exactly the regime
    a tiny image lands in.
    """
    resized_width, resized_height, _ = vision_info(
        image_width=width,
        image_height=height,
        num_frames=1,
        patch_size=_PATCH,
        merge_size=_MERGE,
        temporal_patch_size=_TEMPORAL,
        max_tokens=_MAX_IMAGE_TOKENS)
    hf_height, hf_width = hf_smart_resize(num_frames=_TEMPORAL,
                                          height=height,
                                          width=width,
                                          temporal_factor=_TEMPORAL,
                                          factor=_FACTOR,
                                          min_pixels=_MIN_IMAGE_TOKENS,
                                          max_pixels=_MAX_IMAGE_TOKENS)
    assert (resized_height, resized_width) == (hf_height, hf_width)


def test_video_tokens_scale_with_frames():
    _, _, tokens = vision_info(image_width=448,
                               image_height=448,
                               num_frames=16,
                               patch_size=_PATCH,
                               merge_size=_MERGE,
                               temporal_patch_size=_TEMPORAL,
                               max_tokens=240000)
    assert tokens == 256 * 8  # grid_t = 16 // 2
