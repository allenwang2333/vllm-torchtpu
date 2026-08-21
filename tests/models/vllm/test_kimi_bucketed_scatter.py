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
"""MoonViT's bucketed scatter must stay on the bucket shape axis.

``postprocess_encoder_output`` cuts each item out of the packed encoder output
and pads it to the model's declared ``mm_embed_bucket`` ladder, so the gather
is keyed on ``(packed rows, bucket)`` rather than on the image size. These
check Kimi's real implementation against that property -- the half that can
drift when the model changes -- and that the gather reads the right rows.
"""

import types

import pytest
import torch

from vllm_torchtpu.models.vllm.kimi_k3.model import \
    KimiK3ForConditionalGeneration as Kimi

HIDDEN = 8


def test_bucket_ladder_is_powers_of_two_above_the_floor():
    assert Kimi.mm_embed_bucket(0) == Kimi.MM_EMBED_BUCKET_MIN
    assert Kimi.mm_embed_bucket(1) == Kimi.MM_EMBED_BUCKET_MIN
    for n, expect in ((64, 64), (65, 128), (128, 128), (129, 256),
                      (1024, 1024), (1025, 2048)):
        assert Kimi.mm_embed_bucket(n) == expect, n


def test_gather_reads_the_real_tokens_and_clamps_the_pad():
    """Bucket padding is junk rows, but the real tokens must be exact."""
    model = types.SimpleNamespace(
        mm_embed_bucket=Kimi.mm_embed_bucket,
        MM_EMBED_BUCKET_MIN=Kimi.MM_EMBED_BUCKET_MIN,
    )
    # 180 rows for 170 real tokens, so the second item's 128-row bucket runs
    # off the end of the packed buffer and has to clamp.
    out = torch.arange(180 * HIDDEN, dtype=torch.float32).view(180, HIDDEN)
    dest: dict = {}
    Kimi.postprocess_encoder_output(model, {"default": out}, [0, 1], [70, 100],
                                    dest,
                                    clone=True)

    assert dest[0].shape[0] == 128 and dest[1].shape[0] == 128
    torch.testing.assert_close(dest[0][:70], out[:70])
    torch.testing.assert_close(dest[1][:100], out[70:170])
    # Rows past the end clamp to the last row rather than reading out of
    # bounds; they are junk either way, but they must not fault.
    torch.testing.assert_close(dest[1][-1], out[-1])
