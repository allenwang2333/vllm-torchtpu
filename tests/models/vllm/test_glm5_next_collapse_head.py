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
"""GLM-5.3-Flash's mHC collapse is an unweighted mean, unlike DeepSeek-V4's.

Both collapse ``hc_mult`` residual streams to one vector and both return the
right shape, so substituting one for the other degrades output instead of
raising. These tests pin the difference.
"""

import pytest
import torch

from vllm_torchtpu.models.vllm.glm5_next.layers import mhc_collapse_head_mean


def test_matches_hf_hyper_head():
    """Equal to transformers' ``Glm5NextTextHyperHead.forward``.

    That method is ``hidden_streams.mean(dim=2)`` on ``[B, T, M, H]``; ours
    takes ``[T, M, H]`` and averages the same (stream) axis.
    """
    torch.manual_seed(0)
    streams = torch.randn(7, 4, 16)

    got = mhc_collapse_head_mean(streams)
    want = streams.unsqueeze(0).mean(dim=2).squeeze(0)

    assert got.shape == (7, 16)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_is_not_the_deepseek_v4_gated_head():
    """A learned gate on the same input gives a different answer.

    Guards against someone "unifying" the two collapse heads: GLM's checkpoint
    ships no ``hc_head_fn`` / ``hc_head_base`` / ``hc_head_scale`` at all, so
    the DeepSeek-V4 head could only run on invented weights.
    """
    from vllm_torchtpu.models.vllm.deepseek_v4.layers import mhc_collapse_head

    torch.manual_seed(0)
    hc_mult, hidden = 4, 16
    streams = torch.randn(7, hc_mult, hidden)

    gated = mhc_collapse_head(
        streams,
        hc_fn=torch.randn(hc_mult, hc_mult * hidden),
        hc_scale=torch.ones(1),
        hc_base=torch.zeros(hc_mult),
        rms_norm_eps=1e-5,
        hc_eps=1e-6,
    )
    mean = mhc_collapse_head_mean(streams)

    assert gated.shape == mean.shape
    assert not torch.allclose(gated, mean, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_preserves_dtype(dtype):
    streams = torch.randn(3, 4, 8, dtype=dtype)
    assert mhc_collapse_head_mean(streams).dtype is dtype


def test_averages_rather_than_sums():
    """Identical streams must collapse to that stream, not to M times it."""
    one = torch.randn(5, 1, 12)
    streams = one.repeat(1, 4, 1)
    torch.testing.assert_close(mhc_collapse_head_mean(streams),
                               one.squeeze(1),
                               rtol=1e-6,
                               atol=1e-6)
