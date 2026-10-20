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
"""TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS: which buckets apply W_UV before
the DCP merge.

0 keeps every bucket (backbone and MTP drafter). m > 0 takes the backbone's
buckets of at least m tokens on the power-of-2 ladder (`T > m // 2`) and never
the drafter, whose one graph serves every bucket.
"""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import attention


@pytest.mark.parametrize("backbone", [True, False])
@pytest.mark.parametrize("tokens", [4, 16, 32, 512, 1024, 4096])
def test_min_tokens_zero_takes_every_bucket(monkeypatch, tokens, backbone):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 0)
    assert attention._uv_before_merge_applies(tokens, backbone)


def test_min_tokens_takes_backbone_buckets_from_the_edge(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 1024)
    taken = [
        t
        for t in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        if attention._uv_before_merge_applies(t, True)
    ]
    assert taken == [1024, 2048, 4096]


@pytest.mark.parametrize("tokens", [4, 1024, 4096])
def test_min_tokens_never_takes_the_drafter(monkeypatch, tokens):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 1024)
    assert not attention._uv_before_merge_applies(tokens, False)


def test_retain_buffers_reads_each_buffer_in_warmup_only(monkeypatch):
    import torch

    from vllm_torchtpu.compilation import shape_variants

    out = torch.ones(4, 8, dtype=torch.bfloat16)
    w = torch.full((2, 3, 5), 2.0)
    scale = torch.full((1,), 0.5)

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: False)
    assert attention._retain_buffers(out, w, scale, None) is out

    reads = []
    real_reshape = torch.Tensor.reshape

    def reshape(self, *shape):
        if self is w or self is scale:
            reads.append(self)
        return real_reshape(self, *shape)

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(torch.Tensor, "reshape", reshape)
    got = attention._retain_buffers(out, w, scale, None)
    assert torch.equal(got, out)
    assert reads == [w, scale]
