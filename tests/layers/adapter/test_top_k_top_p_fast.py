# Copyright 2025 Google LLC
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
"""The bf16 key search in `apply_top_k_top_p` must match the float32 search."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    _apply_top_k_top_p_float_search,
    apply_top_k_top_p,
    bf16_order_key,
)

VOCAB_SIZE = 512
TEMPERATURE_EPS = 1e-9


def _params(num_rows, generator, device):
    """Per-row (temperature, top_k, top_p) covering every branch."""
    temperatures = torch.rand(num_rows, 1, generator=generator) * 1.5 + 0.2
    top_k = torch.randint(1, 64, (num_rows, 1), generator=generator)
    top_k[0::7] = 0  # top-k disabled
    top_k[1::7] = VOCAB_SIZE  # vLLM's "no top-k"
    top_k[2::7] = VOCAB_SIZE + 5  # more than the vocabulary
    top_p = torch.rand(num_rows, 1, generator=generator) * 0.9 + 0.05
    top_p[3::7] = 1.0  # top-p disabled
    top_p[4::7] = 0.999999
    return (
        temperatures.to(device),
        top_k.to(dtype=torch.int32, device=device),
        top_p.to(device),
    )


def _logits(num_rows, generator, device, ties=False):
    logits = torch.randn(num_rows, VOCAB_SIZE, generator=generator) * 4
    if ties:
        # Coarse values -> many ties at every threshold, plus signed zeros.
        logits = torch.round(logits)
        logits[:, ::11] = -0.0
    return logits.to(torch.bfloat16).to(device)


def test_bf16_order_key_is_strictly_monotonic():
    bits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[~torch.isnan(values)]
    order = torch.argsort(values.float(), stable=True)
    keys = bf16_order_key(values)[order]
    # -0.0 and +0.0 compare equal; the stable sort keeps -0.0 first, and it
    # gets the smaller key, so keys are strictly increasing throughout.
    assert torch.all(keys[1:] > keys[:-1])
    assert keys.min() >= -(1 << 15) and keys.max() <= (1 << 15) - 1


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_on_bf16_scaled_logits(device, seed, ties):
    # sample_from_logits: bf16 logits / bf16 temperatures.
    g = torch.Generator().manual_seed(seed)
    logits = _logits(70, g, device, ties)
    temperatures, top_k, top_p = _params(70, g, device)
    scaled = logits / temperatures.to(torch.bfloat16)

    expected = _apply_top_k_top_p_float_search(scaled, top_k, top_p)

    # sample_from_logits passes nothing: order_source defaults to `scaled`.
    assert torch.equal(apply_top_k_top_p(scaled, top_k, top_p), expected)
    assert torch.equal(
        apply_top_k_top_p(scaled, top_k, top_p, order_source=logits), expected
    )


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_on_float32_scaled_logits(device, seed, ties):
    # Rejection sampler: float32(bf16 logits) / (temperature + eps), with
    # some temperature-0 rows that become huge.
    g = torch.Generator().manual_seed(seed)
    logits = _logits(70, g, device, ties)
    temperatures, top_k, top_p = _params(70, g, device)
    temperatures[5::9] = 0.0
    scaled = logits.to(torch.float32) / (temperatures + TEMPERATURE_EPS)

    expected = _apply_top_k_top_p_float_search(scaled, top_k, top_p)
    got = apply_top_k_top_p(scaled, top_k, top_p, order_source=logits)

    assert torch.equal(got, expected)


def test_matches_reference_with_masked_vocabulary(device):
    # Structured output sets disallowed logits to -inf; fewer finite logits
    # than top_k must still mask the -inf ones like topk_mask does.
    g = torch.Generator().manual_seed(0)
    logits = _logits(4, g, device)
    logits[:, 3:] = float("-inf")
    top_k = torch.tensor([[2], [3], [10], [0]], dtype=torch.int32, device=device)
    top_p = torch.tensor([[1.0], [0.5], [1.0], [0.9]], device=device)
    scaled = logits.to(torch.float32) / 0.7

    expected = _apply_top_k_top_p_float_search(scaled, top_k, top_p)
    got = apply_top_k_top_p(scaled, top_k, top_p, order_source=logits)

    assert torch.equal(got, expected)


def test_keeps_all_ties_at_the_kth_value(device):
    logits = torch.full((1, VOCAB_SIZE), -5.0)
    logits[0, :4] = torch.tensor([9.0, 8.0, 7.0, 6.0])
    logits[0, 100:120] = 1.0
    logits = logits.to(torch.bfloat16).to(device)
    top_k = torch.tensor([[5]], dtype=torch.int32, device=device)
    top_p = torch.ones((1, 1), device=device)

    got = apply_top_k_top_p(logits, top_k, top_p, order_source=logits)

    assert int((got > -1e11).sum()) == 24
    assert torch.equal(got, _apply_top_k_top_p_float_search(logits, top_k, top_p))


def test_float32_logits_use_the_float32_search(device):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(8, VOCAB_SIZE, generator=g).to(device)
    top_k = torch.full((8, 1), 5, dtype=torch.int32, device=device)
    top_p = torch.full((8, 1), 0.8, device=device)

    got = apply_top_k_top_p(logits, top_k, top_p, order_source=logits)

    assert torch.equal(got, _apply_top_k_top_p_float_search(logits, top_k, top_p))
