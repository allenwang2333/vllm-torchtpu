# SPDX-License-Identifier: Apache-2.0
"""Determinism and distribution sanity for the keyed Gumbel primitive."""

import torch

from vllm_torchtpu.layers.vllm.sample.keyed_gumbel import (keyed_gumbel_noise,
                                                           keyed_uniform,
                                                           keyed_uniform_vector
                                                           )


def test_same_key_same_draw_across_call_sites():
    # The whole point of the primitive: the drafter and the verifier call it
    # independently and must see identical noise for identical keys.
    seeds = torch.tensor([7, -3, 12345678901234], dtype=torch.int64)
    pos = torch.tensor([0, 17, 4096], dtype=torch.int32)
    assert torch.equal(keyed_uniform(seeds, pos), keyed_uniform(seeds, pos))
    assert torch.equal(keyed_uniform_vector(seeds, pos, 64),
                       keyed_uniform_vector(seeds, pos, 64))
    assert torch.equal(keyed_gumbel_noise(seeds, pos, 64),
                       keyed_gumbel_noise(seeds, pos, 64))


def test_different_pos_different_draw():
    seeds = torch.full((4, ), 42, dtype=torch.int64)
    u0 = keyed_uniform(seeds, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert len(set(u0.tolist())) == 4


def test_different_seed_different_draw():
    pos = torch.full((4, ), 5, dtype=torch.int32)
    u = keyed_uniform(torch.tensor([1, 2, 3, 4], dtype=torch.int64), pos)
    assert len(set(u.tolist())) == 4


def test_vector_columns_are_distinct_and_prefix_stable():
    seeds = torch.tensor([9], dtype=torch.int64)
    pos = torch.tensor([3], dtype=torch.int32)
    wide = keyed_uniform_vector(seeds, pos, 128)[0]
    narrow = keyed_uniform_vector(seeds, pos, 32)[0]
    # Column j must not depend on the requested width (drafter and verifier
    # may materialize different vocab paddings).
    assert torch.equal(wide[:32], narrow)
    assert wide.unique().numel() > 120


def test_uniform_range_and_moments():
    seeds = torch.arange(512, dtype=torch.int64)
    pos = torch.arange(512, dtype=torch.int32)
    u = keyed_uniform_vector(seeds, pos, 256).flatten()
    assert u.min() > 0.0
    assert u.max() < 1.0
    assert abs(u.mean().item() - 0.5) < 0.01
    assert abs(u.var().item() - 1.0 / 12) < 0.005


def test_gumbel_noise_is_finite_and_gumbel_shaped():
    seeds = torch.arange(256, dtype=torch.int64)
    pos = torch.zeros(256, dtype=torch.int32)
    g = keyed_gumbel_noise(seeds, pos, 256).flatten()
    assert torch.isfinite(g).all()
    # Standard Gumbel: mean = Euler-Mascheroni (~0.5772), var = pi^2/6.
    assert abs(g.mean().item() - 0.5772) < 0.05
    assert abs(g.var().item() - 1.6449) < 0.15


def test_gumbel_argmax_matches_categorical_frequencies():
    # Gumbel-max over fixed logits should sample tokens with softmax
    # frequencies across many (seed, pos) keys.
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    probs = torch.softmax(logits, dim=-1)
    n = 20000
    seeds = torch.arange(n, dtype=torch.int64)
    pos = torch.full((n, ), 11, dtype=torch.int32)
    g = keyed_gumbel_noise(seeds, pos, 4)
    picks = (logits.unsqueeze(0) + g).argmax(dim=-1)
    freq = torch.bincount(picks, minlength=4).float() / n
    assert torch.allclose(freq, probs, atol=0.02)
