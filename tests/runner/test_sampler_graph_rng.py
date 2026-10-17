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
"""TPU_SAMPLER_GRAPH_RNG: the fused verify chain draws its three uniforms
in-graph from the default generator.

Exactness (TPU only): over several steps, the in-graph chain on the default
generator seeded S selects the same tokens, bit for bit, as the eager-draw
chain on a dedicated generator seeded S (the runner's other mode), including
across a verify-trim variant and with eager draws interleaved (the stream stays
shared). Positive control: a different seed changes some output.
"""

import pytest
import torch

from test_sampler_verify_trim import (  # noqa: F401
    PADDED_NUM_REQS,
    VARIANTS,
    _fused,
    _inputs,
    _raise_dynamo_recompile_limit,
    _run,
    _vocab,
    dev,
)

CASES = [[3], [3, 3], [2, 1], [1, 1, 3], [3, 3, 2]]


def _default_gen(dev):
    return getattr(torch, dev.type).default_generators[dev.index or 0]


def _eager_draws(gen, padded, vocab, dev):
    b = torch.rand((PADDED_NUM_REQS, vocab), dtype=torch.float32, device=dev, generator=gen)
    a = torch.rand((padded,), dtype=torch.float32, device=dev, generator=gen)
    r = torch.rand((padded, vocab), dtype=torch.float32, device=dev, generator=gen)
    return b, a, r


def _steps(dev, seed, graph, interleave):
    vocab, fn = _vocab(dev), _fused(dev)
    if graph:
        gen = _default_gen(dev)
    else:
        gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    outs = []
    for i, case in enumerate(CASES):
        _, logits, md, temps, top_k, top_p, padded = _inputs(case, 100 + i, vocab, dev)
        kw = dict(VARIANTS[0])
        if sum(case) <= 8 and padded > 8:
            kw["verify_rows"] = 8
        if graph:
            out = _run(fn, dev, logits, md, temps, top_k, top_p, (None, None, None), 0,
                       graph_rng=True, **kw)
        else:
            draws = _eager_draws(gen, padded, vocab, dev)
            out = _run(fn, dev, logits, md, temps, top_k, top_p, draws, 0, **kw)
        outs.append(out)
        if interleave:  # an eager draw on the sampling generator between steps
            outs.append(torch.rand((8, 128), device=dev, generator=gen).cpu())
    return outs


@pytest.mark.parametrize("interleave", [False, True])
def test_graph_rng_is_exact(dev, interleave):
    if dev.type != "tpu":
        pytest.skip("default-generator equivalence is a TPU property")
    eager = _steps(dev, 7, False, interleave)
    graph = _steps(dev, 7, True, interleave)
    for i, (e, g) in enumerate(zip(eager, graph)):
        assert torch.equal(e, g), (i, e, g)


def test_positive_control_seed_is_caught(dev):
    if dev.type != "tpu":
        pytest.skip("TPU only")
    a = _steps(dev, 7, True, False)
    b = _steps(dev, 8, True, False)
    assert any(not torch.equal(x, y) for x, y in zip(a, b))
