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
"""How far the fused EP kernel's semantics sit from the GMM path's, on CPU.

The Pallas kernel cannot run here (TPU interpret mode trips a JAX defect on the
kernel's reshaped DMA refs), so this models both paths' *arithmetic* in plain
JAX and measures the gap. Two differences are structural and cannot be tuned
away:

* the GMM path selects with ``torch.topk``; the kernel runs ``top_k`` passes of
  ``argmax`` with the previous winner masked to -inf. These agree exactly when
  the scores are fp32 -- which is why the kernel scores in fp32 rather than in
  the token dtype it receives the logits in.
* the GMM path quantizes each matmul's activations to fp8 per row-block
  (``gmm_v2.maybe_quantize_lhs``); this kernel does not, and keeps them in the
  token dtype. That makes the kernel the *more* accurate of the two and the
  slower one -- bf16 x fp8 runs at half the v7x MXU rate -- and it is a
  permanent property of this port, not a flag.

``test_bf16_scoring_would_reorder_near_ties`` is the evidence for the first
bullet, and the numbers the others print are the tolerance an e2e numerics gate
must accept.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

TOP_K = 10
NUM_EXPERTS = 512
HIDDEN = 256
INTERMEDIATE = 128
TOKENS = 256
FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(rng.normal(0, 1, (TOKENS, HIDDEN)), jnp.bfloat16)
    logits = jnp.asarray(rng.normal(0, 1, (TOKENS, NUM_EXPERTS)), jnp.bfloat16)
    w13 = rng.normal(0, 0.05, (NUM_EXPERTS, 2, HIDDEN, INTERMEDIATE))
    w2 = rng.normal(0, 0.05, (NUM_EXPERTS, INTERMEDIATE, HIDDEN))
    # Per-output-channel fp8, exactly what the loader produces.
    s13 = np.abs(w13).max(axis=2, keepdims=True) / FP8_MAX
    s2 = np.abs(w2).max(axis=1, keepdims=True) / FP8_MAX
    q13 = jnp.asarray(w13 / s13, jnp.float8_e4m3fn)
    q2 = jnp.asarray(w2 / s2, jnp.float8_e4m3fn)
    return tokens, logits, q13, jnp.asarray(s13), q2, jnp.asarray(s2)


def _gmm_routing(logits):
    """fp32 softmax + top_k + renormalize -- moe_routing.select_experts."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    weights, ids = jax.lax.top_k(probs, TOP_K)
    return weights / jnp.sum(weights, axis=-1, keepdims=True), ids


def _kernel_routing(logits, score_dtype=jnp.float32):
    """fp32 softmax + k passes of argmax + renormalize -- kernel.get_top_k."""
    scores = jax.nn.softmax(logits.astype(score_dtype),
                            axis=-1).astype(jnp.float32)
    iota = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    weights, ids = [], []
    for k in range(TOP_K):
        top = jnp.max(scores, axis=1, keepdims=True)
        idx = jnp.argmax(scores, axis=1, keepdims=True)
        weights.append(top)
        ids.append(idx)
        scores = jnp.where(iota == idx, -jnp.inf, scores)
    weights = jnp.concatenate(weights, axis=1)
    return (weights / jnp.sum(weights, axis=-1, keepdims=True),
            jnp.concatenate(ids, axis=1))


def _quantize_rows(x, block):
    """Per-row-block fp8 round-trip, as ``gmm_v2.maybe_quantize_lhs`` does."""
    parts = []
    for start in range(0, x.shape[-1], block):
        chunk = x[..., start:start + block]
        scale = jnp.max(jnp.abs(chunk), axis=-1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        parts.append(
            ((chunk * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32) *
             scale))
    return jnp.concatenate(parts, axis=-1)


def _moe(tokens, routing, q13, s13, q2, s2, *, quant_lhs, block=512):
    weights, ids = routing
    w13 = q13.astype(jnp.float32) * s13
    w2 = q2.astype(jnp.float32) * s2
    x = tokens.astype(jnp.float32)
    out = jnp.zeros((tokens.shape[0], HIDDEN), jnp.float32)
    for k in range(TOP_K):
        expert = ids[:, k]
        gate_w = w13[expert, 0]
        up_w = w13[expert, 1]
        down_w = w2[expert]
        lhs = _quantize_rows(x, block) if quant_lhs else x
        gate = jnp.einsum("td,tdf->tf", lhs, gate_w)
        up = jnp.einsum("td,tdf->tf", lhs, up_w)
        act = jax.nn.silu(gate) * up
        act = _quantize_rows(act, block) if quant_lhs else act
        out += jnp.einsum("tf,tfd->td", act, down_w) * weights[:, k:k + 1]
    return out


def _report(name, actual, expected, routing_a, routing_b):
    same = np.asarray(
        jnp.all(jnp.sort(routing_a[1], axis=-1) == jnp.sort(routing_b[1],
                                                            axis=-1),
                axis=-1))
    diff = np.asarray(actual - expected, np.float32)
    ref = np.asarray(expected, np.float32)
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(ref)
    rel_l2_same = (np.linalg.norm(diff[same]) /
                   np.linalg.norm(ref[same]) if same.any() else float("nan"))
    print(f"[{name}] routing agrees on {same.mean() * 100:.2f}% of tokens | "
          f"max_abs={np.abs(diff).max():.4g} rel_l2={rel_l2:.4g} "
          f"rel_l2(same routing)={rel_l2_same:.4g}")
    return same.mean(), rel_l2, rel_l2_same


def test_fused_kernel_semantics_track_the_gmm_path(capsys):
    tokens, logits, q13, s13, q2, s2 = _inputs()
    gmm_route = _gmm_routing(logits)
    ker_route = _kernel_routing(logits)

    gmm = _moe(tokens, gmm_route, q13, s13, q2, s2, quant_lhs=True)
    fused = _moe(tokens, ker_route, q13, s13, q2, s2, quant_lhs=False)

    with capsys.disabled():
        agree, rel_l2, rel_l2_same = _report("fused vs gmm", fused, gmm,
                                             ker_route, gmm_route)

    # Both paths score in fp32, so the routing agrees; the floor is here to
    # catch a regression, not to license a gap.
    assert agree > 0.90
    # What is left on the tokens both paths route identically is the GMM
    # reference's *own* fp8 activation error, which this kernel does not incur.
    assert rel_l2_same < 0.10
    assert rel_l2 < 0.25


@pytest.mark.parametrize("sigma", [4.0, 1.0, 0.25, 0.05])
def test_kernel_routing_matches_topk_exactly_when_scored_in_fp32(sigma):
    """The k-pass argmax *is* top-k, at any logit spread, if scored in fp32."""
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(0, sigma, (1024, NUM_EXPERTS)),
                         jnp.bfloat16)
    _, gmm_ids = _gmm_routing(logits)
    _, ker_ids = _kernel_routing(logits)
    np.testing.assert_array_equal(np.asarray(jnp.sort(gmm_ids, axis=-1)),
                                  np.asarray(jnp.sort(ker_ids, axis=-1)))


@pytest.mark.parametrize("sigma,floor", [(1.0, 0.99), (0.25, 0.90),
                                         (0.05, 0.40)])
def test_bf16_scoring_would_reorder_near_ties(sigma, floor, capsys):
    """Why the kernel widens the router block to fp32 before scoring.

    Softmax in the token dtype collapses near-ties among 512 experts, and the
    top-10 set then disagrees with the torch router on a growing fraction of
    tokens as the logits get more concentrated.
    """
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(0, sigma, (1024, NUM_EXPERTS)),
                         jnp.bfloat16)
    _, gmm_ids = _gmm_routing(logits)
    _, bf16_ids = _kernel_routing(logits, score_dtype=jnp.bfloat16)
    agree = np.asarray(
        jnp.all(jnp.sort(bf16_ids, -1) == jnp.sort(gmm_ids, -1), -1)).mean()
    with capsys.disabled():
        print(f"[bf16 scoring, sigma={sigma}] top-k set agreement "
              f"{agree * 100:.2f}%")
    assert floor <= agree < 1.0
