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
"""Does the fused EP router pick the same experts, and give them the same
weights, as the router the model actually runs?

`fused_moe.v2.layer.select_experts` is a SECOND implementation of the classic
top-k branch of `moe_routing.select_experts` -- in JAX, above the Mosaic
kernel, on a Pallas selector rather than `torch.topk`. Two implementations of
one rule diverge silently: the kernel emits a fully-formed answer for the
wrong experts, and every band downstream of it is wide enough to hide that.

So this file compares against the REAL reference -- `moe_routing.select_experts`
imported and called, not re-derived here -- and asserts:

  * the selected expert ids are IDENTICAL, exactly, no tolerance; and
  * the combine weights agree to `WEIGHT_TOL` relative.

BOTH assertions are load-bearing and neither implies the other. At zero
correction bias the indices agree even for the OLD softmax router, because
softmax and sigmoid are both strictly increasing in the logit and preserve the
argsort: an index-only test passes the broken router at sigma=0.
`test_the_shipped_softmax_router_would_have_failed_this` fixes that in place,
so a future edit cannot quietly weaken the file back to the test that proved
nothing.

Two tokens are excluded from the index assertion and counted in the failure
message: those whose 8th- and 9th-largest selection keys are within
`BOUNDARY_ULPS` float32 ulps of each other. There the boundary is decided by
the last bit of a transcendental and by the tie-break convention -- torch's
`topk` against `pallas_select`'s lowest-matching-column rule
(`router_ops.py`) -- rather than by the routing rule.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------- the shapes
# GLM-5.2-FP8: n_routed_experts 256, num_experts_per_tok 8, scoring_func
# sigmoid, topk_method noaux_tc, norm_topk_prob true, routed_scaling_factor
# 2.5, n_group = topk_group = 1 (so the GROUPED branch is not taken).
NUM_EXPERTS = 256
TOPK = 8
RENORMALIZE = True
RSF = 2.5
TOKENS = 512  # a whole number of MAX_ROUTING_BLOCK=256 rows
BLOCK_ROWS = 256

WEIGHT_TOL = 1e-5  # relative, fixed before any result was seen
BOUNDARY_ULPS = 8

# The correction-bias scales the arm must hold at. sigma=0 is the control that
# an index-only test cannot fail; sigma=0.5 is far past any real checkpoint.
SIGMAS = (0.0, 0.01, 0.05, 0.5)

SEED = 20260909


def _rng():
    return np.random.default_rng(SEED)


# ------------------------------------------------------------- the reference
def _reference(logits, bias, *, scoring_fn="sigmoid", rsf=RSF):
    """`moe_routing.select_experts`, the router GLM-5.2 is served by today."""
    import torch
    from types import SimpleNamespace

    from vllm_torchtpu.layers.vllm import moe_routing

    layer = SimpleNamespace(
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        hash_indices_table=None,
        e_score_correction_bias=(None if bias is None else torch.from_numpy(
            np.asarray(bias, np.float32))),
        routed_scaling_factor=rsf,
    )
    g = torch.from_numpy(np.asarray(logits, np.float32))
    h = torch.zeros(g.shape[0], 1)  # only its dtype and .size(0) are read
    w, ids = moe_routing.select_experts(h,
                                        g,
                                        topk=TOPK,
                                        renormalize=RENORMALIZE,
                                        scoring_fn=scoring_fn,
                                        layer=layer)
    return ids.numpy().astype(np.int64), w.float().numpy().astype(np.float64)


# ----------------------------------------------------------- the fused path
def _fused(logits, bias, *, scoring_fn="sigmoid", rsf=RSF):
    """THE FUNCTION UNDER TEST -- imported from the layer, not re-derived."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v2.layer import select_experts

    # The bias arrives as the [1, E] per-shard operand the bridge builds.
    router_bias = (None if bias is None else jnp.asarray(
        np.asarray(bias, np.float32)).reshape(1, -1))
    w, idx = select_experts(jnp.asarray(np.asarray(logits, np.float32)),
                            topk=TOPK,
                            block_rows=BLOCK_ROWS,
                            renormalize=RENORMALIZE,
                            scoring_fn=scoring_fn,
                            router_bias=router_bias,
                            routed_scaling_factor=rsf)
    return np.asarray(idx).astype(np.int64), np.asarray(w).astype(np.float64)


# --------------------------------------------------------------- comparison
def _selection_key(logits, bias, scoring_fn):
    """The key each path sorts on, in float32, formed the same way both do."""
    x = np.asarray(logits, np.float32)
    if scoring_fn == "sigmoid":
        s = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    else:
        e = np.exp(x - x.max(-1, keepdims=True))
        s = (e / e.sum(-1, keepdims=True)).astype(np.float32)
    if bias is not None:
        s = (s + np.asarray(bias, np.float32)[None, :]).astype(np.float32)
    return s


def _strict_boundary(key, ulps=BOUNDARY_ULPS):
    """Tokens whose top-k boundary is resolved by more than `ulps` ulps."""
    part = np.sort(np.asarray(key, np.float32), axis=-1)[:, ::-1]
    hi, lo = part[:, TOPK - 1], part[:, TOPK]
    return (hi - lo) > ulps * np.spacing(np.abs(hi).astype(np.float32))


def _index_mismatch(ref_ids, got_ids, mask):
    """Tokens under `mask` whose selected expert SET differs. Order is not
    compared: `pallas_select` returns slots in descending key order and so
    does `torch.topk`, but the SET is what the kernel's routing tables read
    and the weights are compared per expert id rather than per slot."""
    bad = []
    for t in np.nonzero(np.asarray(mask, bool))[0]:
        if set(ref_ids[t].tolist()) != set(got_ids[t].tolist()):
            bad.append(int(t))
    return bad


def _weight_error(ref_ids, ref_w, got_ids, got_w, mask):
    """Worst relative weight error, aligned by EXPERT ID, over tokens whose
    sets agree. Aligned by id and not by slot so that a mere reordering of two
    equal-key slots is not reported as a weight error."""
    worst, worst_abs, n = 0.0, 0.0, 0
    for t in np.nonzero(np.asarray(mask, bool))[0]:
        r = dict(zip(ref_ids[t].tolist(), ref_w[t].tolist()))
        g = dict(zip(got_ids[t].tolist(), got_w[t].tolist()))
        if set(r) != set(g):
            continue
        n += 1
        for e, rv in r.items():
            d = abs(rv - g[e])
            worst_abs = max(worst_abs, d)
            worst = max(worst, d / max(abs(rv), 1e-12))
    return worst, worst_abs, n


def _malformed(ids):
    """Tokens that did not come back as TOPK distinct real expert ids."""
    return [
        int(t) for t, row in enumerate(ids)
        if len(set(row.tolist())) != TOPK or row.min() < 0
        or row.max() >= NUM_EXPERTS
    ]


# ------------------------------------------------------------- input classes
def _logits(rng, scale=1.0, tokens=TOKENS):
    return rng.normal(0.0, scale, (tokens, NUM_EXPERTS)).astype(np.float32)


def _near_tie_logits(rng, gap=1e-4, tokens=TOKENS):
    """Every token's 8th/9th boundary is nearly degenerate: the top 24 experts
    sit inside `gap`, so any reordering at all flips the chosen set."""
    base = rng.normal(0.0, 1.0, (tokens, NUM_EXPERTS)).astype(np.float32)
    for t in range(tokens):
        hot = rng.choice(NUM_EXPERTS, size=24, replace=False)
        base[t, hot] = 2.0 + rng.normal(0, 0.1) + rng.uniform(
            -gap, gap, size=24)
    return base


# =============================================================== the tests
@pytest.mark.parametrize("sigma", SIGMAS)
def test_indices_and_weights_match_the_reference(sigma):
    """The headline: exact index equality AND tight weight agreement, at each
    correction-bias scale the arm has to hold at."""
    rng = _rng()
    logits = _logits(rng)
    bias = (np.zeros(NUM_EXPERTS, np.float32)
            if sigma == 0.0 else rng.normal(0.0, sigma, NUM_EXPERTS).astype(
                np.float32))

    ref_ids, ref_w = _reference(logits, bias)
    got_ids, got_w = _fused(logits, bias)

    assert not _malformed(got_ids), (
        f"the fused router returned malformed selections at sigma={sigma}: "
        f"{_malformed(got_ids)[:8]}")

    key = _selection_key(logits, bias, "sigmoid")
    strict = _strict_boundary(key)
    degenerate = int((~strict).sum())

    bad = _index_mismatch(ref_ids, got_ids, strict)
    assert not bad, (
        f"sigma={sigma}: the fused router chose a DIFFERENT set of {TOPK} "
        f"experts on {len(bad)}/{int(strict.sum())} tokens whose top-k "
        f"boundary is strict at {BOUNDARY_ULPS} ulps "
        f"({degenerate} tokens excluded as f32-degenerate). "
        f"First: token {bad[0]} ref={sorted(ref_ids[bad[0]].tolist())} "
        f"got={sorted(got_ids[bad[0]].tolist())}")

    rel, abs_, n = _weight_error(ref_ids, ref_w, got_ids, got_w, strict)
    assert n > 0, "no comparable tokens -- the check did not actually run"
    assert rel <= WEIGHT_TOL, (
        f"sigma={sigma}: combine weights differ by {rel:.3e} relative "
        f"({abs_:.3e} absolute) over {n} tokens, past the {WEIGHT_TOL:.0e} "
        f"tolerance. renormalize={RENORMALIZE} routed_scaling_factor={RSF}")


def test_the_shipped_softmax_router_would_have_failed_this():
    """The guard on the test itself.

    At sigma=0 the INDICES agree even for the router as shipped: softmax and
    sigmoid are both strictly increasing in the logit, so they induce the same
    argsort and the same top-k set. An index-only check therefore passes the
    broken kernel at sigma=0 and proves nothing. The WEIGHTS are what separate
    them, and this asserts that they do -- at sigma=0, the friendliest case.
    """
    rng = _rng()
    logits = _logits(rng)
    zero = np.zeros(NUM_EXPERTS, np.float32)

    ref_ids, ref_w = _reference(logits, zero)
    # The router exactly as db16bef1 shipped it: softmax, no bias, no scaling.
    old_ids, old_w = _fused(logits, None, scoring_fn="softmax", rsf=1.0)

    strict = _strict_boundary(_selection_key(logits, zero, "sigmoid"))
    assert not _index_mismatch(ref_ids, old_ids, strict), (
        "the shipped softmax router is expected to AGREE on indices at zero "
        "bias -- if it does not, this file's premise is wrong")
    rel, _, n = _weight_error(ref_ids, ref_w, old_ids, old_w, strict)
    assert n > 0
    assert rel > WEIGHT_TOL, (
        f"the shipped softmax router's weights came within {rel:.3e} of the "
        f"reference's at zero bias, so the weight assertion above is not "
        f"discriminating and this whole file is decoration")


def test_a_correction_bias_really_does_move_the_selection():
    """The other guard: that the bias is doing something at these sigmas.

    If the shipped router had agreed on indices at sigma=0.05 too, then the
    index assertion above would be passing for free.
    """
    rng = _rng()
    logits = _logits(rng)
    bias = rng.normal(0.0, 0.05, NUM_EXPERTS).astype(np.float32)

    ref_ids, _ = _reference(logits, bias)
    old_ids, _ = _fused(logits, None, scoring_fn="softmax", rsf=1.0)
    strict = _strict_boundary(_selection_key(logits, bias, "sigmoid"))
    moved = _index_mismatch(ref_ids, old_ids, strict)
    assert len(moved) > 0.1 * int(strict.sum()), (
        f"only {len(moved)}/{int(strict.sum())} tokens are routed differently "
        f"by a sigma=0.05 correction bias; the index assertion is not "
        f"discriminating at that scale")


@pytest.mark.parametrize("sigma", [0.01, 0.05])
def test_near_ties_do_not_break_it(sigma):
    """The hostile input: 24 experts per token inside 1e-4 of each other, so
    the top-8 boundary is decided in the last bits."""
    rng = _rng()
    logits = _near_tie_logits(rng)
    bias = rng.normal(0.0, sigma, NUM_EXPERTS).astype(np.float32)
    ref_ids, ref_w = _reference(logits, bias)
    got_ids, got_w = _fused(logits, bias)
    strict = _strict_boundary(_selection_key(logits, bias, "sigmoid"))
    if not strict.any():
        pytest.skip("every token's boundary is f32-degenerate here")
    bad = _index_mismatch(ref_ids, got_ids, strict)
    assert not bad, (f"near-tie sigma={sigma}: {len(bad)} of "
                     f"{int(strict.sum())} strictly-resolved tokens diverged")
    rel, _, n = _weight_error(ref_ids, ref_w, got_ids, got_w, strict)
    assert rel <= WEIGHT_TOL, f"near-tie weight error {rel:.3e}"


def test_the_weight_operand_returns_the_value_not_the_key():
    """`pallas_select(values=...)` is the mechanism that keeps the correction
    bias out of the weights without a second [tokens, experts] gather in XLA.
    It has to return the VALUES array's entry at the selected column, bit for
    bit -- not the key's, and not something a max sentinel floored."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select

    rng = _rng()
    key = rng.normal(0.0, 1.0, (BLOCK_ROWS, NUM_EXPERTS)).astype(np.float32)
    values = rng.normal(0.0, 1.0e30,
                        (BLOCK_ROWS, NUM_EXPERTS)).astype(np.float32)
    w, idx = pallas_select(jnp.asarray(key),
                           topk=TOPK,
                           block_rows=BLOCK_ROWS,
                           values=jnp.asarray(values))
    idx = np.asarray(idx)
    want = np.take_along_axis(values, idx, axis=-1)
    np.testing.assert_array_equal(np.asarray(w), want)

    # And the selection itself is unchanged by supplying values at all.
    _, idx_plain = pallas_select(jnp.asarray(key),
                                 topk=TOPK,
                                 block_rows=BLOCK_ROWS)
    np.testing.assert_array_equal(idx, np.asarray(idx_plain))


if __name__ == "__main__":  # a plain script, for a cdk pod without pytest args
    import sys
    import traceback

    cases = [(f"indices_and_weights sigma={s}",
              lambda s=s: test_indices_and_weights_match_the_reference(s))
             for s in SIGMAS]
    cases += [
        ("shipped_softmax_router_would_have_failed",
         test_the_shipped_softmax_router_would_have_failed_this),
        ("correction_bias_moves_the_selection",
         test_a_correction_bias_really_does_move_the_selection),
        ("near_ties sigma=0.01", lambda: test_near_ties_do_not_break_it(0.01)),
        ("near_ties sigma=0.05", lambda: test_near_ties_do_not_break_it(0.05)),
        ("weight_operand_returns_the_value",
         test_the_weight_operand_returns_the_value_not_the_key),
    ]
    failed = []
    for label, fn in cases:
        try:
            fn()
        except Exception:  # noqa: BLE001 - a script, reporting every case
            failed.append(label)
            print(f"[FAIL] {label}")
            traceback.print_exc()
        else:
            print(f"[PASS] {label}")
    print(f"\n{len(cases) - len(failed)}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
