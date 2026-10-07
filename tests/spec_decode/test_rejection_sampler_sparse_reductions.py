# SPDX-License-Identifier: Apache-2.0
"""The sparse verify sampler written with masked reductions (gathers,
argmaxes and scatters replaced, int32 indices) against the gather/scatter
version it replaced (copied below as the reference, from bbddf10b6):
bit-identical tokens on the same inputs, plus a positive control that a
broken placement is caught."""

import pytest
import torch
import torch._dynamo

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


# ---- reference: the implementation before the masked-reduction rewrite ----
def _ref_window_logits(
    vals: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Top-k/top-p over a row's sorted top-``width`` window.

    ``vals`` [rows, width] are temperature-scaled logits sorted descending
    (``_chunked_topk`` on the raw logits, then divided by a positive
    temperature, which keeps the order). Every row must have
    ``0 < top_k <= width``, so all of its survivors lie in the window, and
    the result equals ``apply_top_k_top_p`` restricted to the window:
    survivors keep their logit, the rest are ``-inf``. The only difference
    is a tie of the k-th value outside the window, which the full-vocab
    path keeps and this one drops.
    """
    width = vals.shape[-1]
    k_index = (top_k.clamp(min=1, max=width) - 1).to(torch.int64)
    cutoff = torch.gather(vals, -1, k_index)
    survives = vals >= cutoff
    neg_inf = torch.tensor(float("-inf"), dtype=vals.dtype, device=vals.device)
    masked = torch.where(survives, vals, neg_inf)
    probs = torch.softmax(masked, dim=-1)  # non-increasing along the row
    cum_mass = torch.cumsum(probs, dim=-1)
    reached = cum_mass >= top_p
    reached_in_window = torch.any(reached, dim=-1, keepdim=True)
    first_reached = torch.argmax(reached.to(torch.int32), dim=-1, keepdim=True)
    threshold_in_window = torch.gather(probs, -1, first_reached)
    # The survivors' mass is 1 up to rounding; if the cumsum rounds below p,
    # keep every survivor.
    threshold = torch.where(
        reached_in_window, threshold_in_window, torch.zeros_like(threshold_in_window)
    )
    keep = survives & ((probs >= threshold) | (top_p >= 1.0))
    return torch.where(keep, masked, neg_inf)


def _ref_window_sample(
    vals: torch.Tensor,
    idx: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    u: torch.Tensor,
    greedy_eps: float,
) -> torch.Tensor:
    """One token per row from its top-k window, as the runner's
    ``sample_from_logits`` draws it from the full-vocab logits: Gumbel-max
    over the temperature-scaled, top-k/top-p-masked logits, and the argmax
    for rows with ``temperature <= greedy_eps``.

    ``vals`` [rows, width] are the raw logits sorted descending, ``idx``
    their vocab ids, ``u`` [rows, width] uniform noise; parameters are
    [rows, 1]. Returns [rows] vocab ids.
    """
    is_greedy = temperatures <= greedy_eps
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled = vals.to(torch.float32) / safe_temperatures.to(torch.float32)
    window_logits = _ref_window_logits(scaled, top_k, top_p)
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    pos = torch.argmax(window_logits + gumbel_noise, dim=-1, keepdim=True)
    sampled = torch.gather(idx, -1, pos)
    # Greedy rows: the lowest id holding the row maximum, as torch.argmax
    # over the full logits picks it (the window's tie order is unspecified).
    no_id = torch.full_like(idx, torch.iinfo(idx.dtype).max)
    lowest_max = torch.amin(
        torch.where(vals == vals[..., :1], idx, no_id), dim=-1, keepdim=True
    )
    return torch.where(is_greedy, lowest_max, sampled).view(-1)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _ref_rejection_sparse(
    draft_token_ids: torch.Tensor,
    target_vals: torch.Tensor,
    target_idx: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
) -> torch.Tensor:
    """``_random_rejection_sample_with_segment`` on each draft row's top-k
    window (``target_vals`` sorted descending, ``target_idx`` their vocab
    ids) instead of its full-vocab logits.

    Valid when every row has ``0 < top_k <= width``: the target distribution
    is then zero outside the window, so p(draft) is read from the window
    (0 when the draft is not in it) and the recovery distribution
    max(0, p - q) lives in the window too. ``recover_u`` is [rows, width].
    """
    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft_token_ids = draft_token_ids >= 0
    row_req = segment_ids.to(torch.int64).clamp(min=0, max=temperatures.shape[0] - 1)
    temperatures = temperatures[row_req]
    top_k = top_k[row_req]
    top_p = top_p[row_req]

    scaled = target_vals.to(torch.float32) / (
        temperatures.to(torch.float32) + rs.TEMPERATURE_EPS
    )
    target_probs = torch.softmax(_ref_window_logits(scaled, top_k, top_p), dim=-1)
    is_draft = (target_idx.to(torch.int64) == draft_token_ids.unsqueeze(-1)) & (
        valid_draft_token_ids.unsqueeze(-1)
    )
    zero = torch.tensor(0.0, dtype=target_probs.dtype, device=target_probs.device)
    target_token_probs = torch.sum(torch.where(is_draft, target_probs, zero), dim=-1)
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        conditional_rates = synthetic_conditional_rates[group_indices.to(torch.int64)]
        accepted = accept_u < conditional_rates
    else:
        accepted = target_token_probs >= accept_u
    accepted = accepted & valid_draft_token_ids

    recovered_dist = torch.where(is_draft, zero, target_probs)
    recover_u = torch.clamp(
        recover_u,
        min=torch.finfo(target_probs.dtype).tiny,
        max=1.0 - torch.finfo(target_probs.dtype).eps,
    )
    exp_noise = -torch.log(recover_u)
    recovered_pos = torch.argmax(
        recovered_dist / (exp_noise + torch.finfo(target_probs.dtype).tiny),
        dim=-1,
        keepdim=True,
    )
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1, keepdim=True)
    recovered_pos = torch.where(
        has_recovery_mass,
        recovered_pos,
        torch.argmax(target_probs, dim=-1, keepdim=True),
    )
    recovered_token_ids = (
        torch.gather(target_idx, -1, recovered_pos).squeeze(-1).to(torch.int32)
    )
    return rs._place_rejection_tokens(
        draft_token_ids,
        accepted,
        recovered_token_ids,
        num_draft_tokens,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )


# ---------------------------------------------------------------------------
WIDTH = 64
K = 3


def _case(seed, rows, num_reqs_padded, device, ties=False, greedy=True):
    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(rows, WIDTH, generator=g) * 2.0
    if ties:
        vals = torch.round(vals * 2) / 2  # coarse values: many ties in the window
    vals = torch.sort(vals.to(torch.bfloat16).float(), dim=-1, descending=True).values
    idx = torch.stack([torch.randperm(200000, generator=g)[:WIDTH] for _ in range(rows)])
    idx = idx.to(torch.int32)
    num_reqs = min(num_reqs_padded, rows // (K + 1))
    draft = torch.full((rows,), -1, dtype=torch.int32)
    seg = torch.full((rows,), num_reqs_padded, dtype=torch.int64)
    grp = torch.zeros(rows, dtype=torch.int32)
    num_draft = torch.zeros(num_reqs_padded, dtype=torch.int32)
    row = 0
    for q in range(num_reqs):
        nd = int(torch.randint(0, K + 1, (1,), generator=g))
        num_draft[q] = nd
        for j in range(nd):
            pick = int(torch.randint(0, 3, (1,), generator=g))
            # a top window entry (likely accepted), a deep one, or out of window
            draft[row] = (
                idx[row, 0] if pick == 0 else idx[row, 30] if pick == 1 else 7
            )
            seg[row], grp[row] = q, j
            row += 1
    temps = torch.rand(num_reqs_padded, 1, generator=g) * 1.5 + 0.1
    if greedy:
        temps[0] = 0.0
    top_k = torch.tensor([1, 5, 20, 64] * 4, dtype=torch.int32)[:num_reqs_padded, None]
    top_p = torch.tensor([0.3, 0.95, 1.0, 0.8] * 4)[:num_reqs_padded, None]
    bonus = torch.randint(0, 200000, (num_reqs_padded,), generator=g, dtype=torch.int32)
    accept_u = torch.rand(rows, generator=g)
    recover_u = torch.rand(rows, WIDTH, generator=g)
    bonus_u = torch.rand(num_reqs_padded, WIDTH, generator=g)
    c = dict(vals=vals, idx=idx, draft=draft, seg=seg, grp=grp, num_draft=num_draft,
             temps=temps, top_k=top_k, top_p=top_p, bonus=bonus, accept_u=accept_u,
             recover_u=recover_u, bonus_u=bonus_u)
    return {k: v.to(device) for k, v in c.items()}


def _rejection(fn, c, rates=None):
    return fn(c["draft"], c["vals"], c["idx"], c["num_draft"], c["bonus"], c["seg"],
              c["grp"], K, c["temps"], c["top_k"], c["top_p"], c["accept_u"],
              c["recover_u"], rates, rates is not None).cpu()


@pytest.mark.parametrize("rows,reqs", [(16, 8), (48, 16), (64, 16)])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
def test_rejection_matches_the_gather_version(device, rows, reqs, ties, synthetic):
    rates = torch.tensor([0.9, 0.7, 0.5], device=device) if synthetic else None
    for seed in range(6):
        c = _case(seed * 31 + rows, rows, reqs, device, ties=ties)
        got = _rejection(rs._random_rejection_sample_sparse, c, rates)
        want = _rejection(_ref_rejection_sparse, c, rates)
        assert torch.equal(got, want), (seed, got, want)


@pytest.mark.parametrize("ties", [False, True])
def test_window_sample_matches_the_gather_version(device, ties):
    for seed in range(6):
        c = _case(seed, 16, 16, device, ties=ties)
        args = (c["vals"][:16], c["idx"][:16], c["temps"], c["top_k"], c["top_p"],
                c["bonus_u"], 1e-5)
        got = rs.sparse_window_sample(*args).cpu()
        want = _ref_window_sample(*args).cpu()
        assert torch.equal(got, want), (seed, got, want)
        got_l = rs.sparse_window_logits(c["vals"][:16], c["top_k"], c["top_p"]).cpu()
        want_l = _ref_window_logits(c["vals"][:16], c["top_k"], c["top_p"]).cpu()
        assert torch.equal(got_l, want_l)


def test_comparison_catches_a_broken_placement(device, monkeypatch):
    # Positive control: a placement that never writes the bonus token must
    # be told apart from the reference on these cases.
    real = rs._place_rejection_tokens_dense

    def no_bonus(draft, accepted, recovered, num_draft, bonus, *rest):
        return real(draft, accepted, recovered, num_draft, torch.full_like(bonus, -1), *rest)

    monkeypatch.setattr(rs, "_place_rejection_tokens_dense", no_bonus)
    torch._dynamo.reset()
    differs = False
    for seed in range(6):
        c = _case(seed * 31 + 48, 48, 16, device)
        got = _rejection(rs._random_rejection_sample_sparse, c)
        want = _rejection(_ref_rejection_sparse, c)
        differs |= not torch.equal(got, want)
    torch._dynamo.reset()
    assert differs
