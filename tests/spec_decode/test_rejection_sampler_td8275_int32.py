# SPDX-License-Identifier: Apache-2.0
"""TD8.275: rejection sampler int32 index paths are bitwise-equal to the int64 ones.

``_get_segment_info`` / ``_valid_slot_mask`` / the greedy and random segment
samplers now keep segment ids, group indices and accepted counts in int32 and
gather with 1D ``index_select`` (random-path draft ids intentionally stay int64).
The ``_old_*`` functions below are the verbatim pre-TD8.275 bodies (minus
``torch.compile``) used as the reference; the new functions run eagerly through
``__wrapped__``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    TEMPERATURE_EPS,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p


def _old_get_segment_info(num_draft_tokens: torch.Tensor, total_tokens: int):
    """Helper to create segment IDs and per-segment indices."""
    batch_size = num_draft_tokens.shape[0]
    device = num_draft_tokens.device

    # segment_ids assigns a unique ID to each token.
    # In Torch, we can use repeat_interleave.
    segment_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_draft_tokens,
    )

    # group_indices creates a within-segment index for each token.
    # E.g., [0, 1, 2, 0, 1, 0, 1, 2, 3] for sequences [3, 2, 4].
    # We can use cumsum and subtraction.
    cum_draft_tokens = torch.cumsum(num_draft_tokens, dim=0)
    segment_starts = torch.cat(
        [
            torch.zeros(1, dtype=num_draft_tokens.dtype, device=device),
            cum_draft_tokens[:-1],
        ]
    )
    broadcast_starts = torch.repeat_interleave(segment_starts, num_draft_tokens)
    group_indices = (torch.arange(total_tokens, device=device) - broadcast_starts).to(
        torch.int32
    )
    return segment_ids, group_indices


def _old_valid_slot_mask(
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """``[T]`` mask of declared draft slots: ``group_index < num_draft[segment]``.

    Equals ``any(segment_ids[None] == arange(B)[:, None] & (group_indices <
    num_draft_tokens[:, None]), 0)`` without the ``[B, T]`` intermediate.
    Padding segment sentinels (outside ``[0, B)``) are never valid.
    """
    batch_size = num_draft_tokens.shape[0]
    if batch_size == 0:
        return torch.zeros_like(segment_ids, dtype=torch.bool)
    in_batch = (segment_ids >= 0) & (segment_ids < batch_size)
    safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
    limits = num_draft_tokens.to(group_indices.dtype)[safe_segment_ids]
    return in_batch & (group_indices < limits)


def _old_place_bonus_tokens(
    selected_tokens: torch.Tensor,
    accepted_count: torch.Tensor,
    should_get_bonus: torch.Tensor,
    bonus_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Write ``bonus_token_ids[b]`` at column ``accepted_count[b]`` of the rows
    in ``should_get_bonus``; other rows are unchanged.

    One ``[B, K + 1]`` select instead of a gather plus a 2D scatter.
    """
    columns = torch.arange(
        selected_tokens.shape[1],
        dtype=accepted_count.dtype,
        device=accepted_count.device,
    )
    place = should_get_bonus.unsqueeze(1) & (
        columns.unsqueeze(0) == accepted_count.unsqueeze(1)
    )
    return torch.where(
        place, bonus_token_ids.to(torch.int32).unsqueeze(1), selected_tokens
    )


def _old_greedy_rejection_sample_with_segment(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    accept_u: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
) -> torch.Tensor:
    """Vectorized greedy speculative decoding validation in Torch."""
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    # Accept either 1D pre-argmax'd target token IDs (from spec_bonus_and_target_logits)
    # or 2D [num_tokens, vocab_size] target logits.
    target_logits_argmax = (
        target_logits.to(torch.int32)
        if target_logits.ndim == 1
        else torch.argmax(target_logits, dim=-1).to(torch.int32)
    )
    draft_token_ids = draft_token_ids.to(torch.int32)

    # Find the first mismatch index per segment using matrix masking.
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    valid_slot_mask = _old_valid_slot_mask(segment_ids, group_indices, num_draft_tokens)

    # Large value for positions with no mismatches
    # Create an array where mismatched positions hold their `group_index`
    # and matched positions hold a large value.
    large_value = total_tokens

    if total_tokens > 0:
        if synthetic_mode:
            assert accept_u is not None
            assert synthetic_conditional_rates is not None
            conditional_rates = synthetic_conditional_rates[
                group_indices.to(torch.int64)
            ]
            accepted = (accept_u < conditional_rates) & (draft_token_ids >= 0)
        else:
            accepted = draft_token_ids == target_logits_argmax
        mismatch_indices = torch.where(
            valid_slot_mask & ~accepted,
            group_indices,
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )

        masked_mismatch_indices = torch.where(
            match_mask,
            mismatch_indices.unsqueeze(0),
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        first_mismatch_idx_per_segment = torch.min(
            masked_mismatch_indices, dim=1
        ).values
    else:
        first_mismatch_idx_per_segment = torch.full(
            (batch_size,), large_value, dtype=torch.int32, device=device
        )

    # Step 3: Broadcast Mismatch Info and Generate Main Token Output
    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_mismatch_idx_broadcast = first_mismatch_idx_per_segment[safe_segment_ids]

        before_mismatch = valid_slot_mask & (
            group_indices < first_mismatch_idx_broadcast
        )
        at_mismatch = valid_slot_mask & (group_indices == first_mismatch_idx_broadcast)
        main_tokens = torch.where(
            before_mismatch,
            draft_token_ids,
            torch.where(
                at_mismatch,
                target_logits_argmax,
                torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device),
            ),
        )
    else:
        main_tokens = torch.tensor([], dtype=torch.int32, device=device)

    # Step 4: Handle Bonus Tokens
    all_accepted = first_mismatch_idx_per_segment == large_value
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    # Step 5: Arrange into a [batch_size, max_draft_tokens + 1] matrix
    # Route padding positions to an extra sink row for a safe scatter.
    selected_tokens = torch.full(
        (batch_size + 1, max_draft_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # Ensure all indices are uniformly int64 to prevent stablehlo.concatenate
    # type errors
    segment_ids = segment_ids.to(torch.int64)
    group_indices = group_indices.to(torch.int64)
    scatter_segment_ids = torch.where(
        valid_slot_mask,
        segment_ids,
        torch.full_like(segment_ids, batch_size),
    )
    scatter_group_indices = torch.where(
        valid_slot_mask, group_indices, torch.zeros_like(group_indices)
    )

    # Place main tokens at their exact per-request positions
    selected_tokens[scatter_segment_ids, scatter_group_indices] = main_tokens

    # Place bonus tokens immediately after the accepted draft sequence
    # ONLY if all tokens were accepted or there were no draft tokens.
    # If a mismatch occurred at k, the request's selected_tokens[k] already
    # contains target_logits_argmax[k].
    accepted_count = torch.where(
        first_mismatch_idx_per_segment == large_value,
        num_draft_tokens,
        first_mismatch_idx_per_segment,
    ).to(torch.int64)

    return _old_place_bonus_tokens(
        selected_tokens[:batch_size],
        accepted_count,
        should_get_bonus,
        bonus_token_ids,
    )


def _old_random_rejection_sample_with_segment(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
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
    """Vectorized non-greedy speculative decoding validation in Torch."""
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device
    vocab_size = target_logits.shape[-1]

    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft_token_ids = draft_token_ids >= 0
    safe_draft_token_ids = draft_token_ids.clamp(min=0)

    # Shape the target distribution with temperature + top-k/top-p, then
    # softmax. Greedy rows carry temperature == 0.0, so dividing by
    # (temperatures + TEMPERATURE_EPS) collapses them to a one-hot at argmax.
    scaled_logits = target_logits.to(torch.float32) / (
        temperatures.to(torch.float32) + TEMPERATURE_EPS
    )
    masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)
    target_probs = torch.softmax(masked_logits, dim=-1)

    # Accept test is the Leviathan ratio min(1, p(x) / q(x))
    # Our draft is greedy for now, so for q: drafted token
    # x (q(x) == 1, 0 elsewhere) and the ratio collapses to p(x).
    target_token_probs = target_probs.gather(
        -1, safe_draft_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        conditional_rates = synthetic_conditional_rates[group_indices.to(torch.int64)]
        accepted = accept_u < conditional_rates
    else:
        accepted = target_token_probs >= accept_u
    accepted = accepted & valid_draft_token_ids

    # On rejection, resample from the residual max(0, p - q). With q a delta at
    # x, subtracting q only touches entry x, so the residual is just p with the
    # drafted token zeroed out.
    # Broadcast compare instead of `one_hot(...).to(bool)`, which builds an
    # int64 [T, V] tensor before the cast.
    draft_token_mask = safe_draft_token_ids.unsqueeze(-1) == torch.arange(
        vocab_size, device=device, dtype=safe_draft_token_ids.dtype
    )
    # Placeholder drafts have no q mass to subtract, so their recovery
    # distribution remains the full target distribution. In particular, do
    # not let the safe clamp above accidentally mask vocabulary token 0.
    draft_token_mask = draft_token_mask & valid_draft_token_ids.unsqueeze(-1)
    recovered_dist = torch.where(
        draft_token_mask,
        torch.tensor(0.0, dtype=target_probs.dtype, device=device),
        target_probs,
    )
    recover_u = torch.clamp(
        recover_u,
        min=torch.finfo(target_probs.dtype).tiny,
        max=1.0 - torch.finfo(target_probs.dtype).eps,
    )
    exp_noise = -torch.log(recover_u)
    recovered_token_ids = torch.argmax(
        recovered_dist / (exp_noise + torch.finfo(target_probs.dtype).tiny),
        dim=-1,
    ).to(torch.int32)
    # A synthetic rejection can reject a greedy draft that matches the target
    # argmax. In that case p and q are the same one-hot distribution, leaving
    # no residual mass. Match upstream's greedy path by recovering the target
    # argmax instead of letting argmax(all-zero) fall through to token 0.
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1)
    recovered_token_ids = torch.where(
        has_recovery_mass,
        recovered_token_ids,
        torch.argmax(target_probs, dim=-1).to(torch.int32),
    )

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    valid_slot_mask = _old_valid_slot_mask(segment_ids, group_indices, num_draft_tokens)
    large_value = total_tokens

    if total_tokens > 0:
        rejection_indices = torch.where(
            valid_slot_mask & ~accepted,
            group_indices,
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        masked_rejection_indices = torch.where(
            match_mask,
            rejection_indices.unsqueeze(0),
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        first_rejection_idx_per_segment = torch.min(
            masked_rejection_indices, dim=1
        ).values
    else:
        first_rejection_idx_per_segment = torch.full(
            (batch_size,),
            large_value,
            dtype=torch.int32,
            device=device,
        )

    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_rejection_idx_broadcast = first_rejection_idx_per_segment[
            safe_segment_ids
        ]
        before_rejection = valid_slot_mask & (
            group_indices < first_rejection_idx_broadcast
        )
        at_rejection = valid_slot_mask & (
            group_indices == first_rejection_idx_broadcast
        )
        main_tokens = torch.where(
            before_rejection,
            draft_token_ids.to(torch.int32),
            torch.where(
                at_rejection,
                recovered_token_ids,
                torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device),
            ),
        )
    else:
        main_tokens = torch.tensor([], dtype=torch.int32, device=device)

    all_accepted = first_rejection_idx_per_segment == large_value
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    # Route padding positions to an extra sink row for a safe scatter.
    selected_tokens = torch.full(
        (batch_size + 1, max_draft_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    segment_ids = segment_ids.to(torch.int64)
    group_indices = group_indices.to(torch.int64)
    scatter_segment_ids = torch.where(
        valid_slot_mask,
        segment_ids,
        torch.full_like(segment_ids, batch_size),
    )
    scatter_group_indices = torch.where(
        valid_slot_mask, group_indices, torch.zeros_like(group_indices)
    )
    selected_tokens[scatter_segment_ids, scatter_group_indices] = main_tokens

    accepted_count = torch.where(
        all_accepted,
        num_draft_tokens,
        first_rejection_idx_per_segment,
    ).to(torch.int64)

    return _old_place_bonus_tokens(
        selected_tokens[:batch_size],
        accepted_count,
        should_get_bonus,
        bonus_token_ids,
    )


def _case(seed, idx_dtype, num_draft, pad, vocab=64):
    g = torch.Generator().manual_seed(seed)
    nd = torch.tensor(num_draft, dtype=idx_dtype)
    total = int(nd.sum()) + pad
    seg, grp = _old_get_segment_info(nd, int(nd.sum()))
    # Padding slots carry the out-of-batch sentinel (as the runner does).
    seg = torch.cat([seg, torch.full((pad,), len(num_draft), dtype=seg.dtype)])
    grp = torch.cat([grp, torch.zeros(pad, dtype=grp.dtype)])
    target = torch.randn(total, vocab, generator=g)
    draft = target.argmax(-1)
    flip = torch.rand(total, generator=g) < 0.3
    draft = torch.where(flip, torch.randint(0, vocab, (total,), generator=g), draft)
    draft[-1] = -1
    bonus = torch.randint(0, vocab, (len(num_draft),), generator=g)
    return g, nd, seg.to(idx_dtype), grp, target, draft.to(idx_dtype), bonus, total


CASES = [([3, 0, 4, 2], 3), ([1, 1, 1], 0), ([5], 2), ([0, 0], 1)]


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("num_draft", "pad"), CASES)
def test_segment_info_and_valid_mask_match(idx_dtype, num_draft, pad):
    nd = torch.tensor(num_draft, dtype=idx_dtype)
    total = int(nd.sum())
    seg, grp = rs._get_segment_info(nd, total)
    ref_seg, ref_grp = _old_get_segment_info(nd, total)
    assert seg.dtype == torch.int32 and grp.dtype == torch.int32
    assert torch.equal(seg.long(), ref_seg.long()) and torch.equal(grp, ref_grp)
    seg = torch.cat([seg, torch.full((pad,), len(num_draft), dtype=torch.int32)])
    seg = torch.cat([seg, torch.tensor([-1], dtype=torch.int32)])
    grp = torch.cat([grp, torch.zeros(pad + 1, dtype=grp.dtype)])
    assert torch.equal(
        rs._valid_slot_mask(seg, grp, nd), _old_valid_slot_mask(seg, grp, nd)
    )


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("target_1d", [False, True])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("num_draft", "pad"), CASES)
def test_greedy_matches_int64_reference(
    synthetic, target_1d, idx_dtype, num_draft, pad
):
    g, nd, seg, grp, target, draft, bonus, total = _case(1, idx_dtype, num_draft, pad)
    if target_1d:
        target = target.argmax(-1)
    k = max(num_draft) if max(num_draft) else 1
    accept_u = torch.rand(total, generator=g) if synthetic else None
    rates = torch.rand(k + 1, generator=g) if synthetic else None
    args = (draft, target, nd, bonus, seg, grp, k, accept_u, rates, synthetic)
    out = rs._greedy_rejection_sample_with_segment.__wrapped__(*args)
    ref = _old_greedy_rejection_sample_with_segment(*args)
    assert out.dtype == ref.dtype == torch.int32
    assert torch.equal(out, ref)


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("num_draft", "pad"), CASES)
def test_random_matches_int64_reference(synthetic, idx_dtype, num_draft, pad):
    g, nd, seg, grp, target, draft, bonus, total = _case(2, idx_dtype, num_draft, pad)
    k = max(num_draft) if max(num_draft) else 1
    temps = torch.rand(total, 1, generator=g)
    temps[0] = 0.0
    top_k = torch.full((total, 1), 8, dtype=torch.int32)
    top_p = torch.full((total, 1), 0.9)
    accept_u = torch.rand(total, generator=g)
    recover_u = torch.rand(total, target.shape[-1], generator=g)
    rates = torch.rand(k + 1, generator=g) if synthetic else None
    args = (
        draft,
        target,
        nd,
        bonus,
        seg,
        grp,
        k,
        temps,
        top_k,
        top_p,
        accept_u,
        recover_u,
        rates,
        synthetic,
    )
    out = rs._random_rejection_sample_with_segment.__wrapped__(*args)
    ref = _old_random_rejection_sample_with_segment(*args)
    assert out.dtype == ref.dtype == torch.int32
    assert torch.equal(out, ref)


def test_td8295_random_recovery_avoids_extra_vocab_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD8.295: random recovery uses 1 argmax and 0 torch.any [T, V] reductions."""
    vocab = 16
    # 5 rows covering:
    # 0: synthetic reject of greedy draft==0 (no recovery mass, draft==0)
    # 1: synthetic reject of greedy draft==7 (no recovery mass, draft>0)
    # 2: reject where only token 0 has residual mass (recovered_token_ids==0, mass>0)
    # 3: reject where token 5 has residual mass (recovered_token_ids==5, mass>0)
    # 4: placeholder draft (-1) recovering target argmax
    draft = torch.tensor([0, 7, 3, 2, -1], dtype=torch.int32)
    nd = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32)
    seg, grp = rs._get_segment_info(nd, 5)
    bonus = torch.tensor([9, 10, 11, 12, 13], dtype=torch.int32)
    target = torch.full((5, vocab), -100.0, dtype=torch.float32)
    target[0, 0] = 100.0
    target[1, 7] = 100.0
    target[2, 3] = 0.0
    target[2, 0] = 10.0
    target[3, 2] = 0.0
    target[3, 5] = 10.0
    target[4, 6] = 100.0

    temps = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    top_k = torch.zeros((5, 1), dtype=torch.int32)
    top_p = torch.ones((5, 1), dtype=torch.float32)
    accept_u = torch.ones(5, dtype=torch.float32)
    recover_u = torch.full((5, vocab), 0.5, dtype=torch.float32)
    rates = torch.zeros(2, dtype=torch.float32)

    argmax_calls = 0
    any_calls = 0
    orig_argmax = torch.argmax
    orig_any = torch.any

    def _counting_argmax(*a: object, **kw: object) -> torch.Tensor:
        nonlocal argmax_calls
        argmax_calls += 1
        return orig_argmax(*a, **kw)  # type: ignore[arg-type]

    def _counting_any(*a: object, **kw: object) -> torch.Tensor:
        nonlocal any_calls
        any_calls += 1
        return orig_any(*a, **kw)  # type: ignore[arg-type]

    args = (
        draft,
        target,
        nd,
        bonus,
        seg,
        grp,
        1,
        temps,
        top_k,
        top_p,
        accept_u,
        recover_u,
        rates,
        True,
    )
    ref = _old_random_rejection_sample_with_segment(*args)

    monkeypatch.setattr(torch, "argmax", _counting_argmax)
    monkeypatch.setattr(torch, "any", _counting_any)
    out = rs._random_rejection_sample_with_segment.__wrapped__(*args)

    assert argmax_calls == 1
    assert any_calls == 0
    assert torch.equal(out, ref)
    assert out[:, 0].tolist() == [0, 7, 0, 5, 6]
