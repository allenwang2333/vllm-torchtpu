# SPDX-License-Identifier: Apache-2.0
"""Fused per-step device programs for the async spec-decode hot path.

The runner is host-bound at small batch: every eager aten op costs tens of
microseconds of Python/dispatch time and every compiled-region call ~100us,
largely independent of the work it does. The functions here fold chains of
small ops that used to be issued one by one into single fixed-shape compiled
programs. Each body is a plain function (so it runs eagerly on CPU for unit
tests) and the compiled variant is derived from it; the dispatchers pick the
compiled one for TPU tensors, mirroring the read-offset helpers in
tpu_runner.py.

All shapes are bucketed by the callers, so none of these recompile as the
real request count changes.
"""
import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _random_rejection_sample_with_segment)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

INVALID_TOKEN_ID = -1
# Must match tpu_runner.SAMPLING_EPS.
SAMPLING_EPS = 1e-5


def _orig(fn):
    """The undecorated body of a torch.compile'd function, so it inlines
    into the enclosing fused graph instead of being a nested region."""
    return getattr(fn, "_torchdynamo_orig_callable",
                   getattr(fn, "__wrapped__", fn))


_random_rejection_body = _orig(_random_rejection_sample_with_segment)


def async_spec_input_fixup_impl(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    idx: torch.Tensor,
    next_tokens: torch.Tensor,
    num_rejected: torch.Tensor,
    final_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Previous-step async corrections for one verify chunk, in one program.

    Replaces, for the async spec path, the sequence
    ``subtract_num_rejected_tokens`` -> ``_substitute_placeholder_token`` ->
    ``extract_draft_token_ids`` (plus their three index uploads).

    ``idx`` packs three int32 index vectors, each ``-1`` where the slot has no
    source in the previous step:
      * ``[0, S)``           seq_lens slot  -> index into ``num_rejected``
      * ``[S, S+T)``         position slot  -> index into ``num_rejected``
      * ``[S+T, S+2T)``      input_ids slot -> index into ``next_tokens``
    with ``S = len(seq_lens)`` and ``T = len(input_ids) == len(positions)``.

    Returns ``(input_ids, positions, seq_lens, draft_token_ids)``, all new
    tensors (no input is mutated).
    """
    n_seq = seq_lens.shape[0]
    n_tok = input_ids.shape[0]
    seq_idx = idx[:n_seq]
    pos_idx = idx[n_seq:n_seq + n_tok]
    src_idx = idx[n_seq + n_tok:n_seq + 2 * n_tok]
    zero = num_rejected.new_zeros(())
    seq_lens = seq_lens - torch.where(
        seq_idx >= 0, num_rejected[seq_idx.clamp(min=0)], zero)
    positions = positions - torch.where(
        pos_idx >= 0, num_rejected[pos_idx.clamp(min=0)], zero)
    new_ids = next_tokens[src_idx.clamp(min=0)].to(input_ids.dtype)
    input_ids = torch.where(src_idx >= 0, new_ids, input_ids)
    draft_token_ids = input_ids[final_logits_indices][target_logits_indices +
                                                      1]
    return input_ids, positions, seq_lens, draft_token_ids


def spec_verify_sample_impl(
    logits: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    params: torch.Tensor,
    u: torch.Tensor,
    max_draft_tokens: int,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
    read_offsets: torch.Tensor | None,
    mamba_indices: torch.Tensor | None,
    mamba_row_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Non-greedy spec verify for one chunk, in one program.

    Replaces the bonus/target gather, the bonus-token sample, the three RNG
    draws, the per-token sampling-param gathers, the rejection sampler and
    the mamba read-offset scatter that ``_sample_spec_verify_chunk`` +
    ``_update_mamba_slot_read_offsets`` used to issue separately.

    Args:
        params: ``[3, P]`` float32 per-request (temperature, top_k, top_p),
            neutral-padded to the bucketed request count ``P``.
        u: flat float32 uniforms, ``P*V + T + T*V`` long: bonus-sample
            noise, accept draws, recovery noise (``T`` = padded logits rows).
        read_offsets: mamba read-offset buffer, updated in place; ``None``
            skips the scatter.
        mamba_indices: ``[W]`` (one group) or ``[G, W]`` slot ids per batch
            position; padded positions hold the null slot.
        mamba_row_mask: ``[W]`` bool, True for the chunk's real requests.

    Returns the rejection output ``[P, max_draft_tokens + 1]``.
    """
    bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
    target_logits = torch.index_select(logits, 0, target_logits_indices)
    num_bonus, vocab = bonus_logits.shape
    num_target = target_logits.shape[0]

    temperatures = params[0].to(logits.dtype).unsqueeze(1)
    top_k = params[1].to(torch.int32).unsqueeze(1)
    top_p = params[2].unsqueeze(1)

    n0 = num_bonus * vocab
    bonus_u = u[:n0].view(num_bonus, vocab).to(bonus_logits.dtype)
    accept_u = u[n0:n0 + num_target]
    recover_u = u[n0 + num_target:].view(num_target, vocab)

    # Bonus token: same math as TPUModelRunner.sample_from_logits.
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled = bonus_logits / safe_temperatures
    masked = apply_top_k_top_p(scaled, top_k, top_p)
    u_clamped = torch.clamp(bonus_u,
                            min=torch.finfo(bonus_u.dtype).tiny,
                            max=1.0 - torch.finfo(bonus_u.dtype).eps)
    noisy = masked + (-torch.log(-torch.log(u_clamped)))
    final = torch.where(is_greedy, bonus_logits, noisy)
    bonus_token_ids = torch.argmax(final, dim=-1)

    next_tokens = _random_rejection_body(
        draft_token_ids,
        target_logits,
        draft_lengths,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        temperatures[segment_ids],
        top_k[segment_ids],
        top_p[segment_ids],
        accept_u,
        recover_u,
        synthetic_conditional_rates,
        synthetic_mode,
    )

    if read_offsets is not None:
        width = mamba_indices.shape[-1]
        num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1).to(
            torch.int32)
        m = min(num_valid.shape[0], width)
        off = torch.where(mamba_row_mask[:m], (num_valid[:m] - 1).clamp(min=0),
                          0).to(torch.int32)
        if m < width:
            off = torch.cat([off, off.new_zeros(width - m)])
        groups = mamba_indices.numel() // width
        if groups > 1:
            off = off.repeat(groups)
        read_offsets.index_put_((mamba_indices.reshape(-1).long(), ),
                                off.to(read_offsets.dtype))
    return next_tokens


def assemble_async_spec_source_impl(
    next_tokens: torch.Tensor,
    drafts: torch.Tensor,
    draft_lengths: torch.Tensor,
    source_len: int,
    rejected_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Async substitution source + rejected count for one verify chunk.

    Fixed-shape twin of ``assemble_spec_next_tokens`` +
    ``compute_num_rejected`` + the source/rejected padding: operates on the
    bucketed rows (``min(len(next_tokens), len(drafts), len(draft_lengths))``
    of them) instead of slicing to the real request count, so it compiles
    once per bucket. Rows past the real request count carry garbage, which
    is never read: the next step's index builder only addresses real
    requests' positions.

    Returns ``(source[source_len], num_rejected[rejected_len])``.
    """
    rows = min(next_tokens.shape[0], drafts.shape[0], draft_lengths.shape[0])
    nt = next_tokens[:rows]
    num_valid = (nt != INVALID_TOKEN_ID).sum(dim=1)
    last_valid_col = num_valid.clamp(min=1) - 1
    bonus = nt.gather(1, last_valid_col.unsqueeze(1))
    src = torch.cat([bonus.to(drafts.dtype), drafts[:rows]], dim=1).reshape(-1)
    num_rejected = (draft_lengths[:rows] - num_valid + 1).clamp(min=0).to(
        torch.int32)
    src = torch.nn.functional.pad(src, (0, source_len - src.shape[0]))
    num_rejected = torch.nn.functional.pad(
        num_rejected, (0, rejected_len - num_rejected.shape[0]))
    return src, num_rejected


def draft_seed_prep_impl(
    input_ids: torch.Tensor,
    next_tokens: torch.Tensor,
    draft_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    row_idx: torch.Tensor,
    rejected_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Async first-pass draft inputs for one chunk, in one program.

    Replaces the async branch of ``Eagle3Proposer._prepare_draft_inputs``
    (seed token / rejected count / seed position from the on-device
    rejection output) plus the compiled ``_draft_seed_input_ids`` scatter and
    the gather-index / rejected-count padding, at fixed bucket shapes.

    Args:
        next_tokens: ``[>=n, K+1]`` rejection output, ``-1`` past each
            request's accepted prefix + bonus.
        draft_lengths: ``[>=n]`` per-request verified draft count.
        query_start_loc: ``[>=n+1]`` chunk query start offsets.
        row_idx: ``[p]`` request row per loop-bucket slot; slots past the
            real count ``n`` repeat row ``n-1`` so their seed scatter
            rewrites that request's own slot with the same value.
        rejected_idx: ``[R]`` request row per rejected-count slot, ``-1``
            for the zero-filled tail.

    Returns ``(draft_input_ids [T], gather_indices [p] int64,
    num_rejected [R] int32)``.
    """
    nt = next_tokens[row_idx]
    num_valid = (nt != INVALID_TOKEN_ID).sum(dim=1).clamp(min=1)
    next_token_ids = nt.gather(1, (num_valid - 1).unsqueeze(1)).squeeze(1)
    num_draft = draft_lengths[row_idx].to(torch.int64)
    num_rejected = num_draft - torch.minimum(num_valid - 1, num_draft)
    last_token_indices = (query_start_loc[row_idx + 1].to(torch.int64) - 1 -
                          num_rejected)
    shifted = input_ids
    if input_ids.shape[0] > 1:
        shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
    draft_input_ids = shifted.index_put((last_token_indices, ),
                                        next_token_ids.to(input_ids.dtype))
    rejected_out = torch.where(rejected_idx >= 0,
                               num_rejected[rejected_idx.clamp(min=0)],
                               0).to(torch.int32)
    return draft_input_ids, last_token_indices, rejected_out


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_async_spec_input_fixup_compiled = _tpu_compile(async_spec_input_fixup_impl)
_spec_verify_sample_compiled = _tpu_compile(spec_verify_sample_impl)
_assemble_async_spec_source_compiled = _tpu_compile(
    assemble_async_spec_source_impl)
_draft_seed_prep_compiled = _tpu_compile(draft_seed_prep_impl)


def async_spec_input_fixup(*args):
    fn = (_async_spec_input_fixup_compiled if args[0].device.type == "tpu"
          else async_spec_input_fixup_impl)
    return fn(*args)


def spec_verify_sample(*args):
    fn = (_spec_verify_sample_compiled
          if args[0].device.type == "tpu" else spec_verify_sample_impl)
    return fn(*args)


def assemble_async_spec_source(*args):
    fn = (_assemble_async_spec_source_compiled
          if args[0].device.type == "tpu" else assemble_async_spec_source_impl)
    return fn(*args)


def draft_seed_prep(*args):
    fn = (_draft_seed_prep_compiled
          if args[0].device.type == "tpu" else draft_seed_prep_impl)
    return fn(*args)
