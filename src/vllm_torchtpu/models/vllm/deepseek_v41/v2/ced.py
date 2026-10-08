"""Host planning for decoder SWA bounded replay across scheduled chunks."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int
    is_identity: bool = False


def plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    """Keep the same prompt-tail window across every chunk and later decode.

    The encoder publishes global KV for the complete chunk before this plan
    narrows the decoder batch. Prefix caching must be disabled: a cached prefix
    may not contain the encoder outputs required to rebuild decoder SWA.
    """
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None  # Profiling/warmup executes both stages at the full bucket.
    if window < 1 or len(bounds) != len(lengths) + 1 or len(prompts) != len(lengths):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds[0] != 0 or bounds[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    indices, replay_starts, output_bounds = [], [], [0]
    for start, end, length, prompt in zip(bounds, bounds[1:], lengths, prompts):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = max(0, prompt - window)
        skip = min(count, max(0, replay_start - (length - count)))
        indices.extend(range(start + skip, end))
        replay_starts.append(replay_start)
        output_bounds.append(len(indices))
    kept = len(indices)
    # Each stage dispatches by its own token bucket. Never introduce a second,
    # independent symbolic token dimension into vLLM's guard-free full graph.
    buckets = sorted({num_tokens, *(int(n) for n in compile_sizes if int(n) > 0)})
    padded = next(n for n in buckets if n >= kept) if kept else 0
    is_identity = kept > 0 and kept == bounds[-1] and padded == num_tokens
    return DecoderReplayPlan(
        tuple(indices),
        tuple(output_bounds),
        tuple(replay_starts),
        padded,
        is_identity=is_identity,
    )


def sync_decoder_replay_plan(
    plan: DecoderReplayPlan | None,
    metadata,
    num_tokens: int,
    dp_cpu_group=None,
) -> DecoderReplayPlan | None:
    """Synchronize decoder stage execution and padded token bucket across DP ranks.

    Under DP+EP, decoder layers execute cross-DP MoE collectives, so every DP
    rank must agree on whether to skip ``decoder_stage`` (``padded_num_tokens == 0``)
    or execute ``decoder_stage`` at the exact same ``padded_num_tokens`` bucket.
    """
    if dp_cpu_group is None:
        return plan
    state = torch.tensor(
        [
            1 if plan is not None else 0,
            plan.padded_num_tokens if plan is not None else 0,
        ],
        dtype=torch.int64,
        device="cpu",
    )
    torch.distributed.all_reduce(
        state, op=torch.distributed.ReduceOp.MAX, group=dp_cpu_group
    )
    if int(state[0].item()) == 0:
        return None
    target_padded = int(state[1].item())
    if plan is None:
        seq_lens = getattr(metadata, "seq_lens", None)
        requests = int(seq_lens.shape[0]) if seq_lens is not None else 1
        return DecoderReplayPlan(
            (),
            (0,) * (requests + 1),
            (0,) * requests,
            target_padded,
            is_identity=False,
        )
    if plan.padded_num_tokens == target_padded:
        return plan
    bounds = getattr(metadata, "query_start_loc_cpu", None)
    kept = len(plan.indices)
    is_identity = (
        kept > 0
        and bounds is not None
        and kept == bounds[-1]
        and target_padded == num_tokens
    )
    return DecoderReplayPlan(
        plan.indices,
        plan.query_start_loc,
        plan.replay_starts,
        target_padded,
        is_identity=is_identity,
    )
