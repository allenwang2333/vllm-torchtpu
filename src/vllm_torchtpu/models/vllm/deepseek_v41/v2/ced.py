"""Host planning for decoder SWA bounded replay across scheduled chunks."""

import functools
from dataclasses import dataclass

import torch

from vllm_torchtpu import envs


def prefix_replay_enabled(vllm_config) -> bool:
    """Whether CED runs with prefix caching through SWA bounded replay.

    Upstream's contract: the SWA group leaves prefix caching, and the
    scheduler resumes each hit one window early so replay rebuilds it. CED
    requires unsharded token rows, so prefill context parallelism stays out.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return bool(
        envs.TPU_DEEPSEEK_V41_CED
        and getattr(cache_config, "enable_prefix_caching", False)
        and getattr(cache_config, "swa_bounded_replay", False)
        and getattr(parallel_config, "prefill_context_parallel_size", 1) == 1
    )


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int
    is_identity: bool = False


@functools.lru_cache(maxsize=64)
def _identity_indices(kept: int) -> tuple[int, ...]:
    return tuple(range(kept))


@functools.lru_cache(maxsize=64)
def _zero_hits(count: int) -> tuple[int, ...]:
    return (0,) * count


@functools.lru_cache(maxsize=256)
def _padded_bucket(kept: int, num_tokens: int, compile_sizes: tuple) -> int:
    """Smallest compile size in ``[kept, num_tokens)``, else ``num_tokens``."""
    if kept == 0:
        return 0
    padded = num_tokens
    for size in compile_sizes:
        n = int(size)
        if kept <= n < padded:
            padded = n
    return padded


def plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    """Keep the same prompt-tail window across every chunk and later decode.

    The encoder publishes global KV for the complete chunk before this plan
    narrows the decoder batch. With prefix caching, the cache holds no SWA
    rows, so the scheduler resumes a hit at its replay start (SWA bounded
    replay): the decoder window starts no earlier than that.
    """
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None  # Profiling/warmup executes both stages at the full bucket.
    bounds_list = bounds.tolist() if isinstance(bounds, torch.Tensor) else bounds
    lengths_list = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
    prompts_list = prompts.tolist() if isinstance(prompts, torch.Tensor) else prompts
    hits = getattr(metadata, "replay_starts_cpu", None)
    if hits is None:
        hits_list = _zero_hits(len(lengths_list))
    elif isinstance(hits, torch.Tensor):
        hits_list = hits.tolist()
    else:
        hits_list = hits
    if (
        window < 1
        or len(bounds_list) != len(lengths_list) + 1
        or len(prompts_list) != len(lengths_list)
        or len(hits_list) != len(lengths_list)
    ):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds_list[0] != 0 or bounds_list[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    replay_starts = []
    any_skipped = False
    for start, end, length, prompt, hit in zip(
        bounds_list, bounds_list[1:], lengths_list, prompts_list, hits_list
    ):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = prompt - window if prompt > window else 0
        if hit > replay_start:
            replay_start = hit
        replay_starts.append(replay_start)
        if replay_start > length - count and count > 0:
            any_skipped = True
    if not any_skipped:
        kept = bounds_list[-1]
        indices_tuple = _identity_indices(kept)
        output_bounds_tuple = tuple(bounds_list)
    else:
        indices = []
        output_bounds = [0]
        for start, end, length, replay_start in zip(
            bounds_list, bounds_list[1:], lengths_list, replay_starts
        ):
            prior_tokens = length - (end - start)
            keep_start = (
                start + replay_start - prior_tokens
                if replay_start > prior_tokens
                else start
            )
            indices.extend(range(keep_start, end))
            output_bounds.append(len(indices))
        kept = len(indices)
        indices_tuple = tuple(indices)
        output_bounds_tuple = tuple(output_bounds)
    # Each stage dispatches by its own token bucket. Never introduce a second,
    # independent symbolic token dimension into vLLM's guard-free full graph.
    if not isinstance(compile_sizes, tuple):
        compile_sizes = tuple(compile_sizes)
    required_kept = kept
    owner_layout = getattr(metadata, "owner_token_layout", None)
    req_owners = getattr(owner_layout, "owners", None)
    if kept > 0 and owner_layout is not None and req_owners is not None:
        owners_list = (
            req_owners.tolist() if hasattr(req_owners, "tolist") else req_owners
        )
        tp_size = int(owner_layout.query_start_loc.shape[0])
        owner_kept = [0] * tp_size
        for owner, start, end in zip(
            owners_list, output_bounds_tuple, output_bounds_tuple[1:]
        ):
            owner_kept[owner] += end - start
        required_kept = max(kept, tp_size * max(owner_kept))
    padded = _padded_bucket(required_kept, int(num_tokens), compile_sizes)
    is_identity = kept > 0 and kept == bounds_list[-1] and padded == num_tokens
    return DecoderReplayPlan(
        indices_tuple,
        output_bounds_tuple,
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
