"""Decoder replay planning and its bounded SWA state for shared CSA."""

import bisect
import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int


@functools.lru_cache(maxsize=64)
def _range_tuple(n: int) -> tuple[int, ...]:
    return tuple(range(n))


@functools.lru_cache(maxsize=64)
def _sorted_buckets(num_tokens: int, compile_sizes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({num_tokens, *(int(n) for n in compile_sizes if int(n) > 0)}))


@functools.lru_cache(maxsize=256)
def _padded_bucket(kept: int, num_tokens: int, compile_sizes: tuple[int, ...]) -> int:
    if kept <= 0:
        return 0
    buckets = _sorted_buckets(num_tokens, compile_sizes)
    return buckets[bisect.bisect_left(buckets, kept)]


@functools.lru_cache(maxsize=64)
def _static_no_skip_spec(
    bounds: tuple[int, ...],
    prompts: tuple[int, ...],
    window: int,
    num_tokens: int,
    compile_sizes: tuple[int, ...],
):
    replay_starts = []
    min_lens = []
    counts = []
    for start, end, prompt in zip(bounds, bounds[1:], prompts):
        count = end - start
        if count < 0 or prompt < 0:
            return None
        replay_start = prompt - window if prompt > window else 0
        counts.append(count)
        replay_starts.append(replay_start)
        min_lens.append(replay_start + count if count > 0 else 0)
    kept = bounds[-1]
    padded = _padded_bucket(kept, num_tokens, compile_sizes)
    plan = DecoderReplayPlan(_range_tuple(kept), bounds, tuple(replay_starts), padded)
    return tuple(min_lens), plan, max(min_lens, default=0), tuple(counts)


_LAST_NO_SKIP_CACHE: list = [None, None, None, None]


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
    sizes_key = (
        compile_sizes if isinstance(compile_sizes, tuple) else tuple(compile_sizes)
    )
    if (
        bounds == (last := _LAST_NO_SKIP_CACHE)[0]
        and prompts == last[1]
        and (window, num_tokens, sizes_key) == last[2]
    ):
        spec = last[3]
    else:
        bounds_t = bounds if isinstance(bounds, tuple) else tuple(bounds)
        prompts_t = prompts if isinstance(prompts, tuple) else tuple(prompts)
        spec = _static_no_skip_spec(bounds_t, prompts_t, window, num_tokens, sizes_key)
        last[:] = [bounds, prompts, (window, num_tokens, sizes_key), spec]
    if spec is None:
        raise ValueError("Invalid scheduled query or prompt length")
    if min(lengths, default=0) >= spec[2]:
        return spec[1]
    indices: list[int] | None = None
    output_bounds = [0]
    kept = 0
    for start, count, min_len, length in zip(bounds, spec[3], spec[0], lengths):
        if length < count:
            raise ValueError("Invalid scheduled query or prompt length")
        if length < min_len:
            skip = min_len - length
            if skip > count:
                skip = count
            if indices is None:
                indices = list(range(kept))
            indices.extend(range(start + skip, start + count))
            kept += count - skip
        else:
            if indices is not None:
                indices.extend(range(start, start + count))
            kept += count
        output_bounds.append(kept)
    if indices is None:
        return spec[1]
    # Each stage dispatches by its own token bucket. Never introduce a second,
    # independent symbolic token dimension into vLLM's guard-free full graph.
    padded = _padded_bucket(kept, num_tokens, sizes_key)
    return DecoderReplayPlan(
        tuple(indices), tuple(output_bounds), spec[1].replay_starts, padded
    )
