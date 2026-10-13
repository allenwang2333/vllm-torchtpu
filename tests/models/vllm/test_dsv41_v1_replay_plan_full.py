# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v1/replay_plan.py."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class _StubModule(ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        return MagicMock()


for _mod_name in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.model_executor.models.config",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _StubModule(_mod_name)

v1_replay = importlib.import_module(
    "vllm_torchtpu.models.vllm.deepseek_v41.v1.replay_plan"
)


def _ref_plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None
    indices = None
    replay_starts, output_bounds = [], [0]
    kept = 0
    for start, end, length, prompt in zip(bounds, bounds[1:], lengths, prompts):
        count = end - start
        replay_start = max(0, prompt - window)
        skip = min(count, max(0, replay_start - (length - count)))
        if skip > 0 and indices is None:
            indices = list(range(kept))
        if indices is not None:
            indices.extend(range(start + skip, end))
        kept += count - skip
        replay_starts.append(replay_start)
        output_bounds.append(kept)
    indices_tuple = tuple(range(kept)) if indices is None else tuple(indices)
    sizes_key = (
        compile_sizes if isinstance(compile_sizes, tuple) else tuple(compile_sizes)
    )
    buckets = tuple(sorted({num_tokens, *(int(n) for n in sizes_key if int(n) > 0)}))
    padded = next(n for n in buckets if n >= kept) if kept else 0
    return v1_replay.DecoderReplayPlan(
        indices_tuple, tuple(output_bounds), tuple(replay_starts), padded
    )


def test_plan_decoder_replay_validation_and_fast_paths() -> None:
    meta_none = SimpleNamespace(
        query_start_loc_cpu=None, seq_lens_cpu=[2], prompt_lens_cpu=[2]
    )
    assert v1_replay.plan_decoder_replay(meta_none, 4, 8, (4, 8)) is None

    meta_bad_len = SimpleNamespace(
        query_start_loc_cpu=[0, 2], seq_lens_cpu=[2, 3], prompt_lens_cpu=[2]
    )
    with pytest.raises(ValueError, match="Invalid decoder replay scheduling"):
        v1_replay.plan_decoder_replay(meta_bad_len, 4, 8, (4, 8))

    meta_oob = SimpleNamespace(
        query_start_loc_cpu=[0, 10], seq_lens_cpu=[10], prompt_lens_cpu=[10]
    )
    with pytest.raises(ValueError, match="exceed the input token bucket"):
        v1_replay.plan_decoder_replay(meta_oob, 4, 8, (4, 8))

    # Invalid count < 0 / prompt < 0 -> _static_no_skip_spec returns None,
    # fallback raises:
    meta_neg_prompt = SimpleNamespace(
        query_start_loc_cpu=(0, 2), seq_lens_cpu=[2], prompt_lens_cpu=(-1,)
    )
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt"):
        v1_replay.plan_decoder_replay(meta_neg_prompt, 4, 8, (4, 8))

    # Valid spec, but length < count on fallback raises:
    meta_short_len = SimpleNamespace(
        query_start_loc_cpu=[0, 3], seq_lens_cpu=[1], prompt_lens_cpu=[2]
    )
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt"):
        v1_replay.plan_decoder_replay(meta_short_len, 4, 8, (4, 8))

    # All-kept fast path (cold + warm _LAST_NO_SKIP_CACHE hit + per-element ge check):
    meta_all_kept = SimpleNamespace(
        query_start_loc_cpu=[0, 1, 3, 3],
        seq_lens_cpu=[10, 6, 5],
        prompt_lens_cpu=[8, 5, 2],
    )
    plan1 = v1_replay.plan_decoder_replay(meta_all_kept, 4, 8, (0, 4, 8, 16))
    meta_all_kept_step2 = SimpleNamespace(
        query_start_loc_cpu=[0, 1, 3, 3],
        seq_lens_cpu=[5, 3, 0],  # min(lengths)==0 < max_min_len, but all(ge) is True
        prompt_lens_cpu=[8, 5, 2],
    )
    plan2 = v1_replay.plan_decoder_replay(meta_all_kept_step2, 4, 8, (0, 4, 8, 16))
    ref_kept = _ref_plan_decoder_replay(meta_all_kept, 4, 8, (0, 4, 8, 16))
    assert plan1 == ref_kept
    assert plan2 is plan1

    # Partial-skip path (skip == 0 on req 0, skip > 0 on req 1, skip == 0 on req 2,
    # skip > 0 on req 3):
    meta_partial = SimpleNamespace(
        query_start_loc_cpu=[0, 2, 6, 8, 10],
        seq_lens_cpu=[2, 4, 10, 2],
        prompt_lens_cpu=[2, 6, 10, 6],
    )
    plan_partial = v1_replay.plan_decoder_replay(meta_partial, 2, 10, [4, 8, 10])
    assert plan_partial == _ref_plan_decoder_replay(meta_partial, 2, 10, [4, 8, 10])

    # Zero-kept path (kept == 0 -> padded == 0):
    meta_zero = SimpleNamespace(
        query_start_loc_cpu=[0, 2],
        seq_lens_cpu=[2],
        prompt_lens_cpu=[6],
    )
    plan_zero = v1_replay.plan_decoder_replay(meta_zero, 2, 8, (4, 8))
    assert plan_zero == _ref_plan_decoder_replay(meta_zero, 2, 8, (4, 8))
    assert plan_zero.padded_num_tokens == 0
