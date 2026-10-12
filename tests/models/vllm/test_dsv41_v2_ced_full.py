"""Unit and bit-exactness tests covering 100% of DeepSeek-V4.1 v2 ced.py."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

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

ced = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v2.ced")


def test_prefix_replay_enabled_and_padded_bucket():
    cfg_on = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    cfg_off = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    with patch.object(ced.envs, "TPU_DEEPSEEK_V41_CED", True):
        assert ced.prefix_replay_enabled(cfg_on)
        assert not ced.prefix_replay_enabled(cfg_off)

    assert ced._padded_bucket(0, 64, (16, 32, 64)) == 0
    assert ced._padded_bucket(12, 64, (64, 32, 16)) == 16
    assert ced._padded_bucket(40, 64, (16, 32)) == 64


def test_plan_decoder_replay_all_branches():
    # Profiling / warmup with None metadata fields
    assert (
        ced.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=None, seq_lens_cpu=None, prompt_lens_cpu=None
            ),
            window=16,
            num_tokens=32,
            compile_sizes=[16, 32],
        )
        is None
    )

    # Validation errors
    with pytest.raises(ValueError, match="Invalid decoder replay scheduling metadata"):
        ced.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=[0, 4],
                seq_lens_cpu=[4],
                prompt_lens_cpu=[4],
                replay_starts_cpu=None,
            ),
            window=0,
            num_tokens=8,
            compile_sizes=[8],
        )

    with pytest.raises(ValueError, match="Query boundaries exceed"):
        ced.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=torch.tensor([1, 4], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
                prompt_lens_cpu=torch.tensor([3], dtype=torch.int32),
                replay_starts_cpu=torch.tensor([0], dtype=torch.int32),
            ),
            window=16,
            num_tokens=8,
            compile_sizes=(8,),
        )

    # Invalid negative prompt deferred from _static_no_skip_spec -> raises in full path
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        ced.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=[0, 4],
                seq_lens_cpu=[4],
                prompt_lens_cpu=[-1],
                replay_starts_cpu=[0],
            ),
            window=16,
            num_tokens=8,
            compile_sizes=(8,),
        )

    # Invalid seq_lens < count in full path
    with pytest.raises(ValueError, match="Invalid scheduled query or prompt length"):
        ced.plan_decoder_replay(
            SimpleNamespace(
                query_start_loc_cpu=[0, 6],
                seq_lens_cpu=[4],
                prompt_lens_cpu=[10],
                replay_starts_cpu=[8],
            ),
            window=16,
            num_tokens=8,
            compile_sizes=(8,),
        )

    # Steady-state decode tensor hit across steps on _LAST_NO_SKIP_CACHE
    meta_step1 = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([8, 12], dtype=torch.int32),
        replay_starts_cpu=torch.tensor([6, 0], dtype=torch.int32),
    )
    meta_step2 = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([11, 21], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([8, 12], dtype=torch.int32),
        replay_starts_cpu=torch.tensor([6, 0], dtype=torch.int32),
    )
    p1 = ced.plan_decoder_replay(meta_step1, 16, 8, (8,))
    p2 = ced.plan_decoder_replay(meta_step2, 16, 8, (8,))
    assert p1 is p2

    # Full fallback path without skipped tokens (e.g. count == 0 empty request slot)
    meta_zero_req = SimpleNamespace(
        query_start_loc_cpu=[0, 4, 4],
        seq_lens_cpu=[4, 0],
        prompt_lens_cpu=[4, 0],
        replay_starts_cpu=[4, 0],
    )
    # Force fallback past spec by making first request shorter than spec max_min_len
    meta_fallback_noskip = SimpleNamespace(
        query_start_loc_cpu=(0, 2, 2),
        seq_lens_cpu=[6, 0],
        prompt_lens_cpu=(20, 0),
        replay_starts_cpu=(2, 0),
    )
    p_noskip = ced.plan_decoder_replay(meta_fallback_noskip, 16, 4, (4,))
    assert p_noskip is not None and p_noskip.indices == (0, 1)
    with patch.object(ced, "_static_no_skip_spec", return_value=None):
        p_full_noskip = ced.plan_decoder_replay(meta_step1, 16, 8, [8])
        assert p_full_noskip is not None and p_full_noskip.indices == (0, 1)

    p_zero = ced.plan_decoder_replay(meta_zero_req, 2, 4, (4,))
    assert p_zero is not None and p_zero.padded_num_tokens == 0

    # Skipped prefix tokens path + unskipped second request
    meta_skip = SimpleNamespace(
        query_start_loc_cpu=[0, 8, 12],
        seq_lens_cpu=[8, 10],
        prompt_lens_cpu=[20, 10],
        replay_starts_cpu=None,
    )
    p_skip = ced.plan_decoder_replay(meta_skip, 4, 16, (8, 16))
    assert p_skip is not None
    assert p_skip.indices == (8, 9, 10, 11)
    assert p_skip.query_start_loc == (0, 0, 4)


def test_sync_decoder_replay_plan_and_scatter_rows():
    sample_plan = ced.DecoderReplayPlan((0, 1), (0, 2), (0,), 4, is_identity=False)
    assert (
        ced.sync_decoder_replay_plan(
            sample_plan, SimpleNamespace(), 8, dp_cpu_group=None
        )
        is sample_plan
    )

    def _mock_all_reduce_zero(buf, op=None, group=None):
        buf[0] = 0
        buf[1] = 0

    with patch("torch.distributed.all_reduce", side_effect=_mock_all_reduce_zero):
        assert (
            ced.sync_decoder_replay_plan(None, SimpleNamespace(), 8, dp_cpu_group="dp")
            is None
        )

    def _mock_all_reduce_eight(buf, op=None, group=None):
        buf[0] = 1
        buf[1] = 8

    with patch("torch.distributed.all_reduce", side_effect=_mock_all_reduce_eight):
        synced_none = ced.sync_decoder_replay_plan(
            None, SimpleNamespace(seq_lens=torch.zeros(2)), 8, dp_cpu_group="dp"
        )
        assert synced_none is not None and synced_none.padded_num_tokens == 8

        same_padded = ced.DecoderReplayPlan((0, 1), (0, 2), (0,), 8, is_identity=False)
        assert (
            ced.sync_decoder_replay_plan(
                same_padded, SimpleNamespace(), 8, dp_cpu_group="dp"
            )
            is same_padded
        )

        promoted = ced.sync_decoder_replay_plan(
            sample_plan,
            SimpleNamespace(query_start_loc_cpu=[0, 2]),
            2,
            dp_cpu_group="dp",
        )
        assert promoted is not None and promoted.padded_num_tokens == 8

    # Verify _ced_inverse_rows CPU/device caches and _scatter_decoder_rows parity
    inv_cpu = ced._ced_inverse_rows((1, 3), 2, 5, "cpu")
    assert inv_cpu.tolist() == [2, 0, 2, 1, 2]

    ced._CED_INVERSE_DEVICE_CACHE.clear()
    for idx in range(65):
        with patch.object(torch.Tensor, "to", return_value=inv_cpu):
            got_meta = ced._ced_inverse_rows((idx,), 1, 66, "meta")
            assert got_meta is inv_cpu
    assert ced._ced_inverse_rows((64,), 1, 66, "meta") is inv_cpu

    x = torch.randn((2, 16), dtype=torch.bfloat16)
    ref_out = torch.nn.functional.pad(x, (0, 0, 0, 1)).index_select(0, inv_cpu)
    opt_out = ced._scatter_decoder_rows(x, inv_cpu)
    assert torch.equal(opt_out.view(torch.uint16), ref_out.view(torch.uint16))

    empty_x = torch.zeros((0, 16), dtype=torch.bfloat16)
    zero_inv = ced._ced_inverse_rows((), 0, 4, "cpu")
    ref_empty = torch.nn.functional.pad(empty_x, (0, 0, 0, 1)).index_select(0, zero_inv)
    opt_empty = ced._scatter_decoder_rows(empty_x, zero_inv)
    assert torch.equal(opt_empty.view(torch.uint16), ref_empty.view(torch.uint16))
