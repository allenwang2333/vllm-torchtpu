# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 v2 CED with prefix caching through SWA bounded replay.

The scheduler resumes a prefix hit at its replay start (the hit minus the SWA
window). The model must bound SWA visibility by that start and leave the
prefix-cached KV of the replayed tokens untouched; the runner keeps the start
per request.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
    plan_decoder_replay,
    prefix_replay_enabled,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

WINDOW = 128


def _plan_metadata(bounds, lengths, prompts, hits=None):
    return SimpleNamespace(
        query_start_loc_cpu=tuple(bounds),
        seq_lens_cpu=tuple(lengths),
        prompt_lens_cpu=tuple(prompts),
        replay_starts_cpu=None if hits is None else tuple(hits),
    )


def test_plan_without_hits_keeps_prompt_tail():
    # One 1024-token prompt in one chunk: the decoder runs the last window.
    plan = plan_decoder_replay(
        _plan_metadata([0, 1024], [1024], [1024]), WINDOW, 1024, ()
    )
    assert plan.replay_starts == (1024 - WINDOW,)
    assert plan.indices == tuple(range(1024 - WINDOW, 1024))


def test_plan_fresh_hit_resumes_inside_the_scheduled_chunk():
    # A 1024-token prompt hit at 640 resumes at 512. The decoder still starts
    # at the prompt tail, which the scheduled chunk [512, 1024) covers.
    metadata = _plan_metadata([0, 512], [1024], [1024], hits=[512])
    plan = plan_decoder_replay(metadata, WINDOW, 512, ())
    assert plan.replay_starts == (1024 - WINDOW,)
    assert plan.indices == tuple(range(512 - WINDOW, 512))


def test_plan_resumed_hit_past_the_prompt_bounds_the_decoder():
    # A preempted request (prompt 256, 1000 tokens so far) resumes at 768:
    # no decoder SWA row exists below it.
    metadata = _plan_metadata([0, 232], [1000], [256], hits=[768])
    plan = plan_decoder_replay(metadata, WINDOW, 256, ())
    assert plan.replay_starts == (768,)
    assert plan.indices == tuple(range(232))


def test_plan_rejects_mismatched_hits():
    metadata = _plan_metadata([0, 4, 8], [4, 8], [4, 8], hits=[0])
    with pytest.raises(ValueError, match="Invalid decoder replay"):
        plan_decoder_replay(metadata, WINDOW, 8, ())


def test_replayed_marks_only_the_hit_window():
    # Two requests: the first resumes at 256 (tokens 256..300 scheduled),
    # the second has no hit (replay start 0).
    positions = torch.cat((torch.arange(256, 400), torch.arange(0, 16)))
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 144, 160]))
    replayed = v2_attn._replayed(metadata, positions, torch.tensor([256, 0]), WINDOW)
    expected = torch.cat(
        (
            torch.arange(256, 400) < 256 + WINDOW,
            torch.zeros(16, dtype=torch.bool),
        )
    )
    assert torch.equal(replayed, expected)


@pytest.mark.parametrize(
    "ced,caching,bounded,pcp,expected",
    [
        (True, True, True, 1, True),
        (False, True, True, 1, False),
        (True, False, True, 1, False),
        (True, True, False, 1, False),
        (True, True, True, 2, False),
    ],
)
def test_prefix_replay_enabled(monkeypatch, ced, caching, bounded, pcp, expected):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", ced, raising=False)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=caching, swa_bounded_replay=bounded
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )
    assert prefix_replay_enabled(config) is expected


def _runner(replay_tokens=WINDOW):
    runner = object.__new__(TPUModelRunner)
    spec = SimpleNamespace(prefix_replay_tokens=replay_tokens)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    runner._replay_starts = {}
    return runner


def _scheduler_output(new=(), cached=(), resumed=()):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, replay_start=s, num_computed_tokens=s)
            for r, s in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _ in cached],
            num_computed_tokens=[n for _, n in cached],
            resumed_req_ids=set(resumed),
        ),
    )


def test_record_replay_starts_for_new_and_resumed_requests():
    runner = _runner()
    output = _scheduler_output(
        new=[("hit", 512), ("cold", 0)],
        cached=[("running", 900), ("resumed", 768), ("resumed_cold", 0)],
        resumed=["resumed", "resumed_cold"],
    )
    runner._record_replay_starts(output)
    assert runner._replay_starts == {"hit": 512, "resumed": 768}


def test_record_replay_starts_is_off_without_a_replay_group():
    runner = _runner(replay_tokens=0)
    runner._record_replay_starts(_scheduler_output(new=[("hit", 512)]))
    assert runner._replay_starts == {}


class _Buffer:
    def __init__(self, rows, depth):
        self.np = np.zeros((rows, depth), dtype=np.int64)

    def copy_to_gpu(self):
        return torch.from_numpy(self.np.copy())


def test_lookback_reads_committed_tokens_on_a_replayed_admission():
    runner = _runner()
    runner.lookback_token_ids = _Buffer(rows=3, depth=2)
    tokens = np.arange(1000, dtype=np.int64)[None, :].repeat(3, axis=0)
    runner.input_batch = SimpleNamespace(
        req_ids=["resumed", "later", "fresh"],
        num_computed_tokens_cpu=np.array([768, 800, 512]),
        num_prompt_tokens=np.array([256, 256, 1024]),
        num_tokens_no_spec=np.array([1000, 1000, 1024]),
        token_ids_cpu=tokens,
    )
    # "later" resumed at 768 earlier, so this step is not its admission.
    runner._replay_starts = {"resumed": 768, "later": 768, "fresh": 512}
    ids = runner._prepare_lookback_token_ids(3)
    assert ids[0].tolist() == [767, 766]
    # Generated positions on later steps may hold async placeholders.
    assert ids[1].tolist() == [-1, -1]
    assert ids[2].tolist() == [511, 510]


def test_plan_decoder_replay_accepts_tensor_hits():
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 512], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
        prompt_lens_cpu=torch.tensor([1024], dtype=torch.int32),
        replay_starts_cpu=torch.tensor([512], dtype=torch.int32),
    )
    plan = plan_decoder_replay(metadata, WINDOW, 512, ())
    assert plan is not None
    assert plan.replay_starts == (1024 - WINDOW,)


def test_swa_cache_init_sets_bounded_replay_only_when_enabled(monkeypatch):
    captured = []

    def fake_super_init(self, *args, **kwargs):
        captured.append(dict(kwargs))

    monkeypatch.setattr(v2_attn.DeepseekV4SWACache, "__init__", fake_super_init)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", True, raising=False)

    # 1. When prefix_replay_enabled is False, bounded_replay is omitted.
    monkeypatch.setattr(v2_attn, "get_current_vllm_config_or_none", lambda: None)
    swa = object.__new__(v2_attn.SWACache)
    v2_attn.SWACache.__init__(swa, cache_config=SimpleNamespace(block_size=32))
    assert "bounded_replay" not in captured[-1]

    # 2. When prefix_replay_enabled is True, bounded_replay=True is passed.
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    monkeypatch.setattr(v2_attn, "get_current_vllm_config_or_none", lambda: cfg)
    swa2 = object.__new__(v2_attn.SWACache)
    v2_attn.SWACache.__init__(swa2, cache_config=SimpleNamespace(block_size=32))
    assert captured[-1]["bounded_replay"] is True


def test_window_starts_stage_cache_and_attention_paths(monkeypatch):
    try:
        from tests.models.vllm.test_dsv41_v2_attention_td756 import (
            _make_attn,
            _make_paged_cache_mock,
        )
    except ModuleNotFoundError:
        from test_dsv41_v2_attention_td756 import (  # type: ignore[import-not-found]
            _make_attn,
            _make_paged_cache_mock,
        )

    pages = torch.zeros((16, 32, 512), dtype=torch.bfloat16)
    swa_cache = _make_paged_cache_mock(pages, row_offset=0, rows_per_page=32)
    main_cache = _make_paged_cache_mock(pages, row_offset=0, rows_per_page=32)
    idx_cache = _make_paged_cache_mock(pages, row_offset=0, rows_per_page=32)
    attn = _make_attn(
        swa_cache=swa_cache,
        main_cache=main_cache,
        idx_cache=idx_cache,
        has_indexer=True,
    )
    attn.compressor = lambda kv_score, pos: torch.zeros(
        (pos.shape[0], 8), dtype=torch.bfloat16
    )
    attn._quantize_indexer = lambda k: k

    meta = SimpleNamespace(
        seq_lens=torch.tensor([300, 16], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        block_tables=torch.arange(32, dtype=torch.int32).reshape(2, 16),
    )
    win_starts = torch.tensor([256, 0], dtype=torch.int32)
    ctx = SimpleNamespace(
        attn_metadata={"swa_pfx": meta, "src_pfx": meta, "idx_pfx": meta},
        additional_kwargs={"deepseek_v41_window_starts": win_starts},
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
    monkeypatch.setattr(
        v2_attn,
        "sparse_fp8_attention",
        lambda q, *a, **kw: torch.zeros_like(q),
    )

    positions = torch.tensor([256, 257, 0, 1], dtype=torch.int32)
    hidden = torch.zeros((4, 8), dtype=torch.bfloat16)
    qr = torch.zeros((4, 16), dtype=torch.bfloat16)
    kv = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    kv_score = torch.zeros((4, 9), dtype=torch.float32)
    idx_w = torch.zeros((4, 2), dtype=torch.bfloat16)
    out = torch.zeros((4, 2, 8), dtype=torch.bfloat16)

    # 1. Unscoped _prepare_tpu + _write_global_kv with window_starts != None
    assert v2_attn._window_starts() is win_starts
    attn._prepare_tpu(hidden, qr, kv, None, kv_score, idx_w, positions, out)

    # 2. Stage-scoped _prepare_tpu across two calls (populates and reuses window_starts)
    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(hidden, qr, kv, None, kv_score, idx_w, positions, out)
        assert v2_attn._active_stage_attn_cache["window_starts"] is win_starts
        assert v2_attn._window_starts() is win_starts


@pytest.mark.parametrize("concurrency", [64, 96, 112])
def test_td8202_high_conc_c64_c96_c112_prefix_replay_hot_path(concurrency):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41Model

    b = concurrency * 6
    qsl = tuple(range(0, b + 1, 6))
    seq_lens = (4096,) * concurrency
    prompts = (1024,) * concurrency
    zero_hits = (0,) * concurrency

    meta_none = SimpleNamespace(
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq_lens,
        prompt_lens_cpu=prompts,
        replay_starts_cpu=None,
    )
    meta_tuple = SimpleNamespace(
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq_lens,
        prompt_lens_cpu=prompts,
        replay_starts_cpu=zero_hits,
    )
    plan_none = plan_decoder_replay(meta_none, WINDOW, b, (b,))
    plan_tuple = plan_decoder_replay(meta_tuple, WINDOW, b, (b,))
    assert plan_none == plan_tuple
    assert plan_none.is_identity is True
    assert plan_none.padded_num_tokens == b

    dummy_model = SimpleNamespace()
    positions = torch.zeros(b, dtype=torch.int32)
    w1 = DeepseekV41Model._get_ced_replay_starts(
        dummy_model,
        zero_hits,
        positions,
        cache_attr="_ced_window_starts_cache",
    )
    w2 = DeepseekV41Model._get_ced_replay_starts(
        dummy_model,
        zero_hits,
        positions,
        cache_attr="_ced_window_starts_cache",
    )
    assert w1 is w2
    assert int(w1.sum().item()) == 0
