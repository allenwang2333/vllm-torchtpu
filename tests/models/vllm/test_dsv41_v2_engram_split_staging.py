# SPDX-License-Identifier: Apache-2.0
"""Split Engram staging for async speculative mixed steps.

The prompt-only suffix of a mixed step is hashed on the host and staged before
the previous step's device sync; the device hashes only the decode-row prefix.
The staged rows must equal the whole-buffer path bit for bit.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import Engram, NgramHashState
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    HostEngramInputs,
    _prompt_row_start,
    capture_host_engram_inputs,
)

COLS, DIM, BLOCKS = 2, 32, 2


def _engram(max_tokens: int = 16) -> Engram:
    engram = object.__new__(Engram)
    table = torch.randint(0, 256, (64, DIM), dtype=torch.uint8)
    scales = torch.randint(100, 140, (64, BLOCKS), dtype=torch.uint8)

    def gather_raw(indices, raw_out, scale_out):
        rows = indices[:, :COLS].long() % 64
        raw_out.copy_(table[rows])
        scale_out.copy_(scales[rows])

    engram.embed_tokens = SimpleNamespace(
        cpu_offload=True,
        host_offload=True,
        dp_size=1,
        part_n_hash_cols=COLS,
        n_hash_cols=COLS,
        head_dim=DIM,
        block_size=DIM // BLOCKS,
        tp_size=1,
        gather_raw=gather_raw,
    )
    with patch.object(v2_engram.upstream.Engram, "_init_staging"):
        engram._init_staging(max_tokens=max_tokens, head_dim=DIM)
    return engram


def test_split_staging_matches_whole_buffer_path():
    engram = _engram()
    assert engram.supports_split_staging()
    g = torch.Generator().manual_seed(1)
    for step in range(3):
        hash_ids = torch.randint(0, 1000, (12, COLS), generator=g, dtype=torch.int32)
        engram.prepare_embeddings(hash_ids)
        expected = engram._ready_rows(12).clone()
        head = 4
        engram.prepare_tail(hash_ids[head:], head, 12)
        assert engram._pending_split == (head, 12)
        engram.prepare_head(hash_ids[:head])
        assert engram._pending_split is None
        got = engram._ready_rows(12)
        # Random FP8 bytes include NaN encodings: compare bit patterns.
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (
            f"step {step}"
        )
        assert torch.equal(
            engram._ready_rows(5).view(torch.int16), expected[:5].view(torch.int16)
        )
    # Two alternating device tensors per bucket; the whole-buffer tensor is
    # separate, so the early tail upload never touches a buffer in use.
    pair = engram._split_staged_by_tokens[12]
    assert pair[0] is not pair[1]
    assert pair[0] is not engram._staged_by_tokens.get(12)


def test_split_staging_requires_word_aligned_scale_rows():
    engram = _engram()
    engram._host_scale = engram._host_scale[:, :, :1]  # cols*blocks == 2
    assert not engram.supports_split_staging()


def test_hash_rows_prefix_matches_full_device_hash():
    g = torch.Generator().manual_seed(0)
    layers, ngram, heads, depth = 2, 3, 4, 4
    state = object.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.token_map = torch.randint(1, 50, (100,), generator=g)
    state.multipliers = torch.randint(1, 1 << 20, (layers, ngram), generator=g)
    state.primes = torch.randint(1000, 5000, (layers, ngram - 1, heads), generator=g)
    state.offsets = torch.randint(0, 9999, (layers, (ngram - 1) * heads), generator=g)
    state.lookback_depth = depth
    state.pad_id = 0
    state.use_slot_cache = False
    state.block_size = 8
    state._cache = None
    ids = torch.randint(1, 90, (16,), generator=g, dtype=torch.int32)
    pos = torch.cat(
        [torch.arange(6), torch.arange(5), torch.zeros(5, dtype=torch.long)]
    )
    qsl = torch.tensor([0, 6, 11, 11], dtype=torch.int32)
    lookback = torch.randint(1, 90, (4, depth), generator=g, dtype=torch.int32)
    dead = torch.zeros(16, dtype=torch.bool)
    dead_lb = torch.zeros_like(lookback, dtype=torch.bool)
    block_table = torch.arange(16).reshape(4, 4)
    full = state.forward(ids, pos, qsl, dead, lookback, dead_lb, None, block_table)
    head = state.forward(
        ids, pos, qsl, dead, lookback, dead_lb, None, block_table, hash_rows=7
    )
    assert head.shape[0] == 7
    assert torch.equal(head, full[:7])
    whole = state.forward(
        ids, pos, qsl, dead, lookback, dead_lb, None, block_table, hash_rows=99
    )
    assert torch.equal(whole, full)


def _spec_runner(seq_lens, num_prompt, qsl):
    return types.SimpleNamespace(
        _last_sequence_layout_plan=types.SimpleNamespace(
            kind=SequenceLayoutKind.ALL, start_index=0
        ),
        input_batch=types.SimpleNamespace(
            num_prompt_tokens=np.array(num_prompt),
            num_computed_tokens_cpu=np.array(seq_lens) - 1,
            token_ids_cpu=np.zeros((4, 64), dtype=np.int32),
            req_ids=[f"r{i}" for i in range(len(seq_lens))],
        ),
        seq_lens_np=np.array(seq_lens),
        input_ids_cpu=torch.zeros(32, dtype=torch.int32),
        positions_cpu=torch.zeros(32, dtype=torch.int64),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        lookback_token_ids=types.SimpleNamespace(
            cpu=torch.zeros(4, 4, dtype=torch.int32)
        ),
        _pre_async_results=types.SimpleNamespace(
            spec_decode_num_rejected_tokens=torch.zeros(2, dtype=torch.int32)
        ),
        speculative_config=object(),
    )


def test_capture_marks_prompt_only_suffix_under_spec_decode():
    # Two decoding requests (6 verify rows each) then a chunked prefill.
    runner = _spec_runner([20, 30, 16], [10, 10, 40], [0, 6, 12, 28, 28])
    captured = capture_host_engram_inputs(runner, 3)
    assert isinstance(captured, HostEngramInputs)
    assert captured.host_ready is False and not captured.can_hash_on_host
    assert captured.prompt_row_start == 12
    # No prompt-only suffix: every request is decoding.
    runner = _spec_runner([20, 30], [10, 10], [0, 6, 12, 12, 12])
    assert capture_host_engram_inputs(runner, 2).prompt_row_start is None
    # A decode request after the prompt request breaks the suffix.
    runner = _spec_runner([16, 30], [40, 10], [0, 16, 22, 22, 22])
    assert capture_host_engram_inputs(runner, 2).prompt_row_start is None
    # Every request prompt-only: the suffix starts at row 0 (plain host path).
    assert (
        _prompt_row_start(
            _spec_runner([16, 8], [40, 8], [0, 16, 24, 24, 24]),
            2,
            0,
            [16, 8],
            [40, 8],
        )
        == 0
    )
    assert _prompt_row_start(runner, 2, 1, [16, 30], [40, 10]) is None


def test_model_dispatches_split_path_only_for_mixed_spec_steps():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41Model

    model = object.__new__(DeepseekV41Model)
    model._engram_device_rows = 12
    model.start_layer, model.end_layer = 0, 2
    engram = SimpleNamespace(supports_split_staging=lambda: True)
    model.layers = [SimpleNamespace(engram=engram), SimpleNamespace(engram=None)]
    ids = torch.zeros(32, dtype=torch.int32)
    snap = HostEngramInputs(
        input_ids=ids,
        positions=torch.zeros(32, dtype=torch.int64),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        lookback_token_ids=torch.zeros(4, 4, dtype=torch.int32),
        all_prompt=False,
        host_ready=False,
        prompt_row_start=8,
    )
    with patch.object(
        v2_engram.upstream, "get_engram_dp_size", return_value=1, create=True
    ):
        assert model._engram_split_rows(snap, ids, "encoder") == 12
        assert model._engram_split_rows(snap, ids, "decoder") is None
        assert model._engram_split_rows(snap, ids[:12], "encoder") is None
        ready = HostEngramInputs(**{**snap.__dict__, "host_ready": True})
        assert model._engram_split_rows(ready, ids, "encoder") is None
        late = HostEngramInputs(**{**snap.__dict__, "prompt_row_start": 13})
        assert model._engram_split_rows(late, ids, "encoder") is None
        model._engram_device_rows = 0
        assert model._engram_split_rows(snap, ids, "encoder") is None
