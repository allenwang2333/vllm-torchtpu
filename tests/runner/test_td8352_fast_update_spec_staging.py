# SPDX-License-Identifier: Apache-2.0
"""TD8.352: `_try_fast_update_states` spec-token staging fast paths.

- Uniform drafts (async scheduling's placeholder rows) broadcast one row over
  the `[n_reqs, k]` index instead of `np.fromiter` over every list.
- A request's `spec_token_ids` row is rewritten only when it changed.
- Request states looked up in the first loop are reused by the spec loops.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_IDS = ["r0", "r1", "r2"]


def _runner(spec_rows):
    runner = object.__new__(TPUModelRunner)
    runner._pp_is_last = True
    runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: False)
    runner.use_async_spec_decode = False
    fallback = []
    runner.requests = {
        rid: SimpleNamespace(
            req_id=rid,
            num_computed_tokens=0,
            output_token_ids=[1],
            block_ids=([i],),
            prev_num_draft_len=0,
        )
        for i, rid in enumerate(_IDS)
    }
    runner.input_batch = SimpleNamespace(
        num_reqs=3,
        req_ids=list(_IDS),
        num_computed_tokens_cpu=np.zeros(4, dtype=np.int32),
        num_prompt_tokens=np.array([4, 5, 6, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([5, 6, 7, 0], dtype=np.int32),
        token_ids_cpu=np.zeros((4, 16), dtype=np.int32),
        is_token_ids=np.zeros((4, 16), dtype=bool),
        spec_token_ids=[list(r) for r in spec_rows] + [[]],
        block_table=SimpleNamespace(append_row=lambda nb, idx: None),
        update_req_spec_token_ids=lambda req, _s: fallback.append(req),
        refresh_metadata=lambda: None,
    )
    return runner, fallback


def _sched(spec):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        new_block_ids_to_zero=[],
        free_encoder_mm_hashes=[],
        num_scheduled_tokens=dict.fromkeys(_IDS, 3),
        scheduled_spec_decode_tokens=spec,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(_IDS),
            num_computed_tokens=[5, 6, 7],
            num_output_tokens=[1, 1, 1],
            new_block_ids=[None, None, None],
        ),
    )


def _old_staging(batch, spec_list):
    """Pre-TD8.352 reference for token_ids_cpu / is_token_ids."""
    tok = batch.token_ids_cpu.copy()
    is_tok = batch.is_token_ids.copy()
    n, k = len(spec_list), len(spec_list[0])
    starts = batch.num_tokens_no_spec[:n]
    max_len = tok.shape[1]
    flat = ((np.arange(n) * max_len + starts)[:, None] + np.arange(k)).ravel()
    tok.ravel()[flat] = np.fromiter(
        itertools.chain.from_iterable(spec_list), dtype=np.int32, count=n * k
    )
    is_tok.ravel()[flat] = True
    return tok, is_tok


@pytest.mark.parametrize(
    "spec",
    [
        dict.fromkeys(_IDS, [-1, -1]),  # async placeholders, shared list
        {rid: [-1, -1] for rid in _IDS},  # equal but distinct lists
        {"r0": [7, 8], "r1": [7, 8], "r2": [7, 9]},  # non-uniform
    ],
)
def test_spec_staging_matches_fromiter_reference(spec):
    runner, fallback = _runner([[-1, -1], [3], []])
    batch = runner.input_batch
    unchanged_row = batch.spec_token_ids[0]
    ref_tok, ref_is = _old_staging(batch, [spec[r] for r in _IDS])

    assert TPUModelRunner._try_fast_update_states(runner, _sched(spec)) is True

    np.testing.assert_array_equal(batch.token_ids_cpu, ref_tok)
    np.testing.assert_array_equal(batch.is_token_ids, ref_is)
    assert batch.token_ids_cpu.dtype == np.int32
    assert batch.spec_token_ids[:3] == [spec[r] for r in _IDS]
    # Rows are updated in place, never rebound to the scheduler's lists.
    assert batch.spec_token_ids[0] is unchanged_row
    assert all(batch.spec_token_ids[i] is not spec[r] for i, r in enumerate(_IDS))
    assert [runner.requests[r].prev_num_draft_len for r in _IDS] == [2, 2, 2]
    assert fallback == []


def test_ragged_spec_falls_back_with_cached_states_in_order():
    runner, fallback = _runner([[], [], []])
    spec = {"r0": [1, 2], "r1": [3]}  # r2 missing
    assert TPUModelRunner._try_fast_update_states(runner, _sched(spec)) is True
    assert fallback == [runner.requests[r] for r in _IDS]
    assert not runner.input_batch.is_token_ids.any()
