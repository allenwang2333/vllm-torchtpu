# SPDX-License-Identifier: Apache-2.0
"""TD8.346: steady-state async spec-decode host fast paths in `_prepare_inputs`.

(b) When the previous step's `req_id_to_index_copy` equals the current
    `input_batch.req_id_to_index` and the chunk starts at 0, `seq_idx_prefix`
    is the identity `arange_np[:num_reqs]` and the reject-subtract key uses an
    `("identity", num_reqs)` tag instead of `np.fromiter` + `.tobytes()`.
(c) `num_draft_tokens` is built from a C-level `map` chain instead of a
    Python generator; values are unchanged.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_REQ_IDS = ["req-0", "req-1"]


class _SpecManager:
    """Records the `num_draft_tokens` handed to `get_spec_decode_metadata`."""

    def __init__(self):
        self.calls = []

    def get_spec_decode_metadata(
        self, num_draft_tokens, cu_num_scheduled, padded_num_reqs, **kwargs
    ):
        self.calls.append(num_draft_tokens.copy())
        return SimpleNamespace(final_logits_indices=torch.zeros(4, dtype=torch.int32))


def _runner(prev_map, *, spec_tokens=None):
    """Minimal fake runner for `_prepare_inputs` on a 2-request 1+2 spec step."""
    from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner

    max_num_reqs = 2
    max_model_len = 32
    token_ids_cpu = np.arange(max_num_reqs * max_model_len, dtype=np.int32).reshape(
        max_num_reqs, max_model_len
    )
    positions_cpu = torch.full((max_model_len,), -1, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    runner = SimpleNamespace(
        reorder_batch_threshold=1,
        sequence_layout_planner=AllSequenceLayoutPlanner(),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=list(_REQ_IDS),
            req_id_to_index={rid: i for i, rid in enumerate(_REQ_IDS)},
            num_computed_tokens_cpu=np.array([4, 8], dtype=np.int32),
            token_ids_cpu=token_ids_cpu,
            token_ids_cpu_tensor=torch.from_numpy(token_ids_cpu),
            num_prompt_tokens=np.full(max_num_reqs, 16, dtype=np.int32),
        ),
        _wait_input_staging_fence=lambda: None,
        most_model_len=None,
        num_reqs_max_model_len=max_num_reqs,
        num_reqs_most_model_len=max_num_reqs,
        max_num_reqs=max_num_reqs,
        uses_mrope=False,
        arange_np=np.arange(max_model_len, dtype=np.int32),
        positions_cpu=positions_cpu,
        positions_np=positions_cpu.numpy(),
        input_ids_cpu=torch.full((max_model_len,), -1, dtype=torch.int32),
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc_np=query_start_loc_cpu.numpy(),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_cpu.numpy(),
        _check_attention_schedule=lambda *a: None,
        _pcp_enabled=False,
        _mtp_enabled=False,
        device=torch.device("cpu"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        spec_decode_manager=_SpecManager(),
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy=prev_map,
            spec_decode_num_rejected_tokens=torch.tensor([1, 2], dtype=torch.int32),
            next_tokens_tpu=None,
        ),
        _decode_device_cache_key=None,
        _request_distribution_cpu=torch.zeros(3, dtype=torch.int32),
        _has_mamba_state=False,
        mamba_slot_read_offsets=None,
        _mamba_ckpt_window=1,
        _unified_kv_layout=False,
        _replay_starts={},
        empty_slot_mappings=None,
        _build_attention_metadata=lambda **kw: ({}, None),
        lora_config=None,
        _prepare_async_token_substitution_indices=lambda *a: (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ),
        _apply_async_token_substitution=lambda ids, *a: ids,
        _collect_mamba_state_seed_copies=lambda *a: None,
        num_tokens_paddings=[8, 16, 32],
        pcp_size=1,
        dcp_size=1,
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=6,
        num_scheduled_tokens={"req-0": 3, "req-1": 3},
        scheduled_spec_decode_tokens=spec_tokens or {},
    )
    return runner, sched


def _prepare(runner, sched):
    TPUModelRunner._prepare_inputs(runner, sched, start_index=0, num_decode_reqs=0)


def _old_seq_idx_prefix(prev_map, chunk_req_ids):
    """Pre-TD8.346 reference: per-request lookup with a -1 default."""
    get = prev_map.get
    return np.fromiter(
        (get(r, -1) for r in chunk_req_ids), dtype=np.int32, count=len(chunk_req_ids)
    )


def test_identity_batch_uses_arange_view_and_tag_key():
    prev = {"req-0": 0, "req-1": 1}  # equal to, but not, input_batch's dict
    runner, sched = _runner(prev)
    assert prev is not runner.input_batch.req_id_to_index
    _prepare(runner, sched)

    prefix = runner._last_async_seq_idx_prefix
    assert np.shares_memory(prefix, runner.arange_np)
    assert prefix.dtype == np.int32
    np.testing.assert_array_equal(prefix, _old_seq_idx_prefix(prev, _REQ_IDS))
    assert runner._last_async_seq_idx_bytes == ("identity", 2)
    assert runner._cached_reject_subtract_key[2] == ("identity", 2)
    # Same device indices the fromiter path builds: seq [0, 1], pos repeat x3.
    assert runner._cached_reject_seq_idx.tolist() == [0, 1]
    assert runner._cached_reject_pos_idx[:6].tolist() == [0, 0, 0, 1, 1, 1]
    # positions [4,5,6] / [8,9,10] minus rejected [1] / [2].
    assert runner.position_ids[:6].tolist() == [3, 4, 5, 6, 7, 8]

    cached_seq = runner._cached_reject_seq_idx
    cached_pos = runner._cached_reject_pos_idx
    _prepare(runner, sched)
    assert runner._cached_reject_seq_idx is cached_seq
    assert runner._cached_reject_pos_idx is cached_pos


def test_reordered_batch_falls_back_to_lookup():
    prev = {"req-0": 1, "req-1": 0}
    runner, sched = _runner(prev)
    _prepare(runner, sched)

    prefix = runner._last_async_seq_idx_prefix
    assert not np.shares_memory(prefix, runner.arange_np)
    np.testing.assert_array_equal(prefix, _old_seq_idx_prefix(prev, _REQ_IDS))
    assert runner._last_async_seq_idx_bytes == prefix.tobytes()
    assert runner._cached_reject_seq_idx.tolist() == [1, 0]
    assert runner._cached_reject_pos_idx[:6].tolist() == [1, 1, 1, 0, 0, 0]
    # Rows swapped: req-0 subtracts 2, req-1 subtracts 1.
    assert runner.position_ids[:6].tolist() == [2, 3, 4, 7, 8, 9]


def test_partial_previous_batch_falls_back_with_minus_one():
    prev = {"req-1": 0}
    runner, sched = _runner(prev)
    _prepare(runner, sched)

    np.testing.assert_array_equal(
        runner._last_async_seq_idx_prefix, np.array([-1, 0], dtype=np.int32)
    )
    assert isinstance(runner._last_async_seq_idx_bytes, bytes)


def test_identity_tag_and_bytes_keys_never_collide():
    """Switching identity -> lookup with the same values must rebuild."""
    runner, sched = _runner({"req-0": 0, "req-1": 1})
    _prepare(runner, sched)
    identity_key = runner._cached_reject_subtract_key

    runner._pre_async_results.req_id_to_index_copy = {"req-0": 0, "req-1": 1, "x": 9}
    _prepare(runner, sched)
    lookup_key = runner._cached_reject_subtract_key
    assert lookup_key != identity_key
    assert lookup_key[2] == np.array([0, 1], dtype=np.int32).tobytes()
    assert runner._cached_reject_seq_idx.tolist() == [0, 1]


def test_subst_indices_identity_tag_matches_bytes_key():
    """`_prepare_async_token_substitution_indices` consumes the cached prefix
    and its key; the identity tag must yield the same index pair as bytes."""

    def run(prefix, key):
        fake = SimpleNamespace(
            _pre_async_results=SimpleNamespace(
                req_id_to_index_copy={"req-0": 0, "req-1": 1},
                spec_decode_num_rejected_tokens=object(),
            ),
            speculative_config=SimpleNamespace(num_speculative_tokens=2),
            _last_sequence_layout_plan=None,
            input_batch=SimpleNamespace(req_ids=list(_REQ_IDS)),
            _last_async_seq_idx_prefix=prefix,
            _last_async_seq_idx_bytes=key,
            _last_sched_tokens_bytes=None,
            arange_np=np.arange(32, dtype=np.int32),
            query_start_loc_np=None,
        )
        return TPUModelRunner._prepare_async_token_substitution_indices(
            fake,
            start_index=0,
            num_reqs=2,
            num_scheduled_tokens_per_req=np.array([3, 3], dtype=np.int32),
        )

    arange = np.arange(32, dtype=np.int32)
    cur_t, pre_t = run(arange[:2], ("identity", 2))
    ref = np.array([0, 1], dtype=np.int32)
    cur_b, pre_b = run(ref, ref.tobytes())
    np.testing.assert_array_equal(cur_t, cur_b)
    np.testing.assert_array_equal(pre_t, pre_b)
    assert pre_t.dtype == pre_b.dtype == np.int32
    assert pre_t.tolist() == [0, 1, 2, 3, 4, 5]


def test_num_draft_tokens_map_matches_generator():
    spec = {"req-0": [7, 8]}  # req-1 has no drafts -> 0
    runner, sched = _runner({"req-0": 0, "req-1": 1}, spec_tokens=spec)
    _prepare(runner, sched)

    (got,) = runner.spec_decode_manager.calls
    ref = np.fromiter((len(spec.get(r, ())) for r in _REQ_IDS), dtype=np.int32, count=2)
    np.testing.assert_array_equal(got, ref)
    assert got.dtype == np.int32
    assert got.tolist() == [2, 0]
