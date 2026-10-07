# SPDX-License-Identifier: Apache-2.0
"""Spec-decode warmup modes: with the sparse spec sampler on
(TPU_SPARSE_SPEC_SAMPLING, bound = TPU_SAMPLING_TOPK_BOUND), the synthetic
warmup also runs non-greedy requests with top_k = bound, so the sparse verify
programs (and the vocab-parallel windows) are compiled before serving."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 3


class _Runner(TPUModelRunner):
    _is_async_drafter = True

    def _precompile_timed(self, name):
        return contextlib.nullcontext()


class _Stop(Exception):
    pass


def _runner(bound, max_num_seqs=4, reqs_paddings=(1, 4), token_paddings=(16, 64)):
    r = object.__new__(_Runner)
    r._sampling_topk_bound = bound
    r.enforce_eager = False
    r.input_batch = SimpleNamespace(num_reqs=0)
    r.num_xla_graphs = 0
    r.max_model_len = 4096
    r.max_num_tokens = 8192
    r.num_tokens_paddings = list(token_paddings)
    r.num_reqs_paddings = list(reqs_paddings)
    r.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    r.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    group = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
    r.kv_cache_config = SimpleNamespace(num_blocks=100000, kv_cache_groups=[group, group])
    return r


@pytest.mark.parametrize("bound", [0, 64])
def test_modes_add_top_k_bounded_non_greedy_only_when_sparse_sampler_is_on(bound):
    modes = TPUModelRunner._spec_warmup_modes(_runner(bound))
    expected = [(True, 0), (False, 0)] + ([(False, bound)] if bound else [])
    assert modes == expected


@pytest.mark.parametrize("bound", [0, 32])
def test_warmup_sweeps_every_mode_per_prompt_bucket_and_batch(bound):
    r = _runner(bound)
    one, batch = [], []
    r._warmup_one_spec_request = lambda P, idx, quiet=False, greedy=True, top_k=0: (
        one.append((P, greedy, top_k)) or True
    )
    r._warmup_spec_batch = lambda R, P, greedy=True, top_k=0: batch.append(
        (R, P, greedy, top_k)
    )
    with (
        patch.object(tr.envs, "SPEC_WARMUP", True),
        patch.object(tr, "_suspend_kv_transfer_group", contextlib.nullcontext),
    ):
        r._warmup_spec_decode()
    modes = TPUModelRunner._spec_warmup_modes(r)
    assert one == [(P, g, k) for P in (16, 64) for g, k in modes]
    assert batch == [(4, 16, g, k) for g, k in modes]
    assert len(modes) == (3 if bound else 2)


def _capture_sampling_params(r):
    seen = []

    def execute_model(so):
        seen.extend(req.sampling_params for req in so.scheduled_new_reqs)
        raise _Stop

    r.execute_model = execute_model
    r._warmup_spec_decode_cleanup = lambda rids: True
    return seen


@pytest.mark.parametrize(
    "greedy, top_k, temperature, want_top_k",
    [(True, 0, 0.0, None), (False, 0, 1.0, None), (False, 64, 1.0, 64)],
)
def test_single_request_warmup_carries_top_k(greedy, top_k, temperature, want_top_k):
    r = _runner(64)
    seen = _capture_sampling_params(r)
    ok = TPUModelRunner._warmup_one_spec_request(
        r, 16, idx=0, quiet=True, greedy=greedy, top_k=top_k
    )
    assert ok is False  # the stub stopped it after the prefill was built
    (sp,) = seen
    assert sp.temperature == temperature
    if want_top_k is None:
        assert sp.top_k in (0, -1, None)  # vLLM's "unset"
    else:
        assert sp.top_k == want_top_k


@pytest.mark.parametrize("top_k", [0, 64])
def test_batch_warmup_carries_top_k_to_every_request(top_k):
    r = _runner(64)
    seen = _capture_sampling_params(r)
    TPUModelRunner._warmup_spec_batch(r, 4, 16, greedy=False, top_k=top_k)
    assert len(seen) == 4
    for sp in seen:
        assert sp.temperature == 1.0
        assert (sp.top_k == top_k) if top_k else (sp.top_k in (0, -1, None))
