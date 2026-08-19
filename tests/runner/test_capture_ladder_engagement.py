# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``capture_model``'s ladder is what decides which MoE program each bucket has.

vLLM compiles one program per token bucket and the fused EP kernel's engagement
is a branch *inside* that program, so the ladder's own declaration of DP
coordination selects the kernel for the whole run. A ladder that declares
itself uncoordinated compiles the GMM path into every bucket and no runtime
predicate can reach the kernel afterwards -- the fix that unblocked the boot
was also what guaranteed the kernel never ran (W6 blocker 6).
"""

import contextlib
from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.vllm import fused_moe_ep
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(*, lockstep, most_model_len=None):
    calls = []
    barriers = []

    @contextlib.contextmanager
    def _noop(*args, **kwargs):
        yield

    runner = SimpleNamespace(
        enforce_eager=False,
        num_tokens_paddings=[16, 128, 4096],
        num_reqs_max_model_len=8,
        max_num_blocks_per_req=4,
        num_reqs_most_model_len=4,
        num_blocks_per_most_len_req=2,
        most_model_len=most_model_len,
        lora_config=None,
        _is_async_drafter=False,
        maybe_setup_dummy_loras=_noop,
        _precompile_timed=_noop,
        _precompile_sampling_subgraphs=lambda: None,
        _dp_lockstep_enabled=lambda: lockstep,
        _dummy_run=lambda *args, **kwargs: calls.append((args, kwargs)),
        _capture_ladder_rendezvous=lambda n: barriers.append(n),
    )
    runner._fused_moe_needs_rendezvous = (
        TPUModelRunner._fused_moe_needs_rendezvous.__get__(runner))
    return runner, calls, barriers


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    fused_moe_ep.reset_engagement_state()
    monkeypatch.setattr(fused_moe_ep, "report_engagement", lambda **k: None)
    yield
    fused_moe_ep.reset_engagement_state()


@pytest.mark.parametrize("lockstep", [True, False])
def test_the_ladder_traces_with_the_coordination_it_actually_has(lockstep):
    runner, calls, _ = _runner(lockstep=lockstep)
    TPUModelRunner.capture_model(runner)

    assert len(calls) == 3
    assert all(kwargs["dp_lockstep"] is lockstep for _, kwargs in calls), (
        "every bucket in the ladder must declare the same coordination; a "
        "bucket traced uncoordinated carries the GMM path forever")
    assert [args[0] for args, _ in calls] == [16, 128, 4096]


def test_both_request_shapes_of_a_bucket_are_traced_the_same_way():
    runner, calls, barriers = _runner(lockstep=True, most_model_len=1024)
    TPUModelRunner.capture_model(runner)
    assert len(calls) == 6 and len(barriers) == 6
    assert all(kwargs["dp_lockstep"] for _, kwargs in calls)


def test_every_coordinated_bucket_is_preceded_by_a_rendezvous():
    """"Coordinated in fact" is an argument; the barrier is a guarantee."""
    runner, calls, barriers = _runner(lockstep=True)
    TPUModelRunner.capture_model(runner)
    assert barriers == [16, 128, 4096]
    assert len(barriers) == len(calls)


def test_the_rendezvous_is_skipped_where_it_could_only_deadlock(monkeypatch):
    """No DP lockstep, or no layer on the fused path: nothing to rendezvous.

    A CPU-group barrier is only safe where every rank reaches it. Both
    conditions are the runner's own, and both are cheap to get wrong in the
    direction that hangs a boot.
    """
    # No layer on the fused path yet, whatever the lockstep setting says.
    assert _runner(lockstep=False)[0]._fused_moe_needs_rendezvous() is False
    assert _runner(lockstep=True)[0]._fused_moe_needs_rendezvous() is False

    fused_moe_ep._register_enabled_layer(4)
    assert _runner(lockstep=True)[0]._fused_moe_needs_rendezvous() is True
    assert _runner(lockstep=False)[0]._fused_moe_needs_rendezvous() is False


def test_the_run_is_reported_before_the_sampling_subgraphs():
    """The ledger must print even if a later precompile stage explodes."""
    order = []
    runner, _, _ = _runner(lockstep=True)
    runner._precompile_sampling_subgraphs = lambda: order.append("sampling")
    import unittest.mock as mock
    with mock.patch.object(fused_moe_ep, "report_engagement",
                           lambda **k: order.append("report")):
        TPUModelRunner.capture_model(runner)
    assert order == ["report", "sampling"]
