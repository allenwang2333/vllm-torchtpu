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
"""What the boot path owes an armed fused-EP layer: equal shapes, bounded skew.

The kernel's rendezvous is an all-device barrier, so every rank must enter each
forward with the same shape and none may be so far behind that the barrier's
counting semaphore has already been reused. Arming moves *which* forwards this
covers: the layer no longer declines an uncoordinated step, so every backbone
forward from ``profile_run`` onwards runs the kernel.

Two different guarantees, from two different places, and this file pins both:

* **equal shapes** are structural. Every dummy forward's token count is config,
  not schedule -- ``profile_run`` takes ``max_num_tokens`` and the capture
  ladder walks ``num_tokens_paddings``, both of which are
  ``compilation_config.compile_sizes`` and identical on every rank.
* **bounded skew** is not structural over a *ladder*: compile time differs per
  rank and accumulates across eleven buckets until one rank is a whole bucket
  ahead of another, so ``capture_model`` puts one CPU-group barrier before each
  bucket. A single-shape phase has nothing to accumulate, which is why
  ``profile_run`` does not need one -- and why it must not have one, since a
  barrier is also a way for one rank that never arrives to hang the rest.
"""

import contextlib
from types import SimpleNamespace

import pytest

from vllm_torchtpu import utils
from vllm_torchtpu.layers.vllm import fused_moe_ep
from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings
from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker


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
    # The real ladder walk, not a stub of it: the properties under test are
    # that every bucket is preceded by a rendezvous and that `dp_lockstep`
    # reaches `_dummy_run`, and both live inside this method. Main moved the
    # loop here from `capture_model`; binding the real one is what keeps these
    # tests pointed at the code rather than at a copy of it.
    runner._precompile_backbone = (
        TPUModelRunner._precompile_backbone.__get__(runner))
    return runner, calls, barriers


@pytest.fixture(autouse=True)
def _no_shape_variants(monkeypatch):
    """Stub only the retrace bookkeeping the ladder walks through.

    `shape_variants` decides whether a bucket needs its own trace, which is a
    separate feature with its own tests and needs a real compiled model. Here
    it must simply not interfere: no retraces, nothing refused.
    """

    @contextlib.contextmanager
    def _warmup():
        yield SimpleNamespace(needs_retrace=lambda _n: False, refused=set())

    monkeypatch.setattr(tpu_runner_mod.shape_variants, "warmup", _warmup)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    fused_moe_ep.reset_engagement_state()
    monkeypatch.setattr(fused_moe_ep, "report_engagement", lambda **k: None)
    yield
    fused_moe_ep.reset_engagement_state()


@pytest.mark.parametrize("lockstep,armed,coordinated", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, False),
])
def test_the_ladder_is_coordinated_only_when_a_fused_layer_armed(
        lockstep, armed, coordinated):
    """The ladder's coordination must follow the feature, not the topology.

    ``dp_lockstep`` changes what every bucket compiles: it sets
    ``num_tokens_across_dp`` in ``set_forward_context``, changes ``DPMetadata``,
    and adds an idle-engine logits collective. Keying it on
    ``_dp_lockstep_enabled()`` -- which is ``dp_size > 1 and
    enable_expert_parallel``, with **no MoE flag in it** -- changed the compiled
    ladder on every DP+EP boot whether or not this kernel was wanted, which is
    our production cell and the opposite of default-off. It has to read
    ``_fused_moe_needs_rendezvous()``, which is the same DP+EP condition **and**
    an armed layer.
    """
    if armed:
        fused_moe_ep._register_enabled_layer(4)
    runner, calls, _ = _runner(lockstep=lockstep)
    TPUModelRunner.capture_model(runner)

    assert len(calls) == 3
    assert all(kwargs["dp_lockstep"] is coordinated for _, kwargs in calls)
    assert [args[0] for args, _ in calls] == [16, 128, 4096]


def test_with_the_flag_off_the_ladder_is_what_main_compiles():
    """"Default off" has to mean the compiled program is unchanged.

    Base passes nothing to ``_dummy_run``, whose ``dp_lockstep`` defaults to
    False. So with no layer armed -- which is what an unset flag produces, on
    any topology -- every bucket must be traced with ``dp_lockstep=False``, and
    no rank may enter a barrier that base does not have.
    """
    runner, calls, barriers = _runner(lockstep=True)  # DP+EP, feature off
    TPUModelRunner.capture_model(runner)
    assert [kwargs["dp_lockstep"] for _, kwargs in calls] == [False] * 3
    assert runner._fused_moe_needs_rendezvous() is False


def test_both_request_shapes_of_a_bucket_are_traced_the_same_way():
    fused_moe_ep._register_enabled_layer(4)
    runner, calls, barriers = _runner(lockstep=True, most_model_len=1024)
    TPUModelRunner.capture_model(runner)
    assert len(calls) == 6 and len(barriers) == 6
    assert all(kwargs["dp_lockstep"] for _, kwargs in calls)


def test_every_coordinated_bucket_is_preceded_by_a_rendezvous():
    """"Coordinated in fact" is an argument; the barrier is a guarantee.

    One barrier per bucket, because the skew this bounds is *accumulated*: the
    ranks walk the same list in the same order, so they cannot disagree about
    the shape, but they compile it at different speeds and the lead grows down
    the ladder until two ranks' barriers pair across different buckets.
    """
    fused_moe_ep._register_enabled_layer(4)
    runner, calls, barriers = _runner(lockstep=True)
    TPUModelRunner.capture_model(runner)
    assert barriers == [16, 128, 4096]
    assert len(barriers) == len(calls)


def test_the_barrier_itself_is_gated_and_not_only_its_call_site(monkeypatch):
    """The gate is inside the helper, so a caller cannot forget it.

    ``capture_model`` calls the rendezvous for every bucket unconditionally and
    the helper decides; a test that only counted call sites would therefore
    pass on a build that barriers with the feature off. This drives the real
    helper and watches for the collective.
    """
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    entered = []
    monkeypatch.setattr(dist,
                        "barrier",
                        lambda group=None: entered.append(group))
    monkeypatch.setattr(parallel_state, "get_ep_group",
                        lambda: SimpleNamespace(cpu_group="ep-cpu"))

    runner, _, _ = _runner(lockstep=True)
    TPUModelRunner._capture_ladder_rendezvous(runner, 16)
    assert entered == [], "barriered with no layer armed"

    fused_moe_ep._register_enabled_layer(4)
    TPUModelRunner._capture_ladder_rendezvous(runner, 16)
    assert entered == ["ep-cpu"]


def test_the_rendezvous_is_skipped_where_it_could_only_deadlock(monkeypatch):
    """No DP lockstep, or no armed layer: nothing to rendezvous.

    A CPU-group barrier is only safe where every rank reaches it. Both
    conditions are the runner's own, and both are cheap to get wrong in the
    direction that hangs a boot -- as a stray ``@contextlib.contextmanager`` on
    ``fused_path_is_live`` did, making it return an always-truthy object and
    arming the barrier on control runs.
    """
    assert _runner(lockstep=False)[0]._fused_moe_needs_rendezvous() is False
    assert _runner(lockstep=True)[0]._fused_moe_needs_rendezvous() is False

    fused_moe_ep._register_enabled_layer(4)
    assert _runner(lockstep=True)[0]._fused_moe_needs_rendezvous() is True
    assert _runner(lockstep=False)[0]._fused_moe_needs_rendezvous() is False


def test_the_run_is_reported_before_the_sampling_subgraphs():
    """The census must print even if a later precompile stage explodes."""
    order = []
    fused_moe_ep._register_enabled_layer(4)
    runner, _, _ = _runner(lockstep=True)
    runner._precompile_sampling_subgraphs = lambda: order.append("sampling")
    import unittest.mock as mock
    with mock.patch.object(fused_moe_ep, "report_engagement",
                           lambda **k: order.append("report")):
        TPUModelRunner.capture_model(runner)
    assert order == ["report", "sampling"]


# --------------------------------------------------------------------------
# The profile run, which arming turned into the kernel's first execution
# --------------------------------------------------------------------------


def test_the_profile_shape_is_the_top_of_the_config_ladder(monkeypatch):
    """Every rank profiles at the same shape, and nothing rank-local says so.

    With arming, ``profile_run``'s dummy forward is the **first execution of
    the kernel** on every rank, so its barrier pairs only if the shapes match.
    They do, by construction rather than by coordination: the count is
    ``model_runner.max_num_tokens``, which is ``num_tokens_paddings[-1]``,
    which is ``compilation_config.compile_sizes[-1]`` -- computed once from
    ``scheduler_config.max_num_batched_tokens`` and carried inside the config
    every worker is constructed from. No rank, dp_rank, schedule or device
    state is consulted.

    That is why there is no rendezvous here and must not be one: the barrier
    would bound arrival skew that a single-shape phase cannot accumulate, and
    it would add a collective to a path where a rank that never arrives (an
    engine that took the ``kv_cache_memory_bytes`` short-circuit, say) hangs
    the rest with no diagnostic.
    """
    profiled = []
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None,
                                     gpu_memory_utilization=0.9),
        model_runner=SimpleNamespace(
            max_num_tokens=8192,
            profile_run=lambda num_tokens: profiled.append(num_tokens)),
        devices=[],
        vllm_config=SimpleNamespace(),
    )
    monkeypatch.setattr(
        utils, "compute_hbm_budget",
        lambda devices, util: utils.HbmBudget(total_limit=64 * utils.GBYTES,
                                              total_used=8 * utils.GBYTES,
                                              cap=48 * utils.GBYTES,
                                              available=40 * utils.GBYTES))
    monkeypatch.setattr(utils, "estimate_kv_connector_hbm_reserve",
                        lambda vllm_config: 0)

    TPUWorker.determine_available_memory(worker)

    assert profiled == [worker.model_runner.max_num_tokens]


def test_the_bucket_ladder_is_a_pure_function_of_the_engine_cap():
    """The other half of "equal shapes": the ladder itself is config.

    Two ranks with the same ``max_num_batched_tokens`` get the same list, in
    the same order, so ``capture_model``'s walk is rank-invariant too. The one
    input that is not config is ``TPU_TOKEN_BUCKET_EXTRA``, and it is read
    exactly once -- on the process that first builds the config, whose
    ``compile_sizes`` then travels to every worker (a worker that already has
    the list only sorts it). A change that made a worker recompute the ladder
    would put that env var on the critical path for shape agreement.
    """
    ladder = _get_token_paddings(16, 8192, None)
    assert ladder == _get_token_paddings(16, 8192, None)
    assert ladder[0] == 16 and ladder[-1] >= 8192
    assert _get_token_paddings(16, 4096, None)[-1] < ladder[-1]
    assert 3000 in _get_token_paddings(16, 8192, [3000])


def test_the_smallest_bucket_is_what_the_surviving_trace_sees():
    """The number that made a per-step threshold unreachable, pinned.

    ``capture_model`` walks the ladder in ascending order and vLLM keeps only
    the first trace, so whatever the branch answers at ``num_tokens_paddings[0]``
    is what every bucket gets. At 16 tokens that is 64 node-wide on a DP4
    engine -- below any threshold that keeps decode on the GMM path, which is
    why the kernel reached zero compiled graphs in every serving arm before
    arming. Nothing may reintroduce a token-count branch on the strength of
    "the ladder is coordinated now".
    """
    runner, calls, _ = _runner(lockstep=True)
    TPUModelRunner.capture_model(runner)
    assert [args[0] for args, _ in calls][0] == min(runner.num_tokens_paddings)
