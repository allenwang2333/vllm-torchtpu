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
"""Engagement: is the fused kernel in the compiled graph, and does it say so?

Three device jobs were spent on serving numbers that could not be attributed to
a path, because the branch that selects the kernel is resolved when vLLM traces
a token bucket and nothing recorded which way it went (W6, `j-b18b80fc`,
`j-7337172b`, `j-88be9cc4`). Two instruments answer it now and this file pins
both, plus the invariant that makes them worth having: the ledger written in
eager code and the census taken on the traced graph must agree.
"""

from types import SimpleNamespace

import pytest
import torch


from vllm_torchtpu.compilation import moe_census
from vllm_torchtpu.layers.vllm import fused_moe_ep

HIDDEN = 8


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    fused_moe_ep.reset_engagement_state()
    moe_census.RECORDER.reset()
    # setenv, never setattr: `envs` serves these through `__getattr__`, so a
    # `monkeypatch.setattr` here would survive its own undo as a real module
    # attribute and make every later `setenv` in the session invisible. A
    # sibling file does exactly that; `tests/conftest.py::_unpin_envs_between_tests`
    # is what stops it reaching us.
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "0")
    yield
    fused_moe_ep.reset_engagement_state()
    moe_census.RECORDER.reset()


def _layer(dp_size=4):
    return SimpleNamespace(moe_config=SimpleNamespace(dp_size=dp_size))


# --------------------------------------------------------------------------
# One rule, two callers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows,dp_size,coordinated", [
    (4096, 4, True),
    (2048, 1, True),
    (16, 4, True),
    (512, 4, False),
    (4096, 4, False),
    (1, 8, True),
])
def test_the_eager_verdict_is_the_one_the_trace_applies(
        monkeypatch, rows, dp_size, coordinated):
    """The ledger is only worth writing if it describes the branch taken.

    ``engages_for`` runs inside the traced region and cannot log;
    ``engagement_verdict`` runs in eager code and is what the log prints. If
    they can disagree, the log is decoration. So drive both from the same
    inputs and require the same answer.
    """
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", coordinated, raising=False)
    layer = _layer(dp_size)
    traced = fused_moe_ep.engages_for(layer, torch.zeros(rows, HIDDEN))

    agreed = coordinated or dp_size <= 1
    eager = fused_moe_ep.engagement_verdict(rows * dp_size if agreed else None)
    assert traced == eager.engaged


def test_the_reason_names_the_number_and_the_setting():
    """A decline has to be readable without the source next to it."""
    assert "512 node-wide tokens < MIN_TOKENS=2048" in (
        fused_moe_ep.engagement_verdict(512).reason)
    assert "have not agreed" in fused_moe_ep.engagement_verdict(None).reason


# --------------------------------------------------------------------------
# The ledger.
# --------------------------------------------------------------------------


def test_the_ledger_records_every_declared_forward_shape():
    fused_moe_ep._register_enabled_layer(4)
    for rows in (4096, 4096, 16):
        with fused_moe_ep.dp_agreement(True, rows):
            pass
    with fused_moe_ep.dp_agreement(False, 4096):
        pass

    ledger = fused_moe_ep.engagement_ledger()
    assert ledger[(4096, True)][0].engaged is True
    assert ledger[(4096, True)][1] == 2
    assert ledger[(16, True)][0].engaged is False
    # Same shape, uncoordinated: a different verdict and a separate row, because
    # "declined because nobody agreed" and "declined because it is small" are
    # different failures with different fixes.
    assert ledger[(4096, False)][0].engaged is False
    assert "have not agreed" in ledger[(4096, False)][0].reason


def test_nothing_is_recorded_when_no_layer_took_the_path():
    with fused_moe_ep.dp_agreement(True, 4096):
        pass
    assert fused_moe_ep.engagement_ledger() == {}


def test_the_decline_warning_does_not_fire_in_a_control(monkeypatch):
    """An observable that appears in both arms is not an observable.

    W6 found the decline line in all three arms of a serving job, including the
    two controls, because the runner announces it on every uncoordinated
    forward whether or not any layer uses the fused path.
    """
    warned = []
    monkeypatch.setattr(fused_moe_ep.logger, "warning_once",
                        lambda *a, **k: warned.append(a))

    with fused_moe_ep.dp_agreement(False, 4096):
        pass
    assert warned == []

    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(False, 4096):
        pass
    assert len(warned) == 1


def test_the_agreement_flag_is_restored_after_the_forward():
    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(True, 4096):
        with fused_moe_ep.dp_agreement(False, 16):
            assert fused_moe_ep.dp_step_is_coordinated(4) is False
        assert fused_moe_ep.dp_step_is_coordinated(4) is True
    assert fused_moe_ep.dp_step_is_coordinated(4) is False


# --------------------------------------------------------------------------
# The report, which is the thing that refuses to be quiet.
# --------------------------------------------------------------------------


def _saw_fused_ops(n):
    for _ in range(n):
        moe_census.RECORDER.observe([
            SimpleNamespace(op="call_function",
                            target="pallas.fused_ep_moe_kernel_0.default")
        ], "Range(16384, 16384)")


def test_the_flag_on_and_the_kernel_nowhere_is_an_error():
    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(True, 4096):
        pass
    with pytest.raises(RuntimeError, match="compiled into ZERO graphs"):
        fused_moe_ep.report_engagement()


def test_the_separator_arm_has_to_say_so_out_loud(monkeypatch):
    """An arm that wants the kernel unreachable is legitimate -- and explicit."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "999999999")
    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(True, 4096):
        pass
    with pytest.raises(RuntimeError):
        fused_moe_ep.report_engagement()
    fused_moe_ep.report_engagement(strict=False)  # opted out: logs, no raise


def test_a_ledger_that_says_no_and_a_graph_that_says_yes_is_fatal():
    """The two instruments disagreeing means neither can be trusted."""
    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(True, 16):  # below the threshold
        pass
    _saw_fused_ops(3)
    with pytest.raises(RuntimeError, match="ledger says no forward engaged"):
        fused_moe_ep.report_engagement()


def test_an_engaged_run_reports_and_returns():
    fused_moe_ep._register_enabled_layer(4)
    with fused_moe_ep.dp_agreement(True, 4096):
        pass
    _saw_fused_ops(60)
    fused_moe_ep.report_engagement()  # no raise


def test_the_report_is_silent_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "0")
    fused_moe_ep.report_engagement()


def test_the_flag_on_with_no_layer_accepting_is_reported(monkeypatch):
    errors = []
    monkeypatch.setattr(fused_moe_ep.logger, "error",
                        lambda *a, **k: errors.append(a))
    fused_moe_ep.report_engagement()
    assert errors and "no layer accepted" in errors[0][0]


# --------------------------------------------------------------------------
# The census, on a graph a real tracer produced. This is the gate: everything
# above is bookkeeping unless the label matches what Dynamo actually emits.
# --------------------------------------------------------------------------


@torch.library.custom_op("pallas::fused_ep_moe_kernel_censusgate",
                         mutates_args=())
def _fused_ep_op(tokens: torch.Tensor) -> torch.Tensor:
    return tokens * 2.0


@torch.library.custom_op("pallas::fused_moe_kernel_censusgate", mutates_args=())
def _gmm_op(tokens: torch.Tensor) -> torch.Tensor:
    return tokens * 3.0


@_fused_ep_op.register_fake
def _(tokens):
    return torch.empty_like(tokens)


@_gmm_op.register_fake
def _(tokens):
    return torch.empty_like(tokens)


def _trace_and_census(fn, rows):
    """Compile ``fn`` for one bucket and census the graph the backend gets."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    # One bucket per trace, like vLLM: without the reset Dynamo can serve a
    # cached entry for the same code object and never call the backend.
    torch._dynamo.reset()
    torch.compile(fn, fullgraph=True, dynamic=False,
                  backend=backend)(torch.zeros(rows, HIDDEN))
    assert len(graphs) == 1, f"expected one graph, got {len(graphs)}"
    return moe_census.census(graphs[0].graph.nodes)


@pytest.mark.parametrize("rows,expect_fused", [(4096, True), (16, False)])
def test_the_census_reads_the_branch_out_of_a_real_traced_graph(
        monkeypatch, rows, expect_fused):
    """Which MoE program bucket B carries, answered on bucket B's graph.

    The branch is a Python ``if`` on a static shape, so Dynamo bakes one side
    of it into each bucket's graph and no runtime predicate can reach the
    other. That is the mechanism behind blocker 6, and this is the instrument
    that makes it visible for free instead of costing a 25-minute boot.
    """
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)
    layer = _layer(dp_size=4)

    def moe(x):
        if fused_moe_ep.engages_for(layer, x):
            return _fused_ep_op(x)
        return _gmm_op(x)

    counts = _trace_and_census(moe, rows)
    assert counts["fused_ep"] == (1 if expect_fused else 0)
    assert counts["gmm"] == (0 if expect_fused else 1)


def test_the_ledger_and_the_graph_agree_across_the_whole_ladder(monkeypatch):
    """The invariant both instruments exist for, checked end to end.

    Walk a bucket ladder the way ``capture_model`` does -- declare the shape in
    eager code, then trace it -- and require, bucket by bucket, that the eager
    ledger's verdict equals the number of fused custom calls in the graph the
    compiler received. A ledger that can drift from the graph is worse than no
    ledger, because it reads like evidence.
    """
    fused_moe_ep._register_enabled_layer(4)
    layer = _layer(dp_size=4)

    def moe(x):
        if fused_moe_ep.engages_for(layer, x):
            return _fused_ep_op(x)
        return _gmm_op(x)

    for rows in (8, 64, 512, 1024, 4096):
        with fused_moe_ep.dp_agreement(True, rows):
            counts = _trace_and_census(moe, rows)
        verdict = fused_moe_ep.engagement_ledger()[(rows, True)][0]
        assert verdict.engaged == bool(counts["fused_ep"]), (
            f"{rows} rows: ledger says {verdict.engaged} ({verdict.reason}), "
            f"graph says {counts['fused_ep']} fused calls")

    # dp=4, MIN_TOKENS=2048 -> the 512-row bucket is the first to clear it.
    engaged = {
        rows
        for (rows, _), (verdict,
                        _) in fused_moe_ep.engagement_ledger().items()
        if verdict.engaged
    }
    assert engaged == {512, 1024, 4096}


# --------------------------------------------------------------------------
# The traced predicate under DYNAMIC shapes. This is the gate that was missing.
# --------------------------------------------------------------------------


def test_the_traced_predicate_survives_a_dynamic_shape(monkeypatch):
    """`engages_for` may compute a bool and nothing else.

    `node_tokens` is a Dynamo `SymInt` off `hidden_states.shape[0]`. Comparing
    it adds the guard the bucket already has; *formatting* it forces the symbol
    to a constant and the compile dies with
    `Constraints violated (L['input_ids'].size()[0], ...)` -- inside
    `capture_model`, on device, taking the boot with it. An f-string in the
    decline *reason* did exactly that and voided the v2 arm of job
    `j-2ca2a7d7`; the census caught it (`fused_ep=0`) but only after the slot
    was spent.

    Every other engagement test here traces with `dynamic=False`, which is
    precisely why none of them saw it.
    """
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)
    layer = _layer(dp_size=4)

    def moe(x):
        return _fused_ep_op(x) if fused_moe_ep.engages_for(layer, x) else _gmm_op(x)

    torch._dynamo.reset()
    compiled = torch.compile(moe, fullgraph=True, dynamic=True, backend="eager")
    for rows in (4096, 2048):        # a second shape is what marks dim 0 dynamic
        compiled(torch.zeros(rows, HIDDEN))


def test_nothing_on_the_traced_predicate_formats_a_shape():
    """Structural twin of the test above, for the next person adding a message.

    A dynamic-shape trace catches it today; this catches it in the diff.
    """
    import inspect

    traced = (fused_moe_ep.engages_for, fused_moe_ep.engaged_at,
              fused_moe_ep.dp_agreed_tokens)
    offenders = [
        fn.__name__ for fn in traced
        if any(bad in inspect.getsource(fn)
               for bad in ('f"{', "f'{", ".format(", "str("))
    ]
    assert not offenders, (
        f"{offenders} format a value on the traced engagement path; a SymInt "
        "formatted there specialises the graph and breaks the compile. Put the "
        "text in engagement_verdict, which is eager-only")
