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
a path, and the mechanism underneath all three is one fact about how vLLM
compiles: ``TorchCompileWithNoGuardsWrapper`` traces the backbone **once** with
``guard_filter_fn=skip_all_guards_unsafe`` and never traces it again, while
``support_torch_compile`` has marked the token dimension dynamic. So a Python
branch inside the traced forward is not resolved per bucket -- it is resolved
once, by whichever shape traces first, and per-bucket specialisation happens
below it in the backend. The surviving trace is the capture ladder's *smallest*
bucket, which no token threshold that keeps decode on the GMM path can ever be
true at: a per-step engagement test does not select the kernel per step, it
selects it never.

This file pins the three consequences:

1. arming is the whole decision, and the report says so (below);
2. an armed layer's graph carries the kernel at **every** bucket, from one
   trace (`the census`);
3. nothing on the armed path reads a token count, so two buckets on opposite
   sides of the old threshold compile without a ``ConstraintViolationError``
   (`the compile gate`, with the two controls that fail if the defect returns).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import moe_census
from vllm_torchtpu.layers.vllm import fused_moe_ep
from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

HIDDEN = 8
TOP_K = 2
NUM_EXPERTS = 8
DP_SIZE = 4


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    fused_moe_ep.reset_engagement_state()
    moe_census.RECORDER.reset()
    # setenv, never setattr: `envs` serves these through `__getattr__`, so a
    # `monkeypatch.setattr` here would survive its own undo as a real module
    # attribute and make every later `setenv` in the session invisible. A
    # sibling file does exactly that, and the shared conftest fixture that used
    # to sweep those up is not on this branch -- so clear our own names first
    # rather than depend on the order the files run in.
    for name in ("USE_MOE_FUSED_EP_V2", "USE_MOE_EP_KERNEL",
                 "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                 "MOE_FUSED_EP_KERNEL_MAX_TOKENS",
                 "MOE_FUSED_EP_REQUIRE_ENGAGEMENT"):
        if name in vars(envs):
            delattr(envs, name)
    monkeypatch.setenv("USE_MOE_FUSED_EP_V2", "1")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MAX_TOKENS", "0")
    yield
    fused_moe_ep.reset_engagement_state()
    moe_census.RECORDER.reset()


# --------------------------------------------------------------------------
# The arming record, and the report that refuses to be quiet
# --------------------------------------------------------------------------


def _saw_fused_ops(n):
    for _ in range(n):
        moe_census.RECORDER.observe([
            SimpleNamespace(op="call_function",
                            target="pallas.fused_ep_moe_kernel_0.default")
        ], "Range(16384, 16384)")


def test_the_report_is_silent_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("USE_MOE_FUSED_EP_V2", "0")
    fused_moe_ep.report_engagement()


def test_the_flag_on_with_nothing_armed_is_reported(monkeypatch):
    errors = []
    monkeypatch.setattr(fused_moe_ep.logger, "error",
                        lambda *a, **k: errors.append(a))
    fused_moe_ep.report_engagement()
    assert errors and "no layer armed" in errors[0][0]


def test_armed_and_the_kernel_nowhere_is_an_error():
    """The failure mode that produced three unattributable jobs.

    Under arming this is a bicondition rather than a heuristic: an armed layer
    serves every step, so zero fused custom calls in the phase being reported
    on means the kernel is not in the program -- there is no "the buckets were
    all too small" left to explain it away.
    """
    fused_moe_ep._register_enabled_layer(DP_SIZE)
    with pytest.raises(RuntimeError, match="compiled into ZERO graphs"):
        fused_moe_ep.report_engagement()


def test_the_separator_arm_has_to_say_so_out_loud(monkeypatch):
    """An arm that wants the kernel unreachable is legitimate -- and explicit.

    Two spellings now, and they report differently on purpose. Refusing at the
    band leaves nothing armed and the log says exactly that; keeping the layer
    armed and getting no graph is the *unexplained* case, which still raises
    unless the operator opts out.
    """
    errors = []
    monkeypatch.setattr(fused_moe_ep.logger, "error",
                        lambda *a, **k: errors.append(a))
    fused_moe_ep.report_engagement()  # band refused everything: no raise
    assert errors and "no layer armed" in errors[0][0]

    fused_moe_ep._register_enabled_layer(DP_SIZE)
    with pytest.raises(RuntimeError):
        fused_moe_ep.report_engagement()
    fused_moe_ep.report_engagement(strict=False)  # opted out: logs, no raise


def test_a_graph_that_says_yes_and_nothing_armed_is_fatal():
    """The two instruments disagreeing means neither can be trusted.

    The arming record is the intent and the census is the evidence; this is the
    direction that would otherwise pass silently, because a report that only
    checked "armed implies compiled" would say nothing about a kernel that got
    into a graph nobody armed.
    """
    _saw_fused_ops(3)
    with pytest.raises(RuntimeError, match="no layer armed"):
        fused_moe_ep.report_engagement()


def test_an_armed_run_reports_and_returns():
    fused_moe_ep._register_enabled_layer(DP_SIZE)
    _saw_fused_ops(60)
    fused_moe_ep.report_engagement()  # no raise


def test_the_arming_log_does_not_fire_in_a_control(monkeypatch):
    """An observable that appears in both arms is not an observable.

    A control run has nothing to decline: with the flag off the layer must not
    say anything at all, or the line cannot be grepped for to tell the arms
    apart. With the flag on and a layer it cannot serve, exactly one line.
    """
    warned = []
    monkeypatch.setattr(fused_moe_ep.logger, "warning_once",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(
            use_ep=False),
                                   ep_size=1,
                                   dp_size=DP_SIZE))

    monkeypatch.setenv("USE_MOE_FUSED_EP_V2", "0")
    assert fused_moe_ep.is_enabled(layer) is False
    assert warned == []

    monkeypatch.setenv("USE_MOE_FUSED_EP_V2", "1")
    assert fused_moe_ep.is_enabled(layer) is False
    assert len(warned) == 1


# --------------------------------------------------------------------------
# The census, on the graph a real tracer produced from the real forward.
# Everything above is bookkeeping unless the label matches what Dynamo emits.
# --------------------------------------------------------------------------


@torch.library.custom_op("pallas::fused_ep_moe_kernel_censusgate",
                         mutates_args=())
def _fused_ep_op(tokens: torch.Tensor) -> torch.Tensor:
    return tokens * 2.0


@torch.library.custom_op("pallas::fused_moe_kernel_censusgate",
                         mutates_args=())
def _gmm_op(tokens: torch.Tensor) -> torch.Tensor:
    return tokens * 3.0


@_fused_ep_op.register_fake
def _(tokens):
    return torch.empty_like(tokens)


@_gmm_op.register_fake
def _(tokens):
    return torch.empty_like(tokens)


def _layer():
    """Enough of a RoutedExperts for ``apply_monolithic`` on either path."""
    return SimpleNamespace(
        moe_config=SimpleNamespace(experts_per_token=TOP_K,
                                   dp_size=DP_SIZE,
                                   ep_size=8,
                                   num_experts=NUM_EXPERTS),
        renormalize=True,
        scoring_func="softmax",
        _tpu_activation_str="silu",
        w13_weight=torch.zeros(NUM_EXPERTS, 2, HIDDEN),
        w2_weight=torch.zeros(NUM_EXPERTS, HIDDEN, HIDDEN),
        w13_weight_scale_inv=torch.ones(NUM_EXPERTS, 2),
        w2_weight_scale_inv=torch.ones(NUM_EXPERTS, HIDDEN),
        _ep_replica_rank=None,
        _ep_rank_ids=torch.zeros(1, dtype=torch.int32),
        _experts_start=torch.zeros((), dtype=torch.int32),
        custom_routing_function=lambda hidden_states, gating_output, topk,
        renormalize:
        (torch.ones(hidden_states.shape[0], topk),
         torch.zeros(hidden_states.shape[0], topk, dtype=torch.int32)),
    )


def _method(*, armed: bool):
    """The real quant method with only the arming answer supplied."""
    method = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
    method._fused_ep_enabled = armed
    method._fused_ep_config = dict(top_k=TOP_K,
                                   renormalize=True,
                                   activation="silu",
                                   scoring_fn="softmax",
                                   subc_quant_w1_sz=None,
                                   subc_quant_w2_sz=None,
                                   quant_act=True)
    return method


@pytest.fixture
def stub_kernels(monkeypatch):
    """Both MoE programs, as real custom ops the census can name.

    Only the two ops are stubbed. The op-cache lookup, the process-group query,
    the row padding and the routing call are the shipping ones, which is the
    point: the branch under test is the one ``apply_monolithic`` really takes.
    """
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))
    monkeypatch.setattr(fused_moe_ep, "_get_op",
                        lambda **_kw: lambda *args: _fused_ep_op(args[0]))
    monkeypatch.setattr(fp8_mod, "fused_moe_gmm",
                        lambda **kwargs: _gmm_op(kwargs["hidden_states"]))


# The two ends of a real ladder, straddling the floor the deleted per-step
# predicate used. Two counts on the same side would compile clean even with the
# defect, because the guard the first trace installs is one the second still
# satisfies -- straddling is the whole point.
ACROSS_THE_BAND_EDGE = (16, 8192)


def _trace_across(fn, token_counts, *, args=()):
    """Compile ``fn`` the way vLLM compiles the backbone, at each count.

    ``support_torch_compile`` marks the token dimension dynamic with
    ``torch._dynamo.mark_dynamic``, so a *materialisation* of that dimension is
    not a recompile: it is a ``ConstraintViolationError`` that takes the engine
    down at warmup, and ``backend=<recorder>`` reproduces it without a TPU.

    Returns the graphs the backend was handed. One graph is the shipping
    situation -- vLLM drops Dynamo's guards after the first trace, so a second
    graph would never be selected and the first bucket's answer would silently
    become every bucket's.
    """
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    for num_tokens in token_counts:
        x = torch.ones(num_tokens, HIDDEN)
        logits = torch.zeros(num_tokens, NUM_EXPERTS)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(logits, 0)
        compiled(*args, x, logits)
    return graphs


def test_an_armed_layer_carries_the_kernel_at_every_bucket(stub_kernels):
    """The property arming exists to provide, checked on the compiler's input.

    One trace, one graph, the fused custom call in it -- and the same graph
    serves the 16-token decode bucket and the 8192-token prefill one. Under the
    per-step predicate this was unreachable by construction: the surviving
    trace is the smallest bucket, and the smallest bucket is below every
    sensible threshold, so the census saw ``fused_ep=0`` in every serving arm.
    """
    graphs = _trace_across(_method(armed=True).apply_monolithic,
                           ACROSS_THE_BAND_EDGE,
                           args=(_layer(), ))

    assert len(graphs) == 1, (
        f"{len(graphs)} graphs across the band edge: something on the armed "
        "path is still branching on the token count, and vLLM will keep only "
        "the first graph")
    counts = moe_census.census(graphs[0].graph.nodes)
    assert counts["fused_ep"] == 1 and counts["gmm"] == 0


def test_a_declined_layer_carries_the_gmm_path_at_every_bucket(stub_kernels):
    """The control, and the other half of the bicondition the report checks."""
    graphs = _trace_across(_method(armed=False).apply_monolithic,
                           ACROSS_THE_BAND_EDGE,
                           args=(_layer(), ))

    assert len(graphs) == 1
    counts = moe_census.census(graphs[0].graph.nodes)
    assert counts["fused_ep"] == 0 and counts["gmm"] == 1


def test_the_census_labels_match_what_dynamo_emitted(stub_kernels):
    """Rot detector: the census matches on a name the op builder mints.

    If the op naming changes and the census is not updated, every arm after it
    reports ``fused_ep=0`` -- indistinguishable from the kernel not engaging,
    which is the exact confusion the census exists to remove.
    """
    graphs = _trace_across(_method(armed=True).apply_monolithic, (64, ),
                           args=(_layer(), ))
    targets = [str(node.target) for node in graphs[0].graph.nodes]
    assert any("fused_ep_moe_kernel" in target for target in targets), targets


# --------------------------------------------------------------------------
# The compile gate: what vLLM's compiler will accept from the armed path
# --------------------------------------------------------------------------


def test_the_armed_path_compiles_across_the_band_edge(stub_kernels):
    """The gate the last three serving arms did not have.

    The failure it catches is not a wrong answer, it is a dead engine:
    ``apply_monolithic`` asks the band about ``x.shape[0]``, dynamo specialises
    the dimension vLLM marked dynamic, and the whole compile dies with
    ``Constraints violated (L['input_ids'].size()[0], ...)`` inside
    ``capture_model``, on device, taking the boot with it. It cost a device
    job. Nothing on this path may read that number now, and this is what
    says so at every commit rather than at every boot.
    """
    _trace_across(_method(armed=True).apply_monolithic,
                  ACROSS_THE_BAND_EDGE,
                  args=(_layer(), ))


def test_the_declined_path_compiles_across_the_band_edge(stub_kernels):
    """The same, for the deployment the band turns away: still one trace."""
    _trace_across(_method(armed=False).apply_monolithic,
                  ACROSS_THE_BAND_EDGE,
                  args=(_layer(), ))


def test_the_compile_gate_can_see_the_predicate_that_was_removed():
    """NEGATIVE CONTROL: put the defect back and the gate must catch it.

    This is the deleted ``engages_for``, in shape: derive the node-wide count
    from the token dimension, compare it against the band, and -- the part that
    actually kills the engine -- **name the count in the reason it returns**.
    Interpolating a symbolic size into a string forces a concrete value out of
    it, which is the specialisation dynamo refuses.
    """

    def engages_for(x, router_logits):
        node_tokens = x.shape[0] * DP_SIZE
        if not fused_moe_ep.in_token_band(node_tokens):
            reason = f"{node_tokens} node-wide tokens outside the band"
        else:
            reason = None
        return x * 2 if reason is None else x

    with pytest.raises(Exception, match="specialized|Constraints violated"):
        _trace_across(engages_for, ACROSS_THE_BAND_EDGE)


def test_a_band_read_without_the_message_is_worse_than_a_crash():
    """NEGATIVE CONTROL 2, and why deleting only the f-string is not the fix.

    A bare comparison is expressible as a guard, so dynamo keeps the dimension
    dynamic and simply compiles a **second graph**. That looks like a fix and is
    not: vLLM marks the shapes dynamic on the first compile and then serves
    every later call through ``TorchCompileWithNoGuardsWrapper``, which dropped
    the guards -- so the second graph is never selected and the first bucket's
    answer silently becomes every bucket's. The loud version costs a boot; this
    version costs a campaign, which is what it did.
    """

    def reads_the_band(x, router_logits):
        return x * 2 if fused_moe_ep.in_token_band(x.shape[0] * DP_SIZE) else x

    graphs = _trace_across(reads_the_band, ACROSS_THE_BAND_EDGE)
    assert len(graphs) == 2, (
        f"{len(graphs)} graph(s): the band comparison stopped costing a "
        "recompile, so a per-step band may have become expressible -- check "
        "whether per-bucket specialisation landed, and re-read whether the "
        "guards are still dropped")


def test_the_traced_forward_survives_a_dynamic_shape(stub_kernels):
    """`fused_moe_ep` itself is still traced, so the rule still binds it.

    The bridge pads rows to the kernel's per-rank minimum and slices them per
    replica, all in Python arithmetic over ``hidden_states.shape[0]``. That
    arithmetic is fine -- ``torch.arange`` and a slice take a symbolic size --
    but a *branch* on it would specialise, and the graph count is what says
    which of the two it is.
    """
    layer = _layer()
    graphs = _trace_across(
        lambda x, logits: fused_moe_ep.fused_moe_ep(
            x,
            logits,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            replica_rank=None,
            ep_rank_ids=layer._ep_rank_ids,
            top_k=TOP_K,
            renormalize=True,
            activation="silu",
            scoring_fn="softmax",
            subc_quant_w1_sz=None,
            subc_quant_w2_sz=None,
            quant_act=True), ACROSS_THE_BAND_EDGE)
    assert len(graphs) == 1


def test_nothing_on_the_traced_path_formats_a_shape():
    """Structural twin of the two gates above, for the next diagnostic.

    A dynamic-shape trace catches it today; this catches it in the diff, and
    names the rule rather than the symptom: **the traced path may compute
    tensors and nothing else.** Anything that has to be said about a token
    count is said at weight load, where `out_of_band_reason` can format freely.
    """
    import inspect

    traced = (fused_moe_ep.fused_moe_ep, fused_moe_ep._padded_rows,
              fused_moe_ep._row_slice_index,
              fused_moe_ep._scatter_replica_rows, fused_moe_ep.op_cache_key,
              fused_moe_ep._get_op,
              fp8_mod.VllmFp8MoEMethodTPU.apply_monolithic)
    offenders = [
        fn.__name__ for fn in traced
        if any(bad in inspect.getsource(fn)
               for bad in ('f"{', "f'{", ".format(", "logger.", "print("))
    ]
    assert not offenders, (
        f"{offenders} format a value or log on the traced path; a SymInt "
        "formatted there specialises the graph and a logger call in a captured "
        "graph is refused outright. Put it in out_of_band_reason, which runs "
        "at weight load")


def test_the_band_helpers_are_not_reachable_from_the_forward():
    """The deleted per-step spelling must not come back by another name.

    Anything that reads the band is eager-only by construction now, so no
    function the forward can reach may call one. Checked structurally because
    the next person to want a per-step decision will reach for exactly these.
    """
    import inspect

    band_readers = ("in_token_band", "out_of_band_reason",
                    "max_engine_node_tokens", "_smem_reason")
    for fn in (fused_moe_ep.fused_moe_ep,
               fp8_mod.VllmFp8MoEMethodTPU.apply_monolithic):
        source = inspect.getsource(fn)
        called = [name for name in band_readers if f"{name}(" in source]
        assert not called, (
            f"{fn.__name__} calls {called}; the band is answered once, at "
            "weight load, and a call from here is the defect the compile gate "
            "above exists to catch")
