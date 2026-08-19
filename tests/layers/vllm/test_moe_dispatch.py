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
"""Numerics gate for routing before the DP dispatch (TPU_MOE_LOCAL_ROUTING).

The claim the flag rests on is that routing is a *per-row* function, so
deciding a rank's own rows before the all-gather and shipping the decision is
the same computation, not an approximation of it. These tests hold that claim
row by row and bit by bit, plus the pack/unpack round trip that carries it.
"""

import inspect
import pathlib
import re
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.vllm import moe_dispatch, moe_routing, token_padding

EXPERTS = 512
TOPK = 10


def _quant_method(*, monolithic=True, forwards_input_ids=False, unpacks=True):
    """A stand-in for the six TPU MoE methods, as the wrapper reads them.

    ``forwards_input_ids_to_route`` is the capability that decides whether a
    layer's ``hash_indices_table`` can ever be used: only a method that passes
    ``input_ids`` into ``moe_routing.route`` reaches the hash branch.

    ``unpacks_routing_blob`` is the receiver's own declaration that its
    ``apply_monolithic`` routes through ``moe_routing.route``, which is what
    unpacks the decision. A monolithic method from another MoE stack does not
    declare it and must keep the raw logits.
    """
    return SimpleNamespace(is_monolithic=monolithic,
                           forwards_input_ids_to_route=forwards_input_ids,
                           unpacks_routing_blob=unpacks)


def _layer(topk=TOPK, **kwargs):
    attrs = dict(use_grouped_topk=False,
                 num_expert_group=None,
                 topk_group=None,
                 renormalize=True,
                 scoring_func="softmax",
                 e_score_correction_bias=None,
                 routed_scaling_factor=1.0,
                 hash_indices_table=None,
                 custom_routing_function=None,
                 quant_method=_quant_method(),
                 moe_config=SimpleNamespace(experts_per_token=topk))
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _cpu_rng(seed: int = 0) -> torch.Generator:
    """One CPU generator, instead of seeding every device.

    ``torch.manual_seed`` also seeds each registered device backend, so with
    torch_tpu loaded it calls ``torch.tpu.manual_seed_all`` -- which builds the
    PjRt client inside this process and takes /tmp/libtpu_lockfile away from
    every isolated-subprocess TPU test that runs after this file. Nothing here
    needs a device, so nothing here touches one.
    """
    return torch.Generator().manual_seed(seed)


def test_this_file_never_seeds_a_device_backend():
    """The rule above, pinned, because breaking it fails somewhere else.

    ``torch.manual_seed`` seeds every registered device backend. With
    torch_tpu loaded that reaches ``torch.tpu.manual_seed_all``, which builds
    the PjRt client in this process and holds /tmp/libtpu_lockfile -- and what
    then fails is not this file but the next isolated-subprocess TPU test in
    the session, with a lockfile error that reads like a dirty machine.
    """
    source = pathlib.Path(__file__).read_text()

    # Spelled as patterns so this check does not match itself.
    assert not re.search(r"torch\.manual_seed\(", source)
    assert not re.search(r"torch\.seed\(", source)


def _logits(rows, seed=0):
    return (torch.randn(rows, EXPERTS, generator=_cpu_rng(seed)) * 2.0).to(
        torch.bfloat16)


# The row-wise premise has to hold for every branch of the router, not just the
# default one: `moe_routing.select_experts` picks a different path per layer
# config, and "local routing == global routing" is a claim about all of them.
#
# Deferred coverage: on a tree that also carries the sort-free router
# (TPU_MOE_ROUTER_TOPK, PR #479) this axis must be crossed with both top-k
# implementations, because that one is *not* guaranteed tie-identical between
# [T_local, E] and [T_global, E] under an unstable sort. That env var does not
# exist at this base, so it cannot be parameterised here; it belongs to the
# merged tree's numerics gate.
ROUTER_VARIANTS = {
    "softmax": {},
    "sigmoid_bias":
    dict(scoring_func="sigmoid",
         e_score_correction_bias=torch.linspace(-1.0, 1.0, EXPERTS)),
    "sqrtsoftplus_scaled":
    dict(scoring_func="sqrtsoftplus", routed_scaling_factor=2.5),
    "no_renormalize":
    dict(renormalize=False),
    "grouped":
    dict(use_grouped_topk=True, num_expert_group=8, topk_group=3),
}


@pytest.fixture(autouse=True)
def grouped_topk_without_its_own_compile(monkeypatch):
    """Strip the ``@torch.compile`` off vLLM's grouped_topk for this module.

    It is declared ``@torch.compile(dynamic=True,
    backend=current_platform.simple_compile_backend)``, which on this platform
    is ``"tpu"`` -- a backend that refuses dynamic shapes. Calling it outside a
    compiled region therefore raises for reasons that have nothing to do with
    routing. In serving it is reached from inside the MoE layer's own graph and
    Dynamo inlines it; ``__wrapped__`` is that same function body.

    Whether the decorator survives depends on how vLLM was built, so unwrap
    when there is something to unwrap and leave it alone otherwise: these
    tests only need a callable that accepts eager tensors.
    """
    from vllm.model_executor.layers.fused_moe.router import \
        grouped_topk_router as g
    inner = getattr(g.grouped_topk, "__wrapped__", None)
    if inner is not None:
        monkeypatch.setattr(g, "grouped_topk", inner)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 256])
def test_pack_unpack_round_trip_is_bit_exact(rows, dtype):
    rng = _cpu_rng(rows)
    ids = torch.randint(0,
                        EXPERTS, (rows, TOPK),
                        dtype=torch.int32,
                        generator=rng)
    weights = torch.rand(rows, TOPK, generator=rng).to(dtype)
    blob = moe_dispatch.pack_routing(ids, weights)

    assert blob.dtype == torch.int32 and blob.shape == (rows * 2 * TOPK, )
    got_w, got_i = moe_dispatch.unpack_routing(blob, TOPK, dtype)
    assert torch.equal(got_i, ids)
    assert torch.equal(got_w, weights)


def test_blob_is_distinguishable_from_router_logits():
    assert not moe_dispatch.is_routing_blob(_logits(4))
    assert not moe_dispatch.is_routing_blob(_logits(4).float())
    ids = torch.zeros(4, TOPK, dtype=torch.int32)
    assert moe_dispatch.is_routing_blob(
        moe_dispatch.pack_routing(ids, torch.zeros(4, TOPK)))


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(4, 2 * TOPK,
                    dtype=torch.int32),  # still 2-D: the old format
        torch.zeros(4 * 2 * TOPK + 3,
                    dtype=torch.int32),  # not a whole number of rows
    ])
def test_unpack_rejects_anything_that_is_not_a_flat_blob(bad):
    with pytest.raises(ValueError, match="routing blob"):
        moe_dispatch.unpack_routing(bad, TOPK, torch.bfloat16)


def _chunked_vs_global(layer, rows, dp_size, seed):
    """Route `rows` at once, and in `dp_size` chunks. Returns both decisions."""
    logits = _logits(rows, seed=seed)
    hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)

    chunk = rows // dp_size
    local_w, local_i = [], []
    for r in range(dp_size):
        sl = slice(r * chunk, (r + 1) * chunk)
        w, i = moe_routing.route(layer, hidden[sl], logits[sl])
        local_w.append(w)
        local_i.append(i)
    return torch.cat(local_w), torch.cat(local_i), ref_w, ref_i


@pytest.mark.parametrize("variant", sorted(ROUTER_VARIANTS))
@pytest.mark.parametrize("dp_size", [2, 4, 8])
def test_routing_is_row_wise_so_a_rank_can_decide_its_own_rows(
        dp_size, variant):
    """The premise of the flag: the router is a *row-wise* function.

    Deciding a rank's own rows and concatenating must equal deciding all rows
    at once, for every branch of this tree's router. What this does **not**
    cover, and a reader will assume it does: **tie policy**. `torch.topk` on
    CPU is index-ordered, so chunking cannot change a tie here however tied
    the input is (measured: it passes on input that is 100% tied at the
    boundary, while a different tie rule moves 75.4% of those rows). Whether
    the *device* sort is row-count invariant is a property of the emitted sort
    network and is decided by `tests/layers/vllm/test_topk_shape_invariance.py`
    on a chip.
    """
    layer = _layer(**ROUTER_VARIANTS[variant])
    got_w, got_i, ref_w, ref_i = _chunked_vs_global(layer,
                                                    64 * dp_size,
                                                    dp_size,
                                                    seed=dp_size)
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)


@pytest.mark.parametrize("dp_size", [2, 4, 8])
def test_the_row_wise_check_rejects_a_router_that_is_not_row_wise(
        monkeypatch, dp_size):
    """Negative control for the test above: it must be able to fail.

    A router with a genuine cross-row dependency -- here, centring the scores
    on the batch's own column mean -- is exactly what would make routing before
    the dispatch wrong, and the comparison has to catch it.

    This router is synthetic because the tree's one real whole-batch statistic
    is inert for *selection*: `_uniform_random_routing` shifts its noise by
    `router_logits.float().mean()`, a constant per batch that cannot reorder an
    argsort. What it does change is the RNG draw's shape, so the flag on and
    the flag off pick different experts under
    `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` -- documented next to
    the flag in `envs.py`, and never set in serving.
    """

    def not_row_wise(hidden_states,
                     router_logits,
                     *,
                     topk,
                     renormalize,
                     scoring_fn,
                     layer=None,
                     input_ids=None):
        scores = router_logits.float().softmax(dim=-1)
        scores = scores - scores.mean(dim=0, keepdim=True)
        w, i = torch.topk(scores, k=topk, dim=-1)
        return w.to(hidden_states.dtype), i.to(torch.int32)

    monkeypatch.setattr(moe_routing, "select_experts", not_row_wise)
    got_w, got_i, ref_w, ref_i = _chunked_vs_global(_layer(),
                                                    64 * dp_size,
                                                    dp_size,
                                                    seed=dp_size)
    assert not torch.equal(got_i, ref_i), (
        "a router that reads the whole batch produced the same decision "
        "chunked as whole -- the row-wise test cannot fail")


def test_the_row_wise_check_is_blind_to_tie_policy_on_cpu():
    """Name the blind spot in an executable form, so nobody re-derives it.

    The input here is dense in boundary ties (the bf16 gate output is what
    makes them; f32 logits give none). The row-wise comparison passes anyway,
    because CPU `torch.topk` breaks ties by index and that is a per-row rule.
    On the device the sort is `kUnstable` with a key-only comparator, so the
    question is real there and is answered by the device test named in
    `test_routing_is_row_wise_...`'s docstring.
    """
    logits = _logits(4096, seed=17)
    scores = logits.float().softmax(dim=-1)
    srt = torch.sort(scores, dim=-1, descending=True).values
    tie_rate = (srt[:, TOPK - 1] == srt[:, TOPK]).float().mean().item()
    assert tie_rate > 0.05, (
        f"only {tie_rate:.2%} of rows are tied at the top-k boundary; this "
        "test is not exercising the case it exists to document")
    assert (logits.float().softmax(dim=-1) == scores).all()

    got_w, got_i, ref_w, ref_i = _chunked_vs_global(_layer(), 4096, 4, seed=17)
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


def test_route_checks_for_a_blob_before_any_other_branch(monkeypatch):
    """The blob check must precede every other branch in ``route()``.

    ``route()`` is a decision point two changes want to edit at once: this one
    inserts the unpack at the top, and the sort-free router rewrites the
    selection below it. If a merge ever puts anything ahead of the blob check,
    that something is handed an int32 blob and produces silent garbage rather
    than an error -- so pin the *order*, not just the behaviour.

    Every other path out of ``route()`` is poisoned; reaching one fails.
    """
    logits = _logits(8, seed=11)
    hidden = torch.zeros(8, 8, dtype=torch.bfloat16)
    ref_w, ref_i = moe_routing.route(_layer(), hidden, logits)
    blob = moe_dispatch.pack_routing(ref_i, ref_w)

    def _poison(*args, **kwargs):
        raise AssertionError("a non-blob routing branch ran on a packed blob")

    monkeypatch.setattr(moe_routing, "maybe_simulate_routing", _poison)
    monkeypatch.setattr(moe_routing, "select_experts", _poison)
    got_w, got_i = moe_routing.route(_layer(custom_routing_function=_poison),
                                     hidden, blob)
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


@pytest.mark.parametrize("variant", sorted(ROUTER_VARIANTS))
def test_route_unpacks_a_dispatched_blob(variant):
    """What the far side of the collective does must match local routing."""
    logits = _logits(32, seed=3)
    hidden = torch.zeros(32, 8, dtype=torch.bfloat16)
    layer = _layer(**ROUTER_VARIANTS[variant])

    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    blob = moe_dispatch.pack_routing(ref_i, ref_w)
    got_w, got_i = moe_routing.route(layer, hidden, blob)

    assert torch.equal(got_i, ref_i) and got_i.dtype == torch.int32
    assert torch.equal(got_w, ref_w) and got_w.dtype == hidden.dtype


def test_route_and_pack_round_trips_through_the_dispatch():
    logits = _logits(16, seed=5)
    hidden = torch.zeros(16, 8, dtype=torch.bfloat16)
    layer = _layer()

    blob = moe_dispatch.route_and_pack(layer, hidden, logits)
    got_w, got_i = moe_routing.route(layer, hidden, blob)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


# Every way a layer can decline the pack. Each one must fall back to the
# upstream dispatch -- the raw logits, routed by the receiving rank -- because
# there is no flag left to unset for a model that hits one.
UNSUPPORTED_LAYERS = {
    "hash_routing":
    dict(hash_indices_table=torch.zeros(4, TOPK, dtype=torch.int32),
         quant_method=_quant_method(forwards_input_ids=True)),
    "custom_routing_function":
    dict(custom_routing_function=lambda **kw: None),
    "receiver_does_not_unpack":
    dict(quant_method=_quant_method(unpacks=False)),
}
UNSUPPORTED = sorted(UNSUPPORTED_LAYERS) + ["routing_simulation"]


def _unsupported_layer(name, monkeypatch):
    """A layer (and process state) that cannot route before the dispatch."""
    if name == "routing_simulation":
        # Set at import from VLLM_MOE_ROUTING_SIMULATION_STRATEGY; the object
        # itself is never called on this path.
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
        return _layer()
    return _layer(**UNSUPPORTED_LAYERS[name])


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_route_and_pack_falls_back_to_the_upstream_dispatch(case, monkeypatch):
    """An unpackable layer keeps the logits, byte for byte and object for
    object, so the receiving rank routes exactly as it does upstream."""
    layer = _unsupported_layer(case, monkeypatch)
    logits = _logits(4)
    assert moe_dispatch.unsupported_reason(layer) is not None

    out = moe_dispatch.route_and_pack(layer, torch.zeros(4, 8), logits)

    assert out is logits, "the fallback must not touch router_logits"
    assert not moe_dispatch.is_routing_blob(out)


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_the_fallback_routes_the_way_the_control_does(case, monkeypatch):
    """Not just "it runs": the decision has to be the unpatched one.

    ``route()`` on what the fallback dispatched is what the receiving rank
    computes; comparing it against ``route()`` on the raw logits is the same
    comparison the flag's numerics gate makes, for the layers that opt out.
    """
    layer = _unsupported_layer(case, monkeypatch)
    if case == "custom_routing_function":
        # The poison lambda above cannot route; give it a real row-wise one.
        layer.custom_routing_function = (
            lambda hidden_states, gating_output, topk, renormalize: torch.topk(
                gating_output.float(), k=topk, dim=-1))
    if case == "routing_simulation":
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
        monkeypatch.setattr(moe_routing, "_strategy_name", "uniform_random")

    logits = _logits(16, seed=41)
    hidden = torch.zeros(16, 8, dtype=torch.bfloat16)
    # The simulation strategies draw from the global CPU generator, so the two
    # routes have to start from the same state. Save and restore it rather than
    # re-seeding: torch.manual_seed would seed the device backends too.
    before = torch.random.get_rng_state()
    got_w, got_i = moe_routing.route(
        layer, hidden, moe_dispatch.route_and_pack(layer, hidden, logits))
    torch.random.set_rng_state(before)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)

    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


def _record_info(monkeypatch, module):
    """Capture ``module.logger.info`` calls; vLLM's loggers do not propagate."""
    said = []
    monkeypatch.setattr(
        module, "logger",
        SimpleNamespace(info=lambda msg, *args: said.append(msg % args)))
    return said


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_the_fallback_neither_breaks_the_graph_nor_logs_under_compile(
        case, monkeypatch):
    """The fallback runs inside the compiled MoE forward, so it may not log.

    Dynamo rejects ``logging.Logger`` calls under ``fullgraph=True`` outright
    ("logging.Logger method not supported for non-export cases"), which is
    also torch_tpu's objection. ``_log_fallback_once`` is therefore guarded on
    ``torch.compiler.is_compiling()``; this is the test that keeps it guarded,
    because losing the guard turns every unpackable model into a startup
    crash. The eager path asserts the diagnostic is not simply dead.
    """
    import torch._dynamo as dynamo

    layer = _unsupported_layer(case, monkeypatch)
    monkeypatch.setattr(moe_dispatch, "_FALLBACKS_LOGGED", set())
    said = _record_info(monkeypatch, moe_dispatch)
    logits = _logits(8, seed=8)
    hidden = torch.zeros(8, 8, dtype=torch.bfloat16)

    dynamo.reset()
    explained = dynamo.explain(moe_dispatch.route_and_pack)(layer, hidden,
                                                            logits)
    assert explained.graph_break_count == 0, explained.break_reasons
    dynamo.reset()
    # backend="eager": what is under test is a Dynamo property (no graph
    # break, no logger call under fullgraph), and the platform's default
    # backend is the TPU one, which refuses the CPU tensors these tests use.
    torch.compile(moe_dispatch.route_and_pack,
                  backend="eager",
                  fullgraph=True,
                  dynamic=False)(layer, hidden, logits)
    assert said == [], f"the compiled path logged: {said}"

    moe_dispatch.route_and_pack(layer, hidden, logits)
    moe_dispatch.route_and_pack(layer, hidden, logits)
    assert len(said) == 1, said
    assert "keeps the upstream path" in said[0]


def test_a_hash_table_the_far_side_cannot_reach_still_routes_locally():
    """The DeepSeek-V4-on-fp8 case: table present, hash branch unreachable.

    ``select_experts`` takes the hash branch only when ``input_ids is not
    None``, and only a method that forwards ``input_ids`` into ``route()``
    supplies it -- mxfp4, alone. The same checkpoint on the fp8 method routes
    from the gate today, so falling back for it would give up the saving for a
    reason that is not true of it.
    """
    layer = _layer(hash_indices_table=torch.zeros(4, TOPK, dtype=torch.int32),
                   quant_method=_quant_method(forwards_input_ids=False))
    assert moe_dispatch.unsupported_reason(layer) is None

    logits = _logits(16, seed=13)
    hidden = torch.zeros(16, 8, dtype=torch.bfloat16)
    got_w, got_i = moe_routing.route(
        layer, hidden, moe_dispatch.route_and_pack(layer, hidden, logits))
    # The reference is what fp8's apply_monolithic calls: route() with no
    # input_ids, on the gathered rows.
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


# The six quant methods that reach moe_routing.route, and whether each zeroes
# padded routing weights with the flag off. Only two do, and one of those is
# behind its own flag. Routing before the dispatch must not change this: a
# performance flag that starts zeroing rows four methods leave computed is a
# semantic change, and it would break the claim that the decision is
# bit-identical.
ROUTE_CALLERS = {
    "quantization.fp8":
    "TPU_MOE_SKIP_PADDED_TOKENS",
    "quantization.compressed_tensors.compressed_tensors_moe"
    ".compressed_tensors_moe_w4an_mxfp4":
    "",
    "quantization.mxfp4":
    None,
    "quantization.nvfp4":
    None,
    "quantization.unquantized":
    None,
    "quantization.compressed_tensors.compressed_tensors_moe"
    ".compressed_tensors_moe_w4a16":
    None,
}


def _zeroing_guard(module_name):
    """`None` if the module never zeroes, else the env flag guarding it ("" =
    unguarded). Read out of the AST, so it tracks the source, not a memory."""
    import ast
    import importlib

    mod = importlib.import_module(f"vllm_torchtpu.layers.vllm.{module_name}")
    tree = ast.parse(inspect.getsource(mod))
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "zero_routing_weights_for_padding"):
            continue
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If) and isinstance(cur.test, ast.Attribute):
                return cur.test.attr
            if isinstance(cur, (ast.FunctionDef, ast.Module)):
                break
        return ""
    return None


@pytest.mark.parametrize("module_name", sorted(ROUTE_CALLERS))
def test_the_flag_does_not_change_which_methods_zero_padded_rows(module_name):
    """Flag on zeroes exactly what flag off zeroes, for every route() caller.

    The flag cannot differ, because `route_and_pack` does not mask at all and
    `token_padding` is byte-identical to the base -- but that is exactly the
    invariant worth pinning, since the obvious "optimisation" of masking on the
    local rows silently zeroes rows four of these six leave computed.
    """
    assert _zeroing_guard(module_name) == ROUTE_CALLERS[module_name]


def _route_call_forwards_input_ids(module_name):
    """Does this module's ``route()`` call pass ``input_ids``? From the AST."""
    import ast
    import importlib

    mod = importlib.import_module(f"vllm_torchtpu.layers.vllm.{module_name}")
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "route" and any(kw.arg == "input_ids"
                                              for kw in node.keywords)
        for node in ast.walk(ast.parse(inspect.getsource(mod))))


def test_exactly_one_quant_method_can_reach_hash_routing():
    """Anchors the audit below: it must find something, and only this.

    `unsupported_reason` refuses a layer carrying `hash_indices_table` only
    where the hash branch is reachable, i.e. where the receiving method hands
    `route()` the `input_ids` that branch is keyed on.
    """
    forwarding = sorted(m for m in ROUTE_CALLERS
                        if _route_call_forwards_input_ids(m))
    assert forwarding == ["quantization.mxfp4"], forwarding


@pytest.mark.parametrize("module_name", sorted(ROUTE_CALLERS))
def test_the_hash_refusal_capability_bit_matches_the_route_call(module_name):
    """`forwards_input_ids_to_route` is a claim about the method's own call.

    The predicate reads the class attribute; the attribute is only true if the
    method really forwards `input_ids`. Check the declaration against the
    source rather than trusting that the two were kept in step.
    """
    import importlib

    mod = importlib.import_module(f"vllm_torchtpu.layers.vllm.{module_name}")
    declared = sorted(
        name for name, cls in inspect.getmembers(mod, inspect.isclass)
        if cls.__module__ == mod.__name__
        and getattr(cls, "forwards_input_ids_to_route", False))
    assert bool(declared) == _route_call_forwards_input_ids(module_name), (
        f"{module_name}: classes declaring forwards_input_ids_to_route are "
        f"{declared}, which disagrees with its moe_routing.route() call")


def _quantization_modules():
    """Every module of the TPU quantization package, imported."""
    import importlib
    import pkgutil

    import vllm_torchtpu.layers.vllm.quantization as pkg
    for info in pkgutil.walk_packages(pkg.__path__, f"{pkg.__name__}."):
        yield importlib.import_module(info.name)


def _calls_moe_routing_route(mod) -> bool:
    import ast

    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "route" and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "moe_routing"
        for node in ast.walk(ast.parse(inspect.getsource(mod))))


def test_the_unpack_capability_bit_matches_the_route_call():
    """`unpacks_routing_blob` is a claim that the method unpacks the decision.

    It is what the dispatch packs *for*: `moe_routing.route` is the only
    unpacker, so a method that does not go through it must be handed logits.
    Discovered rather than listed -- a new quant method that declares the bit
    without calling `route()` would read the packed int32 as logits, which is
    the one way this change can be silently wrong rather than merely slower.
    """
    declaring, calling = set(), set()
    for mod in _quantization_modules():
        if _calls_moe_routing_route(mod):
            calling.add(mod.__name__)
        if any("unpacks_routing_blob" in vars(cls)
               for _, cls in inspect.getmembers(mod, inspect.isclass)
               if cls.__module__ == mod.__name__):
            declaring.add(mod.__name__)

    assert calling, "the source scan found no route() call at all"
    assert declaring == calling, (
        f"declared but does not route(): {sorted(declaring - calling)}; "
        f"routes() but does not declare: {sorted(calling - declaring)}")


def test_a_receiver_that_does_not_unpack_is_never_packed_for():
    """Negative control for the bit: the check has to be able to fire.

    A monolithic method from another MoE stack -- tpu-inference ships seven,
    and a stacked install can put one on the far side -- reads whatever the
    dispatch carried as router logits.
    """
    foreign = _layer(quant_method=SimpleNamespace(is_monolithic=True))
    reason = moe_dispatch.unsupported_reason(foreign)
    assert reason is not None and "unpacks_routing_blob" in reason
    assert moe_dispatch.unsupported_reason(_layer()) is None


def test_route_and_pack_does_not_mask_padded_rows():
    """The decision crosses the wire unmasked; masking stays downstream."""
    src = inspect.getsource(moe_dispatch)
    assert "zero_routing_weights_for_padding" not in src
    assert "zero_local_routing_weights_for_padding" not in src

    state = token_padding.TokenPaddingState.create(16, torch.device("cpu"))
    state.update(local_valid_tokens=11, local_padded_tokens=16)
    token_padding.set_padding_state(state)
    try:
        layer = _layer()
        logits = _logits(16, seed=7)
        hidden = torch.zeros(16, 8, dtype=torch.bfloat16)
        w, i = moe_routing.route(
            layer, hidden, moe_dispatch.route_and_pack(layer, hidden, logits))
        ref_w, ref_i = moe_routing.route(layer, hidden, logits)
        # The padded rows are still live in the blob: nothing was zeroed.
        assert torch.equal(w, ref_w) and torch.equal(i, ref_i)
        assert (w[11:] != 0).any(), "route_and_pack masked the padded rows"
    finally:
        token_padding.set_padding_state(None)


def test_token_padding_is_untouched_by_this_change():
    """The module the flag used to edit is byte-identical to the base.

    Cheaper and stronger than testing the behaviour it did not change: if
    nobody edits the file, the gathered-mask path cannot have moved.
    """
    src = inspect.getsource(token_padding)
    assert "TPU_MOE_LOCAL_ROUTING" not in src
    assert "_ROUTED_BEFORE_DISPATCH" not in src
    assert not hasattr(token_padding, "zero_local_routing_weights_for_padding")


@pytest.mark.parametrize("flag_on", [False, True])
def test_the_gathered_padding_mask_runs_identically_with_the_flag_on(
        monkeypatch, flag_on):
    """The collective and the rows it zeroes do not depend on the flag."""
    from vllm.distributed import parallel_state

    dp_size, local = 4, 16
    gathered = []

    class _DPGroup:
        world_size = dp_size

        def all_gather(self, tensor, dim=0):
            gathered.append(tensor)
            return tensor.repeat(dp_size)

    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: _DPGroup())
    import vllm_torchtpu.envs as tpu_envs
    monkeypatch.setattr(
        tpu_envs, "environment_variables",
        dict(tpu_envs.environment_variables,
             TPU_MOE_LOCAL_ROUTING=lambda: flag_on))

    state = token_padding.TokenPaddingState.create(local, torch.device("cpu"))
    state.update(local_valid_tokens=11, local_padded_tokens=local)
    token_padding.set_padding_state(state)
    try:
        ids = torch.zeros(dp_size * local, TOPK, dtype=torch.int32)
        weights = torch.ones(dp_size * local, TOPK)
        _, got = token_padding.zero_routing_weights_for_padding(
            ids, weights.clone())
    finally:
        token_padding.set_padding_state(None)

    assert len(gathered) == 1, "the gathered-mask collective must still run"
    mask = state.get_local_padding_mask(local).repeat(dp_size)
    assert torch.equal(got, weights * (~mask)[:, None])


def _upstream_maybe_dispatch():
    """``MoERunner._maybe_dispatch`` as installed, asserted to be upstream's."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    fn = mr.MoERunner.__dict__["_maybe_dispatch"]
    assert fn.__module__.startswith("vllm."), (
        f"expected unpatched upstream code to start from, got {fn!r} from "
        f"{fn.__module__}")
    return fn


def _install_patch(monkeypatch, enabled, ep_kernel=False):
    """Install (or decline to install) the _maybe_dispatch patch, undoably."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu
    monkeypatch.setattr(
        vllm_torchtpu.envs, "environment_variables",
        dict(vllm_torchtpu.envs.environment_variables,
             TPU_MOE_LOCAL_ROUTING=lambda: enabled,
             USE_MOE_EP_KERNEL=lambda: ep_kernel))
    monkeypatch.setattr(mr.MoERunner,
                        "_maybe_dispatch",
                        mr.MoERunner._maybe_dispatch,
                        raising=False)
    monkeypatch.delattr(mr.MoERunner,
                        "_tpu_local_routing_patch",
                        raising=False)
    vllm_torchtpu._patch_moe_local_routing_dispatch()
    return mr


def test_local_routing_is_off_by_default():
    """With no env set the flag reads False, and only startup reads it.

    The one read site runs once, at patch time. Neither the codec nor the
    routing path reads it, so there is no module-scope cache whose value
    depends on import order relative to the worker's environment.
    """
    import vllm_torchtpu
    import vllm_torchtpu.envs as envs
    assert envs.TPU_MOE_LOCAL_ROUTING is False
    readers = [
        m for m, src in
        (("moe_dispatch", inspect.getsource(moe_dispatch)),
         ("moe_routing", inspect.getsource(moe_routing)),
         ("token_padding", inspect.getsource(token_padding)),
         ("_patch_moe_local_routing_dispatch",
          inspect.getsource(vllm_torchtpu._patch_moe_local_routing_dispatch)))
        if "envs.TPU_MOE_LOCAL_ROUTING" in src
    ]
    assert readers == ["_patch_moe_local_routing_dispatch"], readers


@pytest.mark.parametrize("flag_on", [False, True])
def test_maybe_dispatch_is_wrapped_only_when_the_flag_is_on(
        monkeypatch, flag_on):
    """Default off means the *upstream function object* is still installed.

    Identity, not a sentinel attribute: a wrapper that forgot to set the
    sentinel would pass an attribute check and still change behaviour. The
    ``flag_on`` case is the negative control -- an assertion about the flag-off
    path that also holds with the flag on is not evidence of anything.
    """
    before = _upstream_maybe_dispatch()
    mr = _install_patch(monkeypatch, enabled=flag_on)
    after = mr.MoERunner.__dict__["_maybe_dispatch"]

    if flag_on:
        assert after is not before
        assert after.__module__ == "vllm_torchtpu"
        assert mr.MoERunner._tpu_local_routing_patch is True
    else:
        assert after is before
        assert not getattr(mr.MoERunner, "_tpu_local_routing_patch", False)


def test_patched_dispatch_sends_the_blob_not_the_logits(monkeypatch):
    """End to end through the real MoERunner method: the collective sees the
    packed decision, and nothing else about the call changes."""
    mr = _install_patch(monkeypatch, enabled=True)
    assert mr.MoERunner._tpu_local_routing_patch

    seen = {}

    class _Group:

        def dispatch_router_logits(self, hidden, logits, is_sp):
            seen["logits"] = logits
            seen["hidden"] = hidden
            return hidden, logits

    monkeypatch.setattr(mr, "get_ep_group", lambda: _Group())

    layer = _layer()
    layer.quant_method = _quant_method()

    class _Runner:
        do_naive_dispatch_combine = True
        routed_experts = layer
        moe_config = SimpleNamespace(is_sequence_parallel=False, pcp_size=1)

    runner = _Runner()

    logits = _logits(24, seed=9)
    hidden = torch.zeros(24, 8, dtype=torch.bfloat16)
    out_hidden, out_logits = mr.MoERunner._maybe_dispatch(
        runner, hidden, logits)

    assert seen["logits"].dtype == torch.int32, "raw logits reached the wire"
    assert seen["logits"].shape == (24 * 2 * TOPK, ), (
        "the blob must reach the collective rank-1: as [T, 2K] a downstream "
        "Mosaic layout constraint propagates back into the all-gather and "
        "pads 20 lanes to 128")
    assert seen[
        "hidden"] is hidden, "hidden states must pass through untouched"
    # And the far side reconstructs exactly what global routing would give.
    got_w, got_i = moe_routing.route(layer, out_hidden, out_logits)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_i, ref_i) and torch.equal(got_w, ref_w)


@pytest.mark.parametrize("guard",
                         ["do_naive_dispatch_combine", "is_monolithic"])
def test_the_wrapper_leaves_router_logits_alone_when_a_guard_is_false(
        monkeypatch, guard):
    """The negative controls for the predicate, one term at a time.

    ``do_naive_dispatch_combine`` False is what ``dp_size == 1`` looks like
    from inside the wrapper -- 10 of the 14 points on the InferenceX sweep --
    and upstream's body must then run on the raw logits exactly as it does
    unpatched. ``is_monolithic`` False is the modular path, whose router reads
    ``router_logits`` *after* the dispatch and would be handed an int32 blob.

    Object identity, not equality: a wrapper that packed and unpacked around
    the collective would compare equal and still have cost the collective its
    layout.
    """
    mr = _install_patch(monkeypatch, enabled=True)
    assert mr.MoERunner._tpu_local_routing_patch, (
        "the wrapper is not installed, so this negative control would pass "
        "against unpatched upstream code")

    def _refuse_to_pack(*args, **kwargs):
        raise AssertionError(f"route_and_pack ran with {guard} False")

    monkeypatch.setattr(moe_dispatch, "route_and_pack", _refuse_to_pack)

    seen = {}

    class _Group:

        def dispatch_router_logits(self, hidden, logits, is_sp):
            seen["logits"] = logits
            return hidden, logits

    monkeypatch.setattr(mr, "get_ep_group", lambda: _Group())

    layer = _layer()
    layer.quant_method = _quant_method(monolithic=guard != "is_monolithic")

    class _Runner:
        do_naive_dispatch_combine = guard != "do_naive_dispatch_combine"
        routed_experts = layer
        moe_config = SimpleNamespace(is_sequence_parallel=False, pcp_size=1)

    logits = _logits(24, seed=9)
    hidden = torch.zeros(24, 8, dtype=torch.bfloat16)
    out_hidden, out_logits = mr.MoERunner._maybe_dispatch(
        _Runner(), hidden, logits)

    assert out_logits is logits, "router_logits did not survive the wrapper"
    assert out_hidden is hidden
    if guard == "do_naive_dispatch_combine":
        assert not seen, "upstream dispatches nothing when the term is False"
    else:
        assert seen["logits"] is logits, (
            "the modular path's router must receive the raw logits")


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_the_wrapper_dispatches_raw_logits_for_a_layer_that_cannot_pack(
        case, monkeypatch):
    """The fallback, through the real MoERunner method rather than the codec.

    Object identity again: what reaches the collective must be the very tensor
    upstream would have handed it, so the receiving rank's routing is not
    merely equal to the unpatched decision but produced by the unpatched code.
    """
    mr = _install_patch(monkeypatch, enabled=True)
    assert mr.MoERunner._tpu_local_routing_patch, (
        "the wrapper is not installed, so this would pass against unpatched "
        "upstream code")
    layer = _unsupported_layer(case, monkeypatch)

    seen = {}

    class _Group:

        def dispatch_router_logits(self, hidden, logits, is_sp):
            seen["logits"] = logits
            return hidden, logits

    monkeypatch.setattr(mr, "get_ep_group", lambda: _Group())

    class _Runner:
        do_naive_dispatch_combine = True
        routed_experts = layer
        moe_config = SimpleNamespace(is_sequence_parallel=False, pcp_size=1)

    logits = _logits(24, seed=9)
    hidden = torch.zeros(24, 8, dtype=torch.bfloat16)
    out_hidden, out_logits = mr.MoERunner._maybe_dispatch(
        _Runner(), hidden, logits)

    assert seen["logits"] is logits, "the collective saw something repacked"
    assert out_logits is logits and out_hidden is hidden


def test_the_fused_ep_kernel_keeps_upstreams_dispatch(monkeypatch):
    """``USE_MOE_EP_KERNEL=1`` declines the patch; it does not refuse to run.

    The kernel that reads it lives in tpu-inference
    (``tpu_inference/layers/vllm/interface/moe.py``) and scores the raw logits
    itself, so it cannot be paired with a packed decision. Leaving
    ``_maybe_dispatch`` alone is exactly what that configuration needs, and it
    is the only outcome available now that there is no flag to unset: raising
    would make a working fused-EP deployment unstartable.
    """
    import vllm_torchtpu

    said = _record_info(monkeypatch, vllm_torchtpu)
    before = _upstream_maybe_dispatch()
    mr = _install_patch(monkeypatch, enabled=True, ep_kernel=True)

    assert mr.MoERunner.__dict__["_maybe_dispatch"] is before, (
        "the patch was installed under the fused EP kernel")
    assert not getattr(mr.MoERunner, "_tpu_local_routing_patch", False)
    assert len(said) == 1 and "USE_MOE_EP_KERNEL" in said[0], said


def test_the_patch_chain_installs_the_local_routing_patch():
    """Every other test calls the installer directly; this one checks it runs.

    Dropping the call from ``apply_tpu_patches`` would leave the whole feature
    uninstalled with the rest of this file still green.
    """
    import ast

    from vllm_torchtpu.platforms import tpu_platform

    src = textwrap.dedent(inspect.getsource(tpu_platform.apply_tpu_patches))
    # The chain calls some patches by bare name and others through the module
    # object (`tpu_plugin.foo()`), so match on the attribute too -- what this
    # test is protecting is that the installer is reached, not how it is spelt.
    called = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    assert "_patch_moe_local_routing_dispatch" in called, sorted(called)


@pytest.mark.parametrize("variant", sorted(ROUTER_VARIANTS))
def test_pack_and_unpack_compile_fullgraph(variant):
    """Both halves run inside the region vLLM compiles with Dynamo.

    The bitcast in particular (`Tensor.view(torch.int32)`) is the kind of op
    that traces fine eagerly and breaks the graph under `torch.compile`.
    """
    torch._dynamo.reset()
    layer = _layer(**ROUTER_VARIANTS[variant])
    logits = _logits(16, seed=2)
    hidden = torch.zeros(16, 8, dtype=torch.bfloat16)

    def round_trip(h, r):
        blob = moe_dispatch.route_and_pack(layer, h, r)
        return moe_routing.route(layer, h, blob)

    # backend="eager" for the same reason as above: this asserts a tracing
    # property on CPU tensors, not a lowering.
    compiled = torch.compile(round_trip,
                             backend="eager",
                             fullgraph=True,
                             dynamic=False)
    got_w, got_i = compiled(hidden, logits)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_w, ref_w) and torch.equal(got_i, ref_i)


@pytest.mark.parametrize("experts,topk", [(128, 8), (512, 10), (2048, 10)])
def test_pack_and_unpack_are_o1_in_the_tensor_width(experts, topk):
    """Op count must not grow with the shapes.

    Every aten op here is a node in the MoE layer's compiled graph, and the
    graph is instantiated once per shape bucket per layer. A codec whose op
    count scales with a dimension is what made an earlier fp8-over-ICI codec
    uncompilable at model shapes, so pin the invariant rather than the number.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    rows = 64
    ids = torch.zeros(rows, topk, dtype=torch.int32)
    weights = torch.zeros(rows, topk, dtype=torch.bfloat16)
    blob = torch.zeros(rows * 2 * topk, dtype=torch.int32)

    def n_ops(fn, *args):
        gm = make_fx(fn, tracing_mode="fake")(*args)
        return sum(1 for node in gm.graph.nodes if node.op == "call_function")

    assert n_ops(moe_dispatch.pack_routing, ids, weights) <= 5
    assert n_ops(
        lambda b: moe_dispatch.unpack_routing(b, topk, torch.bfloat16),
        blob) <= 6


# vllm/ray/ray_env.py: the only names copied to a Ray worker are vLLM's own
# registered env vars, anything matching one of these prefixes, PYTHONHASHSEED,
# and whatever the platform names in additional_env_vars. TPU_ is not a prefix,
# and the plugin's env vars are not vLLM's, so a plugin flag read inside the
# worker reaches it *only* via additional_env_vars.
RAY_COPIED_PREFIXES = ("VLLM_", "FLASH_ATTENTION_", "LMCACHE_", "NCCL_",
                       "UCX_", "HF_", "HUGGING_FACE_")

# Env vars this feature reads that are deliberately not propagated, each with
# the reason it is safe. Anything not listed here must be in
# additional_env_vars -- see test_every_env_var_this_feature_reads_reaches_the_worker.
NOT_PROPAGATED = {
    "USE_MOE_EP_KERNEL":
    "read only to leave the patch uninstalled; it is the fused-EP kernel's "
    "own flag and that feature owns its propagation. A worker that never "
    "receives it never runs the fused EP kernel, so the pairing this skip "
    "exists to avoid cannot arise there.",
}


def _feature_env_reads():
    """Every ``envs.NAME`` this feature reads, from the source, not a list."""
    import ast

    import vllm_torchtpu
    from vllm_torchtpu import envs as tpu_envs

    units = [
        inspect.getsource(moe_dispatch),
        inspect.getsource(token_padding),
        inspect.getsource(moe_routing),
        inspect.getsource(vllm_torchtpu._patch_moe_local_routing_dispatch),
    ]
    names = set()
    for src in units:
        for node in ast.walk(ast.parse(textwrap.dedent(src))):
            if isinstance(node, ast.Attribute) and isinstance(
                    node.value, ast.Name) and node.value.id == "envs":
                if node.attr in tpu_envs.environment_variables:
                    names.add(node.attr)
    return names


def test_the_flag_is_declared_for_worker_propagation():
    """The string has to be in the list -- a source property, not a habit.

    ``apply_tpu_patches()`` runs again inside every worker
    (``worker/tpu_worker.py``), and ``token_padding`` caches the flag at
    import there. Under a Ray executor the worker's environment is built from
    ``get_env_vars_to_copy(additional_vars=platform.additional_env_vars)``, so
    a flag missing from that list is set on the driver and nowhere else: the
    patch is never installed in any engine, so a benchmark would measure a
    clean zero with every other assertion still green.
    """
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
    assert "TPU_MOE_LOCAL_ROUTING" in TpuPlatform.additional_env_vars


def test_every_env_var_this_feature_reads_reaches_the_worker():
    """Generalises the above so the next flag cannot be forgotten."""
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    reads = _feature_env_reads()
    # Assert what must be present: a scanner that found nothing would pass
    # the loop below vacuously.
    assert "TPU_MOE_LOCAL_ROUTING" in reads, (
        f"the source scan found {sorted(reads)}, which does not include the "
        "flag this feature is named after -- the scan is broken")

    for name in sorted(reads):
        propagated = (name in TpuPlatform.additional_env_vars
                      or name.startswith(RAY_COPIED_PREFIXES))
        assert propagated or name in NOT_PROPAGATED, (
            f"{name} is read inside the engine worker but is neither in "
            f"TpuPlatform.additional_env_vars nor Ray-prefixed nor "
            f"explained in NOT_PROPAGATED; under a Ray executor it would "
            f"reach the driver only")


def test_the_flat_blob_survives_a_dp_gather_and_rebuilds_global_routing():
    """The wire test, not just the codec test.

    ``all_gather(dim=0)`` on the rank-1 blob concatenates rank chunks; the
    row-major flatten is what makes that equal to concatenating the rows. A
    column-major flatten would pass `pack -> unpack` and shuffle every row on
    the wire, so exercise the gather, not the inverse.
    """
    dp_size, local = 4, 16
    layer = _layer()
    rows = dp_size * local
    logits = _logits(rows, seed=23)
    hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)

    chunks = [
        moe_dispatch.route_and_pack(layer, hidden[r * local:(r + 1) * local],
                                    logits[r * local:(r + 1) * local])
        for r in range(dp_size)
    ]
    assert all(c.dim() == 1 and c.numel() == local * 2 * TOPK for c in chunks)
    gathered = torch.cat(chunks, dim=0)  # exactly all_gather(dim=0)

    got_w, got_i = moe_routing.route(layer, hidden, gathered)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)


# Exactly the aten ops the routing codec is allowed to emit. An allow-list, so
# a *new* copy appearing turns this red -- `_to_copy` is here because the codec
# genuinely converts dtypes (int64 -> int32 ids, bf16 -> f32 for the bitcast),
# which is data conversion, not data movement. Nothing that materialises a
# buffer is on the list, which is what makes the flatten and the reshape back
# provably views.
CODEC_ALLOWED_ATEN_OPS = {
    "aten._to_copy.default",  # ids -> int32, weights -> f32: real casts
    "aten.view.dtype",  # the bitcast
    "aten.view.default",  # the flatten, and the reshape back
    "aten.cat.default",  # the pack
    "aten.slice.Tensor",  # the two column slices in unpack
    "aten.detach.default",  # make_fx bookkeeping
}


def _codec_aten_targets(fn, *args):
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(fn, tracing_mode="fake")(*args)
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


def _codec_example_tensors(rows=64):
    return (torch.zeros(rows, TOPK, dtype=torch.int32),
            torch.zeros(rows, TOPK, dtype=torch.bfloat16),
            torch.zeros(rows * 2 * TOPK, dtype=torch.int32))


def test_the_flatten_and_the_reshape_are_views_not_copies():
    """A materialising reshape would hand back the bytes 1-D exists to save.

    Checked on the traced graph, not in Python: `reshape` on a contiguous
    tensor *is* a view, but only the graph says whether this one is.
    """
    ids, weights, blob = _codec_example_tensors()
    pack = _codec_aten_targets(moe_dispatch.pack_routing, ids, weights)
    unpack = _codec_aten_targets(
        lambda b: moe_dispatch.unpack_routing(b, TOPK, torch.bfloat16), blob)

    for names in (pack, unpack):
        assert set(names) <= CODEC_ALLOWED_ATEN_OPS, (
            f"codec emitted ops outside the allow-list: "
            f"{sorted(set(names) - CODEC_ALLOWED_ATEN_OPS)}")
    # Assert what must be present, not only what must be absent.
    assert "aten.view.default" in pack, pack  # the flatten is a view
    assert "aten.view.default" in unpack, unpack  # so is the reshape back
    assert "aten.view.dtype" in unpack, unpack  # and the bitcast


def test_the_codec_allow_list_rejects_a_codec_that_materialises():
    """Negative control: the allow-list has to be able to fail.

    Same codec with one genuine data movement in it -- a transpose forced
    contiguous -- must fall outside the list.
    """
    ids, weights, _ = _codec_example_tensors()

    def materialising_pack(topk_ids, topk_weights):
        bits = topk_weights.float().view(torch.int32)
        packed = torch.cat((topk_ids.to(torch.int32), bits), dim=-1)
        return packed.t().contiguous().reshape(-1)

    names = set(_codec_aten_targets(materialising_pack, ids, weights))
    extra = names - CODEC_ALLOWED_ATEN_OPS
    assert extra, f"the allow-list accepted a materialising codec: {names}"


def test_a_layer_that_declines_to_pack_still_gets_its_padded_rows_masked(
        monkeypatch):
    """The silent-wrong-answer case the flag must not create.

    With the flag on, a layer whose dispatch did not pack sees ordinary
    logits. If the padding skip were keyed on the flag rather than on this
    layer's own provenance, its padded rows would go unmasked. Nothing is
    keyed on the flag, so they do not -- pin that.
    """
    from vllm.distributed import parallel_state

    dp_size, local = 4, 16
    gathered = []

    class _DPGroup:
        world_size = dp_size

        def all_gather(self, tensor, dim=0):
            gathered.append(tensor)
            return tensor.repeat(dp_size)

    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: _DPGroup())
    import vllm_torchtpu.envs as tpu_envs
    monkeypatch.setattr(
        tpu_envs, "environment_variables",
        dict(tpu_envs.environment_variables,
             TPU_MOE_LOCAL_ROUTING=lambda: True))

    state = token_padding.TokenPaddingState.create(local, torch.device("cpu"))
    state.update(local_valid_tokens=11, local_padded_tokens=local)
    token_padding.set_padding_state(state)
    try:
        layer = _layer()
        rows = dp_size * local
        logits = _logits(rows, seed=31)  # logits, not a blob
        hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
        w, i = moe_routing.route(layer, hidden, logits)
        _, masked = token_padding.zero_routing_weights_for_padding(i, w)
    finally:
        token_padding.set_padding_state(None)

    assert len(gathered) == 1, "the gathered-mask collective must still run"
    mask = state.get_local_padding_mask(local).repeat(dp_size)
    assert (masked[mask] == 0).all(), "padded rows were left unmasked"
    assert (masked[~mask] != 0).any()


# --- what the flag costs per step at DECODE ---------------------------------
# It costs +5.56 us/layer there, +1.64% of a decode step (measured on tpu7x at
# DP4TP2_EP conc 64): the packed s32 blob is 13x smaller in bytes than the
# logit gather but 33% more expensive per call, and at a decode row count a
# collective is pure latency. That cost is in the device program and these
# tests cannot see it; see envs.py for the workload rule it implies.
#
# What they do pin is that no *host* cost is added on top of it. An earlier
# revision lost 20.27% to a per-step device->host sync introduced by a
# diagnostic, on exactly these steps. Decode is most of the steps, so a graph
# break or a sync here would dwarf the collective difference above.
DECODE_ROWS = (8, 16, 32)


def _arm_round_trip(layer, hidden, logits):
    return moe_routing.route(
        layer, hidden, moe_dispatch.route_and_pack(layer, hidden, logits))


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_the_flag_on_path_is_one_graph_with_no_breaks_at_decode(rows):
    """A graph break in the MoE forward at decode would be catastrophic."""
    import torch._dynamo as dynamo

    dynamo.reset()
    layer = _layer()
    logits = _logits(rows, seed=rows)
    hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
    explained = dynamo.explain(_arm_round_trip)(layer, hidden, logits)
    assert explained.graph_count == 1, explained.break_reasons
    assert explained.graph_break_count == 0, explained.break_reasons


def test_the_flag_adds_no_device_host_sync():
    """Count real syncs by wrapping what forces one. Zero, and zero either way.

    `Tensor.float()` is a dtype cast and is not counted; `float(tensor)` would
    be. Both paths are counted on purpose: a property the flag-off path also
    lacks would not be evidence about the flag.
    """
    seen = []
    names = ("item", "tolist", "cpu", "numpy", "__bool__", "__int__",
             "__float__", "__index__")
    originals = {
        n: getattr(torch.Tensor, n)
        for n in names if hasattr(torch.Tensor, n)
    }

    def instrument(name, original):

        def wrapper(self, *a, **kw):
            seen.append(name)
            return original(self, *a, **kw)

        return wrapper

    for n, o in originals.items():
        setattr(torch.Tensor, n, instrument(n, o))
    try:
        layer = _layer()
        counts = {}
        for arm, fn in (("control",
                         lambda h, g: moe_routing.route(layer, h, g)),
                        ("arm", lambda h, g: _arm_round_trip(layer, h, g))):
            seen.clear()
            for rows in DECODE_ROWS:
                logits = _logits(rows, seed=rows)
                hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
                for _ in range(20):
                    fn(hidden, logits)
            counts[arm] = list(seen)
    finally:
        for n, o in originals.items():
            setattr(torch.Tensor, n, o)

    assert counts["arm"] == [], f"the flag-on path synced: {counts['arm']}"
    assert counts["control"] == counts["arm"]


def test_the_codec_is_inlined_into_the_compiled_graph_and_compiled_once():
    """The decisive one: is the added Python per-bucket or per-step?

    `unsupported_reason`, the shape checks and the wrapper's guards are pure
    Python. Inside the region vLLM compiles they are traced through once per
    shape bucket; if the MoE body were an opaque custom op -- which it would
    be but for `_patch_default_moe_runner_select_forward` -- they would run on
    every step, 60 layers deep, which is how an earlier revision lost 20% of
    throughput.

    Two assertions: the backend is invoked exactly once across 25 calls (so
    the frame is not re-entered), and the codec's own aten ops are nodes *in*
    that graph (so it is the codec that got inlined, not something adjacent).
    """
    import torch._dynamo as dynamo

    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    layer = _layer()
    logits = _logits(32, seed=3)
    hidden = torch.zeros(32, 8, dtype=torch.bfloat16)

    dynamo.reset()
    compiled = torch.compile(_arm_round_trip,
                             backend=capture,
                             fullgraph=True,
                             dynamic=False)
    for _ in range(25):
        compiled(layer, hidden, logits)

    assert len(graphs) == 1, (
        f"compiled {len(graphs)} times for 25 invocations of one bucket")
    # A Dynamo graph, so the codec's ops appear as call_function (torch.cat)
    # and call_method (Tensor.view / Tensor.reshape) alike.
    names = [
        str(n.target) for n in graphs[0].graph.nodes
        if n.op in ("call_function", "call_method")
    ]
    assert any("cat" in n for n in names), names  # pack is in the graph
    assert any("view" in n or "reshape" in n for n in names), names
