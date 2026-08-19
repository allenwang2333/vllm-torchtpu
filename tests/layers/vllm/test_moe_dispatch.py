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
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.vllm import moe_dispatch, moe_routing, token_padding

EXPERTS = 512
TOPK = 10


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
                 moe_config=SimpleNamespace(experts_per_token=topk))
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _logits(rows, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(rows, EXPERTS) * 2.0).to(torch.bfloat16)


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
    "sigmoid_bias": dict(scoring_func="sigmoid",
                         e_score_correction_bias=torch.linspace(
                             -1.0, 1.0, EXPERTS)),
    "sqrtsoftplus_scaled": dict(scoring_func="sqrtsoftplus",
                                routed_scaling_factor=2.5),
    "no_renormalize": dict(renormalize=False),
    "grouped": dict(use_grouped_topk=True, num_expert_group=8, topk_group=3),
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
    torch.manual_seed(rows)
    ids = torch.randint(0, EXPERTS, (rows, TOPK), dtype=torch.int32)
    weights = torch.rand(rows, TOPK).to(dtype)
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


@pytest.mark.parametrize("bad", [
    torch.zeros(4, 2 * TOPK, dtype=torch.int32),   # still 2-D: the old format
    torch.zeros(4 * 2 * TOPK + 3, dtype=torch.int32),   # not a whole number of rows
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
    got_w, got_i, ref_w, ref_i = _chunked_vs_global(layer, 64 * dp_size,
                                                    dp_size, seed=dp_size)
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)


@pytest.mark.parametrize("dp_size", [2, 4, 8])
def test_the_row_wise_check_rejects_a_router_that_is_not_row_wise(
        monkeypatch, dp_size):
    """Negative control for the test above: it must be able to fail.

    A router with a genuine cross-row dependency -- here, centring the scores
    on the batch's own column mean -- is exactly what would make routing before
    the dispatch wrong, and the comparison has to catch it.
    """

    def not_row_wise(hidden_states, router_logits, *, topk, renormalize,
                     scoring_fn, layer=None, input_ids=None):
        scores = router_logits.float().softmax(dim=-1)
        scores = scores - scores.mean(dim=0, keepdim=True)
        w, i = torch.topk(scores, k=topk, dim=-1)
        return w.to(hidden_states.dtype), i.to(torch.int32)

    monkeypatch.setattr(moe_routing, "select_experts", not_row_wise)
    got_w, got_i, ref_w, ref_i = _chunked_vs_global(_layer(), 64 * dp_size,
                                                    dp_size, seed=dp_size)
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


@pytest.mark.parametrize("unsupported", [
    dict(hash_indices_table=torch.zeros(4, TOPK, dtype=torch.int32)),
    dict(custom_routing_function=lambda **kw: None),
])
def test_route_and_pack_refuses_what_it_cannot_prove(unsupported):
    layer = _layer(**unsupported)
    assert moe_dispatch.unsupported_reason(layer) is not None
    with pytest.raises(NotImplementedError, match="TPU_MOE_LOCAL_ROUTING"):
        moe_dispatch.route_and_pack(layer, torch.zeros(4, 8), _logits(4))


# The six quant methods that reach moe_routing.route, and whether each zeroes
# padded routing weights with the flag off. Only two do, and one of those is
# behind its own flag. Routing before the dispatch must not change this: a
# performance flag that starts zeroing rows four methods leave computed is a
# semantic change, and it would break the claim that the decision is
# bit-identical.
ROUTE_CALLERS = {
    "quantization.fp8": "TPU_MOE_SKIP_PADDED_TOKENS",
    "quantization.compressed_tensors.compressed_tensors_moe"
    ".compressed_tensors_moe_w4an_mxfp4": "",
    "quantization.mxfp4": None,
    "quantization.nvfp4": None,
    "quantization.unquantized": None,
    "quantization.compressed_tensors.compressed_tensors_moe"
    ".compressed_tensors_moe_w4a16": None,
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


def _install_patch(monkeypatch, enabled):
    """Install (or refuse to install) the _maybe_dispatch patch, undoably."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu
    monkeypatch.setattr(
        vllm_torchtpu.envs, "environment_variables",
        dict(vllm_torchtpu.envs.environment_variables,
             TPU_MOE_LOCAL_ROUTING=lambda: enabled))
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
    """With no env set the flag reads False, and it is read in one place.

    One read site is the whole point: the flag gates the patch installer and
    nothing else, so there is no module-scope cache whose value depends on
    import order relative to the worker's environment.
    """
    import vllm_torchtpu.envs as envs
    import vllm_torchtpu
    assert envs.TPU_MOE_LOCAL_ROUTING is False
    readers = [
        m for m, src in (
            ("moe_dispatch", inspect.getsource(moe_dispatch)),
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
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    before = mr.MoERunner.__dict__["_maybe_dispatch"]
    assert before.__module__.startswith("vllm."), (
        f"expected unpatched upstream code to start from, got {before!r} "
        f"from {before.__module__}")

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
    layer.quant_method = SimpleNamespace(is_monolithic=True)

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


def test_refuses_to_coexist_with_the_fused_ep_kernel(monkeypatch):
    """The fused EP kernel scores the raw logits this patch replaces."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu

    monkeypatch.setattr(
        vllm_torchtpu.envs, "environment_variables",
        dict(vllm_torchtpu.envs.environment_variables,
             TPU_MOE_LOCAL_ROUTING=lambda: True,
             USE_MOE_EP_KERNEL=lambda: True))
    monkeypatch.delattr(mr.MoERunner,
                        "_tpu_local_routing_patch",
                        raising=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        vllm_torchtpu._patch_moe_local_routing_dispatch()


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
    "read only to refuse the pair at install time; it is the fused-EP "
    "kernel's own flag and that feature owns its propagation. A worker that "
    "never receives it never runs the fused EP kernel, so the combination "
    "this refusal exists to prevent cannot arise there.",
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
    gathered = torch.cat(chunks, dim=0)          # exactly all_gather(dim=0)

    got_w, got_i = moe_routing.route(layer, hidden, gathered)
    ref_w, ref_i = moe_routing.route(layer, hidden, logits)
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)


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
        logits = _logits(rows, seed=31)          # logits, not a blob
        hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
        w, i = moe_routing.route(layer, hidden, logits)
        _, masked = token_padding.zero_routing_weights_for_padding(i, w)
    finally:
        token_padding.set_padding_state(None)

    assert len(gathered) == 1, "the gathered-mask collective must still run"
    mask = state.get_local_padding_mask(local).repeat(dp_size)
    assert (masked[mask] == 0).all(), "padded rows were left unmasked"
    assert (masked[~mask] != 0).any()


