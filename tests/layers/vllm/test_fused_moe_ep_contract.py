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
"""Collective contract for the fused expert-parallel MoE path.

The fused kernel returns each rank its own token rows, already summed over
every expert. Two things must therefore hold, and both are checked here against
the *real* upstream ``MoERunner`` on CPU:

1. vLLM must not run its DP dispatch/combine pair around the kernel (the rows
   are not gathered and the output is not a partial sum over DP ranks);
2. the TP all-reduce must still run, because it is what re-assembles the
   disjoint row slices *and* reduces the TP-sharded shared expert.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from jax.sharding import PartitionSpec
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner import \
    moe_runner as moe_runner_mod

from vllm_torchtpu.layers.vllm import fused_moe_ep

HIDDEN = 4


class _ExplodingEpGroup:
    """Any DP dispatch/combine call is a contract violation."""

    def dispatch_router_logits(self, *args, **kwargs):
        raise AssertionError("DP all-gather ran under the fused EP kernel")

    def combine(self, *args, **kwargs):
        raise AssertionError("DP reduce-scatter ran under the fused EP kernel")


class _FusedEpQuantMethod:
    """What ``VllmFp8MoEMethodTPU`` looks like with the fused path active."""

    is_monolithic = True
    supports_internal_mk = True
    skip_forward_padding = True
    has_unpadded_output = False
    moe_kernel = None


class _FusedEpExperts:
    """Routed experts whose monolithic kernel owns dispatch and combine."""

    def __init__(self, fused_out):
        self.quant_method = _FusedEpQuantMethod()
        self._fused_out = fused_out

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_monolithic(self, *, x, router_logits, input_ids=None):
        del router_logits, input_ids
        assert x.shape[0] == self._fused_out.shape[0], (
            "the runner must hand the kernel the DP-local rows, not a gathered "
            "batch")
        return self._fused_out


def _make_runner(*, dp_size, ep_size, fused_out):
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=dp_size,
        dp_rank=0,
        ep_size=ep_size,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe_config = SimpleNamespace(
        hidden_dim=HIDDEN,
        hidden_dim_unpadded=HIDDEN,
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=ep_size,
        sp_size=1,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        moe_parallel_config=parallel_config,
    )
    runner = moe_runner_mod.MoERunner.__new__(moe_runner_mod.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = moe_config
    runner.router = object()
    runner.routed_experts = _FusedEpExperts(fused_out)
    runner._shared_experts = None
    runner.gate = None
    runner.shared_expert_gate = None
    runner._fse_fuse_gate = False
    runner._combined_gate_weight = None
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner.layer_name = "fused_ep_contract_test.moe"
    runner._sequence_parallel_context = contextlib.nullcontext
    runner._forward_entry = moe_runner_mod._moe_forward
    return runner


@pytest.mark.parametrize("dp_size,ep_size", [(8, 8), (4, 8), (1, 8)])
def test_no_dp_dispatch_or_combine_and_the_all_reduce_survives(
        monkeypatch, dp_size, ep_size):
    rows = 2
    hidden_states = torch.arange(rows * HIDDEN,
                                 dtype=torch.float32).reshape(rows, HIDDEN)
    router_logits = torch.zeros(rows, ep_size)
    fused_out = torch.full((rows, HIDDEN), 3.0)

    monkeypatch.setattr(moe_runner_mod, "get_ep_group", _ExplodingEpGroup)
    reduced = []

    def _fake_all_reduce(tensor):
        reduced.append(tensor)
        return tensor * 10.0

    monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce",
                        _fake_all_reduce)

    runner = _make_runner(dp_size=dp_size,
                          ep_size=ep_size,
                          fused_out=fused_out)
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name",
                        lambda _name: runner)

    out = runner(hidden_states, router_logits)

    assert len(reduced) == 1, "the TP all-reduce must still run"
    torch.testing.assert_close(out, fused_out * 10.0)


def test_dp_dispatch_still_runs_without_the_fused_kernel(monkeypatch):
    """Control: the same runner *does* dispatch when the method opts out."""
    rows = 2
    hidden_states = torch.zeros(rows, HIDDEN)
    router_logits = torch.zeros(rows, 8)
    runner = _make_runner(dp_size=8,
                          ep_size=8,
                          fused_out=torch.zeros(rows, HIDDEN))
    runner.routed_experts.quant_method.supports_internal_mk = False
    monkeypatch.setattr(moe_runner_mod, "get_ep_group", _ExplodingEpGroup)
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name",
                        lambda _name: runner)

    with pytest.raises(AssertionError, match="DP all-gather ran"):
        runner(hidden_states, router_logits)


class _RecordingEpGroup:

    def __init__(self):
        self.dispatched = 0
        self.combined = 0

    def dispatch_router_logits(self, hidden_states, router_logits, sp=False):
        self.dispatched += 1
        return (torch.cat([hidden_states, hidden_states]),
                torch.cat([router_logits, router_logits]))

    def combine(self, hidden_states, sp=False):
        self.combined += 1
        return hidden_states.chunk(2)[0]


@pytest.mark.parametrize("dp_size,expect_calls", [(8, 1), (1, 0)])
def test_a_step_below_the_threshold_reinstates_the_dp_pair(
        monkeypatch, dp_size, expect_calls):
    """The hazard the token threshold introduces, and its fix.

    ``supports_internal_mk`` is a property of the layer, so the runner stops
    dispatching for *every* step once the fused path is active -- while the
    threshold is per step. A fallback step must therefore run the DP
    all-gather / reduce-scatter itself, or the other DP ranks' expert
    contributions are silently missing (the TP all-reduce does not span them).

    On a *coordinated* step, which is the only kind where the layer owns the
    pair at all: ``supports_internal_mk`` hands it back to the runner
    otherwise, and ``_needs_fallback_dispatch`` reads the same predicate, so
    the two never both run and never both skip.
    """
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)
    group = _RecordingEpGroup()
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    layer = SimpleNamespace(moe_config=SimpleNamespace(dp_size=dp_size))

    tokens = torch.randn(4, HIDDEN)
    logits = torch.randn(4, 8)
    x, gathered_logits = fused_moe_ep.dispatch_for_fallback(
        layer, tokens, logits)
    out = fused_moe_ep.combine_for_fallback(layer, x)

    assert (group.dispatched, group.combined) == (expect_calls, expect_calls)
    assert x.shape[0] == tokens.shape[0] * (2 if expect_calls else 1)
    assert gathered_logits.shape[0] == x.shape[0]
    assert out.shape == tokens.shape


def _reference_moe(tokens, gate_w, up_w, down_w, logits, top_k):
    """Dense fp32 reference: softmax -> top-k -> renormalize -> SwiGLU -> sum."""
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros_like(tokens, dtype=torch.float32)
    for row in range(tokens.shape[0]):
        for k in range(top_k):
            expert = int(ids[row, k])
            gate = tokens[row].float() @ gate_w[expert].float()
            up = tokens[row].float() @ up_w[expert].float()
            out[row] += (torch.nn.functional.silu(gate) *
                         up) @ down_w[expert].float() * weights[row, k]
    return out


@pytest.mark.parametrize("replica_size", [1, 2, 4])
def test_row_slicing_plus_all_reduce_reproduces_the_dense_moe(
        monkeypatch, replica_size):
    """End-to-end numerics of the torch-side slice/scatter around the kernel.

    The kernel itself is replaced by a CPU reference that computes complete
    rows (that is exactly its contract: every expert's contribution, combined
    over the mesh). Summing the ranks' outputs is what the TP all-reduce does.
    """
    torch.manual_seed(0)
    rows, experts, inter, top_k = 32, 4, 6, 2
    tokens = torch.randn(rows, HIDDEN)
    logits = torch.randn(rows, experts)
    gate_w = torch.randn(experts, HIDDEN, inter) / 8
    up_w = torch.randn(experts, HIDDEN, inter) / 8
    down_w = torch.randn(experts, inter, HIDDEN) / 8

    def _fake_kernel(local_tokens, local_logits, *_weights):
        return _reference_moe(local_tokens, gate_w, up_w, down_w, local_logits,
                              top_k)

    monkeypatch.setattr(fused_moe_ep, "_get_op",
                        lambda **_kwargs: _fake_kernel)

    total = torch.zeros(rows, HIDDEN)
    for rank in range(replica_size):
        monkeypatch.setattr(fused_moe_ep,
                            "_replica_group",
                            lambda r=rank: (replica_size, r))
        total += fused_moe_ep.fused_moe_ep(
            tokens,
            logits,
            w1=None,
            w2=None,
            w1_scale=None,
            w2_scale=None,
            top_k=top_k,
            renormalize=True,
            activation="silu",
            scoring_fn="softmax",
            subc_quant_w1_sz=None,
            subc_quant_w2_sz=None,
            quant_act=True,
            replica_rank=torch.tensor(rank, dtype=torch.int32),
        )

    torch.testing.assert_close(
        total, _reference_moe(tokens, gate_w, up_w, down_w, logits, top_k))


@pytest.mark.parametrize(
    "rows,replica_size,expect_local",
    [
        (6, 4, 8),  # 6 rows, 4 replicas: pad up to the 8-row per-rank minimum
        (4, 1, 8),  # a decode-sized DP8_EP batch still gets 8 rows
        (64, 8, 8),
        (4096, 2, 2048),
        (100, 2, 56),  # 50 -> 56, the next multiple of 8
    ])
def test_rows_are_padded_to_a_legal_per_rank_block(monkeypatch, rows,
                                                   replica_size, expect_local):
    tokens = torch.randn(rows, HIDDEN)
    logits = torch.randn(rows, 2)
    seen = []

    def _fake_kernel(local_tokens, local_logits, *_weights):
        seen.append(local_tokens.shape)
        return local_tokens * 2.0

    monkeypatch.setattr(fused_moe_ep, "_get_op", lambda **_kw: _fake_kernel)
    total = torch.zeros(rows, HIDDEN)
    for rank in range(replica_size):
        monkeypatch.setattr(fused_moe_ep,
                            "_replica_group",
                            lambda r=rank: (replica_size, r))
        out = fused_moe_ep.fused_moe_ep(tokens,
                                        logits,
                                        w1=None,
                                        w2=None,
                                        w1_scale=None,
                                        w2_scale=None,
                                        top_k=1,
                                        renormalize=True,
                                        activation="silu",
                                        scoring_fn="softmax",
                                        subc_quant_w1_sz=None,
                                        subc_quant_w2_sz=None,
                                        quant_act=True,
                                        replica_rank=torch.tensor(
                                            rank, dtype=torch.int32))
        assert out.shape == (rows, HIDDEN)
        total += out

    assert seen == [(expect_local, HIDDEN)] * replica_size
    torch.testing.assert_close(total, tokens * 2.0)


def _stub_op(tokens, logits, *_weights):
    return tokens * 2.0


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("replica_size", [1, 2])
def test_the_forward_path_traces_under_dynamo_without_a_graph_break(
        monkeypatch, replica_size, version):
    """`fullgraph=True` over everything the forward does around the kernel.

    vLLM compiles the model with Dynamo, so anything in this path that Dynamo
    cannot trace -- a lock, a lazily *registered* custom op, a mutation of a
    global -- fails on the first profile run and nowhere else. Only the custom
    op itself is stubbed; the op-cache lookup, the env reads, the process-group
    queries and the row slice/scatter are the real ones.
    """
    kwargs = dict(top_k=2,
                  renormalize=True,
                  activation="silu",
                  scoring_fn="softmax",
                  subc_quant_w1_sz=None,
                  subc_quant_w2_sz=None,
                  quant_act=True)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", str(version))
    monkeypatch.setitem(fused_moe_ep._op_cache,
                        fused_moe_ep.op_cache_key(**kwargs), _stub_op)
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda:
                        (replica_size, 0))

    @torch.compile(fullgraph=True, backend="eager")
    def run(tokens, logits, rank):
        return fused_moe_ep.fused_moe_ep(tokens,
                                         logits,
                                         w1=None,
                                         w2=None,
                                         w1_scale=None,
                                         w2_scale=None,
                                         replica_rank=rank,
                                         ep_rank_ids=torch.zeros(
                                             1, dtype=torch.int32),
                                         **kwargs)

    rows = 16 * replica_size
    out = run(torch.randn(rows, HIDDEN), torch.randn(rows, 4),
              torch.tensor(0, dtype=torch.int32))
    assert out.shape == (rows, HIDDEN)


def test_the_forward_never_builds_the_op(monkeypatch):
    """The prebuild at load must cover the key the forward asks for.

    Registering a custom op inside the Dynamo-compiled forward is a TPU-only
    failure (a sibling workstream lost a slot to exactly that). Our op is built
    in `process_weights_after_loading`; what can still go wrong is a *key* skew
    between that prebuild and the forward, which would move the build into the
    traced region. Pin it: with a counting builder, prebuild once, then call the
    forward and assert the builder never ran again.
    """
    builds = []

    def _fake_jax_op(name, fn, **kwargs):
        builds.append(name)

        class _Op:

            def __call__(self, tokens, *_args):
                return tokens * 2.0

            def register_fake(self, _fn):
                pass

        return _Op()

    monkeypatch.setattr(fused_moe_ep.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(fused_moe_ep, "get_or_create_ep_mesh", lambda: "mesh")
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))
    monkeypatch.setattr(fused_moe_ep, "_op_cache", {})

    kwargs = dict(top_k=2,
                  renormalize=True,
                  activation="silu",
                  scoring_fn="softmax",
                  subc_quant_w1_sz=4096,
                  subc_quant_w2_sz=1024,
                  quant_act=True)
    fused_moe_ep.prebuild_fused_ep_moe_kernel(**kwargs)
    assert len(builds) == 1

    fused_moe_ep.fused_moe_ep(torch.randn(16, HIDDEN),
                              torch.randn(16, 4),
                              w1=None,
                              w2=None,
                              w1_scale=None,
                              w2_scale=None,
                              replica_rank=None,
                              **kwargs)

    assert builds == builds[:1], "the forward rebuilt the custom op"


@pytest.mark.parametrize("version,expect_axes,expect_func", [
    (1, ("data", "model"), "fused_ep_moe_func"),
    (2, ("d", ), "fused_ep_moe_v2_func"),
])
def test_the_version_switch_picks_the_program_and_the_mesh_that_go_together(
        monkeypatch, version, expect_axes, expect_func):
    """v1 and v2 are one switch, and the switch moves three things at once.

    The kernel body, the mesh rank (v2 addresses peers with a 1-tuple
    ``device_id`` and names its own axis) and the partition specs have to move
    together: a v2 program under v1's 2-D mesh raises inside Mosaic lowering
    with "got 1 ids for a 2D mesh", which is a long way from the flag that
    caused it.
    """
    seen = {}

    def _fake_jax_op(name, fn, *, mesh, input_partition_specs):
        seen.update(fn=fn, mesh=mesh, specs=input_partition_specs)

        class _Op:

            def __call__(self, tokens, *_args):
                return tokens

            def register_fake(self, _fn):
                pass

        return _Op()

    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", str(version))
    monkeypatch.setattr(fused_moe_ep.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(fused_moe_ep, "_op_cache", {})
    for builder in ("get_or_create_ep_mesh", "get_or_create_ep_mesh_1d"):
        monkeypatch.setattr(fused_moe_ep, builder,
                            (lambda axes: lambda: axes)(expect_axes))

    fused_moe_ep.prebuild_fused_ep_moe_kernel(top_k=2,
                                              renormalize=True,
                                              activation="silu",
                                              scoring_fn="softmax",
                                              subc_quant_w1_sz=4096,
                                              subc_quant_w2_sz=1024,
                                              quant_act=True)

    assert seen["fn"].func.__name__ == expect_func
    # v2 takes a seventh operand, the shard identity -- it has to be a
    # parameter of the exported module or the SPMD partitioner synthesises a
    # partition-id for it (register_ep_rank_buffer).
    n_operands = 6 if version == 1 else 7
    assert seen["specs"] == (PartitionSpec(expect_axes[-1]), ) * n_operands
    # The mesh the shard_map traces under is the mesh the bridge sizes with.
    assert seen["fn"].keywords["mesh"] is seen["mesh"]


def test_the_v2_op_registers_with_a_tensors_only_schema(monkeypatch):
    """The bridge's signature inspection is a trap, so build the real op.

    ``pallas.jax_op`` classifies every positional parameter that is not
    annotated exactly ``jax.Array`` / ``Optional[jax.Array]`` as *static*
    (``pallas.py:_infer_static_argnums``), by identity -- so a
    ``from __future__ import annotations`` anywhere in the JAX adapter turns all
    six tensors into compile-time constants and the op refuses to build. That is
    invisible to every other test here, which stubs the op out.
    """
    jax = pytest.importorskip("jax")
    import numpy as np
    from jax.sharding import AxisType, Mesh

    # One device is enough: registration only inspects the signature and
    # records the mesh, and the ep width is not in the schema.
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1),
                axis_names=("d", ),
                axis_types=(AxisType.Explicit, ))
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_VERSION", "2")
    monkeypatch.setattr(fused_moe_ep, "get_or_create_ep_mesh_1d", lambda: mesh)
    monkeypatch.setattr(fused_moe_ep, "_op_cache", {})

    op = fused_moe_ep._get_op(top_k=10,
                              renormalize=True,
                              activation="silu",
                              scoring_fn="softmax",
                              subc_quant_w1_sz=4096,
                              subc_quant_w2_sz=1024,
                              quant_act=True)
    schema = str(op._opoverload._schema)
    positional, keyword = schema.split(", *, ")
    # Six tensors in, one out, and nothing positional that is not a tensor:
    # this is the assertion the future import would break.
    assert positional.count("Tensor") == 7, schema  # +1: the shard identity
    assert schema.count("-> Tensor") == 1, schema
    # The mesh is not a torch type, so the bridge drops it from the schema
    # (pallas.py:_get_torch_signature); the rest of the config survives as
    # keyword-only parameters that already carry the value the partial bound,
    # so the forward passes tensors alone and a different constant is a
    # different op name rather than a different argument.
    assert "mesh" not in schema, schema
    assert all("=" in p for p in keyword.rstrip(") -> Tensor").split(", ")), (
        f"a config parameter reached the op schema unbound: {schema}")


def test_an_uncoordinated_step_gives_the_whole_takeover_back(monkeypatch):
    """Half a takeover is worse than none.

    On a step whose DP ranks have not agreed on a shape, the kernel declines --
    but if the layer keeps `supports_internal_mk` it has also told the runner to
    stop dispatching and combining, and then does that itself from inside the
    layer, on ranks that are not arriving together. Same deadlock, different
    collective, and it presents the same way (a SparseCore halt, because an
    offloaded collective is what is outstanding when the TensorCore waits).
    Measured: W6 jobs j-ef24b962 / j-d68b0dd5 / j-2846842c, three boots.

    The runner reads `supports_internal_mk` inside `_maybe_dispatch` and
    `_maybe_combine` on every forward, so it can and must answer per step.
    """

    from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

    method = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
    method._fused_ep_enabled = True
    method._fused_ep_dp_size = 8
    layer = SimpleNamespace(moe_config=SimpleNamespace(dp_size=8))

    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", None, raising=False)
    assert method.supports_internal_mk is False, (
        "the layer kept ownership of dispatch/combine on a step it cannot "
        "coordinate")
    assert fused_moe_ep._needs_fallback_dispatch(layer) is False, (
        "and would then have reinstated the pair the runner is still doing")

    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", True, raising=False)
    assert method.supports_internal_mk is True
    assert fused_moe_ep._needs_fallback_dispatch(layer) is True


def test_without_dp_the_takeover_never_depends_on_a_forward_context(
        monkeypatch):
    """dp_size 1 has nothing to coordinate, and TP8_EP must not be disabled by
    a missing forward context."""

    from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

    method = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
    method._fused_ep_enabled = True
    method._fused_ep_dp_size = 1
    monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", None, raising=False)
    assert method.supports_internal_mk is True


def _layer_dp(dp_size):
    return SimpleNamespace(moe_config=SimpleNamespace(dp_size=dp_size))


def test_nothing_on_the_traced_path_logs(monkeypatch):
    """Dynamo refuses a `logging.Logger` call in a graph it is capturing.

    "logging.Logger method not supported for non-export cases" (W6,
    j-cf06d716) -- fatal in the traced region and invisible in eager, which is
    why 33 decline lines printed happily and the boot still died. The decline
    is announced from `dp_agreement`, which runs eagerly in the runner; nothing
    reachable from the forward may log.

    Checked structurally rather than by inspection, because the next person
    adding a diagnostic to the decline path will not read this docstring.
    """
    import inspect

    traced = (fused_moe_ep.engages_for, fused_moe_ep.dp_agreed_tokens,
              fused_moe_ep.dp_step_is_coordinated, fused_moe_ep.max_tokens,
              fused_moe_ep.capacity_skew, fused_moe_ep.kernel_version,
              fused_moe_ep.fused_moe_ep, fused_moe_ep.dispatch_for_fallback,
              fused_moe_ep.combine_for_fallback, fused_moe_ep.op_cache_key,
              fused_moe_ep.block_size_override)
    offenders = [
        fn.__name__ for fn in traced if "logger." in inspect.getsource(fn)
        or "print(" in inspect.getsource(fn)
    ]
    assert not offenders, (
        f"{offenders} log from inside the Dynamo-traced forward; announce it "
        "from dp_agreement or another eager caller instead")


def test_the_engagement_assertion_flag_cannot_change_a_graph():
    """`MOE_FUSED_EP_REQUIRE_ENGAGEMENT` is in the compile-cache IGNORE list.

    The AOT cache key hashes env + config but validates only *source files* on
    load, so a **graph-changing** flag parked in that ignore set makes arm B
    load arm A's executable and the A/B measures nothing while returning a
    clean, plausible null (`d4/a/REPORT.md`). Ignoring this one is only correct
    because it gates a post-hoc assertion in eager code and is unreachable from
    anything traced. Pin that, structurally: if someone ever reads it from the
    forward, this fails instead of a serving A/B silently comparing an arm with
    itself.
    """
    import inspect

    from vllm_torchtpu.compilation import tpu_compiler

    flag = "MOE_FUSED_EP_REQUIRE_ENGAGEMENT"
    assert flag in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    # …and the two that select which custom op is emitted must NEVER be ignored.
    for graph_changing in ("USE_MOE_EP_KERNEL", "MOE_FUSED_EP_KERNEL_VERSION",
                           "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                           "MOE_FUSED_EP_KERNEL_MAX_TOKENS",
                           "MOE_FUSED_EP_QUANT_ACT"):
        assert graph_changing not in tpu_compiler._TPU_COMPILE_ENV_IGNORED, (
            f"{graph_changing} selects which MoE program is compiled; ignoring "
            "it in the cache key would make every arm load the first arm's "
            "executable")

    traced = (fused_moe_ep.engages_for, fused_moe_ep.dp_agreed_tokens,
              fused_moe_ep.dp_step_is_coordinated, fused_moe_ep.max_tokens,
              fused_moe_ep.engagement_verdict, fused_moe_ep.capacity_skew,
              fused_moe_ep.kernel_version, fused_moe_ep.fused_moe_ep,
              fused_moe_ep.dispatch_for_fallback,
              fused_moe_ep.combine_for_fallback, fused_moe_ep.op_cache_key,
              fused_moe_ep.block_size_override, fused_moe_ep._get_op)
    readers = [
        fn.__name__ for fn in traced if flag in inspect.getsource(fn)
    ]
    assert not readers, (
        f"{readers} read {flag}, which is excluded from the compile-cache key "
        "-- so two arms differing only in it would share an executable")
