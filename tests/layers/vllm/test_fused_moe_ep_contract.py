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

The kernel returns each rank its own token rows, already summed over every
expert on every rank -- reduced over the EXPERT axis, and a row-disjoint
partition over the mesh. It is *not* a TP partial, so the bridge zero-fills the
other ranks' rows and lets vLLM's late all-reduce reassemble the batch. Three
things must therefore hold, all checked against the *real* upstream
``MoERunner`` on CPU:

1. vLLM must not run its DP dispatch/combine pair around the kernel (the rows
   are not gathered and the output is not a partial sum over DP ranks);
2. the late all-reduce must still run, because it is what reassembles the
   disjoint row slices *and*, in the same collective, sums a TP-sharded shared
   expert -- so this layer must never declare ``moe_kernel.output_is_reduced``;
3. every deployment that would withdraw that all-reduce must be refused at
   weight load, not discovered as wrong output.

``_run_tp_group`` drives (2) and (3) properly: it runs the real runner once per
TP rank and replays the all-reduce across them, so the assertion is the
invariant that matters -- **after the layer every rank holds the complete
output** -- and not merely "a collective was called".

The row-slicing path itself is refused today for ``tp_size > 1``
(``fused_moe_ep.unsupported_reason``, "not measured for this kernel"). These
cases are what that refusal has to be lifted against; they also pin vLLM 0.27's
reduce order, which is what would move under us on an upgrade.
"""

import contextlib
import itertools
from types import SimpleNamespace

import pytest
import torch
from jax.sharding import PartitionSpec
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner import \
    moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.shared_experts import \
    SharedExperts

from vllm_torchtpu.layers.vllm import fused_moe_ep

HIDDEN = 4
# Divisible by the TP width the reduction cases use, so a rank's row slice is
# a whole number of rows.
ROWS = 8


class _ExplodingEpGroup:
    """Any DP dispatch/combine call is a contract violation."""

    def dispatch_router_logits(self, *args, **kwargs):
        raise AssertionError("DP all-gather ran under the fused EP kernel")

    def combine(self, *args, **kwargs):
        raise AssertionError("DP reduce-scatter ran under the fused EP kernel")


class _NoPrepareFinalize:
    """``FusedMoEMethodBase.topk_indices_dtype`` reaches through this."""

    @staticmethod
    def topk_indices_dtype():
        return None


class _ReducedOutputMoEKernel:
    """The duck type that declares "this layer reduced its own output".

    Our #3040 branch sets exactly this on its own quant method, because that
    kernel really does return a fully reduced output. Here it is a MUTATION:
    v2 returns a row slice, and this is the spelling that silently drops the
    other ranks' rows. See ``test_declaring_the_output_reduced_drops_...``.
    """

    is_monolithic = True
    can_overlap_shared_experts = False
    prepare_finalize = _NoPrepareFinalize()

    @staticmethod
    def output_is_reduced():
        return True


class _FusedEpQuantMethod:
    """What ``VllmFp8MoEMethodTPU`` looks like with the fused path active."""

    is_monolithic = True
    supports_internal_mk = True
    skip_forward_padding = True
    has_unpadded_output = False
    mk_can_overlap_shared_experts = False

    def __init__(self, moe_kernel=None):
        # None is the shipped answer: `MoERunner._fused_output_is_reduced` is
        # `moe_kernel is not None and moe_kernel.output_is_reduced()`.
        self.moe_kernel = moe_kernel


class _FusedEpExperts:
    """Routed experts whose monolithic kernel owns dispatch and combine."""

    def __init__(self, fused_out, moe_kernel=None):
        self.quant_method = _FusedEpQuantMethod(moe_kernel)
        self._fused_out = fused_out

    def _ensure_moe_quant_config_init(self):
        pass

    def forward_monolithic(self, *, x, router_logits, input_ids=None):
        del router_logits, input_ids
        assert x.shape[0] == self._fused_out.shape[0], (
            "the runner must hand the kernel the DP-local rows, not a gathered "
            "batch")
        return self._fused_out

class _SharedPartial(torch.nn.Module):
    """A TP-sharded shared expert: this rank's partial sum, not the total."""

    def __init__(self, partial):
        super().__init__()
        self._partial = partial

    def forward(self, x):
        del x
        return self._partial


def _make_runner(*,
                 dp_size,
                 ep_size,
                 fused_out,
                 shared_partial=None,
                 moe_kernel=None,
                 skip_final_all_reduce=False):
    """The real ``MoERunner`` around a stubbed monolithic quant method.

    ``shared_partial`` attaches vLLM's real ``SharedExperts`` wrapper around a
    module that returns that tensor -- i.e. a TP-sharded shared expert's own
    partial sum, which is what Qwen3.5 produces (``Qwen3NextMLP`` is built with
    ``reduce_results=False``, so its RowParallel down-projection does not
    all-reduce and the runner is what must).
    """
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
        skip_final_all_reduce=skip_final_all_reduce,
        moe_parallel_config=parallel_config,
    )
    runner = moe_runner_mod.MoERunner.__new__(moe_runner_mod.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = moe_config
    runner.router = object()
    runner.routed_experts = _FusedEpExperts(fused_out, moe_kernel)
    # `MoERunner.shared_experts` is a read-only property over
    # `_shared_experts`, so the wrapper is the only thing to set -- and the
    # entry point moves with it (`_select_forward` picks the two-output op).
    if shared_partial is None:
        runner._shared_experts = None
        runner._forward_entry = moe_runner_mod._moe_forward
    else:
        runner._shared_experts = SharedExperts(
            _SharedPartial(shared_partial),
            moe_config,
            enable_dbo=False,
            mk_can_overlap_shared_experts=lambda: False)
        runner._forward_entry = moe_runner_mod._moe_forward_shared
    runner.gate = None
    runner.shared_expert_gate = None
    runner._fse_fuse_gate = False
    runner._combined_gate_weight = None
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner.layer_name = "fused_ep_contract_test.moe"
    runner._sequence_parallel_context = contextlib.nullcontext
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


def test_the_takeover_is_one_answer_for_the_run(monkeypatch):
    """Half a takeover is worse than none, so there is no per-step half.

    The layer tells the runner "I own DP dispatch and combine"
    (``supports_internal_mk``) and the runner reads that on every forward. An
    earlier shape of this path answered it per step, from a token band and a
    DP-coordination flag, and then reinstated the pair itself when it declined.
    Two ways that lost: a step where one rank engaged and another did not hung
    the device in ``semaphore_wait`` (W6, three boots), and vLLM drops Dynamo's
    guards after the first trace, so the per-step answer was frozen anyway.

    Arming is now settled once, at weight load, and this pins both halves of
    that: the property does not read the step, and the per-step reinstatement
    pair is gone rather than dormant.
    """
    from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

    method = fp8_mod.VllmFp8MoEMethodTPU.__new__(fp8_mod.VllmFp8MoEMethodTPU)
    method._fused_ep_enabled = True
    for agreed in (None, False, True):
        monkeypatch.setattr(fused_moe_ep, "_DP_AGREED", agreed, raising=False)
        assert method.supports_internal_mk is True, (
            "the takeover varied with the step; the runner compiles one graph")
    method._fused_ep_enabled = False
    assert method.supports_internal_mk is False

    for gone in ("dispatch_for_fallback", "combine_for_fallback",
                 "engages_for", "engaged_at"):
        assert not hasattr(fused_moe_ep, gone), (
            f"{gone} is back: a per-step engagement test cannot coexist "
            "with a layer-level supports_internal_mk without splitting the "
            "takeover")


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
            ep_rank_ids=torch.zeros(1, dtype=torch.int32),
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
                                            rank, dtype=torch.int32),
                                        ep_rank_ids=torch.zeros(
                                            1, dtype=torch.int32))
        assert out.shape == (rows, HIDDEN)
        total += out

    assert seen == [(expect_local, HIDDEN)] * replica_size
    torch.testing.assert_close(total, tokens * 2.0)


def _stub_op(tokens, logits, *_weights):
    return tokens * 2.0


@pytest.mark.parametrize("replica_size", [1, 2])
def test_the_forward_path_traces_under_dynamo_without_a_graph_break(
        monkeypatch, replica_size):
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
    monkeypatch.setattr(fused_moe_ep, "get_or_create_ep_mesh_1d",
                        lambda: "mesh")
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
                              ep_rank_ids=torch.zeros(1, dtype=torch.int32),
                              **kwargs)

    assert builds == builds[:1], "the forward rebuilt the custom op"


def test_the_program_and_the_mesh_that_go_together(monkeypatch):
    """The kernel body, the mesh rank and the partition specs move as one.

    v2 addresses its peers with a 1-tuple ``device_id`` and names its own
    single mesh axis, so it is built on the 1-D EP mesh: the same program on a
    2-D mesh raises inside Mosaic lowering with "got 1 ids for a 2D mesh", a
    long way from whatever wired it. It also takes a seventh operand -- the
    shard identity -- which has to be a parameter of the exported module or the
    SPMD partitioner synthesises a ``partition-id`` for it
    (``register_ep_rank_buffer``).
    """
    expect_axes = ("d", )
    seen = {}

    def _fake_jax_op(name, fn, *, mesh, input_partition_specs):
        seen.update(fn=fn, mesh=mesh, specs=input_partition_specs)

        class _Op:

            def __call__(self, tokens, *_args):
                return tokens

            def register_fake(self, _fn):
                pass

        return _Op()

    monkeypatch.setattr(fused_moe_ep.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(fused_moe_ep, "_op_cache", {})
    monkeypatch.setattr(fused_moe_ep, "get_or_create_ep_mesh_1d",
                        lambda: expect_axes)
    fused_moe_ep.prebuild_fused_ep_moe_kernel(top_k=2,
                                              renormalize=True,
                                              activation="silu",
                                              scoring_fn="softmax",
                                              subc_quant_w1_sz=4096,
                                              subc_quant_w2_sz=1024,
                                              quant_act=True)

    assert seen["fn"].func.__name__ == "fused_ep_moe_v2_func"
    assert seen["specs"] == (PartitionSpec(expect_axes[-1]), ) * 7
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


def test_a_deployment_that_cannot_coordinate_is_refused_not_fallen_back(
        monkeypatch):
    """The successor to the per-step decline: refuse the layout at weight load.

    A step where one rank engages and another does not hangs the TensorCore in
    ``semaphore_wait`` (W6, three boots). The old fix answered per step; this
    one answers per deployment, and ``dp_size=1`` -- the layout with no DP
    transport to take over -- is refused by name.
    """
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (8, 0))
    reason = fused_moe_ep.out_of_band_reason(
        SimpleNamespace(moe_config=SimpleNamespace(dp_size=1)))
    assert reason is not None and "dp_size=1" in reason, reason


def test_nothing_on_the_traced_path_logs(monkeypatch):
    """Dynamo refuses a `logging.Logger` call in a graph it is capturing.

    "logging.Logger method not supported for non-export cases" (W6,
    j-cf06d716) -- fatal in the traced region and invisible in eager, which is
    why 33 decline lines printed happily and the boot still died. Arming is
    announced from `is_enabled`, which runs at weight load; nothing reachable
    from the forward may log.

    Checked structurally rather than by inspection, because the next person
    adding a diagnostic to the decline path will not read this docstring.
    """
    import inspect

    traced = (fused_moe_ep.fused_moe_ep, fused_moe_ep.op_cache_key,
              fused_moe_ep.block_size_override, fused_moe_ep.capacity_skew,
              fused_moe_ep.max_tokens, fused_moe_ep._input_partition_specs,
              fused_moe_ep._padded_rows, fused_moe_ep._row_slice_index,
              fused_moe_ep._scatter_replica_rows)
    offenders = [
        fn.__name__ for fn in traced if "logger." in inspect.getsource(fn)
        or "print(" in inspect.getsource(fn)
    ]
    assert not offenders, (
        f"{offenders} log from inside the Dynamo-traced forward; announce it "
        "from is_enabled or another eager caller instead")


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
    # …and the ones selecting which custom op is emitted are NEVER ignored.
    for graph_changing in ("USE_MOE_FUSED_EP_V2", "USE_MOE_EP_KERNEL",
                           "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                           "MOE_FUSED_EP_KERNEL_MAX_TOKENS",
                           "MOE_FUSED_EP_QUANT_ACT"):
        assert graph_changing not in tpu_compiler._TPU_COMPILE_ENV_IGNORED, (
            f"{graph_changing} selects which MoE program is compiled; ignoring "
            "it in the cache key would make every arm load the first arm's "
            "executable")

    traced = (fused_moe_ep.fused_moe_ep, fused_moe_ep.op_cache_key,
              fused_moe_ep.block_size_override, fused_moe_ep.capacity_skew,
              fused_moe_ep.max_tokens, fused_moe_ep._get_op)
    readers = [
        fn.__name__ for fn in traced if flag in inspect.getsource(fn)
    ]
    assert not readers, (
        f"{readers} read {flag}, which is excluded from the compile-cache key "
        "-- so two arms differing only in it would share an executable")


# ---------------------------------------------------------------------------
# The reduction contract: who reduces what, and every spelling that loses half
# of it. Driven through the real MoERunner with a real (TP-sharded) shared
# expert, because the two halves are decided by two different switches and only
# one of them touches the shared expert:
#
#   moe_kernel.output_is_reduced() -> True  skips the late all-reduce AND
#                                           reduces the shared expert alone
#   moe_config.skip_final_all_reduce        skips the late all-reduce and
#                                           does NOTHING for the shared expert
#
# v2 wants neither: its output is a row slice, so the late all-reduce is the
# reassembly and, in the same collective, the shared expert's reduction.
# ---------------------------------------------------------------------------

TP = 4


def _run_tp_group(monkeypatch, *, fused_of_rank, shared_of_rank, moe_kernel,
                  skip_final_all_reduce, tp=TP):
    """Run the real runner once per TP rank; return each rank's final output.

    ``tensor_model_parallel_all_reduce`` is replayed in two passes: pass 1
    records every rank's operand per call index, pass 2 hands back the sum over
    ranks. That is exact here because no all-reduce operand in this graph
    depends on another all-reduce's output -- the one that would,
    ``_maybe_reduce_routed_output_before_transform``, needs a
    ``routed_output_transform`` (latent MoE), which this path refuses.

    Asserting on the returned tensors rather than on a call count is the point:
    a recording stub that returns its input cannot tell "reduced the sum" from
    "reduced the shared expert and skipped the sum".
    """
    x = torch.zeros(ROWS, HIDDEN)
    logits = torch.zeros(ROWS, 8)
    recorded: list[list[torch.Tensor]] = []

    def _one_rank(rank, sums):
        calls = itertools.count()

        def _all_reduce(tensor):
            i = next(calls)
            if sums is None:
                while len(recorded) <= i:
                    recorded.append([])
                recorded[i].append(tensor.clone())
                return tensor
            return sums[i]

        monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce",
                            _all_reduce)
        runner = _make_runner(dp_size=tp,
                              ep_size=tp,
                              fused_out=fused_of_rank(rank),
                              shared_partial=shared_of_rank(rank),
                              moe_kernel=moe_kernel,
                              skip_final_all_reduce=skip_final_all_reduce)
        monkeypatch.setattr(moe_runner_mod, "get_ep_group", _ExplodingEpGroup)
        monkeypatch.setattr(moe_runner_mod, "get_layer_from_name",
                            lambda _name, r=runner: r)
        return runner(x, logits)

    for rank in range(tp):
        _one_rank(rank, None)
    sums = [torch.stack(operands).sum(0) for operands in recorded]
    return [_one_rank(rank, sums) for rank in range(tp)]


def _row_slice_of(rank):
    """What v2 returns on rank ``rank``: its own rows, zero everywhere else."""
    out = torch.zeros(ROWS, HIDDEN)
    rows = ROWS // TP
    out[rank * rows:(rank + 1) * rows] = float(rank + 1)
    return out


def _complete_rows(rank):
    """What a kernel that gathers internally returns: every row, everywhere."""
    del rank
    return sum(_row_slice_of(r) for r in range(TP))


def _shared_partial_of(rank):
    """A TP-sharded shared expert's partial sum on rank ``rank``."""
    return torch.full((ROWS, HIDDEN), 0.25 * (rank + 1))


def _all_shared():
    return sum(_shared_partial_of(r) for r in range(TP))


# --- contract A: what v2 ships -- routed output is this rank's row slice -----


def test_the_late_all_reduce_reassembles_rows_and_reduces_the_shared_expert(
        monkeypatch):
    """The shipped spelling, end to end through the runner.

    One collective does both halves: it sums the disjoint row slices into the
    whole batch and sums the shared expert's TP partials, and every rank comes
    out holding the complete layer output.
    """
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_row_slice_of,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=None,
                         skip_final_all_reduce=False)
    want = _complete_rows(0) + _all_shared()
    for rank, out in enumerate(outs):
        torch.testing.assert_close(out, want, msg=f"rank {rank}")


def test_declaring_the_output_reduced_drops_the_other_ranks_rows(monkeypatch):
    """`moe_kernel.output_is_reduced()` is the #3040 spelling, and it is wrong
    here: v2's rows are a partition, not a sum, so skipping the late all-reduce
    leaves every rank holding only its own slice. The shared expert survives
    (it gets its own reduce), which is exactly what makes this hard to see.
    """
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_row_slice_of,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=_ReducedOutputMoEKernel(),
                         skip_final_all_reduce=False)
    torch.testing.assert_close(outs[0], _row_slice_of(0) + _all_shared())
    with pytest.raises(AssertionError):
        torch.testing.assert_close(outs[0], _complete_rows(0) + _all_shared())


def test_skip_final_all_reduce_loses_the_rows_and_the_shared_expert(
        monkeypatch):
    """The other way to switch the same collective off, and the worse one:
    nothing is reduced at all, and vLLM's own assertion against a pre-reduced
    output cannot see it (that assert is gated on `moe_kernel`, which this path
    leaves None). Hence the weight-load refusal below.
    """
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_row_slice_of,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=None,
                         skip_final_all_reduce=True)
    torch.testing.assert_close(outs[0],
                               _row_slice_of(0) + _shared_partial_of(0))


# --- contract B: the #3040 shape, and any future in-kernel gather ------------


def test_a_reduced_output_still_needs_the_shared_expert_reduced(monkeypatch):
    """If the kernel ever gathers internally, `moe_kernel` becomes required.

    The routed output is then complete on every rank, and the ONLY thing that
    reduces the TP-sharded shared expert is
    `_maybe_reduce_shared_expert_output`, which fires on that switch alone.
    """
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_complete_rows,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=_ReducedOutputMoEKernel(),
                         skip_final_all_reduce=False)
    want = _complete_rows(0) + _all_shared()
    for rank, out in enumerate(outs):
        torch.testing.assert_close(out, want, msg=f"rank {rank}")


def test_skip_final_all_reduce_silently_loses_the_shared_expert(monkeypatch):
    """The trap this whole file exists for, isolated.

    Same reduced routed output, the other switch: the routed half is right, and
    each rank keeps 1/tp of the shared expert forever. The model still produces
    fluent text, which is why this is asserted as an exact wrong answer rather
    than left to a numerics run to notice.
    """
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_complete_rows,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=None,
                         skip_final_all_reduce=True)
    torch.testing.assert_close(outs[0],
                               _complete_rows(0) + _shared_partial_of(0))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(outs[0], _complete_rows(0) + _all_shared())


def test_a_reduced_output_that_declares_nothing_is_counted_tp_times(
        monkeypatch):
    """The double-count, for completeness: a fully reduced output left to the
    late all-reduce is multiplied by the TP width."""
    outs = _run_tp_group(monkeypatch,
                         fused_of_rank=_complete_rows,
                         shared_of_rank=_shared_partial_of,
                         moe_kernel=None,
                         skip_final_all_reduce=False)
    torch.testing.assert_close(outs[0], TP * _complete_rows(0) + _all_shared())


# --- and the two ways to state the contract at its source --------------------


def test_nothing_on_this_path_assigns_moe_kernel():
    """`moe_kernel` is vLLM's switch for "this layer reduced its own output".

    v2 returns a row slice, so it must stay None. Structural because the
    behavioural test above can only see it once a shared expert is wired into
    the fixture, and because the tempting edit -- copying #3040's
    `ReducedOutputMoEKernel` across -- is one line in a file nobody re-reads.
    """
    import inspect
    import re

    from vllm_torchtpu.layers.vllm.quantization import fp8 as fp8_mod

    for module in (fused_moe_ep, fp8_mod):
        offenders = re.findall(r"^\s*(?:self\.)?moe_kernel\s*=.*$",
                               inspect.getsource(module), re.M)
        assert not offenders, (
            f"{module.__name__} assigns moe_kernel ({offenders}); that tells "
            "the runner to skip the all-reduce this path depends on")


def test_a_layer_whose_late_all_reduce_was_deferred_is_refused(monkeypatch):
    """`reduce_results=False` on the MoE layer -> `skip_final_all_reduce` ->
    nothing reassembles the row slices. Refused at weight load, by name,
    because there is no assertion downstream that can catch it: vLLM's own one
    is gated on `moe_kernel`, which this path deliberately leaves None.

    Reachable at dp_size == 1 with any model that forwards
    `reduce_results=False` into `FusedMoEFactory` (DeepSeek-V2/V3 in this vLLM;
    Qwen3.5 does not).
    """
    monkeypatch.setattr(fused_moe_ep, "_replica_group", lambda: (1, 0))
    moe = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True),
                          ep_size=8,
                          dp_size=8,
                          num_experts=64,
                          experts_per_token=8,
                          is_sequence_parallel=False,
                          pcp_size=1,
                          skip_final_all_reduce=True,
                          is_lora_enabled=False)
    layer = SimpleNamespace(moe_config=moe,
                            use_grouped_topk=False,
                            hash_indices_table=None,
                            e_score_correction_bias=None,
                            routed_scaling_factor=1.0,
                            scoring_func="softmax",
                            _tpu_activation_str="silu",
                            has_bias=False)

    reason = fused_moe_ep.unsupported_reason(layer)
    assert reason is not None and "all-reduce" in reason, reason

    moe.skip_final_all_reduce = False
    assert fused_moe_ep.unsupported_reason(layer) is None, (
        "the guard must refuse the deferral and nothing else")

    # The neighbouring pair-consistency guard: vLLM's LoRA layer asserts
    # moe_kernel is not None wherever supports_internal_mk is True, and this
    # path is (True, None) -- so LoRA is refused here rather than asserted
    # inside vllm/lora/layers/fused_moe.py.
    moe.is_lora_enabled = True
    assert "LoRA" in (fused_moe_ep.unsupported_reason(layer) or "")
