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
                                        replica_rank=torch.tensor(
                                            rank, dtype=torch.int32))
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
                  subc_quant_w2_sz=None)
    monkeypatch.setitem(fused_moe_ep._op_cache, tuple(kwargs.values()),
                        _stub_op)
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
                  subc_quant_w2_sz=1024)
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
