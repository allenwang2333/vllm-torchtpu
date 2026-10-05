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
"""Which calls the small-collective bridge hands to a kernel.

The kernels themselves are tested on TPU in
tests/kernels/experimental/small_collectives; the warmup hand-over between the
two sides of the row fork in tests/compilation/test_shape_variants.py.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

pytestmark = pytest.mark.cpu_test

CORES = 8
MESH = tuple(range(CORES))
HIDDEN = 4096


@pytest.fixture
def armed(monkeypatch):
    """Both ops replaced by mocks, inside a warmup trace."""
    ar, rs = Mock(name="all_reduce_op"), Mock(name="reduce_scatter_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    return ar, rs


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize("rows", [1, 8, 64, 96])
def test_small_block_takes_the_all_reduce_kernel(armed, rows):
    ar, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, _fallback) is ar.return_value
    (block, rank), _ = ar.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize("rows", [1, 16, 40])
def test_small_combine_takes_the_reduce_scatter_kernel(armed, rows):
    _, rs = armed
    x = _block(rows * CORES)
    assert op.dp_combine_or(x, _fallback) is rs.return_value
    (block, rank), _ = rs.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize(
    "call,rows",
    [
        (lambda x: op.all_reduce_or(x, MESH, _fallback), op.MAX_ROWS + 1),
        (lambda x: op.dp_combine_or(x, _fallback), CORES * op.COMBINE_MAX_ROWS + CORES),
    ],
    ids=["all_reduce", "dp_combine"],
)
def test_large_block_falls_back_and_reads_the_rank(armed, call, rows):
    ar, rs = armed
    x = _block(rows)
    out = call(x)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    ar.assert_not_called()
    rs.assert_not_called()


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(40, True), (41, False)])
def test_combine_cap_is_40_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, rs = armed
    op.dp_combine_or(_block(rows_per_engine * CORES), _fallback)
    assert rs.called is armed_expected


@pytest.mark.parametrize("rows,armed_expected", [(96, True), (97, False)])
def test_all_reduce_cap_is_96_rows(armed, rows, armed_expected):
    ar, _ = armed
    op.all_reduce_or(_block(rows), MESH, _fallback)
    assert ar.called is armed_expected


def test_large_block_fallback_depends_on_the_rank_buffer(armed, monkeypatch):
    """The retain is a live read: a NaN rank buffer poisons the result, which a
    read the compiler could drop would not."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    out = op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, _fallback)
    assert torch.isnan(out).all()


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), MESH),  # dtype
        (_block(8, hidden=HIDDEN + 128), MESH),  # width does not split over cores
        (_block(8).reshape(2, 4, HIDDEN), MESH),  # not 2-D
        (_block(8), (0, 1, 2, 3)),  # another group than the mesh
    ],
    ids=["fp32", "width", "3d", "group"],
)
def test_ineligible_all_reduce_falls_back_untouched(armed, x, ranks):
    ar, _ = armed
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    fallback.assert_called_once_with(x)
    ar.assert_not_called()


@pytest.mark.parametrize("outside", ["not_compiling", "no_warmup"])
def test_traces_outside_the_backbone_warmup_keep_the_collective(
    armed, monkeypatch, outside
):
    """The MTP drafter is compiled outside the warmup, which could not fork it."""
    ar, rs = armed
    if outside == "not_compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    ar.assert_not_called()
    rs.assert_not_called()


def test_combine_needs_the_dp_group_to_be_the_mesh(armed, monkeypatch):
    _, rs = armed
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    x = _block(8 * CORES)
    fallback = Mock(return_value=object())
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    rs.assert_not_called()


def test_unarmed_is_the_fallback(monkeypatch):
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value


def test_prebuild_is_a_no_op_with_the_flag_off(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", False)
    build = Mock()
    monkeypatch.setattr(op, "build_ep_mesh", build)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild()
    build.assert_not_called()
    assert op._AR_OP is None


def test_prebuild_refuses_a_deployment_without_a_mesh(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op, "_mesh_group", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(NotImplementedError, match="TPU_SMALL_ALL_REDUCE"):
        op.prebuild()


@pytest.fixture
def groups(monkeypatch):
    """vLLM's TP group (ranks 0..7, this worker rank 3) and the two meshes."""
    from types import SimpleNamespace

    from vllm.distributed import parallel_state

    tp = SimpleNamespace(ranks=list(MESH), world_size=CORES)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp)
    ep_mesh, tp_mesh = Mock(name="ep_mesh"), Mock(name="tp_mesh")
    build_ep, build_tp = Mock(return_value=ep_mesh), Mock(return_value=tp_mesh)
    monkeypatch.setattr(op, "build_ep_mesh", build_ep)
    monkeypatch.setattr(op, "build_tp_mesh", build_tp)
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 3)
    monkeypatch.setattr(op, "tp_mesh_index", lambda: 3)
    return SimpleNamespace(
        tp=tp, ep_mesh=ep_mesh, tp_mesh=tp_mesh, build_ep=build_ep, build_tp=build_tp
    )


def test_an_ep_group_keeps_the_ep_mesh(monkeypatch, groups):
    """vLLM builds an EP group for every MoE model, expert parallelism on or
    off; whenever it exists the kernels run over its mesh, as before."""
    from types import SimpleNamespace

    ep = SimpleNamespace(ranks=list(MESH), world_size=CORES)
    monkeypatch.setattr(op, "get_ep_group", lambda: ep)
    assert op._mesh_group() == (groups.ep_mesh, MESH, 3)
    groups.build_tp.assert_not_called()


def test_no_ep_group_takes_the_tp_mesh(monkeypatch, groups):
    """A dense model has no EP group: the all-reduce is still over the TP
    group, so the kernels run over a mesh of the TP group."""
    monkeypatch.setattr(op, "get_ep_group", lambda: None)
    assert op._mesh_group() == (groups.tp_mesh, MESH, 3)
    groups.build_ep.assert_not_called()


def test_one_core_has_no_mesh(monkeypatch, groups):
    monkeypatch.setattr(op, "get_ep_group", lambda: None)
    groups.build_tp.return_value = None
    assert op._mesh_group() is None


@pytest.mark.parametrize("ep_on", [True, False], ids=["ep_group", "tp_only"])
def test_prebuild_arms_the_tp_all_reduce_over_the_group(monkeypatch, groups, ep_on):
    """Either mesh arms the TP all-reduce: its ranks are the TP group's, so
    `all_reduce_or` hands the TP communicator's small blocks to the kernel."""
    from types import SimpleNamespace

    from vllm.distributed import parallel_state

    ep = SimpleNamespace(ranks=list(MESH), world_size=CORES) if ep_on else None
    monkeypatch.setattr(op, "get_ep_group", lambda: ep)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(ranks=[3])
    )
    mesh = groups.ep_mesh if ep_on else groups.tp_mesh
    mesh.shape = {op.EP_AXIS_NAME: CORES}
    built = []
    monkeypatch.setattr(op, "_build_all_reduce", lambda m: built.append(m) or "ar")
    monkeypatch.setattr(op, "_build_reduce_scatter", lambda m: built.append(m) or "rs")
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda data, **kw: real_tensor(data, dtype=kw.get("dtype"))
    )
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    for name in ("_AR_OP", "_RS_OP", "_RANK"):
        monkeypatch.setattr(op, name, None)
    monkeypatch.setattr(op, "_CORES", 0)
    monkeypatch.setattr(op, "_MESH_RANKS", ())
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    op.prebuild()
    assert built == [mesh, mesh]
    assert (op._AR_OP, op._RS_OP, op._CORES) == ("ar", "rs", CORES)
    assert op._MESH_RANKS == tuple(groups.tp.ranks)
    assert op._RANK.tolist() == [3]
    assert op._DP_COMBINE is False


def test_retain_is_the_identity_when_unarmed_or_outside_the_warmup(monkeypatch):
    x = _block(8)
    monkeypatch.setattr(op, "_RANK", None)
    assert op.retain(x) is x
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.retain(x) is x
