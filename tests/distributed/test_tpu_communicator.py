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

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.distributed import tpu_communicator as comm


@pytest.fixture
def communicator(monkeypatch):
    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    return comm.TpuDeviceCommunicator()


@pytest.mark.parametrize("use_all2all", [False, True])
def test_all2all_probe_manager(monkeypatch, use_all2all):
    cpu_group = object()

    def init(instance, *args, **kwargs):
        instance.use_all2all = use_all2all
        instance.cpu_group = cpu_group

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    manager = Mock()
    monkeypatch.setattr(comm, "TpuNullAll2AllManager", manager)
    result = comm.TpuDeviceCommunicator()
    if use_all2all:
        manager.assert_called_once_with(cpu_group)
        assert result.all2all_manager is manager.return_value
    else:
        manager.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
@pytest.mark.parametrize("dp_size", [1, 2])
def test_dispatch_and_combine_use_dp_group(
    monkeypatch, communicator, method, count, dp_size
):
    group = Mock(world_size=dp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    group.reduce_scatter.return_value = torch.tensor([[11, 12]])
    result = getattr(communicator, method)(*tensors)
    if method == "combine":
        if dp_size == 1:
            assert result is tensors[0]
            group.reduce_scatter.assert_not_called()
        else:
            group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
            assert result is group.reduce_scatter.return_value
        group.all_gather.assert_not_called()
    elif dp_size == 1:
        assert len(result) == count
        assert all(actual is expected for actual, expected in zip(result, tensors))
        group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert group.all_gather.call_count == count
        group.reduce_scatter.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_extra_tensors_rejected_before_collectives(
    monkeypatch, communicator, method, count
):
    get_group = Mock()
    monkeypatch.setattr(comm, "get_dp_group", get_group)
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        getattr(communicator, method)(*[torch.ones(1)] * count, extra_tensors=[])
    get_group.assert_not_called()


def test_moe_group_with_sequence_parallel(monkeypatch, communicator):
    ep_group = Mock(world_size=8)
    dp_group = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    assert communicator._moe_group(is_sequence_parallel=True) is ep_group


def test_moe_group_without_sequence_parallel(monkeypatch, communicator):
    dp_group_1 = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_1)
    assert communicator._moe_group(is_sequence_parallel=False) is None

    dp_group_4 = Mock(world_size=4)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_4)
    assert communicator._moe_group(is_sequence_parallel=False) is dp_group_4


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
@pytest.mark.parametrize("ep_size", [1, 4])
def test_dispatch_and_combine_use_ep_group_when_sequence_parallel(
    monkeypatch, communicator, method, count, ep_size
):
    dp_group = Mock(world_size=1)
    ep_group = Mock(world_size=ep_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    # Alternate dtypes so dispatch_router_logits stays on the unpacked path.
    dtypes = [torch.float32, torch.bfloat16, torch.float32]
    tensors = [torch.tensor([[i, i + 1]], dtype=dtypes[i]) for i in range(count)]
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    ep_group.reduce_scatter.return_value = torch.tensor([[11, 12]])
    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    dp_group.all_gather.assert_not_called()
    dp_group.reduce_scatter.assert_not_called()
    if method == "combine":
        if ep_size == 1:
            assert result is tensors[0]
            ep_group.reduce_scatter.assert_not_called()
        else:
            ep_group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
            assert result is ep_group.reduce_scatter.return_value
        ep_group.all_gather.assert_not_called()
    elif ep_size == 1:
        assert len(result) == count
        assert all(actual is expected for actual, expected in zip(result, tensors))
        ep_group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, ep_group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert ep_group.all_gather.call_count == count
        ep_group.reduce_scatter.assert_not_called()


def test_all_reduce_goes_through_the_small_collective_bridge(monkeypatch, communicator):
    base = Mock(return_value=torch.tensor([7.0]))
    monkeypatch.setattr(comm.DeviceCommunicatorBase, "all_reduce", base)
    bridge = Mock(side_effect=lambda x, ranks, fallback: fallback(x))
    monkeypatch.setattr(comm.small_all_reduce_op, "all_reduce_or", bridge)
    communicator.ranks = [0, 1]
    x = torch.ones(2, 4)
    assert communicator.all_reduce(x) is base.return_value
    (block, ranks, _), _ = bridge.call_args
    assert block is x and ranks == [0, 1]
    base.assert_called_once_with(x)


def test_all_reduce_unarmed_is_the_base_all_reduce(monkeypatch, communicator):
    base = Mock(return_value=torch.tensor([7.0]))
    monkeypatch.setattr(comm.DeviceCommunicatorBase, "all_reduce", base)
    monkeypatch.setattr(comm.small_all_reduce_op, "_AR_OP", None)
    communicator.ranks = [0, 1]
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    assert communicator.all_reduce(x) is base.return_value
    base.assert_called_once_with(x)


def test_combine_goes_through_the_small_collective_bridge(monkeypatch, communicator):
    group = Mock(world_size=2)
    group.reduce_scatter.return_value = torch.tensor([[11.0, 12.0]])
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    bridge = Mock(side_effect=lambda x, fallback: fallback(x))
    monkeypatch.setattr(comm.small_all_reduce_op, "dp_combine_or", bridge)
    x = torch.ones(2, 2)
    assert communicator.combine(x) is group.reduce_scatter.return_value
    assert bridge.call_args.args[0] is x
    group.reduce_scatter.assert_called_once_with(x, dim=0)


def _sp_groups(monkeypatch, dp_size, tp_size):
    dp_group = Mock(world_size=dp_size)
    ep_group = Mock(world_size=dp_size * tp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat(
        [x + 100 * r for r in range(ep_group.world_size)], dim=dim
    )
    return dp_group, ep_group


@pytest.mark.parametrize("dp_size,tp_size", [(1, 4), (2, 2)])
def test_sequence_parallel_packs_single_gather(
    monkeypatch, communicator, dp_size, tp_size
):
    """Matching dtypes: hidden_states and router_logits share one EP gather."""
    _, ep_group = _sp_groups(monkeypatch, dp_size=dp_size, tp_size=tp_size)
    chunk = torch.arange(3 * 4, dtype=torch.bfloat16).reshape(3, 4)
    logits = torch.arange(3 * 2, dtype=torch.bfloat16).reshape(3, 2) + 50

    out_hidden, out_logits = communicator.dispatch_router_logits(
        chunk, logits, is_sequence_parallel=True
    )

    ep_group.all_gather.assert_called_once()
    packed = ep_group.all_gather.call_args.args[0]
    assert packed.shape == (3, 6)
    expected_hidden = torch.cat([chunk + 100 * r for r in range(dp_size * tp_size)])
    expected_logits = torch.cat([logits + 100 * r for r in range(dp_size * tp_size)])
    torch.testing.assert_close(out_hidden, expected_hidden)
    torch.testing.assert_close(out_logits, expected_logits)


def test_sequence_parallel_mixed_dtype_uses_two_gathers(monkeypatch, communicator):
    _, ep_group = _sp_groups(monkeypatch, dp_size=2, tp_size=2)
    chunk = torch.zeros(3, 4, dtype=torch.bfloat16)
    logits = torch.zeros(3, 2, dtype=torch.float32)

    out_hidden, out_logits = communicator.dispatch_router_logits(
        chunk, logits, is_sequence_parallel=True
    )

    assert ep_group.all_gather.call_count == 2
    assert ep_group.all_gather.call_args_list[0].args[0] is chunk
    assert ep_group.all_gather.call_args_list[1].args[0] is logits
    assert out_hidden.dtype == torch.bfloat16 and out_logits.dtype == torch.float32
    assert out_hidden.shape == (12, 4) and out_logits.shape == (12, 2)


def test_sequence_parallel_3d_hidden_uses_two_gathers(monkeypatch, communicator):
    """Packing needs 2-D operands; a [1, T/TP, H] hidden falls back cleanly."""
    _, ep_group = _sp_groups(monkeypatch, dp_size=1, tp_size=2)
    chunk = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
    logits = torch.zeros(3, 2, dtype=torch.bfloat16)

    communicator.dispatch_router_logits(chunk, logits, is_sequence_parallel=True)

    assert ep_group.all_gather.call_count == 2


def test_non_sequence_parallel_dispatch_never_packs(monkeypatch, communicator):
    """The DP-only path keeps the two separate gathers even with equal dtypes."""
    dp_group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    dp_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)

    communicator.dispatch_router_logits(torch.zeros(4, 4), torch.zeros(4, 2))

    assert dp_group.all_gather.call_count == 2
