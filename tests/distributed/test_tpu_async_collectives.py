from types import SimpleNamespace

import torch

import vllm_torchtpu.distributed.tpu_communicator as communicator


def _bare_communicator():
    return communicator.TpuDeviceCommunicator.__new__(
        communicator.TpuDeviceCommunicator)


def test_async_collectives_launch_before_explicit_wait(monkeypatch):
    events = []
    group = object()
    monkeypatch.setattr(communicator, "get_dp_group", lambda: SimpleNamespace(
        world_size=8, device_group=group))

    def all_gather(value, group_size, group_name):
        assert group_size == 8
        assert group_name == "test-group"
        events.append(("AG-start", int(value.item())))
        return value + 100

    def reduce_scatter(value, group_size, group_name):
        assert group_size == 8
        assert group_name == "test-group"
        events.append(("RS-start", int(value.item())))
        return value + 200

    def wait(value):
        events.append(("wait", int(value.item())))
        return value

    monkeypatch.setattr(communicator, "_raw_all_gather_tensor", all_gather)
    monkeypatch.setattr(communicator, "_raw_reduce_scatter_tensor",
                        reduce_scatter)
    monkeypatch.setattr(communicator.funcol, "_resolve_group_name",
                        lambda value: "test-group")
    monkeypatch.setattr(communicator.funcol, "wait_tensor", wait)

    comm = _bare_communicator()
    handles = [
        comm.all_gather_start(torch.tensor(0), 0),
        comm.all_gather_start(torch.tensor(1), 1),
        comm.reduce_scatter_start(torch.tensor(0), 0),
        comm.reduce_scatter_start(torch.tensor(1), 1),
    ]
    assert [(h.operation, h.microbatch_index) for h in handles] == [
        ("AG", 0), ("AG", 1), ("RS", 0), ("RS", 1)]
    assert events == [("AG-start", 0), ("AG-start", 1),
                      ("RS-start", 0), ("RS-start", 1)]
    assert all(handle.retained_tensors for handle in handles)

    outputs = [comm.wait(handle) for handle in handles]
    assert [int(value.item()) for value in outputs] == [100, 101, 200, 201]
    assert all(not handle.retained_tensors for handle in handles)
    # Waiting twice is idempotent and must not emit a second wait op.
    comm.wait(handles[0])
    assert len([event for event in events if event[0] == "wait"]) == 4


def test_single_rank_async_collective_is_identity(monkeypatch):
    monkeypatch.setattr(communicator, "get_dp_group", lambda: SimpleNamespace(
        world_size=1, device_group=None))
    value = torch.tensor([1, 2])
    comm = _bare_communicator()
    gather = comm.all_gather_start(value, 0)
    scatter = comm.reduce_scatter_start(value, 0)
    assert comm.wait(gather) is value
    assert comm.wait(scatter) is value
