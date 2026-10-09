# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.53 pipeline hand-off & DCP optimizations and 100% coverage."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import dcp, pp_push, pp_shift, pp_wave


def test_dcp_full_coverage(monkeypatch):
    """Verify 100% statement and branch coverage of dcp.py."""
    from vllm.distributed import parallel_state

    fake_group = SimpleNamespace(ranks=[0, 1], world_size=2, rank_in_group=1)
    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: fake_group)
    assert dcp.get_dcp_group() is fake_group
    assert dcp.get_dcp_rank() == 1
    assert dcp.get_dcp_world_size() == 2

    def _raise_assertion():
        raise AssertionError("DCP group not initialized")

    monkeypatch.setattr(parallel_state, "get_dcp_group", _raise_assertion)
    assert dcp.get_dcp_group() is None
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1

    monkeypatch.setattr(dcp, "get_dcp_group", lambda: fake_group)
    monkeypatch.setattr(dcp, "get_cp_group_layout", lambda g: ("layout", g))
    assert dcp.get_dcp_group_layout() == ("layout", fake_group)

    monkeypatch.setattr(
        dcp, "get_or_create_cp_mesh", lambda axis_name, group: (axis_name, group)
    )
    assert dcp.get_or_create_dcp_mesh() == (dcp.DCP_AXIS_NAME, fake_group)
    assert dcp.get_or_create_dcp_mesh(axis_name="custom_dcp") == (
        "custom_dcp",
        fake_group,
    )


def test_pp_shift_full_coverage(monkeypatch):
    """Verify O(P) rank-to-position lookup, mesh ID caching, and 100% coverage."""
    from vllm.distributed import parallel_state

    monkeypatch.setattr(pp_shift, "_MESH", None)
    monkeypatch.setattr(pp_shift, "_POS_OF_RANK", ())
    monkeypatch.setattr(pp_shift, "_MESH_DEVICE_IDS", {})
    monkeypatch.setattr(pp_shift, "_OPS", {})

    fake_group = SimpleNamespace(ranks=[0, 1, 2])
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: fake_group)
    monkeypatch.setattr(pp_shift, "_get_current_global_rank", lambda: 0)
    monkeypatch.setattr(pp_shift, "_get_tpu_global_device_id", lambda: 20)
    monkeypatch.setattr(
        pp_shift,
        "_collect_rank_to_device_id",
        lambda group, rank, dev_id: {0: 20, 1: 10, 2: 30},
    )

    fake_devs = [SimpleNamespace(id=10), SimpleNamespace(id=20), SimpleNamespace(id=30)]
    monkeypatch.setattr(pp_shift.jax, "devices", lambda: fake_devs)

    class _FakeMesh:
        def __init__(self, devices, axis_names):
            self.devices = list(devices.flat)
            self.axis_names = axis_names

    monkeypatch.setattr(pp_shift, "Mesh", _FakeMesh)

    # First call builds _MESH and populates _POS_OF_RANK + _MESH_DEVICE_IDS
    mesh = pp_shift.get_or_create_pp_mesh()
    assert isinstance(mesh, _FakeMesh)
    # Second call hits cached _MESH
    assert pp_shift.get_or_create_pp_mesh() is mesh
    # Device order is [10, 20, 30]: rank 0->pos 1, rank 1->pos 0, rank 2->pos 2
    assert pp_shift.pp_positions() == (1, 0, 2)
    assert pp_shift._MESH_DEVICE_IDS[id(mesh)] == (10, 20, 30)

    with pytest.raises(ValueError, match="at least one tensor"):
        pp_shift.pp_permute_op(mesh, num_tensors=0)

    monkeypatch.setattr(pp_shift, "shard_map", lambda fn, **kw: fn)
    monkeypatch.setattr(
        pp_shift.jax.lax, "ppermute", lambda x, axis, perm: (x, axis, perm)
    )
    monkeypatch.setattr(pp_shift, "sharded_jax_op", lambda name, fn, **kw: fn)

    # Uncached mesh object exercises `if dev_ids is None:` True branch
    fresh_mesh = SimpleNamespace(
        devices=[SimpleNamespace(id=10), SimpleNamespace(id=20), SimpleNamespace(id=30)]
    )
    op1 = pp_shift.pp_permute_op(fresh_mesh, num_tensors=1)
    assert pp_shift._MESH_DEVICE_IDS[id(fresh_mesh)] == (10, 20, 30)
    # Second call exercises `dev_ids is None` False and `op is not None` True
    assert pp_shift.pp_permute_op(fresh_mesh, num_tensors=1) is op1
    res1 = op1("tensor0")
    assert res1 == ("tensor0", "pp", [(1, 0), (0, 2), (2, 1)])

    # Multi-tensor op on cached fresh_mesh exercises `num_tensors > 1` branch
    op2 = pp_shift.pp_permute_op(fresh_mesh, num_tensors=2)
    assert pp_shift.pp_permute_op(fresh_mesh, num_tensors=2) is op2
    res2 = op2("t0", "t1")
    assert res2 == (
        ("t0", "pp", [(1, 0), (0, 2), (2, 1)]),
        ("t1", "pp", [(1, 0), (0, 2), (2, 1)]),
    )


def test_pp_push_full_coverage(monkeypatch):
    """Verify 100% statement and branch coverage of pp_push.py."""
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.v1.executor import multiproc_executor

    monkeypatch.setattr(pp_push, "_ring_chunks_needed", 0)

    class _Queue:
        def __init__(self, n_reader, n_local_reader, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)

    # Test widen_message_rings(None), stages=1, stages=4, and patched MessageQueue
    pp_push.widen_message_rings(None)
    cfg1 = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
    cfg4 = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=4))
    pp_push.widen_message_rings(cfg1)
    pp_push.widen_message_rings(cfg4)
    pp_push.widen_message_rings(cfg4)
    assert _Queue(1, 1).kwargs["max_chunks"] == 24

    # Test _WaveState branches including settles > 5 and pushes > 5
    exec_stub = SimpleNamespace(
        output_rank=3,
        calls=[],
        collective_rpc=lambda method, **kw: exec_stub.calls.append((method, kw)),
    )
    wave = pp_push._WaveState(stages=4)
    # Closed wave: before_waiting returns early
    wave.before_waiting(exec_stub, 1, 1)
    # Settle when settles == 6 (exercises False branch of debug log condition)
    wave.settles = 5
    s1 = wave.dispatched()
    wave.before_waiting(exec_stub, s1, wave.real_steps)
    assert wave.settles == 6

    # Push when pushes == 6 (exercises False branch of debug log condition)
    wave.pushes = 5
    s2 = wave.dispatched()
    s3 = wave.dispatched()
    # need = 4 - 2 - (3 - 2) = 1 -> wave.pushes becomes 6
    wave.before_waiting(exec_stub, s2, s2)
    assert wave.pushes == 6
    # Calling before_waiting again on s2 has need <= 0
    wave.before_waiting(exec_stub, s2, s2)
    # Settle on s3 when settles == 1 (exercises True branch of `self.settles <= 5`)
    wave.settles = 0
    wave.before_waiting(exec_stub, s3, wave.real_steps)
    assert wave.settles == 1
    # Push when pushes == 1 (exercises True branch of `self.pushes <= 5`)
    wave.pushes = 0
    s4 = wave.dispatched()
    r4 = wave.real_steps
    _ = wave.dispatched()
    wave.before_waiting(exec_stub, s4, r4)
    assert wave.pushes == 1

    # _PushFuture methods
    f_ok = Future()
    pf = pp_push._PushFuture(f_ok, exec_stub, pp_push._WaveState(stages=2), 1, 1)
    seen = []
    pf.add_done_callback(seen.append)
    assert not pf.done() and not pf.running()
    f_ok.set_result("val")
    assert pf.done() and seen == [pf]
    assert pf.result() == "val"
    assert pf.exception() is None

    f_err = Future()
    f_err.set_exception(ValueError("err"))
    pf_err = pp_push._PushFuture(f_err, exec_stub, pp_push._WaveState(stages=2), 1, 1)
    assert isinstance(pf_err.exception(), ValueError)

    f_cancel = Future()
    pf_cancel = pp_push._PushFuture(
        f_cancel, exec_stub, pp_push._WaveState(stages=2), 1, 1
    )
    assert pf_cancel.cancel() and pf_cancel.cancelled()

    # patch_executor_for_pp_wave branches
    pp_push.patch_executor_for_pp_wave(None)
    pp_push.patch_executor_for_pp_wave(cfg1)

    class _FakeMultiprocExecutor:
        def __init__(self):
            self.vllm_config = cfg4
            self.output_rank = 3
            self.calls = []

        def execute_model(self, scheduler_output, non_block=False):
            fut = Future()
            fut.set_result(("exec", scheduler_output.total_num_scheduled_tokens))
            return fut if non_block else fut.result()

        def sample_tokens(self, grammar_output, non_block=False):
            fut = Future()
            fut.set_result(("sample", grammar_output))
            return fut if non_block else fut.result()

        def collective_rpc(self, method, **kwargs):
            self.calls.append((method, kwargs))

    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _FakeMultiprocExecutor)
    pp_push.patch_executor_for_pp_wave(cfg4)
    # Second call hits already-patched guard
    pp_push.patch_executor_for_pp_wave(cfg4)

    ex = _FakeMultiprocExecutor()
    assert ex.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) == (
        "exec",
        0,
    )
    assert ex.execute_model(SimpleNamespace(total_num_scheduled_tokens=8)) == (
        "exec",
        8,
    )
    fut_exec = ex.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=8), non_block=True
    )
    assert fut_exec.result() == ("exec", 8)
    assert ex.sample_tokens("g1") == ("sample", "g1")
    fut_sample = ex.sample_tokens("g2", non_block=True)
    assert fut_sample.result() == ("sample", "g2")


def test_pp_wave_full_coverage_main_thread(monkeypatch):
    """Verify 100% statement and branch coverage of pp_wave.py on the main thread."""
    from torch_tpu._internal.sync import sync
    from vllm.distributed import parallel_state

    # 1. pp_rank_flags for PP=1 and PP=2
    assert pp_wave.pp_rank_flags(SimpleNamespace(pipeline_parallel_size=1)) == (
        True,
        True,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=False, is_last_rank=True, rank_in_group=1, world_size=2
        ),
    )
    assert pp_wave.pp_rank_flags(SimpleNamespace(pipeline_parallel_size=2)) == (
        False,
        True,
    )

    # 2. _launch_now with Tensor, tuple (fast path), and list return values
    sync_calls = []
    monkeypatch.setattr(
        sync, "synchronize", lambda t, wait=False: sync_calls.append((t, wait))
    )
    t0 = torch.zeros(2, 4)
    t1 = torch.ones(2, 4)
    assert pp_wave._launch_now(lambda x: x, (t0,)) == (t0,)
    assert pp_wave._launch_now(lambda x, y: (x, y), (t0, t1)) == (t0, t1)
    assert pp_wave._launch_now(lambda x, y: [x, y], (t0, t1)) == (t0, t1)
    assert len(sync_calls) == 6

    # 3. PPWave.__init__ validation error
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(
        pp_wave,
        "pp_permute_op",
        lambda mesh, num_tensors=1: (lambda *xs: xs[0] if len(xs) == 1 else xs),
    )
    with pytest.raises(ValueError, match="hidden_states and residual"):
        pp_wave.PPWave(
            torch.device("cpu"),
            8,
            {"hidden_states": ((4,), torch.float32)},
        )

    # 4. Synchronous 3-stage PPWave lifecycle without and with side tensor
    for with_side in (False, True):
        template = {
            "hidden_states": ((4,), torch.float32),
            "residual": ((4,), torch.float32),
        }
        if with_side:
            template["topk_indices"] = ((2,), torch.int32)

        waves = []
        for r in range(3):
            monkeypatch.setattr(
                parallel_state,
                "get_pp_group",
                lambda r=r: SimpleNamespace(rank_in_group=r, world_size=3),
            )
            # Make op return expected warmup value `float(r)` on rank > 0
            monkeypatch.setattr(
                pp_wave,
                "pp_permute_op",
                lambda mesh, num_tensors=1, r=r: (
                    lambda *xs: tuple(torch.full_like(x, float(r)) for x in xs)
                ),
            )
            w = pp_wave.PPWave(torch.device("cpu"), 8, template)
            assert w._carried_tail == (("topk_indices",) if with_side else ())
            w.warmup()
            waves.append(w)

        # Also test warmup self-test failure on rank 1
        monkeypatch.setattr(
            parallel_state,
            "get_pp_group",
            lambda: SimpleNamespace(rank_in_group=1, world_size=3),
        )
        monkeypatch.setattr(
            pp_wave,
            "pp_permute_op",
            lambda mesh, num_tensors=1: (
                lambda *xs: tuple(torch.full_like(x, 99.0) for x in xs)
            ),
        )
        w_bad = pp_wave.PPWave(torch.device("cpu"), 8, template)
        with pytest.raises(RuntimeError, match="hand-off self-test tensor"):
            w_bad.warmup()

        w0, w1, w2 = waves
        # Settle when burst is not open -> noop
        w0.settle()

        # Forward 1 with rows < max_rows (exercises _pad concatenation)
        assert w0.start_forward(4) is None
        # Calling _open_burst when already open -> early return
        w0._open_burst()
        out_tensors = {
            "hidden_states": torch.ones(4, 4),
            "residual": torch.ones(4, 4),
        }
        if with_side:
            out_tensors["topk_indices"] = torch.ones(4, 2, dtype=torch.int32)
        w0.end_forward(out_tensors, 4)

        # Forward 2 with rows == max_rows (exercises _pad fast path)
        assert w0.start_forward(8) is None
        out_tensors_full = {
            "hidden_states": torch.ones(8, 4),
            "residual": torch.ones(8, 4),
        }
        if with_side:
            out_tensors_full["topk_indices"] = torch.ones(8, 2, dtype=torch.int32)
        w0.end_forward(out_tensors_full, 8)
        w0.push()
        w0.settle()

        # Rank 1 (middle stage: 0 < rank < last)
        inc1 = w1.start_forward(4)
        assert inc1 is not None
        w1.end_forward(out_tensors, 4)
        # Next start_forward consumes w1.pending
        inc1_second = w1.start_forward(8)
        assert inc1_second is not None
        w1.end_forward(out_tensors_full, 8)
        w1.push()
        # Also test push when pending is None on rank 1
        w1.pending = None
        w1.push()
        w1.settle()

        # Rank 2 (last stage: rank == last)
        inc2 = w2.start_forward(4)
        assert inc2 is not None
        w2.end_forward(None, 4)
        w2.push()
        # Set bursts = 5 before next burst so settle hits `self.bursts == 6`
        w2.settle()
        w2.bursts = 5
        w2.push()
        w2.settle()
        assert w2.bursts == 6

    # Missing tensor error in end_forward and _check_count mismatch error
    with pytest.raises(RuntimeError, match="produced no"):
        w0.end_forward({"hidden_states": torch.ones(4, 4)}, 4)
    w0.burst_open = True
    w0.burst_launches = 99
    with pytest.raises(RuntimeError, match="expected"):
        w0._check_count()
