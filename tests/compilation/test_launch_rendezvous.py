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
"""Tests for the cross-rank rendezvous before compiled-graph launches."""

import contextlib
import multiprocessing
import socket
import time

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state

from vllm_torchtpu.compilation import launch_rendezvous


class _World:
    """A stand-in for vLLM's world group that records barrier calls."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.rank_in_group = 0
        self.barriers = 0

    def barrier(self) -> None:
        self.barriers += 1


@pytest.fixture
def world(monkeypatch):
    fake = _World(world_size=8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake)
    return fake


def test_inactive_by_default():
    assert not launch_rendezvous.lockstep_active()


def test_no_rendezvous_outside_the_block(world):
    launch_rendezvous.rendezvous_before_launch()
    assert world.barriers == 0


def test_rendezvous_inside_the_block(world):
    with launch_rendezvous.lockstep_launches():
        assert launch_rendezvous.lockstep_active()
        launch_rendezvous.rendezvous_before_launch()
        launch_rendezvous.rendezvous_before_launch()
    assert world.barriers == 2
    assert not launch_rendezvous.lockstep_active()


def test_nested_blocks_stay_active_until_the_outermost_exits(world):
    with launch_rendezvous.lockstep_launches():
        with launch_rendezvous.lockstep_launches():
            pass
        assert launch_rendezvous.lockstep_active()
        launch_rendezvous.rendezvous_before_launch()
    assert world.barriers == 1
    assert not launch_rendezvous.lockstep_active()


def test_block_deactivates_when_the_body_raises(world):
    with pytest.raises(RuntimeError):
        with launch_rendezvous.lockstep_launches():
            raise RuntimeError("compile failed")
    assert not launch_rendezvous.lockstep_active()


def test_single_rank_world_needs_no_rendezvous(world):
    world.world_size = 1
    with launch_rendezvous.lockstep_launches():
        launch_rendezvous.rendezvous_before_launch()
    assert world.barriers == 0


def test_uninitialized_distributed_needs_no_rendezvous(world, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with launch_rendezvous.lockstep_launches():
        launch_rendezvous.rendezvous_before_launch()
    assert world.barriers == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _GlooWorld:
    """A world whose barrier is the live gloo process group of this process."""

    def __init__(self) -> None:
        self.world_size = torch.distributed.get_world_size()
        self.rank_in_group = torch.distributed.get_rank()

    def barrier(self) -> None:
        torch.distributed.barrier()


def _launch_after_delay(rank: int, world_size: int, port: int, delay: float,
                        lockstep: bool, results) -> None:
    """Join a gloo world, then launch once `delay` seconds after joining."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    parallel_state.get_world_group = _GlooWorld
    try:
        block = (launch_rendezvous.lockstep_launches()
                 if lockstep else contextlib.nullcontext())
        with block:
            time.sleep(delay)
            launch_rendezvous.rendezvous_before_launch()
            results.put((rank, time.monotonic()))
    finally:
        torch.distributed.destroy_process_group()


def _launch_spread(lockstep: bool) -> float:
    """Seconds between the earliest and latest launch of a skewed 2-rank world."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    delays = {0: 1.5, 1: 0.0}
    procs = [
        ctx.Process(target=_launch_after_delay,
                    args=(rank, 2, port, delays[rank], lockstep, results))
        for rank in (0, 1)
    ]
    for proc in procs:
        proc.start()
    launches = dict(results.get(timeout=120) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    return max(launches.values()) - min(launches.values())


def test_lockstep_absorbs_rank_skew():
    # Without the rendezvous the two ranks launch 1.5s apart; with it, the
    # early rank holds its launch until the late rank arrives.
    assert _launch_spread(lockstep=False) > 1.0
    assert _launch_spread(lockstep=True) < 0.5
