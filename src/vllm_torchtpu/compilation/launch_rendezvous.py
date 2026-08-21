# SPDX-License-Identifier: Apache-2.0
"""Lockstep launch of compiled graphs across the ranks of one model.

A compiled graph that carries collectives (TP all-reduce, PCP ring attention,
EP all-to-all) only completes once every rank has launched it. During warmup
each rank compiles a graph before launching it, and compile time differs by
minutes from rank to rank, so an early rank's program would otherwise sit on
the device waiting for peers that are still compiling. The TPU runtime treats
a program that stays pending for minutes as a slow-operation incident and
answers it with a diagnostic core dump of the worker process, which freezes
that worker for tens of seconds and skews the next launch even further.

Inside a ``lockstep_launches`` block every compiled-graph dispatch first meets
the other ranks on the CPU process group, so all ranks enqueue the program
within milliseconds of each other and no device program waits on a host.
"""

import contextlib
import time
from collections.abc import Iterator

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_lockstep_depth = 0

# Waits below this are the normal cost of the rendezvous; longer ones are the
# rank skew it absorbed and are worth a line in the log.
_LOGGED_WAIT_SECONDS = 1.0


def lockstep_active() -> bool:
    """Whether compiled-graph launches currently rendezvous across ranks."""
    return _lockstep_depth > 0


@contextlib.contextmanager
def lockstep_launches() -> Iterator[None]:
    """Make every compiled-graph launch inside the block wait for all ranks.

    Every rank of the model's world must run the same sequence of launches
    inside the block; a launch that only some ranks perform would leave the
    others waiting forever.
    """
    global _lockstep_depth
    _lockstep_depth += 1
    try:
        yield
    finally:
        _lockstep_depth -= 1


def rendezvous_before_launch() -> None:
    """Block until every rank of the model's world reaches this launch.

    No-op outside ``lockstep_launches`` and in a single-rank world.
    """
    if _lockstep_depth == 0:
        return
    if not torch.distributed.is_initialized():
        return
    from vllm.distributed.parallel_state import get_world_group
    world = get_world_group()
    if world.world_size <= 1:
        return
    start = time.perf_counter()
    world.barrier()
    waited = time.perf_counter() - start
    if waited >= _LOGGED_WAIT_SECONDS:
        logger.info(
            "Rank %d waited %.1fs for peer ranks before launching a compiled "
            "graph.", world.rank_in_group, waited)
