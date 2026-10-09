# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that must not retain TPU runtime state in pytest."""

import multiprocessing
import queue
import subprocess
import time
import traceback
from collections.abc import Callable
from typing import Any

import pytest


def _isolated_test_worker(target: Callable[[], None], result_queue: Any) -> None:
    try:
        target()
    except pytest.skip.Exception as exc:
        result_queue.put(("SKIP", str(exc)))
    except BaseException:
        result_queue.put(("ERROR", traceback.format_exc()))
    else:
        result_queue.put(("OK", None))


def run_in_isolated_process(
    target: Callable[[], None], *, timeout: float = 600
) -> None:
    """Run a TPU test in a spawned process and propagate its result.

    PJRT keeps TPU device handles alive for the lifetime of its process. Running
    device tests in a child prevents those handles from blocking later tests
    that start their own engine processes.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_isolated_test_worker, args=(target, result_queue))
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError(f"Isolated TPU test timed out after {timeout} seconds")
        try:
            status, payload = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError(
                "Isolated TPU test exited without reporting a result "
                f"(exitcode={process.exitcode})"
            ) from exc
    finally:
        result_queue.close()

    if status == "SKIP":
        pytest.skip(payload)
    if status == "ERROR":
        raise AssertionError(f"Isolated TPU test failed:\n{payload}")
    if process.exitcode != 0:
        raise AssertionError(f"Isolated TPU test crashed (exitcode={process.exitcode})")


def prepare_tpu_environment(world_size: int | None = None) -> None:
    """Prepare single-host TPU environment variables across torch_tpu versions.

    Backwards-compatible bridge supporting:
      - torch_tpu >= 0.1.2: torch.tpu.distributed.set_tpu_launch_env (public API)
      - torch_tpu <= 0.1.1:
        torch_tpu._internal.distributed.launchers.singlehost_wrapper.prepare_tpu_environment
    """
    import torch

    tpu = getattr(torch, "tpu", None)
    tpu_dist = getattr(tpu, "distributed", None) if tpu is not None else None
    if tpu_dist is not None and hasattr(tpu_dist, "set_tpu_launch_env"):
        tpu_dist.set_tpu_launch_env(nproc_per_node=world_size)
        return

    # TODO: Remove legacy singlehost_wrapper fallback once torch_tpu >= 0.1.2
    # is baseline.
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment as _legacy_prepare,
    )

    _legacy_prepare(world_size=world_size)


# What a worker group prints when libtpu could not start its TPU session. On
# GKE the session goes through the TPU device plugin's vBAR socket, and a group
# spawned while the chips are still resetting after the previous one exited
# gets START_SESSION failed in one process; the others wait out the
# POST_START_SESSION_BARRIER and surface as "PjRtClient is not initialized".
_TPU_SESSION_FAILURE_MARKERS = (
    "PjRtClient is not initialized",
    "START_SESSION failed",
    "POST_START_SESSION_BARRIER",
)


def tpu_session_failed(output: str) -> bool:
    """Whether a worker group's output shows it never got its TPU session."""
    return any(marker in output for marker in _TPU_SESSION_FAILURE_MARKERS)


def run_worker_group_with_retry(
    launch: Callable[[], subprocess.CompletedProcess[str]],
    *,
    attempts: int = 2,
    wait_s: float = 60.0,
    before_retry: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a worker group, relaunching it while it fails to get its TPU session.

    The failure is the device's, not the test's: a fresh group a minute later
    starts normally. `before_retry` clears whatever the failed attempt left for
    the caller to read, such as per-rank result files.
    """
    completed = launch()
    for _ in range(attempts - 1):
        if not tpu_session_failed(completed.stdout or ""):
            break
        print(f"TPU session start failed; relaunching the workers in {wait_s:.0f}s")
        if before_retry is not None:
            before_retry()
        time.sleep(wait_s)
        completed = launch()
    return completed
