# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that must not retain TPU runtime state in pytest."""

import multiprocessing
import queue
import traceback
from collections.abc import Callable
from typing import Any

import pytest


def _isolated_test_worker(target: Callable[[], None],
                          result_queue: Any) -> None:
    import os
    lockfile = "/tmp/libtpu_lockfile"
    if os.path.exists(lockfile):
        try:
            os.remove(lockfile)
        except OSError:
            pass
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        target()
    except pytest.skip.Exception as exc:
        result_queue.put(("SKIP", str(exc)))
    except BaseException:
        result_queue.put(("ERROR", traceback.format_exc()))
    else:
        result_queue.put(("OK", None))

    # XLA/libtpu crashes with a SIGSEGV in tcmalloc::ThreadCache::ReleaseToCentralCache
    # during teardown of Python `spawn` multiprocessing workers. We must brutally terminate
    # the process skipping standard C++ static destructors to prevent the crash from failing the CI.
    try:
        result_queue.close()
        result_queue.join_thread()
    except Exception:
        pass
    import os
    os._exit(0)


def run_in_isolated_process(target: Callable[[], None],
                            *,
                            timeout: float = 600) -> None:
    """Run a TPU test in a spawned process and propagate its result.

    PJRT keeps TPU device handles alive for the lifetime of its process. Running
    device tests in a child prevents those handles from blocking later tests
    that start their own engine processes.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_isolated_test_worker,
                          args=(target, result_queue))
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError(
                f"Isolated TPU test timed out after {timeout} seconds")
        try:
            status, payload = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError(
                "Isolated TPU test exited without reporting a result "
                f"(exitcode={process.exitcode})") from exc
    finally:
        result_queue.close()

    if status == "SKIP":
        pytest.skip(payload)
    if status == "ERROR":
        raise AssertionError(f"Isolated TPU test failed:\n{payload}")
    if process.exitcode != 0:
        raise AssertionError(
            f"Isolated TPU test crashed (exitcode={process.exitcode})")
