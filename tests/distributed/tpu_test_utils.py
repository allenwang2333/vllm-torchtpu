# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that must not retain TPU runtime state in pytest."""

import functools
import multiprocessing
import os
import queue
import traceback
from collections.abc import Callable
from typing import Any

import pytest


def _isolated_test_worker(target: Callable[..., Any], args: tuple[Any, ...],
                          kwargs: dict[str, Any], result_queue: Any) -> None:
    os.environ["VLLM_ISOLATED_PROCESS"] = "1"
    try:
        target(*args, **kwargs)
    except pytest.skip.Exception as exc:
        result_queue.put(("SKIP", str(exc)))
    except BaseException:  # noqa: BLE001
        result_queue.put(("ERROR", traceback.format_exc()))
    else:
        result_queue.put(("OK", None))


def _run_isolated(target: Callable[..., Any],
                  args: tuple[Any, ...],
                  kwargs: dict[str, Any],
                  *,
                  timeout: float = 600) -> None:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_isolated_test_worker,
                          args=(target, args, kwargs, result_queue))
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


def run_in_isolated_process(target: Callable[..., Any] | None = None,
                            *,
                            timeout: float = 600) -> Any:
    """Run a TPU test in a spawned process and propagate its result.

    Can be invoked directly:
        run_in_isolated_process(func, timeout=60)

    Or used as a decorator on a test function/method:
        @run_in_isolated_process(timeout=60)
        def test_something(self): ...
    """
    if target is None:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> None:
                if os.environ.get("VLLM_ISOLATED_PROCESS") == "1":
                    func(*args, **kwargs)
                    return
                _run_isolated(func, args, kwargs, timeout=timeout)

            return wrapper

        return decorator

    if os.environ.get("VLLM_ISOLATED_PROCESS") == "1":
        target()
        return

    _run_isolated(target, (), {}, timeout=timeout)
