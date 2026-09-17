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
    try:
        target()
    except pytest.skip.Exception as exc:
        result_queue.put(("SKIP", str(exc)))
    except Exception:  # noqa: BLE001
        result_queue.put(("ERROR", traceback.format_exc()))
    else:
        result_queue.put(("OK", None))


class _TargetRunner:
    """Pickleable target invoker for functions and methods across process spawn."""

    def __init__(
        self,
        mod_name: str,
        qualname: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.mod_name = mod_name
        self.qualname = qualname
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> None:
        import importlib

        mod = importlib.import_module(self.mod_name)
        obj: Any = mod
        for part in self.qualname.split("."):
            obj = getattr(obj, part)
        while hasattr(obj, "__wrapped__"):
            obj = obj.__wrapped__
        obj(*self.args, **self.kwargs)


def run_in_isolated_process(target: Callable[..., Any] | None = None,
                            *,
                            timeout: float = 600) -> Any:
    """Run a TPU test in a spawned process and propagate its result.

    Can be invoked directly:
        run_in_isolated_process(_test_func, timeout=60)

    Or used as a decorator on a test function/method:
        @run_in_isolated_process(timeout=60)
        def test_something(self): ...
    """
    if target is None:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            import functools

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> None:
                if not args and not kwargs:
                    run_in_isolated_process(func, timeout=timeout)
                else:
                    runner = _TargetRunner(func.__module__, func.__qualname__,
                                           args, kwargs)
                    run_in_isolated_process(runner, timeout=timeout)

            return wrapper

        return decorator

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
