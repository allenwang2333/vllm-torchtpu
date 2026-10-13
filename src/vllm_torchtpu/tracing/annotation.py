# Copyright 2026 Google LLC
"""PyTorch Kineto and TPU profiler trace annotation utilities."""

import contextlib
from typing import Any

import torch

_AUTOGRAD = torch.autograd
_PROFILER = torch.profiler


def is_trace_annotation_enabled() -> bool:
    """Checks whether PyTorch Kineto profiler session is active."""
    return _AUTOGRAD._profiler_enabled()


class TraceAnnotation(contextlib.ContextDecorator):
    __slots__ = ("name", "kwargs", "_rf")

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self._rf = None

    is_enabled = staticmethod(is_trace_annotation_enabled)

    def __enter__(self):
        if _AUTOGRAD._profiler_enabled():
            kwargs = self.kwargs
            self._rf = rf = _PROFILER.record_function(
                f"{self.name}#{','.join([f'{k}={v}' for k, v in kwargs.items()])}#"
                if kwargs
                else self.name
            )
            rf.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._rf is not None:
            self._rf.__exit__(exc_type, exc_val, exc_tb)
            self._rf = None
