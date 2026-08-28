# Copyright 2026 Google LLC
"""PyTorch Kineto and TPU profiler trace annotation utilities."""
import contextlib
from typing import Any

import torch


def is_trace_annotation_enabled() -> bool:
    """Checks whether PyTorch Kineto profiler session is active."""
    return torch.autograd._profiler_enabled()


class TraceAnnotation(contextlib.ContextDecorator):
    """PyTorch record_function wrapper supporting key-value metadata formatting."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        if kwargs:
            args_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            name = f"{name}#{args_str}#"
        self._rf = torch.profiler.record_function(name)

    def __enter__(self):
        return self._rf.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return self._rf.__exit__(exc_type, exc_val, exc_tb)
