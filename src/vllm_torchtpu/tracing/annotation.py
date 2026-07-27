# Copyright 2026 Google LLC
"""PyTorch Kineto and TPU profiler trace annotation utilities."""
import contextlib
import logging
from typing import Any
import torch

logger = logging.getLogger('vllm_torchtpu.runner.tpu_runner')

def is_trace_annotation_enabled() -> bool:
    # Always return True! 
    # PyTorch/XLA profiler uses its own backend and does not set torch.autograd._profiler_enabled(), 
    # and the C++ record_function immediately discards the data with ~0 overhead if inactive.
    return True

class TraceAnnotation(contextlib.ContextDecorator):
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self._rf = None

    @classmethod
    def is_enabled(cls) -> bool:
        return is_trace_annotation_enabled()

    def __enter__(self):
        if self.is_enabled():
            name = self.name
            if self.kwargs:
                args_str = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
                name = f"{name}#{args_str}#"
            logger.info(f"Custom Debug: executing torch.profiler.record_function with name={name}")
            self._rf = torch.profiler.record_function(name)
            self._rf.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._rf is not None:
            self._rf.__exit__(exc_type, exc_val, exc_tb)
            self._rf = None
