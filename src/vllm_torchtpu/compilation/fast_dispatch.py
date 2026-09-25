"""Low-overhead dispatch into small ``torch.compile``'d host helpers.

Calling a ``torch.compile``'d function costs a full trip through dynamo's
``compile_wrapper``: config patching, eval-frame callback install, and a guard
walk over every cache entry of the frame. For the step-critical helpers (a few
tens of microseconds of device work each, called a dozen times per decode
step) that trip is most of their host cost.

``tpu_compile`` returns a wrapper that, once a call has gone through dynamo
normally, remembers which cache entry served it (keyed by the argument
signature) and serves later calls with the same signature by checking just that
entry's guards and running its transformed bytecode directly — the same code
dynamo would run, minus the wrapper. A guard miss or an unseen signature falls
back to the regular compiled call, so dynamo still compiles, recompiles and
guards exactly as before. Keyword arguments and calls made while dynamo is
tracing always take the regular path.

The fast path stays off until ``activate()`` (called once warmup is done), so
cache entries dropped by the profiling run's ``torch._dynamo.reset()`` are
never remembered. Set ``VLLM_TPU_FAST_COMPILED_DISPATCH=0`` to disable.
"""

import functools
import os
import types
import weakref
from typing import Any, Callable

import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

_ENABLED = os.environ.get("VLLM_TPU_FAST_COMPILED_DISPATCH", "1") != "0"
_active = False
_instances: "weakref.WeakSet[_FastCompiled]" = weakref.WeakSet()


def activate() -> None:
    """Turn the fast path on (idempotent). Call after the last
    ``torch._dynamo.reset()`` of startup."""
    global _active
    for inst in list(_instances):
        inst._entries.clear()
    _active = _ENABLED


def _signature(args: tuple) -> tuple:
    # Picks the candidate cache entry; the entry's guards still run on every
    # call, so this only needs to separate the common variants cheaply.
    key = []
    for a in args:
        if isinstance(a, torch.Tensor):
            key.append((a.shape, a.dtype, a.device))
        elif a is None or isinstance(a, (bool, int, float, str)):
            key.append((type(a), a))
        else:
            key.append(id(a))
    return tuple(key)


class _FastCompiled:
    __slots__ = ("_compiled", "_orig", "_argnames", "_entries", "__wrapped__",
                 "__name__", "__qualname__", "__doc__", "__weakref__")

    def __init__(self, compiled: Callable, orig: Callable):
        self._compiled = compiled
        self._orig = orig
        code = orig.__code__
        self._argnames = code.co_varnames[:code.co_argcount]
        # signature -> (direct function, dynamo cache entry)
        self._entries: dict[tuple, tuple[Callable, Any]] = {}
        self.__wrapped__ = orig
        self.__name__ = orig.__name__
        self.__qualname__ = orig.__qualname__
        self.__doc__ = orig.__doc__
        _instances.add(self)

    def __get__(self, obj, objtype=None):
        # Used as a method: bind like a plain function.
        if obj is None:
            return self
        if torch.compiler.is_compiling():
            # Called from inside another compiled function: trace the body
            # into that graph, as dynamo does for a nested torch.compile.
            return functools.partial(self._orig, obj)
        return types.MethodType(self, obj)

    def __call__(self, *args, **kwargs):
        if not _active or kwargs or torch.compiler.is_compiling():
            return self._compiled(*args, **kwargs)
        key = _signature(args)
        hit = self._entries.get(key)
        if hit is not None:
            # Read the manager off the entry each time: dynamo swaps in an
            # always-failing one when a guarded object is deallocated.
            if hit[1].guard_manager.check(dict(zip(self._argnames, args))):
                return hit[0](*args)
        out = self._compiled(*args)
        self._remember(key, args)
        return out

    def _remember(self, key: tuple, args: tuple) -> None:
        if len(args) != len(self._argnames):
            return
        f_locals = dict(zip(self._argnames, args))
        orig = self._orig
        for entry in _debug_get_cache_entry_list(orig.__code__):
            if entry.guard_manager.check(f_locals):
                direct = types.FunctionType(entry.code, orig.__globals__,
                                            orig.__name__, orig.__defaults__,
                                            orig.__closure__)
                self._entries[key] = (direct, entry)
                return
        self._entries.pop(key, None)

    @property
    def orig(self) -> Callable:
        """The undecorated function (for inlining into another compiled
        function's trace)."""
        return self._orig

    @property
    def _torchdynamo_orig_callable(self) -> Callable:
        # Same attribute torch.compile's wrapper carries; callers use it to
        # reach the undecorated body.
        return self._orig


def tpu_compile(fn: Callable) -> Callable:
    """``torch.compile(backend="tpu", fullgraph=True, dynamic=False)`` with
    the low-overhead dispatch above."""
    compiled = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    if not _ENABLED:
        return compiled
    return _FastCompiled(compiled, fn)


def disable_aot_guard_checks(*roots: Any) -> int:
    """Skip the per-call guard check of vLLM's AOT-compiled model forwards.

    vLLM already drops it for artifacts loaded from disk unless
    ``dynamic_shapes_config.evaluate_guards`` is set; a forward compiled in
    this process keeps it, and it costs a signature bind plus a guard walk on
    every draft and target forward. The check only raises on failure — it
    selects nothing — so the caller applies the same ``evaluate_guards``
    condition. Returns the number of forwards changed.
    """
    if not _ENABLED:
        return 0
    n = 0
    for root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for module in root.modules():
            fn = getattr(module, "aot_compiled_fn", None)
            if fn is not None and hasattr(fn, "disable_guard_check"):
                fn.disable_guard_check()
                n += 1
    return n
