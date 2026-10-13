"""Bind device implementations into reused upstream Python code."""

import sys
from threading import RLock
from types import FunctionType, ModuleType

_lock = RLock()
_acquire_lock = _lock.acquire
_release_lock = _lock.release
_SYS_MODULES = sys.modules
_NAMESPACE_PREFIX = f"{__name__}."


class device_hooks:
    __slots__ = ("_module", "_previous", "_replacements")

    def __init__(self, module, **replacements):
        self._module = module
        self._replacements = replacements
        self._previous = None

    def __enter__(self):
        _acquire_lock()
        replacements = self._replacements
        if replacements:
            module = self._module
            self._previous = [(name, getattr(module, name)) for name in replacements]
            module.__dict__.update(replacements)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        previous = self._previous
        if previous is not None:
            mod_dict = self._module.__dict__
            for name, value in previous:
                mod_dict[name] = value
        _release_lock()
        return False


def bind_device_ops(function, **operations):
    """Reuse upstream bytecode with per-adapter device function bindings.

    Unlike runtime module monkeypatches, bindings are immutable during forward
    and TorchDynamo can trace them without locks or global mutation.
    """
    # Dynamo guards resolve global functions by module name. A copied globals
    # dict with the upstream name would make guards inspect the GPU operation.
    qualname = function.__qualname__
    fn_name = function.__name__
    name = f"{_NAMESPACE_PREFIX}{qualname.replace('.', '_')}"
    namespace = ModuleType(name)
    ns_dict = namespace.__dict__
    ns_dict.update(function.__globals__)
    ns_dict.update(operations)
    ns_dict["__name__"] = name
    _SYS_MODULES[name] = namespace
    bound = FunctionType(
        function.__code__,
        ns_dict,
        fn_name,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = qualname
    ns_dict[fn_name] = bound
    return bound
