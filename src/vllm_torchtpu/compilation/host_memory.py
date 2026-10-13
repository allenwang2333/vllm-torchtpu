# SPDX-License-Identifier: Apache-2.0
"""Return idle glibc heap pages after a compiler call releases its locals."""

import ctypes
import functools
import gc
import inspect
import sys
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_RECLAIM_ERROR_MSG = (
    "TPU_RECLAIM_COMPILE_MEMORY requires Linux with glibc malloc; "
    "disable it when using another host allocator."
)
_C_CHAR_P = ctypes.c_char_p
_C_VOID_P = ctypes.c_void_p
_C_SIZE_T = ctypes.c_size_t
_C_INT = ctypes.c_int
_cast = ctypes.cast
_byref = ctypes.byref


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", _C_CHAR_P),
        ("dli_fbase", _C_VOID_P),
        ("dli_sname", _C_CHAR_P),
        ("dli_saddr", _C_VOID_P),
    ]


_DL_INFO_PTR = ctypes.POINTER(_DlInfo)
_TRIM_ARGTYPES = (_C_SIZE_T,)
_DLADDR_ARGTYPES = (_C_VOID_P, _DL_INFO_PTR)


@functools.cache
def _glibc_allocator() -> ctypes.CDLL:
    if sys.platform != "linux":
        raise RuntimeError(_RECLAIM_ERROR_MSG)
    libc = ctypes.CDLL(None)
    try:
        malloc = libc.malloc
        trim = libc.malloc_trim
        version = libc.gnu_get_libc_version
        dladdr = libc.dladdr
    except AttributeError as error:
        raise RuntimeError(_RECLAIM_ERROR_MSG) from error
    version.restype = _C_CHAR_P
    trim.argtypes = _TRIM_ARGTYPES
    trim.restype = _C_INT
    dladdr.argtypes = _DLADDR_ARGTYPES
    dladdr.restype = _C_INT
    info = _DlInfo()
    info_ref = _byref(info)
    if (
        not dladdr(_cast(malloc, _C_VOID_P), info_ref)
        or not (base := info.dli_fbase)
        or not dladdr(_cast(trim, _C_VOID_P), info_ref)
        or info.dli_fbase != base
        or not dladdr(_cast(version, _C_VOID_P), info_ref)
        or info.dli_fbase != base
        or not version()
    ):
        raise RuntimeError(_RECLAIM_ERROR_MSG)
    return libc


def reclaim_after_compile(
    enabled: Callable[[], bool],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Collect dead compiler cycles, then trim only after its frame is gone."""
    this_mod = sys.modules[__name__]

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        params = inspect.signature(function).parameters.values()
        decl: list[str] = []
        call: list[str] = []
        env: dict[str, Any] = {
            "_fn": function,
            "_en": enabled,
            "_hm": this_mod,
            "_gc": gc,
        }
        for idx, param in enumerate(params):
            if param.default is not inspect.Parameter.empty:
                dname = f"_d{idx}"
                env[dname] = param.default
                decl.append(f"{param.name}={dname}")
            else:
                decl.append(param.name)
            call.append(param.name)
        decl_s = ", ".join(decl)
        call_s = ", ".join(call)
        exec(
            f"def wrapped({decl_s}):\n"
            f"    if not _en():\n"
            f"        return _fn({call_s})\n"
            f"    _al = _hm._glibc_allocator()\n"
            f"    _res = _fn({call_s})\n"
            f"    _gc.collect()\n"
            f"    _al.malloc_trim(0)\n"
            f"    return _res\n",
            env,
        )
        return functools.wraps(function)(env["wrapped"])

    return decorate
