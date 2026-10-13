# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for vllm_torchtpu.compilation.host_memory."""

from __future__ import annotations

import ctypes
import gc
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.compilation", ("vllm_torchtpu", "compilation")),
):
    if _pkg_name not in sys.modules:
        _mod = types.ModuleType(_pkg_name)
        _mod.__path__ = [str(Path(_SRC_DIR, *_rel_parts))]
        sys.modules[_pkg_name] = _mod

from vllm_torchtpu.compilation import host_memory  # noqa: E402


def test_glibc_allocator_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise all branches of _glibc_allocator and reclaim_after_compile."""
    # 1. Non-Linux platform rejection branch
    host_memory._glibc_allocator.cache_clear()
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", "darwin")
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()

    # 2. Missing symbol AttributeError branch
    host_memory._glibc_allocator.cache_clear()
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", "linux")
        m.setattr(ctypes, "CDLL", lambda *_a, **_kw: MagicMock(spec=[]))
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()

    # 3. dladdr / dli_fbase / gnu_get_libc_version failure & success branches
    real_libc = ctypes.CDLL("libc.so.6")
    cases = [
        ((0, 1, 1), (0x1000, 0x1000, 0x1000), b"2.39", False),
        ((1, 1, 1), (0, 0x1000, 0x1000), b"2.39", False),
        ((1, 0, 1), (0x1000, 0x1000, 0x1000), b"2.39", False),
        ((1, 1, 1), (0x1000, 0x2000, 0x1000), b"2.39", False),
        ((1, 1, 0), (0x1000, 0x1000, 0x1000), b"2.39", False),
        ((1, 1, 1), (0x1000, 0x1000, 0x3000), b"2.39", False),
        ((1, 1, 1), (0x1000, 0x1000, 0x1000), None, False),
        ((1, 1, 1), (0x1000, 0x1000, 0x1000), b"2.39", True),
    ]
    for dladdr_rets, fbases, ver_bytes, should_succeed in cases:
        call_idx = 0

        def fake_dladdr(
            _addr: int,
            info_ptr: ctypes._Pointer,  # type: ignore[type-arg]
            _rets: tuple[int, int, int] = dladdr_rets,
            _bases: tuple[int, int, int] = fbases,
        ) -> int:
            nonlocal call_idx
            idx = min(call_idx, 2)
            call_idx += 1
            ret = _rets[idx]
            if ret:
                info_ptr.contents.dli_fbase = _bases[idx]
            return ret

        c_dladdr = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(host_memory._DlInfo)
        )(fake_dladdr)
        c_ver = (
            real_libc.gnu_get_libc_version
            if ver_bytes is not None
            else ctypes.CFUNCTYPE(ctypes.c_char_p)(lambda: None)
        )

        class FakeLibc:
            malloc = real_libc.malloc
            malloc_trim = real_libc.malloc_trim
            gnu_get_libc_version = c_ver
            dladdr = c_dladdr

        fake_instance = FakeLibc()
        host_memory._glibc_allocator.cache_clear()
        with monkeypatch.context() as m:
            m.setattr(sys, "platform", "linux")
            m.setattr(
                ctypes,
                "CDLL",
                lambda *_a, _inst=fake_instance, **_kw: _inst,
            )
            if should_succeed:
                resolved = host_memory._glibc_allocator()
                assert resolved is fake_instance
            else:
                with pytest.raises(
                    RuntimeError,
                    match="TPU_RECLAIM_COMPILE_MEMORY requires Linux",
                ):
                    host_memory._glibc_allocator()

    host_memory._glibc_allocator.cache_clear()

    # 4. reclaim_after_compile disabled & enabled branches
    mock_libc = MagicMock()
    mock_libc.malloc_trim.return_value = 1
    mock_alloc = MagicMock(return_value=mock_libc)
    mock_collect = MagicMock()

    with monkeypatch.context() as m:
        m.setattr(host_memory, "_glibc_allocator", mock_alloc)
        m.setattr(gc, "collect", mock_collect)

        active = False

        @host_memory.reclaim_after_compile(lambda: active)
        def compile_step(x: int, scale: int = 2) -> int:
            return x * scale

        assert compile_step(21, scale=2) == 42
        assert mock_alloc.call_count == 0
        assert mock_collect.call_count == 0

        active = True
        assert compile_step(21, scale=3) == 63
        assert mock_alloc.call_count == 1
        assert mock_collect.call_count == 1
        mock_libc.malloc_trim.assert_called_once_with(0)
