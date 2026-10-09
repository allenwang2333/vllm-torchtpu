# SPDX-License-Identifier: Apache-2.0
"""Tests for host memory reclamation during compilation (PR #1793)."""

import ctypes
import gc
import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import host_memory


def test_glibc_allocator_success_glibc():
    """Verify glibc allocator detection succeeds when glibc malloc is active."""
    host_memory._glibc_allocator.cache_clear()
    try:
        with patch("ctypes.CDLL", return_value=ctypes.CDLL("libc.so.6")):
            libc = host_memory._glibc_allocator()
            assert libc is not None
            version = libc.gnu_get_libc_version()
            assert version is not None
            assert len(version) > 0
    finally:
        host_memory._glibc_allocator.cache_clear()


def test_glibc_allocator_rejects_non_glibc_malloc_or_tcmalloc():
    """Verify non-glibc malloc (e.g. TCMalloc or Python malloc) is rejected."""
    host_memory._glibc_allocator.cache_clear()
    try:
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()
    finally:
        host_memory._glibc_allocator.cache_clear()


def test_glibc_allocator_rejects_non_linux(monkeypatch):
    """Verify non-Linux platforms raise RuntimeError before compilation."""
    monkeypatch.setattr(sys, "platform", "darwin")
    host_memory._glibc_allocator.cache_clear()
    try:
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()
    finally:
        host_memory._glibc_allocator.cache_clear()


def test_glibc_allocator_missing_symbol_raises(monkeypatch):
    """Verify missing malloc/trim symbols raise RuntimeError."""
    mock_libc = MagicMock(spec=[])
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: mock_libc)
    host_memory._glibc_allocator.cache_clear()
    try:
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()
    finally:
        host_memory._glibc_allocator.cache_clear()


@pytest.mark.parametrize(
    "dladdr_ret, fbase1, fbase2, fbase3, ver_none",
    [
        ((0, 1, 1), 0x1000, 0x1000, 0x1000, False),  # dladdr malloc fails
        ((1, 0, 1), 0x1000, 0x1000, 0x1000, False),  # dladdr trim fails
        ((1, 1, 0), 0x1000, 0x1000, 0x1000, False),  # dladdr version fails
        ((1, 1, 1), 0, 0x1000, 0x1000, False),  # dli_fbase is 0 / None
        ((1, 1, 1), 0x1000, 0x2000, 0x1000, False),  # trim fbase mismatch
        ((1, 1, 1), 0x1000, 0x1000, 0x3000, False),  # version fbase mismatch
        ((1, 1, 1), 0x1000, 0x1000, 0x1000, True),  # version returns None
    ],
)
def test_glibc_allocator_dladdr_and_version_checks(
    monkeypatch, dladdr_ret, fbase1, fbase2, fbase3, ver_none
):
    """Verify dladdr and symbol resolution checks reject incompatible allocators."""
    bases = [fbase1, fbase2, fbase3]
    call_idx = 0

    real_libc = ctypes.CDLL("libc.so.6")

    def fake_dladdr(addr, info_ptr):
        nonlocal call_idx
        idx = min(call_idx, len(dladdr_ret) - 1)
        ret = dladdr_ret[idx]
        if ret:
            info_ptr.contents.dli_fbase = bases[idx]
        call_idx += 1
        return ret

    c_dladdr = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(host_memory._DlInfo)
    )(fake_dladdr)

    c_ver = (
        ctypes.CFUNCTYPE(ctypes.c_char_p)(lambda: None)
        if ver_none
        else real_libc.gnu_get_libc_version
    )

    class FakeLibc:
        malloc = real_libc.malloc
        malloc_trim = real_libc.malloc_trim
        gnu_get_libc_version = c_ver
        dladdr = c_dladdr

    fake_libc = FakeLibc()
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: fake_libc)

    host_memory._glibc_allocator.cache_clear()
    try:
        with pytest.raises(
            RuntimeError, match="TPU_RECLAIM_COMPILE_MEMORY requires Linux"
        ):
            host_memory._glibc_allocator()
    finally:
        host_memory._glibc_allocator.cache_clear()


def test_reclaim_after_compile_enabled(monkeypatch):
    """Verify gc.collect and malloc_trim are invoked when enabled is True."""
    mock_libc = MagicMock()
    mock_libc.malloc_trim.return_value = 1
    monkeypatch.setattr(host_memory, "_glibc_allocator", lambda: mock_libc)
    mock_gc_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_gc_collect)

    @host_memory.reclaim_after_compile(lambda: True)
    def dummy_compile(x, y):
        return x + y

    res = dummy_compile(20, 22)
    assert res == 42
    assert mock_gc_collect.call_count == 1
    mock_libc.malloc_trim.assert_called_once_with(0)


def test_reclaim_after_compile_disabled(monkeypatch):
    """Verify allocator is not called when enabled is False."""
    mock_allocator_func = MagicMock()
    monkeypatch.setattr(host_memory, "_glibc_allocator", mock_allocator_func)
    mock_gc_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_gc_collect)

    @host_memory.reclaim_after_compile(lambda: False)
    def dummy_compile(x, y):
        return x * y

    res = dummy_compile(6, 7)
    assert res == 42
    assert mock_allocator_func.call_count == 0
    assert mock_gc_collect.call_count == 0


def test_tpu_reclaim_compile_memory_env(monkeypatch):
    """Verify TPU_RECLAIM_COMPILE_MEMORY defaults to False and parses correctly."""
    name = "TPU_RECLAIM_COMPILE_MEMORY"
    monkeypatch.delenv(name, raising=False)
    assert envs.environment_variables[name]() is False

    monkeypatch.setenv(name, "1")
    assert envs.environment_variables[name]() is True

    monkeypatch.setenv(name, "true")
    assert envs.environment_variables[name]() is True

    monkeypatch.setenv(name, "0")
    assert envs.environment_variables[name]() is False
