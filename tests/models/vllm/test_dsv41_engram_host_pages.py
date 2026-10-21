# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the packed Engram H2D source allocation."""

import ctypes
import errno
import importlib.util
import mmap
import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

# Load this CPU-only allocator without registering the vLLM model package.
_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/host_pages.py"
)
_SPEC = importlib.util.spec_from_file_location("engram_host_pages", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
host_pages = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(host_pages)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def allocation(monkeypatch, tmp_path):
    size_file = tmp_path / "hpage_pmd_size"
    size_file.write_text(str(2 << 20))
    monkeypatch.setattr(host_pages, "_THP_SIZE", size_file)
    monkeypatch.setattr(host_pages, "logger", Mock(spec=["warning_once"]))
    return host_pages


@pytest.mark.parametrize("nbytes", [1, 1_824_768, 3_000_001])
def test_aligned_zeroed_buffer_keeps_mapping_alive_through_views(
    allocation, monkeypatch, nbytes
):
    # THP availability is host policy; check alignment/lifetime independently.
    monkeypatch.setattr(allocation, "_hugepage_backed", lambda *args: True)
    original = mmap.mmap
    owners = []

    def track_mapping(*args, **kwargs):
        mapping = original(*args, **kwargs)
        owners.append(weakref.ref(mapping))
        return mapping

    monkeypatch.setattr(mmap, "mmap", track_mapping)
    tensor = allocation.alloc_packed_staging(nbytes)
    assert tensor.dtype == torch.uint8 and tensor.device.type == "cpu"
    assert tensor.shape == (nbytes,) and tensor.is_contiguous()
    assert tensor.data_ptr() % (2 << 20) == 0
    assert torch.count_nonzero(tensor).item() == 0
    view = tensor[-1:]
    del tensor
    assert owners[0]() is not None
    view.fill_(17)
    assert view.item() == 17
    del view
    assert owners[0]() is None
    allocation.logger.warning_once.assert_not_called()


@pytest.mark.parametrize("failure", ["partial", "madvise", "sysfs", "smaps"])
def test_unavailable_huge_pages_release_mapping_and_fall_back(
    allocation, monkeypatch, tmp_path, failure
):
    mappings = []

    class Mapping(mmap.mmap):
        def madvise(self, *args):
            if failure == "madvise":
                raise OSError(errno.EINVAL, "THP advice unavailable")
            return super().madvise(*args)

    def track_mapping(*args, **kwargs):
        mapping = Mapping(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    monkeypatch.setattr(mmap, "mmap", track_mapping)
    if failure == "sysfs":
        monkeypatch.setattr(allocation, "_THP_SIZE", tmp_path / "missing")
    elif failure == "smaps":
        monkeypatch.setattr(allocation, "_SMAPS", tmp_path / "missing")
    else:
        monkeypatch.setattr(allocation, "_hugepage_backed", lambda *args: False)
    tensor = allocation.alloc_packed_staging(1024)
    assert torch.equal(tensor, torch.zeros(1024, dtype=torch.uint8))
    assert all(mapping.closed for mapping in mappings)
    allocation.logger.warning_once.assert_called_once()


@pytest.mark.parametrize("nbytes,huge", [(0, True), (0, False), (1024, False)])
def test_control_and_empty_buffers_do_not_request_huge_pages(
    allocation, monkeypatch, nbytes, huge
):
    request = Mock(side_effect=AssertionError("unexpected huge-page allocation"))
    monkeypatch.setattr(mmap, "mmap", request)
    tensor = allocation.alloc_packed_staging(nbytes, huge=huge)
    assert torch.equal(tensor, torch.zeros(nbytes, dtype=torch.uint8))
    request.assert_not_called()
    allocation.logger.warning_once.assert_not_called()


@pytest.mark.parametrize("huge", [False, True])
def test_negative_buffer_size_is_rejected(huge):
    with pytest.raises(ValueError, match="nonnegative"):
        host_pages.alloc_packed_staging(-1, huge=huge)


@pytest.mark.parametrize(
    "regions,address,size,expected",
    [
        ([(0x1000, 0x5000, 16)], 0x2000, 0x2000, True),
        ([(0x1000, 0x5000, 8)], 0x2000, 0x2000, False),
        ([(0x1000, 0x3000, 8), (0x3000, 0x5000, 8)], 0x2000, 0x2000, True),
        ([(0x1000, 0x3000, 8)], 0x2000, 0x2000, False),
    ],
)
def test_backing_check_covers_the_entire_range(
    monkeypatch, tmp_path, regions, address, size, expected
):
    smaps = tmp_path / "smaps"
    smaps.write_text(
        "".join(
            f"{start:x}-{end:x} rw-p 00000000 00:00 0\nAnonHugePages: {kb} kB\n"
            for start, end, kb in regions
        )
    )
    monkeypatch.setattr(host_pages, "_SMAPS", smaps)
    assert host_pages._hugepage_backed(address, size) is expected


def test_native_small_page_mapping_is_not_reported_as_huge():
    with mmap.mmap(-1, 2 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as buf:
        buf.madvise(mmap.MADV_NOHUGEPAGE)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        ctypes.memset(address, 0, len(buf))
        assert not host_pages._hugepage_backed(address, len(buf))


def test_native_allocation_returns_writable_zeroed_bytes(allocation):
    tensor = allocation.alloc_packed_staging(3_000_001)
    assert torch.count_nonzero(tensor).item() == 0
    tensor.fill_(29)
    assert tensor[0].item() == tensor[-1].item() == 29
