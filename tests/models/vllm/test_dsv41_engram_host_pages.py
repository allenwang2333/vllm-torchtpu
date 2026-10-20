# SPDX-License-Identifier: Apache-2.0
"""Huge-page Engram staging (TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common import host_pages
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import Engram


def _thp_or_skip() -> None:
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            mode = f.read()
    except FileNotFoundError:
        pytest.skip("no transparent huge pages")
    if "[never]" in mode:
        pytest.skip("transparent huge pages disabled")


@pytest.mark.parametrize("huge", [True, False])
def test_alloc_staging_is_aligned_zeroed_and_paged_as_asked(huge):
    _thp_or_skip()
    staging = host_pages.alloc_staging(3_000_001, huge=huge)
    t = staging.tensor
    assert t.dtype == torch.uint8 and t.numel() == 3_000_001
    assert t.data_ptr() % (2 << 20) == 0
    assert int(t.sum()) == 0
    assert staging.mapped_kb == 4096
    assert staging.huge_kb == (4096 if huge else 0)
    t.fill_(7)
    assert int(t[-1]) == 7


def test_hugepage_staging_falls_back_to_torch_zeros_with_a_warning(monkeypatch):
    monkeypatch.setattr(host_pages, "_vma_anon_huge_kb", lambda addr: 0)
    warnings = []
    monkeypatch.setattr(
        common_engram.logger, "warning_once", lambda *a: warnings.append(a)
    )
    t = common_engram._alloc_packed_staging(1 << 20, huge=True, small=False)
    assert t.numel() == 1 << 20 and int(t.sum()) == 0
    assert len(warnings) == 1
    assert "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING" in warnings[0][0]
    assert warnings[0][1:3] == (0, 2048)


def test_smallpage_staging_refuses_huge_pages(monkeypatch):
    monkeypatch.setattr(host_pages, "_vma_anon_huge_kb", lambda addr: 2048)
    with pytest.raises(RuntimeError, match="small-page staging"):
        common_engram._alloc_packed_staging(1 << 20, huge=False, small=True)


def test_h2d_step_log_flushes_rows(tmp_path):
    log = host_pages.H2DStepLog(str(tmp_path), dp_rank=3)
    assert log.cpu() >= 0
    log.add(10, 2_000_010, 5, 6, num_tokens=288)
    log.flush()
    (path,) = tmp_path.glob("h2d-*.jsonl")
    row = json.loads(path.read_text())
    assert row["ms"] == 2.0 and row["cpu"] == [5, 6] and row["dp_rank"] == 3
    assert row["num_tokens"] == 288


def test_tpu_numa_node_requires_an_open_vfio_group(monkeypatch):
    monkeypatch.setattr(host_pages, "tpu_vfio_groups", lambda pid="self": [])
    with pytest.raises(RuntimeError, match="cannot derive one TPU NUMA node"):
        host_pages.tpu_numa_node()


def test_summarize_pages_runs():
    t = torch.ones(5 * 4096, dtype=torch.uint8)
    try:
        pages = host_pages.summarize_pages(t.data_ptr(), t.numel())
    except PermissionError:
        pytest.skip("move_pages not permitted here")
    assert sum(run[2] for run in pages["runs"]) == pages["pages"] >= 5


def _offload_engram() -> Engram:
    engram = object.__new__(Engram)
    nn.Module.__init__(engram)
    engram.embed_tokens = SimpleNamespace(
        host_offload=True,
        cpu_offload=False,
        part_n_hash_cols=4,
        block_size=8,
        dp_size=1,
    )
    return engram


def test_hugepage_staging_matches_default_buffers(monkeypatch):
    """The switches change only the packed buffer's pages, not any contents."""
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING", False)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_SMALLPAGE_STAGING", False)
    ref = _offload_engram()
    Engram._init_staging(ref, max_tokens=10, head_dim=16)

    calls = []

    def fake_alloc(nbytes, *, huge):
        calls.append((nbytes, huge))
        kb = -(-nbytes // (2 << 20)) * 2048
        return host_pages.Staging(
            torch.zeros(nbytes, dtype=torch.uint8), kb if huge else 0, kb
        )

    monkeypatch.setattr(host_pages, "alloc_staging", fake_alloc)
    for huge in (True, False):
        monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING", huge)
        monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_SMALLPAGE_STAGING", not huge)
        new = _offload_engram()
        Engram._init_staging(new, max_tokens=10, head_dim=16)
        assert calls[-1] == (ref._host_packed_buf.numel(), huge)
        for name in ("_host_packed_buf", "_host_scale_buf", "_staged_raw"):
            assert torch.equal(getattr(ref, name), getattr(new, name)), name


def test_huge_and_small_pages_are_exclusive(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING", True)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_SMALLPAGE_STAGING", True)
    with pytest.raises(ValueError, match="both"):
        Engram._init_staging(_offload_engram(), max_tokens=10, head_dim=16)


def test_staging_log_records_each_buffer(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HUGEPAGE_STAGING", False)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_SMALLPAGE_STAGING", False)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_STAGING_LOG", str(tmp_path))
    monkeypatch.setattr(host_pages, "tpu_vfio_groups", lambda pid="self": [])
    monkeypatch.setattr(
        host_pages,
        "summarize_pages",
        lambda addr, nbytes, pid=0: {"pages": -(-nbytes // 4096)},
    )
    engram = _offload_engram()
    Engram._init_staging(engram, max_tokens=10, head_dim=16)
    (log,) = tmp_path.glob("staging-*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["buffer"] for r in records] == ["packed", "scale"]
    assert records[0]["addr"] == engram._host_packed_buf.data_ptr()
    assert records[0]["nbytes"] == engram._host_packed_buf.numel()
    assert engram._h2d_log is not None
