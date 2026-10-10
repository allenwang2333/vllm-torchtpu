# SPDX-License-Identifier: Apache-2.0
"""TD8.234: no stray background ``[ep-prefetch]`` after weights are loaded.

Job 1307: loading the DSpark draft (``runner_type == "draft"``, every expert
local under ``_force_draft_tp1``) started a range prefetch over all 48 target
checkpoint files. The draft loaded in 0.65 s and the runner evicted the page
cache, but the prefetch kept reading 233.2 GiB/rank for 83-182 s during
``_precompile_backbone``, refilling the page cache. Now the prefetch is
skipped for draft models and single-file checkpoints, and stops when the
weights iterator ends or closes, or when ``evict_loaded_weights_page_cache``
runs.
"""

from __future__ import annotations

import os
import shutil
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import model_loader_patches as mlp

_E = "model.layers.0.mlp.experts"


def _save_shards(tmp_path, count=2):
    from safetensors.torch import save_file

    first = str(tmp_path / "model-00001.safetensors")
    save_file(
        {
            f"{_E}.0.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
            f"{_E}.7.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
            "model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
        },
        first,
    )
    shards = [first]
    for i in range(2, count + 1):
        shards.append(str(tmp_path / f"model-{i:05d}.safetensors"))
        shutil.copy(first, shards[-1])
    return shards


@pytest.fixture
def active_events(monkeypatch):
    events: list[threading.Event] = []
    monkeypatch.setattr(mlp, "_active_prefetch_stop_events", events)
    return events


@pytest.fixture
def patched_iterator(monkeypatch):
    """Install the patch over upstream's iterator; record prefetch starts."""
    from vllm.model_executor.model_loader import (
        default_loader,
        ep_weight_filter,
        weight_utils,
    )

    for module, name in (
        (weight_utils, "should_skip_weight"),
        (ep_weight_filter, "should_skip_weight"),
        (weight_utils, "safetensors_weights_iterator"),
        (default_loader, "safetensors_weights_iterator"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    starts: list[tuple[list[str], threading.Event]] = []
    monkeypatch.setattr(
        mlp,
        "_start_ep_range_prefetch",
        lambda files, ids, threads, block, stop_event: starts.append(
            (files, stop_event)
        ),
    )
    mlp.patch_default_loader_ep_weight_filter()
    return weight_utils.safetensors_weights_iterator, starts


def _names(iterator):
    return {name for name, _ in iterator}


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def test_draft_model_skips_prefetch(tmp_path, monkeypatch, patched_iterator):
    iterator, starts = patched_iterator
    shards = _save_shards(tmp_path)
    monkeypatch.setattr(mlp, "_loading_draft_model", lambda: True)
    names = _names(iterator(shards, False, "prefetch", local_expert_ids={0}))
    assert starts == []
    assert f"{_E}.0.w1.weight_packed" in names
    assert f"{_E}.7.w1.weight_packed" not in names


def test_single_file_skips_prefetch(tmp_path, monkeypatch, patched_iterator):
    iterator, starts = patched_iterator
    shards = _save_shards(tmp_path, count=1)
    monkeypatch.setattr(mlp, "_loading_draft_model", lambda: False)
    names = _names(iterator(shards, False, "prefetch", local_expert_ids={0}))
    assert starts == []
    assert f"{_E}.7.w1.weight_packed" not in names


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(model_config=SimpleNamespace(runner_type="draft")), True),
        (SimpleNamespace(model_config=SimpleNamespace(runner_type="generate")), False),
        (SimpleNamespace(model_config=None), False),
    ],
)
def test_loading_draft_model_reads_current_config(monkeypatch, config, expected):
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    assert mlp._loading_draft_model() is expected


# ---------------------------------------------------------------------------
# Iterator end / close stops the prefetch
# ---------------------------------------------------------------------------


def test_iterator_end_and_close_stop_prefetch(tmp_path, monkeypatch, patched_iterator):
    iterator, starts = patched_iterator
    shards = _save_shards(tmp_path)
    monkeypatch.setattr(mlp, "_loading_draft_model", lambda: False)

    full = iterator(list(reversed(shards)), False, "prefetch", local_expert_ids={0})
    (files, event) = starts[-1]
    assert files == shards  # natural sort order
    assert not event.is_set()
    names = _names(full)
    assert event.is_set()
    assert f"{_E}.0.w1.weight_packed" in names

    partial = iterator(shards, False, "prefetch", local_expert_ids={0})
    event = starts[-1][1]
    next(partial)
    assert not event.is_set()
    partial.close()
    assert event.is_set()


# ---------------------------------------------------------------------------
# _start_ep_range_prefetch honours stop_event
# ---------------------------------------------------------------------------


def _deferred_threads(monkeypatch):
    threads: list[threading.Thread] = []

    class _Deferred(threading.Thread):
        def start(self):
            # mlp.threading is the threading module: only defer the prefetch
            # thread, so ThreadPoolExecutor workers still start.
            if self.name == "ep-range-prefetch":
                threads.append(self)
            else:
                super().start()

    monkeypatch.setattr(mlp.threading, "Thread", _Deferred)
    return threads


def test_preset_stop_event_reads_nothing(tmp_path, monkeypatch, active_events):
    shards = _save_shards(tmp_path)
    threads = _deferred_threads(monkeypatch)
    opened: list[str] = []
    real_open = os.open
    monkeypatch.setattr(
        mlp.os, "open", lambda p, f: opened.append(p) or real_open(p, f)
    )
    event = threading.Event()
    mlp._start_ep_range_prefetch(shards, {0}, 2, 16, stop_event=event)
    assert active_events == [event]
    event.set()
    threads[0].run()
    assert opened == []
    assert active_events == []  # finished prefetches unregister themselves


def test_stop_mid_run_skips_remaining_reads(tmp_path, monkeypatch, active_events):
    shards = _save_shards(tmp_path)
    threads = _deferred_threads(monkeypatch)
    event = threading.Event()
    reads: list[int] = []
    real_pread = os.pread

    def pread(fd, n, off):
        reads.append(off)
        event.set()  # e.g. the weights iterator finished meanwhile
        return real_pread(fd, n, off)

    monkeypatch.setattr(mlp.os, "pread", pread)
    infos: list[str] = []
    monkeypatch.setattr(mlp.logger, "info", lambda msg, *a: infos.append(msg % a))
    mlp._start_ep_range_prefetch(shards, {0}, 1, 16, stop_event=event)
    threads[0].run()
    assert len(reads) == 1
    assert any("stopped after" in m for m in infos)
    assert active_events == []


def test_evict_stops_active_prefetches(tmp_path, monkeypatch, active_events):
    shards = _save_shards(tmp_path)
    threads = _deferred_threads(monkeypatch)
    monkeypatch.setattr(mlp.envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False)
    first, second = threading.Event(), threading.Event()
    mlp._start_ep_range_prefetch(shards, {0}, 2, 16, stop_event=first)
    mlp._start_ep_range_prefetch(shards, {0}, 2, 16, stop_event=second)
    assert active_events == [first, second]

    mlp.evict_loaded_weights_page_cache()
    assert first.is_set() and second.is_set()
    assert active_events == []
    reads: list[int] = []
    monkeypatch.setattr(mlp.os, "pread", lambda fd, n, off: reads.append(off))
    for thread in threads:
        thread.run()  # already unregistered: nothing to remove, nothing read
    assert reads == []
    assert active_events == []


def test_pread_job_skips_once_stopped(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"0123456789")
    fd = os.open(path, os.O_RDONLY)
    try:
        event = threading.Event()
        assert mlp._pread_job((fd, 2, 4)) == 4
        assert mlp._pread_job((fd, 2, 4), event) == 4
        event.set()
        assert mlp._pread_job((fd, 2, 4), event) == 0
    finally:
        os.close(fd)
