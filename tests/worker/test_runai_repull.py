# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the per-rank Run:AI aux-file pull (runai_repull)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.worker.runai_repull import (
    COMPLETE_SENTINEL,
    PULLING_SENTINEL,
    ensure_runai_aux_files,
)

MODEL_URI = "gs://bucket/models/kimi/k3"
TOK_URI = "gs://bucket/models/kimi/tok"


class FakeModelConfig:
    """Stands in for vllm's ModelConfig: carries the three fields the
    repull reads/writes and a maybe_pull_model_tokenizer_for_runai that
    behaves like upstream's (pull into a URI-keyed dir, rewrite the
    fields), against a tmp cache root."""

    def __init__(self, cache_root, model, tokenizer=None, model_weights=""):
        self.cache_root = str(cache_root)
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else model
        self.model_weights = model_weights
        self.pull_calls = []
        self.crash_after_config = False

    def dir_for(self, uri):
        return os.path.join(self.cache_root, uri.rsplit("/", 1)[1])

    def maybe_pull_model_tokenizer_for_runai(self, model, tokenizer):
        self.pull_calls.append((model, tokenizer))
        if self.model_weights:
            return
        if not model.startswith("gs://"):
            return
        d = self.dir_for(model)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{}")
        if self.crash_after_config:
            raise RuntimeError("simulated SIGKILL mid-pull")
        with open(os.path.join(d, "modeling_custom.py"), "w") as f:
            f.write("")
        self.model_weights = model
        self.model = d
        if tokenizer == model:
            self.tokenizer = d
        elif tokenizer.startswith("gs://"):
            td = self.dir_for(tokenizer)
            os.makedirs(td, exist_ok=True)
            self.tokenizer = td


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path / "assets"


def _v2_config(cache_root, **kw):
    """Ray V2 state: head's rewritten config (local dir + URI)."""
    local = os.path.join(str(cache_root), "k3")
    return FakeModelConfig(cache_root, model=local, model_weights=MODEL_URI, **kw)


def test_noop_for_non_object_storage_model(cache_root):
    cfg = FakeModelConfig(cache_root, model="Qwen/Qwen3-0.6B")
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == []


def test_v2_pulls_when_dir_missing_and_restores_fields(cache_root):
    cfg = _v2_config(cache_root)
    local = cfg.model
    ensure_runai_aux_files(cfg)
    # Upstream's pull was asked for model AND tokenizer from the model URI
    # (tokenizer shared the model dir on the head).
    assert cfg.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert cfg.model == local and cfg.tokenizer == local
    assert cfg.model_weights == MODEL_URI
    assert os.path.exists(os.path.join(local, "config.json"))
    assert os.path.exists(os.path.join(local, PULLING_SENTINEL))
    assert os.path.exists(os.path.join(local, COMPLETE_SENTINEL))


def test_v2_skips_when_sibling_rank_completed(cache_root):
    cfg = _v2_config(cache_root)
    ensure_runai_aux_files(cfg)
    again = _v2_config(cache_root)
    ensure_runai_aux_files(again)
    assert again.pull_calls == []
    assert again.model_weights == MODEL_URI


def test_v2_skips_on_head_host_dir_pulled_by_api_server(cache_root):
    # The API server's own pull leaves config.json and no worker sentinel.
    cfg = _v2_config(cache_root)
    os.makedirs(cfg.model)
    with open(os.path.join(cfg.model, "config.json"), "w") as f:
        f.write("{}")
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == []


def test_v2_repulls_after_interrupted_worker_pull(cache_root):
    crashed = _v2_config(cache_root)
    crashed.crash_after_config = True
    with pytest.raises(RuntimeError):
        ensure_runai_aux_files(crashed)
    local = crashed.model
    # config.json landed but the pull never completed.
    assert os.path.exists(os.path.join(local, "config.json"))
    assert os.path.exists(os.path.join(local, PULLING_SENTINEL))
    assert not os.path.exists(os.path.join(local, COMPLETE_SENTINEL))

    retry = _v2_config(cache_root)
    ensure_runai_aux_files(retry)
    assert retry.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert os.path.exists(os.path.join(local, COMPLETE_SENTINEL))


def test_v1_pre_pull_state_pulls_and_rewrites(cache_root):
    # Ray V1 restores the URI into model and clears model_weights.
    cfg = FakeModelConfig(cache_root, model=MODEL_URI, model_weights=None)
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert cfg.model == cfg.dir_for(MODEL_URI)
    assert cfg.tokenizer == cfg.model
    assert cfg.model_weights == MODEL_URI
    assert os.path.exists(os.path.join(cfg.model, COMPLETE_SENTINEL))


def test_v1_separate_tokenizer_uri_passes_through(cache_root):
    cfg = FakeModelConfig(
        cache_root, model=MODEL_URI, tokenizer=TOK_URI, model_weights=None
    )
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, TOK_URI)]
    assert cfg.tokenizer == cfg.dir_for(TOK_URI)


def test_v2_separate_tokenizer_dir_missing_is_reported(cache_root):
    # Head pulled a distinct tokenizer URI into its own dir; the URI is not
    # retained, so the worker can only flag the gap.
    missing_tok = os.path.join(str(cache_root), "tok-only-on-head")
    cfg = _v2_config(cache_root, tokenizer=missing_tok)
    with patch("vllm_torchtpu.worker.runai_repull.logger") as log:
        ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, missing_tok)]
    assert cfg.tokenizer == missing_tok
    messages = [str(c.args[0]) % c.args[1:] for c in log.error.call_args_list]
    assert any(
        "[runai-repull] tokenizer dir" in m and missing_tok in m for m in messages
    ), messages


def test_noop_for_mock_model_config():
    # TPUWorker unit tests construct the worker with MagicMock configs.
    cfg = MagicMock()
    ensure_runai_aux_files(cfg)
    cfg.maybe_pull_model_tokenizer_for_runai.assert_not_called()


def test_patch_runai_pull_once_partial_then_full_pull_and_refresh(
    cache_root, monkeypatch
):
    import fnmatch
    import sys
    from types import ModuleType
    from typing import Any

    from vllm_torchtpu import runai_pull_patches as rpp

    bucket = {
        MODEL_URI + "/": {
            "config.json": '{"arch": "v1"}',
            "modeling_custom.py": "# code",
            "merges.txt": "a b\n",
            "chat_template.jinja": "{{ messages }}",
            "sub/extra.json": '{"nested": true}',
            "model.safetensors": "weights",
        },
        TOK_URI + "/": {
            "tokenizer.json": '{"tok": 1}',
            "merges.txt": "t o k\n",
        },
    }

    def fake_runai_pull_files(
        model_path, dst_dir, allow_pattern=None, ignore_pattern=None
    ):
        for rel_path, content in bucket[model_path].items():
            fname = os.path.basename(rel_path)
            if allow_pattern is not None and not any(
                fnmatch.fnmatch(fname, pat) for pat in allow_pattern
            ):
                continue
            if ignore_pattern is not None and any(
                fnmatch.fnmatch(fname, pat) for pat in ignore_pattern
            ):
                continue
            out_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                f.write(content)

    class FakeObjectStorageModel:
        def __init__(self, url: str):
            self.dir = os.path.join(str(cache_root), url.rsplit("/", 1)[1])
            os.makedirs(self.dir, exist_ok=True)

        def pull_files(self, model_path="", allow_pattern=None, ignore_pattern=None):
            if not model_path.endswith("/"):
                model_path = model_path + "/"
            fake_runai_pull_files(model_path, self.dir, allow_pattern, ignore_pattern)

    def simulate_upstream_pull(model: str, tokenizer: str) -> tuple[str, str]:
        osm = FakeObjectStorageModel(model)
        osm.pull_files(model, allow_pattern=["*.model", "*.py", "*.json"])
        if model == tokenizer:
            osm.pull_files(model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
            return osm.dir, osm.dir
        tok_osm = FakeObjectStorageModel(tokenizer)
        tok_osm.pull_files(tokenizer, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        return osm.dir, tok_osm.dir

    fake_runai_utils: Any = ModuleType("vllm.transformers_utils.runai_utils")
    fake_runai_utils.ObjectStorageModel = FakeObjectStorageModel
    monkeypatch.setitem(
        sys.modules, "vllm.transformers_utils.runai_utils", fake_runai_utils
    )

    rpp.patch_runai_pull_once()
    # Second call is idempotent.
    rpp.patch_runai_pull_once()

    # 1. Launch with model != tokenizer: only *.model/*.py/*.json pulled for model.
    model_dir, tok_dir = simulate_upstream_pull(MODEL_URI, TOK_URI)
    assert os.path.exists(os.path.join(model_dir, "config.json"))
    assert os.path.exists(os.path.join(model_dir, "modeling_custom.py"))
    assert os.path.exists(os.path.join(model_dir, "sub", "extra.json"))
    assert not os.path.exists(os.path.join(model_dir, "merges.txt"))
    assert not os.path.exists(os.path.join(model_dir, "chat_template.jinja"))
    assert os.path.exists(os.path.join(tok_dir, "merges.txt"))

    # 2. Subsequent launch on the same host with model == tokenizer must pull
    # the remaining tokenizer/template files (merges.txt, chat_template.jinja).
    model_dir_2, tok_dir_2 = simulate_upstream_pull(MODEL_URI, MODEL_URI)
    assert model_dir_2 == model_dir == tok_dir_2
    assert os.path.exists(os.path.join(model_dir, "merges.txt"))
    assert os.path.exists(os.path.join(model_dir, "chat_template.jinja"))
    assert not os.path.exists(os.path.join(model_dir, "model.safetensors"))

    # 3. Bucket updates are picked up on subsequent pulls, and no temp dirs leak.
    bucket[MODEL_URI + "/"]["config.json"] = '{"arch": "v2"}'
    simulate_upstream_pull(MODEL_URI, MODEL_URI)
    with open(os.path.join(model_dir, "config.json")) as f:
        assert f.read() == '{"arch": "v2"}'
    leftover_tmp = [name for name in os.listdir(cache_root) if ".tmp." in name]
    assert leftover_tmp == []


def test_patch_runai_pull_once_multiprocess_no_torn_reads(cache_root, monkeypatch):
    import json
    import multiprocessing
    import sys
    import time
    from types import ModuleType
    from typing import Any

    from vllm_torchtpu import runai_pull_patches as rpp

    payload = json.dumps({"vocab": {f"tok_{i}": i for i in range(4000)}})
    chunks = [payload[i : i + 512] for i in range(0, len(payload), 512)]

    class FakeObjectStorageModel:
        def __init__(self, url: str):
            self.dir = os.path.join(str(cache_root), url.rsplit("/", 1)[1])
            os.makedirs(self.dir, exist_ok=True)

        def pull_files(self, model_path="", allow_pattern=None, ignore_pattern=None):
            # Simulate runai_pull_files streaming in-place without temp+rename:
            # opening with "w" truncates immediately, then writes chunks slowly.
            target = os.path.join(self.dir, "tokenizer.json")
            with open(target, "w") as f:
                for idx, chunk in enumerate(chunks):
                    f.write(chunk)
                    if idx % 8 == 0:
                        f.flush()
                        time.sleep(0.0002)

    fake_runai_utils: Any = ModuleType("vllm.transformers_utils.runai_utils")
    fake_runai_utils.ObjectStorageModel = FakeObjectStorageModel
    monkeypatch.setitem(
        sys.modules, "vllm.transformers_utils.runai_utils", fake_runai_utils
    )

    rpp.patch_runai_pull_once()

    # Initial pull populates the file before concurrent readers/writers race.
    osm = FakeObjectStorageModel(MODEL_URI)
    osm.pull_files(MODEL_URI)
    tok_path = os.path.join(osm.dir, "tokenizer.json")

    ctx = multiprocessing.get_context("fork")
    stop_event = ctx.Event()
    errors = ctx.Queue()

    def writer_loop() -> None:
        try:
            for _ in range(10):
                FakeObjectStorageModel(MODEL_URI).pull_files(MODEL_URI)
        except BaseException as exc:  # pragma: no cover
            errors.put(f"writer: {exc!r}")

    def reader_loop() -> None:
        reads = 0
        try:
            while not stop_event.is_set() or reads < 20:
                with open(tok_path) as f:
                    data = f.read()
                parsed = json.loads(data)
                if len(parsed["vocab"]) != 4000:
                    errors.put(f"unexpected vocab len: {len(parsed['vocab'])}")
                    return
                reads += 1
        except BaseException as exc:
            errors.put(f"reader: {exc!r}")

    writers = [ctx.Process(target=writer_loop) for _ in range(3)]
    readers = [ctx.Process(target=reader_loop) for _ in range(3)]
    for p in readers + writers:
        p.start()
    for p in writers:
        p.join(timeout=30)
        assert p.exitcode == 0
    stop_event.set()
    for p in readers:
        p.join(timeout=30)
        assert p.exitcode == 0

    collected_errors = []
    while not errors.empty():
        collected_errors.append(errors.get_nowait())
    assert collected_errors == []


def test_general_plugins_entry_point_registered():
    import tomllib
    from pathlib import Path

    from vllm_torchtpu.runai_pull_patches import patch_runai_pull_once

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    general_plugins = data["project"]["entry-points"]["vllm.general_plugins"]
    assert (
        general_plugins.get("torchtpu_runai_pull")
        == "vllm_torchtpu.runai_pull_patches:patch_runai_pull_once"
    )
    assert callable(patch_runai_pull_once)
