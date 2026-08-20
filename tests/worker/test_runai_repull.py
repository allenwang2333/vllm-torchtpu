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
"""Tests for the per-rank Run:AI aux-file repull (runai_repull)."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_torchtpu.worker.runai_repull import ensure_runai_aux_files

GS_URI = "gs://bucket/models/kimi/k3"


def _model_config(model: str,
                  model_weights: str = GS_URI,
                  tokenizer: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        model_weights=model_weights,
        tokenizer=tokenizer if tokenizer is not None else model)


def _storage_model_mock(local_dir: str, populate: bool = True) -> MagicMock:
    """ObjectStorageModel stand-in resolving to local_dir; pull_files
    creates config.json like the real pull would."""
    storage = MagicMock()
    storage.dir = local_dir

    def pull_files(model_path="", allow_pattern=None, ignore_pattern=None):
        if populate:
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, "config.json"), "w") as f:
                f.write("{}")

    storage.pull_files = MagicMock(side_effect=pull_files)
    return storage


def test_noop_when_not_object_storage(tmp_path):
    config = _model_config(model=str(tmp_path / "missing"), model_weights="")
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel") as om:
        ensure_runai_aux_files(config)
    om.assert_not_called()


def test_noop_when_model_not_rewritten(tmp_path):
    # Pull skipped upstream: model is still the gs:// URI.
    config = _model_config(model=GS_URI)
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel") as om:
        ensure_runai_aux_files(config)
    om.assert_not_called()


def test_noop_when_config_already_present(tmp_path):
    local_dir = tmp_path / "bd38833a"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")
    config = _model_config(model=str(local_dir))
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel") as om:
        ensure_runai_aux_files(config)
    om.assert_not_called()


def test_pulls_model_and_tokenizer_files_when_dir_missing(tmp_path):
    local_dir = str(tmp_path / "bd38833a")
    config = _model_config(model=local_dir)
    storage = _storage_model_mock(local_dir)
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel",
               return_value=storage) as om:
        ensure_runai_aux_files(config)
    om.assert_called_once_with(url=GS_URI)
    # Two pulls, mirroring maybe_pull_model_tokenizer_for_runai: config/code
    # files, then the tokenizer's everything-but-weights pass.
    assert storage.pull_files.call_count == 2
    _, first_kwargs = storage.pull_files.call_args_list[0]
    assert first_kwargs["allow_pattern"] == ["*.model", "*.py", "*.json"]
    _, second_kwargs = storage.pull_files.call_args_list[1]
    assert "*.safetensors" in second_kwargs["ignore_pattern"]
    assert os.path.exists(os.path.join(local_dir, "config.json"))


def test_skips_tokenizer_pull_for_separate_tokenizer(tmp_path):
    local_dir = str(tmp_path / "bd38833a")
    config = _model_config(model=local_dir,
                           tokenizer=str(tmp_path / "other-tokenizer"))
    storage = _storage_model_mock(local_dir)
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel",
               return_value=storage):
        ensure_runai_aux_files(config)
    assert storage.pull_files.call_count == 1


def test_bails_out_on_cache_dir_mismatch(tmp_path):
    local_dir = str(tmp_path / "bd38833a")
    config = _model_config(model=local_dir)
    storage = _storage_model_mock(str(tmp_path / "elsewhere"))
    with patch("vllm_torchtpu.worker.runai_repull.ObjectStorageModel",
               return_value=storage):
        ensure_runai_aux_files(config)
    storage.pull_files.assert_not_called()
    assert not os.path.exists(os.path.join(local_dir, "config.json"))
