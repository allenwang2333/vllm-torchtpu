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
"""Re-pull Run:AI object-storage aux files on hosts that never resolved them.

When a model is served from object storage (``vllm serve gs://...``
with ``--load-format runai_streamer``), the API-server process pulls the
non-weight files (config/tokenizer/custom code) into a local cache dir
(``ObjectStorageModel.dir``, hashed from the URI) and rewrites
``model_config.model`` to that local path, keeping the original URI in
``model_config.model_weights`` (``ModelConfig.
maybe_pull_model_tokenizer_for_runai``). On a multi-host slice the Ray
worker ranks on OTHER hosts receive the rewritten local path but nothing
ever creates it there, so any per-rank config/tokenizer re-parse (e.g.
custom-config models like Kimi-K3) dies with "Repo id must be in the
form 'repo_name'" naming the missing local dir.

``ensure_runai_aux_files`` mirrors the API server's pull on such ranks:
same deterministic dir (the hash only depends on the URI), same file
patterns. An exclusive flock serializes the ranks sharing a host — the
first one pulls, the rest see config.json and skip.
"""

import fcntl
import os

from vllm.transformers_utils.runai_utils import (ObjectStorageModel,
                                                 is_runai_obj_uri)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def ensure_runai_aux_files(model_config) -> None:
    """Make sure the local Run:AI aux-file dir exists on this host.

    No-op unless ``model_config`` was rewritten by
    ``maybe_pull_model_tokenizer_for_runai`` (i.e. ``model_weights`` is
    an object-storage URI and ``model`` is a local path).
    """
    model_weights = getattr(model_config, "model_weights", "") or ""
    if not is_runai_obj_uri(model_weights):
        return
    local_dir = model_config.model
    if is_runai_obj_uri(local_dir):
        # Not rewritten (pull skipped upstream); nothing to mirror.
        return
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        return

    os.makedirs(local_dir, exist_ok=True)
    lock_path = local_dir.rstrip("/") + ".pull.lock"
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            if os.path.exists(config_path):
                return  # another rank on this host pulled while we waited
            logger.info(
                "[runai-repull] %s missing on this host; pulling aux files "
                "from %s", config_path, model_weights)
            storage_model = ObjectStorageModel(url=model_weights)
            if os.path.realpath(
                    storage_model.dir) != os.path.realpath(local_dir):
                # The hash is a pure function of the URI, so a mismatch means
                # model_config.model is not this URI's cache dir — bail out
                # rather than populate the wrong directory.
                logger.warning(
                    "[runai-repull] cache dir mismatch: model_config.model=%s "
                    "but ObjectStorageModel(%s).dir=%s; skipping repull",
                    local_dir, model_weights, storage_model.dir)
                return
            # Same two pulls as maybe_pull_model_tokenizer_for_runai: model
            # config/code files, then (tokenizer shares the dir) everything
            # except the weights.
            storage_model.pull_files(
                model_weights, allow_pattern=["*.model", "*.py", "*.json"])
            if model_config.tokenizer == local_dir:
                storage_model.pull_files(model_weights,
                                         ignore_pattern=[
                                             "*.pt",
                                             "*.safetensors",
                                             "*.bin",
                                             "*.tensors",
                                             "*.pth",
                                         ])
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
