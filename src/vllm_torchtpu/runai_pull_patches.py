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
"""Stage Run:AI object-storage file pulls atomically to prevent torn reads.

``ModelConfig.__post_init__`` calls
``maybe_pull_model_tokenizer_for_runai``, which mirrors a
``gs://``/``s3://``/``az://`` model's non-weight files into
``$VLLM_ASSETS_CACHE/model_streamer/<sha256(uri)[:8]>`` and rewrites
``model``/``tokenizer`` to that dir. It pulls TWICE per process: once for
``*.model``/``*.py``/``*.json``, then again for everything but the weights
when ``model == tokenizer`` (the usual case, since ``--tokenizer`` is not
passed).

That dir name is a pure function of the URI, so every process in the pod
picks the same one, and ``ObjectStorageModel.pull_files`` has no lock, no
skip-if-present and no temp+rename -- it downloads straight to the final
path. With ``--api-server-count 8`` that is sixteen concurrent rewrites of
the same ~11 MB ``tokenizer.json`` while sibling API servers are reading
it, so a reader eventually observes a half-written file::

    json.decoder.JSONDecodeError: Unterminated string starting at:
        line 449789 column 7 (char 9852170)
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in
        position 3661823: unexpected end of data

which kills that API server and takes the engine down with::

    RuntimeError: API server ApiServer_6 closed its address pipe without
        reporting its bound ZMQ addresses

The offsets differ per occurrence (char 0, 3.1 MB, 3.7 MB, 9.9 MB), which
is the signature of a torn read rather than a corrupt source object.

To prevent readers from observing partially written files without suppressing
subsequent pulls or bucket updates, ``patch_runai_pull_once`` wraps
``ObjectStorageModel.pull_files`` so each call downloads into a sibling
temporary directory on the same filesystem and atomically moves every pulled
file into ``self.dir`` with ``os.replace``. Readers that opened a file before
``os.replace`` finish reading the previous complete inode; readers that open
it after read the new complete inode.

Installed via a ``vllm.general_plugins`` entry point, which runs before
``create_model_config()`` with ``vllm`` already imported.

TODO(https://github.com/vllm-project/vllm/issues/36141): Remove once upstream
stages ``ObjectStorageModel.pull_files`` atomically
(https://github.com/vllm-project/vllm/pull/36696).
"""

import functools
import os
import tempfile
from typing import Any


def patch_runai_pull_once() -> None:
    """Stage ``ObjectStorageModel.pull_files`` via a temp dir and ``os.replace``."""
    from vllm.transformers_utils.runai_utils import ObjectStorageModel

    if getattr(ObjectStorageModel, "_tpu_runai_pull_once", False):
        return

    original_pull_files = ObjectStorageModel.pull_files

    @functools.wraps(original_pull_files)
    def pull_files(
        self,
        model_path: str = "",
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:
        orig_dir = self.dir
        target_dir = os.path.abspath(orig_dir)
        os.makedirs(target_dir, exist_ok=True)
        parent_dir = os.path.dirname(target_dir)
        prefix = f".{os.path.basename(target_dir)}.tmp."
        with tempfile.TemporaryDirectory(dir=parent_dir, prefix=prefix) as tmp_dir:
            self.dir = tmp_dir
            try:
                original_pull_files(self, model_path, allow_pattern, ignore_pattern)
            finally:
                self.dir = orig_dir
            for root, dirs, files in os.walk(tmp_dir):
                rel_root = os.path.relpath(root, tmp_dir)
                dst_root = (
                    target_dir
                    if rel_root == "."
                    else os.path.join(target_dir, rel_root)
                )
                for d in dirs:
                    os.makedirs(os.path.join(dst_root, d), exist_ok=True)
                for fname in files:
                    os.replace(
                        os.path.join(root, fname),
                        os.path.join(dst_root, fname),
                    )

    osm_cls: Any = ObjectStorageModel
    osm_cls.pull_files = pull_files
    osm_cls._tpu_runai_pull_once = True
