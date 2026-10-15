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
"""Stage Run:AI object-storage file pulls atomically and once per host.

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

Two problems, two mechanisms
----------------------------

1. **Torn reads.** ``patch_runai_pull_once`` wraps
   ``ObjectStorageModel.pull_files`` so each call downloads into a sibling
   temporary directory on the same filesystem and atomically moves every
   pulled file into ``self.dir`` with ``os.replace``. Readers that opened a
   file before ``os.replace`` finish reading the previous complete inode;
   readers that open it after read the new complete inode.

2. **Request fan-out.** Atomic staging alone still lets every process
   download: with 1 launcher + 8 API servers pulling twice, a host issues
   ~18 pulls of the same objects at start-up, and a multi-host deployment
   issues them all at the same moment (Kubernetes starts every pod at
   once). On a 5-host P/D deployment (3 prefill + 2 decode hosts) that was
   ~90 concurrent pulls of the same objects, and GCS throttled them::

       google.api_core.exceptions.RetryError: Timeout of 120.0s exceeded,
           last exception: ('Request failed with status code', 429, ...)
       RuntimeError: API server ApiServer_5 closed its address pipe
           without reporting its bound ZMQ addresses

   or, when the retries did succeed, delayed one host's API servers past
   vLLM's 5-minute engine/front-end handshake. 2- and 3-host deployments
   of the same model never hit the limit, which is why this only showed
   up at 5 hosts. To collapse the fan-out, the wrapper serialises pulls
   into the same ``self.dir`` with a file lock and records each
   *completed* pull in a marker that names exactly what was pulled
   (``model_path``, ``allow_pattern``, ``ignore_pattern``). A sibling that
   acquires the lock and finds a fresh marker for the same pull skips the
   network. The marker is only trusted for ``TPU_RUNAI_PULL_REUSE_S``
   seconds (default 900): long enough that all processes of one server
   start share a single download per pull, short enough that a later
   restart on the same host refreshes config/tokenizer/chat-template from
   the bucket as before. ``TPU_RUNAI_PULL_REUSE_S=0`` disables the skip and
   keeps only the atomic staging.

   Keying the marker on the pull arguments matters: a ``model != tokenizer``
   launch pulls only ``*.model``/``*.py``/``*.json`` into the model dir, and
   a later ``model == tokenizer`` launch must still fetch ``merges.txt``,
   ``chat_template.jinja`` and friends. Locking on ``self.dir`` (not on the
   URI) covers two servers that reach the same dir through different
   ``(model, tokenizer)`` pairs. The lock file lives next to the pulled
   dirs in the assets cache, so processes that share the cache but not
   ``/tmp`` still see each other's lock.

Installed via a ``vllm.general_plugins`` entry point, which runs before
``create_model_config()`` with ``vllm`` already imported.

TODO(https://github.com/vllm-project/vllm/issues/36141): Remove once upstream
stages ``ObjectStorageModel.pull_files`` atomically
(https://github.com/vllm-project/vllm/pull/36696).
"""

import functools
import hashlib
import json
import os
import tempfile
import time
from typing import Any

from filelock import Timeout

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Marker written into the pulled dir after a pull completes; one per distinct
# (model_path, allow_pattern, ignore_pattern). Hidden so it never collides with
# model files, and distinct from runai_repull's per-dir sentinels.
_MARKER_PREFIX = ".vllm_torchtpu_pulled."
_REUSE_ENV = "TPU_RUNAI_PULL_REUSE_S"
_DEFAULT_REUSE_S = 900


def _reuse_window_s() -> float:
    value = os.getenv(_REUSE_ENV)
    if value is None or value.strip() == "":
        return float(_DEFAULT_REUSE_S)
    try:
        return max(0.0, float(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid value '{value}' for {_REUSE_ENV}: must be a number of seconds."
        ) from exc


def _pull_key(
    model_path: str,
    allow_pattern: list[str] | None,
    ignore_pattern: list[str] | None,
) -> str:
    payload = json.dumps(
        [
            model_path,
            sorted(allow_pattern) if allow_pattern else None,
            sorted(ignore_pattern) if ignore_pattern else None,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _marker_path(target_dir: str, key: str) -> str:
    return os.path.join(target_dir, _MARKER_PREFIX + key)


def _marker_age_s(path: str) -> float | None:
    """Seconds since the marker was written, or ``None`` if there is none."""
    try:
        with open(path) as f:
            written = float(f.read().strip() or "nan")
    except (OSError, ValueError):
        return None
    if written != written:  # NaN: unreadable content, treat as absent.
        return None
    return max(0.0, time.time() - written)


def _write_marker(path: str) -> None:
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=os.path.basename(path) + ".tmp."
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"{time.time():.3f}\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _move_tree(src_dir: str, dst_dir: str) -> None:
    """Move every file under ``src_dir`` into ``dst_dir`` with ``os.replace``."""
    for root, dirs, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        dst_root = dst_dir if rel_root == "." else os.path.join(dst_dir, rel_root)
        for d in dirs:
            os.makedirs(os.path.join(dst_root, d), exist_ok=True)
        for fname in files:
            os.replace(os.path.join(root, fname), os.path.join(dst_root, fname))


def patch_runai_pull_once() -> None:
    """Stage ``ObjectStorageModel.pull_files`` atomically, once per host."""
    from vllm.model_executor.model_loader.weight_utils import get_lock
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
        reuse_s = _reuse_window_s()
        marker = _marker_path(
            target_dir, _pull_key(model_path, allow_pattern, ignore_pattern)
        )

        # Lock on the destination dir (not the URI) so every pull that writes
        # here is serialised, whichever (model, tokenizer) pair led to it. The
        # lock file sits in the assets cache next to the pulled dirs.
        lock = get_lock(target_dir, cache_dir=parent_dir)
        try:
            lock.acquire(blocking=False)
        except Timeout:
            logger.info(
                "[runai-pull] waiting for a sibling process to finish pulling %s "
                "into %s",
                model_path,
                target_dir,
            )
            lock.acquire()
        try:
            age = _marker_age_s(marker) if reuse_s > 0 else None
            if age is not None and age <= reuse_s:
                logger.info(
                    "[runai-pull] reusing files pulled %.0fs ago by a sibling "
                    "process for %s (allow=%s ignore=%s); set %s=0 to always "
                    "re-pull",
                    age,
                    model_path,
                    allow_pattern,
                    ignore_pattern,
                    _REUSE_ENV,
                )
                return

            prefix = f".{os.path.basename(target_dir)}.tmp."
            with tempfile.TemporaryDirectory(dir=parent_dir, prefix=prefix) as tmp:
                self.dir = tmp
                try:
                    original_pull_files(self, model_path, allow_pattern, ignore_pattern)
                finally:
                    self.dir = orig_dir
                _move_tree(tmp, target_dir)
            if reuse_s > 0:
                _write_marker(marker)
        finally:
            lock.release()

    osm_cls: Any = ObjectStorageModel
    osm_cls.pull_files = pull_files
    osm_cls._tpu_runai_pull_once = True
