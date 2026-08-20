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
"""Pull Run:AI object-storage aux files on worker ranks whose host lacks them.

``ModelConfig.__post_init__`` (API-server process) pulls the non-weight
files of a ``gs://``/``s3://`` model — config, tokenizer, custom code —
into a local cache dir and rewrites ``model_config.model`` /
``tokenizer`` to it, keeping the URI in ``model_weights``
(``ModelConfig.maybe_pull_model_tokenizer_for_runai``). That runs once,
on the head host; ``__post_init__`` is not re-run when the config is
shipped to workers. On a multi-host slice the remote ranks therefore
arrive in one of two states, neither of which has the files locally:

* Ray V2 executor: the head's config as-is — ``model`` is a local dir
  that exists only on the head host, ``model_weights`` is the URI.
* Ray V1 executor (``ray_distributed_executor.py``): the URI restored
  into ``model`` and ``model_weights`` cleared, i.e. the pre-pull state.

Either way a per-rank config/tokenizer re-parse (custom-config models
such as Kimi-K3) dies with "Repo id must be in the form 'repo_name'"
naming the missing dir. ``ensure_runai_aux_files`` runs upstream's own
pull on such ranks — same dir, same file set, same field rewrites, so
nothing here can drift from the API server — serialized per host by
upstream's ``get_lock`` on the URI.

Completion is tracked with two sentinel files in the pulled dir rather
than by ``config.json`` existing: upstream downloads in place (no
temp+rename) and ``config.json`` is the first file written, so a
sibling rank could otherwise observe a half-populated dir.
``.vllm_torchtpu_pulling`` is created before a worker-side pull starts
and ``.vllm_torchtpu_aux_complete`` after it finishes; a dir with
``config.json`` and neither sentinel was populated by this run's API
server (which finished before any worker was spawned), so head-host
ranks do not re-pull.
"""

import os

from vllm.model_executor.model_loader.weight_utils import (atomic_writer,
                                                           get_lock)
from vllm.transformers_utils.runai_utils import is_runai_obj_uri

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PULLING_SENTINEL = ".vllm_torchtpu_pulling"
COMPLETE_SENTINEL = ".vllm_torchtpu_aux_complete"


def _touch(path: str, content: str) -> None:
    with atomic_writer(path) as fh:
        fh.write(content)


def _is_complete(local_dir: str) -> bool:
    if os.path.exists(os.path.join(local_dir, COMPLETE_SENTINEL)):
        return True
    # Populated by the API server's own pull (no worker-side sentinel) —
    # complete by construction: __post_init__ returned before the engine
    # spawned any worker.
    return (os.path.exists(os.path.join(local_dir, "config.json"))
            and not os.path.exists(os.path.join(local_dir, PULLING_SENTINEL)))


def ensure_runai_aux_files(model_config) -> None:
    """Make sure this host has the model's Run:AI aux files and that
    ``model_config`` points at them.

    No-op unless the model comes from object storage (see module doc).
    """
    model_weights = getattr(model_config, "model_weights", None) or ""
    if is_runai_obj_uri(model_weights):
        # Ray V2: rewritten on the head; the local dir may be missing here.
        uri = model_weights
        rewritten = True
    elif is_runai_obj_uri(model_config.model):
        # Ray V1: pre-pull state, nobody has pulled on this host yet.
        uri = model_config.model
        rewritten = False
    else:
        return

    tokenizer = model_config.tokenizer
    if rewritten and tokenizer == model_config.model:
        # Upstream pulled the tokenizer into the model dir; ask it to do
        # the same here.
        tokenizer = uri

    with get_lock(uri):
        if rewritten and _is_complete(model_config.model):
            _report_tokenizer_gap(model_config)
            return
        logger.info(
            "[runai-repull] pulling aux files for %s on this host "
            "(model_config.model=%s)", uri, model_config.model)
        # Upstream treats a set model_weights as "already pulled".
        model_config.model_weights = None
        if rewritten:
            os.makedirs(model_config.model, exist_ok=True)
            _touch(os.path.join(model_config.model, PULLING_SENTINEL), uri)
        model_config.maybe_pull_model_tokenizer_for_runai(uri, tokenizer)
        # Without the V2 rewrite the dir is only known after the pull.
        if not rewritten:
            _touch(os.path.join(model_config.model, PULLING_SENTINEL), uri)
        _touch(os.path.join(model_config.model, COMPLETE_SENTINEL), uri)
        _report_tokenizer_gap(model_config)


def _report_tokenizer_gap(model_config) -> None:
    """A tokenizer served from a different object-storage URI than the
    model is pulled into its own dir on the head host, and ModelConfig
    does not retain that URI — so it cannot be mirrored here. Say so
    instead of letting the worker die on an opaque repo-id error."""
    tokenizer = model_config.tokenizer
    if (tokenizer and tokenizer != model_config.model
            and not is_runai_obj_uri(tokenizer) and os.path.isabs(tokenizer)
            and not os.path.isdir(tokenizer)):
        logger.error(
            "[runai-repull] tokenizer dir %s is missing on this host and its "
            "object-storage URI is not retained on ModelConfig, so it cannot "
            "be re-pulled; any worker-side tokenizer use will fail. Pass the "
            "model URI as --tokenizer (shared dir) or stage the tokenizer on "
            "every host.", tokenizer)
