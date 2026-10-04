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
"""Per-process record of what a served arm imported and lowered.

The scorer puts this directory first on the server's PYTHONPATH, so every vLLM
process (API server, engine core, each TP worker) runs it at start-up. With
GMMV2_SCORE_RECORD_DIR and GMMV2_SCORE_ARM_ROOT set it appends JSON lines to
`<dir>/<kind>.<pid>.jsonl`:

  module  every Python source file imported from the arm's tree, with the
          sha256 of its bytes: what the process loaded.
  mosaic  every Mosaic kernel lowered into a compiled program, with the sha256
          of its serialized backend config and the LIBTPU_INIT_ARGS in force:
          what the process put on the device.

It then runs the sitecustomize it shadows, so the image's own start-up hook
still applies.
"""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import threading

_REC = os.environ.get("GMMV2_SCORE_RECORD_DIR")
_ROOT = os.environ.get("GMMV2_SCORE_ARM_ROOT")
_LOCK = threading.Lock()
_SEEN: set = set()
_TPU_CUSTOM_CALL = "jax._src.tpu_custom_call"


def _emit(kind: str, rec: dict) -> None:
    line = json.dumps({"pid": os.getpid(), **rec}, sort_keys=True)
    with _LOCK, open(os.path.join(_REC, f"{kind}.{os.getpid()}.jsonl"), "a") as f:
        f.write(line + "\n")


def _record_imports() -> None:
    loader = importlib.machinery.SourceFileLoader
    get_code = loader.get_code
    root = os.path.realpath(_ROOT) + os.sep

    def recording_get_code(self, fullname):
        code = get_code(self, fullname)
        path = os.path.realpath(self.get_filename(fullname))
        if path.startswith(root) and (os.getpid(), path) not in _SEEN:
            _SEEN.add((os.getpid(), path))
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            _emit("module", {"path": path[len(root) :], "sha256": digest})
        return code

    loader.get_code = recording_get_code


def _record_mosaic(module) -> None:
    from jax._src.interpreters import mlir

    lowering = module._tpu_custom_call_lowering

    def recording_lowering(ctx, *in_nodes, config, kernel_name, **kw):
        _emit(
            "mosaic",
            {
                "name": kernel_name or "",
                "sha256": hashlib.sha256(config.to_json()).hexdigest(),
                "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
            },
        )
        return lowering(ctx, *in_nodes, config=config, kernel_name=kernel_name, **kw)

    # Both paths into the lowering: the module attribute, which is looked up at
    # call time, and the rule pallas_call reaches through the primitive.
    module._tpu_custom_call_lowering = recording_lowering
    mlir.register_lowering(module.tpu_custom_call_p, recording_lowering, platform="tpu")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TPU_CUSTOM_CALL:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            _record_mosaic(module)

        spec.loader.exec_module = patched_exec_module
        return spec


def _run_shadowed_sitecustomize() -> None:
    here = os.path.dirname(os.path.realpath(__file__))
    for entry in sys.path:
        cand = os.path.join(entry or os.getcwd(), "sitecustomize.py")
        if os.path.realpath(os.path.dirname(cand)) == here or not os.path.isfile(cand):
            continue
        spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", cand)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        if _REC:
            _emit("chained", {"path": cand})
        return


if _REC and _ROOT:
    os.makedirs(_REC, exist_ok=True)
    _record_imports()
    if _TPU_CUSTOM_CALL in sys.modules:
        _record_mosaic(sys.modules[_TPU_CUSTOM_CALL])
    else:
        sys.meta_path.insert(0, _PatchOnImport())
_run_shadowed_sitecustomize()
