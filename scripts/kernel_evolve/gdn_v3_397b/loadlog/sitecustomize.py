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
"""Record where every `vllm_torchtpu` module a served arm imports came from.

Put this directory first on PYTHONPATH and set KE_LOADLOG to a file: every
process of the server (API server, engine core, workers) then appends one
`pid<TAB>module<TAB>origin` line per `vllm_torchtpu` module it imports. The
scorer reads the log to prove the server ran the arm's own files and to hash
exactly the code that was loaded. The finder only observes: it returns the
spec the rest of `sys.meta_path` would have returned.
"""

import os
import sys


def _chain_next_sitecustomize() -> None:
    """Run the sitecustomize this file shadows, if the image has one."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                code = compile(fh.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


class _LoadRecorder:
    def __init__(self, log_path: str):
        self._log_path = log_path

    def find_spec(self, name, path=None, target=None):
        if name != "vllm_torchtpu" and not name.startswith("vllm_torchtpu."):
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                with open(self._log_path, "a") as fh:
                    fh.write(f"{os.getpid()}\t{name}\t{spec.origin}\n")
                return spec
        return None


_log = os.environ.get("KE_LOADLOG")
if _log:
    sys.meta_path.insert(0, _LoadRecorder(_log))
_chain_next_sitecustomize()
