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
"""Process-wide defaults required before torch_tpu autoloads."""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types
from collections.abc import Sequence
from typing import Any

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")


class _VllmOmniCompatFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Lazy compatibility shim for vllm-omni==0.32.0rc1 against vLLM main HEAD.

    vllm-omni 0.32.0rc1 imports `vllm.entrypoints.launchers.cli_args` (moved to
    `vllm.entrypoints.common.cli_args` in vllm#62151) and
    `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` (moved to
    `vllm.v1.core.kv_cache_planning` in vllm#61183) at `import vllm_omni` time,
    before `vllm_torchtpu.omni` platform plugins are loaded.
    """

    _LAUNCHERS_PKG = "vllm.entrypoints.launchers"
    _LAUNCHERS_CLI_ARGS = "vllm.entrypoints.launchers.cli_args"

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == self._LAUNCHERS_PKG:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        if fullname == self._LAUNCHERS_CLI_ARGS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=False)
        return None

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        if spec.name == self._LAUNCHERS_CLI_ARGS:
            return importlib.import_module("vllm.entrypoints.common.cli_args")
        return None

    def exec_module(self, module: Any) -> None:
        if module.__name__ == self._LAUNCHERS_PKG:
            module.__path__ = []
            cli_args = importlib.import_module("vllm.entrypoints.common.cli_args")
            sys.modules[self._LAUNCHERS_CLI_ARGS] = cli_args
            module.cli_args = cli_args
        from vllm.v1.core import kv_cache_planning, kv_cache_utils

        kv_cache_utils_any: Any = kv_cache_utils
        kv_cache_utils_any.get_kv_cache_configs = kv_cache_planning.get_kv_cache_configs


if not any(type(f).__name__ == "_VllmOmniCompatFinder" for f in sys.meta_path):
    sys.meta_path.append(_VllmOmniCompatFinder())
