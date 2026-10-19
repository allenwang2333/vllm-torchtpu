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
"""Hook through which blocks reach the experimental fused pool I/O helpers.

`GQABlock` and `GDNBlock` never import `experimental.fused_pool_io`; when
`ex.FUSED` is set they call `fused_io()`, which returns the implementation that
`experimental/fused_pool_io.py` registers on import (`megakernel/__init__.py`
imports `experimental`). Without it, fused builds fail loudly.
"""

from __future__ import annotations

from typing import Any

from ..errors import MegakernelConfigError

_REGISTRY: dict[str, Any] = {}


def register_fused_io(impl: Any) -> None:
    """Install the fused pool I/O helpers (`gqa_fused_*`, `gdn_fused_*`)."""
    _REGISTRY["fused_pool_io"] = impl


def fused_io() -> Any:
    """Return the registered fused pool I/O helpers (only used when `ex.FUSED`)."""
    impl = _REGISTRY.get("fused_pool_io")
    if impl is None:
        raise MegakernelConfigError(
            "ex.FUSED needs megakernel.experimental.fused_pool_io, which is not "
            "loaded (import vllm_torchtpu.megakernel.experimental)"
        )
    return impl


__all__ = ["fused_io", "register_fused_io"]
