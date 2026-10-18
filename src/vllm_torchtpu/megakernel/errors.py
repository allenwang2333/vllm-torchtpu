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
"""Exception types of the megakernel library (framework-free)."""

from __future__ import annotations


class MegakernelConfigError(RuntimeError):
    """Invalid megakernel configuration (fail loudly; never a silent fallback).

    Raised for requests the configuration cannot satisfy: VMEM / SMEM budget
    overflow, `pool_gather=True` or `fused_pool_io=True` on a bucket or pool
    backend that cannot run it, `compile_hlo=True` on a mock-device mesh.

    Deliberately not a `ValueError`: framework adapters treat `ValueError` from
    cache binding as an incompatible KV layout (and fall back to the stock
    path); configuration mistakes must surface instead.
    """


__all__ = ["MegakernelConfigError"]

