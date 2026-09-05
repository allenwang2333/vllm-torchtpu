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
"""Stand-ins for packages that only exist on a TPU host.

Each module here fakes just enough of a device-only dependency that the
model, layer and config tests can be collected and run on a CPU box. They are
opt-in -- see ``VLLM_TORCHTPU_CPU_STUB`` in ``tests/conftest.py`` -- so a real
TPU run is never shadowed by one.
"""
