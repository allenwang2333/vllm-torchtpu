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
"""plan(): the plumbing registry (P-01..P-22)."""

from scenario.plan.plumbing import ITEMS


def test_registry_is_complete() -> None:
    assert sorted(ITEMS) == [f"P-{n:02d}" for n in range(1, 23)]
    assert all(item.title and item.rule for item in ITEMS.values())
