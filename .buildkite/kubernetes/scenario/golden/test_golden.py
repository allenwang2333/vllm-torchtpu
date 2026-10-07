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
"""Golden check: the cases and the scenarios they name.

Each case runs a pipeline step of today's scripts with stub programs
(golden_capture.py) for a scenario under .buildkite/kubernetes/scenarios/.
"""

from golden_capture import SCENARIO_ROOT
from golden_cases import qwen_point, sharding_table
from scenario.resolve import Resolver


def test_qwen_points_are_the_sharding_table() -> None:
    resolver = Resolver(SCENARIO_ROOT)
    points = resolver.resolve(resolver.file_of("qwen3.5-397b/inferencex-sweep"))
    names = [qwen_point(conc, sharding) for _, _, conc, sharding in sharding_table()]
    assert [p.point for p in points] == names
