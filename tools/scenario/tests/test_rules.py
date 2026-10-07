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
"""One case per validation rule message, on scenarios that are otherwise valid."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from scenario.errors import Issue
from scenario.model.rules import RULES, check_scenario, rule_table
from scenario.model.scenario import Scenario
from scenario.model.server import HARDWARE_FIELDS
from scenario.paths import parse_path

Data = dict[str, Any]
DROP = object()


def put(data: Data, text: str, value: object) -> None:
    *parents, last = parse_path(text)
    node: Any = data
    for key in parents:
        if isinstance(key, str) and key not in node:
            node[key] = {}
        node = node[key]
    if value is DROP:
        del node[last]
    else:
        node[last] = value


@dataclass(frozen=True)
class Case:
    rule: str
    fixture: str
    edits: dict[str, object]
    path: str
    message: str
    hint: str | None = None
    severity: str = "error"
    extra: Callable[[Data], None] | None = field(default=None, compare=False)


def apply(case: Case, data: Data) -> Data:
    for text, value in case.edits.items():
        put(data, text, value)
    if case.extra is not None:
        case.extra(data)
    return data


LM = "tasks.eval.lm_eval"
CUSTOM_ENV = "tasks.custom[0].env"

CASES = [
    Case(
        "V-02",
        "single",
        {f"server.{name}": DROP for name in HARDWARE_FIELDS},
        "server",
        "set server.accelerator, topology, hosts and chips_per_host for one role, "
        "or server.roles for several",
    ),
    Case(
        "V-02",
        "single",
        {"server.hosts": DROP, "server.chips_per_host": DROP},
        "server",
        "needs hosts and chips_per_host too; one role sets accelerator, topology, "
        "hosts and chips_per_host",
    ),
    Case(
        "V-02",
        "pd",
        {"server.accelerator": "tpu7x", "server.hosts": 1},
        "server.roles",
        "server.accelerator, server.hosts are set too; with roles, each role sets "
        "its own hardware",
        hint="set them to null under server to drop inherited values",
    ),
    Case(
        "V-03",
        "pd",
        {"server.roles.decode": DROP},
        "server.roles",
        "has one role; set a single role's hardware on server instead",
    ),
    Case(
        "V-04",
        "single",
        {"server.multihost": DROP},
        "server.hosts",
        "is 2; set server.multihost: mp or ray",
    ),
    Case(
        "V-04",
        "pd",
        {"server.multihost": "mp"},
        "server.multihost",
        "is set but every role has one host; remove it",
    ),
    Case(
        "V-42",
        "single",
        {"server.chips_per_host": 2},
        "server.topology",
        "topology 2x2x2 has 8 chips, but hosts 2 x chips_per_host 2 = 4",
    ),
    Case(
        "V-42",
        "pd",
        {"server.roles.decode.topology": "2x4"},
        "server.roles.decode.topology",
        "topology 2x4 has 8 chips, but hosts 1 x chips_per_host 4 = 4",
    ),
    Case(
        "V-05",
        "single",
        {"server.router": {"type": "toy-proxy", "command": ["p"]}},
        "server.router",
        "needs server.roles",
    ),
    Case(
        "V-06",
        "pd",
        {"server.roles.decode.kv.role": "producer"},
        "server.router.type",
        "toy-proxy needs one producer and one consumer role; found producers "
        "[prefill, decode] and consumers []",
    ),
    Case(
        "V-08",
        "pd",
        {"tasks.driver": DROP},
        "tasks",
        "tasks.driver is required with server.roles (one of: prefill, decode, client)",
    ),
    Case(
        "V-08", "pd", {"tasks.driver": "router"}, "tasks.driver", "router is not a role"
    ),
    Case(
        "V-08",
        "single",
        {"tasks.driver": "client"},
        "tasks.driver",
        "a scenario without roles runs tasks on its first host",
    ),
    Case(
        "V-09",
        "single",
        {"server.args.load-format": DROP},
        "server.model.path",
        "is object storage; set load-format: runai_streamer (no other loader",
    ),
    Case(
        "V-10",
        "single",
        {"server.args.distributed-executor-backend": "mp"},
        "server.args.distributed-executor-backend",
        "is mp but server.multihost is ray",
    ),
    Case(
        "V-11",
        "single",
        {"server.env.TPU_TOPOLOGY": "2x2x2"},
        "server.env.TPU_TOPOLOGY",
        "TPU_TOPOLOGY is set by the tool (multi-host and backend wiring); remove it",
    ),
    Case(
        "V-11",
        "pd",
        {"server.roles.decode.env.SCENARIO_HOST_INDEX": "1"},
        "server.roles.decode.env.SCENARIO_HOST_INDEX",
        "SCENARIO_HOST_INDEX is set by the tool (SCENARIO_* names are reserved)",
    ),
    Case(
        "V-11",
        "pd",
        {f"{CUSTOM_ENV}.VLLM_PLUGINS": ""},
        f"{CUSTOM_ENV}.VLLM_PLUGINS",
        "VLLM_PLUGINS is set by the tool on a client driver",
    ),
    Case(
        "V-11",
        "agentx",
        {"tasks.agentx.env.CONC": "4"},
        "tasks.agentx.env.CONC",
        "CONC is set by the tool from tasks.agentx.concurrency; remove it",
    ),
    Case(
        "V-12",
        "single",
        {"server.env.VLLM_ENGINE_READY_TIMEOUT_S": "1800"},
        "server.env.VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_ENGINE_READY_TIMEOUT_S is set from server.startup_timeout; remove it",
        hint="set server.startup_timeout instead (now 2h)",
    ),
    Case(
        "V-13",
        "single",
        {"server.env.JAX_PLATFORMS": "tpu"},
        "server.env.JAX_PLATFORMS",
        "JAX_PLATFORMS is set by the ray backend; remove it",
    ),
    Case(
        "V-14",
        "single",
        {"server.args.data-parallel-size-local": 8},
        "server.args.data-parallel-size-local",
        "srt-slurm style: do not set data-parallel-size-local; the tool derives it "
        "from data-parallel-size and hosts",
    ),
    Case(
        "V-14",
        "pd",
        {"server.roles.decode.args.data-parallel-start-rank": 4},
        "server.roles.decode.args.data-parallel-start-rank",
        "srt-slurm style: do not set data-parallel-start-rank; the tool derives it",
    ),
    Case(
        "V-14",
        "single",
        {"server.args.headless": True},
        "server.args.headless",
        "srt-slurm style: do not set headless; the tool derives it",
    ),
    Case(
        "V-14",
        "single",
        {"server.args.host": "0.0.0.0"},
        "server.args.host",
        "is set by the tool (address binding); remove it",
    ),
    Case(
        "V-14",
        "pd",
        {"server.router.args.port": 9000},
        "server.router.args.port",
        "is set by the tool (router wiring); remove it",
    ),
    Case(
        "V-14",
        "single",
        {"tasks.perf.client_args.num-prompts": 8},
        "tasks.perf.client_args.num-prompts",
        "is set by tasks.perf.num_prompts; remove it",
    ),
    Case(
        "V-17",
        "pd",
        {"server.roles.decode.args.data-parallel-size": "8"},
        "server.roles.decode.args.data-parallel-size",
        "must be a positive integer; got '8'",
    ),
    Case(
        "V-17",
        "single",
        {"server.multihost": "mp", "server.args.data-parallel-size": 15},
        "server.args.data-parallel-size",
        "multihost mp needs data-parallel-size divisible by 2 hosts; got 15",
    ),
]


def issues_for(data: Data) -> list[Issue]:
    return check_scenario(Scenario.model_validate(data))


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{case.rule}-{index}" for index, case in enumerate(CASES)],
)
def test_rule_message(case: Case, request: pytest.FixtureRequest) -> None:
    data = apply(case, request.getfixturevalue(case.fixture))
    found = [issue for issue in issues_for(data) if issue.rule == case.rule]
    assert len(found) == 1, found
    (issue,) = found
    assert issue.path == parse_path(case.path)
    assert issue.message.startswith(case.message), issue.message
    assert issue.severity == case.severity
    if case.hint is not None:
        assert issue.hint == case.hint


def test_every_scenario_rule_has_a_case() -> None:
    covered = {case.rule for case in CASES}
    scenario_rules = {rule.id for rule in RULES.values() if rule.stage == "scenario"}
    assert scenario_rules <= covered


def test_rule_table_is_complete_and_ordered() -> None:
    ids = [rule.id for rule in rule_table()]
    # V-16 is unused: whether TP x PP x PCP x DP fits a role would need the
    # devices per chip, which the tool does not know.
    assert ids == [*(f"V-{n:02d}" for n in range(1, 18) if n != 16), "V-42"]
    assert all(rule.summary for rule in rule_table())


@pytest.mark.parametrize(
    "name", ["SPEC_DECODING", "PREFILL_TP", "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING"]
)
def test_agentx_env_names_the_tool_sets(agentx: Data, name: str) -> None:
    put(agentx, f"tasks.agentx.env.{name}", "x")
    del agentx["sweep"]
    (issue,) = issues_for(agentx)
    assert issue.rule == "V-11"
    assert issue.path == ("tasks", "agentx", "env", name)


@pytest.mark.parametrize("name", ["MODEL", "TP", "CONC", "RESULT_DIR"])
def test_agentx_names_are_free_outside_agentx(
    single: Data, pd: Data, name: str
) -> None:
    """The tool sets them on the AgentX command only; elsewhere they are free."""
    put(single, f"server.env.{name}", "x")
    put(single, f"tasks.perf.env.{name}", "x")
    put(single, f"{LM}.tasks", [{"name": "mmlu_pro", "env": {name: "x"}}])
    assert issues_for(single) == []
    put(pd, f"server.roles.decode.env.{name}", "x")
    put(pd, f"{CUSTOM_ENV}.{name}", "x")
    assert [issue for issue in issues_for(pd) if issue.rule == "V-32"] == []


def test_inferencex_perf_env_keeps_the_import_path(single: Data) -> None:
    put(single, "tasks.perf.client", "inferencex")
    put(single, "tasks.perf.range_ratio_style", DROP)
    put(single, "tasks.perf.env", {"PYTHONPATH": "/x"})
    (issue,) = issues_for(single)
    assert issue.rule == "V-11"
    assert "inferencex client" in issue.message
