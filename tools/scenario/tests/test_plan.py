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
"""plan(): the Plan of a point, its gates and --only."""

import pathlib
import re
import textwrap
from typing import Any

import pytest
from plan_fixtures import Data, build
from scenario.errors import DiagnosticError
from scenario.plan.build import PlanError, plan, select_point
from scenario.plan.codec import to_json
from scenario.plan.plumbing import ITEMS
from scenario.plan.types import (
    EvalTaskPlan,
    PerfTaskPlan,
    Plan,
)
from scenario.resolve import resolve

GLM_PATH = "gs://tpu-commons-ci/glm/GLM-5.2-FP8"
GLM = "zai-org/GLM-5.2-FP8"


def task(plan_: Plan, kind: type) -> Any:
    (found,) = [t for t in plan_.driver.tasks if isinstance(t, kind)]
    return found


def test_registry_is_complete() -> None:
    assert sorted(ITEMS) == [f"P-{n:02d}" for n in range(1, 23)]
    assert all(item.title and item.rule for item in ITEMS.values())


def test_gates(variants: dict[str, Data]) -> None:
    single = build(variants["single"])
    gate = task(single, PerfTaskPlan).gate
    assert gate is not None and (gate.tolerance, gate.source) == (0.05, "inline")
    assert gate.as_dict() == {
        "isl8192_osl1024_c1024": {"completed": 2048, "median_tpot_ms": 276.99}
    }
    (run,) = task(single, EvalTaskPlan).runs
    assert run.gate is not None
    assert run.gate.as_dict() == {"mmlu_pro": {"exact_match,custom-extract": 0.7629}}
    assert run.gate.tolerance == 0.025
    experiment = build(variants["single"], overrides=("server.args.max-num-seqs=32",))
    assert experiment.experiment == ("server.args.max-num-seqs=32",)
    assert task(experiment, PerfTaskPlan).gate is None
    assert task(experiment, EvalTaskPlan).runs[0].gate is None


def test_task_order_and_only(variants: dict[str, Data]) -> None:
    data = variants["single"]
    data["tasks"]["custom"] = [{"name": "probe", "command": ["true"], "timeout": 60}]
    kinds = [t.kind for t in build(data).driver.tasks]
    assert kinds == ["perf", "custom", "eval"]
    only = build(data, only=["eval", "probe"])
    assert [t.kind for t in only.driver.tasks] == ["custom", "eval"]
    assert only.only == ("eval", "probe")
    with pytest.raises(PlanError, match="--only names agentx, which this scenario"):
        build(data, only=["agentx"])


def test_plan_identity(variants: dict[str, Data]) -> None:
    single = build(variants["single"])
    assert (single.format, single.scenario_name, single.point) == (
        1,
        "glm-5.2-fp8-dp16-ep",
        None,
    )
    assert (single.image, single.model_name, single.model_path) == (None, GLM, GLM_PATH)
    assert single.experiment == () and single.only is None


def test_shape(variants: dict[str, Data]) -> None:
    shape = build(variants["pd"]).shape()
    assert [r.name for r in shape.roles] == ["prefill", "decode", "client"]
    client = shape.roles[-1]
    assert (client.chipless, client.driver, client.ports) == (True, True, (8100, 8090))
    assert shape.roles[0].ports == (8000,)
    one = build(variants["single_host"]).shape().roles[0]
    assert (one.driver, one.ports, one.backend) == (True, (8000,), "single")
    k3 = build(variants["k3_pd"]).shape().roles
    assert k3[0].ports == (8000, 8100, 8090) and k3[0].scratch_bytes == 64 * 2**30


_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_no_runtime_values_in_a_plan(variants: dict[str, Data]) -> None:
    for data in variants.values():
        text = to_json(build(data))
        assert set(_IPV4.findall(text)) <= {"127.0.0.1", "0.0.0.0"}
        for banned in ("HF_TOKEN", "POD_IP", "JOB_COMPLETION_INDEX", "-0-0."):
            assert banned not in text


DEFAULTS = """\
schema: 1
server: {startup_timeout: 2h, args: {api-server-count: 1}}
"""


def write(root: pathlib.Path, ref: str, text: str) -> pathlib.Path:
    path = root / "scenarios" / f"{ref}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def test_plan_from_files_and_points(tmp_path: pathlib.Path) -> None:
    write(tmp_path, "defaults", DEFAULTS)
    ladder = write(
        tmp_path,
        "ladder",
        """\
        schema: 1
        name: ladder
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
          args: {max-num-seqs: 8}
        sweep:
          name: s{server.args.max-num-seqs}
          zip: {server.args.max-num-seqs: [8, 16]}
        """,
    )
    points = resolve(ladder)
    with pytest.raises(PlanError, match="sweep of 2 points.*--point: s8, s16"):
        select_point(points)
    with pytest.raises(PlanError, match="has no point s4"):
        select_point(points, "s4")
    planned = plan(select_point(points, "s16"))
    assert planned.point == "s16"
    assert "16" in planned.roles[0].host_plans[0].serve.argv


def test_dp_errors_come_from_validation(tmp_path: pathlib.Path) -> None:
    write(tmp_path, "defaults", DEFAULTS)
    bad = write(
        tmp_path,
        "bad",
        """\
        schema: 1
        name: bad
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x4
          hosts: 4
          chips_per_host: 4
          multihost: mp
          args: {data-parallel-size: 6, data-parallel-size-local: 2}
        """,
    )
    with pytest.raises(DiagnosticError) as info:
        resolve(bad)
    messages = [d.render() for d in info.value.diagnostics]
    assert messages == [
        "scenarios/bad.yaml:10: server.args.data-parallel-size-local: srt-slurm "
        "style: do not set data-parallel-size-local; the tool derives it from "
        "data-parallel-size and hosts",
        "scenarios/bad.yaml:10: server.args.data-parallel-size: multihost mp needs "
        "data-parallel-size divisible by 4 hosts; got 6",
    ]
