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
"""Resolution: layers, extends, defaults and provenance."""

import pathlib
import textwrap

import pytest
from scenario.errors import DiagnosticError
from scenario.merge import TOOL, Origin
from scenario.resolve import Resolved, find_root, resolve

DEFAULTS = """\
schema: 1
server:
  image: us-docker.pkg.dev/test/images/vllm:test
  startup_timeout: 2h
  args:
    tensor-parallel-size: 1
    data-parallel-size: 1
    max-model-len: 16384
    max-num-seqs: 512
    api-server-count: 1
tasks:
  perf:
    dataset: {name: random}
    num_prompts: 320
    range_ratio_style: symmetric
    warmup_runs: 0
    percentile_metrics: [ttft, tpot, itl, e2el]
    tolerance: 0.05
    client_args: {backend: vllm, request-rate: inf}
  inferencex: {framework: vllm, spec_decoding: none, kv_offloading: none}
  agentx:
    duration: 1h
    timeout: 3h
    fast: false
  eval:
    lm_eval:
      model: local-chat-completions
      model_args: {num_concurrent: 128}
      seed: "0,1234,None,1234"
      limit: 100
      apply_chat_template: true
      fewshot_as_multiturn: true
"""

BASE = """\
schema: 1
name: glm-base
server:
  model: {name: zai-org/GLM, path: gs://bucket/glm}
  accelerator: tpu7x
  topology: 2x2x2
  hosts: 2
  chips_per_host: 4
  multihost: ray
  args:
    load-format: runai_streamer
    data-parallel-size: 16
    max-num-seqs: 64
    compilation-config: {backend: x, compile_sizes: [1, 8]}
tasks:
  perf:
    isl_osl: [[8192, 1024]]
    concurrency: [1024]
    random_range_ratio: 0.0
    baseline:
      isl8192_osl1024_c1024: {completed: 2048}
  eval:
    lm_eval: {tasks: [mmlu_pro]}
    tolerance: 0.025
    baseline:
      mmlu_pro: {"exact_match,custom-extract": 0.76}
"""


class Repo:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.write("defaults", DEFAULTS)
        self.write("glm/base", BASE)

    def write(self, ref: str, text: str) -> pathlib.Path:
        path = self.root / "scenarios" / f"{ref}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        return path

    def resolve(self, ref: str) -> tuple[Resolved, ...]:
        return resolve(self.root / "scenarios" / f"{ref}.yaml")

    def one(self, ref: str) -> Resolved:
        (resolved,) = self.resolve(ref)
        return resolved

    def errors(self, ref: str) -> list[str]:
        with pytest.raises(DiagnosticError) as info:
            self.resolve(ref)
        return [d.render() for d in info.value.diagnostics]


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> Repo:
    return Repo(tmp_path)


def test_base_resolves_with_defaults(repo: Repo) -> None:
    resolved = repo.one("glm/base")
    assert resolved.ref == "glm/base"
    server = resolved.model.server
    assert server.image == "us-docker.pkg.dev/test/images/vllm:test"
    assert server.startup_timeout == 7200
    assert server.args["max-num-seqs"] == 64
    assert server.args["api-server-count"] == 1
    perf = resolved.model.tasks.perf
    assert perf is not None and perf.num_prompts == 320
    assert perf.client_args == {"backend": "vllm", "request-rate": "inf"}
    prov = resolved.provenance
    assert prov.origin(("server", "image")) == Origin(
        "defaults", "scenarios/defaults.yaml", 3
    )
    assert prov.origin(("server", "args", "max-num-seqs")) == Origin(
        "scenario", "scenarios/glm/base.yaml", 13
    )
    assert prov.superseded[("server", "args", "max-num-seqs")] == [
        Origin("defaults", "scenarios/defaults.yaml", 9)
    ]
    assert prov.origin(("tasks", "perf", "client")) == TOOL


def test_defaults_only_for_declared_tasks(repo: Repo) -> None:
    repo.write(
        "plain",
        """\
        schema: 1
        name: plain
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          custom:
            - {name: probe, command: ["true"], timeout: 1m}
        """,
    )
    tasks = repo.one("plain").data["tasks"]
    assert set(tasks) == {"custom"}
    repo.write(
        "child",
        """\
        schema: 1
        name: child
        extends: glm/base
        tasks:
          perf: null
        """,
    )
    resolved = repo.one("child")
    assert resolved.model.tasks.perf is None
    assert resolved.model.tasks.eval is not None
    assert (
        resolved.provenance.deleted[("tasks", "perf")].source == "scenarios/child.yaml"
    )


def test_inferencex_eval_gets_no_lm_eval_defaults(repo: Repo) -> None:
    """lm_eval defaults need tasks.eval.lm_eval, which client inferencex lacks."""
    repo.write(
        "ix-eval",
        """\
        schema: 1
        name: ix-eval
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          inferencex: {model_prefix: kimik3, precision: fp4}
          eval:
            client: inferencex
            concurrency: 4
            timeout: 8h
        """,
    )
    tasks = repo.one("ix-eval").model.tasks
    evals = tasks.eval
    assert (evals.client, evals.lm_eval, evals.tolerance) == ("inferencex", None, None)
    assert tasks.inferencex.framework == "vllm"
    assert repo.one("glm/base").model.tasks.eval.lm_eval.model == (
        "local-chat-completions"
    )


def test_lm_eval_defaults_need_lm_eval(repo: Repo) -> None:
    repo.write(
        "evals",
        """\
        schema: 1
        name: evals
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval:
              tasks: [gsm8k]
              concurrency: [1, 2]
        """,
    )
    lm_eval = repo.one("evals").model.tasks.eval.lm_eval
    assert lm_eval.model == "local-chat-completions"
    assert lm_eval.model_args == {}
    repo.write(
        "evals2",
        """\
        schema: 1
        name: evals2
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval: {tasks: [gsm8k], concurrency: [1], model_args: {num_concurrent: 4}}
        """,
    )
    assert repo.errors("evals2") == [
        "scenarios/evals2.yaml:12: tasks.eval.lm_eval.model_args.num_concurrent: "
        "is set per run from tasks.eval.lm_eval.concurrency; remove it"
    ]


def test_inferencex_perf_gets_no_vllm_bench_defaults(repo: Repo) -> None:
    repo.write(
        "ix",
        """\
        schema: 1
        name: ix
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          perf:
            client: inferencex
            isl_osl: [[8192, 1024]]
            concurrency: [4]
            random_range_ratio: 0.8
            num_prompts: {per_concurrency: 10}
            num_warmups: {per_concurrency: 2}
        """,
    )
    perf = repo.one("ix").model.tasks.perf
    assert perf.client_args == {}
    assert perf.range_ratio_style is None
    assert perf.percentile_metrics == ["ttft", "tpot", "itl", "e2el"]
    assert perf.num_warmups.per_concurrency == 2


def test_extends_merges_and_strips(repo: Repo) -> None:
    repo.write(
        "glm/child",
        """\
        schema: 1
        name: glm-child
        extends: ./base.yaml
        server:
          args:
            compilation-config: {compile_sizes: [1, 8, 1296]}
        """,
    )
    resolved = repo.one("glm/child")
    assert resolved.model.server.args["compilation-config"] == {
        "backend": "x",
        "compile_sizes": [1, 8, 1296],
    }
    assert resolved.model.name == "glm-child"
    assert resolved.data["extends"] == "glm/base"
    assert resolved.model.tasks.perf.baseline is None
    assert resolved.model.tasks.eval.baseline is None


SWEPT = """\
schema: 1
name: glm-sweep
extends: glm/base
sweep:
  name: s{server.args.max-num-seqs}
  zip:
    server.args.max-num-seqs: [64, 128]
  baselines:
    s64:
      perf:
        isl8192_osl1024_c1024: {completed: 2048}
"""


def test_extends_problems(repo: Repo) -> None:
    repo.write("loop/a", "schema: 1\nname: a\nextends: loop/b\n")
    repo.write("loop/b", "schema: 1\nname: b\nextends: loop/a\n")
    assert repo.errors("loop/a") == [
        "scenarios/loop/b.yaml:3: extends: extends cycle: loop/a -> loop/b -> loop/a"
    ]
    repo.write("typo", "schema: 1\nname: t\nextends: glm/bse\n")
    assert repo.errors("typo") == [
        "scenarios/typo.yaml:3: extends: no scenario glm/bse (did you mean glm/base?)"
    ]
    repo.write("out", "schema: 1\nname: o\nextends: ../../elsewhere.yaml\n")
    assert repo.errors("out") == [
        "scenarios/out.yaml:3: extends: ../../elsewhere.yaml is outside scenarios/"
    ]
    for depth in range(7):
        parent = f"extends: deep/d{depth + 1}\n" if depth < 6 else ""
        repo.write(f"deep/d{depth}", f"schema: 1\nname: d{depth}\n{parent}")
    (message,) = repo.errors("deep/d0")
    assert message.startswith(
        "scenarios/deep/d5.yaml:3: extends: extends chains are at most 5 deep"
    )


def test_typos_are_reported_where_they_are(repo: Repo) -> None:
    repo.write(
        "p",
        """\
        schema: 1
        name: p
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          perf:
            isl_osll: [[1, 1]]
        """,
    )
    repo.write("c", "schema: 1\nname: c\nextends: p\ntasks: {perf: null}\n")
    assert repo.errors("c") == [
        "scenarios/p.yaml:11: tasks.perf.isl_osll: unknown key; did you mean "
        "tasks.perf.isl_osl?"
    ]


def test_missing_field_points_at_its_parent(repo: Repo) -> None:
    repo.write("nomodel", "schema: 1\nname: n\nserver:\n  hosts: 1\n")
    assert repo.errors("nomodel") == [
        "scenarios/nomodel.yaml:3: server.model: required"
    ]


def test_every_file_states_its_schema(repo: Repo) -> None:
    repo.write("noschema", "name: n\nextends: glm/base\n")
    assert repo.errors("noschema") == [
        "scenarios/noschema.yaml:1: schema: required: every scenario file starts "
        "with schema: 1"
    ]


def test_root_discovery(repo: Repo, tmp_path_factory: pytest.TempPathFactory) -> None:
    nested = repo.root / "scenarios" / "glm" / "base.yaml"
    assert find_root(nested) == repo.root
    scratch_dir = tmp_path_factory.mktemp("scratch")
    scratch = scratch_dir / "exp.yaml"
    scratch.write_text("schema: 1\nname: exp\nextends: glm/base\n")
    with pytest.raises(DiagnosticError, match="pass --root"):
        find_root(scratch)
    assert find_root(scratch, cwd=repo.root) == repo.root
    (resolved,) = resolve(scratch, root=repo.root)
    assert resolved.ref == f"@{scratch.resolve()}"
    assert resolved.model.name == "exp"
