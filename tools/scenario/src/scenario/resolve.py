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
"""From a scenario file to a resolved, validated scenario.

Layers, later wins:

    tool (model defaults) < scenarios/defaults.yaml < extends chain, root
    parent first < the scenario

Defaults for a task apply only when a later layer declares that task, and
perf defaults that only mean something for vllm bench serve stay out of an
inferencex perf task. `name`, `extends` and baselines are not inherited.
Resolution reads only the files and its arguments: no environment, clock,
network or subprocess.
"""

import copy
import dataclasses
import difflib
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from scenario.errors import (
    Diagnostic,
    DiagnosticError,
    Issue,
    issues_from_validation_error,
    sort_diagnostics,
)
from scenario.keycheck import check_keys
from scenario.load import LoadedFile, load_file
from scenario.merge import Origin, OriginOf, Provenance, merge_into
from scenario.model.rules import check_scenario
from scenario.model.scenario import SCHEMA_VERSION, Defaults, Scenario
from scenario.model.tasks import (
    BUILTIN_TASKS,
    GATED_TASKS,
    VLLM_BENCH_ONLY_DEFAULTS,
)
from scenario.paths import KeyPath, format_path

SCENARIOS_DIR = "scenarios"
DEFAULTS_FILE = "defaults.yaml"
MAX_EXTENDS_DEPTH = 5


@dataclass(frozen=True)
class Resolved:
    """A resolved, validated scenario."""

    ref: str
    file: pathlib.Path
    data: Mapping[str, Any]
    model: Scenario
    provenance: Provenance
    warnings: tuple[Diagnostic, ...] = ()
    # Layer names, lowest first: tool, defaults, extends:<ref>..., scenario.
    layers: tuple[str, ...] = ()


def find_root(
    file: pathlib.Path,
    root: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
) -> pathlib.Path:
    """The directory holding scenarios/defaults.yaml.

    It is `root` when given, else the nearest ancestor of `file`, else of
    `cwd` (which the CLI passes, for scratch files outside the repo).
    """
    if root is not None:
        candidates = [root]
    else:
        candidates = list(file.resolve().parents)
        if cwd is not None:
            candidates += [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / SCENARIOS_DIR / DEFAULTS_FILE).is_file():
            return candidate
    where = f" in {root}" if root is not None else " above this file; pass --root"
    message = f"no {SCENARIOS_DIR}/{DEFAULTS_FILE}{where}"
    raise DiagnosticError([Diagnostic(str(file), None, "", message)])


def resolve(
    file: pathlib.Path,
    *,
    root: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
) -> tuple[Resolved, ...]:
    """Resolve and validate a scenario file.

    Raises:
        DiagnosticError: every problem found, in every file.
    """
    resolver = Resolver(find_root(file, root, cwd))
    return resolver.resolve(file.resolve())


@dataclass(frozen=True)
class _FileLayer:
    path: pathlib.Path
    ref: str
    layer: str
    loaded: LoadedFile
    data: dict[str, Any]

    def origin_of(self) -> OriginOf:
        """Origins of this file's data."""
        name = self.layer
        loaded = self.loaded
        return lambda path: Origin(name, loaded.source, loaded.line(path))

    def origin(self, path: KeyPath) -> Origin:
        return self.origin_of()(path)


@dataclass(frozen=True)
class _Merge:
    """A step that merges a layer over the data built so far."""

    data: dict[str, Any]
    origin_of: OriginOf


@dataclass
class _Report:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def at(self, origin: Origin, issue: Issue) -> None:
        self.diagnostics.append(
            Diagnostic(
                origin.source,
                origin.line,
                format_path(issue.path),
                issue.message,
                hint=issue.hint,
                rule=issue.rule,
                severity=issue.severity,
            )
        )

    def in_file(self, layer: _FileLayer, issues: Sequence[Issue]) -> None:
        for issue in issues:
            self.at(layer.origin(issue.path), issue)

    def located(self, prov: Provenance, issues: Sequence[Issue]) -> None:
        for issue in issues:
            self.at(prov.locate(issue.path), issue)

    @property
    def failed(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)


def _get(data: Any, path: KeyPath) -> Any:
    node = data
    for key in path:
        in_dict = isinstance(node, dict) and key in node
        in_list = isinstance(node, list) and isinstance(key, int) and key < len(node)
        if not (in_dict or in_list):
            return None
        node = node[key]
    return node


def _without(data: Any, paths: Sequence[KeyPath]) -> Any:
    data = copy.deepcopy(data)
    for path in paths:
        del _get(data, path[:-1])[path[-1]]
    return data


def _drop_baseline(tasks: Any) -> Any:
    if not isinstance(tasks, dict):
        return tasks
    return {
        name: (
            {k: v for k, v in task.items() if k != "baseline"}
            if name in GATED_TASKS and isinstance(task, dict)
            else task
        )
        for name, task in tasks.items()
    }


def _strip_inherited(data: dict[str, Any]) -> dict[str, Any]:
    """A parent's layer without what a child never inherits."""
    data = {k: v for k, v in data.items() if k not in ("name", "extends")}
    if "tasks" in data:
        data["tasks"] = _drop_baseline(data["tasks"])
    return data


def _scoped_defaults(defaults: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    """The defaults layer for a scenario whose own layers merge to `upper`.

    Task defaults apply only to declared tasks; tasks.inferencex, tasks.perf,
    tasks.agentx, tasks.eval and tasks.eval.lm_eval count separately. Defaults that a
    declared setting replaces wholesale stay out: vllm-bench-only perf
    entries for another client, and num_concurrent when lm_eval sweeps
    concurrency.
    """
    scoped = copy.deepcopy(defaults)
    tasks = scoped.get("tasks")
    if not isinstance(tasks, dict):
        return scoped
    for name in ("inferencex", *BUILTIN_TASKS):
        if _get(upper, ("tasks", name)) is None:
            tasks.pop(name, None)
    perf = tasks.get("perf")
    client = _get(upper, ("tasks", "perf", "client")) or "vllm-bench"
    if isinstance(perf, dict) and client != "vllm-bench":
        for key in VLLM_BENCH_ONLY_DEFAULTS:
            perf.pop(key, None)
    evals = tasks.get("eval")
    if isinstance(evals, dict):
        if _get(upper, ("tasks", "eval", "lm_eval")) is None:
            evals.pop("lm_eval", None)
        model_args = _get(evals, ("lm_eval", "model_args"))
        sweeps = _get(upper, ("tasks", "eval", "lm_eval", "concurrency")) is not None
        if isinstance(model_args, dict) and sweeps:
            model_args.pop("num_concurrent", None)
    return scoped


def _schema_issues(data: dict[str, Any]) -> list[Issue]:
    """Every scenario file states its schema version (V-01)."""
    if "schema" not in data:
        message = f"required: every scenario file starts with schema: {SCHEMA_VERSION}"
        return [Issue(("schema",), message, rule="V-01")]
    version = data["schema"]
    if version != SCHEMA_VERSION or isinstance(version, bool):
        message = (
            f"schema {version!r} is not supported; this tool reads schema "
            f"{SCHEMA_VERSION}"
        )
        return [Issue(("schema",), message, rule="V-01")]
    return []


class Resolver:
    """Resolves scenarios under one root, sharing loaded files."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.scenarios = root / SCENARIOS_DIR
        self._files: dict[pathlib.Path, LoadedFile | DiagnosticError] = {}

    # Files and reference ids ---------------------------------------------

    def scenario_files(self) -> list[pathlib.Path]:
        """Every scenario under scenarios/, by reference id."""
        return [self.file_of(ref) for ref in self.all_refs()]

    def declared_name(self, file: pathlib.Path) -> tuple[str, int] | None:
        """The file's own `name` and its line, or None when it has none."""
        try:
            loaded = self._load(file)
        except DiagnosticError:
            return None
        name = loaded.data.get("name") if isinstance(loaded.data, dict) else None
        return (name, loaded.line(("name",))) if isinstance(name, str) else None

    def ref_of(self, file: pathlib.Path) -> str:
        """The path under scenarios/ without .yaml; `@<path>` outside it."""
        if file.is_relative_to(self.scenarios):
            return file.relative_to(self.scenarios).with_suffix("").as_posix()
        return f"@{file}"

    def file_of(self, ref: str) -> pathlib.Path:
        return self.scenarios / f"{ref}.yaml"

    def source_of(self, file: pathlib.Path) -> str:
        if file.is_relative_to(self.root):
            return file.relative_to(self.root).as_posix()
        return str(file)

    def all_refs(self) -> list[str]:
        return sorted(
            self.ref_of(path)
            for path in self.scenarios.rglob("*.yaml")
            if path.name != DEFAULTS_FILE or path.parent != self.scenarios
        )

    def _near(self, ref: str) -> str:
        matches = difflib.get_close_matches(ref, self.all_refs(), n=3, cutoff=0.5)
        return f" (did you mean {', '.join(matches)}?)" if matches else ""

    def _load(self, file: pathlib.Path) -> LoadedFile:
        if file not in self._files:
            try:
                self._files[file] = load_file(file, self.source_of(file))
            except DiagnosticError as error:
                self._files[file] = error
        loaded = self._files[file]
        if isinstance(loaded, DiagnosticError):
            raise loaded
        return loaded

    def _layer(
        self,
        file: pathlib.Path,
        layer: str,
        model: type[BaseModel],
        report: _Report,
    ) -> _FileLayer | None:
        """A file as a layer: loaded, key-checked, unknown keys left out."""
        try:
            loaded = self._load(file)
        except DiagnosticError as error:
            report.diagnostics.extend(error.diagnostics)
            return None
        unknown = check_keys(loaded.data, model)
        data = _without(loaded.data, [issue.path for issue in unknown])
        file_layer = _FileLayer(file, self.ref_of(file), layer, loaded, data)
        report.in_file(file_layer, [*unknown, *_schema_issues(loaded.data)])
        return file_layer

    # The extends chain ----------------------------------------------------

    def _parent(self, layer: _FileLayer, report: _Report) -> str | None:
        text = layer.data["extends"]
        problem: str | None = None
        ref = text if isinstance(text, str) else ""
        if not isinstance(text, str):
            problem = "expected a scenario reference"
        elif text.startswith(("./", "../")):
            target = (layer.path.parent / text).resolve()
            if target.is_relative_to(self.scenarios):
                ref = self.ref_of(target)
            else:
                problem = f"{text} is outside {SCENARIOS_DIR}/"
        if problem is None and not self.file_of(ref).is_file():
            problem = f"no scenario {ref}{self._near(ref)}"
        if problem is not None:
            report.in_file(layer, [Issue(("extends",), problem, rule="V-29")])
            return None
        return ref

    def _chain(self, file: pathlib.Path, report: _Report) -> list[_FileLayer] | None:
        """The scenario's layer, then its parents', nearest first (V-29)."""
        layer = self._layer(file, "scenario", Scenario, report)
        if layer is None:
            return None
        chain = [layer]
        while layer.data.get("extends") is not None:
            ref = self._parent(layer, report)
            if ref is None:
                return None
            refs = [link.ref for link in chain]
            if ref in refs or len(chain) > MAX_EXTENDS_DEPTH:
                walk = " -> ".join([*refs, ref])
                message = (
                    f"extends cycle: {walk}"
                    if ref in refs
                    else f"extends chains are at most {MAX_EXTENDS_DEPTH} deep: {walk}"
                )
                report.in_file(layer, [Issue(("extends",), message, rule="V-29")])
                return None
            parent = self._layer(self.file_of(ref), f"extends:{ref}", Scenario, report)
            if parent is None:
                return None
            chain.append(parent)
            layer = parent
        return chain

    # Resolution -----------------------------------------------------------

    def resolve(self, file: pathlib.Path) -> tuple[Resolved, ...]:
        """Resolve `file`; see `resolve()`."""
        report = _Report()
        results = self._resolve(file, report)
        diagnostics = sort_diagnostics(dict.fromkeys(report.diagnostics))
        if report.failed:
            raise DiagnosticError(diagnostics)
        warnings = tuple(d for d in diagnostics if d.severity == "warning")
        return tuple(dataclasses.replace(r, warnings=warnings) for r in results)

    def _resolve(self, file: pathlib.Path, report: _Report) -> list[Resolved]:
        defaults = self._layer(
            self.scenarios / DEFAULTS_FILE, "defaults", Defaults, report
        )
        chain = self._chain(file, report)
        if defaults is None or chain is None:
            return []
        try:
            Defaults.model_validate(defaults.data)
        except ValidationError as error:
            report.in_file(defaults, issues_from_validation_error(error, defaults.data))

        scenario, parents = chain[0], chain[1:]
        own = copy.deepcopy(scenario.data)
        if parents:
            own["extends"] = parents[0].ref
        else:
            own.pop("extends", None)
        file_steps = [
            _Merge(_strip_inherited(layer.data), layer.origin_of())
            for layer in reversed(parents)
        ]
        file_steps.append(_Merge(own, scenario.origin_of()))

        results = []
        declared, _ = self._apply(file_steps)
        scoped = _scoped_defaults(defaults.data, declared)
        data, prov = self._apply([_Merge(scoped, defaults.origin_of()), *file_steps])
        resolved = self._validate(scenario, data, prov, report)
        if resolved is not None:
            layers = (
                "tool",
                "defaults",
                *(f"extends:{layer.ref}" for layer in reversed(parents)),
                "scenario",
            )
            results.append(dataclasses.replace(resolved, layers=layers))
        return results

    def _apply(self, steps: Sequence[_Merge]) -> tuple[dict, Provenance]:
        data: dict[str, Any] = {}
        prov = Provenance()
        for step in steps:
            merge_into(data, prov, step.data, step.origin_of)
        return data, prov

    def _validate(
        self,
        scenario: _FileLayer,
        data: dict[str, Any],
        prov: Provenance,
        report: _Report,
    ) -> Resolved | None:
        try:
            model = Scenario.model_validate(data)
        except ValidationError as error:
            report.located(prov, issues_from_validation_error(error, data))
            return None
        issues = check_scenario(model)
        report.located(prov, issues)
        if any(issue.severity == "error" for issue in issues):
            return None
        return Resolved(
            ref=scenario.ref,
            file=scenario.path,
            data=data,
            model=model,
            provenance=prov,
        )
