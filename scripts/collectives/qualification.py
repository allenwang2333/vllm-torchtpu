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
"""Conservative admission of exact shapes from independent Torch campaigns."""

import hashlib
import json
import math
import re
from pathlib import Path

BATCHES = (1, 2, 4, 8, 16, 32, 64)
PROTOCOL = {"low": 20, "high": 60, "trials": 25, "warmup": 5}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(samples, low, high):
    if not samples or high <= low:
        raise ValueError("expected samples and increasing chain lengths")
    if any(
        len(pair) != 2 or any(not math.isfinite(x) or x <= 0 for x in pair)
        for pair in samples
    ):
        raise ValueError("durations must be finite and positive")
    slopes = [(pair[1] - pair[0]) * 1e6 / (high - low) for pair in samples]
    if not all(math.isfinite(value) for value in slopes):
        raise ValueError("derived slopes must be finite")
    return {
        "samples_seconds": samples,
        "slopes_us": slopes,
        "p10_us": percentile(slopes, 0.1),
        "median_us": percentile(slopes, 0.5),
        "p90_us": percentile(slopes, 0.9),
    }


def chain_dependencies(text, implementation, length):
    """Follow data operands, mapping fusion inputs through used body parameters."""
    computations, roots = {}, {}
    current = entry = None
    for line in text.splitlines():
        header = re.match(r"^(ENTRY )?%([\w.-]+)\s*\(", line)
        if header:
            current = header[2]
            computations[current] = {}
            if header[1]:
                entry = current
        instruction = re.match(
            r"\s*(ROOT\s+)?%([\w.-]+)\s*=\s*(.*?)\s+([a-z][\w-]*)\(([^)]*)\)(.*)",
            line,
        )
        if instruction and current:
            root, name, shape, opcode, arguments, attributes = instruction.groups()
            for marker in (
                ", metadata=",
                ", backend_config=",
                ", frontend_attributes=",
            ):
                attributes = attributes.split(marker, 1)[0]
            called = re.search(r"calls=%([\w.-]+)", attributes)
            computations[current][name] = {
                "opcode": opcode,
                "shape": shape,
                "literal": arguments if opcode in ("constant", "parameter") else None,
                "attributes": attributes,
                "operands": re.findall(r"%([\w.-]+)", arguments),
                "parameter": int(arguments) if opcode == "parameter" else None,
                "fusion": called[1] if called else None,
            }
            if root:
                roots[current] = name
    if entry is None:
        return {
            "data_dependencies_ok": False,
            "dependent_edges": 0,
            "output_depends_on_final": False,
            "collective_dag_sha256": None,
            "collective_dag": [],
        }

    parameter_cache = {}

    def operands(computation, name):
        node = computations.get(computation, {}).get(name)
        if node is None:
            return []
        inputs = node["operands"]
        if node["opcode"] == "fusion":
            body = node["fusion"]
            indices = used_parameters(body, roots.get(body))
            return [inputs[i] for i in sorted(indices) if i < len(inputs)]
        if node["opcode"] in ("call", "while", "conditional"):
            return []
        return inputs

    def used_parameters(computation, name):
        key = (computation, name)
        if key not in parameter_cache:
            parameter_cache[key] = set()
            node = computations.get(computation, {}).get(name)
            if node:
                if node["opcode"] == "parameter":
                    parameter_cache[key] = {node["parameter"]}
                else:
                    for operand in operands(computation, name):
                        parameter_cache[key].update(
                            used_parameters(computation, operand)
                        )
        return parameter_cache[key]

    def depends_on(name, target, seen):
        pending = [name]
        while pending:
            name = pending.pop()
            if name == target:
                return True
            if name not in seen:
                seen.add(name)
                pending.extend(operands(entry, name))
        return False

    collectives = [
        name
        for name, node in computations[entry].items()
        if (
            node["opcode"] in ("all-reduce", "all-reduce-start")
            if implementation == "native"
            else node["opcode"] == "custom-call" and "two_shot_all_reduce" in name
        )
    ]
    ordinals = {name: i for i, name in enumerate(collectives)}
    signatures = {}

    def signature(computation, name, expand_collective=False):
        key = (computation, name, expand_collective)
        if key in signatures:
            return signatures[key]
        node = computations.get(computation, {}).get(name)
        if node is None:
            raise ValueError("HLO data operand or called computation is missing")
        if computation == entry and name in ordinals and not expand_collective:
            return digest({"collective": ordinals[name]})

        # Computation and instruction names are deliberately absent from the hash.
        def reference(match):
            reference_name = match[1]
            if reference_name in computations:
                return signature(reference_name, roots.get(reference_name))
            return signature(computation, reference_name)

        attributes = re.sub(r"%([\w.-]+)", reference, node["attributes"])
        value = {
            "opcode": node["opcode"],
            "shape": node["shape"],
            "literal": node["literal"],
            "attributes": attributes,
            "operands": [
                signature(computation, x) for x in operands(computation, name)
            ],
        }
        if node["opcode"] == "fusion":
            body = node["fusion"]
            value["used_parameters"] = sorted(used_parameters(body, roots.get(body)))
        signatures[key] = digest(value)
        return signatures[key]

    graph = [
        {
            "ordinal": index,
            "opcode": computations[entry][name]["opcode"],
            "shape": computations[entry][name]["shape"],
            "predecessors": [
                i
                for i, before in enumerate(collectives[:index])
                if depends_on(name, before, set())
            ],
            "operand_graph_sha256": signature(entry, name, True),
        }
        for index, name in enumerate(collectives)
    ]
    edges = sum(index - 1 in node["predecessors"] for index, node in enumerate(graph))
    reaches_output = bool(collectives) and depends_on(
        roots.get(entry), collectives[-1], set()
    )
    return {
        "data_dependencies_ok": len(collectives) == length
        and edges == length - 1
        and reaches_output,
        "dependent_edges": edges,
        "output_depends_on_final": reaches_output,
        "collective_dag_sha256": digest(graph),
        "collective_dag": graph,
    }


def require_graph_agreement(audits, process_count, lengths):
    """Fail closed unless all processes execute the same canonical collective DAG."""
    expected = {(p, n) for p in range(process_count) for n in lengths}
    if (
        len(audits) != len(expected)
        or {(item["process"], item["chain_length"]) for item in audits} != expected
    ):
        raise ValueError("incomplete process HLO evidence")
    for length in lengths:
        signatures = {
            item.get("collective_dag_sha256")
            for item in audits
            if item["chain_length"] == length
        }
        if len(signatures) != 1 or None in signatures:
            raise ValueError("collective dependency graphs differ across processes")


def audit_hlo(text, implementation, length):
    counts = {}
    custom = 0
    pallas_calls = 0
    dependency_reductions = 0
    for line in text.splitlines():
        # Metadata may mention an opcode without executing it.
        instruction = line.split(", metadata=", 1)[0].split(", backend_config=", 1)[0]
        if not re.match(r"^\s*(?:ROOT\s+)?%?[^\s=]+\s*=", instruction):
            continue
        native = re.search(
            r"\s(all-reduce|all-gather|reduce-scatter|collective-permute)(?:-start)?\(",
            instruction,
        )
        if native:
            counts[native[1]] = counts.get(native[1], 0) + 1
        if re.search(r"\sreduce\(", instruction):
            dependency_reductions += 1
        if (
            re.search(r"\bcustom-call\(", instruction)
            and 'custom_call_target="tpu_custom_call"' in line
        ):
            pallas_calls += 1
        if (
            re.search(r"\bcustom-call\(", instruction)
            and 'custom_call_target="tpu_custom_call"' in line
            and "two_shot_all_reduce" in instruction.split("=", 1)[0]
        ):
            custom += 1
    ok = (
        counts == {"all-reduce": length} and pallas_calls == 0
        if implementation == "native"
        else not counts and custom == pallas_calls == length
    )
    dependencies = chain_dependencies(text, implementation, length)
    return {
        "ok": ok
        and dependency_reductions >= length - 1
        and dependencies["data_dependencies_ok"],
        **dependencies,
        "chain_length": length,
        "native_opcodes": counts,
        "two_shot_calls": custom,
        "pallas_calls": pallas_calls,
        "dependency_reductions": dependency_reductions,
    }


def read_run(path):
    path = Path(path)
    data = json.loads(path.read_text())
    data["evidence_file"] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return data


def checked_arm(arm, run, length_keys):
    timing = summarize(
        arm["samples_seconds"], run["protocol"]["low"], run["protocol"]["high"]
    )
    if len(timing["slopes_us"]) != run["protocol"]["trials"]:
        raise ValueError("incomplete timing trials")
    if any(arm.get(key) != value for key, value in timing.items()):
        raise ValueError("saved timing statistics disagree with raw samples")
    durations = arm["per_process_seconds"]
    if len(durations) != run["protocol"]["trials"]:
        raise ValueError("missing per-process timing")
    for pair, gathered in zip(arm["samples_seconds"], durations, strict=True):
        if len(gathered) != 2 or any(
            len(times) != run["process_count"]
            or any(not math.isfinite(value) or value <= 0 for value in times)
            for times in gathered
        ):
            raise ValueError("incomplete process timings")
        if pair != [max(times) for times in gathered]:
            raise ValueError("timing did not use the slowest process")
    evidence = arm["correctness"]
    if not evidence or any(not item["ok"] for item in evidence):
        raise ValueError("missing or failed correctness")
    expected_checks = {
        (r, p, i, n)
        for r in range(32)
        for p in ("exact", "cancellation")
        for i in range(2)
        for n in {1, 2, 3, *length_keys}
    }
    actual_checks = {
        (c["rank"], c["pattern"], c["iteration"], c["length"]) for c in evidence
    }
    if actual_checks != expected_checks or len(evidence) != len(expected_checks):
        raise ValueError("incomplete all-rank changed-input correctness grid")
    all_checks = [*evidence, *arm.get("eager_correctness", [])]
    for item in all_checks:
        if (
            not math.isfinite(item["max_abs_error"])
            or item["max_abs_error"] < 0
            or not 0 <= item["max_scaled_error"] <= 1
            or not isinstance(item["output_sha256"], str)
            or len(item["output_sha256"]) != 64
        ):
            raise ValueError("invalid numerical correctness evidence")
    if arm["implementation"] == "two_shot":
        for pattern, iteration, length in {
            (c["pattern"], c["iteration"], c["length"]) for c in all_checks
        }:
            hashes = {
                c["output_sha256"]
                for c in all_checks
                if (c["pattern"], c["iteration"], c["length"])
                == (pattern, iteration, length)
            }
            if len(hashes) != 1:
                raise ValueError("Two Shot outputs differ across ranks")
    eager = arm.get("eager_correctness", [])
    if (
        len(eager) != 128
        or {(x["rank"], x["pattern"], x["iteration"], x["length"]) for x in eager}
        != {
            (r, p, i, 1)
            for r in range(32)
            for p in ("exact", "cancellation")
            for i in range(2)
        }
        or any(not x["ok"] or not 0 <= x["max_scaled_error"] <= 1 for x in eager)
    ):
        raise ValueError("missing or failed all-rank eager correctness")
    if len(arm["hlo"]) != run["process_count"] * len(length_keys):
        raise ValueError("incomplete or duplicate HLO evidence")
    for item in arm["hlo"]:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("HLO artifact changed")
        audit = audit_hlo(path.read_text(), arm["implementation"], item["chain_length"])
        if not audit["ok"] or any(item[key] != value for key, value in audit.items()):
            raise ValueError("failed repeated HLO audit")
    require_graph_agreement(arm["hlo"], run["process_count"], length_keys)
    return timing


def qualifies(native, candidate):
    saving = native["median_us"] - candidate["median_us"]
    positive_slopes = all(
        math.isfinite(value) and value > 0
        for arm in (native, candidate)
        for value in arm.get("slopes_us", [arm["median_us"]])
    )
    return (
        positive_slopes
        and candidate["median_us"] > 0
        and saving > 0.5
        and saving > native["median_us"] * 0.05
        and candidate["p90_us"] < native["p10_us"]
    )


def qualify(primary, confirmation):
    for run in (primary, confirmation):
        if (
            run.get("schema_version") != 1
            or run.get("framework") != "torch"
            or run.get("protocol") != PROTOCOL
            or run.get("mode") != "forced"
            or run.get("smoke")
            or run.get("world_size") != 32
            or run.get("process_count") != 32
            or not run.get("complete")
        ):
            raise ValueError("serving qualification requires complete full Torch runs")
    for key in (
        "identity",
        "harness_sha256",
        "protocol",
        "world_size",
        "process_count",
    ):
        if primary[key] != confirmation[key]:
            raise ValueError(f"campaigns disagree on {key}")
    if (
        primary["run_id"] == confirmation["run_id"]
        or primary["started_unix"] == confirmation["started_unix"]
    ):
        raise ValueError("campaigns must come from independent launches")
    admitted, decisions = [], []
    for batch in BATCHES:
        rows = []
        passed = True
        for run in (primary, confirmation):
            matches = [
                case
                for case in run["cases"]
                if case["batch"] == batch
                and case["width"] == 7168
                and case["dtype"] == "bfloat16"
            ]
            if len(matches) != 1:
                raise ValueError("campaign missing or duplicating a primary shape")
            if any(
                matches[0]["arms"][key].get("implementation") != key
                for key in ("native", "two_shot")
            ):
                raise ValueError(
                    "benchmark arm label differs from executed implementation"
                )
            native = checked_arm(matches[0]["arms"]["native"], run, (20, 60))
            candidate = checked_arm(matches[0]["arms"]["two_shot"], run, (20, 60))
            accepted = qualifies(native, candidate)
            passed &= accepted
            rows.append(
                {
                    "run_id": run["run_id"],
                    "accepted": accepted,
                    "native": native,
                    "two_shot": candidate,
                }
            )
        if passed:
            admitted.append(batch)
        decisions.append({"batch": batch, "accepted": passed, "runs": rows})
    evidence = {
        "runs": [run["run_id"] for run in (primary, confirmation)],
        "files": [run.get("evidence_file") for run in (primary, confirmation)],
        "minimum_saving_us": 0.5,
        "minimum_saving_fraction": 0.05,
        "separation": "candidate_p90 < native_p10",
        "protocol": PROTOCOL,
    }
    manifest = {"schema_version": 1, "entries": []}
    if admitted:
        manifest["entries"].append(
            {"identity": primary["identity"], "batches": admitted, "evidence": evidence}
        )
    return manifest, {"evidence": evidence, "decisions": decisions}
