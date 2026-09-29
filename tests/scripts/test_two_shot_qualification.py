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
"""Fail-closed evidence admission without importing accelerator runtimes."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/collectives/qualification.py"
_SPEC = importlib.util.spec_from_file_location("two_shot_qualification_script", _PATH)
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


def test_signed_slopes_are_retained():
    result = qualification.summarize([[30e-6, 10e-6], [10e-6, 30e-6]], 20, 60)
    assert result["slopes_us"] == pytest.approx([-0.5, 0.5])
    assert result["median_us"] == 0
    assert result["p10_us"] < 0 < result["p90_us"]


@pytest.mark.parametrize("samples", [[[0, 1]], [[1, float("nan")]], [[1]], []])
def test_invalid_samples_rejected(samples):
    with pytest.raises(ValueError):
        qualification.summarize(samples, 20, 60)


def _hlo(arm, length):
    rows = [
        "ENTRY %main (x: bf16[4,7168]) -> bf16[4,7168] {",
        "%x = bf16[4,7168] parameter(0)",
        "%zero = f32[] constant(0)",
    ]
    operand = "x"
    for index in range(length):
        name = f"ar.{index}" if arm == "native" else f"two_shot_all_reduce.{index}"
        if arm == "native":
            rows.append(
                f"%{name} = bf16[4,7168] all-reduce(%{operand}), "
                "replica_groups={{0,1}}"
            )
        else:
            rows.append(
                f"%{name} = bf16[4,7168] custom-call(%{operand}), "
                'custom_call_target="tpu_custom_call", custom_call_has_side_effect=true'
            )
        if index < length - 1:
            rows.extend(
                [
                    f"%convert.{index} = f32[4,7168] convert(%{name})",
                    f"%sum.{index} = f32[] reduce(%convert.{index}, %zero), "
                    "dimensions={0,1}",
                    f"%cast.{index} = bf16[] convert(%sum.{index})",
                    f"%broadcast.{index} = bf16[4,7168] broadcast(%cast.{index}), "
                    "dimensions={}",
                    f"%next.{index} = bf16[4,7168] multiply(%x, %broadcast.{index})",
                ]
            )
            operand = f"next.{index}"
    rows.extend([f"ROOT %result = bf16[4,7168] copy(%{name})", "}"])
    return "\n".join(rows)


@pytest.mark.parametrize("arm", ["native", "two_shot"])
def test_hlo_requires_complete_dependent_chain(arm):
    text = _hlo(arm, 3)
    assert qualification.audit_hlo(text, arm, 3)["ok"]
    assert not qualification.audit_hlo(text, arm, 4)["ok"]
    assert not qualification.audit_hlo(text.replace(" reduce(", " copy("), arm, 3)["ok"]


def test_metadata_does_not_count_as_executed_communication():
    text = '%fake = bf16[4,7168] copy(%x), metadata={op_name="all-reduce(%x)"}'
    assert not qualification.audit_hlo(text, "native", 1)["ok"]
    assert not qualification.audit_hlo(
        _hlo("two_shot", 3) + "\n" + _hlo("native", 1), "two_shot", 3
    )["ok"]


@pytest.mark.parametrize(
    "candidate",
    [
        {"median_us": 9.5, "p90_us": 9.6},
        {"median_us": 9.0, "p90_us": 9.8},
        {"median_us": -1.0, "p90_us": -0.5},
    ],
)
def test_admission_rejects_threshold_ties_overlap_and_negative_slope(candidate):
    assert not qualification.qualifies({"median_us": 10.0, "p10_us": 9.8}, candidate)


def test_admission_accepts_separated_measured_win():
    assert qualification.qualifies(
        {"median_us": 10, "p10_us": 9.8}, {"median_us": 9, "p90_us": 9.4}
    )


def _campaign(name):
    return {
        "schema_version": 1,
        "framework": "torch",
        "protocol": qualification.PROTOCOL,
        "mode": "forced",
        "smoke": False,
        "world_size": 32,
        "process_count": 32,
        "complete": True,
        "identity": {"source": "s", "runtime": "r"},
        "harness_sha256": "h",
        "run_id": name,
        "started_unix": float(ord(name)),
        "cases": [
            {
                "batch": b,
                "width": 7168,
                "dtype": "bfloat16",
                "arms": {
                    "native": {
                        "implementation": "native",
                        "median_us": 20,
                        "p10_us": 19,
                        "p90_us": 21,
                    },
                    "two_shot": {
                        "implementation": "two_shot",
                        "median_us": 15,
                        "p10_us": 14,
                        "p90_us": 16,
                    },
                },
            }
            for b in qualification.BATCHES
        ],
    }


def test_manifest_requires_both_runs_and_preserves_losers(monkeypatch):
    monkeypatch.setattr(qualification, "checked_arm", lambda arm, *_: arm)
    primary, confirmation = _campaign("a"), _campaign("b")
    confirmation["cases"][0]["arms"]["two_shot"]["p90_us"] = 19
    manifest, report = qualification.qualify(primary, confirmation)
    assert manifest["entries"][0]["batches"] == list(qualification.BATCHES[1:])
    assert not report["decisions"][0]["accepted"]
    assert len(report["decisions"]) == len(qualification.BATCHES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("framework", "jax"),
        ("smoke", True),
        ("run_id", "a"),
        ("identity", {"source": "different"}),
        ("complete", False),
        ("mode", "auto"),
    ],
)
def test_bad_cohort_cannot_qualify(field, value):
    first, second = _campaign("a"), _campaign("b")
    second[field] = value
    with pytest.raises(ValueError):
        qualification.qualify(first, second)


@pytest.fixture
def evidence(tmp_path):
    run = {"protocol": {"low": 2, "high": 3, "trials": 2}, "process_count": 32}
    checks = [
        {
            "rank": rank,
            "pattern": pattern,
            "iteration": iteration,
            "length": length,
            "ok": True,
            "max_abs_error": 0,
            "max_scaled_error": 0,
            "output_sha256": "0" * 64,
        }
        for rank in range(32)
        for pattern in ("exact", "cancellation")
        for iteration in range(2)
        for length in (1, 2, 3)
    ]
    arm = {
        "implementation": "two_shot",
        "correctness": checks,
        "eager_correctness": [c.copy() for c in checks if c["length"] == 1],
        "hlo": [],
        "per_process_seconds": [[[20e-6] * 32, [30e-6] * 32]] * 2,
        **qualification.summarize([[20e-6, 30e-6]] * 2, 2, 3),
    }
    for length in (2, 3):
        path = tmp_path / f"{length}.hlo"
        path.write_text(_hlo("two_shot", length))
        for rank in range(32):
            arm["hlo"].append(
                {
                    **qualification.audit_hlo(path.read_text(), "two_shot", length),
                    "process": rank,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return arm, run


def test_raw_evidence_reaudit(evidence):
    arm, run = evidence
    assert qualification.checked_arm(arm, run, (2, 3))["median_us"] == pytest.approx(10)


@pytest.mark.parametrize("corruption", ["timing", "rank", "hlo", "eager", "replicas"])
def test_corrupt_raw_evidence_cannot_qualify(evidence, corruption):
    arm, run = evidence
    if corruption == "timing":
        arm["median_us"] += 1
    elif corruption == "rank":
        arm["correctness"].pop()
    elif corruption == "hlo":
        Path(arm["hlo"][0]["path"]).write_text(_hlo("two_shot", 1))
    elif corruption == "eager":
        arm["eager_correctness"][0]["ok"] = False
    else:
        arm["correctness"][0]["output_sha256"] = "1" * 64
    with pytest.raises(ValueError):
        qualification.checked_arm(arm, run, (2, 3))


def test_candidate_reference_enforces_fp32_accumulation(monkeypatch):
    import numpy as np

    monkeypatch.syspath_prepend(str(_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "two_shot_benchmark_test", _PATH.with_name("two_shot_benchmark.py")
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    expected, candidate_bound = benchmark.reference(
        1, 16, "bfloat16", "cancellation", 0, "two_shot"
    )
    native_expected, native_bound = benchmark.reference(
        1, 16, "bfloat16", "cancellation", 0, "native"
    )
    np.testing.assert_array_equal(expected, native_expected)
    assert np.all(candidate_bound < native_bound / 10)
    checks = benchmark.check_outputs(
        [(0, expected + native_bound / 2)], expected, candidate_bound
    )
    assert not checks[0]["ok"]


@pytest.mark.parametrize("arm", ["native", "two_shot"])
def test_nonpositive_raw_slope_is_retained_but_not_admitted(arm):
    statistics = {
        "native": {"median_us": 20, "p10_us": 19, "p90_us": 21, "slopes_us": [20]},
        "two_shot": {"median_us": 15, "p10_us": 14, "p90_us": 16, "slopes_us": [15]},
    }
    statistics[arm]["slopes_us"].append(-1)
    assert not qualification.qualifies(statistics["native"], statistics["two_shot"])


def test_arm_cannot_relabel_native_hlo_as_two_shot(monkeypatch):
    monkeypatch.setattr(qualification, "checked_arm", lambda arm, *_: arm)
    primary, confirmation = _campaign("a"), _campaign("b")
    primary["cases"][0]["arms"]["two_shot"]["implementation"] = "native"
    with pytest.raises(ValueError, match="arm label"):
        qualification.qualify(primary, confirmation)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_abs_error", float("nan")),
        ("output_sha256", "x"),
        ("output_sha256", "1" * 64),
    ],
)
def test_eager_evidence_requires_valid_errors_and_replica_parity(
    evidence, field, value
):
    arm, run = evidence
    arm["eager_correctness"][0][field] = value
    with pytest.raises(ValueError):
        qualification.checked_arm(arm, run, (2, 3))


@pytest.mark.parametrize("arm", ["native", "two_shot"])
def test_independent_calls_cannot_pass_with_dead_checksum_reductions(arm):
    text = _hlo(arm, 3).replace("(%next.0)", "(%x)").replace("(%next.1)", "(%x)")
    audit = qualification.audit_hlo(text, arm, 3)
    assert audit["dependency_reductions"] == 2
    assert audit["dependent_edges"] == 0
    assert not audit["ok"]


@pytest.mark.parametrize("use_input", [True, False])
def test_fusion_operand_must_contribute_to_its_root(use_input):
    body = """%fused (p: bf16[4,7168]) -> bf16[4,7168] {
  %p = bf16[4,7168] parameter(0)
  %zero = bf16[] constant(0)
  %zeros = bf16[4,7168] broadcast(%zero), dimensions={}
  ROOT %result = bf16[4,7168] copy(%REPLACEMENT)
}
""".replace("REPLACEMENT", "p" if use_input else "zeros")
    text = _hlo("two_shot", 2).replace(
        "%next.0 = bf16[4,7168] multiply(%x, %broadcast.0)",
        "%next.0 = bf16[4,7168] fusion(%broadcast.0), kind=kLoop, calls=%fused",
    )
    assert qualification.audit_hlo(body + text, "two_shot", 2)["ok"] == use_input


def test_control_predecessor_is_not_a_data_dependency():
    text = _hlo("two_shot", 2).replace(
        "custom-call(%next.0)",
        "custom-call(%x), control-predecessors={%two_shot_all_reduce.0}",
    )
    assert not qualification.audit_hlo(text, "two_shot", 2)["ok"]


def test_final_collective_must_contribute_to_output():
    text = _hlo("two_shot", 3).replace("copy(%two_shot_all_reduce.2)", "copy(%x)")
    audit = qualification.audit_hlo(text, "two_shot", 3)
    assert audit["dependent_edges"] == 2
    assert not audit["output_depends_on_final"]
    assert not audit["ok"]


def test_dag_signature_ignores_names_and_debug_metadata():
    text = _hlo("two_shot", 3)
    renamed = text.replace("%x", "%another_input").replace("%next.", "%chain.")
    renamed = renamed.replace(
        "custom_call_has_side_effect=true",
        'custom_call_has_side_effect=true, metadata={op_name="rank-specific.name"}',
    )
    first = qualification.audit_hlo(text, "two_shot", 3)
    second = qualification.audit_hlo(renamed, "two_shot", 3)
    assert first["ok"] and second["ok"]
    assert first["collective_dag_sha256"] == second["collective_dag_sha256"]
    assert first["collective_dag"][2]["predecessors"] == [0, 1]


@pytest.mark.parametrize(
    "before,after",
    [
        ("bf16[4,7168]", "bf16[4,7169]"),
        ("constant(0)", "constant(1)"),
        (" multiply(", " add("),
        ("dimensions={0,1}", "dimensions={1,0}"),
        ("custom_call_has_side_effect=true", "custom_call_has_side_effect=false"),
    ],
)
def test_dag_signature_retains_shapes_operations_literals_and_attributes(before, after):
    text = _hlo("two_shot", 3)
    first = qualification.audit_hlo(text, "two_shot", 3)
    second = qualification.audit_hlo(text.replace(before, after), "two_shot", 3)
    assert first["ok"] and second["ok"]
    assert first["collective_dag_sha256"] != second["collective_dag_sha256"]


def test_admission_rejects_individually_valid_but_different_rank_graphs(
    evidence, tmp_path
):
    arm, run = evidence
    item = arm["hlo"][0]
    text = Path(item["path"]).read_text().replace(" multiply(", " add(")
    path = tmp_path / "different-rank.hlo"
    path.write_text(text)
    audit = qualification.audit_hlo(text, "two_shot", item["chain_length"])
    assert audit["ok"]
    item.update(
        audit, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="graphs differ"):
        qualification.checked_arm(arm, run, (2, 3))


def test_long_chain_retains_complete_transitive_dependency_graph():
    audit = qualification.audit_hlo(_hlo("two_shot", 60), "two_shot", 60)
    assert audit["ok"]
    assert audit["collective_dag"][-1]["predecessors"] == list(range(59))
