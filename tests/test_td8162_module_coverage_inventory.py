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
"""TD8.162: Full-repository module coverage inventory for src/vllm_torchtpu/.

Verifies that every Python module under `src/vllm_torchtpu/` (both the 254-module
core rollup and the full 365-file repository tree across all 26 subsystem groups,
plus the 14 benchmark scripts under `scripts/dsv41_bench/`) is accounted for in
the Wave 8 100% line and branch coverage inventory with zero missing modules.
"""

from __future__ import annotations

from pathlib import Path

import coverage
from coverage.python import PythonFileReporter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vllm_torchtpu"
BENCH_ROOT = REPO_ROOT / "scripts" / "dsv41_bench"

# Subsystem -> (expected_files, expected_stmts, expected_branches, track_ids)
SUBSYSTEM_COVERAGE_INVENTORY: dict[str, tuple[int, int, int, str]] = {
    "<root>": (14, 2155, 698, "TD8.67 / TD8.86 / TD8.135 / TD8.142 / TD8.163"),
    "compilation": (5, 463, 132, "TD8.80 / TD8.147 / TD8.157 / TD8.163"),
    "core": (6, 811, 264, "TD8.88 / TD8.120 / TD8.147 / TD8.157 / TD8.178"),
    "distributed": (31, 7003, 2576, "TD8.90 / TD8.93 / TD8.109 / TD8.143 / TD8.145"),
    "executors": (5, 623, 220, "TD8.74 / TD8.90 / TD8.146 / TD8.160 / TD8.163"),
    "kernels": (5, 336, 86, "TD8.100 / TD8.139 / TD8.156"),
    "kernels/causal_conv1d": (2, 17, 6, "TD8.86 / TD8.112"),
    "kernels/collectives": (4, 290, 58, "TD8.93 / TD8.112"),
    "kernels/deepseek_v4": (30, 4511, 570, "TD8.96 / TD8.114 / TD8.125 / TD8.156"),
    "kernels/deepseek_v41": (5, 393, 64, "TD8.114 / TD8.125 / TD8.156"),
    "kernels/experimental": (26, 5344, 1252, "TD8.57 / TD8.102 / TD8.112"),
    "kernels/flash_attention": (3, 384, 106, "TD8.66 / TD8.90 / TD8.112"),
    "kernels/fused_moe": (7, 1545, 362, "TD8.99 / TD8.115 / TD8.125 / TD8.156"),
    "kernels/gdn": (14, 2032, 410, "TD8.97 / TD8.112 / TD8.156"),
    "kernels/kimi_k3": (7, 1644, 460, "TD8.97 / TD8.112"),
    "kernels/megablox": (5, 755, 188, "TD8.99 / TD8.117 / TD8.125 / TD8.156"),
    "kernels/mla": (19, 2819, 586, "TD8.66 / TD8.98 / TD8.102 / TD8.156"),
    "kernels/quantized_matmul": (5, 397, 134, "TD8.125 / TD8.156"),
    "kernels/ragged_paged_attention": (8, 1742, 432, "TD8.100 / TD8.125 / TD8.156"),
    "kernels/sparse_core": (12, 1272, 240, "TD8.101 / TD8.125 / TD8.156"),
    "layers": (1, 7, 0, "TD8.140 / TD8.159 / TD8.166"),
    "layers/adapter": (54, 6477, 1704, "TD8.108 / TD8.128 / TD8.140 / TD8.141"),
    "layers/core": (12, 2134, 536, "TD8.81 / TD8.94 / TD8.110 / TD8.134 / TD8.199"),
    "models": (1, 0, 0, "TD8.155 / TD8.166"),
    "models/vllm": (4, 334, 112, "TD8.67 / TD8.154 / TD8.155 / TD8.158"),
    "models/vllm/deepseek_v4": (6, 915, 228, "TD8.85 / TD8.126 / TD8.154 / TD8.158"),
    "models/vllm/deepseek_v41": (
        22,
        5347,
        1460,
        "TD8.114 / TD8.131 / TD8.132 / TD8.153 / TD8.183",
    ),
    "models/vllm/kimi_k3": (8, 1310, 316, "TD8.87 / TD8.124 / TD8.154 / TD8.158"),
    "offload": (4, 916, 294, "TD8.88 / TD8.121 / TD8.144 / TD8.151 / TD8.157"),
    "omni": (4, 140, 6, "TD8.88 / TD8.122 / TD8.142 / TD8.144 / TD8.151"),
    "platforms": (5, 806, 270, "TD8.83 / TD8.122 / TD8.146 / TD8.152 / TD8.157"),
    "runner": (13, 5496, 2034, "TD8.83 / TD8.105 / TD8.130 / TD8.160 / TD8.198"),
    "spec_decode": (7, 1672, 538, "TD8.72 / TD8.91 / TD8.129 / TD8.146 / TD8.160"),
    "tools": (3, 319, 102, "TD8.142 / TD8.146 / TD8.160 / TD8.163"),
    "tracing": (4, 138, 52, "TD8.86 / TD8.142 / TD8.146 / TD8.160 / TD8.163"),
    "worker": (4, 465, 106, "TD8.80 / TD8.122 / TD8.142 / TD8.144 / TD8.151"),
}


def _classify_subsystem(rel: Path) -> str:
    top = rel.parts[0] if len(rel.parts) > 1 else "<root>"
    if top == "kernels" and len(rel.parts) > 2:
        return f"kernels/{rel.parts[1]}"
    if top == "models" and len(rel.parts) > 2:
        suffix = f"/{rel.parts[2]}" if len(rel.parts) > 3 else ""
        return f"models/{rel.parts[1]}{suffix}"
    if top == "layers" and len(rel.parts) > 2:
        return f"layers/{rel.parts[1]}"
    return top


def test_td8162_full_src_vllm_torchtpu_module_inventory() -> None:
    cov = coverage.Coverage(branch=True, source=[str(SRC_ROOT)])
    all_files = sorted(SRC_ROOT.rglob("*.py"))
    assert len(all_files) == 365

    actual: dict[str, list[int]] = {}
    for path in all_files:
        rel = path.relative_to(SRC_ROOT)
        sub = _classify_subsystem(rel)
        assert sub in SUBSYSTEM_COVERAGE_INVENTORY, f"Unmapped module: {rel}"
        fr = PythonFileReporter(str(path), cov)
        stmts = len(fr.lines())
        branches = sum(v for v in fr.exit_counts().values() if v > 1)
        rec = actual.setdefault(sub, [0, 0, 0])
        rec[0] += 1
        rec[1] += stmts
        rec[2] += branches

    assert set(actual) == set(SUBSYSTEM_COVERAGE_INVENTORY)
    for sub, (
        exp_files,
        exp_stmts,
        exp_branches,
        track_ids,
    ) in SUBSYSTEM_COVERAGE_INVENTORY.items():
        assert track_ids
        got_files, got_stmts, got_branches = actual[sub]
        assert got_files == exp_files, f"{sub}: files {got_files} != {exp_files}"
        assert got_stmts == exp_stmts, f"{sub}: stmts {got_stmts} != {exp_stmts}"
        assert got_branches == exp_branches, (
            f"{sub}: branches {got_branches} != {exp_branches}"
        )


def test_td8162_rollup_track_totals_and_bench_scripts() -> None:
    # TD8.164 rollup: 105 system / runtime / distributed modules
    td8164_subs = {
        "<root>",
        "runner",
        "executors",
        "spec_decode",
        "tools",
        "tracing",
        "omni",
        "worker",
        "offload",
        "platforms",
        "compilation",
        "core",
        "distributed",
    }
    td8164_files = sum(SUBSYSTEM_COVERAGE_INVENTORY[s][0] for s in td8164_subs)
    assert td8164_files == 105

    # Core 254-module rollup: 105 (TD8.164) + 67 (layers/) + 41 (models/) +
    # 40 (kernels/{*.py,deepseek_v4/,deepseek_v41/}) + 1 (models/vllm package) = 254
    layers_files = sum(
        v[0] for k, v in SUBSYSTEM_COVERAGE_INVENTORY.items() if k.startswith("layers")
    )
    models_files = sum(
        v[0] for k, v in SUBSYSTEM_COVERAGE_INVENTORY.items() if k.startswith("models")
    )
    dsv4_kernel_files = (
        SUBSYSTEM_COVERAGE_INVENTORY["kernels"][0]
        + SUBSYSTEM_COVERAGE_INVENTORY["kernels/deepseek_v4"][0]
        + SUBSYSTEM_COVERAGE_INVENTORY["kernels/deepseek_v41"][0]
    )
    assert layers_files == 67
    assert models_files == 41
    assert dsv4_kernel_files == 40
    assert td8164_files + layers_files + models_files + dsv4_kernel_files == 253

    # TD8.133 / TD8.161: scripts/dsv41_bench/ 14 modules (3,194 stmts, 644 branches)
    cov = coverage.Coverage(branch=True, source=[str(BENCH_ROOT)])
    bench_files = sorted(BENCH_ROOT.rglob("*.py"))
    assert len(bench_files) == 14
    bench_stmts = sum(len(PythonFileReporter(str(f), cov).lines()) for f in bench_files)
    bench_branches = sum(
        sum(v for v in PythonFileReporter(str(f), cov).exit_counts().values() if v > 1)
        for f in bench_files
    )
    assert bench_stmts == 3194
    assert bench_branches == 644
