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
"""Unit tests for .buildkite/scripts/select_pr_tests.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".buildkite" / "scripts" / "select_pr_tests.py"

spec = importlib.util.spec_from_file_location("select_pr_tests", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
select_pr_tests = importlib.util.module_from_spec(spec)
sys.modules["select_pr_tests"] = select_pr_tests
spec.loader.exec_module(select_pr_tests)


_SAMPLE_TESTS_KUBE_YAML = """\
steps:
  - group: ":kubernetes: Unit Tests on TPU"
    steps:
      - label: ":kubernetes: Unit Tests - General (PR)"
        key: "unit_tests_general_pr"
        <<: *tpu_step
        if: 'build.env("NIGHTLY") != "1"'
        parallelism: 3
        env:
          SHAPE: *shape_single
        plugins:
          - *kube_plugin
          - tests#v1.0.0:
              <<: *tests_plugin_defaults
              test-cmd: "pytest -sv"

      - label: ":kubernetes: Unit Tests - Speculative Decoding (PR)"
        key: "unit_tests_spec_decode_pr"
        <<: *tpu_step
        if: 'build.env("NIGHTLY") != "1"'
        parallelism: 3
"""


class SelectPrTestsTest(unittest.TestCase):
    """Tests diff-aware PR test selection and YAML step filtering."""

    def test_skip_all_ci_for_non_pr_files(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "docs/README.md",
                ".github/workflows/pre-commit.yml",
                "support_matrices/nightly.csv",
                ".buildkite/features/moe.yml",
                ".buildkite/pipeline_dynamic.yml",
                "scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-tp4-ep.sh",
                "scripts/vllm/integration/run_dsv4_offload_correctness.sh",
            ]
        )
        self.assertTrue(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ())

    def test_cpu_only_tooling_skips_integration_and_tpu_unit_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "tools/check_comment_language.py",
                "tests/scripts/test_verify_regression.py",
                "scripts/vllm/benchmarking/verify_regression.py",
            ]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("typecheck_full", plan.skip_steps)
        self.assertNotIn("unit_tests_cpu_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_omni_only_runs_omni_and_cpu_tests(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "src/vllm_torchtpu/omni/runner.py",
                "tests_omni/test_omni_smoke.py",
            ]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(plan.omni_image_skip, "")
        self.assertNotIn("unit_tests_omni_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)

    def test_integration_script_only_runs_matching_step(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["scripts/vllm/integration/run_qwen35_apc_correctness.sh"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration"))
        self.assertTrue(plan.omni_image_skip)
        self.assertNotIn("integration_qwen35_apc_correctness", plan.skip_steps)
        self.assertIn(
            "integration_k3_pruned75_apc_offload_correctness",
            plan.skip_steps,
        )
        self.assertIn("vllm_torchtpu_multihost_presubmit", plan.skip_steps)

    def test_spec_decode_unit_test_only(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["tests/entrypoints/llm/test_speculative_decoding.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertNotIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_spec_decode_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_single_general_unit_test_narrows_pattern_and_parallelism(
        self,
    ) -> None:
        plan = select_pr_tests.select_for_files(["tests/lora/test_lora.py"])
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertEqual(plan.general_test_pattern, "tests/lora/test_lora.py")
        self.assertEqual(plan.general_test_parallelism, 1)
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)

    def test_dsv4_custom_ops_skips_integration_and_spec_decode(self) -> None:
        plan = select_pr_tests.select_for_files(
            [
                "src/vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/"
                "deepseek_v4_indexer.py"
            ]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "tests"))
        self.assertNotIn("unit_tests_general_pr", plan.skip_steps)
        self.assertNotIn("unit_tests_multichip_pr", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_kimi_k3_src_change_runs_only_k3_integration(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/models/vllm/kimi_k3/kimi_k3.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertNotIn(
            "integration_k3_pruned75_apc_offload_correctness",
            plan.skip_steps,
        )
        self.assertIn("integration_qwen35_apc_correctness", plan.skip_steps)
        self.assertIn("unit_tests_spec_decode_pr", plan.skip_steps)
        self.assertIn("unit_tests_omni_pr", plan.skip_steps)

    def test_core_src_change_runs_full_non_omni_suite(self) -> None:
        plan = select_pr_tests.select_for_files(
            ["src/vllm_torchtpu/worker/model_runner.py"]
        )
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(list(plan.skip_steps.keys()), ["unit_tests_omni_pr"])
        self.assertTrue(plan.omni_image_skip)

    def test_dockerfile_change_runs_everything_including_omni(self) -> None:
        plan = select_pr_tests.select_for_files(["docker/Dockerfile"])
        self.assertFalse(plan.skip_all_ci)
        self.assertEqual(plan.lanes, ("verify", "integration", "tests"))
        self.assertEqual(plan.skip_steps, {})
        self.assertEqual(plan.omni_image_skip, "")

    def test_filter_yaml_injects_skip_and_test_pattern(self) -> None:
        yaml_inputs = [_SAMPLE_TESTS_KUBE_YAML]
        tests_kube_path = REPO_ROOT / ".buildkite" / "pipeline_tests_kube.yml"
        if tests_kube_path.exists():
            yaml_inputs.append(tests_kube_path.read_text(encoding="utf-8"))

        plan = select_pr_tests.select_for_files(["tests/lora/test_lora.py"])
        for tests_kube in yaml_inputs:
            filtered = select_pr_tests.filter_yaml_content(
                yaml_text=tests_kube,
                skip_steps=plan.skip_steps,
                general_test_pattern=plan.general_test_pattern,
                general_test_parallelism=plan.general_test_parallelism,
                multichip_test_pattern=plan.multichip_test_pattern,
            )
            self.assertIn(
                '              test-file-pattern: "tests/lora/test_lora.py"\n',
                filtered,
            )
            self.assertIn("        parallelism: 1\n", filtered)
            self.assertIn(
                '        key: "unit_tests_spec_decode_pr"\n        skip: "',
                filtered,
            )
        for reason in plan.skip_steps.values():
            self.assertLessEqual(len(reason), 70)


if __name__ == "__main__":
    unittest.main()
