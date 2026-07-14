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

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_SCRIPT = (REPO_ROOT / "scripts" / "vllm" / "integration" /
                "run_qwen35_397b_pcp8_compile_repro.sh")
MODEL_CONFIG = (REPO_ROOT / "tests" / "fixtures" / "models" /
                "qwen3_5_397b_a17b_fp8" / "config.json")
WORKFLOW = (REPO_ROOT / ".github" / "workflows" /
            "qwen35-397b-pcp8-compile-repro.yml")
PRESUBMIT_WORKFLOW = (REPO_ROOT / ".github" / "workflows" /
                      "presubmit-tpu.yml")


def _compilation_config(mode: str) -> dict:
    env = os.environ.copy()
    env.update({
        "PRINT_CONFIG_ONLY": "1",
        "PYTHON_BIN": sys.executable,
    })
    result = subprocess.run(
        ["bash", str(REPRO_SCRIPT), mode],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_checked_in_fixture_is_qwen35_397b_fp8_gdn_model():
    from transformers import AutoConfig

    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    loaded_config = AutoConfig.from_pretrained(
        MODEL_CONFIG.parent,
        local_files_only=True,
        trust_remote_code=True,
    )

    assert config["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert config["model_type"] == "qwen3_5_moe"
    assert config["quantization_config"]["quant_method"] == "fp8"
    assert config["text_config"]["num_hidden_layers"] == 60
    assert config["text_config"]["num_experts"] == 512
    assert config["text_config"]["layer_types"].count("linear_attention") == 45
    assert loaded_config.text_config.num_hidden_layers == 60
    assert loaded_config.text_config.num_experts == 512
    assert {path.name
            for path in MODEL_CONFIG.parent.iterdir()
            } == {"README.md", "config.json"}


def test_split_mode_lists_every_pcp_gdn_op():
    config = _compilation_config("split")
    expected_ops = [
        "pallas::gdn_attention_pcp_"
        f"language_model_model_layers_{layer}_linear_attn"
        for layer in range(60) if layer % 4 != 3
    ]

    assert config["splitting_ops"] == expected_ops
    assert config["compile_sizes"] == [512]
    assert config["use_inductor_graph_partition"] is False
    assert config["pass_config"]["fuse_rope_kvcache_cat_mla"] is False


def test_full_mode_uses_same_compiler_config_without_splits():
    split_config = _compilation_config("split")
    full_config = _compilation_config("full")

    assert full_config["splitting_ops"] == []
    split_config["splitting_ops"] = []
    assert full_config == split_config


def test_repro_script_uses_dummy_load_without_eager_or_model_download():
    script = REPRO_SCRIPT.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(REPRO_SCRIPT)], check=True)
    assert "--load-format dummy" in script
    assert "--skip-tokenizer-init" in script
    assert "--prefill-context-parallel-size 8" in script
    assert "--cp-kv-cache-interleave-size 256" in script
    assert "--block-size 256" in script
    assert "--max-num-batched-tokens 4096" in script
    assert "--enforce-eager" not in script
    assert "gsutil" not in workflow
    assert "model_gcs" not in workflow.lower()


def test_workflow_runs_isolated_split_and_full_graph_cases():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "fail-fast: false" in workflow
    assert "mode: split" in workflow
    assert "expected_outcome: healthy" in workflow
    assert "mode: full" in workflow
    assert "expected_outcome: core_halt" in workflow
    assert "qwen35-397b-pcp8-compile-${{ matrix.case.mode }}-logs" in workflow
    assert "HF_HUB_OFFLINE: \"1\"" in workflow
    assert "TRANSFORMERS_OFFLINE: \"1\"" in workflow


def test_presubmit_runs_and_gates_compile_reproduction():
    presubmit = PRESUBMIT_WORKFLOW.read_text(encoding="utf-8")

    assert "run_qwen35_397b_pcp8_compile_repro:" in presubmit
    assert "uses: ./.github/workflows/qwen35-397b-pcp8-compile-repro.yml" in presubmit
    assert "needs.run_qwen35_397b_pcp8_compile_repro.result" in presubmit
    assert "qwen35 397b pcp8 compile reproduction failed" in presubmit
    assert "qwen35 397b pcp8 compile reproduction was cancelled" in presubmit
    assert "run_tests:" not in presubmit
    assert "run_perf_eval:" not in presubmit
    assert "run_qwen35_p4d2_disagg_correctness:" not in presubmit
    assert not (REPO_ROOT / ".github" / "workflows" /
                "qwen35-p4d2-disagg-correctness.yml").exists()
