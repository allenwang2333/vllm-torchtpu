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
"""``register_models()`` must not import GLM-5.3-Flash.

``register_models`` runs on **every** model's boot path, and everything it
registers except the MTP speculative-config patch is registered lazily, by
dotted string. The patch is the one thing that has to be applied eagerly, so
it is the one thing that can drag a model's implementation into every other
model's process -- and Python runs a parent package before any submodule, so
``from ...glm5_next import spec_patch`` would execute
``glm5_next/__init__.py`` and with it the attention, layer, model, MTP,
indexer and kpool-kernel modules, plus the DeepSeek-V4 pieces they import.

The cost is not the import time. It is that an ``ImportError``, a missing
optional dependency or a syntax error anywhere in GLM's tree would become a
**boot failure for Qwen, DeepSeek-V4 and Kimi**. ``tpu_runner.py`` documents
exactly this hazard for its per-layer cache predicate and answers it without
importing anything; this test is the same rule for the other end.

Run out of process, in a clean interpreter, because an in-process check would
pass or fail on whatever an earlier test in the session happened to import.
"""

import json
import os
import subprocess
import sys

#: Prefixes that must not appear in ``sys.modules`` after ``register_models``.
#: ``glm5_next_spec_patch`` is deliberately a sibling module, not a member of
#: the package, so the package prefix carries its trailing dot.
FORBIDDEN_PREFIXES = (
    "vllm_torchtpu.models.vllm.glm5_next.",
    "vllm_torchtpu.models.vllm.deepseek_v4.",
    "vllm_torchtpu.models.vllm.kimi_k3.",
    "vllm_torchtpu.kernels.glm5_kpool",
)

_PROBE = """
import json, sys
from typing import get_args

from vllm_torchtpu.models.vllm import register_models

register_models()

import vllm.config.speculative as spec

forbidden = %(forbidden)r
leaked = sorted(
    name for name in sys.modules
    if name == "vllm_torchtpu.models.vllm.glm5_next"
    or name.startswith(forbidden))
print("RESULT" + json.dumps({
    "leaked": leaked,
    "patched": "glm5_next_mtp" in get_args(spec.MTPModelTypes),
    "spec_patch_imported":
        "vllm_torchtpu.models.vllm.glm5_next_spec_patch" in sys.modules,
}))
"""


def _run_probe():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # No conftest runs in the child, so the torch_tpu stub is not installed.
    # Nothing on this path may need it -- that is part of what is under test.
    env.pop("VLLM_TORCHTPU_CPU_STUB", None)
    probe = _PROBE % {"forbidden": FORBIDDEN_PREFIXES}
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True,
                          text=True,
                          env=env)
    assert proc.returncode == 0, (
        f"register_models() probe failed:\n{proc.stdout}\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise AssertionError(f"probe printed no result:\n{proc.stdout}")


def test_register_models_does_not_import_any_model_implementation():
    result = _run_probe()
    assert result["leaked"] == [], (
        "register_models() imported model implementation modules: "
        f"{result['leaked']}. Every architecture must be registered by dotted "
        "string so vLLM imports it only for the model actually being served; "
        "an eager import puts one model's import errors on every other "
        "model's boot path.")


def test_the_mtp_speculative_patch_is_still_applied():
    """The isolation above must not have been bought by skipping the work."""
    result = _run_probe()
    assert result["spec_patch_imported"], (
        "glm5_next_spec_patch was not imported, so the test above proves "
        "nothing")
    assert result["patched"], (
        "MTPModelTypes does not carry glm5_next_mtp after register_models(); "
        "a speculative run would fail with 'Unsupported speculative method'")
