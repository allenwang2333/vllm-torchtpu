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
"""End-to-end numeric parity for the vision tower, off-device.

``test_glm5_next_vision.py`` pins the SwiGLU clamp in isolation. This pins the
*whole* tower: build it, load the tiny fixture's ``model.visual.*`` weights
through the production stacked-param loader, run it, and diff every stage
against transformers' ``Glm5NextVisionModel``. It is the CPU rehearsal of
exactly what ``scripts/glm5_next/vision_probe.py`` runs on a TPU, so a
regression is caught in ~40 seconds instead of a queue wait.

Run in a subprocess, not in-process, deliberately. The probe installs a fake
``torch_tpu`` and a fake ``torchvision`` into ``sys.modules`` and registers a
``_C::silu_and_mul`` op, and it initialises a gloo process group -- all
per-process, none of it undoable. Doing that inside the pytest interpreter
would leak into every test that runs after it.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts" / "glm5_next"
#: The tiny random fixture from ``scripts/glm5_next/build_tiny_fixture.py``.
_FIXTURE = pathlib.Path(
    os.environ.get("GLM5_NEXT_TINY",
                   pathlib.Path.home() / "data" / "glm5_next_tiny"))


def _skip_reasons() -> str | None:
    if not (_FIXTURE / "config.json").exists():
        return f"no glm5_next fixture at {_FIXTURE}"
    try:
        import transformers.models.glm5_next  # noqa: F401
    except ImportError:
        return "transformers has no glm5_next"
    try:
        import torch_tpu
    except ImportError:
        return None
    # `_is_cpu_stub` is set by `tests/stubs/torch_tpu.py`, which conftest
    # installs under VLLM_TORCHTPU_CPU_STUB=1 -- that is still a CPU box and
    # this test should run. Only a *real* torch_tpu means the tower belongs on the
    # device, where the CPU stubs this test relies on would shadow the runtime.
    if getattr(torch_tpu, "_is_cpu_stub", False):
        return None
    return "real torch_tpu present; run vision_probe.py --device tpu instead"


def _run(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # The worktree ahead of anything installed, so the tower under test is the
    # one in this checkout.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "src"),
         str(_REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("VLLM_TARGET_DEVICE", "cpu")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # A distinct port from the probe's default: pytest may be sharing the box
    # with a manual run.
    env.setdefault("MASTER_PORT", "12459")
    return subprocess.run([sys.executable, *args],
                          cwd=cwd,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=900)


@pytest.fixture(scope="module")
def vision_oracle(tmp_path_factory):
    reason = _skip_reasons()
    if reason:
        pytest.skip(reason)
    out = tmp_path_factory.mktemp("glm5_next_vision") / "vision_oracle.npz"
    result = _run([
        str(_SCRIPTS / "dump_vision_reference.py"),
        "--model",
        str(_FIXTURE),
        "--out",
        str(out),
    ],
                  cwd=_REPO)
    assert result.returncode == 0, (
        f"oracle dump failed:\n{result.stdout}\n{result.stderr}")
    return out


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_tower_matches_transformers(vision_oracle, dtype):
    """Every stage of the ported tower tracks the fp32 HF reference.

    fp32 and bf16 are both run because they fail differently: fp32 lands at
    ~1e-6 and so catches a *small* systematic error (a wrong epsilon, a
    transposed weight shard) that bf16's ~1e-2 noise floor would hide, while
    bf16 is the configuration that actually ships.
    """
    result = _run([
        str(_SCRIPTS / "vision_probe.py"),
        "--model",
        str(_FIXTURE),
        "--vision-oracle",
        str(vision_oracle),
        "--device",
        "cpu",
        "--cpu-stub",
        "--dtypes",
        dtype,
        "--dump",
        str(vision_oracle.parent / f"device_{dtype}.npz"),
    ],
                  cwd=_REPO)
    assert "RESULT: PASS" in result.stdout, (
        f"vision_probe failed for {dtype}:\n{result.stdout}\n{result.stderr}")
    assert result.returncode == 0


def test_rope_tables_agree_exactly_in_fp32(vision_oracle):
    """The flagged risk, called out on its own so a failure is unambiguous.

    transformers rotates halves of a ``head_dim``-wide ``cat(emb, emb)``
    table; vLLM gathers a ``head_dim/2``-wide one out of
    ``get_rope(partial_rotary_factor=0.5)``. Both shapes are right either way,
    so nothing but the numbers can tell them apart -- and in fp32 they must
    agree to the last bit, not merely be close.
    """
    result = _run([
        str(_SCRIPTS / "vision_probe.py"),
        "--model",
        str(_FIXTURE),
        "--vision-oracle",
        str(vision_oracle),
        "--device",
        "cpu",
        "--cpu-stub",
        "--dtypes",
        "float32",
        "--dump",
        str(vision_oracle.parent / "device_rope.npz"),
    ],
                  cwd=_REPO)
    line = next((ln for ln in result.stdout.splitlines()
                 if ln.startswith("rope cos/sin")), None)
    assert line is not None, f"probe printed no rope row:\n{result.stdout}"
    assert "position_ids" in result.stdout
    assert "identical=True" in result.stdout, result.stdout
    # "rope cos/sin  max|diff| <cos> / <sin> (bar ...)"
    cos_diff = float(line.split("max|diff|")[1].split("/")[0])
    assert cos_diff == 0.0, line


def _load_dump_module():
    """Import ``dump_vision_reference`` without importing torch_tpu or vLLM.

    The grid helpers this exercises are pure, but the oracle imports
    ``transformers.models.glm5_next`` at module scope, so on a box without the
    transformers overlay the import is what fails -- not the thing under test.
    Skip rather than error, so the suite stays honest about which environment
    it is in; every other test in this file already routes through
    ``_skip_reasons``.
    """
    spec = importlib.util.spec_from_file_location(
        "_glm5_dump_vision_reference", _SCRIPTS / "dump_vision_reference.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"cannot import dump_vision_reference: {exc}")
    return module


def test_grid_must_be_a_multiple_of_the_spatial_merge():
    """The oracle's own guard, because the failure it prevents is silent.

    ``downsample`` reshapes the token axis to ``(-1, m, m, C)``. For a grid
    whose ``h`` or ``w`` is odd, that view still *succeeds* whenever
    ``t*h*w`` happens to be divisible by ``m*m`` -- it just merges patches
    from unrelated rows. A probe built on such a grid would report a
    divergence that is the fixture's fault, not the port's.
    """
    module = _load_dump_module()
    assert module.parse_grid("1,4,4;2,2,6") == [[1, 4, 4], [2, 2, 6]]

    module.validate_grid([[1, 4, 4], [1, 2, 6]], 2)  # must not raise
    # 3*5 = 15 tokens per frame, and 15 % 4 != 0, so this one would also blow
    # up on the view; use a grid that reshapes cleanly but regroups wrongly.
    with pytest.raises(ValueError, match="spatial_merge_size"):
        module.validate_grid([[1, 2, 2], [1, 1, 4]], 2)
