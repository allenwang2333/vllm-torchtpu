import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def _declared_entry_points(group: str) -> dict[str, str]:
    """``{name: "module:attr"}`` for one entry-point group in pyproject."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject.is_file(), f"expected a pyproject at {pyproject}"
    table = tomllib.loads(pyproject.read_text())["project"]["entry-points"]
    return table.get(group, {})


def test_plugin_entry_points_are_declared():
    """Guard against the table being emptied or a plugin renamed wholesale."""
    assert _declared_entry_points("vllm.platform_plugins") == {
        "torchtpu": "vllm_torchtpu.platforms:register_tpu_platform"
    }
    assert {"torchtpu_layers", "torchtpu_models"
            } <= set(_declared_entry_points("vllm.general_plugins"))


@pytest.mark.parametrize(
    ("name", "target"),
    sorted(_declared_entry_points("vllm.general_plugins").items()),
    ids=lambda value: str(value))
def test_general_plugin_entry_point_target_resolves(name: str, target: str):
    """Every declared general-plugin target must import and be callable.

    vLLM resolves these at startup, long after any test that imports the
    implementation module directly. A target whose attribute lives somewhere
    else -- because the function moved, or because the package never re-exported
    it -- passes every other test in the suite and then fails serving with
    ``AttributeError: module '...' has no attribute '...'``.

    Only the general plugins are checked. The platform plugin is loaded while
    vLLM is still resolving its platform, so importing
    ``vllm_torchtpu.platforms`` on its own -- in a bare interpreter or under
    pytest -- circles back through a half-initialised ``vllm.platforms`` and
    raises. Its name is pinned by the test above instead.
    """
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute, None)
    assert resolved is not None, (
        f"entry point {name} = {target!r} does not resolve: "
        f"{module_name} has no attribute {attribute!r}")
    assert callable(
        resolved), f"entry point {name} = {target!r} is not callable"


def test_package_imports_before_vllm():
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_torchtpu; import vllm"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_submodule_imports_before_vllm_initialization_completes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_torchtpu.layers.vllm.quantization.fp8; import vllm",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
