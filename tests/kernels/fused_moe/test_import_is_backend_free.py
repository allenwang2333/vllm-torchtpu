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
"""Importing a fused-MoE test module must not initialise a JAX backend.

``jax.default_backend()`` and friends do not *report* a backend, they bring one
up. Called at module scope -- most temptingly inside a ``pytest.mark.skipif``
-- that happens during **collection**, in the parent process, before pytest has
selected a single test:

* a CPU gate that only wants three unrelated files pays for it, and pays again
  in the parent even when the module is then deselected;
* a module later in the same session that needs a different device count finds
  one already up, and has to ``clear_backends()`` to get its own;
* on a chip, collection claims the device before anything has been selected.

The fix is always the same and always cheap: ask inside a fixture or the test.
``test_fused_ep_v2_device.py`` did it in a ``skipif`` and this is the guard that
keeps it fixed.

**Scope, and why it is not the whole tree.** Every module below is one this
branch adds. A repo-wide version of this test would pass today -- an AST scan
of ``tests/`` finds **zero** module-scope backend calls at ``gh/main``
``08ff5d019``, zero at ``86bb551c8``, and one on this branch before the fix
above; the six ``jax.default_backend()`` call sites already on main are all
inside functions. Widening ``_OUR_MODULES`` to ``("tests",)`` is therefore a
one-token change, but turning a general repo rule on from inside a MoE PR is
the PR's business to propose, not to assume.
"""

import ast
import pathlib

import pytest

# Backend entry points: each of these initialises the runtime as a side effect.
_BACKEND_CALLS = {
    ("jax", "default_backend"),
    ("jax", "devices"),
    ("jax", "local_devices"),
    ("jax", "device_count"),
    ("jax", "local_device_count"),
    ("jax", "device_put"),
}

_TESTS_ROOT = pathlib.Path(__file__).resolve().parents[2]  # tests/

# The fused-MoE test modules this branch adds, relative to `tests/`.
_OUR_MODULES = (
    "kernels/fused_moe/*.py",
    "layers/common/test_fused_moe_ep_numerics.py",
    "layers/vllm/test_fused_moe_ep*.py",
    "runner/test_capture_ladder_engagement.py",
    "compilation/test_moe_census.py",
    "distributed/test_ep_mesh_device_order.py",
)


def _module_scope_calls(source: str):
    """Backend calls reachable when the module body runs, with line numbers.

    Walks only statements that execute at import: a call inside a ``def`` or a
    ``class`` body's methods is exactly what this rule asks for, so those
    subtrees are skipped rather than searched. Decorator expressions are *not*
    skipped -- ``@pytest.mark.skipif(jax.default_backend() != "tpu")`` is the
    spelling this exists to catch.
    """
    found = []
    stack = list(ast.parse(source).body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The decorators run at import even though the body does not.
            stack.extend(node.decorator_list)
            continue
        if isinstance(node, ast.ClassDef):
            stack.extend(node.decorator_list)
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and (func.value.id, func.attr) in _BACKEND_CALLS):
                found.append((sub.lineno, f"{func.value.id}.{func.attr}()"))
    return sorted(found)


def _our_test_modules():
    paths = {
        path
        for pattern in _OUR_MODULES
        for path in _TESTS_ROOT.glob(pattern)
        if path.name != pathlib.Path(__file__).name
    }
    assert paths, f"no modules matched {_OUR_MODULES} under {_TESTS_ROOT}"
    return sorted(paths)


@pytest.mark.parametrize("path", _our_test_modules(), ids=lambda p: p.name)
def test_no_backend_is_initialised_at_import(path):
    offenders = _module_scope_calls(path.read_text())
    assert not offenders, (
        f"{path.relative_to(_TESTS_ROOT.parent)} calls " +
        ", ".join(f"{call} at line {line}" for line, call in offenders) +
        " while the module body runs, so importing it -- which pytest does to "
        "every selected file at collection, before running anything -- brings "
        "up a JAX backend. Move it into a fixture or the test body; see "
        "test_fused_ep_v2_device.py::require_tpu_backend")


def test_the_scan_sees_the_pattern_it_is_written_for():
    """Negative control: the shape this rule exists to catch, and two it must
    not catch (the same call inside a function, and inside a fixture body)."""
    caught = _module_scope_calls('''
import jax, pytest
pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="x")
''')
    assert [call for _, call in caught] == ["jax.default_backend()"]

    assert _module_scope_calls('''
import jax, pytest

@pytest.fixture(autouse=True)
def require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("x")

def test_thing():
    assert jax.devices()
''') == []
