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
"""Pin the vLLM KV-cache spec API that the TPU deployment actually imports.

The reason this is a test and not a comment is that **the CPU rung imports a
different vLLM from the pod**:

* the pod imports the image's ``site-packages`` vLLM, which is the
  ``pyproject.toml`` pin, ``v0.27.0``;
* a local run usually puts a vLLM *source checkout* on ``PYTHONPATH``, and
  that checkout is normally much newer.

Those two have *incompatibly* different KV-cache spec APIs -- ``v0.27.0`` has
``MLAAttentionSpec.compress_ratio`` and ``.storage_block_size``, while a recent
``main`` replaced both with ``tokens_per_state``/``num_states`` and added
``CircularBufferSpec`` and ``SlidingWindowSpec.extra_retained_tokens``. So a
spec that constructs fine on a developer box can be an ``AttributeError`` or a
``TypeError`` inside the pod, and vice versa, with nothing on the CPU rung to
say so.

Every assertion here is written to hold under *both* imports. What it refuses
is a **third**, unrecorded shape, and any use in ``src/`` of a name the pinned
vLLM does not have. See ``docs/vllm-kv-cache-api-pin.md`` for the recorded
shapes and the rule that follows from them.
"""

import ast
import dataclasses
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The git ref in ``pyproject.toml``'s vllm dependency. Confirmed against the
#: pod's own logs, which print a bare ``Initializing a V1 LLM engine
#: (v0.27.0)`` -- a build from any later commit would render as
#: ``0.27.1.devN+g<sha>`` -- and whose ``model.py:645`` / ``model.py:1883`` log
#: sites match this tag's source lines exactly.
PINNED_VLLM_REF = "v0.27.0"


def _api_shape() -> dict[str, bool]:
    """Fingerprint the *imported* vLLM's KV-cache spec surface."""
    from vllm.v1 import kv_cache_interface as kvci

    mla_fields = {f.name for f in dataclasses.fields(kvci.MLAAttentionSpec)}
    swa_fields = {f.name for f in dataclasses.fields(kvci.SlidingWindowSpec)}
    return {
        "MLAAttentionSpec.compress_ratio":
        "compress_ratio" in mla_fields,
        "MLAAttentionSpec.storage_block_size":
        isinstance(getattr(kvci.MLAAttentionSpec, "storage_block_size", None),
                   property),
        "MLAAttentionSpec.tokens_per_state":
        "tokens_per_state" in mla_fields,
        "SlidingWindowSpec.extra_retained_tokens":
        "extra_retained_tokens" in swa_fields,
        "CircularBufferSpec":
        hasattr(kvci, "CircularBufferSpec"),
    }


#: What ``v0.27.0`` -- the pin, and therefore the pod -- declares. This is the
#: shape every cache spec in this repo is written against.
SHAPE_PINNED_V0_27_0 = {
    "MLAAttentionSpec.compress_ratio": True,
    "MLAAttentionSpec.storage_block_size": True,
    "MLAAttentionSpec.tokens_per_state": False,
    "SlidingWindowSpec.extra_retained_tokens": False,
    "CircularBufferSpec": False,
}

#: What the developer checkout declared when this was written
#: (vLLM ``main`` @ ``60857baa53``). Recorded, not endorsed: nothing in this
#: repo may depend on it.
SHAPE_DEV_MAIN_60857BAA53 = {
    "MLAAttentionSpec.compress_ratio": False,
    "MLAAttentionSpec.storage_block_size": False,
    "MLAAttentionSpec.tokens_per_state": True,
    "SlidingWindowSpec.extra_retained_tokens": True,
    "CircularBufferSpec": True,
}

#: Names that exist only *after* the pin. Using any of them in ``src/`` builds
#: something that cannot run in the pod -- and that will not be caught on the
#: CPU rung, because the dev checkout does have them. ``CircularBufferSpec`` is
#: on this list even though it is the natural way to express a fixed-depth
#: per-request ring; see ``docs/vllm-kv-cache-api-pin.md``.
NAMES_ABSENT_FROM_THE_PINNED_VLLM = (
    "CircularBufferSpec",
    "extra_retained_tokens",
    "tokens_per_state",
)


def test_pyproject_still_pins_the_vllm_this_repo_was_written_against():
    """The pin *is* the deployment: the image installs it from pyproject."""
    text = (_REPO_ROOT / "pyproject.toml").read_text()
    matches = re.findall(
        r'"vllm @ git\+https://github\.com/vllm-project/vllm\.git@([^"]+)"',
        text)
    assert matches, (
        "no `vllm @ git+...@<ref>` dependency found in pyproject.toml; the "
        "deployment's vLLM can no longer be identified from the repo, which "
        "is the thing docs/vllm-kv-cache-api-pin.md exists to keep true")
    assert matches == [
        PINNED_VLLM_REF
    ], (f"vLLM pin moved from {PINNED_VLLM_REF} to {matches}. Every KV-cache "
        "spec in this repo (deepseek_v4_indexer.py:113, "
        "deepseek_v4/attention.py:244, tpu_runner.py:5908/:5914) is written "
        "against v0.27.0's `compress_ratio`/`storage_block_size` API, which "
        "vLLM main replaced with `tokens_per_state`/`num_states`. Re-derive "
        "docs/vllm-kv-cache-api-pin.md before changing this constant.")


def test_imported_vllm_kv_cache_api_is_one_of_the_two_recorded_shapes():
    """Fail on an unrecorded API shape rather than on the wrong one.

    Passing here does *not* mean the imported vLLM is the deployment's; it
    means it is one of the two we have characterised. Which one is printed.
    """
    shape = _api_shape()
    known = {
        "pinned v0.27.0 (what the pod imports)": SHAPE_PINNED_V0_27_0,
        "dev checkout, vLLM main @ 60857baa53": SHAPE_DEV_MAIN_60857BAA53,
    }
    for name, recorded in known.items():
        if shape == recorded:
            print(f"imported vLLM KV-cache API shape: {name}")
            return
    diffs = {
        label: {
            k: (v, shape[k])
            for k, v in recorded.items() if shape[k] != v
        }
        for label, recorded in known.items()
    }
    pytest.fail(
        "The imported vLLM's KV-cache spec API matches neither recorded "
        f"shape.\nobserved: {shape}\ndiff vs each recorded shape "
        f"(recorded, observed): {diffs}\n"
        "Update docs/vllm-kv-cache-api-pin.md and this test together -- an "
        "unrecognised shape means nobody has checked whether the repo's cache "
        "specs still construct inside the pod.")


def test_repo_cache_specs_construct_against_the_pinned_api():
    """The compress-ratio API the repo calls really behaves as assumed."""
    from vllm.v1 import kv_cache_interface as kvci

    if _api_shape() != SHAPE_PINNED_V0_27_0:
        pytest.skip(
            "imported vLLM is not the pinned v0.27.0 (this is the expected "
            "state on the CPU rung, where PYTHONPATH points at the developer "
            "checkout). The pod-side behaviour asserted here cannot be "
            "reproduced locally; see docs/vllm-kv-cache-api-pin.md.")

    import torch

    # Exactly the DeepSeek-V4 lightning-indexer spec at
    # layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:106-115, whose
    # production block_size is the MLA page size (1024) per
    # platforms/tpu_platform.py:1150-1157.
    spec = kvci.MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        compress_ratio=4,
        alignment=None,
    )
    # tpu_runner.py:5908 reads this to size the pooled page.
    assert spec.storage_block_size == 256
    # ... and tpu_runner.py:5914 branches on this.
    assert spec.compress_ratio == 4


@pytest.mark.parametrize("name", NAMES_ABSENT_FROM_THE_PINNED_VLLM)
def test_src_does_not_depend_on_apis_absent_from_the_pinned_vllm(name):
    """Catch a pod-only ImportError/AttributeError on the CPU rung.

    The dev checkout *has* these names, so nothing else on this rung -- not
    imports, not construction, not the model tree -- will notice that the
    deployment does not.

    Scanning *code* rather than text, deliberately. A raw substring scan also
    fires on the comment explaining why the name is avoided -- and the whole
    point is to be able to write that explanation down next to the spec.
    The cheapest repair for a tripwire that punishes documentation is to
    delete the documentation, so tokenize and skip comments and strings.
    """
    hits = []
    for path in sorted((_REPO_ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - not our problem to police
            continue
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            ident = None
            if isinstance(node, ast.Name):
                ident = node.id
            elif isinstance(node, ast.Attribute):
                ident = node.attr
            elif isinstance(node, ast.keyword):
                ident = node.arg
            elif isinstance(node, ast.alias):
                ident = node.name.rsplit(".", 1)[-1]
            if ident == name:
                hits.append(f"{rel}:{getattr(node, 'lineno', '?')}: {name}")
    assert not hits, (
        f"`{name}` does not exist in vLLM {PINNED_VLLM_REF}, which is the "
        "vLLM the TPU pod imports -- this code would raise inside the pod "
        "while passing every check on the CPU rung, because the developer "
        "checkout on PYTHONPATH does have it. See "
        "docs/vllm-kv-cache-api-pin.md.\n" + "\n".join(hits))
