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
"""Load a patched copy of `streamindex_topk.py` as a fresh module.

Why a source patch rather than a monkeypatch: the thing option C changes,
`out_dtype`, is a local variable inside `_common_path`, which is a closure
inside the jitted `streamindex_topk`. There is no attribute to rebind. And why
a patch rather than a checked-in 866-line copy: a copy silently stops tracking
the incumbent, and then the A/B is against a stale baseline rather than
against HEAD. This applies the edit to whatever source is installed at run
time and asserts each anchor matched exactly once, so a drift in the incumbent
is a loud failure instead of a quiet one.

The same substitution list is what `half_width_key.diff` records.
"""

import importlib.util
import inspect
import sys
import types


def load_patched(base_module_name: str, new_name: str, subs):
    """Return a new module: `base_module_name`'s source with `subs` applied.

    `subs` is a list of (old, new) string pairs. Each `old` must occur exactly
    once in the source; anything else raises.
    """
    spec = importlib.util.find_spec(base_module_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"cannot locate source of {base_module_name}")
    with open(spec.origin) as f:
        src = f.read()

    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise AssertionError(
                f"patch anchor matched {n} times, expected exactly 1, in "
                f"{spec.origin}:\n  {old!r}\n"
                "The incumbent has drifted; re-derive the diff rather than "
                "loosening the anchor.")
        src = src.replace(old, new, 1)

    mod = types.ModuleType(new_name)
    mod.__file__ = spec.origin + f" [patched:{new_name}]"
    mod.__package__ = base_module_name.rsplit(".", 1)[0]
    mod.__loader__ = None
    sys.modules[new_name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def source_of(mod):
    return inspect.getsource(mod) if hasattr(mod, "__file__") else ""
