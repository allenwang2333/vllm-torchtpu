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
"""Import helpers shared by the variant-A drivers.

`variants/narrow-record/` is not an importable package name (the hyphen), and
adding an `__init__.py` would not fix that. Both modules in it are therefore
loaded from their file path.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# .../vllm-torchtpu -- the directory `kernel_work` lives in.
REPO = HERE.rsplit("/kernel_work/", 1)[0]
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load(name):
    """Load `<name>.py` from this directory as a module named `name`."""
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
