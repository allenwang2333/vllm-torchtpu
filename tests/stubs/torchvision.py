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
"""An import-only stand-in for ``torchvision``, so the ViT can be built on CPU.

vLLM's ``glm4_1v`` -- which ``glm_ocr`` and therefore this repo's vision tower
sit on top of -- imports ``transformers.models.glm4v.image_processing_glm4v``,
which does ``from torchvision.transforms.v2 import functional as tvF`` at
module scope. That single line makes the whole tower unimportable on a box
without torchvision, which pushes every structural and numerical question
about the ViT to the back of a TPU queue.

Installing real torchvision into the CPU dev venv was tried and is worse: its
compiled ops do not load against this torch build, and a half-imported
torchvision breaks unrelated tests with ``operator torchvision::nms does not
exist``. So this provides the *name* and nothing else.

That is enough because the only uses of ``tvF`` in those modules are
``PILImageResampling | tvF.InterpolationMode | int | None`` annotations, which
are strings and never evaluated. Anything that actually resizes an image will
fail loudly here -- correctly, because a stub cannot resize an image. This is
for the tower's weights and arithmetic, not for preprocessing.

Guarded: if a real torchvision is importable (as on the TPU image), this is a
no-op.
"""

from __future__ import annotations

import enum
import importlib.machinery
import sys
import types


def _unavailable(path: str):
    """A module ``__getattr__`` that raises instead of returning junk.

    Dunders are exempt: ``inspect`` walks ``sys.modules`` asking every module
    for ``__file__`` whenever anything anywhere resolves a stack frame, so a
    stub that raises on those turns into an unrelated crash somewhere else.
    """

    def getattr_(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            f"torchvision is stubbed out on this host; {path}.{name} was "
            "actually called. Image/video preprocessing needs the real "
            "package -- run it on the TPU image instead.")

    return getattr_


def install() -> None:
    try:
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass

    torchvision = types.ModuleType("torchvision")
    torchvision.__version__ = "0.0.0+stub"
    transforms = types.ModuleType("torchvision.transforms")
    v2 = types.ModuleType("torchvision.transforms.v2")
    functional = types.ModuleType("torchvision.transforms.v2.functional")

    # The full member list from the real `torchvision.transforms.v2`, spelled
    # out rather than proxied: `transformers.image_utils` builds a
    # PIL<->torchvision interpolation mapping over all of them at import time
    # and dies on the first one that is absent.
    class InterpolationMode(str, enum.Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    functional.InterpolationMode = InterpolationMode
    functional.__getattr__ = _unavailable(
        "torchvision.transforms.v2.functional")
    v2.functional = functional
    v2.InterpolationMode = InterpolationMode

    # The v1 namespace, which `transformers.image_utils` imports directly.
    tf_functional = types.ModuleType("torchvision.transforms.functional")
    tf_functional.InterpolationMode = InterpolationMode
    tf_functional.pil_to_tensor = _decoder(
        "torchvision.transforms.functional.pil_to_tensor")
    tf_functional.__getattr__ = _unavailable(
        "torchvision.transforms.functional")

    transforms.v2 = v2
    transforms.functional = tf_functional
    transforms.InterpolationMode = InterpolationMode
    torchvision.transforms = transforms

    ops = types.ModuleType("torchvision.ops")
    ops.__getattr__ = _unavailable("torchvision.ops")
    torchvision.ops = ops

    # `transformers.image_utils` does a hard `from torchvision.io import
    # ImageReadMode, decode_image` at module scope, and that module is on the
    # path of `transformers.models.auto`, i.e. of importing anything at all.
    io = types.ModuleType("torchvision.io")

    class ImageReadMode:
        UNCHANGED = "unchanged"
        GRAY = "gray"
        RGB = "rgb"

    io.ImageReadMode = ImageReadMode
    io.decode_image = _decoder("torchvision.io.decode_image")
    io.read_image = _decoder("torchvision.io.read_image")
    io.__getattr__ = _unavailable("torchvision.io")
    torchvision.io = io

    modules = {
        "torchvision": torchvision,
        "torchvision.transforms": transforms,
        "torchvision.transforms.functional": tf_functional,
        "torchvision.transforms.v2": v2,
        "torchvision.transforms.v2.functional": functional,
        "torchvision.ops": ops,
        "torchvision.io": io,
    }
    for name, module in modules.items():
        # `transformers.utils.is_torchvision_available` answers with
        # `importlib.util.find_spec`, which raises `ValueError` outright on a
        # module in `sys.modules` whose `__spec__` is None -- so a stub with no
        # spec fails harder than no stub at all. `__path__` is what makes the
        # top-level name a *package*, without which `import torchvision.io`
        # fails even though the entry is already in `sys.modules`.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__file__ = f"<{name} stub>"
        module.__path__ = []
    sys.modules.update(modules)


def _decoder(path: str):

    def fn(*args, **kwargs):
        raise RuntimeError(
            f"torchvision is stubbed out on this host; {path} was actually "
            "called. Image decoding needs the real package.")

    return fn
