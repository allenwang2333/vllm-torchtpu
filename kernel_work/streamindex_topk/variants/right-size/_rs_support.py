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
"""Shared plumbing for the Right-Size (Option B) drivers.

Two jobs, and deliberately nothing else:

1.  Load the variant kernel from this folder by path. `variants/right-size` is
    not an importable package name (the hyphen), and it must not become one --
    the variant is not part of the shipped tree.
2.  Adapt an implementation module so the READ-ONLY instruments can drive it.
    `microbench.run_regime` and `test_correctness.run` both take a module
    object and call `impl.streamindex_topk(...)` with a fixed kwargs list. The
    variant needs two extra *static* kwargs that list does not contain, so it
    is bound here with `functools.partial` behind a namespace that quacks like
    a module. No instrument is edited, and both arms therefore go through the
    identical timing and checking code.

The sixth regime is NOT here -- it is in `extra_regimes.py`, committed
separately, because it is an instrument addition and not part of Option B.
"""

import functools
import importlib.util
import os
import types

from kernel_work.streamindex_topk.instruments import shapes as S

_HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_PATH = os.path.join(_HERE, "streamindex_topk_rs.py")

INCUMBENT_MODULE = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"


def load_variant():
    """Import `streamindex_topk_rs.py` from this folder by absolute path."""
    spec = importlib.util.spec_from_file_location("streamindex_topk_rs",
                                                  VARIANT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_incumbent():
    return importlib.import_module(INCUMBENT_MODULE)


def bind(impl, **extra):
    """Return a module-like object whose `streamindex_topk` has `extra` bound.

    `types.SimpleNamespace` is enough: the instruments touch exactly one
    attribute. Binding rather than editing the instrument is what keeps the
    yardstick identical across arms.
    """
    return types.SimpleNamespace(
        streamindex_topk=functools.partial(impl.streamindex_topk, **extra),
        __name__=getattr(impl, "__name__", "impl"))


# The sixth regime and the two masked-store correctness shapes live in
# `extra_regimes.py`, which is committed separately from the Option B diff
# because it is an instrument addition rather than part of the variant.
from extra_regimes import (CONFOUND, EXTRA_REGIMES, MASK_DECODE,  # noqa: E402
                           MASK_MIXED)



def regime(name: str) -> S.Regime:
    """Look up a step-1 regime or one of the additions above."""
    if name in EXTRA_REGIMES:
        return EXTRA_REGIMES[name]
    return S.REGIMES_BY_NAME[name]


def widths(r: S.Regime) -> tuple[int, int]:
    """(incumbent scores columns, right-sized scores columns) for a regime."""
    inc = S.num_sublanes_total(r.max_model_len) * 128
    pps = S.pages_per_seq(r.max_model_len)
    live = min(r.kv_len, pps * S.PAGE_SIZE)
    rs = max(S.cdiv(live, 128), S.cdiv(S.INDEX_TOPK, 128)) * 128
    return inc, rs
