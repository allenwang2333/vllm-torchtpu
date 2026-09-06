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
"""Run the Option B bucket-correctness gate against the INTEGRATED `src/` tree.

The gate itself is not re-implemented here and not copied. This file re-points
`probes/bucket_correctness/run_bucket_correctness.py` -- byte for byte the same
oracle, the same rank/recall criterion, the same index-set comparison, the same
poison arm and the same two negative controls -- at two different modules:

    variant   ->  vllm_torchtpu.kernels.deepseek_v4.streamindex_topk
                  i.e. the PORTED production file, not `variants/right-size/`.
    incumbent ->  `baseline_streamindex_topk.py` beside this file, which is
                  `git show 097fb5c4:src/.../streamindex_topk.py` verbatim
                  (md5 1e221f10cbf3573aa0d2ae78c3463532, identical to
                  `origin/main`'s copy). The pod has no `.git`, so the baseline
                  is carried in as a file rather than extracted on the fly.

Both substitutions are made by rebinding two attributes of `_rs_support`, which
is the only place the gate names a module. Nothing under `instruments/`, nothing
under `variants/` and nothing under `probes/bucket_correctness/` is edited; all
three are read-only to this script.

`--bkv` rebinds `shapes.NUM_KV_PAGES_PER_BLOCK`, so the same shapes can be run
at the incumbent `(2,2,2)` and at Option E's `(4,2,4)`. `shapes.py` is
read-only; rebinding a module attribute from here is the sanctioned mechanism.
Both `predict()` and the kernel call read that attribute at call time, so both
follow the rebind. (`shapes.num_sublanes_total` does NOT -- its default binds at
def time -- but the gate never calls it.)

Everything after `--` is passed through to the gate's own argument parser.
"""

import argparse
import functools
import importlib
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = __file__.rsplit("/kernel_work/", 1)[0]
RS_DIR = os.path.join(REPO, "kernel_work", "streamindex_topk", "variants",
                      "right-size")
GATE = os.path.join(REPO, "kernel_work", "streamindex_topk", "probes",
                    "bucket_correctness", "run_bucket_correctness.py")
BASELINE = os.path.join(_HERE, "baseline_streamindex_topk.py")

SRC_MODULE = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"

sys.path.insert(0, REPO)
sys.path.insert(0, RS_DIR)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def _ported():
    """The integrated production kernel -- the thing under test."""
    return importlib.import_module(SRC_MODULE)


@functools.lru_cache(maxsize=None)
def _baseline():
    """Pre-port `src/`, loaded under its own module name so its jit cache is
    separate from the ported module's."""
    return _load_by_path("baseline_streamindex_topk", BASELINE)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--bkv", default="",
                    help="comma tuple to rebind shapes.NUM_KV_PAGES_PER_BLOCK")
    args, rest = ap.parse_known_args(argv)

    from kernel_work.streamindex_topk.instruments import shapes as S

    if args.bkv:
        tup = tuple(int(x) for x in args.bkv.split(","))
        assert len(tup) == 3, args.bkv
        S.NUM_KV_PAGES_PER_BLOCK = tup
    print(f"[src_port] NUM_KV_PAGES_PER_BLOCK = {S.NUM_KV_PAGES_PER_BLOCK}",
          flush=True)

    import _rs_support as RS
    RS.load_variant = _ported
    RS.load_incumbent = _baseline

    ported, baseline = _ported(), _baseline()
    print(f"[src_port] variant   (ported)   = {ported.__file__}", flush=True)
    print(f"[src_port] incumbent (baseline) = {baseline.__file__}", flush=True)

    # The ported file must be the one under src/, not a stale site-packages
    # copy -- that is the single way this whole check could test nothing.
    assert os.path.realpath(ported.__file__).endswith(
        "/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py"), (
            ported.__file__)
    # The port must carry B and the baseline must not. Without both of these
    # the arms below could be comparing a thing to itself.
    import inspect

    def _params(mod):
        fn = mod.streamindex_topk
        fn = getattr(fn, "__wrapped__", fn)
        return set(inspect.signature(fn).parameters)

    assert "kv_len_bucket" in _params(ported), "ported kernel has no bucket"
    assert "kv_len_bucket" not in _params(baseline), "baseline has a bucket"

    # The shipping bucket formula, checked here because nothing else runs it.
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        MAX_SLICE_WORDS)
    for mml, want in ((66560, min(66560, MAX_SLICE_WORDS)),
                      (9216, 9216), (131072, MAX_SLICE_WORDS)):
        got = ported.static_kv_len_bucket(mml)
        print(f"[src_port] static_kv_len_bucket({mml}) = {got} "
              f"(expect {want})", flush=True)
        assert got == want, (mml, got, want)
        assert isinstance(got, int), type(got)

    gate = _load_by_path("run_bucket_correctness_gate", GATE)
    return gate.main(rest)


if __name__ == "__main__":
    sys.exit(main())
