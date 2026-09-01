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
"""Reject a bad `vllm serve` argv in seconds instead of in a server lifetime.

    preflight_serve_argv.py <arg> [<arg> ...]      # everything after `vllm serve`
    preflight_serve_argv.py --self-test            # prove the gate bites

WHY THIS EXISTS. `--model-loader-extra-config` parses cleanly and is then refused
by `dummy_loader.py:29` under `--load-format dummy`. vLLM raises it inside the
engine core at worker init, so the visible failure is
`Engine core initialization failed. See root cause above` with the cause several
hundred lines deep in a worker log the caller usually only tails. It cost three
CDK arms across two jobs (`j-94240ea7`, `j-8cbec927`) and would have cost the
first cell of a pod job.

Argparse acceptance is NOT the gate: the rejections live one layer down, in the
config objects. Those objects are constructible on a chipless host -- a full
`VllmConfig` is not (it demands real TPU devices), but the loader is, and the
loader is where this class of refusal lives. So the gate builds what it can and
says plainly what it did not build, rather than implying more coverage than it
has.

Exit 0 = accepted, 1 = rejected (with the reason on stdout), 2 = usage.
"""
from __future__ import annotations

import os
import sys

# The repo root holds a `vllm/` directory (the Dockerfile copies one in), and
# `python3 <script>` puts the caller's cwd on sys.path as "". Importing vllm
# from the repo root therefore resolves to that directory as a namespace
# package and dies with `cannot import name 'SamplingParams' from 'vllm'
# (unknown location)`. Drop cwd before importing anything of vLLM's.
sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd())]


def _stub_tpu_device_if_needed() -> str:
    """Let the parser build on a host with no TPU, and say that we did.

    `DeviceConfig.__post_init__` calls `torch.device("tpu")`, which torch
    refuses without the TPU backend registered, and vLLM builds a DeviceConfig
    just to compute an argparse default. That is the ONLY concession this gate
    makes, and it is unrelated to anything it checks -- it cannot hide a
    loader or CLI rejection. Stated in the output so nobody reads more
    coverage into a green than is there.
    """
    import torch
    try:
        torch.device("tpu")
        return "native"
    except RuntimeError:
        import vllm.config.device as device_config

        def _post_init(self):
            self.device_type = "tpu"
            self.device = torch.device("cpu")

        device_config.DeviceConfig.__post_init__ = _post_init
        return "stubbed (no TPU on this host; affects the device default only)"


def check(argv: list[str]) -> tuple[bool, str]:
    device_note = _stub_tpu_device_if_needed()
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    try:
        ns = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    except SystemExit as exc:                      # argparse exits rather than raises
        return False, f"argv rejected by the CLI parser (exit {exc.code})"

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader

    load_format = getattr(ns, "load_format", "auto")
    extra = getattr(ns, "model_loader_extra_config", None) or {}
    try:
        loader = get_model_loader(
            LoadConfig(load_format=load_format, model_loader_extra_config=extra))
    except Exception as exc:                       # noqa: BLE001 - any refusal is a fail
        return False, f"{type(exc).__name__}: {exc}"

    return True, (f"load_format={load_format} loader={type(loader).__name__} "
                  f"extra_config={extra or '{}'} "
                  f"block_size={getattr(ns, 'block_size', None)} "
                  f"max_num_seqs={getattr(ns, 'max_num_seqs', None)} "
                  f"[device={device_note}; VllmConfig NOT built -- it needs "
                  f"real devices]")


def self_test() -> int:
    """A gate never seen to fail is not a gate."""
    bad = ["m", "--load-format", "dummy",
           "--model-loader-extra-config", '{"memory_limit":17179869184}']
    good = ["m", "--load-format", "dummy"]
    ok_bad, why_bad = check(bad)
    ok_good, why_good = check(good)
    print(f"[preflight] self-test negative: accepted={ok_bad} ({why_bad})")
    print(f"[preflight] self-test positive: accepted={ok_good} ({why_good})")
    if ok_bad:
        print("[preflight] SELF-TEST FAILED: the gate does not bite on a known bad argv")
        return 1
    if not ok_good:
        print("[preflight] SELF-TEST FAILED: the gate rejects a known good argv")
        return 1
    print("[preflight] self-test OK")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "--self-test":
        return self_test()
    ok, why = check(argv[1:])
    print(f"[preflight] {'ACCEPTED' if ok else 'REJECTED'} {why}")
    if not ok:
        print("[preflight] the serve command would have died at engine init, "
              "several hundred log lines after the last thing you would read.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
