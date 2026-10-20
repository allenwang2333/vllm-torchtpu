# SPDX-License-Identifier: Apache-2.0
"""Loads the opt-in routing tap in every process that has this directory first
on PYTHONPATH, then runs the sitecustomize this file shadows.

Inert unless DSV41_ROUTING_TAP_OUT is set. See README.md.
"""

import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))


def _chain_shadowed_sitecustomize() -> None:
    for entry in sys.path:
        path = os.path.join(entry or os.getcwd(), "sitecustomize.py")
        if os.path.dirname(os.path.abspath(path)) == _here or not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


_chain_shadowed_sitecustomize()

if os.environ.get("DSV41_ROUTING_TAP_OUT"):
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import routing_tap

    routing_tap.install()
