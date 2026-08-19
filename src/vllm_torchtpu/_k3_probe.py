# SPDX-License-Identifier: Apache-2.0
"""One-shot diagnostic probe for the Kimi K3 speculative-decode bring-up.

Temporary hackathon instrumentation. Every entry point is defensive: a probe
failure must never affect the run it is measuring.

Markers are single-line and greppable:

    [K3PROBE_<TAG>] key=value key=value ...

Environment:
    K3_PROBE=0              disable all markers (default: enabled)
    K3_PROBE_ALLOW_SPEC=1   bypass the KDA speculative-decode guard so the
                            engine can be started with spec decode enabled.
                            Outputs are EXPECTED TO BE WRONG until the KDA
                            kernels learn the gdn/v3 checkpoint-window
                            protocol; this exists to measure the throughput
                            ceiling and to find what breaks next.
"""

from __future__ import annotations

import os

_seen: set[str] = set()


def _enabled() -> bool:
    return os.environ.get("K3_PROBE", "1") != "0"


def allow_spec() -> bool:
    """True when the KDA spec-decode guard should be bypassed."""
    return os.environ.get("K3_PROBE_ALLOW_SPEC", "0") == "1"


def emit(tag: str, once_key: str | None = None, **fields: object) -> None:
    """Print one greppable marker line. Never raises."""
    try:
        if not _enabled():
            return
        if once_key is not None:
            if once_key in _seen:
                return
            _seen.add(once_key)
        body = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[K3PROBE_{tag}] {body}", flush=True)
    except Exception:
        pass


def emit_versions() -> None:
    """Report the runtime pins actually present in the container."""
    try:
        if "versions" in _seen:
            return
        versions: dict[str, object] = {}
        for name in ("torch", "torch_tpu", "jax", "jaxlib", "vllm", "libtpu"):
            try:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                versions[name] = "missing"
        emit("VERSIONS", once_key="versions", **versions)
    except Exception:
        pass
