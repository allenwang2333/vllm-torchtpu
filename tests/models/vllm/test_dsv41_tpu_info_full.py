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

import importlib
import logging
import sys
import types
from types import SimpleNamespace

from jax.experimental.pallas import tpu as pltpu


class _StubRequestException(Exception):
    pass


_stub_requests = types.ModuleType("requests")
_stub_requests.RequestException = _StubRequestException  # type: ignore[attr-defined]
_stub_requests.get = lambda *args, **kwargs: None  # type: ignore[attr-defined]
requests = sys.modules.setdefault("requests", _stub_requests)

_stub_vllm = types.ModuleType("vllm")
_stub_vllm_logger = types.ModuleType("vllm.logger")
_stub_vllm_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
_stub_vllm_logger._print_warning_once = SimpleNamespace(cache_clear=lambda: None)  # type: ignore[attr-defined]
_stub_vllm.logger = _stub_vllm_logger  # type: ignore[attr-defined]
sys.modules.setdefault("vllm", _stub_vllm)
sys.modules.setdefault("vllm.logger", _stub_vllm_logger)

from vllm_torchtpu import tpu_info  # noqa: E402


def test_tpu_info_all_branches_and_metadata(monkeypatch) -> None:
    """Verify 100% statement and branch coverage of vllm_torchtpu.tpu_info."""
    mod = importlib.reload(tpu_info)

    # 1. get_tpu_metadata: 200 non-empty, 200 empty, non-200, RequestException.
    captured_urls: list[str] = []

    def _ok_get(url: str, headers: dict[str, str]):
        captured_urls.append(url)
        assert headers == mod.GCE_TPU_HEADERS
        return SimpleNamespace(status_code=200, text="tpu7x-8")

    monkeypatch.setattr(mod.requests, "get", _ok_get)
    assert mod.get_tpu_metadata("accelerator-type") == "tpu7x-8"
    assert captured_urls[-1].endswith("/accelerator-type")

    monkeypatch.setattr(
        mod.requests,
        "get",
        lambda url, headers: SimpleNamespace(status_code=200, text=""),
    )
    assert mod.get_tpu_metadata("accelerator-type") is None

    monkeypatch.setattr(
        mod.requests,
        "get",
        lambda url, headers: SimpleNamespace(status_code=404, text="not found"),
    )
    assert mod.get_tpu_metadata("accelerator-type") is None

    def _raise_req(url: str, headers: dict[str, str]):
        raise requests.RequestException("metadata unreachable")

    monkeypatch.setattr(mod.requests, "get", _raise_req)
    assert mod.get_tpu_metadata("accelerator-type") is None

    # 2. get_tpu_type: env set vs env unset (fallback to metadata).
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    assert mod.get_tpu_type() == "v6e-8"

    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(mod, "get_tpu_metadata", lambda key="": "v5litepod-4")
    assert mod.get_tpu_type() == "v5litepod-4"

    # 3. get_chip_version: explicit recognized, alias recognized, None fallback,
    #    and unrecognized warning fallback.
    mod.get_chip_version.cache_clear()
    assert mod.get_chip_version("tpu7x-8") == pltpu.ChipVersion.TPU_7X
    assert mod.get_chip_version("v7x-16") == pltpu.ChipVersion.TPU_7X
    assert mod.get_chip_version("v5litepod-4") == pltpu.ChipVersion.TPU_V5E
    assert mod.get_chip_version("v6e-8") == pltpu.ChipVersion.TPU_V6E
    assert mod.get_chip_version(None) == pltpu.ChipVersion.TPU_V5E

    monkeypatch.setattr(mod, "get_tpu_type", lambda: None)
    mod.get_chip_version.cache_clear()
    assert mod.get_chip_version(None) == pltpu.ChipVersion.TPU_7X
    assert mod.get_chip_version("unrecognized-chip-99") == pltpu.ChipVersion.TPU_7X
    mod.get_chip_version.cache_clear()

    # 4. get_node_name: env set vs env empty/unset (fallback to metadata).
    monkeypatch.setenv("TPU_NAME", "tpu-node-0")
    assert mod.get_node_name() == "tpu-node-0"

    monkeypatch.setenv("TPU_NAME", "")
    monkeypatch.setattr(mod, "get_tpu_metadata", lambda key="": "gce-instance-1")
    assert mod.get_node_name() == "gce-instance-1"
    monkeypatch.delenv("TPU_NAME", raising=False)

    # 5. get_node_worker_id: env set, metadata set, and both None -> 0.
    monkeypatch.setenv("TPU_WORKER_ID", "3")
    assert mod.get_node_worker_id() == 3

    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.setattr(mod, "get_tpu_metadata", lambda key="": "6")
    assert mod.get_node_worker_id() == 6

    monkeypatch.setattr(mod, "get_tpu_metadata", lambda key="": None)
    assert mod.get_node_worker_id() == 0

    # 6. get_num_cores_per_chip: single-core vs dual-core (tpu7x, None).
    monkeypatch.setattr(mod, "get_tpu_type", lambda: "v5litepod-8")
    assert mod.get_num_cores_per_chip() == 1
    monkeypatch.setattr(mod, "get_tpu_type", lambda: "v6e-4")
    assert mod.get_num_cores_per_chip() == 1
    monkeypatch.setattr(mod, "get_tpu_type", lambda: "tpu7x-8")
    assert mod.get_num_cores_per_chip() == 2
    monkeypatch.setattr(mod, "get_tpu_type", lambda: None)
    assert mod.get_num_cores_per_chip() == 2

    # 7. get_num_chips: /dev/accel* present, /dev/vfio entries, empty /dev/vfio,
    #    and FileNotFoundError fallback.
    monkeypatch.setattr(
        mod.glob,
        "glob",
        lambda pattern: ["/dev/accel0", "/dev/accel1", "/dev/accel2", "/dev/accel3"],
    )
    assert mod.get_num_chips() == 4

    monkeypatch.setattr(mod.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(
        mod.os,
        "listdir",
        lambda path: ["0", "1", "2", "3", "vfio", "noiommu-0"],
    )
    assert mod.get_num_chips() == 4

    monkeypatch.setattr(mod.os, "listdir", lambda path: [])
    assert mod.get_num_chips() == 0

    def _raise_fnf(path: str):
        raise FileNotFoundError(path)

    monkeypatch.setattr(mod.os, "listdir", _raise_fnf)
    assert mod.get_num_chips() == 0
