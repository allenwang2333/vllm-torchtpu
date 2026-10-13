"""100% statement and branch coverage test suite for TPUMultiConnector."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_vllm_multi_connector_stub() -> None:
    if "vllm" not in sys.modules:
        vllm_mod = types.ModuleType("vllm")
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod._print_warning_once = MagicMock()  # type: ignore[attr-defined]
        vllm_mod.logger = logger_mod  # type: ignore[attr-defined]
        sys.modules["vllm"] = vllm_mod
        sys.modules["vllm.logger"] = logger_mod

    mod_name = "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ModuleNotFoundError:
            for parent in (
                "vllm.distributed",
                "vllm.distributed.kv_transfer",
                "vllm.distributed.kv_transfer.kv_connector",
                "vllm.distributed.kv_transfer.kv_connector.v1",
            ):
                sys.modules.setdefault(parent, types.ModuleType(parent))
            mc_mod = types.ModuleType(mod_name)

            class _StubMultiConnector:
                def __init__(self, connectors=None) -> None:
                    self._connectors = list(connectors or [])

            mc_mod.MultiConnector = _StubMultiConnector  # type: ignore[attr-defined]
            sys.modules[mod_name] = mc_mod


_ensure_vllm_multi_connector_stub()

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (  # noqa: E402
    TPUMultiConnector,
)


def _make_connector(children: list[object]) -> TPUMultiConnector:
    conn = object.__new__(TPUMultiConnector)
    conn._connectors = list(children)
    return conn


def test_register_runner_empty_and_mixed_children() -> None:
    """Verify register_runner handles empty, accepting, and non-accepting children."""
    empty_conn = _make_connector([])
    runner = SimpleNamespace(name="tpu_runner")
    empty_conn.register_runner(runner)

    received: list[object] = []
    child_with_runner = SimpleNamespace(register_runner=lambda r: received.append(r))
    child_without_runner = SimpleNamespace()

    conn = _make_connector([child_with_runner, child_without_runner])
    conn.register_runner(runner)
    assert received == [runner]


def test_get_block_ids_with_load_errors_group_index_all_branches() -> None:
    """Verify group index across 0-fn, 1-fn, multi-fn, and conflicting children."""
    empty_conn = _make_connector([])
    assert empty_conn.get_block_ids_with_load_errors_group_index() is None
    assert empty_conn.get_block_ids_with_load_errors_group_index() is None

    c_no_attr = SimpleNamespace()
    c_none = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: None)
    c_group2_a = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 2)
    c_group2_b = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 2)
    c_group3 = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 3)

    single_fn_conn = _make_connector([c_no_attr, c_group2_a])
    assert single_fn_conn.get_block_ids_with_load_errors_group_index() == 2
    assert single_fn_conn.get_block_ids_with_load_errors_group_index() == 2

    multi_fn_conn = _make_connector([c_no_attr, c_none, c_group2_a, c_group2_b])
    assert multi_fn_conn.get_block_ids_with_load_errors_group_index() == 2
    assert multi_fn_conn.get_block_ids_with_load_errors_group_index() == 2

    conflict_conn = _make_connector([c_group2_a, c_group3])
    with pytest.raises(RuntimeError, match="Conflicting KV-load-error group indices"):
        conflict_conn.get_block_ids_with_load_errors_group_index()
