# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the v6d TPU DMA provider."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_get_provider_does_not_import_handler_module():
    sys.modules.pop("vllm_torchtpu.offload.v6d.handler", None)
    from vllm_torchtpu.offload.v6d.provider import get_provider

    provider = get_provider()

    assert provider.api_version == 1
    assert "vllm_torchtpu.offload.v6d.handler" not in sys.modules


def test_create_handler_passes_context_rank():
    from vllm_torchtpu.offload.v6d.provider import TpuDmaProvider

    handler_cls = MagicMock()
    context = SimpleNamespace(rank=7)
    with patch("vllm_torchtpu.offload.v6d.provider._load_handler",
               return_value=handler_cls):
        result = TpuDmaProvider().create_handler(context)

    assert result is handler_cls.return_value
    handler_cls.assert_called_once_with(rank_id=7)
