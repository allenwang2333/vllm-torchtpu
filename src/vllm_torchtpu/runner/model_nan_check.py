# SPDX-License-Identifier: Apache-2.0
"""Opt-in final hidden/logits NaN detection without changing model outputs."""

from __future__ import annotations

import functools
import socket
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorStats,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan

logger = init_logger(__name__)


def _has_nan_jax(tensor: jax.Array, valid_rows: jax.Array) -> jax.Array:
    return jnp.isnan(tensor).any(where=valid_rows[:, None])


@functools.cache
def _get_has_nan_op() -> torch.library.CustomOpDef:
    return pallas.jax_op("pallas::model_has_nan", _has_nan_jax)


def _stats_paths(
    connector_cls: type[KVConnectorBase_V1], vllm_config: VllmConfig
) -> list[tuple[str, ...]]:
    if issubclass(connector_cls, MultiConnector):
        children = connector_cls._get_connector_classes_and_configs(vllm_config)
        return [
            (child_cls.__name__, *path)
            for child_cls, child_config in children
            for path in _stats_paths(child_cls, child_config)
        ]
    return (
        [()]
        if isinstance(connector_cls.build_kv_connector_stats(), TpuKVConnectorStats)
        else []
    )


class ModelNaNChecker:
    """Check each real forward chunk; log errors and keep serving.

    Like RTP-LLM's PyWrappedModel final-output checks, this detects NaN,
    allowing Inf. Padding is excluded and errors are logged rather than raised.
    The JAX reduction returns one boolean scalar to the host.
    """

    def __init__(
        self,
        rank: int,
        stats_path: tuple[str, ...] | None = None,
        *,
        engine_idx: int = 0,
    ) -> None:
        self.rank = rank
        self.engine_idx = engine_idx
        self._hostname = socket.gethostname()
        self._stats_path = stats_path
        self._round_has_nan: bool | None = None

    @classmethod
    def from_vllm_config(cls, rank: int, vllm_config: VllmConfig) -> ModelNaNChecker:
        """Resolve the telemetry carrier once using registered connector APIs."""
        engine_idx = vllm_config.parallel_config.data_parallel_rank
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            return cls(rank, engine_idx=engine_idx)
        connector_cls = KVConnectorFactory.get_connector_class(transfer_config)
        paths = _stats_paths(connector_cls, vllm_config)
        if len(paths) > 1:
            raise ValueError(
                "NaN summary reporting requires at most one TPU stats leaf"
            )
        return cls(rank, paths[0] if paths else None, engine_idx=engine_idx)

    def take_result(self) -> bool | None:
        """Consume the OR of all checks since the previous output collection."""
        result, self._round_has_nan = self._round_has_nan, None
        return result

    def attach_stats(self, stats: KVConnectorStats | None) -> KVConnectorStats | None:
        """Move one round's result into its configured connector stats leaf."""
        result = self.take_result()
        if result is None or self._stats_path is None:
            return stats
        leaf = TpuKVConnectorStats()
        leaf.record_model_nan(result)
        update: KVConnectorStats = leaf
        for key in reversed(self._stats_path):
            update = MultiKVConnectorStats(data={key: update})
        return update if stats is None else stats.aggregate(update)

    def check_hidden(self, hidden: torch.Tensor, plan: SequenceLayoutPlan) -> bool:
        """Check all real rank-local tokens before hidden-state gathering."""
        token_indices = plan._packed_to_request_major_token_indices
        if token_indices is None:
            token_indices = np.arange(plan.global_padded_num_tokens)
        token_indices = token_indices[plan.token_slice]
        valid_rows = torch.from_numpy(
            (token_indices >= 0) & (token_indices < plan.global_num_tokens)
        )
        return self._check(hidden, valid_rows, "last_hidden")

    def check_logits(self, logits: torch.Tensor, num_valid_rows: int) -> bool:
        """Check real sampling/verification rows before logits processing."""
        assert 0 <= num_valid_rows <= logits.shape[0], (
            "Real sampling/verification rows must fit the logits buffer: "
            f"rank={self.rank}, real={num_valid_rows}, shape={logits.shape}"
        )
        valid_rows = torch.arange(logits.shape[0], device="cpu") < num_valid_rows
        return self._check(logits, valid_rows, "logits")

    def _check(self, tensor: torch.Tensor, valid_rows: torch.Tensor, name: str) -> bool:
        assert valid_rows.shape == (tensor.shape[0],), (
            "NaN check row mapping must cover the rank-local output: "
            f"rank={self.rank}, tensor={name}, shape={tensor.shape}, "
            f"mapping={valid_rows.shape}"
        )
        result = _get_has_nan_op()(tensor, valid_rows.to(tensor.device))
        synchronize_tensors(result, wait=True)
        has_nan = bool(result.to("cpu").item())
        self._round_has_nan = has_nan or bool(self._round_has_nan)
        if has_nan:
            logger.error(
                "NaN detected in %s: host=%s, engine=%d, rank=%d, shape=%s",
                name,
                self._hostname,
                self.engine_idx,
                self.rank,
                tuple(tensor.shape),
            )
        return has_nan
