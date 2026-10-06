# SPDX-License-Identifier: Apache-2.0
"""Request-level Raiden errors; receive completion remains a separate barrier."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import utils as dist_utils

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def raiden_load_failure_feedback_enabled(vllm_config: "VllmConfig") -> bool:
    """Enable early request errors for non-inline Stage-3 fail-policy loads."""
    kv = vllm_config.kv_transfer_config
    if (
        kv is None
        or str(envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() != "raiden"
        or kv.kv_role != "kv_consumer"
        or kv.kv_load_failure_policy != "fail"
        or dist_utils.get_raiden_inline_load()
    ):
        return False
    connectors = (
        kv.kv_connector_extra_config["connectors"]
        if kv.kv_connector == "TPUMultiConnector"
        else [{"kv_connector": kv.kv_connector}]
    )
    return any(
        child["kv_connector"] in ("TPUConnector", "TPURaidenConnector")
        for child in connectors
    )


@dataclass
class RaidenLoadFailures(KVConnectorWorkerMetadata):
    # Request identity avoids treating shared prefix blocks as failed receives.
    failed_req_ids: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "RaidenLoadFailures":
        assert isinstance(other, RaidenLoadFailures), (
            "Raiden load failures must aggregate with the same connector type"
        )
        self.failed_req_ids.update(other.failed_req_ids)
        return self
