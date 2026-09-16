# SPDX-License-Identifier: Apache-2.0
"""Scheduler->worker metadata contract shared by the TPU KV connectors.

These dataclasses are the connector-neutral request bookkeeping (what to
send, what to load) attached to scheduler output. They are used by both the
zmq/shm transport stack (zmq_shm_base) and the Raiden connector, so they
live here rather than inside either transport module.
"""

from dataclasses import dataclass, field
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorMetadata

ReqId = str


@dataclass
class SendMeta:
    uuid: int
    local_block_ids: list[int]
    expiration_time: float
    # Exact transfer extent for controller-planned PCP striping. Legacy send
    # paths leave this unset.
    num_tokens: int | None = None
    # Uniform-mamba-layout source state slots, one per mamba kv-cache group
    # ordinal (the block holding the final recurrent state); None for FA-only
    # Stage-3 models and all legacy paths.
    mamba_state_block_ids: list[int] | None = None
    # Where the consumer wants this request's KV written, as published by the
    # consumer through the proxy while this prefill was still running. Present
    # only when the producer is the side that starts the transfer; None means
    # the consumer will start it itself, as it always has.
    dst_params: dict | None = None
    # Stage 6.5. A chunked prefill ships one request as several sends, and
    # these three say which one this is. The send covers tokens
    # [start_token, num_tokens): 0 and the whole prompt for an unchunked
    # request, which is why the defaults reproduce the single-send behaviour
    # exactly. ``chunk_index`` only ever appears in logs. ``is_final_chunk``
    # is what tells the producer this send closes the request out, so the
    # registration can be retired and the blocks freed.
    start_token: int = 0
    chunk_index: int = 0
    is_final_chunk: bool = True
    # Stage 7. Which attention layer group this send carries, as an index into
    # the worker's FA pool tags. None means all of them, which is every send
    # until the layer split is turned on. The early group's send covers the
    # whole prompt but only its own layers, so it is split on a different axis
    # from a chunk and the two compose.
    fa_group: int | None = None
    # Stage 7c. Set when this send was built before the forward that writes
    # its pages, to be released by that group's doorbell rather than fired on
    # arrival. It is the whole difference between building the transfer early
    # and shipping it early. The worker must honour it: firing a send marked
    # this way without waiting for the bell ships pages the model has not
    # written, which is why it is a field on the send rather than a choice the
    # worker makes for itself.
    arm_on_bell: int | None = None


@dataclass
class LoadMeta:
    uuid: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str | list[str]
    remote_port: int | list[int]
    remote_side_channel_port: Optional[int] = None
    # Whether the worker reports this load's completion to the scheduler as
    # finished_recving. False when this connector doesn't own the request's
    # load state (the request is not WAITING_FOR_REMOTE_KVS here, e.g. a full
    # local cache hit or delegation to another MultiConnector child), where a
    # report would trip the scheduler's assert or prematurely resume the
    # request.
    report_completion: bool = True


@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)
