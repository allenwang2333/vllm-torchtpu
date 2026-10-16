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
"""Asynchronous TPU HBM <-> Host DRAM KV and hybrid state offloading connector.

Supports multi-block prefix cache offloading, asynchronous DMA transfers, and
mixed local-HBM / offloaded-DRAM prefix hit composition for hybrid models.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    config as offloading_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    get_offload_group_idx,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

logger = init_logger(__name__)


def _enqueue_pool_writes(pool_tensors: list[torch.Tensor]) -> None:
    """Asynchronously dispatches graph execution for KV pool buffers without blocking.

    Triggers compilation and submission of any deferred operations targeting the
    pool buffers so that subsequent stream events order behind them. Performs a
    non-blocking device synchronization (`wait=False`), which dispatches no new
    work when the writes are already enqueued.
    """
    from torch_tpu._internal import sync

    sync.synchronize(pool_tensors, wait=False)


def _record_fence_event() -> torch.Event:
    """Records a TPU stream event to track completion of enqueued device operations.

    The event orders after work already submitted to the stream, so callers
    enqueue the writes they need fenced (via `_enqueue_pool_writes`) before
    recording.
    """
    event = torch.tpu.Event()
    event.record()
    return event


def _event_is_retired(event: torch.Event) -> bool:
    """Non-blockingly verifies fence event completion and surfaces deferred errors.

    `event.query()` polls device status without blocking the host thread. Once
    execution terminates, calling `event.synchronize()` returns immediately and
    raises any deferred asynchronous runtime errors before acknowledging the fence.
    """
    if not event.query():
        return False
    event.synchronize()
    return True


@dataclass
class RaidenOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler -> worker metadata.

    Transfers are controller-driven (load_jobs and store_jobs are empty);
    extra fields carry fence requests and completion echoes.
    """

    # Store jobs awaiting device fence from all ranks before launching save.
    fence_job_ids: set[int] = field(default_factory=set)
    # Completed store jobs to echo back to the scheduler state machine.
    finished_store_job_ids: list[int] = field(default_factory=list)
    # Completed load jobs (job_id -> req_id) to echo back and unblock requests.
    finished_load_jobs: dict[int, str] = field(default_factory=dict)
    # Destination device blocks of failed loads, reported for scheduler recompute.
    failed_load_device_block_ids: list[int] = field(default_factory=list)


@dataclass
class RaidenOffloadingWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler metadata: contains completed-job echoes and
    per-rank fence acks."""

    fenced_jobs: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other):
        assert isinstance(other, RaidenOffloadingWorkerMetadata)
        merged_completed = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged_completed[job_id] = merged_completed.get(job_id, 0) + v
        merged_fenced = dict(self.fenced_jobs)
        for job_id, v in other.fenced_jobs.items():
            merged_fenced[job_id] = merged_fenced.get(job_id, 0) + v
        # completed_jobs and transfer_stats are inherited from vLLM base dataclass.
        return RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=merged_completed,
            # pyrefly: ignore[unexpected-keyword]
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            fenced_jobs=merged_fenced,
        )


@dataclass
class _ParkedStore:
    """A store job parked until every worker rank acknowledges the device fence."""

    keys: list[OffloadKey]
    device_block_ids: list[int]
    acks: int = 0


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.
    """

    def __init__(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}
        self._scheduler: Any = None
        self._pending_failed_jobs: list[Any] = []
        self._init_store_split()
        self._init_state_demotion(spec.dedup_layout is not None)

    def _init_store_split(self) -> None:
        """Sets up the per-class count of launched store keys.

        Keys of the full-attention groups count as split_fa_keys and all
        others as split_mamba_keys in the `offload-tier store-split` line.
        """
        self._fa_group_ids = frozenset(
            g.group_idx
            for g in self.config.kv_group_configs
            if g.sliding_window_size_in_chunks is None and not g.requires_cow_source
        )

    def _init_state_demotion(self, default_on: bool) -> None:
        """Sets up demotion of superseded Mamba/GDN boundary states.

        A conversation's next turn resumes from the newest boundary state its
        last turn stored; older boundaries of the same chain serve only forks.
        With TPU_RAIDEN_TIER_DEMOTE_STATE on (default: on with the deduplicated
        layout), a finished request's boundary states below its newest one,
        and the boundary it was admitted from, are demoted: they still answer
        lookups, but the tier evicts them before any live entry.
        """
        env = os.environ.get("TPU_RAIDEN_TIER_DEMOTE_STATE")
        self._demote_stale_state = (env == "1") if env is not None else default_on
        self._state_group_ids = frozenset(
            g.group_idx for g in self.config.kv_group_configs if g.requires_cow_source
        )
        # req_id -> boundary tokens -> the state keys stored for that boundary.
        self._state_boundaries: dict[str, dict[int, list[OffloadKey]]] = {}
        # req_id -> (boundary tokens, state keys) of the admission's hit.
        self._state_hits: dict[str, tuple[int, list[OffloadKey]]] = {}

    def _complete_jobs_directly(
        self,
        scheduler: Any,
        finished_jobs: Iterable[Any],
        extra_store_job_ids: Iterable[int] = (),
    ) -> int:
        """Complete finished Raiden store/load jobs directly on the scheduler
        without a 2-step worker echo round-trip."""
        completed_jobs: dict[int, int] = {}
        load_req_ids: list[str] = []
        num_workers = self.config.num_workers
        for jid in extra_store_job_ids:
            if jid in self._jobs:
                completed_jobs[jid] = num_workers
        for finished_job in finished_jobs:
            if finished_job.failed_device_block_ids:
                if not hasattr(self, "_pending_failed_jobs"):
                    self._pending_failed_jobs = []
                self._pending_failed_jobs.append(finished_job)
                continue
            if finished_job.job_id in self._jobs:
                completed_jobs[finished_job.job_id] = num_workers
            if finished_job.op == AdmissionOp.LOAD and finished_job.req_id is not None:
                load_req_ids.append(finished_job.req_id)

        finished_recving: set[str] = set()
        for req_id in load_req_ids:
            if req_id not in scheduler.requests:
                continue
            req_status = self._req_status.get(req_id)
            has_other_loads = req_status is not None and any(
                jid not in completed_jobs
                and jid in self._jobs
                and not self._jobs[jid].is_store
                for jid in req_status.transfer_jobs
            )
            if not has_other_loads:
                finished_recving.add(req_id)

        if completed_jobs or finished_recving:
            kv_out = KVConnectorOutput(
                finished_recving=finished_recving,
                kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                    # pyrefly: ignore[unexpected-keyword]
                    completed_jobs=completed_jobs,
                    fenced_jobs={},
                ),
            )
            self.update_connector_output(kv_out)
            if finished_recving:
                scheduler._update_from_kv_xfer_finished(
                    KVConnectorOutput(finished_recving=finished_recving)
                )
        return len(finished_recving)

    def poll_and_complete_transfers(self, scheduler: Any) -> tuple[int, int, int, bool]:
        """Poll Raiden C++ transfer completions at the start of ``schedule()``
        so completed H2D loads promote ``WAITING_FOR_REMOTE_KVS`` requests in
        the very next step instead of waiting 2 steps for the worker echo."""
        self._scheduler = scheduler
        t0_ns = time.perf_counter_ns()
        manager = self._raiden_manager
        initial_finished = manager.poll_finished_jobs()
        early_reqs = (
            self._complete_jobs_directly(scheduler, initial_finished)
            if initial_finished
            else 0
        )
        waited_reqs = 0
        waited = False

        has_pending_loads = any(not js.is_store for js in self._jobs.values())
        if has_pending_loads and early_reqs == 0:
            # A request parked on an async load holds its KV blocks, so the
            # scheduler queues it in kv_holding_waiting.
            has_waiting_remote = any(
                r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and r.request_id not in scheduler.finished_recving_kv_req_ids
                for r in scheduler.kv_holding_waiting
            )
            if has_waiting_remote:
                wait_ms = float(tpu_envs.VLLM_TPU_RAIDEN_LOAD_WAIT_MS)
                if wait_ms > 0.0:
                    if not getattr(scheduler, "running", None):
                        wait_ms = max(wait_ms, 5.0)
                    deadline_ns = time.perf_counter_ns() + int(wait_ms * 1_000_000.0)
                    waited = True
                    while time.perf_counter_ns() < deadline_ns:
                        time.sleep(0.0001)
                        polled = manager.poll_finished_jobs()
                        if polled:
                            waited_reqs += self._complete_jobs_directly(
                                scheduler, polled
                            )
                            if waited_reqs > 0 or not any(
                                not js.is_store for js in self._jobs.values()
                            ):
                                break
                        elif not manager.has_pending_work():
                            break

        elapsed_ns = time.perf_counter_ns() - t0_ns
        tel = self._raiden_manager.telemetry
        tel["poll_ns"] += elapsed_ns
        if waited:
            tel["wait_ns"] += elapsed_ns
            tel["waited_polls"] += 1
        return elapsed_ns, early_reqs, waited_reqs, waited

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)

        manager = self._raiden_manager

        # Park new store jobs: wait for all ranks to complete device fence.
        new_fence_ids: set[int] = set()
        for job_id, job in meta.store_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, RaidenLoadStoreSpec)
            device_block_ids = src_spec.block_ids.tolist()
            assert len(device_block_ids) == len(dst_spec.keys), (
                len(device_block_ids),
                len(dst_spec.keys),
            )
            self._fence_pending[job_id] = _ParkedStore(
                keys=dst_spec.keys, device_block_ids=device_block_ids
            )
            new_fence_ids.add(job_id)

        # Submit load jobs immediately: destination HBM blocks are freshly
        # allocated and safe to write into.
        for job_id, job in meta.load_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, RaidenLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            device_block_ids = dst_spec.block_ids.tolist()
            assert len(device_block_ids) == len(src_spec.keys), (
                len(device_block_ids),
                len(src_spec.keys),
            )
            manager.submit_load_job(
                job_id,
                src_spec.keys,
                device_block_ids,
                job.req_id,
                pinned=src_spec.pinned,
            )

        # Handle jobs_to_flush (source blocks are about to be recycled):
        # - Cancel fence-pending stores before they start.
        # - Synchronously drain launched stores so DMA finishes before blocks
        #   are overwritten.
        finished_store_ids: list[int] = []
        finished_loads: dict[int, str] = {}
        failed_load_block_ids: list[int] = []
        flush_ids = meta.jobs_to_flush or set()
        launched_flush_ids: set[int] = set()
        for job_id in flush_ids:
            parked = self._fence_pending.pop(job_id, None)
            if parked is not None:
                manager.on_store_job_cancelled(parked.keys)
                finished_store_ids.append(job_id)
                # Avoid useless TPU sync if created and cancelled in the same step.
                new_fence_ids.discard(job_id)
                logger.debug(
                    "TPURaidenOffloadingConnector: cancelled fence-pending "
                    "store job %d (source blocks reused)",
                    job_id,
                )
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
        drained = manager.drain_jobs(launched_flush_ids) if launched_flush_ids else []
        polled = manager.poll_finished_jobs()

        sched_ref = getattr(self, "_scheduler", None)
        if sched_ref is not None:
            self._complete_jobs_directly(
                sched_ref,
                list(drained) + polled,
                extra_store_job_ids=finished_store_ids,
            )
            finished_store_ids = []
            pending_failed = getattr(self, "_pending_failed_jobs", None)
            if pending_failed:
                for failed_job in pending_failed:
                    if failed_job.op == AdmissionOp.STORE:
                        finished_store_ids.append(failed_job.job_id)
                    else:
                        assert failed_job.req_id is not None
                        finished_loads[failed_job.job_id] = failed_job.req_id
                        failed_load_block_ids.extend(failed_job.failed_device_block_ids)
                pending_failed.clear()
        else:
            # Collect completed/drained jobs for the worker echo channel.
            for finished_job in list(drained) + polled:
                if finished_job.op == AdmissionOp.STORE:
                    finished_store_ids.append(finished_job.job_id)
                else:
                    assert finished_job.req_id is not None
                    finished_loads[finished_job.job_id] = finished_job.req_id
                    failed_load_block_ids.extend(finished_job.failed_device_block_ids)

        if self._demote_stale_state:
            self._demote_superseded_states(scheduler_output.finished_req_ids or ())

        # Return empty load_jobs/store_jobs: workers act as passive DMA endpoints.
        return RaidenOffloadingConnectorMetadata(
            load_jobs={},  # pyrefly: ignore[unexpected-keyword]
            store_jobs={},  # pyrefly: ignore[unexpected-keyword]
            # pyrefly: ignore[unexpected-keyword]
            jobs_to_flush=meta.jobs_to_flush,
            fence_job_ids=new_fence_ids,
            finished_store_job_ids=finished_store_ids,
            finished_load_jobs=finished_loads,
            failed_load_device_block_ids=failed_load_block_ids,
        )

    def _lookup(
        self, req_status: Any, max_num_new_tokens: int | None = None
    ) -> int | None:
        """Upstream's lookup, with partial tails probed only above the local hit.

        A partial-tail boundary at or below the local HBM hit
        (``_tpu_local_hit_tokens``, which keeps the sub-block tail the
        scheduler strips before asking the connector) loads nothing HBM does
        not already hold.
        """
        complete_hit = self._lookup_complete_chunks(req_status, max_num_new_tokens)
        req_status.partial_tail_boundary = None
        if complete_hit is None or not self.config.supports_partial_tail:
            return complete_hit

        local_tokens = req_status.num_locally_computed_tokens
        complete_boundary = local_tokens + complete_hit
        tokens_per_hash = self.config.tokens_per_hash
        block_end = complete_boundary + self._partial_tail_block_size
        max_boundary = min(req_status.req.num_prompt_tokens - 1, block_end - 1)
        if max_num_new_tokens is not None:
            max_boundary = min(max_boundary, local_tokens + max_num_new_tokens)
        if req_status.max_load_tokens is not None:
            max_boundary = min(max_boundary, local_tokens + req_status.max_load_tokens)
        max_boundary = round_down(max_boundary, tokens_per_hash)
        min_boundary = complete_boundary
        if complete_hit == 0:
            local_hit = int(getattr(req_status.req, "_tpu_local_hit_tokens", 0) or 0)
            if local_hit > min_boundary:
                min_boundary = local_hit
        if max_boundary <= min_boundary:
            return complete_hit

        pending = False
        for boundary in range(max_boundary, min_boundary, -tokens_per_hash):
            boundary_pending = False
            boundary_missed = False
            boundary_keys = []
            for group_config in self.config.kv_group_configs:
                key = self._make_boundary_key(
                    req_status.req,
                    group_config.group_idx,
                    boundary,
                    req_status.req_context,
                )
                boundary_keys.append(key)
                result = self.manager.lookup(key, req_status.req_context)
                if result is LookupResult.MISS:
                    boundary_missed = True
                    break
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    boundary_pending = True

            pending |= boundary_pending
            if not boundary_missed and not boundary_pending:
                for group_config, key in zip(
                    self.config.kv_group_configs, boundary_keys
                ):
                    self._events_tracker.record_partial_lookup(
                        req_status.req, group_config, boundary, key
                    )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens

        if pending and complete_hit == 0:
            return None
        return complete_hit

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        if (
            num_computed_tokens > 0
            and self.config.kv_group_configs
            and num_computed_tokens % self.config.kv_group_configs[0].tokens_per_chunk
            != 0
        ):
            req_status = self._req_status.get(request.request_id)
            if req_status is None:
                self.on_new_request(request)
                req_status = self._req_status.get(request.request_id)
            if req_status is not None:
                req_status.num_locally_computed_tokens = num_computed_tokens
                req_status.update_offload_keys()
                req_status.update_num_hit_chunks(num_computed_tokens)
                self._touch(req_status)
            return 0, False
        max_loads = getattr(self, "_max_async_load_reqs", None)
        if max_loads is None:
            try:
                max_loads = int(os.environ.get("TPU_OFFLOAD_MAX_ASYNC_LOAD_REQS", "4"))
            except ValueError:
                max_loads = 4
            self._max_async_load_reqs = max_loads
        if max_loads > 0:
            active_loads = sum(1 for js in self._jobs.values() if not js.is_store)
            if active_loads >= max_loads:
                req_status = self._req_status.get(request.request_id)
                if req_status is None:
                    self.on_new_request(request)
                    req_status = self._req_status.get(request.request_id)
                if req_status is not None and not req_status.transfer_jobs:
                    req_status.update_offload_keys()
                    orig_local = req_status.num_locally_computed_tokens
                    orig_partial = req_status.partial_tail_boundary
                    try:
                        probe_hit = self._quiet_lookup(req_status, num_computed_tokens)
                    finally:
                        req_status.num_locally_computed_tokens = orig_local
                        req_status.partial_tail_boundary = orig_partial
                    if probe_hit is None or probe_hit > 0:
                        self._raiden_manager.telemetry["load_cap_deferrals"] += 1
                        return None, False
        num_hit_tokens, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens, max_num_new_tokens
        )
        req_status = self._req_status.get(request.request_id)
        if req_status is not None and not req_status.transfer_jobs:
            # Upstream leaves cache recency to the manager, and the Raiden
            # store learns it only through touch: refresh this request's keys
            # on every lookup that ran (one deferred for in-flight transfers
            # did not).
            self._touch(req_status)
        if not num_hit_tokens:
            local_hit = int(getattr(request, "_tpu_local_hit_tokens", 0) or 0)
            if local_hit > num_computed_tokens and req_status is not None:
                req_status.num_locally_computed_tokens = local_hit
                if (
                    self.config.supports_partial_tail
                    and local_hit % self._partial_tail_block_size != 0
                    and local_hit % self.config.tokens_per_hash == 0
                    and local_hit // self.config.tokens_per_hash - 1
                    < len(request.block_hashes)
                ):
                    self.manager.touch(
                        tuple(
                            self._make_boundary_key(
                                request,
                                group.group_idx,
                                local_hit,
                                req_status.req_context,
                            )
                            for group in self.config.kv_group_configs
                        ),
                        req_status.req_context,
                    )
        return num_hit_tokens, load_async

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        req_status = self._req_status.get(request.request_id)
        if req_status is not None:
            self._record_state_hit(
                request, req_status.num_locally_computed_tokens + num_external_tokens
            )
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.sliding_window_size_in_chunks is None:
                    # Keep next_stored_chunk_idx = 0 for FullAttention so
                    # _build_store_jobs lets prepare_store probe 0..num_chunks
                    # (skipping resident HOST/HOST_AND_HBM blocks in memory and
                    # re-storing any HBM-resident prefix chunks displaced from
                    # the host LRU).
                    group_state.next_stored_chunk_idx = 0

    def _build_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        store_jobs = super()._build_store_jobs(scheduler_output)
        # Every request that offloads this step refreshes its keys, as each
        # lookup does in get_num_new_matched_tokens.
        for req_id in dict.fromkeys(job.req_id for job in store_jobs.values()):
            req_status = self._req_status.get(req_id)
            if req_status is not None:
                self._touch(req_status)
        return store_jobs

    def _build_aligned_boundary_store_jobs(
        self, handoffs: dict[str, list[tuple[int, int, int]]]
    ) -> dict[int, TransferJob]:
        jobs = super()._build_aligned_boundary_store_jobs(handoffs)
        if self._demote_stale_state:
            for req_id, entries in handoffs.items():
                req_status = self._req_status.get(req_id)
                if req_status is None:
                    continue
                boundaries = self._state_boundaries.setdefault(req_id, {})
                for group_idx, _, boundary in entries:
                    if group_idx in self._state_group_ids:
                        boundaries.setdefault(boundary, []).append(
                            self._make_boundary_key(
                                req_status.req,
                                group_idx,
                                boundary,
                                req_status.req_context,
                            )
                        )
        return jobs

    def _record_state_hit(self, request: Request, hit_tokens: int) -> None:
        if not self._demote_stale_state or hit_tokens <= 0 or not self._state_group_ids:
            return
        if hit_tokens // self.config.tokens_per_hash > len(request.block_hashes):
            return
        req_status = self._req_status.get(request.request_id)
        if req_status is None:
            return
        self._state_hits[request.request_id] = (
            hit_tokens,
            [
                self._make_boundary_key(
                    request, group_idx, hit_tokens, req_status.req_context
                )
                for group_idx in sorted(self._state_group_ids)
            ],
        )

    def _demote_superseded_states(self, finished_req_ids: Iterable[str]) -> None:
        for req_id in finished_req_ids:
            boundaries = self._state_boundaries.pop(req_id, None)
            hit = self._state_hits.pop(req_id, None)
            if not boundaries:
                continue
            newest = max(boundaries)
            stale = [
                key
                for boundary, keys in boundaries.items()
                if boundary < newest
                for key in keys
            ]
            if hit is not None and hit[0] < newest:
                stale.extend(hit[1])
            if stale:
                self._raiden_manager.demote(stale)

    def _touch(self, req_status: Any) -> None:
        """Refreshes the Raiden host LRU for one request's offloaded keys.

        Keys touched later end nearer the MRU end, so the order below decides
        what an active conversation loses first under host pressure.
        """
        local_tokens = int(getattr(req_status, "num_locally_computed_tokens", 0) or 0)
        local_hit = int(getattr(req_status.req, "_tpu_local_hit_tokens", 0) or 0)
        if local_hit > local_tokens:
            local_tokens = local_hit
        shared_boundary = int(getattr(req_status.req, "shared_prefix_boundary", 0) or 0)
        # 1. Touch ALL FullAttention chunks in reverse order (tail -> chunk_0)
        #    so the entire active conversation stays ahead of inactive
        #    conversations AND chunk_0 (the prefix root) is the most recently
        #    used entry and can never be evicted before the tail.
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is not None:
                continue
            if group_state.offload_keys:
                self.manager.touch(
                    list(reversed(group_state.offload_keys)),
                    req_status.req_context,
                )

        # 2. Touch Mamba / sliding-window boundary chunks AFTER FullAttention so
        #    active recurrent boundary states stay at the MRU head of the host cache.
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                continue
            chunks_to_skip = max(
                0,
                group_state.num_hit_chunks - group_config.sliding_window_size_in_chunks,
            )
            keys_to_touch = list(group_state.offload_keys[chunks_to_skip:])
            if shared_boundary >= group_config.tokens_per_chunk:
                fork_idx = (shared_boundary // group_config.tokens_per_chunk) - 1
                if 0 <= fork_idx < chunks_to_skip and fork_idx < len(
                    group_state.offload_keys
                ):
                    keys_to_touch.append(group_state.offload_keys[fork_idx])
            if keys_to_touch:
                self.manager.touch(keys_to_touch, req_status.req_context)

        # 3. Touch partial-tail boundary keys LAST (both CPU-matched partial
        #    tails and HBM-matched partial tails).
        boundaries_to_touch: list[int] = []
        if req_status.partial_tail_boundary is not None:
            boundaries_to_touch.append(int(req_status.partial_tail_boundary))
        if (
            self.config.supports_partial_tail
            and local_tokens > 0
            and local_tokens % self._partial_tail_block_size != 0
            and local_tokens % self.config.tokens_per_hash == 0
            and local_tokens not in boundaries_to_touch
        ):
            boundaries_to_touch.append(local_tokens)
        for boundary in boundaries_to_touch:
            if boundary // self.config.tokens_per_hash - 1 < len(
                req_status.req.block_hashes
            ):
                self.manager.touch(
                    tuple(
                        self._make_boundary_key(
                            req_status.req,
                            group.group_idx,
                            boundary,
                            req_status.req_context,
                        )
                        for group in self.config.kv_group_configs
                    ),
                    req_status.req_context,
                )

    def estimate_external_hit_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        """Side-effect-free probe of how many additional tokens beyond
        ``num_local_tokens`` can be loaded from Raiden CPU offload."""
        if getattr(request, "skip_reading_prefix_cache", False):
            return 0
        if not self.config.kv_group_configs:
            return 0
        tokens_per_chunk = self.config.kv_group_configs[0].tokens_per_chunk
        if tokens_per_chunk <= 0:
            return 0
        aligned_local = (
            max(0, int(num_local_tokens)) // tokens_per_chunk
        ) * tokens_per_chunk
        if request.num_tokens - aligned_local < self.config.tokens_per_hash:
            return 0
        num_hashes = len(request.block_hashes)
        now = time.monotonic()
        cache = getattr(request, "_paw_ext_hit_cache", None)
        if (
            cache is not None
            and cache[0] == num_hashes
            and cache[1] == aligned_local
            and now - cache[2] < 0.25
        ):
            return max(0, (aligned_local + cache[3]) - num_local_tokens)
        req_status = self._req_status.get(request.request_id)
        if req_status is None:
            self.on_new_request(request)
            req_status = self._req_status.get(request.request_id)
        if req_status is None or req_status.transfer_jobs:
            return 0
        req_status.update_offload_keys()
        orig_local = req_status.num_locally_computed_tokens
        orig_partial = req_status.partial_tail_boundary
        try:
            ext = self._quiet_lookup(req_status, aligned_local)
        finally:
            req_status.num_locally_computed_tokens = orig_local
            req_status.partial_tail_boundary = orig_partial
        ext_tokens = int(ext) if ext is not None and ext > 0 else 0
        request._paw_ext_hit_cache = (
            num_hashes,
            aligned_local,
            now,
            ext_tokens,
        )
        return max(0, (aligned_local + ext_tokens) - num_local_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, RaidenOffloadingWorkerMetadata) and meta.fenced_jobs:
            ready_jobs: list[tuple[int, _ParkedStore]] = []
            for job_id, count in meta.fenced_jobs.items():
                parked = self._fence_pending.get(job_id)
                if parked is None:
                    # Cancelled while the acks were in flight.
                    continue
                parked.acks += count
                assert parked.acks <= self.config.num_workers
                if parked.acks == self.config.num_workers:
                    del self._fence_pending[job_id]
                    ready_jobs.append((job_id, parked))
            ready_jobs.sort(key=lambda item: (-len(item[1].keys), item[0]))
            tel = self._raiden_manager.telemetry
            for job_id, parked in ready_jobs:
                fa = sum(
                    1
                    for key in parked.keys
                    if get_offload_group_idx(key) in self._fa_group_ids
                )
                tel["split_fa_keys"] += fa
                tel["split_mamba_keys"] += len(parked.keys) - fa
                self._raiden_manager.submit_store_job(
                    job_id, parked.keys, parked.device_block_ids
                )
        super().update_connector_output(connector_output)

    def _quiet_lookup(self, req_status: Any, base_tokens: int) -> int | None:
        """Probe ``self._lookup(req_status)`` starting from ``base_tokens``
        without emitting duplicate lookup events."""
        req_status.num_locally_computed_tokens = base_tokens
        tracker = getattr(self, "_events_tracker", None)
        orig_record = (
            getattr(tracker, "record_lookup", None) if tracker else None
        )
        orig_partial = (
            getattr(tracker, "record_partial_lookup", None) if tracker else None
        )
        try:
            if tracker is not None:
                tracker.record_lookup = lambda *a, **kw: None
                tracker.record_partial_lookup = lambda *a, **kw: None
            return self._lookup(req_status)
        finally:
            if tracker is not None:
                if orig_record is not None:
                    tracker.record_lookup = orig_record
                if orig_partial is not None:
                    tracker.record_partial_lookup = orig_partial

    def resolve_divergent_local_hit_tokens(
        self,
        request: Request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        """Return the block-aligned local FullAttention prefix length to retain
        in HBM when HBM has a divergent hybrid hit (FullAttention > Mamba), or
        None if Raiden CPU cannot advance past ``num_reconciled``."""
        if getattr(request, "skip_reading_prefix_cache", False):
            return None
        req_status = self._req_status.get(request.request_id)
        if req_status is None:
            self.on_new_request(request)
            req_status = self._req_status.get(request.request_id)
        if req_status is None:
            return None
        if req_status.transfer_jobs:
            return (num_local // block_size) * block_size

        req_status.update_offload_keys()
        fa_aligned = (num_local // block_size) * block_size
        reconciled_aligned = (num_reconciled // block_size) * block_size

        # 1. Check if Raiden CPU has tokens strictly above fa_aligned
        #    (e.g. a partial tail at fa_aligned + 128 or additional chunks).
        ext_above = self._quiet_lookup(req_status, fa_aligned)
        if ext_above is None:
            return fa_aligned
        if ext_above > 0 and fa_aligned + ext_above > num_reconciled:
            return fa_aligned

        # 2. Common case: FullAttention in HBM covers all full blocks of the
        #    previous turn (or its partial tail), and Raiden CPU holds the
        #    Mamba boundary in that final block [fa_aligned - block_size, fa_aligned].
        #    Back off by 1 block so get_num_new_matched_tokens loads only the
        #    final block (1 FA block + 1 Mamba block) while keeping the other
        #    (fa_aligned // block_size - 1) FA blocks in HBM.
        one_back = fa_aligned - block_size
        if one_back >= reconciled_aligned and one_back >= 0:
            ext_last = self._quiet_lookup(req_status, one_back)
            if ext_last is None:
                return one_back
            if ext_last > 0 and one_back + ext_last > num_reconciled:
                ext_end = one_back + ext_last
                return min(fa_aligned, ((ext_end - 1) // block_size) * block_size)

        # 3. General case: FullAttention in HBM extends past the latest Mamba
        #    boundary stored in Raiden CPU (e.g. shared prefix from a longer
        #    request). Scan downward from one_back - block_size to
        #    reconciled_aligned to find the highest chunk where Raiden CPU can
        #    supply both the Mamba boundary and its corresponding FA chunk.
        min_chunk = max(0, reconciled_aligned // block_size)
        max_chunk = (one_back // block_size) - 1
        for chunk_idx in range(max_chunk, min_chunk - 1, -1):
            base = chunk_idx * block_size
            ext_hit = self._quiet_lookup(req_status, base)
            if ext_hit is None:
                return base
            if ext_hit > 0 and base + ext_hit > num_reconciled:
                ext_end = base + ext_hit
                return min(fa_aligned, ((ext_end - 1) // block_size) * block_size)

        return None


class TPURaidenOffloadingConnector(OffloadingConnector):
    """Top-level facade connector for the store-backed (V2) raiden KV offload path."""

    supports_divergent_local_hybrid_hits: bool = True

    def resolve_divergent_local_hit_tokens(
        self,
        request: Request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        if self.connector_scheduler is None:
            return None
        return self.connector_scheduler.resolve_divergent_local_hit_tokens(
            request, num_local, num_reconciled, block_size
        )

    def estimate_external_hit_tokens(
        self, request: Request, num_local_tokens: int
    ) -> int:
        if self.connector_scheduler is None:
            return 0
        return self.connector_scheduler.estimate_external_hit_tokens(
            request, num_local_tokens
        )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """No layout preference, unlike the `OffloadingConnector` this extends.

        Upstream hardcodes "HND" for NIXL on CUDA. On TPU "HND" names
        SEQ_ALONG_LANE, so inheriting it would select a layout nobody asked
        for; these pages are opaque bytes to this connector.
        """
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Opt in to uniform cross-layer KV cache allocation when block-major layout is
        # enabled. Satisfies `KVConnectorModelRunnerMixin.use_uniform_kv_cache()`
        # preconditions to enable bundled tensor allocation in the TPU model runner.
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Skip OffloadingConnector.__init__: default factory cannot carry
        # vllm_config / kv_cache_config directly into the spec.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # The raiden singleton worker cannot coexist with this connector: the
        # disagg stack runs its own control plane in the same process and the
        # two collide on ports.
        if not tpu_envs.RAIDEN_DISABLE_SINGLETON_WORKER:
            raise ValueError(
                "TPURaidenOffloadingConnector requires the raiden singleton "
                "worker to be disabled (it collides with the in-process "
                "control plane's ports), but RAIDEN_DISABLE_SINGLETON_WORKER "
                "was explicitly set to off. Unset it or set it to '1'."
            )
        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"

        # Call through module to preserve PCP-aware patch on build_offloading_config.
        offloading_config = offloading_config_module.build_offloading_config(
            vllm_config, kv_cache_config
        )
        spec = TPURaidenStoreOffloadingSpec(
            offloading_config, vllm_config, kv_cache_config
        )

        # pyrefly: ignore[bad-override-mutable-attribute]
        self.connector_scheduler: TPURaidenOffloadingScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPURaidenOffloadingScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            # Worker wrapper is only used for KV buffer registration; job dicts
            # remain empty.
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )
            # Track per-step fence acknowledgments and completion echoes for
            # worker metadata.
            self._fenced_jobs: dict[int, int] = {}
            self._completed_jobs: dict[int, int] = {}
            self._load_error_block_ids: set[int] = set()
            # Registered KV pool byte views whose deferred writes are dispatched before
            # recording fence events.
            self._pool_sync_tensors: list[torch.Tensor] = []
            # In-flight fence events awaiting device retirement before acknowledging
            # store jobs, recorded as pairs of (fence_job_ids, event).
            self._pending_fences: list[tuple[set[int], torch.Event]] = []
            # QX_OFFLOAD_FENCE_AFTER_DRAFT=1: the runner calls get_finished()
            # after the target forward but BEFORE the speculative (MTP/eagle)
            # draft propose, which writes the draft layer's KV into the same
            # pool blocks. Deferring the fence for a step's store jobs to the
            # next get_finished() call orders it after every KV write of that
            # step, draft included.
            self._fence_after_draft: bool = (
                os.environ.get("QX_OFFLOAD_FENCE_AFTER_DRAFT", "0") == "1"
            )
            self._deferred_fence_ids: set[int] = set()
            if self._fence_after_draft:
                logger.info(
                    "TPURaidenOffloadingConnector: QX_OFFLOAD_FENCE_AFTER_DRAFT=1: "
                    "store fences are recorded one get_finished() call later so "
                    "they order after the step's draft (MTP) KV writes"
                )

    # -- scheduler-side hooks ----------------------------------------------
    def reset_cache(self) -> bool | None:
        # External cache reset is unsupported by the Raiden store. Report
        # failure through the endpoint's documented channel instead of raising an
        # exception here
        logger.warning(
            "TPURaidenOffloadingConnector: external cache reset is not "
            "supported by the Raiden offload store; reporting failure."
        )
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.take_events()
        return []

    def poll_and_complete_transfers(self, scheduler: Any) -> tuple[int, int, int, bool]:
        if isinstance(self.connector_scheduler, TPURaidenOffloadingScheduler):
            return self.connector_scheduler.poll_and_complete_transfers(scheduler)
        return (0, 0, 0, False)

    # -- worker-side hooks -------------------------------------------------
    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]]
    ):
        """Register the TPU UBP as zero-copy canonical offload tensors.

        vLLM's generic per-layer registration does not support the TPU
        unified block pool layout or Mamba tensor lists, so this
        implementation deduplicates the UBP backing storages and exposes
        each as an `int8` byte view.
        """
        assert self.connector_worker is not None
        num_blocks = self.connector_worker.kv_cache_config.num_blocks

        pool_storages: dict[int, tuple[torch.UntypedStorage, torch.device]] = {}
        for layer_kv_cache in kv_caches.values():
            state_tensors = (
                [layer_kv_cache]
                if isinstance(layer_kv_cache, torch.Tensor)
                else layer_kv_cache
            )
            for tensor in state_tensors:
                storage = tensor.untyped_storage()
                pool_storages.setdefault(storage.data_ptr(), (storage, tensor.device))

        pool_tensors: list[CanonicalKVCacheTensor] = []
        for storage, device in pool_storages.values():
            storage_bytes = storage.nbytes()
            assert storage_bytes % num_blocks == 0, (
                "KV pool storage must hold whole scheduler blocks: "
                f"storage_bytes={storage_bytes}, num_blocks={num_blocks}"
            )
            page_size_bytes = storage_bytes // num_blocks
            int8_view = (
                torch.empty(0, dtype=torch.int8, device=device)
                .set_(storage)
                .view(num_blocks, page_size_bytes)
            )
            pool_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=int8_view, page_size_bytes=page_size_bytes
                )
            )

        # Retain pool tensor references to dispatch their writes during save fences.
        self._pool_sync_tensors = [pool_tensor.tensor for pool_tensor in pool_tensors]

        group_data_refs = [
            [
                CanonicalKVCacheRef(
                    tensor_idx=idx, page_size_bytes=pool_tensor.page_size_bytes
                )
                for idx, pool_tensor in enumerate(pool_tensors)
            ]
            for _ in self.connector_worker.kv_cache_config.kv_cache_groups
        ]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors, group_data_refs=group_data_refs)
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, RaidenOffloadingConnectorMetadata)

        fence_job_ids = set(meta.fence_job_ids)
        if getattr(self, "_fence_after_draft", False):
            fence_job_ids, self._deferred_fence_ids = (
                self._deferred_fence_ids,
                fence_job_ids,
            )

        if fence_job_ids:
            # Establish an asynchronous device fence before acknowledging store jobs:
            # 1. Dispatch any deferred pool writes so they precede the fence event.
            # 2. Record a stream event to track completion without host-side blocking.
            assert self._pool_sync_tensors, (
                "KV pool tensors must be registered before the save fence"
            )
            _enqueue_pool_writes(self._pool_sync_tensors)
            self._pending_fences.append(
                (fence_job_ids, _record_fence_event())
            )

        if self._pending_fences:
            # Non-blockingly drain retired events and acknowledge completed fences.
            still_pending: list[tuple[set[int], torch.Event]] = []
            for job_ids, event in self._pending_fences:
                if _event_is_retired(event):
                    for job_id in job_ids:
                        self._fenced_jobs[job_id] = 1
                else:
                    still_pending.append((job_ids, event))
            self._pending_fences = still_pending

        finished_recving: set[str] = set()
        for job_id in meta.finished_store_job_ids:
            self._completed_jobs[job_id] = 1
        for job_id, req_id in meta.finished_load_jobs.items():
            self._completed_jobs[job_id] = 1
            finished_recving.add(req_id)
        self._load_error_block_ids.update(meta.failed_load_device_block_ids)

        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids = self._load_error_block_ids
        self._load_error_block_ids = set()
        return block_ids

    def get_block_ids_with_load_errors_group_index(self) -> int | None:
        """Returns None: offload load errors are not scoped to one KV group.

        The runner reads this whenever ``get_block_ids_with_load_errors``
        returns a non-empty set. Failed device block ids come from the
        scheduler-side block table and resolve against every cache group,
        so there is no single group index to report.
        """
        return None

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if not (self._fenced_jobs or self._completed_jobs):
            return None
        # completed_jobs is inherited from vLLM base dataclass.
        meta = RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=self._completed_jobs,
            fenced_jobs=self._fenced_jobs,
        )
        self._completed_jobs = {}
        self._fenced_jobs = {}
        return meta
