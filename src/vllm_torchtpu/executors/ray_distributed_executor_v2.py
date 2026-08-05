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

import os
import threading
import time
import weakref
from collections import defaultdict, deque
from typing import Any, Dict, List

import ray
import vllm.envs as envs
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.platforms import current_platform
from vllm.utils.network_utils import (get_distributed_init_method, get_ip,
                                      get_open_port)
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.executor.ray_env_utils import get_driver_env_vars
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2, RayWorkerHandle
from vllm.v1.executor.ray_utils import (WORKER_SPECIFIC_ENV_VARS,
                                        _wait_until_pg_ready, build_actor_name,
                                        get_bundles_for_indices,
                                        get_bundles_sorted_by_node)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.distributed.utils import set_node_kv_ip_port
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import TPU_MULTIHOST_TOPOLOGY_MAP
from vllm_torchtpu.utils import get_dp_size
from vllm_torchtpu.worker.tpu_rank_binding import slice_binding_env

logger = init_logger(__name__)

# How long a DP engine waits for a peer to publish the shared slice placement
# group before giving up.
_SLICE_PG_TIMEOUT_S = float(os.environ.get("TPU_SLICE_PG_TIMEOUT_S", "600.0"))
_SLICE_PG_POLL_INTERVAL_S = 0.5


class RayDistributedExecutorV2(RayExecutorV2):
    """Ray-based distributed executor V2 for TPU.

    Inherits from vLLM V1's RayExecutorV2, leveraging the high-performance
    MessageQueue-based control plane, while keeping TPU-specific cluster
    initialization and PyTorch TPU/XLA parallelisms.
    """

    # Set when this engine is the one that created the shared slice placement
    # group, so only its owner tears it down.
    _created_slice_pg: bool = False
    _slice_placement_group: Any = None

    def _get_dp_geometry(self) -> tuple[int, int, int]:
        """Return (dp_size, dp_rank, slice_world_size) for this engine.

        vLLM runs one executor per DP engine, each sized `world_size` =
        TP x PP x PCP. TorchTPU instead needs the whole DP*TP grid bootstrapped
        as a single slice, so the executor reasons in slice-global terms and
        carves out the window of ranks belonging to this engine.
        """
        # Read the per-engine width off ParallelConfig rather than
        # self.world_size: placement group setup needs the geometry before
        # _get_parallel_sizes() has populated the executor attribute.
        parallel_config = self.parallel_config
        world_size = parallel_config.world_size
        dp_size = get_dp_size(parallel_config)
        if dp_size <= 1:
            return 1, 0, world_size

        dp_rank = parallel_config.data_parallel_index
        if dp_rank is None:
            dp_rank = parallel_config.data_parallel_rank
        if dp_rank is None:
            raise ValueError(
                "ParallelConfig.data_parallel_index must be resolved when "
                f"data_parallel_size={dp_size}.")
        return dp_size, int(dp_rank), world_size * dp_size

    def _slice_placement_group_name(self) -> str:
        """Return the name of the placement group shared by all DP engines.

        The DP coordinator rendezvous is the one identifier every engine of
        this server already agrees on before any of them talks to Ray, so it
        keys the group without needing a side channel.
        """
        parallel_config = self.parallel_config
        return (f"vllm-tpu-slice-{parallel_config.data_parallel_master_ip}-"
                f"{parallel_config.data_parallel_master_port}")

    def _get_or_create_shared_placement_group(
            self, placement_group_specs: List[Dict[str, float]]):
        """Create, or attach to, the placement group shared by all DP engines.

        Every DP engine must see the same bundles in the same order so that a
        slice-global rank means the same chip everywhere, and only one of them
        may claim the cluster's chips. Whichever engine gets there first
        creates the group; the rest attach to it by name.
        """
        name = self._slice_placement_group_name()
        expected_bundles = len(placement_group_specs)
        deadline = time.monotonic() + _SLICE_PG_TIMEOUT_S

        while True:
            try:
                existing = ray.util.get_placement_group(name)
            except ValueError:
                existing = None

            if existing is not None:
                actual_bundles = len(existing.bundle_specs)
                if actual_bundles != expected_bundles:
                    raise RuntimeError(
                        f"Ray placement group {name!r} has {actual_bundles} "
                        f"bundles but this slice needs {expected_bundles}. "
                        "It is most likely a leftover from a differently "
                        "shaped run; remove it with "
                        "`ray.util.remove_placement_group("
                        f"ray.util.get_placement_group({name!r}))` and retry.")
                logger.info(
                    "RayDistributedExecutorV2 | Attached to shared slice "
                    "placement group %s (%d bundles).", name, actual_bundles)
                return existing

            try:
                created = ray.util.placement_group(
                    placement_group_specs,
                    strategy="PACK",
                    name=name,
                    # Outlives the creating engine's driver so peers that start
                    # later can still attach; removed again in shutdown().
                    lifetime="detached",
                )
            except ValueError as exc:
                # Ray reports both "name already taken" and a malformed
                # request as ValueError. Only the former is retryable, and it
                # is exactly the case where the group now exists.
                try:
                    ray.util.get_placement_group(name)
                except ValueError:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out resolving shared slice placement group "
                        f"{name!r} after {_SLICE_PG_TIMEOUT_S}s.") from exc
                logger.info(
                    "RayDistributedExecutorV2 | Slice placement group %s was "
                    "created by a peer engine; attaching to it instead.", name)
                time.sleep(_SLICE_PG_POLL_INTERVAL_S)
                continue

            self._created_slice_pg = True
            self._slice_placement_group = created
            logger.info(
                "RayDistributedExecutorV2 | Created shared slice placement "
                "group %s (%d bundles).", name, expected_bundles)
            return created

    def _remove_slice_placement_group(self) -> None:
        placement_group = self._slice_placement_group
        if not self._created_slice_pg or placement_group is None:
            return
        self._created_slice_pg = False
        self._slice_placement_group = None
        try:
            ray.util.remove_placement_group(placement_group)
        except Exception:
            logger.exception(
                "RayDistributedExecutorV2 | Failed to remove shared slice "
                "placement group %s.", self._slice_placement_group_name())

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            # Detached groups survive their creator, so the engine that made
            # this one must release the cluster's chips explicitly.
            self._remove_slice_placement_group()

    def _get_actor_resource_kwargs(self) -> dict[str, Any]:
        """Return Ray actor resource kwargs for the TPU platform.

        For torchtpu-vllm, we always use 1 TPU per worker similar to
        Multiprocexecutor.
        """
        device_key = current_platform.ray_device_key
        return {"num_gpus": 0, "resources": {device_key: 1.0}}

    def _init_executor(self) -> None:
        """Initialize the RayDistributedExecutorV2 executor.

        NOTE(ranlihao): This method shares the exact same 10-step structure as
        vLLM's base RayExecutorV2._init_executor(). The main differences are:
          1. We use a custom, TPU-optimized placement group initialization.
          2. We construct and propagate a rich set of TorchTPU multi-host
             environment variables (e.g. RANK, LOCAL_RANK, WORLD_SIZE,
             TORCH_TPU_SLICEBUILDER_ADDRESSES, etc.) as part of the worker's
             env_vars.

        TODO(ranlihao): Eventually, we should add a hook in the upstream vLLM
        RayExecutorV2 to customize/extend the placement group creation and
        the worker env_vars. This will allow us to completely delete this
        duplicated _init_executor and reuse the upstream implementation.
        """
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None
        self.shutting_down = False
        self.shutdown_lock = threading.Lock()

        # Step 1: Initialize Ray cluster and retrieve placement group
        if ray is None:
            raise ImportError(
                "Using Ray backend requires installation of ray.")

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). ")

        dp_size, dp_rank, slice_world_size = self._get_dp_geometry()
        if dp_size > 1 and pcp_size > 1:
            raise NotImplementedError(
                "Prefill context parallelism is not supported together with "
                "multihost data parallelism.")

        self._initialize_ray_cluster()
        placement_group = self.parallel_config.placement_group

        # Step 2: Build bundle assignments for worker rank placement
        # while respecting VLLM_RAY_BUNDLE_INDICES. Bundles are ordered
        # identically in every DP engine, so bundle i is slice-global rank i
        # everywhere and this engine owns one contiguous window of them.
        if envs.VLLM_RAY_BUNDLE_INDICES:
            bundle_to_node_id = get_bundles_for_indices(
                placement_group,
                list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(","))),
                slice_world_size,
            )
        else:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)

        if len(bundle_to_node_id) < slice_world_size:
            raise ValueError(
                f"Placement group provides {len(bundle_to_node_id)} bundles "
                f"but the TPU slice needs {slice_world_size} "
                f"(world_size={self.world_size} x "
                f"data_parallel_size={dp_size}).")

        driver_node = ray.get_runtime_context().get_node_id()

        # Placement of every rank in the slice, not just this engine's: the
        # TorchTPU bootstrap env below describes the whole DP*TP slice.
        slice_assignments: list[dict[str, Any]] = []
        for slice_rank, (bundle_id_idx, node_id, node_ip) in enumerate(
                bundle_to_node_id[:slice_world_size]):
            slice_assignments.append({
                "slice_rank": slice_rank,
                "bundle_id_idx": bundle_id_idx,
                "node_id": node_id,
                "node_ip": node_ip,
            })

        # This engine's shard of the slice. vLLM ranks stay per-engine
        # (0..world_size-1); the slice-global rank only ever reaches the worker
        # as environment, exactly as the single-host DP spawn shim does.
        rank_offset = dp_rank * self.world_size
        bundle_assignments: list[dict[str, Any]] = [{
            "rank": rank,
            **assignment
        } for rank, assignment in enumerate(
            slice_assignments[rank_offset:rank_offset + self.world_size])]

        # Step 3: Resolve the IP for torch.distributed TCPStore.
        # The TCPStore server runs on rank 0's node, so all workers
        # must be able to reach this address.
        dist_ip = bundle_assignments[0]["node_ip"]
        port = self._select_tcpstore_port(
            self.parallel_config.data_parallel_rank_local,
            self.parallel_config.data_parallel_master_port,
        )
        distributed_init_method = get_distributed_init_method(dist_ip, port)

        # Step 4: Create broadcast MessageQueue.
        # Workers on the driver node use shared memory; the rest use TCP.
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        n_local = sum(1 for a in bundle_assignments
                      if a["node_id"] == driver_node)
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            n_local,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=ray.util.get_node_ip_address(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Step 5: Spawn RayWorkerProc actors into PG bundles.
        # Use the standard RayWorkerProc.
        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

        self.ray_worker_handles: list[RayWorkerHandle] = []
        instance_id = self.vllm_config.instance_id

        # Collect driver env vars
        self.driver_env_vars = get_driver_env_vars(
            worker_specific_vars=WORKER_SPECIFIC_ENV_VARS, )

        runtime_env = self._build_runtime_env()
        resource_kwargs = self._get_actor_resource_kwargs()

        for bundle_idx in range(self.world_size):
            bundle = bundle_assignments[bundle_idx]
            is_driver_worker = self._is_driver_worker(bundle["rank"])
            is_driver_node = bundle["node_id"] == driver_node

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle["bundle_id_idx"],
            )

            actor_name = build_actor_name(instance_id, bundle["rank"], tp_size,
                                          pp_size, pcp_size)
            if dp_size > 1:
                # Every DP engine spawns the same per-engine ranks, so the
                # upstream name is only unique once the DP rank is in it.
                actor_name = f"{actor_name}_dp{dp_rank}"

            actor = (ray.remote(RayWorkerProc).options(
                name=actor_name,
                num_cpus=0,
                **resource_kwargs,
                scheduling_strategy=scheduling_strategy,
                runtime_env=runtime_env,
            ).remote(
                vllm_config=self.vllm_config,
                rank=bundle["rank"],
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
                is_driver_worker=is_driver_worker,
                is_driver_node=is_driver_node,
            ))

            handle = RayWorkerHandle(
                actor=actor,
                rank=bundle["rank"],
                local_rank=-1,  # Set in Step 7 after TPU ID discovery
                node_id=bundle["node_id"],
                bundle_id_idx=bundle["bundle_id_idx"],
            )
            self.ray_worker_handles.append(handle)

        # Step 6: Discover physical TPU IDs assigned to each worker via Ray.
        worker_node_and_physical_tpu_ids = ray.get([
            h.actor.get_node_and_physical_gpu_ids.remote()
            for h in self.ray_worker_handles
        ])

        node_physical_tpu_ids: dict[str, list[int]] = defaultdict(list)
        for node_id, physical_tpu_ids in worker_node_and_physical_tpu_ids:
            node_physical_tpu_ids[node_id].extend(physical_tpu_ids)
        for node_id, physical_tpu_ids in node_physical_tpu_ids.items():
            node_physical_tpu_ids[node_id] = sorted(physical_tpu_ids)

        # Step 7: Prepare the environment variables for TorchTPU/XLA.
        # This includes construction of slice builder addresses, topology
        # lookup, and assigning rank/local_rank for the unified multi-host
        # slice. All of it is derived from the slice-wide assignment so that
        # every DP engine computes an identical view and merely reads different
        # entries out of it.
        host_slice_ranks: dict[str, list[int]] = defaultdict(list)
        host_order: list[str] = []
        for assignment in slice_assignments:
            node_ip = assignment["node_ip"]
            if node_ip not in host_slice_ranks:
                host_order.append(node_ip)
            host_slice_ranks[node_ip].append(assignment["slice_rank"])

        # Position of each slice rank among the slice's chips on its own host.
        host_local_rank: dict[int, int] = {}
        for ip in host_order:
            for local_rank, slice_rank in enumerate(host_slice_ranks[ip]):
                host_local_rank[slice_rank] = local_rank

        base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
        # Indexed by slice rank, so entry i is the address of rank i however
        # the bundles happen to be ordered across hosts.
        sb_addresses = [
            f"{a['node_ip']}:{base_port + host_local_rank[a['slice_rank']]}"
            for a in slice_assignments
        ]
        slicebuilder_addresses = ",".join(sb_addresses)
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = slicebuilder_addresses
        logger.info(
            f"RayDistributedExecutorV2 | Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {slicebuilder_addresses}"
        )

        node_rank_by_ip = {ip: rank for rank, ip in enumerate(host_order)}
        num_nodes = len(host_order)
        master_addr = slice_assignments[0]["node_ip"]
        if dp_size > 1:
            # Every worker of every DP engine joins one TorchTPU world, so the
            # rendezvous port must be agreed on without a side channel. The
            # slicebuilder ports occupy [base_port, base_port + chips per host)
            # so the first port past that range is free by construction.
            max_chips_per_host = max(
                len(ranks) for ranks in host_slice_ranks.values())
            master_port = (os.environ.get("TORCH_TPU_DP_MASTER_PORT")
                           or str(base_port + max_chips_per_host))
        else:
            master_port = str(get_open_port())

        topology = TPU_MULTIHOST_TOPOLOGY_MAP.get(slice_world_size, None)
        if topology is None:
            raise ValueError(
                f"Cannot find topology for {slice_world_size} chips "
                f"(world_size={self.world_size} x "
                f"data_parallel_size={dp_size}). The supported number of "
                f"chips are {list(TPU_MULTIHOST_TOPOLOGY_MAP.keys())}")
        xprof_session_id = os.environ.get("TORCH_TPU_XPROF_SESSION_ID",
                                          str(time.time_ns()))
        os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id

        # Initialize workers with correct environment variables and local_rank.
        init_worker_refs = []
        for i, (node_id, _) in enumerate(worker_node_and_physical_tpu_ids):
            assignment = bundle_assignments[i]
            if node_id != assignment["node_id"]:
                raise RuntimeError(
                    f"Worker {i} started on Ray node {node_id} but its bundle "
                    f"was placed on {assignment['node_id']}; the slice rank "
                    "to chip mapping would be wrong.")
            node_ip = assignment["node_ip"]
            slice_rank = assignment["slice_rank"]
            local_rank = host_local_rank[slice_rank]
            local_world_size = len(host_slice_ranks[node_ip])
            node_rank = node_rank_by_ip[node_ip]
            assigned_physical_tpu_ids = sorted(node_physical_tpu_ids[node_id])

            worker_env_vars = {
                "LOCAL_WORLD_SIZE": str(local_world_size),
                "NNODES": str(num_nodes),
                "NODE_RANK": str(node_rank),
                "TPU_NUM_HOSTS": str(num_nodes),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "TORCH_TPU_TOPOLOGY": topology,
                "TORCH_TPU_XPROF_SESSION_ID": xprof_session_id,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
            }
            if dp_size > 1:
                # TPUWorker rebuilds MASTER_ADDR/PORT from the DP pair, and
                # TORCH_TPU_DP_SIZE is what makes get_dp_size() in the worker
                # agree with the slice the executor actually built.
                worker_env_vars.update({
                    "TORCH_TPU_DP_SIZE": str(dp_size),
                    "TORCH_TPU_DP_MASTER_ADDR": master_addr,
                    "TORCH_TPU_DP_MASTER_PORT": master_port,
                })
                worker_env_vars.update(
                    slice_binding_env(
                        rank=slice_rank,
                        local_rank=local_rank,
                        world_size=slice_world_size,
                        local_world_size=local_world_size,
                    ))
            self.ray_worker_handles[i].local_rank = local_rank
            logger.info(
                "RayDistributedExecutorV2 | Worker rank=%d dp_rank=%d -> "
                "slice_rank=%d local_rank=%d node=%s (node_rank=%d)", i,
                dp_rank, slice_rank, local_rank, node_ip, node_rank)
            # Print all environment variables that will be set on the worker
            combined_env = {**self.driver_env_vars, **worker_env_vars}
            logger.debug(
                f"RayDistributedExecutorV2 | Worker {i} (slice rank {slice_rank}) environment variables: {combined_env}"
            )
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    local_rank,
                    worker_env_vars,
                    self.driver_env_vars,
                    assigned_physical_gpu_ids=assigned_physical_tpu_ids,
                ))
        if len(node_physical_tpu_ids) == 1:
            node_id_0 = worker_node_and_physical_tpu_ids[0][0]
            self.vllm_config.parallel_config.assigned_physical_gpu_ids = sorted(
                node_physical_tpu_ids[node_id_0])
        ray.get(init_worker_refs)

        # Step 8: Collect response MQ handles
        init_results = ray.get(
            [h.actor.wait_for_init.remote() for h in self.ray_worker_handles])

        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != RayWorkerProc.READY_STR:
                raise RuntimeError(
                    f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0))

        # Step 9: Start run() before wait_until_ready() to avoid deadlock
        for handle in self.ray_worker_handles:
            handle.run()

        # Step 10: wait_until_ready() barrier
        self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        self.futures_queue = deque[FutureWrapper]()
        self._post_init_executor()

        self.start_worker_monitor()
        self.output_rank = self._get_output_rank()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        if self.has_connector:
            ip_port = self.collective_rpc("get_node_kv_ip_port")
            for item in ip_port:
                set_node_kv_ip_port(item)

    def _initialize_ray_cluster(self) -> None:
        """Initialize the distributed cluster with Ray.

        Creates a TPU-optimized placement group where all chips on a node
        are packed into a single bundle, instead of 1 GPU per bundle.
        """
        # Check if placement group is already provided (e.g., by ray.serve.llm)
        if self.parallel_config.placement_group is not None:
            logger.info(
                f"Using existing placement group: {self.parallel_config.placement_group}"
            )
            return

        if ray.is_initialized():
            logger.info(
                "Ray is already initialized. Skipping Ray initialization.")
        else:
            logger.warning("Ray is not initialized, this is mainly for test.")
            ray.init()

        device_str = current_platform.ray_device_key
        if not device_str:
            raise ValueError(
                f"current platform {current_platform.device_name} does not "
                "support ray.")

        pp_size = self.parallel_config.pipeline_parallel_size
        placement_group_specs: List[Dict[str, float]] = []

        ray_nodes = ray.nodes()
        logger.info(f"RayDistributedExecutorV2 | ray_nodes={ray_nodes}")

        # Filter nodes that have the required TPU resource
        nodes_with_device = [
            n for n in ray_nodes if device_str in n.get("Resources", {})
        ]
        logger.info(
            f"RayDistributedExecutorV2 | nodes_with_device={len(nodes_with_device)} "
            f"(filtered from {len(ray_nodes)} total nodes)")

        if pp_size == 1:
            placement_group_specs = []
            for node in nodes_with_device:
                num_devices = int(node['Resources'][device_str])
                for _ in range(num_devices):
                    placement_group_specs.append({device_str: 1.0})
        else:
            assert pp_size == len(
                nodes_with_device
            ), f"Cannot use PP across hosts, please set --pipeline-parallel-size to 1 or {len(nodes_with_device)}"
            num_devices_per_pp_rank = self.parallel_config.world_size // pp_size
            placement_group_specs = [{
                device_str: num_devices_per_pp_rank
            } for _ in range(pp_size)]

        # Bind the first bundle to the current node (vLLM engine node)
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_info = next(
            (n for n in ray.nodes() if n["NodeID"] == current_node_id), None)
        current_node_resource = (current_node_info.get("Resources", {})
                                 if current_node_info else {})
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")

        # Ensure the first bundle is created on the current node
        placement_group_specs[0][f"node:{current_ip}"] = 0.001
        logger.info(
            f"RayDistributedExecutorV2 | placement_group_specs={placement_group_specs}"
        )

        # By default, Ray packs resources as much as possible.
        dp_size, _, _ = self._get_dp_geometry()
        if dp_size > 1:
            # One group for the whole slice, shared by all DP engines: they
            # must agree on bundle ordering, and only one of them can claim
            # the cluster's chips.
            current_placement_group = (
                self._get_or_create_shared_placement_group(
                    placement_group_specs))
        else:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        self.parallel_config.placement_group = current_placement_group

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()

        if self.vllm_config.cache_config.num_gpu_blocks_override is None:
            # Compact-mamba sizing sets `num_gpu_blocks_override` on the worker's
            # cache_config during the RPC above; workers are separate processes, so
            # copy it to the engine-side config here.
            overrides = self.collective_rpc("get_num_gpu_blocks_override")
            assert len(set(overrides)) == 1
            self.vllm_config.cache_config.num_gpu_blocks_override = (
                overrides[0])

        return specs

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p, ))
