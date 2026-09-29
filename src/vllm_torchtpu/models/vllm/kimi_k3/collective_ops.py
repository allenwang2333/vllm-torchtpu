# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridges for K3 collectives."""

import hashlib
from functools import lru_cache

import jax
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import get_tp_group

from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_rank_order, get_ep_group
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

from vllm_torchtpu.kernels.kimi_k3.collectives import (  # isort: skip
    gather_moe_inputs,
    gather_project,
    mla_chip_layout,
    mla_preprocess,
    pack_mla_gate,
    project_reduce_scatter,
    ring_collective,
    ring_tables,
)

_OPS = {}


def _build_mla_ops(mesh, owners, layer, eps):
    key = (
        "mla_preprocess",
        layer,
        eps,
        tuple(d.id for d in mesh.devices.flat),
        tuple(owners),
    )
    if key in _OPS:
        return _OPS[key]
    pairs, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)

    def pack(weight: jax.Array) -> jax.Array:
        return pack_mla_gate(weight, mesh, pairs)

    pack_op = sharded_jax_op(
        f"pallas::kimi_mla_pack_gate_{layer}",
        pack,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P("k3_sp", None),
    )
    pack_op.register_fake(lambda weight: weight.new_empty((7168, 6144)))

    def preprocess(
        x: jax.Array,
        weight: jax.Array,
        packed_gate: jax.Array,
        qnorm: jax.Array,
        kvnorm: jax.Array,
        sequence_parallel: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return mla_preprocess(
            x,
            weight,
            packed_gate,
            qnorm,
            kvnorm,
            mesh,
            owners,
            sequence_parallel=sequence_parallel,
            eps=eps,
        )

    op = sharded_jax_op(
        f"pallas::kimi_mla_preprocess_{layer}",
        preprocess,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),) * 3 + (P(), P(), P()),
        output_partition_specs=(P("k3_sp", None),) * 4,
    )

    def fake(x, weight, packed_gate, qnorm, kvnorm, sequence_parallel):
        rows = x.shape[0] * 32 if sequence_parallel else x.shape[0]
        return tuple(x.new_empty((rows, width)) for width in (1536, 512, 64, 384))

    op.register_fake(fake)
    _OPS[key] = pack_op, op
    return pack_op, op


def _build_moe_gather_op(mesh, tables, collective_id):
    key = (
        "moe_gather",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def gather(
        x: jax.Array, weights: jax.Array, ids: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return gather_moe_inputs(
            x, weights, ids, mesh, tables, collective_id=collective_id
        )

    op = sharded_jax_op(
        f"pallas::kimi_moe_ag_{collective_id}",
        gather,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),) * 3,
        output_partition_specs=(P(),) * 3,
    )

    def fake(x, weights, ids):
        return tuple(
            a.new_empty((a.shape[0] * 32, a.shape[1])) for a in (x, weights, ids)
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_op(mesh, tables, gather, collective_id):
    key = (
        gather,
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def collective(x: jax.Array) -> jax.Array:
        # The bridge assigns a partitioned global shape to this rank's input.
        # RS represents each rank's full-token partial as one leading slice.
        if not gather:
            x = x.reshape(32, x.shape[0] // 32, x.shape[1])
        return ring_collective(
            x, mesh, tables, gather=gather, collective_id=collective_id
        )

    name = f"pallas::kimi_ring_{'ag' if gather else 'rs'}_{collective_id}"
    op = sharded_jax_op(
        name,
        collective,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P() if gather else P("k3_sp", None),
    )

    def fake(x):
        rows = x.shape[0] * 32 if gather else x.shape[0] // 32
        return torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_project_op(mesh, tables, collective_id):
    key = (
        "project",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        return project_reduce_scatter(
            x.reshape(32, x.shape[0] // 32, 384),
            w.reshape(32, 384, 7168),
            mesh,
            tables,
            collective_id=collective_id,
        )

    op = sharded_jax_op(
        f"pallas::kimi_project_rs_{collective_id}",
        project,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None), P("k3_sp", None)),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x, w):
        return torch.empty(
            (x.shape[0] // 32, w.shape[1]), dtype=x.dtype, device=x.device
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_gather_project_op(mesh, tables, collective_id):
    key = (
        "gather_project",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        result = gather_project(
            x, w.reshape(32, 7168, 1792), mesh, tables, collective_id=collective_id
        )
        return result.reshape(-1, 1792)

    op = sharded_jax_op(
        f"pallas::kimi_gather_project_{collective_id}",
        project,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None), P("k3_sp", None)),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x, w):
        return torch.empty(
            (x.shape[0] * 32, w.shape[1]), dtype=x.dtype, device=x.device
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


@lru_cache(maxsize=1)
def _mesh_and_tables():
    ep = get_ep_group()
    base = get_tp_group()
    if ep is None or list(ep.ranks) != list(base.ranks):
        raise ValueError("K3 rings require identical TP32 and EP32 rank order")
    mesh = build_ep_mesh(axis_name="k3_sp")
    owners = ep_rank_order()
    return mesh, ring_tables(list(mesh.devices.flat), owners)


@lru_cache(maxsize=4)
def _build_token_op(world_size, gather):
    mesh, tables = _mesh_and_tables()
    owners = tuple(int(i) for i in tables[0, 2])
    if world_size == 2:
        groups, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)
    elif world_size == 32:
        groups = (tuple(sorted(range(32), key=lambda i: owners[i])),)
    else:
        raise ValueError("K3 SP token collectives require TP2 or TP32")

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            if gather:
                return jax.lax.all_gather(
                    part, "k3_sp", axis=0, tiled=True, axis_index_groups=groups
                )
            return jax.lax.psum_scatter(
                part, "k3_sp", scatter_dimension=0, tiled=True, axis_index_groups=groups
            )

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P("k3_sp", None),
            out_specs=P("k3_sp", None),
            check_vma=False,
        )(x)

    op = sharded_jax_op(
        f"pallas::kimi_token_{world_size}_{'ag' if gather else 'rs'}",
        collective,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x):
        rows = x.shape[0] * world_size if gather else x.shape[0] // world_size
        return x.new_empty((rows, x.shape[1]))

    op.register_fake(fake)
    return op


class TokenShardCollectives:
    """Gather/scatter token shards over TP2 or TP32 without concrete allocations."""

    def __init__(self, base):
        self.world_size = base.world_size
        self.rank_in_group = base.rank_in_group
        self.ag = _build_token_op(self.world_size, True)
        self.rs = _build_token_op(self.world_size, False)

    def all_gather(self, x, dim):
        assert dim == 0
        return self.ag(x)

    def reduce_scatter(self, x, dim):
        assert dim == 0
        return self.rs(x)


class FusedPrefillCollectives:
    """Fused prefill projections and MoE collectives, with standard fallbacks."""

    def __init__(self, prefix):
        self.base = get_tp_group()
        mesh, tables = _mesh_and_tables()
        # Fixed layer-derived IDs keep independent ring barriers distinct.
        components = prefix.split(".")
        layer = int(components[components.index("layers") + 1])
        self.layer = layer
        self.ag = _build_op(mesh, tables, True, 20000 + 2 * layer)
        self.rs = _build_op(mesh, tables, False, 20001 + 2 * layer)
        self.project_rs = _build_project_op(mesh, tables, 22000 + layer)
        self.gather_project = _build_gather_project_op(mesh, tables, 23000 + layer)
        self._moe_gather_op = _build_moe_gather_op(mesh, tables, 24000 + layer)

    def gather_moe_inputs(self, latent, weights, ids, *, num_experts):
        """Gather token activations and routes, packing compatible BF16 inputs."""
        weights, ids = weights.to(latent.dtype), ids.to(torch.int32)
        if num_experts <= 32768 and weights.dtype == torch.bfloat16:
            return self._moe_gather_op(latent, weights, ids)
        return tuple(self.all_gather(x, dim=0) for x in (latent, weights, ids))

    def mla_ops(self, eps):
        mesh, _ = _mesh_and_tables()
        return _build_mla_ops(mesh, ep_rank_order(), self.layer, eps)

    def all_gather(self, x, dim):
        if (
            dim == 0
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] in (2176, 3584, 7168)
            and x.dtype == torch.bfloat16
        ):
            return self.ag(x)
        return self.base.all_gather(x, dim=dim)

    def reduce_scatter(self, x, dim):
        if (
            dim == 0
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[0] % 32 == 0
            and x.shape[1] in (2176, 3584, 7168)
            and x.dtype == torch.bfloat16
        ):
            return self.rs(x)
        return self.base.reduce_scatter(x, dim=dim)

    def project_reduce_scatter(self, x, projection):
        # Preserve the linear method for other weight formats.
        weight = getattr(projection, "weight", None)
        if (
            x.shape[0] > 0
            and x.shape[0] % 32 == 0
            and x.shape[1] == 384
            and x.dtype == torch.bfloat16
            and weight is not None
            and weight.dtype == torch.bfloat16
            and weight.shape == (384, 7168)
            and getattr(projection, WEIGHT_FLIPPED_ATTR, False)
            and getattr(projection, "bias", None) is None
        ):
            return self.project_rs(x, weight)
        return self.reduce_scatter(projection(x)[0], dim=0)

    def all_reduce(self, x):
        return self.base.all_reduce(x)


_TWO_SHOT_CONTEXTS = {}
_KIMI_TWO_SHOT_BATCHES = {}


def prepare_two_shot_context(tp_group):
    """Resolve physical placement once, before compiling any serving bucket."""
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh

    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    ranks = tuple(int(rank) for rank in tp_group.ranks)
    if (
        not dist.is_initialized()
        or int(tp_group.world_size) != 32
        or ranks != tuple(range(32))
        or dist.get_world_size() != 32
    ):
        raise ValueError("K3 Two Shot requires a full-world, rank-ordered TP32 group")
    key = (tp_group.unique_name, ranks)
    if key in _TWO_SHOT_CONTEXTS:
        result = _TWO_SHOT_CONTEXTS[key]
        if isinstance(result, str):
            raise ValueError(result)
        return result

    try:
        device = torch.get_device_module("tpu").get_local_device_attributes()
        report = {
            "rank": dist.get_rank(),
            "device": {
                "id": int(device["id"]),
                "device_kind": device["device_kind"],
                "coords": tuple(device["coords"]),
                "core_on_chip": int(device["core_on_chip"]),
                "slice_index": int(device.get("slice_index", 0)),
            },
            "source": qualification.source_fingerprint(),
            "runtime": qualification.runtime_identity(),
            "manifest": qualification.compilation_identity()["manifest"],
        }
    except Exception as error:  # noqa: BLE001 - every rank must join the startup gather
        report = {"rank": dist.get_rank(), "error": f"{type(error).__name__}: {error}"}
    reports: list[dict | None] = [None] * 32
    dist.all_gather_object(reports, report, group=tp_group.cpu_group)
    try:
        for index, item in enumerate(reports):
            if item is None or "error" in item:
                reason = "empty report" if item is None else item["error"]
                raise ValueError(
                    f"K3 Two Shot device discovery failed on rank {index}: {reason}"
                )
        valid_reports = [item for item in reports if item is not None]
        if [item["rank"] for item in valid_reports] != list(ranks):
            raise ValueError("K3 Two Shot device reports differ from TP rank order")
        if any(
            any(
                item[field] != valid_reports[0][field]
                for field in ("source", "runtime", "manifest")
            )
            for item in valid_reports
        ):
            raise ValueError(
                "K3 Two Shot source, runtime, or qualification differs across ranks"
            )
        devices = [item["device"] for item in valid_reports]
        identity = qualification.qualification_identity(
            devices, tuple(device["id"] for device in devices)
        )
    except ValueError as error:
        _TWO_SHOT_CONTEXTS[key] = str(error)
        raise
    mesh = get_pallas_mesh(axis_names=("k3_ar",))
    if tuple(mesh.axis_names) != ("k3_ar",) or mesh.size != 32:
        raise ValueError("K3 Two Shot bridge requires a flat 32-device mesh")
    qualification.register_context(tp_group.unique_name, identity)
    _TWO_SHOT_CONTEXTS[key] = mesh, identity
    return mesh, identity


def build_kimi_two_shot_all_reduce(prefix, tp_group):
    """Build the uniform full-world KDA bridge used by hardware qualification.

    Every rank must execute the same live collective graph. Repeated invocations
    must be data-dependent or run in separate synchronized executions; independent
    overlap and rank-dependent removal or merging of calls are unsupported.

    TorchTPU partitions the JAX export, then partitions the containing Torch graph
    again. The first pass removes the manual sharding required for side-effecting
    calls in the second pass. The private uniform-SPMD entry uses functional
    compiler semantics while retaining the kernel's barrier and all DMA waits.
    """
    from vllm_torchtpu.distributed.two_shot_qualification import COLLECTIVE_ID
    from vllm_torchtpu.kernels.collectives.two_shot import (
        _two_shot_all_reduce_uniform_spmd,
    )

    mesh, _ = prepare_two_shot_context(tp_group)
    key = ("two_shot", prefix, tp_group.unique_name, tuple(tp_group.ranks))
    if key in _OPS:
        return _OPS[key]
    spec = P("k3_ar", None, None)

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            return _two_shot_all_reduce_uniform_spmd(
                part[0], axis_name="k3_ar", axis_size=32, collective_id=COLLECTIVE_ID
            )[None]

        return jax.shard_map(
            local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )(x)

    group_key = hashlib.sha256(repr(key[2:]).encode()).hexdigest()[:16]
    name = f"pallas::kimi_two_shot_{prefix.replace('.', '_')}_{group_key}"
    op = sharded_jax_op(
        name,
        collective,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )
    op.register_fake(lambda x: x.new_empty(x.shape))

    def all_reduce(x):
        return op(x[None])[0]

    _OPS[key] = all_reduce
    return all_reduce


class KimiTwoShotAllReduce:
    """Use the measured batch sizes; leave all other inputs on the native path."""

    def __init__(self, op, batches):
        self.op = op
        self.batches = tuple(batches)

    def __call__(self, x, native):
        if (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == 7168
            and x.shape[0] in self.batches
        ):
            return self.op(x)
        return native(x)


def _qualified_kimi_two_shot_batches(group, config):
    """Agree on opt-in admission before any rank enters topology discovery."""
    import torch.distributed as dist

    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    key = (group.unique_name, tuple(group.ranks))
    if key in _KIMI_TWO_SHOT_BATCHES:
        return _KIMI_TWO_SHOT_BATCHES[key]
    manifest = None
    report: dict = {"eligible": False}
    try:
        if qualification.serving_context_supported(
            config, sp_prefill=envs.TPU_K3_SP_PREFILL
        ):
            manifest = qualification.load_manifest()
            if qualification.has_runtime_match(manifest):
                report = {
                    "eligible": True,
                    "source": qualification.source_fingerprint(),
                    "runtime": qualification.runtime_identity(),
                    "manifest": qualification.compilation_identity()["manifest"],
                }
    except (OSError, ValueError):
        # An unreadable or stale local installation must not strand peers in
        # topology discovery. All participants take the native path together.
        pass
    reports = [None] * group.world_size
    dist.all_gather_object(reports, report, group=group.cpu_group)
    batches: tuple[int, ...] = ()
    if report["eligible"] and all(item == report for item in reports):
        try:
            _, identity = prepare_two_shot_context(group)
            batches = qualification.qualified_batches(identity, manifest)
        except ValueError:
            pass
    if not batches:
        qualification.register_context(group.unique_name, {"batches": []})
    # The manifest, runtime and model configuration are immutable for this
    # process-group lifetime; subsequent KDA layers reuse the startup decision.
    _KIMI_TWO_SHOT_BATCHES[key] = batches
    return batches


def prepare_kimi_two_shot_all_reduce(prefix, config):
    """Prepare opt-in admission at model construction, outside traced forwards.

    The feature flag is a process-wide launch setting shared by all TP ranks.
    When enabled, even unsupported model configurations join the CPU consensus.
    Uniformly empty qualification never probes physical devices.
    """
    from vllm_torchtpu import envs

    if not envs.TPU_K3_TWO_SHOT_ALL_REDUCE:
        return None
    group = get_tp_group()
    batches = _qualified_kimi_two_shot_batches(group, config)
    if not batches:
        return None
    return KimiTwoShotAllReduce(build_kimi_two_shot_all_reduce(prefix, group), batches)
