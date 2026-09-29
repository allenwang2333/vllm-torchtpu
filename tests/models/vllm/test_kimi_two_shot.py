# SPDX-License-Identifier: Apache-2.0
"""The KDA callsite preserves native behavior outside its exact admission."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.kimi_k3 import collective_ops

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "shape,dtype,custom",
    [
        ((1, 7168), torch.bfloat16, True),
        ((16, 7168), torch.bfloat16, True),
        ((4, 7168), torch.bfloat16, False),
        ((3, 7168), torch.bfloat16, False),
        ((0, 7168), torch.bfloat16, False),
        ((1, 7167), torch.bfloat16, False),
        ((1, 7168), torch.float32, False),
        ((1, 1, 7168), torch.bfloat16, False),
    ],
)
def test_only_exact_shapes_use_custom(shape, dtype, custom):
    x = torch.zeros(shape, dtype=dtype)
    op = Mock(return_value=x + 1)
    native = Mock(return_value=x + 2)
    adapter = collective_ops.KimiTwoShotAllReduce(op, (1, 16))
    result = adapter(x, native)
    assert result is (op.return_value if custom else native.return_value)
    (op if custom else native).assert_called_once_with(x)
    (native if custom else op).assert_not_called()


def test_disabled_switch_does_not_discover_devices(monkeypatch):
    monkeypatch.setattr(envs, "TPU_K3_TWO_SHOT_ALL_REDUCE", False)
    prepare = Mock(side_effect=AssertionError("startup discovery must stay disabled"))
    monkeypatch.setattr(collective_ops, "prepare_two_shot_context", prepare)
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    prepare.assert_not_called()


def test_bridge_keeps_rank_slot_and_registers_symbolic_fake(monkeypatch):
    from vllm_torchtpu.kernels.collectives import two_shot

    group = SimpleNamespace(unique_name="tp", ranks=tuple(range(32)))
    mesh = object()
    monkeypatch.setattr(collective_ops, "_OPS", {})
    monkeypatch.setattr(
        collective_ops, "prepare_two_shot_context", lambda group: (mesh, {})
    )
    captured = {}
    operator = Mock(side_effect=lambda x: x.clone())

    def register(name, fn, **kwargs):
        captured.update(kwargs, function=fn)
        return operator

    monkeypatch.setattr(collective_ops, "sharded_jax_op", register)
    bridge = collective_ops.build_kimi_two_shot_all_reduce("layers.0", group)
    x = torch.zeros((3, 7168), dtype=torch.bfloat16)
    assert bridge(x).shape == x.shape
    assert operator.call_args.args[0].shape == (1, 3, 7168)
    assert captured["input_partition_specs"] == (P("k3_ar", None, None),)
    assert captured["output_partition_specs"] == P("k3_ar", None, None)
    fake = operator.register_fake.call_args.args[0]
    assert fake(torch.empty((1, 5, 7168))).shape == (1, 5, 7168)
    assert collective_ops.build_kimi_two_shot_all_reduce("layers.0", group) is bridge
    # Only this uniform full-world adapter may select the functional entry.
    functional = Mock(side_effect=lambda value, **kwargs: value)
    monkeypatch.setattr(two_shot, "_two_shot_all_reduce_uniform_spmd", functional)
    # The factory captures its import at construction, so build another prefix.
    monkeypatch.setattr(
        two_shot, "two_shot_all_reduce", Mock(side_effect=AssertionError("public path"))
    )
    monkeypatch.setattr(collective_ops.jax, "shard_map", lambda fn, **kwargs: fn)
    collective_ops.build_kimi_two_shot_all_reduce("layers.1", group)
    assert captured["function"](x[None]).shape == (1, 3, 7168)
    assert functional.call_args.args[0].shape == (3, 7168)
    assert functional.call_args.kwargs == {
        "axis_name": "k3_ar",
        "axis_size": 32,
        "collective_id": 1,
    }


@pytest.fixture
def startup(monkeypatch):
    import torch.distributed as dist
    from torch_tpu._internal import pallas

    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    group = SimpleNamespace(
        unique_name="tp", ranks=tuple(range(32)), world_size=32, cpu_group=object()
    )
    devices = [
        {
            "id": 31 - rank,
            "device_kind": "TPU7x",
            "coords": (rank // 16, rank // 8 % 2, rank // 2 % 4),
            "core_on_chip": rank % 2,
        }
        for rank in range(32)
    ]
    monkeypatch.setattr(collective_ops, "_TWO_SHOT_CONTEXTS", {})
    monkeypatch.setattr(qualification, "_CONTEXT_IDENTITIES", {})
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 32)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda name: SimpleNamespace(get_local_device_attributes=lambda: devices[0]),
    )
    monkeypatch.setattr(qualification, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(
        qualification, "runtime_identity", lambda: {"version": "runtime"}
    )
    mesh = SimpleNamespace(axis_names=("k3_ar",), size=32)
    monkeypatch.setattr(pallas, "get_pallas_mesh", lambda **kwargs: mesh)
    monkeypatch.setattr(
        collective_ops.jax,
        "devices",
        Mock(side_effect=AssertionError("no JAX runtime")),
    )

    def gather(reports, local, *, group):
        reports[:] = [
            dict(local, rank=i, device=device) for i, device in enumerate(devices)
        ]

    return group, mesh, gather


def test_startup_uses_native_metadata_and_caches_rank_order(startup, monkeypatch):
    import torch.distributed as dist

    group, mesh, gather = startup
    collect = Mock(side_effect=gather)
    monkeypatch.setattr(dist, "all_gather_object", collect)
    actual_mesh, identity = collective_ops.prepare_two_shot_context(group)
    assert actual_mesh is mesh
    assert identity["rank_mapping"][0] == [0, 0, 0, 0]
    assert identity["rank_mapping"][-1] == [1, 1, 3, 1]
    assert collective_ops.prepare_two_shot_context(group) == (mesh, identity)
    collect.assert_called_once()


@pytest.mark.parametrize("failure", ["device", "runtime"])
def test_startup_all_ranks_observe_discovery_failure(startup, monkeypatch, failure):
    import torch.distributed as dist

    group, _, gather = startup

    def bad_report(reports, local, *, group):
        gather(reports, local, group=group)
        if failure == "device":
            reports[7] = {"rank": 7, "error": "missing coordinates"}
        else:
            reports[7]["runtime"] = "different version"

    monkeypatch.setattr(dist, "all_gather_object", bad_report)
    match = (
        "rank 7: missing coordinates" if failure == "device" else "differs across ranks"
    )
    with pytest.raises(ValueError, match=match):
        collective_ops.prepare_two_shot_context(group)


def test_group_lifetimes_use_distinct_registration_names(monkeypatch):
    monkeypatch.setattr(collective_ops, "_OPS", {})
    monkeypatch.setattr(
        collective_ops, "prepare_two_shot_context", lambda group: (object(), {})
    )
    register = Mock(side_effect=lambda *args, **kwargs: Mock())
    monkeypatch.setattr(collective_ops, "sharded_jax_op", register)
    for name in ("tp:0", "tp:1"):
        collective_ops.build_kimi_two_shot_all_reduce(
            "layers.0", SimpleNamespace(unique_name=name, ranks=tuple(range(32)))
        )
    assert register.call_args_list[0].args[0] != register.call_args_list[1].args[0]


@pytest.fixture
def admission(monkeypatch):
    import torch.distributed as dist

    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    group = SimpleNamespace(
        unique_name="tp", ranks=tuple(range(32)), world_size=32, cpu_group=object()
    )
    identity = {"source": "source", "runtime": {"version": "runtime"}}
    manifest = {
        "schema_version": 1,
        "entries": [{"identity": identity, "batches": [1, 4]}],
    }
    monkeypatch.setattr(envs, "TPU_K3_TWO_SHOT_ALL_REDUCE", True)
    monkeypatch.setattr(envs, "TPU_K3_SP_PREFILL", False)
    monkeypatch.setattr(collective_ops, "_KIMI_TWO_SHOT_BATCHES", {})
    monkeypatch.setattr(collective_ops, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        qualification, "serving_context_supported", lambda *args, **kwargs: True
    )
    compilation_identity = qualification.compilation_identity
    monkeypatch.setattr(qualification, "_CONTEXT_IDENTITIES", {})
    monkeypatch.setattr(qualification, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(qualification, "runtime_identity", lambda: identity["runtime"])
    monkeypatch.setattr(
        qualification, "compilation_identity", lambda: {"manifest": "manifest"}
    )
    monkeypatch.setattr(qualification, "load_manifest", lambda: manifest)

    def gather(reports, local, **kwargs):
        reports[:] = [dict(local) for _ in range(32)]

    gather = Mock(side_effect=gather)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    discover = Mock(return_value=(object(), identity))
    build = Mock(return_value=Mock())
    monkeypatch.setattr(collective_ops, "prepare_two_shot_context", discover)
    monkeypatch.setattr(collective_ops, "build_kimi_two_shot_all_reduce", build)
    return SimpleNamespace(
        manifest=manifest,
        gather=gather,
        discover=discover,
        build=build,
        compilation_identity=compilation_identity,
    )


def test_empty_qualification_stays_native_before_device_discovery(admission):
    admission.manifest["entries"] = []
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.1", None) is None
    admission.gather.assert_called_once()
    admission.discover.assert_not_called()
    admission.build.assert_not_called()


def test_native_admission_replaces_previous_qualified_cache_identity(admission):
    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    qualification.register_context("tp", {"source": "previously-qualified"})
    previous_identity = admission.compilation_identity()
    admission.manifest["entries"] = []
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    assert qualification._CONTEXT_IDENTITIES["tp"] == {"batches": []}
    assert admission.compilation_identity() != previous_identity


@pytest.mark.parametrize("local_eligible", [False, True])
def test_mixed_runtime_eligibility_never_enters_topology_discovery(
    admission, monkeypatch, local_eligible
):
    if not local_eligible:
        admission.manifest["entries"] = []

    def mismatched(reports, local, **kwargs):
        eligible = {
            "eligible": True,
            "source": "source",
            "runtime": {"version": "runtime"},
            "manifest": "manifest",
        }
        reports[:] = [dict(eligible) for _ in range(32)]
        reports[0] = local
        reports[7] = {"eligible": False}

    admission.gather.side_effect = mismatched
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    admission.gather.assert_called_once()
    admission.discover.assert_not_called()
    admission.build.assert_not_called()


@pytest.mark.parametrize("field", ["source", "runtime", "manifest"])
def test_disagreeing_admission_identity_uses_native(admission, field):
    gather = admission.gather.side_effect

    def mismatched(reports, local, **kwargs):
        gather(reports, local, **kwargs)
        reports[7][field] = "different"

    admission.gather.side_effect = mismatched
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    admission.discover.assert_not_called()


def test_unsupported_context_still_joins_cpu_consensus(admission, monkeypatch):
    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    monkeypatch.setattr(
        qualification, "serving_context_supported", lambda *args, **kwargs: False
    )
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    admission.gather.assert_called_once()
    admission.discover.assert_not_called()


def test_unreadable_local_manifest_still_joins_cpu_consensus(admission, monkeypatch):
    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    monkeypatch.setattr(qualification, "load_manifest", Mock(side_effect=ValueError))
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    admission.gather.assert_called_once()
    admission.discover.assert_not_called()


def test_missing_manifest_keeps_native_compilation_identity_available(
    admission, monkeypatch, tmp_path
):
    from vllm_torchtpu.distributed import two_shot_qualification as qualification

    path = tmp_path / "missing.json"
    monkeypatch.setattr(qualification, "_MANIFEST", path)
    monkeypatch.setattr(
        qualification, "compilation_identity", admission.compilation_identity
    )
    monkeypatch.setattr(
        qualification, "load_manifest", Mock(side_effect=FileNotFoundError)
    )
    assert collective_ops.prepare_kimi_two_shot_all_reduce("layers.0", None) is None
    identity = qualification.compilation_identity()
    assert identity == {"manifest": "unavailable", "contexts": {"tp": {"batches": []}}}
    path.write_text('{"schema_version": 1, "entries": []}')
    assert qualification.compilation_identity() != identity
    admission.discover.assert_not_called()


def test_matching_admission_is_resolved_once_per_group(admission):
    for prefix in ("layers.0", "layers.1"):
        selected = collective_ops.prepare_kimi_two_shot_all_reduce(prefix, None)
        assert selected.batches == (1, 4)
    admission.gather.assert_called_once()
    admission.discover.assert_called_once()
    assert admission.build.call_count == 2
