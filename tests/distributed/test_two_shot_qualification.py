# SPDX-License-Identifier: Apache-2.0
"""Admission uses exact source, runtime, shape, and physical placement."""

import copy
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import two_shot_qualification as qualification

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def devices():
    return [
        {
            "id": index,
            "device_kind": "TPU v7",
            "coords": (index // 16, index // 8 % 2, index // 2 % 4),
            "core_on_chip": index % 2,
        }
        for index in range(32)
    ]


@pytest.fixture
def identity(monkeypatch, devices):
    monkeypatch.setattr(qualification, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(
        qualification, "runtime_identity", lambda: {"version": "runtime"}
    )
    return qualification.qualification_identity(devices, tuple(range(32)))


def test_rank_order_is_preserved_and_coordinate_origin_is_irrelevant(devices, identity):
    order = tuple(reversed(range(32)))
    permuted = qualification.qualification_identity(devices, order)
    assert permuted["rank_mapping"] == list(reversed(identity["rank_mapping"]))
    for device in devices:
        device["coords"] = tuple(coord + 4 for coord in device["coords"])
    assert qualification.qualification_identity(devices, tuple(range(32))) == identity


@pytest.mark.parametrize(
    "corruption", ["kind", "coordinates", "core", "duplicate", "gap", "v70"]
)
def test_invalid_physical_inventory_rejected(devices, identity, corruption):
    if corruption == "kind":
        devices[0]["device_kind"] = "TPU v6e"
    elif corruption == "coordinates":
        devices[0]["coords"] = (4, 0, 0)
    elif corruption == "core":
        devices[0]["core_on_chip"] = 1
    elif corruption == "duplicate":
        devices[0]["id"] = 1
    elif corruption == "gap":
        for device in devices:
            x, y, z = device["coords"]
            device["coords"] = (2 * x, y, z)
    else:
        for device in devices:
            device["device_kind"] = "TPU v70"
    with pytest.raises(ValueError):
        qualification.qualification_identity(devices, tuple(range(32)))


def test_exact_admission(identity):
    manifest = {
        "schema_version": 1,
        "entries": [{"identity": identity, "batches": [16, 1, 4]}],
    }
    assert qualification.qualified_batches(identity, manifest) == (1, 4, 16)
    for field in ("source", "runtime", "topology", "rank_mapping"):
        other = copy.deepcopy(identity)
        other[field] = "changed"
        assert qualification.qualified_batches(other, manifest) == ()
    assert (
        qualification.qualified_batches(identity, {"schema_version": 1, "entries": []})
        == ()
    )


@pytest.mark.parametrize("batches", [[3], [True], [128], "16"])
def test_invalid_allowlist_fails_closed(identity, batches):
    manifest = {
        "schema_version": 1,
        "entries": [{"identity": identity, "batches": batches}],
    }
    assert qualification.qualified_batches(identity, manifest) == ()


def test_ambiguous_or_future_manifest_fails_closed(identity):
    entry = {"identity": identity, "batches": [16]}
    assert (
        qualification.qualified_batches(
            identity, {"schema_version": 1, "entries": [entry, entry]}
        )
        == ()
    )
    assert (
        qualification.qualified_batches(
            identity, {"schema_version": 2, "entries": [entry]}
        )
        == ()
    )


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=1),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 8),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_dbo", True),
        ("ubatch_size", 2),
    ],
)
def test_other_parallel_contexts_use_native(config, field, value):
    assert qualification.serving_context_supported(config, sp_prefill=False)
    setattr(config.parallel_config, field, value)
    assert not qualification.serving_context_supported(config, sp_prefill=False)


@pytest.mark.parametrize(
    "field,value",
    [("async_scheduling", True), ("async_scheduling", None), ("max_num_seqs", 2)],
)
def test_potentially_overlapping_scheduling_uses_native(config, field, value):
    assert qualification.serving_context_supported(config, sp_prefill=False)
    setattr(config.scheduler_config, field, value)
    assert not qualification.serving_context_supported(config, sp_prefill=False)


def test_sp_speculation_and_effective_dp_use_native(config, monkeypatch):
    assert not qualification.serving_context_supported(config, sp_prefill=True)
    config.speculative_config = object()
    assert not qualification.serving_context_supported(config, sp_prefill=False)
    config.speculative_config = None
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    assert not qualification.serving_context_supported(config, sp_prefill=False)


def test_manifest_and_placement_change_compilation_identity(tmp_path, monkeypatch):
    path = tmp_path / "qualification.json"
    path.write_text('{"schema_version": 1, "entries": []}')
    monkeypatch.setattr(qualification, "_MANIFEST", path)
    monkeypatch.setattr(qualification, "_CONTEXT_IDENTITIES", {})
    before = qualification.compilation_identity()
    path.write_text('{"schema_version": 1, "entries": [{"identity": {}}]}')
    assert qualification.compilation_identity() != before
    before = qualification.compilation_identity()
    qualification.register_context("tp", {"rank_mapping": [1, 0]})
    assert qualification.compilation_identity() != before


def test_runtime_prefilter_does_not_admit_changed_sources(identity):
    manifest = {
        "schema_version": 1,
        "entries": [{"identity": identity, "batches": [16]}],
    }
    assert qualification.has_runtime_match(manifest)
    manifest["entries"][0]["identity"] = dict(identity, source="changed")
    assert not qualification.has_runtime_match(manifest)


def test_manifest_is_not_part_of_kernel_source_identity(tmp_path, monkeypatch):
    manifest = tmp_path / "qualification.json"
    monkeypatch.setattr(qualification, "_MANIFEST", manifest)
    manifest.write_text('{"schema_version": 1, "entries": []}')
    source = qualification.source_fingerprint()
    manifest.write_text('{"schema_version": 1, "entries": [{"batches": [16]}]}')
    assert qualification.source_fingerprint() == source


def test_runtime_flag_strings_are_preserved_exactly(monkeypatch):
    flags = {
        "LIBTPU_INIT_ARGS": " --xla_example=true ",
        "XLA_FLAGS": "  --xla_another=false  ",
        "TORCH_TPU_INTERNAL_XLA_OPTIONS": '{ "example": true }',
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
    }
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    assert qualification.runtime_identity()["flags"] == flags


def test_vllm_version_is_part_of_runtime_identity(monkeypatch):
    version = "vllm-first"
    monkeypatch.setattr(
        qualification.importlib.metadata,
        "version",
        lambda name: version if name == "vllm" else "unchanged",
    )
    before = qualification.runtime_identity()
    assert before["versions"]["vllm"] == "vllm-first"
    version = "vllm-updated"
    assert qualification.runtime_identity() != before


@pytest.mark.parametrize(
    "source",
    [
        "__init__.py",
        "platforms/tpu_platform.py",
        "runner/tpu_runner.py",
        "runner/tpu_runner_async_output.py",
    ],
)
def test_serving_execution_sources_invalidate_qualification(
    tmp_path, monkeypatch, source
):
    module = tmp_path / "distributed" / "two_shot_qualification.py"
    monkeypatch.setattr(qualification, "__file__", str(module))
    for name in qualification._SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    before = qualification.source_fingerprint()
    (tmp_path / source).write_text("updated serving compiler")
    assert qualification.source_fingerprint() != before
