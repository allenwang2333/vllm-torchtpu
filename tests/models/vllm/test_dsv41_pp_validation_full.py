# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for platforms/pp_validation.py."""

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel_parts in (
    ("vllm_torchtpu", ("vllm_torchtpu",)),
    ("vllm_torchtpu.platforms", ("vllm_torchtpu", "platforms")),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg_dir = str(Path(_SRC_DIR, *_rel_parts))
        _pkg.__path__ = [_pkg_dir]
        _pkg.__file__ = str(Path(_pkg_dir, "__init__.py"))
        _spec = importlib.machinery.ModuleSpec(_pkg_name, None, is_package=True)
        _spec.submodule_search_locations = [_pkg_dir]
        _pkg.__spec__ = _spec
        sys.modules[_pkg_name] = _pkg

_vllm_pkg = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
if "vllm.logger" not in sys.modules:
    _vlog = types.ModuleType("vllm.logger")
    _vlog.init_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
    _vlog._print_warning_once = SimpleNamespace(cache_clear=lambda: None)  # type: ignore[attr-defined]
    _vllm_pkg.logger = _vlog  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = _vlog

if "vllm.envs" not in sys.modules:
    _venvs = types.ModuleType("vllm.envs")
    _venvs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = False  # type: ignore[attr-defined]
    _vllm_pkg.envs = _venvs  # type: ignore[attr-defined]
    sys.modules["vllm.envs"] = _venvs

if "vllm.distributed.utils" not in sys.modules:
    _vdist = sys.modules.setdefault(
        "vllm.distributed", types.ModuleType("vllm.distributed")
    )
    _vdutils = types.ModuleType("vllm.distributed.utils")

    def _stub_get_pp_indices(
        num_layers: int, stage: int, stages: int
    ) -> tuple[int, int]:
        per_stage = num_layers // stages
        return stage * per_stage, (stage + 1) * per_stage

    _vdutils.get_pp_indices = _stub_get_pp_indices  # type: ignore[attr-defined]
    _vdist.utils = _vdutils  # type: ignore[attr-defined]
    sys.modules["vllm.distributed.utils"] = _vdutils

from vllm_torchtpu import envs as tpu_envs  # noqa: E402
from vllm_torchtpu.platforms import pp_validation as pp_mod  # noqa: E402
from vllm_torchtpu.platforms.pp_validation import (  # noqa: E402
    _legacy_ray_executor,
    _stage_without_full_attention,
    validate_pipeline_parallel_config,
)


def _cfg(
    *,
    pp: int = 2,
    tp: int = 1,
    async_scheduling: bool = False,
    speculative: object = None,
    kv_transfer: object = None,
    layer_types: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            tensor_parallel_size=tp,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative,
        kv_transfer_config=kv_transfer,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types)
        ),
    )


def test_pp_validation_all_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    importlib.reload(pp_mod)
    assert tpu_envs is not None
    raiden = SimpleNamespace(kv_connector="TPURaidenConnector")
    other_kv = SimpleNamespace(kv_connector="OtherConnector")
    hybrid_8 = [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(8)
    ]

    # 1. _legacy_ray_executor: cold import paths when modules removed from sys.modules
    saved_tpu_envs = sys.modules.pop("vllm_torchtpu.envs", None)
    try:
        monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)
        assert _legacy_ray_executor() is False
    finally:
        if saved_tpu_envs is not None:
            sys.modules["vllm_torchtpu.envs"] = saved_tpu_envs

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    vllm_dir = tmp_path / "vllm"
    vllm_dir.mkdir(exist_ok=True)
    (vllm_dir / "envs.py").write_text(
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND = False\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys.modules["vllm"], "__path__", [str(vllm_dir)], raising=False)
    saved_vllm_envs = sys.modules.pop("vllm.envs", None)
    try:
        assert _legacy_ray_executor() is True
    finally:
        if saved_vllm_envs is not None:
            sys.modules["vllm.envs"] = saved_vllm_envs

    # Warm sys.modules lookup with V2 enabled
    monkeypatch.setattr(
        sys.modules["vllm.envs"],
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND",
        True,
        raising=False,
    )
    assert _legacy_ray_executor() is False
    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)

    # 2. _stage_without_full_attention: cold _get_pp_indices import + warm reuse
    pp_mod._get_pp_indices = None
    assert _stage_without_full_attention(_cfg(pp=2, layer_types=None)) is None
    assert (
        _stage_without_full_attention(
            _cfg(pp=2, layer_types=["deepseek_sparse_attention"] * 4)
        )
        is None
    )
    assert _stage_without_full_attention(_cfg(pp=2, layer_types=hybrid_8)) is None
    msg = _stage_without_full_attention(_cfg(pp=4, layer_types=hybrid_8))
    assert msg == "stage 0 holds layers 0-1, linear_attention only"

    # 3. validate_pipeline_parallel_config: all branches
    validate_pipeline_parallel_config(_cfg(pp=1, tp=4))
    validate_pipeline_parallel_config(_cfg(pp=2))

    with pytest.raises(NotImplementedError, match="tensor parallelism inside a stage"):
        validate_pipeline_parallel_config(_cfg(pp=2, tp=2))

    with pytest.raises(NotImplementedError, match="async scheduling"):
        validate_pipeline_parallel_config(_cfg(pp=2, async_scheduling=True))

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        validate_pipeline_parallel_config(_cfg(pp=2, speculative=object()))

    with pytest.raises(NotImplementedError, match="other than TPURaidenConnector"):
        validate_pipeline_parallel_config(_cfg(pp=2, kv_transfer=other_kv))

    with pytest.raises(
        NotImplementedError, match="stage holding no full-attention layer"
    ):
        validate_pipeline_parallel_config(
            _cfg(pp=4, kv_transfer=raiden, layer_types=hybrid_8)
        )

    validate_pipeline_parallel_config(
        _cfg(pp=2, kv_transfer=raiden, layer_types=hybrid_8)
    )

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setattr(
        sys.modules["vllm.envs"],
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND",
        False,
        raising=False,
    )
    with pytest.raises(NotImplementedError, match="the legacy Ray executor"):
        validate_pipeline_parallel_config(_cfg(pp=2))
