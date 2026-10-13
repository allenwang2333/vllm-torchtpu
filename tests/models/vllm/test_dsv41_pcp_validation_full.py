# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for platforms/pcp_validation.py."""

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

from vllm_torchtpu.platforms import pcp_validation as pcp_mod  # noqa: E402
from vllm_torchtpu.platforms.pcp_validation import (  # noqa: E402
    _DEFAULT_PCP_STATIC_CONFIG,
    PcpStaticSupportValidator,
)


def _vllm_cfg(
    *,
    pcp_size: int = 4,
    interleave_size: int = 128,
    dcp_size: int = 1,
    pipeline_parallel_size: int = 1,
    async_scheduling: bool = False,
    speculative_method: str | None = None,
    num_speculative_tokens: int = 0,
    is_moe_model: bool | None = False,
    enable_expert_parallel: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            decode_context_parallel_size=dcp_size,
            pipeline_parallel_size=pipeline_parallel_size,
            is_moe_model=is_moe_model,
            enable_expert_parallel=enable_expert_parallel,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=(
            None
            if speculative_method is None
            else SimpleNamespace(
                method=speculative_method,
                num_speculative_tokens=num_speculative_tokens,
            )
        ),
    )


def test_pcp_validation_all_branches() -> None:
    importlib.reload(pcp_mod)
    default_cfg = pcp_mod._DEFAULT_PCP_STATIC_CONFIG
    validator = pcp_mod.PcpStaticSupportValidator
    pcp_mod._PCP_CONFIG_CACHE.clear()
    pcp_mod._VALIDATED_PCP_CACHE.clear()

    # 1. None and default singleton fast path
    c_none = validator.from_vllm_config(None)
    assert c_none is default_cfg
    assert c_none.enabled is False

    c_def = validator.from_vllm_config(_vllm_cfg(pcp_size=1, interleave_size=1))
    assert c_def is default_cfg
    assert _DEFAULT_PCP_STATIC_CONFIG is not None
    assert PcpStaticSupportValidator is not None

    # 2. Disabled PCP with non-default interleave_size (populates _PCP_CONFIG_CACHE)
    c_dis = PcpStaticSupportValidator.validate_platform_config(
        _vllm_cfg(pcp_size=1, interleave_size=64), kv_cache_layout="HND"
    )
    assert c_dis.enabled is False
    # Hit _PCP_CONFIG_CACHE on second call
    assert (
        PcpStaticSupportValidator.from_vllm_config(
            _vllm_cfg(pcp_size=1, interleave_size=64)
        )
        is c_dis
    )

    # 3. Valid enabled PCP without speculation and with MTP speculation
    c_val = PcpStaticSupportValidator.validate_platform_config(
        _vllm_cfg(pcp_size=4, interleave_size=128), kv_cache_layout="HND"
    )
    assert c_val.enabled is True
    assert (
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, interleave_size=128), kv_cache_layout="HND"
        )
        is c_val
    )

    c_mtp = PcpStaticSupportValidator.validate_platform_config(
        _vllm_cfg(
            pcp_size=4,
            interleave_size=16,
            speculative_method="mtp",
            num_speculative_tokens=1,
            is_moe_model=True,
            enable_expert_parallel=True,
        ),
        kv_cache_layout="NHD",
    )
    assert c_mtp.speculative_enabled is True

    # 4. All validation error branches
    with pytest.raises(ValueError, match="multiple of 128"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, interleave_size=64), kv_cache_layout="HND"
        )

    with pytest.raises(NotImplementedError, match="DCP"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, dcp_size=2), kv_cache_layout="NHD"
        )

    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, pipeline_parallel_size=2), kv_cache_layout="NHD"
        )

    with pytest.raises(NotImplementedError, match="enable-expert-parallel"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, is_moe_model=True, enable_expert_parallel=False),
            kv_cache_layout="NHD",
        )

    with pytest.raises(NotImplementedError, match="method=mtp"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(
                pcp_size=4, speculative_method="eagle3", num_speculative_tokens=1
            ),
            kv_cache_layout="NHD",
        )

    with pytest.raises(NotImplementedError, match="num_speculative_tokens=1"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, speculative_method="mtp", num_speculative_tokens=2),
            kv_cache_layout="NHD",
        )

    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size > 0"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_cfg(pcp_size=4, interleave_size=0), kv_cache_layout="NHD"
        )
