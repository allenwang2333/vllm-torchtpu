# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_torchtpu.distributed.kv_transfer.raiden.tags."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
if (_SRC_ROOT / "distributed" / "kv_transfer" / "raiden").is_dir():
    for _pkg_name, _pkg_dir in (
        ("vllm_torchtpu", _SRC_ROOT),
        ("vllm_torchtpu.distributed", _SRC_ROOT / "distributed"),
        (
            "vllm_torchtpu.distributed.kv_transfer",
            _SRC_ROOT / "distributed" / "kv_transfer",
        ),
        (
            "vllm_torchtpu.distributed.kv_transfer.raiden",
            _SRC_ROOT / "distributed" / "kv_transfer" / "raiden",
        ),
    ):
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_pkg_dir)]
        sys.modules.setdefault(_pkg_name, _pkg)

from vllm_torchtpu.distributed.kv_transfer.raiden import tags  # noqa: E402


def test_tag_constants_and_dsv4_group_tag() -> None:
    """Verify frozen Raiden pool tag vocabulary and dsv4_group_tag caching."""
    tags.dsv4_group_tag.cache_clear()
    expected_constants = {
        "TAG_FA": "fa",
        "TAG_GDN_CONV": "gdn.conv",
        "TAG_GDN_SSM": "gdn.ssm",
        "TAG_DSA_IDX": "dsa.idx",
        "TAG_MLA_NOPE": "mla.nope",
        "TAG_MLA_ROPE": "mla.rope",
        "TAG_DSV4_CSA_NOPE": "dsv4.csa.nope",
        "TAG_DSV4_CSA_ROPE": "dsv4.csa.rope",
        "TAG_DSV4_IDX": "dsv4.idx",
        "TAG_DSV4_HCA": "dsv4.hca",
        "TAG_DSV4_SWA": "dsv4.swa",
        "TAG_DSV4_STATE_CSA": "dsv4.state.csa",
        "TAG_DSV4_STATE_IDX": "dsv4.state.idx",
        "TAG_DSV4_STATE_HCA": "dsv4.state.hca",
    }
    for attr_name, value in expected_constants.items():
        assert getattr(tags, attr_name) == value

    first = tags.dsv4_group_tag(tags.TAG_DSV4_SWA, 3)
    second = tags.dsv4_group_tag(tags.TAG_DSV4_SWA, 3)
    assert first == "dsv4.swa.g3"
    assert first is second
    assert tags.dsv4_group_tag.cache_info().hits >= 1


def test_layer_tag_validation_and_caching() -> None:
    """Verify layer_tag formatting, negative index rejection, and caching."""
    tags.layer_tag.cache_clear()
    for group_idx in range(4):
        group_tag = tags.dsv4_group_tag(tags.TAG_DSV4_CSA_NOPE, group_idx)
        for layer_idx in (0, 1, 30, 60):
            formatted = tags.layer_tag(group_tag, layer_idx)
            assert formatted == f"dsv4.csa.nope.g{group_idx}.l{layer_idx}"
            assert tags.layer_tag(group_tag, layer_idx) is formatted

    with pytest.raises(ValueError, match="layer index must be non-negative: -1"):
        tags.layer_tag(tags.TAG_FA, -1)

    with pytest.raises(ValueError, match="layer index must be non-negative: -99"):
        tags.layer_tag(tags.TAG_DSV4_HCA, -99)


def test_split_layer_tag_and_class_tag_round_trip() -> None:
    """Verify split_layer_tag and class_tag across layered and unlayered tags."""
    tags.split_layer_tag.cache_clear()
    tags.class_tag.cache_clear()

    base_tags = (
        tags.TAG_FA,
        tags.TAG_MLA_NOPE,
        tags.TAG_MLA_ROPE,
        tags.dsv4_group_tag(tags.TAG_DSV4_CSA_NOPE, 0),
        tags.dsv4_group_tag(tags.TAG_DSV4_CSA_ROPE, 1),
        tags.dsv4_group_tag(tags.TAG_DSV4_IDX, 2),
        tags.dsv4_group_tag(tags.TAG_DSV4_HCA, 3),
        tags.dsv4_group_tag(tags.TAG_DSV4_SWA, 4),
        tags.dsv4_group_tag(tags.TAG_DSV4_STATE_CSA, 0),
        tags.dsv4_group_tag(tags.TAG_DSV4_STATE_IDX, 1),
        tags.dsv4_group_tag(tags.TAG_DSV4_STATE_HCA, 2),
    )

    for base in base_tags:
        assert tags.split_layer_tag(base) == (base, None)
        assert tags.split_layer_tag(base) is tags.split_layer_tag(base)
        assert tags.class_tag(base) == base
        assert tags.class_tag(base) is tags.class_tag(base)

        for layer_idx in (0, 7, 60):
            layered = tags.layer_tag(base, layer_idx)
            split_res = tags.split_layer_tag(layered)
            assert split_res == (base, layer_idx)
            assert tags.split_layer_tag(layered) is split_res

            cls_res = tags.class_tag(layered)
            assert cls_res == base
            assert tags.class_tag(layered) is cls_res

    for non_numeric_suffix in ("fa.l", "fa.labc", "dsv4.swa.g0.l-1", "dsv4.swa.g0.l3x"):
        assert tags.split_layer_tag(non_numeric_suffix) == (non_numeric_suffix, None)
        assert tags.class_tag(non_numeric_suffix) == non_numeric_suffix
