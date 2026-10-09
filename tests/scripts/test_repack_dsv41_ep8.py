"""Focused checks for the offline DSv4.1 EP8 shard layout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

_WEIGHTS = Path(__file__).resolve().parents[2] / "scripts" / "weights"
sys.path.insert(0, str(_WEIGHTS))
from repack_dsv41_ep8 import read_expert, store_expert  # noqa: E402


def test_read_expert_recovers_all_projection_keys(tmp_path: Path) -> None:
    directory = tmp_path / "prepared-weights"
    directory.mkdir()
    tensors = {}
    records = {}
    for projection in (1, 2, 3):
        for kind in ("fp4", "scale"):
            key = f"layers.0.ffn.experts.7.w{projection}.weight::{kind}"
            value = np.full(
                (2, 2), projection, dtype=np.uint8 if kind == "fp4" else np.float32
            )
            tensors[key] = value
            records[key] = {
                "file": "prepared-weights/sample.fp4",
                "dtype": "U8" if kind == "fp4" else "F32",
                "shape": [2, 2],
            }
    save_file(tensors, str(directory / "sample.fp4"))
    parts = read_expert(tmp_path, records, 0, 7)
    assert set(parts) == {
        f"w{i}::{kind}" for i in (1, 2, 3) for kind in ("fp4", "scale")
    }
    for projection in (1, 2, 3):
        np.testing.assert_array_equal(parts[f"w{projection}::fp4"], projection)


def test_store_expert_preserves_fp4_and_scale_order() -> None:
    tensors = {
        "w13": np.empty((1, 5120, 2560), dtype=np.uint8),
        "w2": np.empty((1, 2560, 2560), dtype=np.uint8),
        "s13": np.empty((1, 10, 1, 5120), dtype=np.float32),
        "s2": np.empty((1, 5, 1, 5120), dtype=np.float32),
    }
    parts = {
        "w1::fp4": np.full((5120, 1280), 1, dtype=np.uint8),
        "w2::fp4": np.full((2560, 2560), 2, dtype=np.uint8),
        "w3::fp4": np.full((5120, 1280), 3, dtype=np.uint8),
        "w1::scale": np.full((10, 1, 2560), 11, dtype=np.float32),
        "w2::scale": np.full((5, 1, 5120), 12, dtype=np.float32),
        "w3::scale": np.full((10, 1, 2560), 13, dtype=np.float32),
    }
    store_expert(tensors, parts, 0)
    assert np.all(tensors["w13"][0, :, :1280] == 1)
    assert np.all(tensors["w13"][0, :, 1280:] == 3)
    assert np.all(tensors["w2"] == 2)
    assert np.all(tensors["s13"][0, :, :, :2560] == 11)
    assert np.all(tensors["s13"][0, :, :, 2560:] == 13)
    assert np.all(tensors["s2"] == 12)
