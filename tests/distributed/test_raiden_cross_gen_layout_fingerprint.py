# SPDX-License-Identifier: Apache-2.0
"""Unit verification suite for Raiden cross-generation layout fingerprinting."""

from __future__ import annotations

import unittest
from typing import Any

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.raiden import (
    layout_fingerprint as rlf,
    pool_manifest as rpm,
)


def _synthetic_fa_manifest(*, page_tokens: int = 4096) -> rpm.PoolManifest:
    """Builds a lightweight in-memory PoolManifest without TPU device allocation."""
    storage = object()
    return rpm.PoolManifest(
        binding=rpm.BINDING_PRIVATE_TYPED,
        storages=[storage],
        pools=[
            rpm.PoolEntry(
                tag=rpm.TAG_FA,
                layer_name="model.layers.0.self_attn.attn",
                storage_index=0,
                base_offset_bytes=0,
                block_stride_bytes=page_tokens * 1024,
                num_blocks=16,
                regions=(
                    rpm.RegionSpec(
                        name="fa_payload",
                        offset_bytes=0,
                        stride_bytes=1024,
                        unit_bytes=512,
                        num_units=page_tokens,
                        units_per_stride=2,
                    ),
                ),
                dtype_tag="float8_e4m3fn",
            )
        ],
    )


class RaidenCrossGenLayoutFingerprintTest(unittest.TestCase):
    """Verifies layout fingerprint behavior across Flag-OFF, Flag-ON, and edge cases."""

    def setUp(self):
        self.manifest = _synthetic_fa_manifest()
        self.valid_layout = ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 8)

    def test_exact_golden_backward_compatibility(self):
        """Verifies exact bit-for-bit SHA-256 match against production golden."""
        fp, payload = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 0),
            package_version=lambda pkg: {
                "torch_tpu": "0.1.1.dev20260630090312",
                "libtpu": "0.0.42.1",
            }[pkg],
            allow_cross_generation=False,
        )
        expected = "491e9b375a38f65d175e6e38919d29af59bf5149a2c617d0bea69872e3431b77"
        self.assertEqual(fp, expected)

    def test_flag_off_preserves_strict_legacy_behavior(self):
        """Flag OFF (default): Fingerprints diverge when environment packages differ."""
        # Simulate node 1 (e.g. v6e host)
        fp_node1, payload_node1 = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: self.valid_layout,
            package_version=lambda pkg: f"node1-{pkg}-v1.0",
            allow_cross_generation=False,
        )
        # Simulate node 2 (e.g. v7x host with different libtpu build)
        fp_node2, payload_node2 = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: self.valid_layout,
            package_version=lambda pkg: f"node2-{pkg}-v2.0",
            allow_cross_generation=False,
        )

        self.assertNotEqual(fp_node1, fp_node2)
        self.assertEqual(payload_node1["torch_tpu"], "node1-torch_tpu-v1.0")
        self.assertEqual(payload_node2["torch_tpu"], "node2-torch_tpu-v2.0")

    def test_flag_on_matches_semantic_physical_invariants(self):
        """Flag ON: Identical physical layouts match across heterogeneous host packages."""
        fp_v6e, payload_v6e = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: self.valid_layout,
            package_version=lambda pkg: "v6e-trillium-custom-driver",
            allow_cross_generation=True,
        )
        fp_v7x, payload_v7x = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: self.valid_layout,
            package_version=lambda pkg: "v7x-ghostfish-custom-driver",
            allow_cross_generation=True,
        )

        # Wire hash matches, enabling Raiden C++ controller admission
        self.assertEqual(fp_v6e, fp_v7x)
        # Observability payloads preserve node-local diagnostics
        self.assertEqual(payload_v6e["torch_tpu"], "v6e-trillium-custom-driver")
        self.assertEqual(payload_v7x["torch_tpu"], "v7x-ghostfish-custom-driver")

    def test_dependency_injection_custom_policy(self):
        """Custom LayoutFingerprintPolicy can be injected directly."""
        class StubFixedPolicy:
            def compute(self, payload: dict[str, Any]) -> str:
                return "stubbed-policy-fingerprint"

        fp, _ = rlf.measured_fa_layout_fingerprint(
            self.manifest,
            layout_getter=lambda _: self.valid_layout,
            policy=StubFixedPolicy(),
        )
        self.assertEqual(fp, "stubbed-policy-fingerprint")

    def test_flag_on_fails_closed_on_physical_layout_divergence(self):
        """Flag ON still strictly rejects any physical tile or minor-to-major mismatch."""
        # Non-matching tile ((8, 128) instead of (4, 128))
        invalid_tile_layout = ([4, 3, 2, 1, 0], [[8, 128], [4, 1]], 8)
        with self.assertRaisesRegex(RuntimeError, "tile gate"):
            rlf.measured_fa_layout_fingerprint(
                self.manifest,
                layout_getter=lambda _: invalid_tile_layout,
                allow_cross_generation=True,
            )

        # Non-matching minor-to-major dimension
        invalid_dim_layout = ([3, 4, 2, 1, 0], [[4, 128], [4, 1]], 8)
        with self.assertRaisesRegex(RuntimeError, "minor-to-major"):
            rlf.measured_fa_layout_fingerprint(
                self.manifest,
                layout_getter=lambda _: invalid_dim_layout,
                allow_cross_generation=True,
            )


if __name__ == "__main__":
    unittest.main()
