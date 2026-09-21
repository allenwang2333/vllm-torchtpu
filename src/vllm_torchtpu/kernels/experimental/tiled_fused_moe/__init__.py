# SPDX-License-Identifier: Apache-2.0
"""Experimental output-tiled FP8 MoE, based on the original V2 transport."""
from .layer import tiled_fused_moe

__all__ = ["tiled_fused_moe"]
