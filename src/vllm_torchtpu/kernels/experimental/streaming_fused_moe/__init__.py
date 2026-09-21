# SPDX-License-Identifier: Apache-2.0
"""Experimental streaming FP8/FP4 MoE, based on the original V2 transport."""
from .layer import streaming_fused_moe

__all__ = ["streaming_fused_moe"]
