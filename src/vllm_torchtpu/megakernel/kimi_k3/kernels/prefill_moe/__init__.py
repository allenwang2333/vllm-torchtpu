# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed EP16/TP2 prefill MoE kernels: the FP32-reduction schedule for
512- through 8192-token windows (``fused``), the BF16-reduction layer for
4096- and 8192-token windows (``layer``) and the Pallas pieces they share."""
