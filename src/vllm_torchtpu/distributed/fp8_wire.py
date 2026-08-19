# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8 payloads for the MoE activation collectives.

Halves the bytes an activation all-gather puts on ICI: quantize each token row
to ``float8_e4m3fn`` with an fp32 scale per ``WIRE_BLOCK``-wide slice of the
hidden dimension -- the whole row by default -- gather the fp8 rows and the
scales as two collectives, and rebuild the original dtype on arrival.

The scale granularity is a compile-time trade, and the cheap side is the one
that ships. ``gmm_v2`` re-derives its own lhs scale from the dequantized rows,
per row x ``LHS_QUANT_BLOCK_SIZE``-wide k block; a wire that used the *same*
blocks would hand it back the scale it is about to derive, but expressing that
needs a reshape that relayouts the payload or eight sliced multiplies and a
concatenate -- 98 ops and 480 MB of temp on the receive side at the gathered
prefill shape, against 6 ops and 0 MB for one scale per row. The price of the
row form is numerical, and it is not free: rel-L2 of the layer output against an
fp32 reference is 4.65e-2 for bf16 dispatch, 4.62e-2 for the block form and
5.93e-2 per row; against main's own output the row form moves 7.23e-2 where one
bf16 ulp of input jitter moves 4.98e-2.

Rows and scales travel as **two** collectives, not one packed buffer. Packing
both into uint8 saves an 8-rank all-gather's ~9 us of fixed cost, and costs the
asynchronous form: XLA compiles a uint8 all-gather as a plain synchronous op
where bf16 and float8_e4m3fn get the sparsecore-offloaded pair that overlaps the
compute it sits between. Both device runs that packed lost ~60%. The packed form
stays reachable behind ``TPU_MOE_DISPATCH_FP8_WIRE=uint8`` so the two can be
measured against each other; see ``packed_wire``.

Everything here is 2-D elementwise work on lane-tile-aligned slices, which is
why it fuses into the collective's producer/consumer: reshaping ``[T, H]`` to
``[T, H/B, B]`` to express the same arithmetic instead costs a relayout of the
whole payload (AOT peak temp 672 MB vs 81 MB for the slice form).

Ported from tpu-inference's ``_apply_all_gather_fp8``
(``tpu_inference/layers/common/fused_moe_gmm.py``), which scales per row and
dequantizes right after the collective. Its degenerate-row rule is made
explicit here: that implementation divides by a zero scale and ships NaN for an
all-zero row (padding), which the ``amax > 0`` guard below avoids.
"""

import torch

import vllm_torchtpu.envs as envs

WIRE_DTYPE = torch.float8_e4m3fn
WIRE_MAX = float(torch.finfo(WIRE_DTYPE).max)
# One scale per row. Matching kernels.megablox.gmm_v2.LHS_QUANT_BLOCK_SIZE (512)
# is more accurate but does not compile into a single pass: at the gathered
# prefill shape the receive side is 98 ops and 480 MB of temp against 6 ops and
# 0 MB per row. ``None`` means "the whole row"; the code below takes either.
WIRE_BLOCK = None


def _blocks(width: int, block: int) -> range:
    return range(0, width, block)


def _block_for(width: int) -> int:
    if WIRE_BLOCK is None or width % WIRE_BLOCK:
        return width
    return WIRE_BLOCK


def packed_wire() -> bool:
    """Whether the payload crosses as one uint8 buffer instead of fp8 + scales.

    Packing saves a collective (~9 us of fixed cost) and costs the async form:
    XLA compiles a uint8 all-gather as a plain synchronous op, where bf16 and
    float8_e4m3fn get the sparsecore-offloaded pair. Both device runs that
    used the packed wire lost ~60%, which is why it is not the default.
    """
    return envs.TPU_MOE_DISPATCH_FP8_WIRE == "uint8"


def encode(x: torch.Tensor) -> torch.Tensor:
    """``[..., H]`` floats -> one uint8 wire buffer of fp8 rows + fp32 scales."""
    q, scale = quantize_rows(x)
    return torch.cat([q.view(torch.uint8), scale.view(torch.uint8)], dim=-1)


def decode(payload: torch.Tensor, width: int,
           dtype: torch.dtype) -> torch.Tensor:
    """Inverse of ``encode`` for a payload whose rows are ``width`` wide."""
    q = payload[..., :width].view(WIRE_DTYPE)
    scale = payload[..., width:].view(torch.float32)
    return dequantize_rows(q, scale, dtype)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., H]`` to fp8 with one fp32 scale per hidden block.

    Returns ``(q, scale)`` with ``scale`` shaped ``[..., H/block]``;
    ``x ~= q.float() * scale``, exactly zero where the input is.
    """
    width = x.shape[-1]
    block = _block_for(width)
    x_f32 = x.to(torch.float32)
    q, scales = [], []
    for lo in _blocks(width, block):
        chunk = x_f32[..., lo:lo + block]
        amax = chunk.abs().amax(dim=-1, keepdim=True)
        # A zero block (padding) would give scale 0; keep 1 so it stays zero
        # instead of turning into NaN on the wire.
        scale = torch.where(amax > 0, amax / WIRE_MAX, torch.ones_like(amax))
        q.append((chunk / scale).clamp(-WIRE_MAX, WIRE_MAX).to(WIRE_DTYPE))
        scales.append(scale)
    if len(q) == 1:  # no concatenate: it would materialize the payload
        return q[0], scales[0]
    return torch.cat(q, dim=-1), torch.cat(scales, dim=-1)


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
    width = q.shape[-1]
    block = width // scale.shape[-1]
    if block == width:  # one fused pass, no buffer-forming op
        return (q.to(torch.float32) * scale).to(dtype)
    out = [
        q[..., lo:lo + block].to(torch.float32) * scale[..., i:i + 1]
        for i, lo in enumerate(_blocks(width, block))
    ]
    return torch.cat(out, dim=-1).to(dtype)
