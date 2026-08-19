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
to ``float8_e4m3fn`` with an fp32 scale per ``wire_block()``-wide slice of the
hidden dimension -- the whole row by default -- gather the fp8 rows and the
scales as two collectives, and rebuild the original dtype on arrival.

The scale granularity is the one knob that changes what the layer computes.
``gmm_v2`` re-derives its own lhs scale from the dequantized rows, per row x
``LHS_QUANT_BLOCK_SIZE``-wide k block, so a per-row wire rounds the payload to
fp8 **twice on different grids** and multiplies the layer's fp8 error by 1.38x.
On the kernel's own grid (``TPU_MOE_DISPATCH_FP8_BLOCK=512``) that second
rounding is a bitwise no-op: every block arrives saturated at ``WIRE_MAX``, the
kernel re-derives the identical scale and its requantization hands back these
exact fp8 codes, so the layer's output is **bit-identical** to the bf16
dispatch it replaces (measured, not argued: tests/distributed/test_fp8_wire.py).

That costs eight sliced multiplies and a concatenate on the receive side, which
is cheap -- the concatenate writes each slice in place. AOT-compiled for tpu7x
at the gathered prefill shape [16384, 4096]: peak temp 81.53 vs 80.53 MB per row
(+1.2%), and XLA's own cycle estimate for the whole dispatch 550.7k vs 473.7k
(+16.3%, i.e. ~+11 us/layer against the ~546 us/layer the fp8 wire saves at that
bucket). The two formulations that look cheaper are not: broadcasting the scale
(``repeat_interleave``) or multiplying against a ``[T, H/B, 1]`` view both make
XLA materialize an f32 ``[T, H/B, B]`` scale and relayout it through a
``sparse-core-data-format-call`` -- 594.5 MB of temp, 7.3x the sliced form.

Rows and scales travel as **two** collectives, not one packed buffer. Packing
both into uint8 saves an 8-rank all-gather's ~9 us of fixed cost, and costs the
asynchronous form: XLA compiles a uint8 all-gather as a plain synchronous op
where bf16 and float8_e4m3fn get the sparsecore-offloaded pair that overlaps the
compute it sits between. Two device runs on the packed wire lost 60.8% and
61.5%, but each changed the scale grid and the receive side as well as the
container, so what attributes it to the container is the compiled form (an AOT
compile shows sync vs async in seconds) plus the two-collective run that gained
+1.1% in the same configuration. The packed form stays reachable behind
``TPU_MOE_DISPATCH_FP8_WIRE=uint8`` so the two can be measured against each
other; see ``packed_wire``.

Everything here is 2-D elementwise work on lane-tile-aligned slices, so the send
side can fuse into the producer of ``hidden_states``. The receive side cannot
fuse into its consumer: that is ``ragged_gather_v2``, a Mosaic custom call, and
XLA cannot fuse an elementwise op into one.

Upstream owns this shape and we should converge on it, not diverge further:
vLLM quantizes before the wire and gathers the scale through the communicator's
``extra_tensors`` channel (``fused_moe/prepare_finalize/naive_dp_ep.py:16-52``),
hands the experts ``(a1q, a1q_scale)`` (``fused_moe/modular_kernel.py``
``FusedMoEKernelMonolithicImpl.apply``), and dequantizes **only** when the
wire's block grid disagrees with the kernel's
(``fused_moe/prepare_finalize/deepep_ll.py:166-179``) -- upstream's own
statement of what ``TPU_MOE_DISPATCH_FP8_BLOCK=512`` buys here. We dequantize on
every path because ``gmm_v2`` takes an ``rhs_scale`` and no ``lhs_scale`` and so
re-derives one; giving it that argument is the change that deletes
``dequantize_rows`` outright. Line numbers are at the pinned vLLM v0.27.0.

Ported from tpu-inference's ``_apply_all_gather_fp8``
(``tpu_inference/layers/common/fused_moe_gmm.py``), which scales per row and
dequantizes right after the collective. Its zero-row behaviour is accidental
rather than stated: ``quantize_tensor`` divides by a zero scale and is saved by
``nan_to_num``'s default ``posinf`` substitution (the ``nan=`` argument it means
to pass lands on ``copy``), so a padding row comes back 0 and not NaN -- checked
under the image's jax 0.10.2, against the claim this docstring used to make. The
guard below states the rule instead of inheriting it, and covers what the
reference does not: an fp32 payload whose ``amax / WIRE_MAX`` underflows to 0.
"""

import torch

import vllm_torchtpu.envs as envs

WIRE_DTYPE = torch.float8_e4m3fn
WIRE_MAX = float(torch.finfo(WIRE_DTYPE).max)


def _blocks(width: int, block: int) -> range:
    return range(0, width, block)


def wire_block() -> int | None:
    """Hidden-axis width one scale covers; ``None`` means the whole row.

    Only two values are useful, and the test asserts it: the row, or exactly
    ``kernels.megablox.gmm_v2.LHS_QUANT_BLOCK_SIZE``. Anything in between
    mismatches the kernel's grid and pays the second rounding anyway.
    """
    return envs.TPU_MOE_DISPATCH_FP8_BLOCK or None


def _block_for(width: int) -> int:
    block = wire_block()
    if block is None or width % block:
        return width
    return block


def packed_wire() -> bool:
    """Whether the payload crosses as one uint8 buffer instead of fp8 + scales.

    Packing saves a collective (~9 us of fixed cost) and costs the async form:
    XLA compiles a uint8 all-gather as a plain synchronous op, where bf16 and
    float8_e4m3fn get the sparsecore-offloaded pair. The two device runs that
    used the packed wire lost 60.8% and 61.5%; each differed from the run it is
    compared against in more than the container, so the attribution rests on
    the compiled form rather than on those two numbers alone.
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
    kernel_grid = block != width
    # On the kernel's own grid, stay in the payload's dtype and multiply by a
    # reciprocal, which is how gmm_v2 spells it (gmm_v2.py:551-557): its lhs
    # requantization then reproduces these codes bit for bit instead of
    # rounding a second time, and the amax of a bf16 block is exact in bf16
    # anyway, so the row form's fp32 upcast buys nothing and costs a pass.
    src = x if kernel_grid else x.to(torch.float32)
    q, scales = [], []
    for lo in _blocks(width, block):
        chunk = src[..., lo:lo + block]
        amax = chunk.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        # A zero block (padding) would give scale 0; keep 1 so it stays zero
        # instead of turning into NaN on the wire. Guard the scale, not the
        # amax: amax / WIRE_MAX can underflow fp32 to 0 while amax > 0.
        scale = amax / WIRE_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        if kernel_grid:
            # No clamp, for the reason the kernel needs none: the block max
            # comes out at WIRE_MAX x (1 + 2^-9), inside e4m3's round-to-
            # nearest range (which reaches WIRE_MAX x 1.036).
            q.append((chunk * (1 / scale.to(x.dtype))).to(WIRE_DTYPE))
        else:
            q.append((chunk / scale).clamp(-WIRE_MAX, WIRE_MAX).to(WIRE_DTYPE))
        scales.append(scale)
    if len(q) == 1:  # no concatenate: it would materialize the payload
        return q[0], scales[0]
    return torch.cat(q, dim=-1), torch.cat(scales, dim=-1)


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
    # TODO: delete this entirely once gmm_v2 accepts an lhs_scale. It exists
    # only because the kernel re-derives a scale it could have been handed;
    # upstream's dispatch keeps the payload quantized end to end for exactly
    # this reason (see the module docstring).
    width = q.shape[-1]
    block = width // scale.shape[-1]
    if block == width:  # one fused pass, no buffer-forming op
        return (q.to(torch.float32) * scale).to(dtype)
    out = [
        q[..., lo:lo + block].to(torch.float32) * scale[..., i:i + 1]
        for i, lo in enumerate(_blocks(width, block))
    ]
    return torch.cat(out, dim=-1).to(dtype)
