# Copyright 2025 Google LLC
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
"""Deterministic (seed, position)-keyed uniforms and Gumbel noise.

Gumbel-coupled speculative sampling needs the drafter and the verifier to see
the SAME random draws for a given (request, position): the drafter samples
token x ~ q via Gumbel-max, the verifier re-derives its accept threshold and
residual resample from noise keyed identically. Upstream vLLM keys Triton's
Philox with `tl.randint(seed, pos)`; TPU has no Triton, so this module derives
the noise from a splitmix64-style integer hash in pure torch ops that XLA
compiles. The two sides only need to agree with EACH OTHER, not with the CUDA
bitstream.

All functions are shape-static and branch-free (safe inside
``torch.compile(backend="tpu", fullgraph=True)``).
"""

import numpy as np
import torch

# splitmix64 multipliers, reinterpreted as signed int64 (torch has no uint64;
# int64 arithmetic wraps to the same bits).
_MIX_A = int(np.int64(np.uint64(0xBF58476D1CE4E5B9)))
_MIX_B = int(np.int64(np.uint64(0x94D049BB133111EB)))
_GOLDEN = int(np.int64(np.uint64(0x9E3779B97F4A7C15)))

# Smallest uniform this generator emits; keeps -log(-log1p(-u)) finite.
_U_MIN = 1.0 / (1 << 24)


def _mix64(x: torch.Tensor) -> torch.Tensor:
    """Finalizer of splitmix64: bijective avalanche mix on int64 bits."""

    # Arithmetic >> on negative int64 drags sign bits in; mask the shifted
    # value back to a logical shift.
    def _lshr(v: torch.Tensor, k: int) -> torch.Tensor:
        return (v >> k) & ((1 << (64 - k)) - 1)

    x = (x ^ _lshr(x, 30)) * _MIX_A
    x = (x ^ _lshr(x, 27)) * _MIX_B
    x = x ^ _lshr(x, 31)
    return x


def _keyed_bits(seeds: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Per-row int64 key from (seed, position). Both [n] (int64/int32)."""
    return _mix64(
        seeds.to(torch.int64) ^ _mix64(pos.to(torch.int64) + _GOLDEN))


def _bits_to_uniform(bits: torch.Tensor) -> torch.Tensor:
    """Map int64 hash bits to float32 uniform in [_U_MIN, 1)."""
    # Take 24 high-quality bits (fits float32 mantissa exactly).
    u24 = (bits >> 16) & 0xFFFFFF
    u = u24.to(torch.float32) * (1.0 / (1 << 24))
    return torch.clamp(u, min=_U_MIN)


def keyed_uniform(seeds: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Scalar uniform per row, keyed by (seed[i], pos[i]).

    Args:
      seeds: [n] int64 per-row seeds.
      pos: [n] int32/int64 per-row position keys.

    Returns:
      [n] float32 uniforms in (0, 1).
    """
    return _bits_to_uniform(_keyed_bits(seeds, pos))


def keyed_uniform_vector(seeds: torch.Tensor, pos: torch.Tensor,
                         num_cols: int) -> torch.Tensor:
    """[n, num_cols] uniforms; column j is keyed by (seed[i], pos[i], j)."""
    key = _keyed_bits(seeds, pos).unsqueeze(-1)  # [n, 1]
    cols = torch.arange(num_cols, dtype=torch.int64,
                        device=seeds.device).unsqueeze(0)  # [1, num_cols]
    return _bits_to_uniform(_mix64(key + cols * _GOLDEN))


def keyed_gumbel_noise(seeds: torch.Tensor, pos: torch.Tensor,
                       num_cols: int) -> torch.Tensor:
    """[n, num_cols] standard Gumbel noise, keyed by (seed[i], pos[i], col).

    Uses -log(-log1p(-u)) so the winning tail (u -> 0) sits where float32
    resolution is fine, mirroring upstream gumbel.py's precision note.
    """
    u = keyed_uniform_vector(seeds, pos, num_cols)
    return -torch.log(-torch.log1p(-u))
