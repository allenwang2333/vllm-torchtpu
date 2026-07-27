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
"""Donating copy for GDN recurrent state arrays."""

import torch

_gdn_state_copy_op = None

def _build_op():
    import jax
    from torch_tpu._internal import pallas

    def _state_copy_tpu(state: jax.Array, new_state: jax.Array) -> jax.Array:
        # Keep the update under cond so torch_tpu's aliasing validation accepts
        # the donated buffer cleanly.
        return jax.lax.cond(
            True,
            lambda s: s.at[...].set(new_state),
            lambda s: s,
            state,
        )

    op = pallas.jax_op("pallas::gdn_state_copy",
                       _state_copy_tpu,
                       donate_argnums=(0, ))
    op.register_fake(lambda state, new_state: torch.empty_like(state))
    return op

def ensure_op_built() -> None:
    """Force pallas op construction outside the hot path."""
    global _gdn_state_copy_op
    if _gdn_state_copy_op is None:
        _gdn_state_copy_op = _build_op()

def gdn_state_copy(state: torch.Tensor, new_state: torch.Tensor) -> None:
    """In-place state copy out-of-band via Pallas."""
    ensure_op_built()
    _gdn_state_copy_op(state, new_state)
