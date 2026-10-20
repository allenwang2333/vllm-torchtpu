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
"""QSA request segmentation using the runner's decode-first ragged ordering."""

import jax
import jax.numpy as jnp


def decode_prefix(starts: jax.Array) -> jax.Array:
    """Count the leading one-token requests; leave later/empty rows in place."""
    return jnp.sum(jnp.cumprod((jnp.diff(starts) == 1).astype(jnp.int32)))
