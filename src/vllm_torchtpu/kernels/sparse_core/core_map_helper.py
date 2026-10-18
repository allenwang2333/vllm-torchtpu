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
"""Run single-mesh SparseCore kernels that produce fresh outputs.

``kernel`` takes the ``pl.kernel`` arguments by keyword (``out_type``,
``scratch_types`` as a sequence or a dict) and returns a jitted program.
"""

import jax
from jax.experimental import pallas as pl


def kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None,
):
    """Builds a jitted ``pl.kernel`` program over one mesh.

    ``body(*operand_refs, *out_refs, *scratch_refs)`` runs with a dict of
    scratch types passed as keyword refs. The outputs are the kernel's own,
    buffers the compiler allocates for the call: an allocation made in the
    program (``lax.empty`` written in place through a ref) is identical in
    every inlined copy of the program, so the torch bridge's
    common-subexpression pass merges one model's layers onto a single buffer,
    which then has to be copied before every in-place write.
    """
    program = pl.kernel(
        body,
        out_type,
        mesh=mesh,
        scratch_types=scratch_types,
        compiler_params=compiler_params,
        interpret=interpret,
        cost_estimate=cost_estimate,
        debug=debug,
        name=name,
        metadata=metadata,
    )
    return jax.jit(program)
