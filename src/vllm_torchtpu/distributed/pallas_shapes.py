# SPDX-License-Identifier: Apache-2.0
"""Shape conversion for native Pallas ops with multi-axis PartitionSpecs.

TorchTPU 20260917 handles explicit output specs, but its shape conversion
assumes one mesh axis per dimension. GDN state is sharded on (tp, pcp) on
one dimension. Install the same conversion for native input and output
placeholders; keep JaxCallable, packed dtype handling and dispatch upstream.
"""

import math


def _convert_partitioned_shape(shape, mesh, partition_spec, *, to_global):
    if mesh is None or partition_spec is None or len(partition_spec) == 0:
        return shape if isinstance(shape, tuple) else tuple(shape)
    mesh_shape = mesh.shape
    spec_len = len(partition_spec)
    result = []
    for index, size in enumerate(shape):
        axes = partition_spec[index] if index < spec_len else None
        if axes is None:
            result.append(size)
            continue
        if isinstance(axes, tuple):
            factor = math.prod(int(mesh_shape[axis]) for axis in axes)
        else:
            factor = int(mesh_shape[axes])
        if not to_global and size % factor:
            raise ValueError(
                f"Global dimension {size} is not divisible by mesh factor {factor}."
            )
        result.append(size * factor if to_global else size // factor)
    return tuple(result)


def get_global_shape(local_shape, mesh, partition_spec):
    return _convert_partitioned_shape(local_shape, mesh, partition_spec, to_global=True)


def get_local_shape(global_shape, mesh, partition_spec):
    return _convert_partitioned_shape(
        global_shape, mesh, partition_spec, to_global=False
    )


def install_pallas_shape_conversion():
    """Apply the shared conversion before custom ops are registered."""
    from torch_tpu._internal.pallas import pallas

    pallas.get_global_shape = get_global_shape
    pallas.get_local_shape = get_local_shape
