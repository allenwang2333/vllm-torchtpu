# SPDX-License-Identifier: Apache-2.0
"""Shape conversion for native Pallas ops with multi-axis PartitionSpecs.

TorchTPU 20260917 handles explicit output specs, but its shape conversion
assumes one mesh axis per dimension. GDN state is sharded on (tp, pcp) on
one dimension. Install the same conversion for native input and output
placeholders; keep JaxCallable, packed dtype handling and dispatch upstream.
"""


def _convert_partitioned_shape(shape, mesh, partition_spec, *, to_global):
    if mesh is None or not partition_spec:
        return shape if isinstance(shape, tuple) else tuple(shape)
    mesh_shape = mesh.shape
    result = list(shape)
    shape_len = len(result)
    for index, axes in enumerate(partition_spec):
        if axes is None or index >= shape_len:
            continue
        if isinstance(axes, tuple):
            factor = 1
            for axis in axes:
                factor *= int(mesh_shape[axis])
        else:
            factor = int(mesh_shape[axes])
        size = result[index]
        if to_global:
            result[index] = size * factor
        else:
            if size % factor:
                raise ValueError(
                    f"Global dimension {size} is not divisible by mesh factor {factor}."
                )
            result[index] = size // factor
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
