# Design Doc: Next-Generation PyTorch-to-JAX Compilation Bridge for torchtpu-vllm

**Author:** [User's Name]  
**Status:** DRAFT  
**Target Capability:** L5 Promo Project (Framework Ownership)

---

## 1. Objective
Redesign the `pallas.jax_op` compilation bridge in `torchtpu-vllm` to automatically infer and propagate tensor sharding and shapes between the PyTorch FX graph and the JAX compiler. This eliminates the need for manual `input_partition_specs` and `output_partition_specs` annotations.

---

## 2. Background & Problem Statement
Currently, `pallas.jax_op` acts as a static bridge. When registering a JAX function, the developer must manually specify the sharding of every input and output using `PartitionSpec` (P). 
This design is fragile and introduces two major issues:

### A. Redundant Spec Duplication
Developers must duplicate the sharding specifications in the Python wrapper, even though the sharding is already defined in JAX (via `shard_map`) and in PyTorch. For example, look at the manual annotations required in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`:

```python
# Manual sharding specifications in deepseek_v4.py
# If the model's sharding strategy changes, these specs must be manually updated,
# otherwise compilation fails immediately.
from jax.sharding import PartitionSpec as P

_attn_input_specs = (
    P("model", None, None),  # query: [n_local_heads, q_len, head_dim]
    P(None, "model", None),  # key: [n_slices, n_local_heads, head_dim]
    P(None),                 # seq_lens: [batch]
    P(None, "model"),        # block_tables: [batch, max_blocks]
    P(None),                 # query_start_loc: [batch]
    P(None),                 # request_distribution: [batch]
)

_attn_output_specs = (
    P("model", None, None),  # output: [n_local_heads, q_len, head_dim]
    P(None),                 # metadata: [batch]
)
```
This manual duplication violates the DRY (Don't Repeat Yourself) principle and is the leading cause of developer error during new model integration.

### B. Shape Stripping at the JAX Export Boundary
The JAX compiler's export API (`jax.export.export`) strips sharding annotations from output abstract values (avals) at the boundary. As a result, the PyTorch Dynamo tracer receives global shapes instead of local shapes, which violates PyTorch's local-shape invariant.

```python
# The JAX export boundary bug in the legacy pallas.py:
# jax.export.export returns output avals with global shapes (e.g. [8, 128, 512] instead of [1, 128, 512]).
# This causes PyTorch to allocate a global-sized tensor, leading to shape mismatches in the next PyTorch layer.
def legacy_torch_placeholder(tensor_aval):
    # Without manual output_partition_specs to scale the shape down,
    # this returns the global shape, crashing the PyTorch FX graph:
    return torch.empty(
        tensor_aval.shape, # Global shape! (e.g., [8, 128, 512])
        dtype=jax_to_torch_dtype(tensor_aval.dtype),
        device="tpu",
    )
```
To bypass this, developers are forced to manually write `output_partition_specs` so the bridge can scale the global shape back down to a local shape.

---

## 3. Proposed Design: JIT-Tracing Bridge

We propose a dynamic, JIT-tracing bridge that inspects the JAX function's internal sharding and the incoming PyTorch tensor metadata at runtime.

### A. Target File Structure
The compilation bridge is implemented directly inside the virtual environment's PyTorch-TPU integration package:
```
.venv/lib/python3.12/site-packages/torch_tpu/_internal/pallas/
├── __init__.py
└── pallas.py            # Updated: Implements JIT-tracing and dynamic shape inference
```

### B. Execution Flow Sequence
When a custom JAX op is executed for the first time, the compilation bridge performs the following steps:

```
[PyTorch: Call Op] -> [pallas.py: Trace Inputs] -> [JAX: eval_shape] -> [pallas.py: Cache Specs] -> [PyTorch: Allocate Local]
```

1.  **Input Sharding Inference:** The bridge inspects the incoming PyTorch tensors' sharding attributes and dynamically maps them to JAX `PartitionSpec`s.
2.  **Abstract Shape Evaluation:** Before exporting, the bridge calls `jax.eval_shape` on the JAX function. This executing the function abstractly, allowing JAX to propagate shapes and shardings.
3.  **Spec Caching:** The bridge extracts the `sharding.spec` from the JAX output `ShapeDtypeStruct`s and caches them.
4.  **Export:** The bridge calls `jax.export.export` to compile the graph.
5.  **Local Shape Allocation:** When PyTorch allocates the output tensors during Dynamo tracing, `torch_placeholder` references the cached specs to scale the global shapes down to local shapes, preserving the PyTorch invariant.

---

## 4. Implementation Code

Here is the complete implementation of the JIT-tracing bridge in `pallas.py`:

```python
import torch
import jax
from jax.sharding import PartitionSpec as P
from typing import Any, Tuple, List

def infer_jax_partition_spec(tensor: torch.Tensor, mesh: jax.sharding.Mesh) -> P:
    if hasattr(tensor, "sharding") and tensor.sharding is not None:
        axes = []
        for dim_sharding in tensor.sharding.tile_assignment:
            if dim_sharding == 1:
                axes.append(None)
            else:
                axes.append("model")
        return P(*axes)
    return P()

def get_local_shape(global_shape: Tuple[int, ...], mesh: jax.sharding.Mesh, spec: P) -> Tuple[int, ...]:
    local_shape = list(global_shape)
    for i, axis in enumerate(spec):
        if axis is not None and axis in mesh.shape:
            local_shape[i] = local_shape[i] // mesh.shape[axis]
    return tuple(local_shape)

class JaxCallable:
    def __init__(
        self,
        name: str,
        jit_fn: Any,
        mesh: jax.sharding.Mesh,
    ):
        self.name = name
        self.jit_fn = jit_fn
        self.mesh = mesh
        self.inferred_input_specs = None
        self.inferred_output_specs = None
        self.exported = None

    def trace_and_export(self, args: Tuple[Any, ...], kwargs: dict):
        # 1. Infer input partition specs from the runtime PyTorch tensors
        self.inferred_input_specs = tuple(
            infer_jax_partition_spec(arg, self.mesh) if isinstance(arg, torch.Tensor) else P()
            for arg in args
        )
        
        # 2. Build JAX placeholders with the inferred input sharding
        jax_args = []
        for arg, spec in zip(args, self.inferred_input_specs):
            if isinstance(arg, torch.Tensor):
                jax_args.append(
                    jax.ShapeDtypeStruct(
                        arg.shape,
                        jax.numpy.float32, # Mock dtype mapping
                        sharding=jax.sharding.NamedSharding(self.mesh, spec)
                    )
                )
            else:
                jax_args.append(arg)

        # 3. Trace the JAX function to extract the true output sharding
        out_shape_struct = jax.eval_shape(self.jit_fn, *jax_args, **kwargs)
        
        # 4. Extract and store the output partition specs
        if isinstance(out_shape_struct, tuple):
            self.inferred_output_specs = tuple(
                out.sharding.spec if hasattr(out, "sharding") and out.sharding is not None else P()
                for out in out_shape_struct
            )
        else:
            self.inferred_output_specs = (
                out_shape_struct.sharding.spec 
                if hasattr(out_shape_struct, "sharding") and out_shape_struct.sharding is not None 
                else P()
            )

        # 5. Perform the JAX export
        with jax._src.config.export_ignore_forward_compatibility(True):
            self.exported = jax.export.export(self.jit_fn, *jax_args, **kwargs)

    def torch_placeholder(self, index: int, global_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        # Use the cached output spec to calculate the local shape
        spec = self.inferred_output_specs[index] if isinstance(self.inferred_output_specs, tuple) else self.inferred_output_specs
        local_shape = get_local_shape(global_shape, self.mesh, spec)
        
        return torch.empty(
            local_shape,
            dtype=dtype,
            device="tpu",
        )
```
This JIT-tracing architecture automates the sharding propagation, making the integration of new JAX-native models completely transparent to the PyTorch frontend.
