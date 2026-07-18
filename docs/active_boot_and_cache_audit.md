# DeepSeek-V4 Active Boot and Weight Serialization Cache Audit

This document details the configuration, modifications, and debugging checkpoints for the active DeepSeek-V4-Flash TPU integration run. It serves as a technical audit record to verify the correctness of the caching system, sharding topology, and cluster volume allocations.

---

## 1. Active Run Environment & Topology

The active inference server is running on a GKE TPU v7x-8 slice under JobSet controller supervision:
*   **Active Pod ID:** `dsv4-inference-v7x-8-vllm-server-0-0-w8hsj`
*   **Sharding Strategy:** Tensor Parallelism size = 8 (`TP=8`), Expert Parallelism size = 8 (`EP=8`), combined with Data Parallel Attention (`enable_dp_attention: true`).
*   **Quantization Spec:** FP8 block-quantization on linear projection layers; 8-bit checkpoint weights are requantized host-side (CPU) during load to 4-bit (`float4_e2m1fn`) block-512 precision for MoE expert weights.
*   **Storage layout:**
    *   `/mnt/workspace`: 1 TB Shared persistent NFS drive (holds codebase and HuggingFace cache snapshots/symlinks).
    *   `/mnt/local-cache`: 500 GB Local NVMe SSD mount (EmptyDir volume). Holds local JIT compiler caches (`inductor_cache`, `triton_cache`, `jax_cache`), raw HuggingFace blobs, and the pre-quantized weight cache.

---

## 2. Issue 1: Weight Caching Omission (Resolved)

### Root Cause Analysis
Although weight serialization load/save loops were present in `fp8.py` and `mxfp4.py`, the cache directories remained empty. We identified a variable propagation mismatch in the model registration hooks:
1. The loader locates the cache path using:
   $$\text{cache\_path} = \text{os.path.join(VLLM\_FP4\_CACHE\_DIR, f"rank\_\{rank\}\_layer\_\{layer\_idx\}.pt")}$$
2. The index `layer_idx` was retrieved via `getattr(layer, "layer_idx", None)`.
3. In `deepseek_v4_patch.py`, the loading hook registered `layer_idx` on the parent MoE block `l.ffn`:
   ```python
   for idx, l in enumerate(self.model.layers):
       if hasattr(l, "ffn"):
           l.ffn.layer_idx = idx
   ```
4. However, the quantization engine in `mxfp4.py` receives the inner `FusedMoE` instance (`l.ffn.experts`). Since `layer_idx` was never set on `experts`, it resolved to `None`. The caching logic silently bypassed writing the files without raising errors.

### Resolution Code Change
We updated the model patcher in [deepseek_v4_patch.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py) to propagate the layer index to the inner `FusedMoE` module:

```diff
     def patched_load_weights(self, weights):
         for idx, l in enumerate(self.model.layers):
             if hasattr(l, "ffn"):
                 l.ffn.layer_idx = idx
+                if hasattr(l.ffn, "experts"):
+                    l.ffn.experts.layer_idx = idx
```

Now, `mxfp4.py` and `fp8.py` successfully resolve the layer index and trigger serialization.

---

## 3. Issue 2: Kubernetes Pod Eviction (Resolved)

### Root Cause Analysis
During serialization on pod `rs4f8`, the pod crashed with exit code 137, and GKE reported:
`Warning  Evicted  kubelet  Usage of EmptyDir volume "local-ssd-volume" exceeds the limit "250Gi".`

Physical memory sizing map:
1.  **HuggingFace Blobs:** `/mnt/local-cache/hf_cache` stores the model's raw safetensors checkpoint files. Total size = **145 GB**.
2.  **JIT Compiler Caches:** `/mnt/local-cache/inductor_cache` + `jax_cache` $\approx$ **15 GB**.
3.  **Pre-quantized Weights Cache:** `/mnt/local-cache/dsv4-flash-fp4` stores serialized `.pt` files.
    *   For 1 layer, the combined MoE weights per rank are ~0.4 GB.
    *   Since all 8 processes run on the same physical host node, they all write to the same EmptyDir directory mount.
    *   Total weight cache footprint = $61 \text{ layers} \times 8 \text{ ranks} \times 0.4\text{ GB/file} \approx$ **195 GB**.
4.  **Accumulated Footprint:** $145\text{ GB} + 15\text{ GB} + 195\text{ GB} = \mathbf{355\text{ GB}}$, which strictly exceeds the `250Gi` limit.

### Resolution Code Change
We edited the GKE JobSet recipe YAML on the host VM (`patemotter-cdk`) to increase the local SSD EmptyDir limit to **500Gi**, providing a 145 GB safety margin:

```diff
               - name: local-ssd-volume
                 emptyDir:
-                  sizeLimit: 250Gi
+                  sizeLimit: 500Gi
```

This successfully prevents eviction during serialization.

---

## 4. File Modifications Checklist

Verify the following files in your code audit:

### 1. `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`
Propagates the unique layer index to `ffn.experts` to enable cache file indexing.
*   [File Link](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py#L88-L92)

### 2. `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`
Implements the load/save cache checks for FP4 expert weights to bypass CPU requantization overhead on restarts.
*   [File Link](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py#L218-L245)

### 3. `recipes/experimental/torchtpu-vllm/dsv4-inference-v7x-8-patemotter.yml` (on `patemotter-cdk` VM)
Sets local SSD cache volume limits to `500Gi`.

---

## 5. Active Verification Checkpoints

To monitor serialization progress on the active pod:
1.  **Check file count:** Run the following command. The final count should be **488** files (8 ranks $\times$ 61 layers):
    ```bash
    kubectl exec -n cloud-devkit-jobs dsv4-inference-v7x-8-vllm-server-0-0-w8hsj -c vllm-server -- ls -al /mnt/local-cache/dsv4-flash-fp4/ | wc -l
    ```
2.  **Verify GCS Sync on exit:** When weight processing completes and the model enters compiler warmup, the launcher script will trigger the GCS copy step:
    ```bash
    gcloud storage cp -r /mnt/local-cache/dsv4-flash-fp4/* gs://cloud-tpu-inference-test-patemotter-archive/workspace/dsv4-flash-fp4/
    ```
    Verify that the files appear in the GCS bucket `gs://cloud-tpu-inference-test-patemotter-archive/workspace/dsv4-flash-fp4/`.
