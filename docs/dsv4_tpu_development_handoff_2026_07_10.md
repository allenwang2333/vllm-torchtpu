# DeepSeek-V4 TPU Integration Handoff Document

This document summarizes the current status of the DeepSeek-V4 TPU integration, active bugs resolved, VM compilation findings, and pending verification actions for the next conversation session.

---

## 1. Executive Summary & Current Status
*   **Active GKE Job Status**: A new JobSet `dsv4-verify-v7x-8` has been submitted to the GKE cluster (v7x-8). It is currently in a **suspended state (`SUSPENDED: true`)** in Kueue's StrictFIFO queue (`lq-patemotter-google-com`), waiting for cluster resources (TPU v7x chips are currently 100% saturated).
*   **Completed Fixes**:
    1.  **GCS Cache Pollution Fix**: Discovered that exactly 20 out of 46 `.safetensors` files stored in the GCS archive bucket were 0-byte placeholders because they were copied as symlinks rather than dereferenced files. We patched the GKE JobSet manifest ([examples/dsv4-inference-v7x-8-patemotter.yml](file:///usr/local/google/home/patemotter/torchtpu-vllm/examples/dsv4-inference-v7x-8-patemotter.yml#L23-L24)) to copy all files dereferenced (`cp -fL`) directly from the NFS share `/mnt/workspace/` before starting, repairing the local disk cache.
    2.  **Defensive Startup Guard**: Added a file validation guard in [examples/verify_dsv4.py](file:///usr/local/google/home/patemotter/torchtpu-vllm/examples/verify_dsv4.py#L26-L41) that scans the model snapshot directory and throws a descriptive runtime exception if it finds any 0-byte weight files, preventing cryptic Rust/Safetensor deserialization crashes.

---

## 2. Environment & VM Infrastructure

*   **Local Debug Host (TPU VM)**: `patemotter-tpu-v6e-8-3` (v6e-8 architecture).
*   **CDK Job Submission Node**: `patemotter-cdk` VM.
*   **NFS Share Volume**: Mounted on `/workspace` on `patemotter-cdk` VM, and on `/mnt/workspace` on the TPU VM.
*   **GCS Archive Bucket**: `gs://cloud-tpu-inference-test-patemotter-archive/workspace/`
*   **Custom Compiler (libtpu.so)**: Patched build resides at `/mnt/workspace/libtpu.so` on both the VM and the NFS share, activated via `TPU_LIBRARY_PATH`.

---

## 3. Core Technical Findings & VM Compilation Limits
During compilation tests on the v6e-8 VM (`test_fp4_bitcast.py`), we isolated a hardware-locked constraint in the compilation pipeline:
*   JAX/Mosaic's `pltpu.bitcast(x, float4)` operates on **axis 0** (the major dimension) of the vector register layout.
*   Even when structuring the matrix block so that K is axis 0 (avoiding explicit reshape operators inside VMEM), performing software dequantization on sub-elements (`astype(jnp.float32)`) lowers to a `"tpu.unpack_subelements"` MLIR operation:
    `%56 = "tpu.unpack_subelements"(%53) : (vector<8x128x8xf4E2M1FN>) -> vector<8x128xf32>`
*   This compilation fails on the **TPU v6e target** with `Unsupported type 'vector<8x128x8xf4>'` because the v6e VPU compiler driver lacks lowering rules for sub-8-bit layout unpacking.
*   **Conclusion**: Custom FP4 software dequantization (unpacking sub-elements inside VPU registers) is physically locked out of the v6e compiler backend and can only compile/run on TPU v7+ hardware targets (which GKE uses).

---

## 4. Pending Actions for Next Session

1.  **Monitor GKE Workload Admission**:
    Check Kueue status and GKE pods until the JobSet is admitted and unsuspended:
    ```bash
    ssh patemotter-cdk "kubectl get jobsets -n cloud-devkit-jobs"
    ssh patemotter-cdk "kubectl get pods -n cloud-devkit-jobs"
    ```
2.  **Stream Run Logs**:
    Once the pod starts running (`STATUS: Running`), stream the container logs using the pod name:
    ```bash
    ssh patemotter-cdk "kubectl logs -f <pod-name> -n cloud-devkit-jobs"
    ```
3.  **Inspect Output Token IDs & Logprobs**:
    *   Verify that model weight loading succeeds without `SafetensorError`.
    *   Inspect the output generation logs:
        *   Prompt: `'The quick brown fox jumps over the lazy dog'`
        *   Verify if the output text is valid English or garbage tokens.
        *   Verify the prompt and generation logprobs are correct.
