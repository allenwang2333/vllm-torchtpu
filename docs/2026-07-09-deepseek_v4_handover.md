# Handover Report: DeepSeek-V4 Gibberish Output Debugging
**Date:** July 9, 2026

## Objective
Isolate and resolve the root cause of the garbage text/gibberish generated during DeepSeek-V4-Flash inference runs on GKE TPU v7x-8 (which occurs even though the model runs to completion without runtime exceptions or NaNs).

---

## Current Status & Diagnoses

### 1. Weight Cache Investigation (Hypothesis 1: Ruled Out)
*   **Action:** We deleted the GCS cache directory `gs://cloud-tpu-inference-test-patemotter-archive/workspace/dsv4-flash-fp4/` to force a clean quantization pass from the raw HuggingFace weights.
*   **Result:** The model performed a fresh host-side CPU FP4 expert quantization and compiled/ran successfully, but the output text remained completely garbage. Stale weights cache layout mismatch is **ruled out**.
*   **Current Hypotheses:**
    *   **Hypothesis 2:** Head sharding / attention layout shuffle mismatch in the custom MLA attention wrapper.
    *   **Hypothesis 3:** KV Cache padding / memory stride index mapping issues in the custom Pallas kernels.
    *   **Hypothesis 4:** Token routing index slicing mismatch in the custom `_fused_moe_sharded_jax` wrapper.

### 2. Active Debugging Setup
*   **Eager Activation Norm Logging:** We modified the attention forward path inside [deepseek_v4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py) to print L2 norms (`torch.linalg.norm`) of `hidden_states`, `qr`, `kv`, `attn_output`, and final block output.
*   **4-Layer Validation Model:** To bypass long JAX compile times and iterate faster, we configured [verify_dsv4.py](file:///Users/patemotter/dsv4/torchtpu-vllm/examples/verify_dsv4.py) to execute eagerly (`VLLM_ENFORCE_EAGER=1`) using a 4-layer configuration.
*   **Random Mock Weights:** We modified [generate_mock_weights.py](file:///Users/patemotter/dsv4/torchtpu-vllm/examples/generate_mock_weights.py) to initialize mock checkpoint parameters with `torch.randn(shape) * 0.02` instead of zeros, allowing non-zero activation norms to propagate and trace.

---

## Active GKE Deployment

We redeployed the GKE JobSet to execute the 4-layer validation pipeline in eager mode.

*   **Namespace:** `cloud-devkit-jobs`
*   **JobSet:** `dsv4-inference-v7x-8`
*   **Pod Name:** `dsv4-inference-v7x-8-vllm-server-0-0-fs252`
*   **Status:** Currently **`Pending`** (FailedScheduling). GKE is temporarily out of TPU capacity because other benchmark jobs took up the nodes. A `kubectl wait` command is running in the background to log readiness once scheduled.

### Execution Flow in GKE Pod:
Once GKE schedules the pod, the container will:
1.  Verify local NFS mount directory `/root/cloud-devkit/torchtpu-vllm/`.
2.  Run `generate_mock_weights.py` to build the 31.7 GB random weight checkpoint on the local NVMe SSD (takes ~2 minutes).
3.  Launch `verify_dsv4.py` eagerly to execute the 4-layer model.
4.  Print the norm logs (`NORM_LOG: ...`) for hidden states layer-by-layer.

---

## Key Commands Cheat Sheet

### GKE Cluster Commands
Run these from the host VM `patemotter-cdk`:

*   **Check Pod Status:**
    ```bash
    kubectl get pods -n cloud-devkit-jobs
    ```
*   **Fetch Live Logs (once Running):**
    ```bash
    kubectl logs -n cloud-devkit-jobs dsv4-inference-v7x-8-vllm-server-0-0-fs252 -c vllm-server -f
    ```
*   **Check Pod Scheduling Details/Errors:**
    ```bash
    kubectl describe pod dsv4-inference-v7x-8-vllm-server-0-0-fs252 -n cloud-devkit-jobs
    ```
*   **Delete Stale JobSet:**
    ```bash
    kubectl delete jobset dsv4-inference-v7x-8 -n cloud-devkit-jobs
    ```
*   **Re-apply/Start JobSet:**
    ```bash
    kubectl apply -n cloud-devkit-jobs -f /home/patemotter_google_com/cloud-devkit/recipes/experimental/torchtpu-vllm/dsv4-inference-v7x-8-patemotter.yml
    ```

### Standalone TPU VM `alyssa` Commands
> [!IMPORTANT]
> Alyssa is currently running a background coding agent (`agy` PID 1037757). Do not execute tests or run computations on `alyssa` until she is finished to prevent TPU resource allocation aborts.

*   **Check Active Processes on VM:**
    ```bash
    ssh alyssa "ps aux | grep -i python"
    ```
*   **Check Stale TPU Lockfile (if processes are dead but TPU is locked):**
    ```bash
    ssh alyssa "ls -la /tmp/libtpu_lockfile"
    ```
*   **Run JAX Unit Tests (once VM is idle):**
    ```bash
    ssh alyssa "cd ~/workspace/dsv4/torchtpu-vllm && source .venv/bin/activate && PYTHONPATH=src pytest tests/kernels/test_dsv4_mla.py"
    ```

---

## Next Steps for Continuation
1.  **Monitor Pod Scheduling:** Keep running `kubectl get pods` until `fs252` goes from `Pending` to `Running`.
2.  **Inspect Norm Logs:** Once the pod runs and prints `NORM_LOG` statements, examine the printed norm values:
    *   Identify the layer number where norms first diverge (e.g. grow exponentially) or collapse (drop to zero or near zero).
    *   If norms blow up/collapse inside attention layers, focus on layout/indexing in `_deepseek_v4_attention_jax` and its sharding specifications inside `deepseek_v4.py`.
    *   If they blow up/collapse inside expert layers, focus on routing and indexing inside `fused_moe_gmm.py`.
3.  **Restore Cache settings:** After completing the debugging phase, remember to edit `VLLM_FP4_CACHE_DIR` back to `"/mnt/local-cache/dsv4-flash-fp4"` inside `dsv4-inference-v7x-8-patemotter-patemotter.yml` to re-enable permanent GCS weights archiving.
