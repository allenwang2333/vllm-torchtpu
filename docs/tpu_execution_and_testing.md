# DeepSeek-V4 TPU Execution & Unit Testing Guide

This document lists the environment configurations, cache directories, end-to-end execution commands, and unit tests for running DeepSeek-V4 on TPU v7x-8 hardware.

---

## 1. Environment & Infrastructure Setup

*   **Default Host VM (`alyssa`)**: Primary host for full v7x-8 model verification (8 chips).
    *   **SSH Target**: `alyssa` (resolves to `t1v-n-b8112fd6-w-0`)
    *   **SSH Key Path**: `/Users/patemotter/.ssh/google_compute_engine`
    *   **Remote Workspace Path**: `/home/patemotter_google_com/workspace/vllm-torchtpu` (mounted to `/home/patemotter_google_com/workspace` in Docker)
    *   **Container Registry**: `us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest`

---

## 2. Weight Caches & Storage Locations

| Directory Path | Filesystem | Contents | Status |
| :--- | :--- | :--- | :--- |
| `/dev/shm/dsv4-flash-real-fp4` | Host RAM Disk (`tmpfs`, 321 GB free) | 344 `.pt` pre-quantized FP4 expert weight files | **Active & Complete** (43 layers × 8 ranks = 344 files) |
| `/home/patemotter_google_com/workspace/fp4_cache` | NVMe SSD (`/dev/nvme1n1`, 201 GB free) | Backup directory for serialized `.pt` cache files | Persistent secondary cache storage |

---

## 3. End-to-End Execution Commands

Run either of these commands from your local workstation shell to launch the model on the remote TPU VM:

### Option 1: Standard Inference Run (Recommended)
This uses mounted host shared memory and proper process initialization hooks:

```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker run --init --privileged --net=host --ipc=host --rm \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e VLLM_TP_SIZE=8 -e VLLM_EP_SIZE=8 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -v /dev/shm:/dev/shm -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  python3 /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py"
```

### Option 2: Full Eager Mode Profiling/Execution
Use this to enforce specific quantization parameters, eager execution limits, memory bounds, and code directory overrides:

```bash
ssh alyssa "sudo docker run --privileged --net=host --shm-size=16g \
  -v /home/patemotter_google_com/torchtpu-vllm:/root/cloud-devkit/torchtpu-vllm \
  -v /home/patemotter_google_com/workspace:/mnt/workspace \
  -w /root/cloud-devkit/torchtpu-vllm \
  -e PYTHONPATH=/root/cloud-devkit/torchtpu-vllm/src \
  -e DSV4_MODEL_PATH=/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
  -e VLLM_FP4_CACHE_DIR=/mnt/workspace/tmp/dsv4-flash-real-fp4 \
  -e TPU_LIBRARY_PATH=/mnt/workspace/libtpu.so \
  -e ENABLE_QUANTIZED_MATMUL_KERNEL=1 \
  -e REQUANTIZE_BLOCK_SIZE=128 \
  -e MOE_REQUANTIZE_BLOCK_SIZE=512 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
  -e NEW_MODEL_DESIGN=1 \
  -e MODEL_IMPL_TYPE=vllm \
  -e VLLM_ENFORCE_EAGER=1 \
  -e VLLM_PLUGINS=torchtpu \
  -e GPU_MEM_UTIL=0.28 \
  -e XLA_CLIENT_MEM_FRACTION=0.32 \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  python3 -u examples/run_dsv4_inference.py"
```

---

## 4. DeepSeek-V4 Unit Tests

The codebase contains the following unit tests under `tests/`:

| Test Path | Component Tested |
| :--- | :--- |
| `tests/kernels/test_dsv4_moe_routing.py` | MoE dynamic gated router correctness |
| `tests/kernels/test_dsv4_compressor.py` | State compressor (C4/C128) kernel correctness |
| `tests/kernels/test_dsv4_mla_swa.py` | Combined MLA + Sliding Window Attention Pallas kernel |
| `tests/kernels/test_dsv4_mla.py` | Standard Compressed MLA attention kernel |
| `tests/kernels/test_dsv4_streamindex_topk.py` | MoE top-K routing selection logic |
| `tests/kernels/test_dsv4_mla_nan.py` | Guard logic against online-softmax NaNs |
| `tests/layers/vllm/test_dsv4_rope.py` | Rotary embeddings (last 64 trailing channels) correctness |
| `tests/layers/vllm/test_dsv4_linear_weight_loading.py` | Standard weights mapping and mapper registry |
| `tests/layers/vllm/test_dsv4_moe_weight_loading.py` | MoE gated expert weight loader mapping |
| `tests/layers/vllm/test_dsv4_correctness_compiled.py` | End-to-end multi-layer correctness compiled execution |
| `tests/layers/vllm/test_dsv4_o_proj.py` | Projection outputs (`o_proj`) correctness |
| `tests/runner/test_dsv4_weight_mapping.py` | Global runner model loader patching registry |

---

## 5. How to Run Unit Tests

To run an individual unit test on the remote TPU VM:

```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker run --rm --net=host --ipc=host --privileged \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  -w /home/patemotter_google_com/workspace/vllm-torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PJRT_DEVICE=TPU \
  -e JAX_PLATFORMS=tpu,cpu \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  pytest -sv --use-tpu tests/kernels/test_dsv4_mla_swa.py"
```
