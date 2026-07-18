# DeepSeek-V4 TPU Development & Execution Guidelines

This document provides localized workspace rules, environment configurations, and execution setups for debugging and running the DeepSeek-V4 model on TPUs. It is automatically loaded by Antigravity at the start of new conversations in this workspace.

---

## 1. Environment & VM Infrastructure

- **Local Debug Host (TPU VM)**: `patemotter-tpu-v6e-8-3` (v6e-8 architecture).
- **CDK Job Submission Node**: `patemotter-cdk` VM (IP: `34.68.6.121`).
- **NFS Share Volume**: Mounted on `/workspace` on `patemotter-cdk` VM (`172.30.171.18:/workspace`).
- **Local NVMe SSD Mount**: Mounted on `/mnt/workspace` on the TPU VM (`/dev/nvme0n2`).

---

## 2. Parity Compiler (libtpu.so) Integration

To compile low-precision FP4 (4-bit float) MoE kernels without backend MLIR type unpacking errors, the execution environment must use the custom patched compiler build.
- **Source**: `/mnt/workspace/libtpu.so` on `patemotter-cdk` (NFS).
- **Target**: Copied to `/mnt/workspace/libtpu.so` on `patemotter-tpu-v6e-8-3` VM.
- **Activation**: Activated inside the running docker container via the environment variable `TPU_LIBRARY_PATH=/mnt/workspace/libtpu.so`.

---

## 3. Sync & Development Workflow

To synchronize local host code changes to the TPU VM before running:
```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '*.safetensors' --exclude '*.safetensors_*' --exclude '*gstmp' -e "ssh" /usr/local/google/home/patemotter/torchtpu-vllm/ patemotter-tpu-v6e-8-3:/home/patemotter_google_com/torchtpu-vllm/
```

---

## 4. Run Commands

To execute the 4-layer verification/compilation test on the TPU VM:
```bash
ssh patemotter-tpu-v6e-8-3 \
  "sudo docker run --privileged --net=host --shm-size=16g \
  -v /home/patemotter_google_com/torchtpu-vllm:/root/cloud-devkit/torchtpu-vllm \
  -v /mnt/workspace:/mnt/workspace \
  -w /root/cloud-devkit/torchtpu-vllm \
  -e PYTHONPATH=/root/cloud-devkit/torchtpu-vllm/src \
  -e DSV4_MODEL_PATH=/mnt/workspace/dsv4-flash-4layer-fp8 \
  -e TPU_LIBRARY_PATH=/mnt/workspace/libtpu.so \
  -e ENABLE_QUANTIZED_MATMUL_KERNEL=1 \
  -e REQUANTIZE_BLOCK_SIZE=128 \
  -e MOE_REQUANTIZE_BLOCK_SIZE=512 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
  -e NEW_MODEL_DESIGN=1 \
  -e MODEL_IMPL_TYPE=vllm \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  python3 -u examples/verify_dsv4.py"
```

---

## 5. Architectural Gotchas & Configurations

- **Dynamic Head Sharding axis**: `ShardingAxisName.ATTN_HEAD` and `KV_CACHE_HEAD` are resolved dynamically via a python metaclass mapping to `'expert'` (for EP), `'model'` (for TP), or `'attn_dp'` (for attention DP) based on the active JAX mesh size.
- **Attention Sinks Parameter**: `attention_sinks` is initialized as a global shape `[64]`. However, the Custom Op has partition spec `P(shard_axis)`. To avoid JAX tracer scaling shape mismatches (`(64, 1) vs (8, 128)`), it must be sliced in PyTorch prior to Custom Op invocation using the worker's rank:
  ```python
  local_sink = self.attn_sink[rank * n_local_heads : (rank + 1) * n_local_heads].clone()
  ```
- **FP4 Software Dequantization**: TPU v6e/v7 do not support native FP4 execution. low-precision weights are unpacked and dequantized in software inside VMEM as part of the custom Pallas GMM kernel.
