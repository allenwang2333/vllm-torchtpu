# DeepSeek V4 TPU Inference — Agent Handoff Guide

## 1. Remote Host & SSH Configuration
- **Host Alias**: `alyssa`
- **SSH Key**: `/Users/patemotter/.ssh/google_compute_engine`
- **SSH Command**:
  ```bash
  ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa
  ```
- **Local Directory**: `/Users/patemotter/dsv4/vllm-torchtpu`
- **Remote Directory**: `/home/patemotter_google_com/workspace/vllm-torchtpu`

---

## 2. File Synchronization (rsync)
To sync modified local files to `alyssa`:
```bash
rsync -avz -e "ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine" \
  src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py \
  alyssa:/home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py
```

---

## 3. TPU Device Reset / Process Cleanup
If PJRT initialization fails with `Device or resource busy` (`open(/dev/vfio/*)`):
```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker ps -q | xargs -r sudo docker kill && sudo pkill -9 python3 || true"
```

---

## 4. Containerized Inference Execution Command
Run the containerized TP8 DeepSeek V4 generation job:
```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker run --init --privileged --net=host --ipc=host --rm \
  -v /home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu:/usr/local/lib/python3.12/site-packages/vllm_torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e HF_HOME=/home/patemotter_google_com/workspace/hf_cache \
  -e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so \
  -e LIBTPU_INIT_ARGS=\"--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1000000000 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1000000000\" \
  -e TORCH_DIST_TIMEOUT=120 \
  -e VLLM_TP_SIZE=8 -e VLLM_EP_SIZE=8 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PYTHONWARNINGS=ignore \
  -e TRANSFORMERS_VERBOSITY=error \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -e JAX_COMPILATION_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache \
  -e DSV4_PROMPT=\"Compare XLA and LLVM in two bullet points.\" \
  -v /dev/shm:/dev/shm \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  vllm-torchtpu-local:latest \
  bash -c 'python3 -u /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py'"
```

---

## 5. Key Architecture & Source Code Files
- **Custom Ops & Layer Wrappers**: [src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py)
- **Pallas MLA Ragged Paged Attention Kernel**: [src/vllm_torchtpu/kernels/deepseek_v4/mla.py](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py)
- **Pallas Sliding Window Attention Kernel**: [src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py)
- **Model Monkey-Patches**: [src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py)
- **Development & Bug Log**: [development_log.md](file:///Users/patemotter/dsv4/vllm-torchtpu/development_log.md)

---

## 6. Current Diagnostic Status & Next Steps
1. **Pipeline Execution**: Runs completely without crashes across all 60 layers.
2. **Current Issue**: Hidden states evaluate to `NaN`, causing logits to evaluate to `NaN` and the sampler to output repeating `<｜image｜>` tokens (`ID 129279`).
3. **Active Instrumentation**: Injected `jax.debug.print` inside `_attention_local` ([custom_ops/deepseek_v4.py:L404](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L404)) to trace `q_nan` and `out_nan` directly on TPU hardware during runtime.
