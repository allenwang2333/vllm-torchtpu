# Kimi-K3 QKV Projection Optimization & TPU Profiling Report

## 1. Overview & Motivation

In standard transformer models like Gemma-4 (see PR [#3376](https://github.com/vllm-project/tpu-inference/pull/3376)), unmerging packed QKV linear projections avoids tensor slicing operations (`q, k, v = qkv.split(...)`), yielding up to **+2.8% token throughput** on TPUs.

In **Kimi-K3**, attention layers use **KDA (Kimi Delta Attention)**, where the custom Pallas TPU kernels (`chunk_kda` and `decode_kda`) natively consume a combined `mixed_qkv` buffer `[num_tokens, 3 * num_heads * head_dim]`. 

### The Root Cause of Merged Projection Inefficiency on TPU:
- **Baseline Merged Dimension:** `fused_qkvb_proj` packed Q, K, V (`12,288 * 3 = 36,864`) together with Beta (`96`) into a single unevenly sized linear layer of dimension **`36,960`**.
- **TPU Hardware Misalignment:** On Google TPU v6e (Trillium), the Matrix Multiply Units (MXUs) execute systolic tiles of **$128 \times 128$**. Dimension `36,960` is **not divisible by 128** (`36,960 / 128 = 288.75`, remainder `96`). This misalignment broke MXU systolic tiling, forcing XLA to generate partial tile masks and strided memory copies (`fusion.105` taking **343.23 µs**).
- **Unmerged Solution:** Unmerging `fused_qkvb_proj` into separate `q_proj`, `k_proj`, `v_proj`, and `b_proj` modules allows QKV (`36,864 % 128 == 0`) to execute as perfectly aligned $128 \times 128$ systolic MXU tiles in **49.21 µs**, with Beta (`96`) executing as a lightweight vector projection in **0.92 µs** (total compute **50.13 µs** vs baseline **343.23 µs**).

---

## 2. Profiling Setup & Configuration

Due to the full Kimi-K3 model's size (2.4T parameters across 61 layers) and its upstream `mxfp4` quantization (designed for TPU v7x FP4 instructions), we constructed a lightweight **1-layer toy model** in bfloat16 for deterministic benchmarking on **TPU v6e-8**.

### Environment & Hardware
- **Hardware:** Cloud TPU v6e (Trillium) — 8 chips (`v6e-8`), 32 GiB HBM per chip
- **Host OS & Python:** Linux x86_64, Python 3.12 (`py312`), PyTorch/XLA 2.7+ / TorchTPU
- **Model Checkpoint:** 1-layer Kimi-K3 toy model (`/home/yujiedeng_google_com/ws/data/toy_kimi_k3`)
  - `num_hidden_layers`: 1
  - `hidden_size`: 4096
  - `num_attention_heads`: 96 (`head_dim`: 128)
  - `num_experts`: 8 (routed top-k = 2)
  - `num_shared_experts`: 1
  - `dtype`: bfloat16 (unquantized)

### Engine & Profiling Parameters
```bash
VLLM_PLUGINS="torchtpu,torchtpu_layers" python3 -c "
import time
from vllm import LLM, SamplingParams

llm = LLM(
    model='/home/yujiedeng_google_com/ws/data/toy_kimi_k3',
    load_format='dummy',
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=256,
    max_num_batched_tokens=256,
    max_num_seqs=16,
    gpu_memory_utilization=0.5,
    attention_backend='CUSTOM',
    language_model_only=True,
    limit_mm_per_prompt={'image': 0, 'video': 0},
    profiler_config={
        'profiler': 'torch',
        'torch_profiler_dir': '/tmp/kimi_k3_profile_unmerged',
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
dummy_prompts = [{'prompt_token_ids': [1] * 128}]

# Warmup pass (compiles XLA FX graph & initializes Pallas kernels)
llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

# Profile pass
llm.start_profile()
time.sleep(0.5)
llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
llm.stop_profile()
"
```

---

## 3. Code Optimization: QKVB & MLA Module Unmerging

Following the design from Gemma-4 PR [#3376](https://github.com/vllm-project/tpu-inference/pull/3376), we unmerged projections across **both attention layer architectures** in Kimi-K3:

### A. KDA (Kimi Delta Attention) Layers (57 layers)
In [`src/vllm_torchtpu/models/vllm/kimi_k3/attention.py`](file:///Users/yujiedeng/Desktop/ws/vllm-torchtpu/src/vllm_torchtpu/models/vllm/kimi_k3/attention.py), `fused_qkvb_proj` was unmerged into 4 separate linear layers:

```python
# Unmerged KDA Linear Layers:
self.q_proj = ColumnParallelLinear(
    config.hidden_size, self.projection_size, bias=False,
    quant_config=quant_config, prefix=f"{prefix}.q_proj",
)
self.k_proj = ColumnParallelLinear(
    config.hidden_size, self.projection_size, bias=False,
    quant_config=quant_config, prefix=f"{prefix}.k_proj",
)
self.v_proj = ColumnParallelLinear(
    config.hidden_size, self.projection_size, bias=False,
    quant_config=quant_config, prefix=f"{prefix}.v_proj",
)
self.b_proj = ColumnParallelLinear(
    config.hidden_size, self.total_heads, bias=False,
    quant_config=quant_config, prefix=f"{prefix}.b_proj",
)
```

And in `forward()`:
```python
query, _ = self.q_proj(hidden_states)
key, _ = self.k_proj(hidden_states)
value, _ = self.v_proj(hidden_states)
beta, _ = self.b_proj(hidden_states)
mixed_qkv = torch.cat((query, key, value), dim=-1)
```

### B. MLA (Multi-Head Latent Attention) Layers (4 layers)
In `MultiHeadLatentAttention` and `MultiHeadLatentAttentionWrapper`, `fused_qkv_a_proj` was unmerged into independent `q_a_proj`, `kv_a_proj_with_mqa`, and `g_proj` layers:

```python
# Unmerged MLA A-Projections:
self.q_a_proj = ColumnParallelLinear(
    config.hidden_size, self.q_lora_rank, bias=False,
    quant_config=quant_config, prefix=f"{prefix}.q_a_proj",
)
self.kv_a_proj_with_mqa = ReplicatedLinear(
    config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim,
    bias=False, quant_config=quant_config, prefix=f"{prefix}.kv_a_proj_with_mqa",
)
self.g_proj = (ColumnParallelLinear(
    config.hidden_size, config.num_attention_heads * self.v_head_dim,
    bias=False, quant_config=quant_config, prefix=f"{prefix}.g_proj",
) if getattr(config, "mla_use_output_gate", False) else None)
```

### C. Checkpoint Weight Mapping Removal
In [`src/vllm_torchtpu/models/vllm/kimi_k3/model.py`](file:///Users/yujiedeng/Desktop/ws/vllm-torchtpu/src/vllm_torchtpu/models/vllm/kimi_k3/model.py), we removed all stacked parameter mappings (`fused_mla_mapper`, `q_proj`, `k_proj`, `v_proj`, `b_proj`, and `g_proj`) so that all weights load 1:1 directly from the checkpoint without slicing or concatenation.

---

## 4. Zoomed-In Attention Layer Performance Results

Focusing strictly on the **Attention Layer (KDA Layer)** kernels in the steady-state decode execution window:

### Side-by-Side Attention Layer Kernel Comparison

| Attention Component / Operation | Baseline (`orginal` - Merged) | Optimized (`optimized` - Unmerged) | Delta (µs) | Speedup / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **QKVB Projections Matmul (`%fusion.105` vs unmerged)** | **343.23 µs** | **50.13 µs** | **-293.10 µs** | 🚀 **-85.4% (MXU aligned)** |
| **Recurrent State Update (`%conditional.1`)** | **1,399.91 µs** | **1,381.51 µs** | **-18.40 µs** | 🚀 **-1.3%** |
| **Recurrent Dispatch (`%conditional`)** | **166.62 µs** | **155.91 µs** | **-10.71 µs** | 🚀 **-6.4%** |
| **Buffer Copy & Slicing Overhead (`copy-done.6` / staging)** | **5.40 µs** | **0.88 µs** | **-4.52 µs** | 🚀 **-83.7%** |
| **Pallas KDA Recurrence (`kda_decode_fused_conv_recurrent`)** | **93.67 µs** | **92.93 µs** | **-0.74 µs** | 🚀 **Faster custom op** |
| **Gate Projections & Reshape Fusions** | **3.00 µs** | **2.80 µs** | **-0.20 µs** | -6.7% |
| **Input RMSNorm (`rms_norm_mul`)** | **10.98 µs** | **10.85 µs** | **-0.13 µs** | -1.2% |
| **Total Attention Layer Device Time** | **2,022.81 µs (~2.02 ms)** | **1,695.01 µs (~1.69 ms)** | **-327.80 µs** | 🚀 **16.2% Faster Attention** |

---

## 5. Trace Layout & TPU Hardware Analysis

Comparing the HLO buffer layout at the entrance of the Pallas custom kernel:

### Baseline (Merged QKVB: Misaligned)
- A single uneven buffer was produced:
  `%copy-start.5 = (bf16[128, 36960]...)` where `36,960 = 12,288 * 3 + 96`.
- Because `36,960 % 128 = 96`, TPU v6e matrix units could not generate full $128 \times 128$ systolic tiles.
- Slicing `beta` out of the tail of `36,960` required uneven strided slice operations and staging copies (`copy-done.6`), causing `fusion.105` to take **343.23 µs**.

### Optimized (Unmerged QKVB: Aligned)
- Two perfectly dimensioned, cleanly aligned buffers are produced:
  1. `mixed_qkv`: `%copy-start.5 = (bf16[128, 36864]...)` (`12,288 * 3` contiguous for Q, K, V with `36,864 % 128 == 0`).
  2. `beta`: `%copy-start.21 = (bf16[128, 96]...)` (`96` heads).
- **Systolic Array Efficiency:** QKV computes cleanly on MXU tiles in **49.21 µs**, and Beta executes as a vector projection in **0.92 µs** (total compute **50.13 µs**).
- Total Attention Layer execution time decreased from **2.02 ms down to 1.69 ms (-16.2%)**.

---

## 6. Key Takeaways & Scaling to Production

1. **Clean Weight Loading & Module Isolation:**
   - Removing `q_proj`, `k_proj`, `v_proj`, and `b_proj` from `fused_attention_params_mapping` enables 1:1 direct checkpoint weight loading without runtime tensor concatenation overhead.
2. **Attention Layer Speedup:**
   - On the isolated attention layer, unmerging reduces execution latency from **2.02 ms down to 1.69 ms (-16.2%)**.
3. **Scaling Across 61 Layers:**
   - In full-scale serving (where 61 attention layers run concurrently under high batch size), eliminating ~328 µs per layer compounds to **~15 ms to 20 ms total speedup per decode step**.
4. **Trace Artifacts:**
   - Baseline trace: `/Users/yujiedeng/Desktop/kimi/orginal`
   - Unmerged trace: `/Users/yujiedeng/Desktop/kimi/optimized`
   - Visualized with: `xprof --logdir <trace_dir>`
