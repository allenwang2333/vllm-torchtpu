# Capacity-Aware MoE Balanced Routing & Local Expert Affinity
# 容量约束均衡路由与本地专家亲和偏置技术文档

---

## 1. Background & Problem Statement / 背景与核心问题

### EN:
Under **Expert Parallelism (EP=8)** on single-node TPUs (e.g. TPU v7x with 8 TensorCores):
- The 64 MoE experts are partitioned evenly across the 8 cores (each core owns 8 local experts).
- In real serving workloads, standard Top-$K$ gating exhibits **severe expert load imbalance** (e.g., Core 0 receives 20 tokens while Core 1 receives only 2 tokens).
- In the Decode phase, faster cores finish their Grouped GEMM (`gmm_v2`) early and **stall at layer-end `AllReduce` barriers** waiting for the slowest straggler core. This creates large **Barrier Core idle bubbles (e.g. 11 µs ~ 40 µs per layer)** in XProf traces, significantly degrading Token-per-Second (TPS) and Time-per-Output-Token (TPOT).

### CN:
在单机 TPU（如包含 8 个 TensorCore 的 TPU v7x）的 **专家并行（EP=8）** 模式下：
- 全局 64 个专家被均分到 8 个 TPU 核上（每个核负责 8 个本地专家）。
- 在真实服务中，原生 Top-$K$ 路由会产生严重的**专家负载倾斜**（例如 Core 0 接收了 20 个 Token，而 Core 1 只分到 2 个 Token）。
- 在生成（Decode）阶段，算得快的核提前完成 GMM 矩阵乘法，**被迫在层间 AllReduce 屏障处长时间挂起空等**最慢的长尾核。这在 XProf 性能分析图上表现为明显的 **Barrier Core 空闲气泡（每层空等 11 µs ~ 40 µs）**，严重拖慢了系统吞吐量（TPS）与单字生成延迟（TPOT）。

---

## 2. End-to-End Algorithm / 端到端算法流程

```
[Input: Router Logits [N, 64] & Hidden States [N, H]]
                      │
                      ▼
Step 1: Softmax / Sigmoid Scoring & Local Affinity Bias
        Scores = scoring_fn(Logits) + ε · Local_Expert_Mask
                      │
                      ▼
Step 2: Static Capacity Ceiling Formulation
        Ideal_Cap = ceil((N * k) / ep_size)
        Max_Cap   = max(k, ceil(Ideal_Cap * Capacity_Factor) + 1)
                      │
                      ▼
Step 3: Vectorized Pure-Tensor Selection Loop (Fixed k iterations)
        ├── 1. argmax(remaining_scores) -> step_ids [N]
        ├── 2. Map expert to EP Rank -> step_ranks [N]
        ├── 3. Cumulative rank load via torch.cumsum -> token_rank_cum [N]
        ├── 4. Detect overflow: is_overflow = (token_rank_cum > Max_Cap)
        ├── 5. Matrix tensor masking (torch.matmul) of overflow rank experts
        └── 6. Re-pick best available expert from non-full ranks
                      │
                      ▼
Step 4: Score Gathering & Renormalization
        TopK_Weights = Scores.gather(Selected_IDs) / sum(Weights)
                      │
                      ▼
[Output: Balanced Expert IDs [N, k] & Weights [N, k] -> Sent to gmm_v2]
```

---

## 3. Detailed Step Breakdown / 详细步骤解析

### Step 1: Scoring & Local Expert Affinity / 前置打分与本地专家偏置
* **Formula**:
  $$S = \text{Softmax}(\text{Logits}) \in \mathbb{R}^{N \times 64}$$
  $$S_{\text{choice}}[i, e] = S[i, e] + \epsilon \cdot \mathbb{I}(e \in \text{Local\_Rank\_Experts})$$
* **EN**: When `MOE_LOCAL_EXPERT_AFFINITY_EPSILON` ($\epsilon = 0.01$) is enabled, a tiny bonus is added to the 8 experts owned by the local TPU core. When expert scores are closely tied, the model prioritizes local experts, cutting inter-chip ICI communication.
* **CN**: 开启 `MOE_LOCAL_EXPERT_AFFINITY_EPSILON`（$\epsilon = 0.01$）后，为本核管辖的 8 个本地专家增加微小偏置。在专家打分相近时优先命中本地专家，减少跨芯片网络传输。

---

### Step 2: Capacity Ceiling Formulation / 动态容量上限推导
* **Formula**:
  $$\text{Ideal\_Capacity} = \left\lceil \frac{N \times k}{\text{ep\_size}} \right\rceil$$
  $$\text{Max\_Capacity} = \max\left(k, \left\lceil \text{Ideal\_Capacity} \times \text{Capacity\_Factor} \right\rceil + 1\right)$$
* **EN**: For $N=16, k=8, \text{ep\_size}=8$, total slots = $128$. Ideal capacity is $16$ tokens/core. With `Capacity_Factor = 1.25`, $\text{Max\_Cap} = 21$ tokens. Any core attempting to take $>21$ tokens triggers automatic overflow rerouting.
* **CN**: 例如 $N=16, k=8, \text{ep\_size}=8$，总路由槽位为 $128$，理想均值为每核 $16$ 个 Token。设置 `Capacity_Factor = 1.25` 时，容量上限 $\text{Max\_Cap} = 21$。任何核接收超过 21 个 Token 就会触发熔断顺延。

---

### Step 3: Vectorized Pure-Tensor Selection Loop / 纯张量向量化选择流水线
* **EN**: To guarantee zero host-device synchronization (no `.item()`, no Python scalar branches), the selection runs over a fixed compile-time unrolled loop of $k$ steps:
  1. **Pick Highest Candidate**: `step_ids = torch.max(remaining_scores, dim=-1)` ($[N]$)
  2. **Map to Core**: `step_ranks = rank_lookup[step_ids]` ($[N]$)
  3. **Track Load via `torch.cumsum`**:
     $$\text{one\_hot} = \text{torch.nn.functional.one_hot}(\text{step\_ranks}, \text{num\_classes}=\text{ep\_size})$$
     $$\text{token\_rank\_cum} = \text{total\_counts} + \text{torch.cumsum}(\text{one\_hot}, \text{dim}=0)$$
  4. **Overflow Detection**: `is_overflow = token_rank_cum > Max_Cap` ($[N]$ boolean tensor on TPU)
  5. **Tensor Matrix Contraction for Masking**:
     $$\text{expert\_overflow\_mask} = (\text{one\_hot}(\text{step\_ranks}) \cdot \text{is\_overflow} \times \text{rank\_expert\_mask}^{T}) > 0 \in \mathbb{R}^{N \times 64}$$
     $$\text{remaining\_scores}[\text{expert\_overflow\_mask}] = -\infty$$
  6. **Re-select Alternative**: `new_ids = torch.max(remaining_scores, dim=-1)`; commit via `torch.where(is_overflow, new_ids, step_ids)`.
* **CN**: 为彻底避免 CPU-TPU 阻塞同步（零 `.item()`、零 Python 条件跳转），算法展开为固定 $k$ 次循环的纯张量流水线：
  1. 向量化提取当前轮次各 Token 最高分专家；
  2. 映射专家所属 Rank；
  3. 利用 `torch.cumsum` 在单个张量算子中统计 $N$ 个 Token 对各 Rank 的实时排队深度；
  4. 生成纯张量溢出布尔掩码；
  5. 利用张量矩阵乘法批量将超载 Rank 的所有专家打分置为 $-\infty$；
  6. 用 `torch.where` 自动挑选非超载 Rank 的次优专家。

---

### Step 4: Gather & Renormalize / 权重提取与归一化
* **EN**: Gathers unbiased routing probabilities from original score tensor $S$ using chosen `selected_ids`, renormalizes across top-$k$, and passes balanced token indices to `gmm_v2`.
* **CN**: 从未加偏置的原始概率矩阵 $S$ 中提取对应权重，归一化后将负载完全对称的索引传递给 `gmm_v2` 矩阵乘内核。

---

## 4. Concrete Numeric Example / 具体数值算例

Suppose **$N=4$ Tokens**, **Top-$k=2$**, **2 TPU Cores** (Core 0 owns Experts 0,1; Core 1 owns Experts 2,3), and $\text{Max\_Capacity} = 4$:

| Token | Scores [Exp 0, 1, 2, 3] | Iteration 1 Selection | Iteration 2 Selection | Action Taken | Final Chosen Experts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Token 1** | `[10.0, 9.0, 2.0, 1.0]` | Expert 0 (Core 0) | Expert 1 (Core 0) | Normal (Core 0 load: 2) | **[0, 1]** |
| **Token 2** | `[9.8, 8.5, 3.0, 2.0]` | Expert 0 (Core 0) | Expert 1 (Core 0) | Normal (Core 0 load: 4 - Full!) | **[0, 1]** |
| **Token 3** | `[9.5, 8.0, 4.0, 1.0]` | Expert 0 (Core 0) | 💥 Expert 1 (Core 0) | **Overflow!** Core 0 is full ($>4$). Mask Core 0 $\to$ Pick **Expert 2** (Core 1). | **[0, 2]** |
| **Token 4** | `[9.0, 7.5, 5.0, 3.0]` | 💥 Expert 0 (Core 0) | Expert 2 (Core 1) | **Overflow!** Core 0 is full. Mask Core 0 $\to$ Pick **Expert 2, 3** (Core 1). | **[2, 3]** |

* **Result**: Core 0 processes 4 tokens, Core 1 processes 4 tokens. **100% Symmetric Execution.**

---

## 5. Configuration Guide / 配置与环境变量指南

### In Kubernetes YAML (`values-sweep_torch.yml`):
```yaml
env:
  # 1. 启用容量约束均衡路由 (推荐 1.10 ~ 1.25，强力削峰可用 1.05)
  - name: MOE_ROUTING_CAPACITY_FACTOR
    value: "1.25"

  # 2. 启用本地专家亲和偏置
  - name: MOE_LOCAL_EXPERT_AFFINITY_EPSILON
    value: "0.01"

  # 3. 细化低并发分桶，避免 Padding 算力浪费
  - name: TPU_TOKEN_BUCKET_EXTRA
    value: "4,8,12,16,20,24,32,48"

  # 4. 全程使用 TensorCore 进行 MoE 重排
  - name: ONEHOT_MOE_PERMUTE_THRESHOLD
    value: "65536"

  # 5. 关闭 SC 集合通信卸载与硬件 Launch Barrier
  - name: LIBTPU_INIT_ARGS
    value: " --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_use_enhanced_launch_barrier=false"
```

### In Shell Script (`server.sh`):
```bash
export MOE_ROUTING_CAPACITY_FACTOR=1.25
export MOE_LOCAL_EXPERT_AFFINITY_EPSILON=0.01
export TPU_TOKEN_BUCKET_EXTRA="4,8,12,16,20,24,32,48"
export ONEHOT_MOE_PERMUTE_THRESHOLD=65536
export LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_use_enhanced_launch_barrier=false"
```
