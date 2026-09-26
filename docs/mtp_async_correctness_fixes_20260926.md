# Async MTP 三项正确性修复

## TL;DR

本次修复三个经真实 TPU UT 验证的问题：统一 KV 池请求暂离 batch 后 checkpoint 读偏移被错误清零、
bootstrap 输出计数被二次回退，以及混合 chunk 丢弃已经生成的 drafts。
修复分别让 checkpoint 跟随状态存储生命周期、区分本步验证数量与下一步 proposal 数量、
统一构造各 chunk 的下一步输入源。未改变 DP 必须一致执行的 offset reset 设备程序。

这些 UT 证明局部缺陷及修复效果，尚不能证明它们就是 2026-09-24 22:28–22:35
服务 `v-8efc4bbc` 的单个 Decode 接受率降至接近零的实际根因。
仍需原部署的高并发压测与请求级状态、接受率对照；未执行 397B 模型精度或吞吐测试。

## 1. 暂离 batch 的请求重入时误选 checkpoint

- **现象与证据**：请求 A 在统一 KV 池中接受到 checkpoint 2，下一步仅调度 B。
  上游从 `InputBatch` 移除 A，但 `runner.requests` 与 A 的 KV blocks 仍在。
  旧实现按 batch 成员裁剪 `_mamba_offset_seeded`，A 重入时错误地将 offset 从 2 清为 0。
- **修复**：初始化标记随 `runner.requests` 保留；结束、抢占和恢复重算时显式失效。
  compact 池与统一池生命周期不同，compact 槽位回收时同步移除标记，防止槽位复用残留。
  同一步结束并重新提交相同请求 ID 时，也清除旧标记及旧异步结果映射。
- **验证契约**：统一池 A → B → A 保留 offset 2；持续请求保持原值；新 owner、同 ID
  新请求、compact 回收后重入均从 0 开始。既有抢占/恢复测试同时验证其他请求的源索引不变。
  compact 用例只验证 offset 初始化，不验证完整 recurrent-state 的保存和恢复。

## 2. Bootstrap 与上游裁剪造成二次回退

- **现象与证据**：prefill 只采样一个 bonus，却把为下一步生成的 K 个 drafts
  记为本步输出占位。上游按 scheduler 的一个输出占位裁剪后，TPU 回写再次扣除 K。
  在 prompt 长度 10、K=3 时，旧实现的 token end 为 `14 → 11 → 8`，并覆盖 prompt。
- **修复**：bootstrap 的 `num_draft_per_req` 设为 0，仅登记本步的一个输出占位；
  下一步 `[bonus, draft_1, …, draft_K]` 源保持完整。明确该字段记录本步验证数量，
  与下一步 proposal 数量无关。
- **验证契约**：串联真实 `AsyncScheduler`、`GPUModelRunner._update_states` 与 TPU
  异步回写；K=0/1/3 均恰好提交一个 bonus，token end 为 11，prompt 与输出历史保持正确。

## 3. 混合 chunk 丢弃 drafts

- **现象与证据**：同一步包含 verify chunk 与非 verify chunk 时，后者虽然已有真实
  proposal，旧 assembler 仍把下一步源填成 `[bonus, -1, …]`。真实 TPU token substitution
  因而把无效值写入下一步输入；相同请求处于纯 bootstrap 步时则正常。
- **修复**：所有 chunk 共用 `assemble_spec_next_tokens` 组装 bonus 与真实 drafts；
  是否存在本步 verify metadata 只影响拒绝数和输出占位计数。
- **验证契约**：K=1/3 下，纯 bootstrap 与混合 chunk 的新请求均得到同样的真实下一步输入。
  已知部署若不会形成此类混合多 chunk，则此项对该事故的解释优先级较低。

## 复现与验证

基线为 `aios/tmp_dev` 的 `1a623487b49532e1d0e14d0f27fdcbcba86c2563`。
测试无需模型权重，使用真实上游状态更新及 TPU 张量操作；仅替代无关的模型、编码器和 PP 初始化。

2026-09-26 实测结果：

| 验证 | 结果 |
| --- | --- |
| 基线代码运行新增 17 个 UT | 7 个契约断言失败、10 通过 |
| 修复后运行相同 UT | 17 通过 |
| 修复后下列完整相关测试组 | 151 通过 |
| 本次变更的 pre-commit 检查（含 Ruff、PyMarkdown、Pyrefly） | 通过 |

7 个失败由统一池暂停重入 1 例、两种池的同 ID 重用 2 例、bootstrap K=1/3 两例、
混合 chunk K=1/3 两例组成。测试不是性能基准。

本机验证环境：Python 3.12.14、vLLM 0.29.0（源码提交 `98dff2a81d747d1dba01a47f939f48c3526d4206`）、
torch 2.13.0+cpu、torch-tpu 0.1.1.dev20260912075415、jax/jaxlib 0.10.2、libtpu 0.0.47。
需要可用 TPU，且环境已安装本仓库及 pytest、pytest-timeout。
固定 torch-tpu wheel 来自 Google 私有 Artifact Registry，新环境需具备相应读取权限。

在仓库根目录运行；本机 Conda 环境为 `/mnt/data/yanzecheng_yzc/miniconda3/envs/torch_tpu`：

```bash
source /mnt/data/yanzecheng_yzc/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/yanzecheng_yzc/miniconda3/envs/torch_tpu
# 本机系统 C++ 库较旧，须优先加载 Conda 的库，避免 sqlite/ICU 导入失败。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
python -m pytest -q --tb=short --timeout=90 --junitxml=/tmp/mtp-fixes.xml \
  tests/runner/test_mtp_async_contracts.py \
  tests/runner/test_tpu_runner.py \
  tests/runner/test_tpu_runner_preemption.py \
  tests/runner/test_mamba_state_seed_copies.py \
  tests/runner/test_tpu_runner_async_output.py \
  tests/runner/test_tpu_runner_kv_finalize.py
```

新增 UT 位于 `tests/runner/test_mtp_async_contracts.py`；断言均描述正确行为，无 `xfail`。
单独运行此文件可验证三个问题及 owner 生命周期边界。
判断以契约断言为准，依赖导入或 fixture 初始化失败不能算缺陷复现。
JUnit 输出在 `/tmp/mtp-fixes.xml`；本机本轮日志保存在
`/mnt/data/yanzecheng_yzc/mtp-ut-20260926/logs/fix-*.log`。
