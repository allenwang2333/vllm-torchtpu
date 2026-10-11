# DeepSeek V4.1 DP8 chunk host/device overlap

状态：**性能验证完成；长上下文数值基线未通过。** 代码尚未提交、推送或开 PR。不应将本轮结果描述为已通过数值验证的稳定基线。

## 根因与最小修复

现有 async scheduler 已能在前块设备执行期间进入下一块 ModelForward。阻断 overlap 的是 Engram 的 DP slot 查询：平台把 `num_tokens_across_dp_cpu` 上传为 TPU tensor，上游 `engram_gathered_num_tokens()` 随后 `.max().item()` 又从设备取回标量，每块等待约 115–117 ms，之后才开始约 55 ms 的 host Engram 准备。

修复在平台上传前保存不可变 CPU tuple 到 forward context；Engram 根据其连续 DP 子组直接计算 Python max。编译路径继续使用设备 metadata，collective 次序不变。未新增预取线程或双缓冲。`pad_id` 已核实是 Python 整数，与此同步无关。

生产改动：
- `src/vllm_torchtpu/platforms/tpu_platform.py`
- `src/vllm_torchtpu/models/vllm/deepseek_v41/v2/host_engram.py`
- 对应 host Engram 与 platform 回归测试。
- 本轮增量补丁：`host-dp-metadata.patch`。

## 配置与性能

V4.1 Flash v2，TP1/DP8/EP8，全功能（FP8 KV、async、CED、SP/local-Q、Engram CPU offload、DSpark K5、prefix cache）。LongBench 独立前缀，65536 hit + 16384 extend，O1，C8。chunk=2048/rank，全局 token budget=16384。每波新增 token=131072，prefix hits=524288，零 preemption。

| 指标 | 原串行路径 | overlap |
|---|---:|---:|
| 非 profile extend 服务端均值 | 1.871810 s | 1.467144 s |
| 客户端 mean TTFT | 2.053531 s | 1.640069 s |
| 客户端波次新增 token 吞吐 | 52537.7/s | 63106.8/s |
| 8 cores XLA Ops 空隙 | 592–599 ms | 71–77 ms |
| rank0 采样 HBM peak | 71.979 GiB | 72.051 GiB |

extend 减少 21.6%，新增 token 吞吐增加 20.1%。服务端 prefill 指标包含调度/回放，不是纯设备时间；吞吐使用客户端整波 wall time，未以 token/平均请求延迟替代。

同服务追加对照再次复现：serial-1 的不带 logprobs measured extend 为 1.895173 s，overlap 为 1.464786 s，fresh staging 为 1.471017 s，serial-2 回到 1.880970 s。完整数字见 `numeric/summary.json`。

## Trace 机制证据

- XLA Modules 从 258 变为 249，唯一消失的模块为 9 次 `tt_jit_max(2020072880951920416)`。
- rank0 长 `aten::item` 等待全部消失。
- Engram CPU span（hash all-gather 至 CED all-reduce）约 444 ms 与前块设备计算重叠；旧 trace 为 0。详见 `overlap-measurement.json`。
- 设备有效计算仍约 1.27–1.30 s；收益主要来自填补 host 空隙。
- 原始计数与时序：`correlation-events.json`、`timeline-summary.json`、`device-metrics.json`。

## 数值与正确性限制

34 项回归通过；O16 不均衡 C8 与 idle C1 smoke 完成。但最初 48 个响应有 15 个不同于历史实验，历史自身重复输入也不稳定。因此继续做了同一服务的 serial-1 / overlap / fresh / serial-2 隔离，每组相同 8 个文档、prime/extend 各 3 波。round0/1 不请求 logprobs，round2 请求 top5；后者没有明显消除 overlap。

不带 logprobs 的两次重复，在 16 个 prime/extend 输入中的输出变化数分别为 **6 / 2 / 6 / 6**。原串行路径和独立 staging 都存在变化，数据不支持把波动简单归因于当前 overlap 的 buffer 复用；但这也不是完整数值正确性证明。部分 logprobs 变化很大，不能解释成仅有微小浮点误差。

例如 extend rank2 在不同原/新模式中可在换行与 `eps` 间变化；`eps` 的 logprob 可为 0，而另一状态的换行 logprob 为 -2.4375。下一步数值调查应独立跟踪长上下文的调度/最后 token 与 logits 对齐、CED/DSpark 状态等，不应通过放宽文本比较来宣称通过。

所有 192 个响应、logprobs、输入 SHA 与 metrics 保存在 `numeric/numeric-results.json`；可用 `numeric/summarize.py` 重建汇总。`compare.py` 保留严格历史比较失败，未隐藏差异。诊断用独立 staging 仅存在于实验，未加入生产源码。

## 实验与复现

- 旧 profile：exp-2257y2oh7d / art-nj3o7jv3a7。
- overlap：exp-jew0a7weqi / art-d58h0sd80i，SUCCEEDED，资源已释放。manifest：`experiment.json`。
- 同服务数值隔离：exp-usduz1nwhn / art-e6moty8stp，manifest：`numeric/experiment.json`。原始 manifest 另通过 Falcon cp 覆盖了 `numeric/client.py`，并使用 `numeric/setup-mode.sh` 让回归测试阶段保持 overlap。准确客户端源码与完整远端补丁分别为 `numeric/client.py`、`numeric/tested.patch`。
- 数值隔离在最后一波后有界保留服务，完成信号 `/tmp/numeric.done` 已发送；实验 SUCCEEDED，资源已释放。最终状态见 `numeric/status.json`。
- 四个离线 analyzer 均 SUCCEEDED：results an-k6xxa734nh、CPU an-rbiarieu0k、timeline an-pilbybc8uo、device an-3x1hgqdnj8。
- 本地 Ruff、语法检查和 diff whitespace 检查通过。未新建定时任务。

详细执行过程与传输 ID 见 `investigation-log.md`。
