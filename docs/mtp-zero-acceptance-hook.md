# 临时 MTP 低接受率诊断

## TL;DR

这是 `to be removed` 提交，排查完成后移除。
配置 speculative decoding 的单体服务和 KV consumer 默认启用；纯 KV producer 不启用。
异步调度只保留 scheduler 的 CPU 接受率统计，不保留诊断 tensor 引用，不执行 tensor
检查或 dump；同步调度保留下述低接受率 tensor 诊断。
启动服务前设置 `TPU_MTP_ZERO_ACCEPTANCE_LOG=0` 可关闭 scheduler hook 和 worker
诊断采集。关闭时不保留诊断用的 hidden/draft 引用，也不执行诊断计数、检查或 dump。

## 开关与验证

`TPU_MTP_ZERO_ACCEPTANCE_LOG` 未设置或为空时默认开启；`1`、`true`、`yes`、`on`
表示开启，`0`、`false`、`no`、`off` 表示关闭，大小写不限。其他值报 `ValueError`。
变量通过 vLLM 的 Ray 环境传递机制送到 worker，不进入 TPU 编译缓存 key。
在服务启动前设置，并重启服务使修改生效：

```bash
export TPU_MTP_ZERO_ACCEPTANCE_LOG=0
# 然后运行原有服务启动命令；恢复默认行为可 unset TPU_MTP_ZERO_ACCEPTANCE_LOG。
```

在安装仓库固定依赖的 CPU 开发环境中运行开关和低接受率回归：

```bash
TPU_MTP_ZERO_ACCEPTANCE_LOG=1 pytest -q \
  tests/test_mtp_zero_acceptance.py tests/test_mtp_diagnostic_output.py \
  tests/test_mtp_low_acceptance_diagnostics.py tests/test_async_mtp_diagnostic_isolation.py
```

验证关闭时保留 scheduler 原方法和 runner output 对象，未设置或开启时继续按下述
阈值记录 scheduler 日志；只有同步调度执行 tensor 检查。`=1` 不会重新开启异步
tensor 采集。Ray 传递与编译缓存契约分别由 platform 和 compiler 测试覆盖。

异步回归覆盖未设置、`0` 和 `1`，确认输出对象与采样值保持原样；scheduler hook
仍遵守开关。2026-10-09 的 r32 单机 PCP4/TP2DP2 隔离补丁实验完成 74 个 P→D
请求（54 个 MTP logits NaN，六轮八并发）：38538 个故障草稿接受数为 0，
14 个关键进程无重启、无 EngineCore fatal 或新 core；关闭注入后两种恢复输出
均 6/6 与基线一致。NaN 仍可能改变输出，此实验不代表任意 tensor NaN 都安全。

以下 tensor 检查时序和 dump 范围均指同步调度下的诊断路径。

## 唯一触发条件

整批有效 draft 数大于零，并且 `accepted * 10 < effective_drafts`。
恰好 10%、高于 10%、没有有效 draft 都不触发。按整批 token 数加权，
不是每请求接受率的平均值，也不会因单个请求接受率低而触发。
有效 draft 数扣除 grammar-invalid tokens；忽略已丢弃的采样行。

runner 复用已有采样结果的 CPU materialization 后才判断阈值。
正常批次不执行诊断 tensor 的 D2H、额外 event/wait、clone、NaN 扫描、
文件写入或逐批诊断日志。即使 hidden tensor 含 NaN，也必须满足低接受率条件。
正常路径仍有少量 CPU 计数/元数据维护，并把 hidden/draft 引用保留到输出消费；
这会延长 activation 的 HBM 生命周期，不能据此声称完全零开销。

## 日志

- `[MTP_TENSOR_DIAGNOSTIC] scope=worker_output_batch`：只有 worker 整批接受率
  严格低于 10% 才检查 tensor 并输出完整摘要，包含本批有效/接受 draft 数。
- `[MTP_LOW_ACCEPTANCE] scope=batch`：scheduler 按实际消费的请求计算，严格低于
  10% 才记录 `dp_index`、`batch_index`、请求 IDs、draft IDs 和诊断摘要。
  若请求在 worker 完成后被取消、丢弃或发生加载失败，scheduler 的请求子集可能
  与 worker 不同。此时 worker 不会为正常批次补做诊断；日志通过
  `diagnostic_status=worker_batch_not_below_threshold_or_unavailable` 说明缺失。
- 不再有正常批次的 `[MTP_TENSOR_CHECK]`，也没有独立于接受率的 NaN 触发。

## 现场与时序

logits 输入按 tensor 引用保留，低接受率时才复制真实 hidden-state 行并检查 NaN，
忽略 logits padding。fallback logits 路径的 index-select 也延迟到触发后。
实际验证的 draft tensor 同样延迟复制，并保持 chunk、request 和 padding 对齐。

GDN/cache 检查发生在采样之后，标记 `cache_observation=after_sampling`。
记录 `state_column`、`previous_state_column`、slot、read offset、实际 read slot
和 checkpoint 列表。`read_offset_phase=after_sampling` 表示下一次恢复的位置；
它不是刚完成的 forward 当时读入的 offset。为避免每批预先复制，不保存历史输入
状态，`input_state_snapshot_available=false`。

cache 检查与下次 `execute_model` 的第一次状态修改互斥。诊断 ticket 已失效时，
旧 cache 和共享 offset 都不再读取，明确报告
`cache_reused_before_acceptance_was_known`、`has_nan=null`、`status=unavailable`。
仍可检查保留的同批 hidden tensor；不会把后续 batch 的 cache 误标为本批现场。

未覆写的 GDN 检查实际写出的 checkpoint；attention 检查相关请求的已分配页，
包括未写尾部/lookahead。页面 NaN 不等于本次 attention 真正读到了 NaN。
pool manifest 按真实 FP32 SSM、BF16 conv、attention dtype 解释有效区域，
忽略字节 padding，attention 布局以解析后的 cache config 为准。

## Dump 文件与范围

低接受率检查发现 NaN 时，将完整异常 hidden tensor 或 pool/block 有效区域写到
worker 本机 `/tmp/vllm-mtp-<pid>-<capture_id>/*.pt`（使用系统临时目录）。
文件含 CPU tensor 和元数据，可用 `torch.load(path, weights_only=True)` 读取。
日志注明路径、NaN 数量、dtype、shape、layer/pool/block。
检查正常则明确 `has_nan=false`；未能检查则 `has_nan=null`，不会报成正常。
磁盘写入失败记录 `dump_error`。

每份报告只涵盖本 rank 的层，保留所有 worker 日志。现有配置门禁仍拒绝
PP+speculative 和 block-major+speculative；没有新增支持这些 serving 组合。
低接受率时的扫描与 dump 仍有同步和 I/O 开销，不修改采样值或 KV 内容。

CPU 回归覆盖阈值、有效 draft、丢弃行、正常批次无额外复制、NaN/正常/未知、
重复取输出、布局、ticket 失效和 mixed dtype；真实 TPU 辅助测试覆盖延迟 tensor
检查，异步 serving 的 NaN 验证覆盖关闭采集后的存活与恢复。性能证据不替代
完整模型服务的吞吐/延迟压测。
