# MTP batch 零接受率诊断 hook

## TL;DR

启动前设置 `TPU_MTP_ZERO_ACCEPTANCE_LOG=1`。当一个 scheduler 输出 batch 的有效
draft 总数大于 0、accepted 总数等于 0 时，输出一条 `[MTP_ZERO_ACCEPTANCE]` WARNING，
记录有效 draft 的所有贡献 request ID，以及本批每条 request **实际参与验证的 draft
token IDs**。默认关闭。

PD 分离时，Prefill（`kv_producer`）不启用；Decode（`kv_consumer`）启用。
`kv_both` 因具备 consumer 能力也启用；没有 KV transfer 配置的单体服务继续支持。
所有这些模式都要求配置 speculative decoding。

日志说明这些请求的有效 drafts 本批全部被拒绝，导致**这一批**接受率为 0；不能据此
确定某一个请求是故障根因，也不等同于监控时间窗口或跨 DP 聚合接受率为 0。
开启后，验证 batch 会额外复制 draft IDs 到 CPU 并等待复制完成，适合排查复现。

## 开启与读取

保持现有模型、MTP、TP/DP/PCP 和采样参数，在启动环境中添加：

```bash
export TPU_MTP_ZERO_ACCEPTANCE_LOG=1
```

变量需要同时传入 EngineCore 和模型 worker 进程，容器部署通过现有环境配置传入，
设置后重启服务。旧 Ray executor 的平台传播白名单已包含此开关；仍需保证各节点
未单独设置冲突值或主动排除此变量。PD 模式可只在 Decode 服务设置。将现有服务日志保存到
`/tmp/mtp-zero-acceptance.log` 后检索：

```bash
rg '\[MTP_ZERO_ACCEPTANCE\]' /tmp/mtp-zero-acceptance.log
```

示例（单条日志）：

```text
[MTP_ZERO_ACCEPTANCE] scope=batch reason=all_effective_drafts_rejected dp_index=2 batch_index=17 num_draft_tokens=4 num_accepted_tokens=0 acceptance_rate=0 verified_draft_token_ids={"req-b": [101, 102, 103], "req-a": [201, 202], "prefill": []} request_ids=["req-b", "req-a"]
```

- `verified_draft_token_ids` 为 JSON 对象，覆盖本 runner batch 的每条 request，不截断。
  没有参与 draft 验证的请求记为 `[]`；去掉 bucket padding，保留每个请求内的 slot 顺序。
- 这些 IDs 是与当前接受率对应的**本轮验证输入**。异步模式从 runner 经 token
  substitution 后的设备张量获取，避免 scheduler 的 `-1` 占位值；不是本轮新生成、
  留给下一轮验证的 proposals，也不包含 bonus token。
- `request_ids` 为 JSON 数组，仅包含有效 draft 的贡献请求。因此 token dump 可能还包含
  没有有效 drafts、被上游跳过或被丢弃输出的请求，不能将 dump 的所有 key 都视为贡献者。
- `num_draft_tokens` 遵循上游 grammar invalid 调整。dump 保留实际验证输入，并不按
  invalid 计数删减 token 列表：例如上述 `req-a` 两个验证 slots 中一个无效，分母为 1。
- request ID 使用上游 scheduler 实际持有的 ID。通过入口请求日志取回对应输入、采样
  参数和到达顺序。`batch_index` 是本 scheduler 消费输出的序号，不是跨进程 step ID。

每个零接受 batch 都记录，包括首次验证即为零、连续为零以及恢复后再次为零。
未要求上一批先大于零；非零 batch 不输出 token dump。排查后取消变量或设为 `0` 并重启。

## 判定与实现边界

scheduler 接入固定 vLLM `v0.29.0` 的 `make_spec_decoding_stats()`，利用上游已经按
request ID 与 runner 行映射得到的真实 accepted 数，不读取未来 batch 的 input batch。
有效 draft 数为 `num_draft_tokens - num_invalid_spec_tokens[request_id]`，accepted
只计 drafts。仅收集上游实际进入统计调用的请求，在 `update_from_output()` 完成后判定；
上游跳过或丢弃的输出不进入贡献列表。

有效 drafts 中只要有一个 accepted token，就不触发；没有有效 drafts 的 batch、
bootstrap 和普通 prefill 不触发。关闭常规统计 `--disable-log-stats` 后仍能工作。
hook 也观察其他使用同一统计入口的 speculative 方法。

runner 仅增加一个输出装配接入点；功能自己的输出子类持有快照，复用既有
`AsyncTPUCopyState` 复制和等待机制，不给公共输出类型或原复制状态添加诊断字段。
chunk 内按非 padding 的每请求 draft 长度拆分，request IDs 与长度在当前 batch
快照，随对应输出传回 EngineCore，支持同步与异步 scheduling。

开启时，所有有验证 drafts 的 batch 都需要额外复制、CPU 拆分和输出传输，归零时再
写日志。未测量其性能影响。关闭、未配置 speculative decoding 或处于纯 producer
角色时，不安装 scheduler 观察器，runner 返回原输出对象。采样和上游统计返回语义
保持不变；诊断状态在每批结束或异常退出时清空。开关不参与编译环境参数哈希。

## 验证与复现

源码基线：`alibaba/release-0.0.1` 的 `ffaf644d48c0d9ea90b1c02b150d47968c7c7d04`。
正常项目测试环境使用分支固定的 Python 3.12、vLLM `v0.29.0`、TorchTPU 及测试依赖：

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_mtp_zero_acceptance.py tests/test_mtp_diagnostic_output.py
python -m pytest -q tests/test_envs_builders.py tests/platforms/test_patch_registry.py
python -m pytest -q tests/platforms/test_tpu_platform.py -k mtp_diagnostic_env_reaches_ray_workers
python -m pytest -q tests/compilation/test_tpu_compiler.py
python tools/check_env_reads.py
```

契约测试涵盖 PD producer 跳过、consumer 启用、关闭开关、整批归零、混合接受和无有效
drafts、统计关闭、连续 batch 与异常后的状态隔离。token 测试使用不同 draft 长度、
多 chunk、padding、无 draft 请求和批次重排，验证 token 对应关系、原输出字段及 pickle
传输；它用 CPU tensor 与事件替身，不要求执行模型。已有测试没有覆盖 token 快照契约。Ray 环境传播测试保护 driver 到 worker 的开关传递。

本地隔离验证结果：

- Python `3.12.14`、CPU Torch `2.14.1`、NumPy `2.5.3`；从固定 vLLM tag 提取真实统计
  方法和输出类型，绕过完整框架初始化，输出消费者与 TPU event 使用替身。
- hook 与 token 的 `18` 个契约用例通过；连同环境变量、registry，共 `126 passed`。
  已有 fork 用例产生 1 条多线程 fork 的 DeprecationWarning。
- 对三请求 batch 枚举 draft 数 `0..3` 和所有合法 invalid/accepted 组合，`8000` 个
  batch 的日志谓词、request ID 集合及上游统计通过。没有 scheduled drafts 的请求按
  上游契约不调用统计方法。
- 单独执行固定上游 Ray 环境选择函数及新传播测试，确认补齐白名单前漏传、补齐后通过。
- 编译环境参数收集函数在开关 `0/1` 下结果一致；Ruff `0.14.0`、环境变量读取检查、
  `git diff --check` 通过。Pyrefly `1.3.0` 对比源码基线，没有新增错误，保留 12 个已有错误。

本机没有完整 vLLM/TorchTPU 环境和 TPU，以上证据不等同于完整 scheduler、PD 服务、
真实设备异步复制或高并发验证。线上复现时保持异常窗口的请求输入、采样参数、版本、
PD 角色与 TP/DP 配置，检查日志中 draft IDs 能与当前归零 batch 对齐；完整环境还需执行
编译缓存测试与类型检查。CPU 测试不能证明生产性能与设备事件行为。
