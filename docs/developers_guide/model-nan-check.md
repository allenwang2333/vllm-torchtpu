# 最后一层 hidden 和 logits 的 NaN/Inf 检测

## TL;DR

设置 `TPU_NAN_LOG_LEVEL=1`（`SUMMARY`）或更高等级，检查每个真实主模型 forward
chunk 的最后一层 hidden
和采样前 logits。默认关闭。命中时记录 ERROR 并继续执行，检测器返回布尔结果，
本检测不修改 tensor、不替换 token，也不触发 dump。

配置唯一 TPU connector stats leaf 且启用 stats logger 时，每轮结果还通过现有
connector metrics 通道上报。只保留异常轮数一个 Counter，发现 NaN/Inf 的轮次增加 1，
正常轮累计值保持不变；绘制累计曲线即可查看历史异常及上涨时间。

检查有效行中的 NaN、`+Inf` 和 `-Inf`；padding 不算异常。设备上保存各 chunk 的布尔结果，
轮末合并后异步回读，由 runner 主线程轮询 FIFO 队列。只消费已完成的队首，遇到
未完成项立即返回，不等待设备。空闲尾项留到下一次 runner 调用；没有后台线程或
空闲定时轮询。设备归约、传输和主机处理仍有成本，吞吐与延迟开销需要在实际部署中测量。

## 启用和输出

在 worker 启动前设置环境变量；runner 构造时读取，修改后需要重启 worker。

```bash
export TPU_NAN_LOG_LEVEL=1
# 使用原有 vllm serve 或服务启动命令。
```

`TPU_NAN_LOG_LEVEL` 是 vllm-torchtpu 与 TPU Sync 共用的累计诊断等级，默认 `0`。
只接受单个数字 `0`–`4`，其他值（包括空字符串、空格和等级名称）报配置错误。
在所有 worker 启动前设置，修改后重启进程；两个包均需升级到支持该协议的版本。
SUMMARY 的主线程轮询不依赖输出消费，支持同步或异步 TP/PCP 多 rank；
没有 TPU connector stats leaf 时仅输出日志。更高等级沿用各诊断原有能力边界。

| 枚举 | 值 | 累计启用的诊断 |
| --- | --- | --- |
| `OFF` | 0 | 全部关闭 |
| `SUMMARY` | 1 | 最后 hidden / logits NaN/Inf 日志与异常轮数指标 |
| `DETAIL` | 2 | 加上逐层 hidden 检查与现有现场快照 |
| `DUMP` | 3 | 加上 TPU Sync 接收 FA/GDN KV 检查与限频定位日志 |
| `ALL` | 4 | 加上 Decode batch 逐层 KV cache 检查 |

该等级控制诊断范围，沿用各诊断原有采样、限频和能力边界。旧变量
`TPU_CHECK_NAN`、`TPU_HIDDEN_NAN_LOG`、`TPU_RAIDEN_CHECK_RECV_FA_NAN`、
`TPU_KV_CACHE_NAN_LOG` 已停用，不再读取。KV 保存保护的 `check_nan_on_save`
保持独立，不受该等级控制。

异常日志示例：

```text
ERROR ... NaN/Inf detected in last_hidden: host=worker-a, engine=2, rank=3, shape=(128, 4096)
ERROR ... NaN/Inf detected in logits: host=worker-a, engine=2, rank=3, shape=(8, 151936)
```

`host` 为 worker hostname，`engine` 为 data-parallel rank；`rank` 使用 runner 的
profiler rank，`shape` 是包含 padding 的 tensor shape。时间使用原有 logger 时间戳。
每个含有效行 NaN/Inf 的 chunk 分别记录一次对应 tensor 的 ERROR。

## 检查边界

| tensor | 检查位置与范围 |
| --- | --- |
| `last_hidden` | 最后一个 PP stage 的模型输出，在 layout gather/选行前检查所有有效 rank-local token，而不仅是最后采样行 |
| `logits` | `compute_selected_logits` 或 `compute_logits_from_hidden_states` 返回后，采样及结构化约束处理前，检查有效采样/verify 行 |

hidden 的有效行来自本 chunk 的 `SequenceLayoutPlan`：普通布局使用实际 token
范围；PCP 使用 packed 到 request-major 的映射及 `token_slice`，排除 `-1` padding。
logits 的有效行数在普通采样时为 `num_reqs`，在 speculative verify 时为
`sum(draft_lengths_cpu + 1)`，覆盖每个草稿的 verify 行和 bonus 行。

检查只接入 `TPUModelRunner.execute_model` 的真实 chunk。DP dummy、预编译、
warmup、非末尾 PP stage 不执行检查。pooling 模型只检查最后一层 hidden，
不把占位的空 logits 当成模型输出。独立 draft proposer 的 forward 不在此次范围内；
主模型对草稿的 verify 在范围内。

TPU 上通过项目已有的 `pallas.jax_op` 调用 JAX 归约：

```python
(~jnp.isfinite(tensor)).any(where=valid_rows[:, None])
```

有效行 mask 送到设备，`isfinite`、padding 过滤及最终 `any()` 均在设备上完成。
每次归约输出标量 `bool`；本轮各 chunk 的标量合并为小型数组，使用
`to("cpu", non_blocking=True)` 回读并记录完成事件。检测器拥有的 FIFO 队列保留本轮
设备标量、CPU 数组及 tensor 名称和 shape，直到该项完成并被消费；后续轮次不会
覆盖尚未消费的结果。
不复制完整 hidden/logits。沿用桥接层已有的 JIT，不额外包一层 `torch.compile`。
关闭时不调用检测器，不在原模型或 logits 编译函数中插入检测算子。

真实 forward 的 chunk 全部提交后入队，在主线程的 `get_finished_kv_transfers()` 中
调用 `poll()`：队首事件 `query()` 成功才出队、读取布尔数组并输出日志，随后继续
消费下一项；队首未完成则结束本次轮询，保持 FIFO 顺序。该入口在没有 KV connector
时也执行；零 token 且无 KV connector 的提前返回点同样执行 poll。
pooling 同样不依赖 sampled-token copy。SUMMARY 不向
`AsyncTPUModelRunnerOutput` 传递状态，也不调用事件 `synchronize()`。
检测前不依赖 CPU 布尔值决定后续 forward 或采样；正常轮也执行归约和小数组回读。
完整服务的吞吐与延迟损失需要通过开关 A/B 测量。

参考 RTP-LLM
[PyWrappedModel](https://github.com/alibaba/rtp-llm/blob/97fbaaaf41ca0f4ebfabdd2d7647f81b2737a88e/rtp_llm/cpp/models/PyWrappedModel.cc#L1437)
的 `isnan(...).any().item<bool>()` 语义，以及
[CHECK_NAN 开关](https://github.com/alibaba/rtp-llm/blob/97fbaaaf41ca0f4ebfabdd2d7647f81b2737a88e/rtp_llm/server/server_args/profile_debug_logging_group_args.py#L115)。
此实现按所需服务策略记录 ERROR 并继续，RTP-LLM 的原检查会报错中止。

本检测在 `SUMMARY` 及以上启用，逐层检查在 `DETAIL` 及以上启用，KV 保存检查保持独立。
NaN/Inf 被观测到只能说明该输出已异常，不能据此定位最先产生 NaN/Inf 的层或算子。

## 指标和历史折线

同一检查轮次的 hidden、logits、chunk 和参与检查的 worker 取 OR。每个 worker
移交 `(round_id, rank, 0/1)`；轮次 ID 只在真实检查入队时递增。最后 PP stage 的
TP × PCP 个 rank 都报告后，异常轮次增加一次异常轮数，多个 rank 同时命中也只增加 1。
一次 poll 可消费多轮，各轮分别聚合；rank 之间完成时间不同也不会合并或重复计数。
正常轮仍上报 0，供聚合器收齐参与 rank；没有实际检查时不更新。

上报复用 `TpuKVConnectorStats` 和 factory 注入的 Counter 类。支持单个
TPUConnector / TPURaidenConnector，以及包含唯一 TPU stats leaf 的 MultiConnector。
没有 TPU leaf 时保持仅日志检测；多个 TPU leaf 在检测器构造入口报 `ValueError`。
沿用原有 stats logger；`--disable-log-stats` 会关闭这条指标输出路径。

| Prometheus 名称 | 含义 |
| --- | --- |
| `vllm:tpu_model_nan_rounds_total` | 发现 NaN/Inf 的调度轮数 |

DashLog 兼容后端将名称中的冒号转为点，保留 `_total`。Counter 沿用
`model_name/engine` 和后端已有环境标签，不增加 request、rank 或 tensor 标签。

Grafana 使用 Time series、range query 和阶梯线，直接显示累计异常轮数：

```promql
vllm:tpu_model_nan_rounds_total
```

DashLog 后端使用名称 matcher：

```promql
{__name__="vllm.tpu_model_nan_rounds_total"}
```

曲线每次上涨表示新增异常轮次；沿实例、engine 和上涨的时间查询上述 ERROR 日志，
定位具体 rank 和 tensor。保留各实例的独立曲线，不聚合掉实例，不补零或连接空白。
需要窗口增量时可使用 `increase(vllm:tpu_model_nan_rounds_total[$__rate_interval])`，
需要窗口内是否异常时可再接 `> bool 0`。窗口增量会处理 Counter 重置，
并按窗口外推，可能不是整数，见 [Prometheus increase 说明](https://prometheus.io/docs/prometheus/latest/querying/functions/#increase)。

Counter 在 logger 进程生命周期内累计，重启后归零，历史样本由监控后端保留。
累计值不变只表示没有新增计数，无法区分正常检测、空闲和关闭检测。
两次采集之间的短暂异常会保留在累计值中；时间精度仍取决于报告、采集和查询窗口，
不逐个还原调度 step，首次采集前及重启处的异常时间须结合日志确认。
队列消费和跨 rank 聚合可能延后上报；空闲时尚未完成的尾项不会主动消费，
要等下一次 runner 调用，日志时间和指标上涨时间均不是异常产生的精确时间。
沿用后端既有传输保障，不新增持久队列或失败重试机制。

## 验证

在已经安装项目固定依赖的环境中运行 CPU 契约测试：

```bash
JAX_PLATFORMS=cpu pytest -q \
  tests/runner/test_model_nan_check.py tests/distributed/test_model_nan_metrics.py \
  tests/test_envs_builders.py
```

测试覆盖默认关闭、两个 tensor 各自的 NaN/Inf、padding NaN/Inf、`+Inf` / `-Inf`、
PCP 非连续有效行、多次检查之间的独立结果，以及正常/异常退出 warmup 后恢复
检查；这些是 CPU 数值、行映射与生命周期验证。
CPU 测试执行同一个 JAX 归约函数，仅替换 TPU 调用和完成事件；生产路径没有 CPU
数值检测分支。另检查归约输出必须是标量 bool，且 mask 排除 padding NaN/Inf。
指标测试覆盖 chunk OR、一次消费、正常 0 的序列化、无传输样本时创建 stats、
MultiConnector 保留其他 child，以及真实 registry 的 Counter 导出和 engine 隔离。
另覆盖前向及未完成队首不等待或读取主机结果、FIFO 已完成前缀消费、重复 poll
不重复日志或指标、日志运行于调用线程、无 KV connector 时主线程仍消费，以及
多个 rank 延迟报告多轮时的 OR 和独立计数。

真实 TPU 验收应在诊断副本中对有效 hidden 和 logits 分别注入 NaN/Inf，检查对应
ERROR、后续 chunk 仍能执行和输出 tensor 未被检测器修改；另将 NaN/Inf 只注入
padding，要求无 ERROR。分别覆盖普通 P/D、MTP verify、PCP prefill 和最后 PP stage；
PCP 与 MTP 组合沿用现有仅 prefill/首 token 的能力边界。
使用同负载多轮比较开关两组的吞吐与延迟，保留
`avg/min/max/count/stddev/p90/p99`。CPU 测试不替代 TPU 编译、collective 和
端到端验收。
