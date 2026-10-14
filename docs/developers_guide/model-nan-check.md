# 最后一层 hidden 和 logits 的 NaN 检测

## TL;DR

设置 `TPU_CHECK_NAN=1`，检查每个真实主模型 forward chunk 的最后一层 hidden
和采样前 logits。默认关闭。命中时记录 ERROR 并继续执行，检测器返回布尔结果，
不修改 tensor、不替换 token，也不触发 dump。

配置唯一 TPU connector stats leaf 且启用 stats logger 时，每轮结果还通过现有
connector metrics 通道上报。只保留异常轮数一个 Counter，发现 NaN 的轮次增加 1，
正常轮累计值保持不变；绘制累计曲线即可查看历史异常及上涨时间。

只检查有效行中的 NaN；padding 和合法 `Inf` 不算异常。开启后增加设备归约和
host 同步，适用于诊断，吞吐与延迟开销需要在实际部署中测量。

## 启用和输出

在 worker 启动前设置环境变量；runner 构造时读取，修改后需要重启 worker。

```bash
export TPU_CHECK_NAN=1
# 使用原有 vllm serve 或服务启动命令。
```

不设置变量或设置 `TPU_CHECK_NAN=0` 即关闭。沿用 `env_bool` 的解析规则，支持
`true/false`、`yes/no`、`on/off`；非法值在启动时抛出 `ValueError`。

异常日志示例：

```text
ERROR ... NaN detected in last_hidden: host=worker-a, engine=2, rank=3, shape=(128, 4096)
ERROR ... NaN detected in logits: host=worker-a, engine=2, rank=3, shape=(8, 151936)
```

`host` 为 worker hostname，`engine` 为 data-parallel rank；`rank` 使用 runner 的
profiler rank，`shape` 是包含 padding 的 tensor shape。时间使用原有 logger 时间戳。
每个含有效行 NaN 的 chunk 分别记录一次对应 tensor 的 ERROR。

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
jnp.isnan(tensor).any(where=valid_rows[:, None])
```

有效行 mask 送到设备，`isnan`、padding 过滤及最终 `any()` 均在设备上完成。
输出是单个标量 bool，同步后 CPU 只读取这个 bool，不再回传或处理行标志，
也不复制完整 hidden/logits。沿用桥接层已有的 JIT，不额外包一层 `torch.compile`。
关闭时不调用检测器，不在原模型或 logits 编译函数中插入检测算子。

与 RTP-LLM 的 `isnan(...).any().item<bool>()` 一样，最终只取回标量。
检测仍有额外设备归约、mask 传输和 host 同步开销，完整服务的吞吐与延迟损失
需要通过开关 A/B 测量。

参考 RTP-LLM
[PyWrappedModel](https://github.com/alibaba/rtp-llm/blob/97fbaaaf41ca0f4ebfabdd2d7647f81b2737a88e/rtp_llm/cpp/models/PyWrappedModel.cc#L1437)
的 `isnan(...).any().item<bool>()` 语义，以及
[CHECK_NAN 开关](https://github.com/alibaba/rtp-llm/blob/97fbaaaf41ca0f4ebfabdd2d7647f81b2737a88e/rtp_llm/server/server_args/profile_debug_logging_group_args.py#L115)。
此实现按所需服务策略记录 ERROR 并继续，RTP-LLM 的原检查会报错中止。

本检测独立于 `TPU_HIDDEN_NAN_LOG` 的逐层检查和 KV 保存检查。NaN 被观测到只能说明该
输出已异常，不能据此定位最先产生 NaN 的层或算子。

## 指标和历史折线

同一调度轮次的 hidden、logits、chunk 和参与检查的 worker 取 OR。每个 worker
移交一个 `0/1`。聚合后异常轮次只增加一次异常轮数，多个 worker 同时命中也只增加 1。
正常轮保留 `[0]`，沿用上报链路输出当前累计值；没有实际检查时不更新。

上报复用 `TpuKVConnectorStats` 和 factory 注入的 Counter 类。支持单个
TPUConnector / TPURaidenConnector，以及包含唯一 TPU stats leaf 的 MultiConnector。
没有 TPU leaf 时保持仅日志检测；多个 TPU leaf 在检测器构造入口报 `ValueError`。
沿用原有 stats logger；`--disable-log-stats` 会关闭这条指标输出路径。

| Prometheus 名称 | 含义 |
| --- | --- |
| `vllm:tpu_model_nan_rounds_total` | 发现 NaN 的调度轮数 |

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
沿用后端既有传输保障，不新增持久队列或失败重试机制。

## 验证

在已经安装项目固定依赖的环境中运行 CPU 契约测试：

```bash
JAX_PLATFORMS=cpu pytest -q \
  tests/runner/test_model_nan_check.py tests/distributed/test_model_nan_metrics.py \
  tests/test_envs_builders.py
```

测试覆盖默认关闭、两个 tensor 各自的 NaN、padding NaN、合法 `±Inf`、
PCP 非连续有效行、多次检查之间的独立结果，以及正常/异常退出 warmup 后恢复
检查；这些是 CPU 数值、行映射与生命周期验证。
CPU 测试执行同一个 JAX 归约函数，仅替换 TPU 调用和同步；生产路径没有 CPU
数值检测分支。另检查归约输出必须是标量 bool，且 mask 排除 padding NaN。
指标测试覆盖 chunk OR、一次消费、正常 0 的序列化、无传输样本时创建 stats、
MultiConnector 保留其他 child，以及真实 registry 的 Counter 导出和 engine 隔离。

真实 TPU 验收应在诊断副本中对有效 hidden 和 logits 分别注入 NaN，检查对应
ERROR、后续 chunk 仍能执行和输出 tensor 未被检测器修改；另将 NaN 只注入
padding，要求无 ERROR。分别覆盖普通 P/D、MTP verify、PCP prefill 和最后 PP stage；
PCP 与 MTP 组合沿用现有仅 prefill/首 token 的能力边界。
使用同负载多轮比较开关两组的吞吐与延迟，保留
`avg/min/max/count/stddev/p90/p99`。CPU 测试不替代 TPU 编译、collective 和
端到端验收。
