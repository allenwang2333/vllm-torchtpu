# 逐层 hidden NaN 日志

## TL;DR

设置 `TPU_HIDDEN_NAN_LOG=1` 后，每个 decoder layer 结束时检查 rank-local
hidden 输出，每个模型只保存长度等于层数的布尔数组。某一调度轮存在 NaN 时，
在 `AsyncTPUModelRunnerOutput.get_output()` 消费采样结果后记录完整数组；正常轮次不打印。
不保存 hidden，不生成 pickle，不读取 KV，不改变模型输出。

当前仍为实验实现：R38 两机性能验收尚未通过。XProf 已完成 Prefill / Decode 各三次
采集，但设备事件耗时不能直接作为稳态 TPOT 增量。

```bash
export TPU_HIDDEN_NAN_LOG=1
```

当前入口支持 PP=1 的 causal LM，以及异步 MTP Draft，不限制 decoder layer 类名。
按语言模型最外层 `ModuleList` / 整数索引 `ModuleDict` 的直接成员安装 hook；每个完整
block 一个 flag，block 内部的 Attention、Linear、Norm 和嵌套容器不单独安装 hook。
多模态包装通过 vLLM 的 `get_language_model()` 取得语言模型，视觉层不计入数组。
block 输出可以是 Tensor 或首项为 hidden 的 tuple；当前 kernel 仍要求 hidden 为非空
二维 Tensor。本次仅扩展 hook 接入，kernel、Torch bridge 和异步回读路径保持原样。
异步 TP/PCP 多 rank 必须配置 KV connector，确保每个 rank 的 AsyncOutput 被消费。
关闭配置时不安装 hook、不分配诊断数组、不提交诊断回读。旧 A/B hidden 快照实现在
`codex/async-decode-nan` 的 `af5843bef8600b9b2f56e9c853a8fa080bb52f9c` 保留。

## 数组与轮次

Target 和 Draft 各自拥有一个 `torch.bool` 数组。按层在容器中的注册索引排序，使用连续
slot；不能把 MTP 层号当作数组下标。Q397 的 Target 为 60 层，R38 Draft 为 1 层，
实际 Draft 层号为 0。数组按模型分别输出，`layer_ids[i]` 对应 `isnan[i]`。

一次 `execute_model` / `sample_tokens` 调度步是一轮。进入该轮 chunk 循环前清零，
同一轮分块、重复 MTP forward 使用 OR 累积，因此一个异常 chunk 不会被后续正常
chunk 覆盖。零 token 的 KV 协调步没有模型 forward，不递增编号。检测包含 padding 和
有输出的调度轮中的 collective dummy forward；独立 idle dummy 没有 AsyncOutput，
不产生诊断日志。检测只反映本 rank
的 decoder hidden；不包含 residual、最终 norm、logits 或 KV。

以下为三层示例日志，真实日志打印完整数组：

```text
Hidden NaN detected: role=target round=1 layer_ids=[7, 11, 19] isnan=[false, true, true]
```

每个异常轮都会记录，没有首次命中后冻结或停检的行为。`round` 是 worker 内部递增
编号，不是 request ID，也不是跨 rank 的全局轮号。

## 执行边界

诊断独立拥有 JAX kernel、私有 Torch bridge、hook/controller。kernel 对 hidden
执行 `isnan().any()`，仅 donate 布尔数组；不 donate、复制或保留 live hidden。
入口统一校验支持范围，原模型、公共 backend 和 KV allocator 不承载诊断状态。

warmup 的 device arm flag 为 false，结束后启用；host arm flag 阻止 warmup 提交
无人消费的诊断 D2H。正式输出组装时异步回读 Target/Draft 的小数组，在最后一份
回读后记录一个 event；CPU 在 AsyncOutput 原 sampled-token wait 后消费此 event。
下一轮清零依赖设备队列顺序，CPU 目的数组由各轮 readback 持有；没有 forward 前
新增 CPU fence。这个较晚 event 的耗时仍必须通过真实模型性能对照验证。

Q397 的每 worker 逻辑持久数据为 60 + 1 个 bool 和一个共享 arm bool；不把逻辑字节数
等同于后端实际分配粒度、编译临时空间或性能无损的证明。

## 复现与验证

保持 R38 全部依赖；测试版本和两机验收证据记录在父仓实验文档中。
先验证 FP32/BF16、后续异常层、同轮 OR、轮次清零、dense layer slot、warmup 无回读
和异步消费顺序。hook 契约测试另覆盖非 Qwen block、Tensor / tuple 输出、稀疏
`ModuleDict` 索引、内部容器不重复检测，以及多模态包装只检测语言模型：

```bash
JAX_PLATFORMS=cpu pytest tests/kernels/test_hidden_nan_flags_kernel.py \
  tests/runner/test_hidden_nan_flags.py tests/runner/test_tpu_runner_async_output.py \
  tests/runner/test_tpu_runner_kv_connector_lifecycle.py
```

R38 CPU-only 测试使用隔离容器及测试用 TPU event mock；它不能代替 native TPU 验证。
实机需验证两份未消费 readback 在下一轮清零后仍各自保留原值，并完成异常/正常轮
日志检查。性能比较必须使用同 wheel 的 env-off 与 env-on，健康轮持续检测、无异常
日志；固定输入、cache salt、输出 token 数，重复运行并记录 avg/min/max/count/stddev/
p90/p99。不能用已 frozen 的旧 A/B feature 或保留 kernel 的 unarmed 模式代替 env-off。

## R38 XProf 与编译 shape

2026-10-14 在 PCP8 Prefill（34.37.32.206）和 TP2/DP4 Decode（34.37.69.96）
各采集三次，保持 R38 全部依赖。以下是 rank0 的实际 NaN 检查 hidden shape；
不是对所有已配置 bucket 的耗时覆盖。

| 阶段 | 模型 / 步骤 | hidden shape（BF16） | token compile size |
| --- | --- | --- | ---: |
| PCP8 Prefill | Target、MTP Draft | `[1024,4096]` | 1024 |
| TP2/DP4 Decode | Target、首次 MTP Draft | `[16,4096]` | 16 |
| TP2/DP4 Decode | 后两次 MTP Draft | `[192,4096]` | 192 |

rank0/core0 的 Target 单层完整 NaN 条件分支平均为 Prefill 11.741 µs、Decode 7.237 µs；
60 层 duration 相加平均为 0.704 / 0.434 ms。纯扫描子事件平均为 7.661 / 3.370 µs。
父条件分支包含子事件，不能重复相加；这些是在 profiler 下的设备 duration，
不代表未开启 profiler 时的关键路径延迟或功能性能验收。

当前 Prefill 的本地 token buckets 为 `[512,1024,2048,4096]`，Decode 配置为
`[16,32,64,80,96,128,192,256,512,1024]`。hook 在 `capture_model()` 预编译前安装；
hidden 特征宽度固定4096，token 维度补齐到 bucket。`layer_slot` 和 arm flag 是数据参数，
不按层编号或 NaN 布尔值做静态特化。

当前未开启 `VLLM_XLA_CHECK_RECOMPILATION`，因此不能由日志中的
`warmup compiled 0 graphs` 宣称没有编译。严格验证需要开启现有监控，重复覆盖所有
bucket、greedy/non-greedy 及 Draft/verify 路径，检查预热后新 cache entry 的实际
compile duration。已有短 trace 未识别到实际 XLA 编译事件，不证明全链路无热编译。

详细版本、采集命令、全部统计与浏览器故障排查位于父仓
`pd_disagg/docs/hidden-nan-xprof-r38-20261014.md`；原始 profile 保留在两机
`/experiment/perf/traces`，归档及 SHA256 清单见父仓证据目录。
