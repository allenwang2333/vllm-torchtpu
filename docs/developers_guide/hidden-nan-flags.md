# 逐层 hidden NaN 现场快照

## TL;DR

设置 `TPU_HIDDEN_NAN_LOG=1` 后，每个 worker 对 Target / Draft 分别检查 decoder
layer 的 rank-local hidden 输出。一轮首次发现 NaN 时保留该层的完整 hidden 和该次
forward 的实际输入 token；每 100 次异常轮只保存前 3 次，即第 1–3、101–103 次，
正常轮不推进异常计数。日志提示快照文件位置，不再输出逐层 true/false 数组。

```bash
export TPU_HIDDEN_NAN_LOG=1
```

当前支持 PP=1 的 causal LM 和异步 MTP / hidden-state Draft。异步 TP/PCP 多 rank
需要 KV connector，以保证每个 rank 的 AsyncOutput 被消费。关闭开关时不安装诊断
hook、不分配诊断 buffer、不提交诊断回读。

## 文件与内容

文件默认位于 Python 临时目录下的 `vllm-hidden-nan/`，通常是
`/tmp/vllm-hidden-nan/hidden-nan-<host>-<pid>-<role>-round<N>-layer<L>.pt`。
临时目录遵循进程启动时的标准 `TMPDIR` 设置。

```python
snapshot = torch.load(path, weights_only=True)
hidden = snapshot["hidden_states"]
tokens = snapshot["input_ids"]
```

文件包含 `role`、`round`、`layer_id`、`nan_event_in_window`、`req_ids`、
`hidden_states` 和 `input_ids`。hidden 保留原 dtype；token 为 int32。
只保存首次异常 forward 的实际长度，包括其 padding，不把缓冲区剩余容量写入文件。
不保存逐层计数、residual、最终 norm、logits 或 KV。

`req_ids` 是该调度轮的请求集合；`input_ids` 是首次异常 forward 的 rank-local
输入，可能仅覆盖其中一个 chunk，两者不是逐项对应的数组。PCP 下保存的是该 rank
本地输入；需要联合各 rank 的快照分析全局 batch。Target 的 token 在异步占位符替换后
绑定；MTP 每个实际 embedding 调用重新绑定，因此后续 draft step 不沿用 step 0 的 token。
多模态包装只检查语言模型的 decoder block，保存送入词嵌入的 token。

## 首次命中与限频

一次有模型输出的 `execute_model` / `sample_tokens` 调度步为一轮。进入该轮后清除
首次命中位置；每个模型沿实际执行顺序检查 block 输出。首次发现 NaN 后，该模型本轮
后续层、chunk 和重复 draft forward 不再扫描，也不覆盖快照。“首次”指执行顺序，
不是全局最小层号。Target 与 Draft、不同 worker 的计数相互独立。

检测范围沿用原功能：包含真实调度轮内的 padding 和用于 collective 配对的 dummy
forward；独立 idle dummy 和启动 warmup 不计数、不保存。零 token 的 KV 协调步不推进
轮次。首次 NaN 之后没有本轮传播统计；这项功能用于保留现场，不能单凭一份快照自动
判定 NaN 的完整传播路径。

限频选择在设备端执行。事件 phase 仅在 0–99 内循环，不依赖可能延迟的 CPU 消费进度，
也不会因后续正常轮或本轮更多异常层而占用新名额。没有进程级“首次异常后永久冻结”。

## 数据与执行边界

诊断拥有独立 JAX kernel、Torch bridge、hook/controller 和快照 buffer。
使用语言模型最外层 `ModuleList` / 整数索引 `ModuleDict` 的完整 block 输出；
Tensor 和首项为 hidden 的 tuple 均支持，hidden 需要是非空二维 Tensor。

每个模型保存 4 个 int32 状态值及一份最大容量 hidden 快照。Target 容量来自 runner
的最大 token bucket；Draft 同时覆盖最多请求数乘以每请求 draft query 长度。
共享词嵌入会触发多个角色的 hook，因此 token staging/snapshot 采用各角色容量的最大值。
固定容量设备 staging 加实际行数，避免无 guards 的已编译模型绑定到旧 Tensor 或旧长度。

存在词嵌入时只在词嵌入入口绑定 token，不再同时安装模型根 hook；普通输入路径的绑定
进入编译图，避免 runner 和根 hook 重复提交 eager copy/fill。不同 TP 的 Draft 使用
`nn.Embedding` 时沿用 vLLM 的 `embed_tokens` 替换约定；位置嵌入、LM head 和视觉分支
不作为 token 输入。`inputs_embeds` 路径由 runner 绑定替换后的实际 token；没有词嵌入的
模型保留根 hook。此调整不改变逐层 kernel、采样限频或快照 ownership。

**只 donate 检测状态，hidden/token 快照不可 donate。** 当前 TorchTPU 的 `clone()`
可以与源共享 PJRT buffer，不能当作物理深复制；让快照以非 donation 的函数输出保留
版本，readback 再持有本轮的逻辑版本，才能跨越后续更新安全地延迟消费。

正式输出组装时异步回读小状态并记录 event，随后关闭本轮设备检测。在原 sampled-token
等待之后消费事件，仅命中采样窗口时才回传大快照并调用 `torch.save()`。这避免正常轮
的大 D2H/文件写入；设备端分支、buffer 更新和版本保留仍有成本，需以真实模型测量。

命中保存时，当前先按缓冲区容量回传，再以实际行数裁剪落盘。故障轮的 D2H 与 CPU
临时内存成本因此取决于最大容量，不能仅用当前 batch 的落盘大小估算。

## 验证

长期测试覆盖首次命中不被后续层覆盖、正常轮不计数、205 次异常的窗口边界、
FP32/BF16、稀疏真实层号、warmup、idle、共享 embedding、大 Draft、unbound MTP 输入绑定
以及异步消费的轮次归属。原生回归必须使用真实 TorchTPU donation 与 D2H；CPU 替身
不能复现 PJRT buffer 失效。

准备好项目依赖、TPU 拓扑及设备权限后，先运行实际 Torch bridge 的 ownership 回归，
再运行 kernel case：

```bash
JAX_PLATFORMS=tpu,cpu pytest -q --use-tpu \
  tests/runner/test_hidden_nan_flags.py::test_native_readback_retains_snapshot_after_later_donated_updates \
  tests/kernels/test_hidden_nan_flags_kernel.py
```

CPU 验证需要使用 CPU transport/event 测试环境，原生 case 会明确跳过；同时验证
`tests/runner/test_hidden_nan_flags.py`、`test_tpu_runner_async_output.py` 和
`test_tpu_runner_kv_connector_lifecycle.py`。性能验证使用同一 wheel、模型、拓扑与输入，
分别测关闭检测、正常检测、异常但限频跳过、异常并保存；重复采样并记录
avg/min/max/count/stddev/p90/p99，区分客户端延迟、模型 forward、设备 trace 和文件写入。
与旧 bool 实现比较时，需要在相同部署重新运行该实现；关闭检测的结果只用于衡量
绝对诊断成本，不能替代旧版本基线。真实 PD/MTP 与不执行 Draft 的 Target-only 探针应分开记录。
