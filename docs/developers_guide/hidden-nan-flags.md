# 逐层 hidden NaN/Inf 现场快照

## TL;DR

设置 `TPU_NAN_LOG_LEVEL=2`（`DETAIL`）或更高等级后，每个 worker 对 Target / Draft
分别检查 decoder
layer 的 rank-local hidden 输出。一轮首次发现 NaN/Inf 时保留该层的完整 hidden 和该次
forward 的实际输入 token。任一模型命中后，全局跳过接下来的 20 个调度轮次；
上一份导出完成后才允许恢复。快照异步复制到 CPU，以 pickle 写盘并记录路径。
每个 rank 同时最多导出一轮，包含该轮命中的 Target / Draft 快照。
每个 rank 在一次 worker 运行期间最多提交 10 轮导出，同轮 Target / Draft 合计一次，
因此最多产生 20 个快照文件。额度用尽后停止 L2 逐层扫描和回读，已提交的导出继续完成；
L1 最后 hidden / logits 检查继续执行。

```bash
export TPU_NAN_LOG_LEVEL=2
```

当前支持 PP=1 的 causal LM 和异步 MTP / hidden-state Draft。异步 TP/PCP 多 rank
需要 KV connector，以保证每个 rank 的 AsyncOutput 被消费。关闭开关时不安装诊断
hook、不分配诊断 buffer、不提交诊断回读。

等级累计启用，`DETAIL` 同时开启最后 hidden / logits 检查；完整配置见
[NaN/Inf 诊断等级](model-nan-check.md#启用和输出)。旧 `TPU_HIDDEN_NAN_LOG` 不再读取。

## 文件与内容

文件默认位于 Python 临时目录下的 `vllm-hidden-nan/`，通常是
`/tmp/vllm-hidden-nan/hidden-nan-<host>-<pid>-<role>-round<N>-layer<L>.pkl`。
临时目录遵循进程启动时的标准 `TMPDIR` 设置。

```python
import pickle

with open(path, "rb") as stream:
    snapshot = pickle.load(stream)
hidden = snapshot["hidden_states"]
tokens = snapshot["input_ids"]
```

文件包含 `role`、`round`、`layer_id`、`req_ids`、
`hidden_states` 和 `input_ids`。hidden 保留原 dtype；token 为 int32。
只保存首次异常 forward 的实际长度，包括其 padding，不把缓冲区剩余容量写入文件。
不保存逐层计数、residual、最终 norm、logits 或 KV。

`req_ids` 是该调度轮的请求集合；`input_ids` 是首次异常 forward 的 rank-local
输入，可能仅覆盖其中一个 chunk，两者不是逐项对应的数组。PCP 下保存的是该 rank
本地输入；需要联合各 rank 的快照分析全局 batch。Target 的 token 在异步占位符替换后
绑定；MTP 每个实际 embedding 调用重新绑定，因此后续 draft step 不沿用 step 0 的 token。
多模态包装只检查语言模型的 decoder block，保存送入词嵌入的 token。

## 首次命中与限频

一次有模型输出的 `execute_model` / `sample_tokens` 调度步为一轮。
每个模型沿实际执行顺序检查 block 输出。首次发现 NaN/Inf 后，该模型本轮
后续层、chunk 和重复 draft forward 不再扫描，也不覆盖快照。“首次”指执行顺序，
不是全局最小层号。同一异常轮仍允许 Target / Draft 分别保留首个异常层，
两者共用冷却状态；不同 worker 独立处理。

检测范围沿用原功能：包含真实调度轮内的 padding 和用于 collective 配对的 dummy
forward；独立 idle dummy 和启动 warmup 不计数、不保存。零 token 的 KV 协调步不推进
轮次。首次 NaN/Inf 之后没有本轮传播统计；这项功能用于保留现场，不能单凭一份快照自动
判定 NaN/Inf 的完整传播路径。

设备端首次命中时立即关闭后续轮次的检测，不依赖可能延迟的 CPU 消费进度。
若第 N 轮命中，第 N+1 至 N+20 轮不检测，最早第 N+21 轮恢复。
若此时 D2H 或 pickle 写盘还没有完成，则继续暂停检测和新 dump，推理继续运行。
导出完成并满足轮次条件后才清除首次命中状态，避免覆盖仍被读取的快照。

导出总额度在提交后台任务时计数，正常轮和重复消费不占用额度，重新 `arm()` 也不重置。
达到 10 轮后不再恢复 L2 检测或提交新的快照回读，重启 worker 后重新计数。
既有文件不自动删除；此上限按每次 worker 运行计数，不累计磁盘中历史运行的文件。

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
模型保留根 hook。

检测 kernel 将 hidden 与检测结果一起通过 optimization barrier 返回，hook 把返回的
hidden 交给下一层。这样下一层依赖本层检测完成，编译器不能把所有检测推迟到模型末尾。
正常分支不复制 hidden；只有首次发现 NaN/Inf 时才写入快照。
快照在设备上使用一维存储，避免跟随模型激活的二维 tiling 变化而在正常分支产生布局
复制；仅异常分支展平 hidden，CPU 导出时恢复原来的行列形状。

**检测状态和快照均允许 donation，导出期间保持设备侧写保护。** 不再用 GPU `clone()`
保留历史版本；每个模型只保留一份固定容量快照。对 4096×8192、BF16 的 Target / Draft，
两份快照分别为 64 MiB，合计 128 MiB，另有少量状态和 token buffer。
编译临时区是否产生额外复制仍需以目标硬件的 HLO 和内存分配验证。

正式输出组装时异步回读小状态并记录 event，随后关闭本轮设备检测。在原 sampled-token
等待之后消费小状态，仅异常轮提交大快照的异步 D2H。单个后台任务等待 D2H 完成、
执行 `pickle.dump()`、原子替换临时文件并记录日志。正常轮没有大快照 D2H，
导出忙时不创建第二份 CPU dump，也不等待写盘完成。

命中保存时，当前先按缓冲区容量回传，再以实际行数裁剪落盘。故障轮的 D2H 与 CPU
临时内存成本因此取决于最大容量，不能仅用当前 batch 的落盘大小估算。

## 验证

长期测试覆盖首次命中不被后续层覆盖、正常轮不复制、20 轮冷却和慢写盘的占用边界、
10 轮导出上限及 Target / Draft 同轮计数、额度用尽后停止回读、
FP32/BF16、稀疏真实层号、warmup、idle、共享 embedding、大 Draft、unbound MTP 输入绑定
以及异步消费的轮次归属。原生回归必须使用真实 TorchTPU donation 与 D2H；CPU 替身
不能复现 PJRT buffer 失效。

准备好项目依赖、TPU 拓扑及设备权限后，先运行实际 Torch bridge 的 ownership 回归，
再运行 kernel case：

```bash
JAX_PLATFORMS=tpu,cpu pytest -q --use-tpu \
  tests/runner/test_hidden_nan_flags.py::test_native_donated_snapshot_survives_delayed_readback \
  tests/kernels/test_hidden_nan_flags_kernel.py
```

CPU 验证需要使用 CPU transport/event 测试环境，原生 case 会明确跳过；同时验证
`tests/runner/test_hidden_nan_flags.py`、`test_tpu_runner_async_output.py` 和
`test_tpu_runner_kv_connector_lifecycle.py`。性能验证使用同一 wheel、模型、拓扑与输入，
分别测关闭检测、正常检测、异常但限频跳过、异常并保存；重复采样并记录
avg/min/max/count/stddev/p90/p99，区分客户端延迟、模型 forward、设备 trace 和文件写入。
与旧 bool 实现比较时，需要在相同部署重新运行该实现；关闭检测的结果只用于衡量
绝对诊断成本，不能替代旧版本基线。真实 PD/MTP 与不执行 Draft 的 Target-only 探针应分开记录。
