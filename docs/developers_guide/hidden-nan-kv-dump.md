# hidden NaN 对应 request 的 KV dump

## TL;DR

设置 `TPU_HIDDEN_NAN_KV_DUMP=1` 后，在真实主模型 forward 结束、sampling 之前，
同步 hidden 并检测 NaN。首次发现异常时，同步缓存及 slot metadata，记录异常
request ID、hidden 行与该 request 在本 worker 上的 FA/GDN tensor。
实际截取的 hidden/KV 切片也执行 `synchronize_tensors(..., wait=True)` 后再复制到 CPU，
避免延迟执行的切片在同步前被读取。
默认关闭；每个 worker 生命周期最多尝试一次，单文件硬上限 64 MiB。

该入口独立于 V6D 的 `check_nan_on_save`，纯 Raiden P→D 也可用。目前入口支持
Qwen GDN 架构、PP=1、layer-major KV pool；不支持的组合在 runner 初始化时报错。
开启后，首次异常之前每个主模型 forward 都会等待同步并检查 hidden，会增加延迟。

## 开启与读取

在所有 prefill/decode worker 的启动环境中设置：

```bash
export TPU_HIDDEN_NAN_KV_DUMP=1
```

出现真实 request hidden NaN 时，日志打印 `Hidden NaN KV dump`、`request_id`、rank、
文件路径、字节数和 `truncated`。文件位于
`${TMPDIR:-/tmp}/vllm-hidden-nan-<pid>-<uuid>/request.pt`。
应在容器内读取，或复制到使用兼容 PyTorch 的分析环境：

```python
import torch

result = torch.load("/tmp/vllm-hidden-nan-<pid>-<uuid>/request.pt", weights_only=True)
print(result["metadata"])
for cache in result["caches"]:
    print(cache["kind"], cache["layer"], cache["block_ids"],
          cache["dtype"], cache["shape"], cache["nan_count"], cache["inf_count"])
    tensor = cache["tensor"]
```

`fa` 保留请求已分配的完整 kernel pages，包含尾部 padding/lookahead，不能把所有
非有限值都归因于有效历史 token。`gdn.conv` / `gdn.ssm` 是统一 pool 中按 canonical
manifest 提取的有效状态；private typed materialization 标为 `gdn_conv` / `gdn_ssm`。
模型预编译期间暂停采集，MTP 合成请求的 NaN 不会消耗真实请求的一次性额度。
该隔离使用启动生命周期，不依赖 request ID 命名。
GDN 记录本次读取的状态及实际写入的 checkpoint slots，不采集未使用的整个窗口。
GDN geometry 复用计算路径的 head 分片函数，在 TP-local heads 上继续应用 PCP，
保留 Q/K 复制语义；直接把 TP head 数用于 PCP pool 会错误地计算状态字节范围。
统一 pool 的 FP8 carrier 先按 TPU 的 128-lane tile 与 packing 还原物理字节，再按
状态真实 dtype 解码；直接对 CPU carrier 做 `view(bfloat16)` 会误报 NaN。
`regions` 描述各片段的拼接顺序，pool 状态 tensor 为一维有效元素，不含 ABI padding。

## 范围与容量

- PCP token layout 用于关联当前 chunk 的 request，忽略 padding 的 hidden NaN。
- 只保存首个异常 request；其他异常 ID 写入 metadata。各 worker 独立触发。
- 只覆盖本 rank 的 layer/page，不能单独代表其他 PCP/TP/DP rank。
- hidden 最多保留 8 个异常行；缓存 payload 预算为 63 MiB。
  超过预算即停止复制后续页，记录 `truncated` 与 `first_omitted`，并校验最终
  序列化文件不超过 64 MiB。文件系统写入失败只记录日志，不无限重试。
- dump 不修改 hidden、KV、sampling 或请求生命周期；后续 step 不再扫描或同步。
  重启 worker 会重置一次性状态；已有 dump 需要由运维按实验周期清理。

这些数据能定位哪些层/状态含 NaN，不能仅凭同一时刻的 NaN 数量证明生成 NaN 的算子。
完整根因仍需结合 token 有效范围、checkpoint、前后状态与模型输入验证。

## 可复现验证

在与部署相同的依赖环境运行：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/runner/test_hidden_nan_kv_dump.py
```

契约覆盖同步等待、PCP request 映射、FA manager/kernel page 映射、GDN read/written
slot、FP8/BF16 字节恢复的误报反例、截断、一次性写入及磁盘错误。硬件 ABI 仍需在
目标 TPU 上与 Mosaic typed word load 独立比对；CPU 反例不能替代真实硬件验收。
