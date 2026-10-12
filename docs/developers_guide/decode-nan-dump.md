# Decode hidden NaN 的 batch 输入与模型缓存转储

## TL;DR

`TPU_DECODE_NAN_DUMP=1` 在 target 和 eagle3/MTP draft 的 logits 调用前检测真实
decode 行的 hidden NaN。每个 worker 的两个模型独立计数，异常索引 `0、20、40…`
落盘，即首次异常立即保存，以后每隔 20 次异常保存。正常 forward、prefill、padding
和启动预热不消耗计数。

每份转储覆盖当前一次模型 forward 的完整输入批次，包含正常和异常 request 的
全部已分配缓存页。target 和 draft 共用实现，各自只保存所属层的缓存。缓存是
**forward 后、logits 前**的当前状态：GDN 的旧递推状态可能已经覆盖，因此这不是
保证精确重放本步计算的执行前 checkpoint。

## 入口与范围

```bash
export TPU_DECODE_NAN_DUMP=1
export TPU_DECODE_NAN_DUMP_DIR=/mounted-disk/decode-nan
```

默认关闭；关闭时不记录 forward，不执行 NaN 检查或复制缓存。
当前支持 Qwen GDN、PP=1、统一 layer-major KV pool，以及未融合的 eagle3/MTP drafting。
不支持组合在 runner 初始化时集中报错。`VLLM_TPU_FUSED_DRAFT_LOOP` 须关闭。

检测在模型执行线程执行，包含同步等待，不能视为无性能开销。
未到落盘点时仅归约 hidden 的 NaN flags；到落盘点才复制输入、metadata、hidden
和缓存。原 MTP 低接受率 hook、异步诊断包装及 one-shot hidden dump 已移除。
独立的 KV 保存 NaN 检查保留原有语义。

这里的 batch 是一次实际 forward 的 request chunk；包含该 chunk 全部 request，
不是仅第一个异常 request。chunked execution 按每次 forward 单独检测。
每个文件记录 rank 和 DP index，覆盖本 rank 的模型层；各 worker 本地独立计数，
没有增加跨 rank 的条件 collective。排查跨 EP/TP 问题时须收集关联 worker 的文件，
不能把单个 worker 的转储解释成整个分布式实例快照。

## 格式及读取

目录名为 `vllm-decode-nan-<target|draft>-<pid>-<异常索引>-<随机后缀>`。
文件均使用 `pickle.HIGHEST_PROTOCOL`：

- `inputs.pkl`：实际 forward 参数（设备替换后的 tokens/positions，draft 输入 hidden
  和 step index）、逐层 attention metadata、request token 记录和分组 block IDs、
  logits 前 hidden、选择 logits 行的索引及对应 request 行归属。
- `layer-NNNN.pkl`：首个 pickle 是层名、group、形状、dtype、页映射和 block 数；
  后续每个 pickle 是一个 manager block 的完整 `uint8` byte tensors。
  统一使用 byte tensors，避免 Torch 2.13 直接 pickle FP8 Tensor 后无法回读的问题。
- `manifest.pkl`：最后写入，存在时表示输入及所有缓存文件均已写完；记录异常索引、
  正常/异常 request IDs、模型归属和缓存采集边界。I/O 失败会记日志，不生成完成 manifest。

```python
import pickle
from pathlib import Path

folder = Path('/mounted-disk/decode-nan/vllm-decode-nan-target-...')
with (folder / 'manifest.pkl').open('rb') as f:
    manifest = pickle.load(f)
with (folder / 'inputs.pkl').open('rb') as f:
    inputs = pickle.load(f)
for layer in manifest['layers']:
    with (folder / layer['path']).open('rb') as f:
        header = pickle.load(f)
        for _ in range(header['num_blocks']):
            block = pickle.load(f)
            # 按 block_id 和 kernel_pages_per_block 恢复原地址范围。
            for raw, dtype, shape in zip(block['tensor_bytes'],
                                         header['tensor_dtypes'],
                                         header['tensor_shapes']):
                selected_shape = (header['kernel_pages_per_block'], *shape[1:])
                tensor = raw.view(dtype).reshape(selected_shape)
```

FA/GDN pool 保留关联请求全部分配页，包括 GDN checkpoint 和尾部未写区域。
没有 64 MiB 截断上限；按页流式写入，避免在 host 同时积累整个 batch 的缓存。
文件可能很大，输出目录应位于有足够空间的挂载盘。
统一 GDN pool 保留 CPU carrier 的原始 dtype 和 bit pattern，不能把 carrier 数值
当作 GDN 的 BF16/FP32 状态；解释物理区域时仍须使用对应版本的 pool layout。
输入 token 记录可能包含异步调度尚未更新的 host 历史，回放应以实际 forward tensor
和设备 metadata 为准。

## 验证

基线为 R38 的 `alibaba/release-0.0.1@a503c18f`；环境依赖应与 R38 的安装清单匹配。

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/runner/test_decode_nan_dump.py
```

契约检查覆盖 `0/20/40` 独立计数、完整 batch 与模型隔离、prefill/padding/预热排除、
spec verify 行归属、manager/kernel page split 和 FP8 carrier
pickle 位模式。真实 TPU 验收需另行运行 PCP8→TP2/DP4、async+MTP，分别受控触发
target/draft，并核对输入、全层/cache block 覆盖、文件可读取及服务继续响应。
