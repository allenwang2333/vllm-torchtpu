# Decode batch 逐层 KV cache NaN 检测

## TL;DR

在 Decode worker 启动前设置 `TPU_KV_CACHE_NAN_LOG=1`，每轮调度检查当前 batch
请求在 Target forward 前已有的有效 FA KV / GDN 输入状态。按层输出布尔数组；
异常轮打印完整数组和本轮 request IDs，正常轮保持静默。只读缓存，不保存完整 tensor。

```bash
export TPU_KV_CACHE_NAN_LOG=1
```

该开关独立于 `TPU_HIDDEN_NAN_LOG` 和 `TPU_CHECK_NAN`，可分别开启，也可同时开启。
关闭时不安装检测器、不分配诊断 flags、不执行扫描或回读。

## 检测时机与数组

一次非零 token 的 `execute_model` / `sample_tokens` 调度步为一轮。各 chunk 完成
attention metadata 准备、GDN state seed copy 和新请求 read-offset reset 后，在
Target forward 前扫描。扫描使用本次设备页表、修正后的 `seq_lens` 和
`query_start_loc`，已有 prefix 长度为 `seq_lens - query_len`；不会使用异步 MTP
尚未回滚的 CPU 乐观长度。

- FA：只检查请求已计算 prefix 的有效 token、KV heads 和 head dimension。
  不扫描其他请求的页、padding 请求、最后一页未写入的 token 或 feature padding。
- GDN：只检查有已有 prefix 的请求，按本次实际 state slot、read offset 和
  checkpoint table 选择 Conv / SSM 输入状态。新请求未使用的状态和已拒绝草稿的
  其他 checkpoint 不参与检查。
- Unified pool：沿用 GDN 的 `StateSourcePlan`，按 Conv 的 BF16 和 SSM 的声明类型
  解释有效字节，不能直接将承载状态的 FP8 pool 当成 FA KV 执行 `isnan`。

每轮先清零，再对各 chunk 和同一 decoder layer 的多个 cache module 做 OR。
`layer_ids[i]` 对应 `isnan[i]`；按数值层号排序，不要求层号连续。示例：

```text
KV cache NaN detected: role=target round=12 layer_ids=[0, 1, 2] isnan=[false, true, false] req_ids=['r0', 'r1']
```

`req_ids` 是被检查的整个 batch，**不是逐请求 NaN 归因结果**。
`round` 为 worker 内部编号；结果仅覆盖本 rank 的缓存，不做跨 rank 归约。
检测发生在本轮写入之前；本轮新写入的坏缓存会在后续读取该状态的调度步被发现。
本功能检查 Target 缓存，MTP 在 Target 上的 verify / rollback 已覆盖；Draft 自身的
缓存不在检测范围内。

只异步回读逐层 bool，并持有该轮独立的 CPU 目的数组；在
`AsyncTPUModelRunnerOutput.get_output()` 原采样结果等待后消费。
下一轮清零不会改变前一轮的快照。warmup 的 device arm flag 关闭，预编译相同扫描
图但不读取缓存或提交诊断回读；预编译结束后才启用检测。

## 支持范围与验证

支持 PP=PCP=DCP=1、非 SPMD 的 causal LM：浮点缓存的 FullAttention RPA / batched RPA，
以及 `VllmGatedDeltaNetAttention` 的独立或 pooled Conv / SSM。
支持普通、HND、统一 pool 和 block-major 缓存布局；暂不支持 MLA、sliding-window
cache 和其他 Mamba 实现。不支持的组合在检测器初始化入口报错。
异步多 TP rank 需要配置 KV connector，以保证每个 rank 的结果被消费。

检测器安装时，FA backend 的 shape 查询在 runner 自身的
`set_current_vllm_config(runner.vllm_config)` 范围内执行；调用方不需要预先设置
current config，查询结束后恢复外层配置。该上下文只属于诊断初始化路径。

CPU 验证命令（需要项目的 TorchTPU / vLLM / JAX 依赖及 pytest；可在项目镜像中执行）：

```bash
cat > /tmp/kv_cache_nan_cpu.py <<'PY'
def pytest_configure(config):
    from types import SimpleNamespace
    import torch
    config.option.use_tpu = False
    torch.tpu = SimpleNamespace(Event=None)
PY

PYTHONPATH="/tmp:${PYTHONPATH:-}" JAX_PLATFORMS=cpu \
TPU_SKIP_MDS_QUERY=true TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q -p kv_cache_nan_cpu \
  tests/kernels/test_kv_cache_nan_flags_kernel.py \
  tests/runner/test_kv_cache_nan_flags.py \
  tests/runner/test_tpu_runner_async_output.py \
  tests/runner/test_hidden_nan_flags.py \
  tests/kernels/test_hidden_nan_flags_kernel.py \
  tests/runner/test_tpu_runner_kv_connector_lifecycle.py \
  tests/runner/test_model_nan_check.py
```

2026-10-15 在 R42 镜像 `6831681426f6` 的隔离 CPU 容器验证通过 92 项测试，
基础源码为 Release `15d96006e`；依赖为 Torch `2.13.0+cpu`、vLLM `0.29.0`、
JAX `0.10.2`。Ruff、格式、文档和环境变量检查通过。
完整依赖环境的 pyrefly 检查在基础源码及修改后均报告 17 个既有错误，无新增错误；
这不等同于仓库完整类型检查通过。

2026-10-16 在 R43 镜像 `ba898e8ddf08` 中，以 Release `9a7f6622e` 加初始化
修复运行上述 CPU 命令，95 项通过。新增回归组合覆盖普通 / batched RPA backend
和无外层配置 / 已有其他外层配置，使用真实 backend shape 查询与 vLLM 配置上下文，
并检查安装后上下文恢复。修复前，两个 batched 组合分别复现
`Current vLLM config is not set` 和误读外层未解析布局的异常；修复后四个组合通过。

CPU 测试中的 TPU event / 数据传输使用测试替身，页表读取和归约执行真实 JAX 逻辑；
CPU 验证不能代替真实 TPU 编译、异步执行和模型端到端验收。实机验收需在独立诊断
实例上，向请求有效缓存与无关页分别注入 NaN，确认前者对应层置 true、后者不告警，
并验证连续异常轮和随后正常轮。该诊断会增加缓存读取和设备工作，尚无实机性能结论。
