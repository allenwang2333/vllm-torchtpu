# D 侧容量抢占即失败

## TL;DR

在 KV consumer 的 `--additional-config` 中设置
`{"abort_on_preemption": true}`，被容量调度选中的请求将结束为
`FINISHED_ERROR`，携带 `model.KVCachePreempt`，不再进入 waiting 重算。
默认关闭；producer、`kv_both` 和没有 KV connector 的实例不能开启。
本实现通过 TPU init patch 接入 vLLM 0.29，使用官方 scheduler，并保留 Raiden
加载失败的独立响应适配器。

这是容量策略，不保证选择最长或占用最多 KV 的请求，也不解决总容量不足。
FCFS 和 PRIORITY 继续使用上游 victim 选择规则。

## 配置与边界

仅在 D 的已有启动参数中增加：

```bash
--additional-config '{"abort_on_preemption": true}'
```

如果已有 `additional_config`，将此字段合并到同一个 JSON 对象。
Python 入口使用 `EngineArgs(additional_config={"abort_on_preemption": True}, ...)`。
值必须是布尔类型。关闭时删除字段或设置 `false`。

该入口替代上游源码补丁中的原生 `--abort-on-preemption` 参数，
不要求修改已安装的 vLLM 文件或扩展其配置 dataclass。
原生 `Scheduler`、`AsyncScheduler` 可用；覆盖了 `schedule()` 的自定义 scheduler
在构造时拒绝开启。Raiden 的临时 scheduler 类已删除，迁移方式见
[响应完成与错误](raiden_pd_completion.md)。
管理接口 `reset_prefix_cache(reset_running_requests=True)` 仍走上游
`_preempt_request()`，不套用容量失败策略。

EngineCore 终态为：

```json
{
  "finish_reason": "error",
  "stop_reason": "kv_cache_preempted",
  "new_token_ids": [],
  "kv_transfer_params": {
    "kv_transfer_error": {
      "status_code": 500,
      "status_name": "model.KVCachePreempt",
      "status_message": "D-side request was preempted due to KV cache capacity."
    }
  }
}
```

原生 OpenAI serving 会将 `FinishReason.ERROR` 转成通用推理错误；
不能据此承诺 HTTP body 包含 `model.KVCachePreempt`。流式响应头发出后的
错误格式也取决于 serving 层。结构化 metadata 面向读取
`RequestOutput.kv_transfer_params` 的消费者；本功能不修改 DashLLM adapter 或重试策略。

## 生命周期与实现

`patch_registry` 在 `platform_activation`、`engine_core` 和 `worker_init`
注册 `vllm_torchtpu:_patch_vllm_abort_on_preemption`。
回调只包装 `Scheduler.__init__`，入口集中校验后，仅对开启的实例绑定容量版
`schedule()` 和输出包装器。关闭实例继续使用原始调度和输出方法。

容量版调度在上游方法的五处精确上下文应用内存补丁，复用其余调度逻辑，
特别是 PRIORITY victim 已在本步排入时的 token、spec 和 encoder 账目撤销。
vLLM 尚无“移除 victim 前”的独立 hook，因此不能只包装 `_preempt_request()`。
上下文不匹配立即报错，升级 vLLM 时需重新核对，不静默降级或模糊应用。
生成的方法通过 `linecache` 保留源码，便于定位 traceback。

选中 victim 后，在移除之前检查
`defer_block_free and last_sched_seq > processed_step_seq`。
若仍在途，退出本次分配内循环并递增请求索引，尝试下一个请求；
若可安全回收，则沿上游账目处理后调用 `finish_requests(FINISHED_ERROR)`，
复用 connector、encoder、finished IDs 和 block free 的生命周期。
每次继续分配内循环都会减少 running 的 victim 数；选中当前请求时退出。
共享 block 仍由上游引用计数管理，终止一个 victim 不保证必然释放独占容量。

终态快照归功能自己的 `_CapacityPreemption` 对象持有，按对应
`scheduler_output.finished_req_ids` 消费一次，包括零 token 步骤。
较早的 async 输出不能提前排空快照。`preempted_req_ids` 同时使 TPU worker
丢弃该请求旧的 async、Mamba 和 MTP 状态；原 Raiden load failure 包装链保留。

## 离线验证

在安装了 `.[test]` 依赖的环境，从本仓库根目录运行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/test_abort_on_preemption.py \
  tests/platforms/test_patch_registry.py \
  tests/test_raiden_pd_scheduler.py \
  tests/test_raiden_pd_decode_scheduler.py \
  tests/test_raiden_pd_async_scheduler.py \
  tests/distributed/test_tpu_connector.py \
  tests/runner/test_tpu_runner_preemption.py
```

Conda 环境应先激活，并确保 `$CONDA_PREFIX/lib` 在 `LD_LIBRARY_PATH` 中。
在 TPU 主机上保留 TorchTPU platform plugin，不要通过空的 `VLLM_PLUGINS`
禁用它后误选上游内置 TPU plugin。离线契约测试不代替真实 TPU 端到端验收。

## TPU 容量对照

在完成现有 P4D4 correctness、模型与依赖已就绪且八个 TPU 设备空闲后运行：

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B-FP8
for policy in recompute abort; do
  PREEMPTION_EXPECT="$policy" \
    bash scripts/vllm/integration/run_qwen35_p4d2_disagg_preemption.sh
done
```

需要宿主 logical-client 会话时，将该命令放入部署环境已有的会话 wrapper，
入口会保留并串联 `VLLM_DISAGG_ROLE_WRAPPER`。
每组使用新的 `RUN_DIR`，运行后清理本次启动的服务，保留日志与
`capacity_preemption_result.json`。脚本沿用既有 launcher 的文件名，实际拓扑
固定为 P4D4，使用 async scheduling、P MTP1 / D MTP3。
两组均在官方 AsyncScheduler 上开启 D 的 `kv_load_failure_policy="fail"`，
同时验证加载失败响应适配器与容量策略的组合；P 非流式调用保持发送完成开关关闭。
P 侧沿用 correctness 的 `PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=256`，
两侧启用 `TPU_RAIDEN_PREFIX_AWARE_LOAD=1`。
D 编译尺寸包含 `256,4096`，调度 token 上限为 4096，使关闭策略时的
完整 prompt 重算也落在已编译范围内。

压力场景使用 31-block 池及两条跨 decode block 边界的并发请求，
输入为边界前 4 tokens（2300），输出为完整的 64 tokens。
把输入放在边界附近，是为减小不同 MTP 接受率造成的请求进度差；
任意长的加载延迟仍可能错开压力，因此必须检查实际抢占计数。
该数字只适用于脚本验证的 Qwen3.5 hybrid layout，不是通用容量公式。
hybrid 布局要求启用 prefix caching；每条请求使用独立 `cache_salt`，
避免请求间共享 prefix 改变压力条件。同一请求在 P/D 使用相同 salt。

`recompute` 对照要求两条请求成功；`abort` 要求一条 HTTP 500、同 request ID
恰有一条 `model.KVCachePreempt` 调度日志，另一条成功。两组都要求真实抢占
计数与 MTP draft 计数增加、请求队列清空，以及成功响应与后续恢复请求的完整
token 序列一致。这是正确性验收，不以单次请求耗时给出性能结论。
