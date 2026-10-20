# 显式 Mamba cache checkpoint

## TL;DR

请求可通过 `SamplingParams.extra_args.cache_control_params.token_positions`
提示一个优先保存的 Mamba checkpoint。插件选取最大有效位置，按
`scheduler_block_size` 向下取整，再覆盖 vLLM 返回的 `shared_prefix_boundary`。
这样复用现有 Mamba chunk 切分、稀疏缓存登记和 boundary-state handoff，
不改变真实缓存命中数，不修改已安装的 vLLM 源码。

## 请求契约

```python
from vllm import SamplingParams

params = SamplingParams(
    max_tokens=1,
    extra_args={"cache_control_params": {"token_positions": [111]}},
)
```

`token_positions` 必须是 list。有效元素为 `type(position) is int` 且
`0 < position <= num_prompt_tokens` 的位置；bool、字符串、浮点数、越界值和
不合法容器作为无效优化提示忽略，不新增请求错误。位置使用最终 engine prompt 的
token 坐标。使用 `x-ds-max-user-theoretical-hit-num` 的服务层需先将 header
映射到上述参数；插件不直接读取 HTTP header，也不保留各位置的来源标记。

最大有效位置按以下公式得到唯一边界：

```text
boundary = position // scheduler_block_size * scheduler_block_size
```

`scheduler_block_size` 是 scheduler 使用的逻辑调度粒度，与
`Scheduler.block_size`、`KVCacheManager.coordinator.scheduler_block_size` 一致。
它不等于各 rank 的物理页大小或 `hash_block_size`；PCP 布局也使用同一逻辑粒度。
例如 `scheduler_block_size=32` 时，111 变为 96，32 仍为 32；31 变为 0，
因此保留引擎原生边界。

有非零显式边界时，它优先于引擎原生产出的 shared-prefix junction，不取并集。
没有有效位置、取整后为 0、prefix caching 关闭或请求跳过 prefix-cache read 时，
沿用原生返回结果。raw 参数保持不变。

## 实现与边界

`core/explicit_cache_checkpoint.py` 在 `_ENGINE` 生命周期包装
`KVCacheManager.get_computed_blocks` 和 `get_computed_blocks_for_connector`。
两个方法均先执行原函数，只替换返回 tuple 的第三项，保留 blocks、实际 computed
tokens 和 connector 的 `hit_diverged`。这同时覆盖普通 admission、connector 的
per-group 查询和无远端命中时的 reconciliation fallback。

请求仍由现有 scheduler 将结果写入 `Request.shared_prefix_boundary`，
原有 split、retention 和 Mamba boundary handoff 继续消费同一字段。
不另建请求状态，不改写 Request、scheduler 或 retention 算法。
实现面向当前 pinned vLLM 0.29；升级时需核对两个返回值契约及消费链。

提示不代表已有 cache hit，也不保证永久免于 eviction。若本地或远端已计算的前缀
越过该位置，不回退重算来补 checkpoint。Mamba checkpoint 的物化仍受现有
cache mode、调度和 finalized-token 规则约束；外部 offload 是否接收边界取决于
connector 自身的 store 范围和粒度，不承诺所有 connector。

## 验证

CPU 契约测试覆盖真实 KV manager 的命中与边界返回、connector 两种查询路径、
无效提示回退、逻辑调度粒度、幂等安装，以及 `retention=0` 时 checkpoint 的
实际 hash 登记、Mamba handoff 与后续请求复用：

```bash
python -m pytest -q \
  tests/platforms/test_explicit_cache_checkpoint.py \
  tests/platforms/test_mamba_scheduler_block_size_patch.py \
  tests/platforms/test_hybrid_pcp_block_size_patch.py \
  tests/platforms/test_patch_registry.py
```

这些 metadata 契约测试不执行模型，不代替真实 TPU P/D 的输出正确性与 KV 传输验收，
也不提供性能结论。
