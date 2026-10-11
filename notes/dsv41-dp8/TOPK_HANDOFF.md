# DeepSeek V4.1 DP8 prefill：性能基线与 indexer/top-k handoff

更新：2026-10-11。目标是在已经修复 host overlap、idle CED 和 DP wave 到达错位的基础上，优化 indexer score → top-k 路径。先复现基线，再改 kernel；不把调度变化混入 kernel 收益。本文所称基线是性能基线，长上下文数值重复性仍有未解决问题。

## 1. 工作目录、版本与资源

- 工作树：`/Users/jiongxuan/work_vllm/dsv41-dp8-prefill`，分支 `develop/dsv41-dp8-prefill`。不要修改原仓库其他工作。
- Epic 基础：`ddf9b3766365ecaf00d4ac678990c55b4d4ca5cd`；vLLM：`e37e51dd246cf421c554e7d6f53e178e3fd29085`。
- runtime：torch `2.13.0+cpu`、torch-tpu `0.1.1.dev20260924085621`、jax `0.10.2`、libtpu `0.0.47`。
- 适配代码、测试、benchmark 脚本和本 handoff 随分支 `develop/dsv41-dp8-prefill` 一起提交。复现时使用该分支最新提交，不要只 checkout Epic 基础 SHA。若后续还有工作区修改，另行保存 diff 和必要新增文件，通过 Falcon cp 上传并校验 SHA。
- 最新实验 `exp-lgy33v281e` / `art-algs8mmspm` 已于 2026-10-11 03:03:27 UTC SUCCEEDED，资源已结束，没有可继续使用的 holding session。
- 后续 4 physical-chip 任务优先 `gke-tpu-train-us-central1-2-prod`，提交前核查实时容量；保留已经获得的资源。遵循 `/Users/jiongxuan/.codex/skills/falcon-workflow/SKILL.md`，先 `source /Users/jiongxuan/work_vllm/env.sh`，通过 Falcon CLI 操作。不要自行建立定时任务、push 或开 PR。
- 历史 setup/运行入口：[experiment.json](cache-aware/experiment.json)、[command.sh](cache-aware/command.sh)。它们包含历史 overlay，不能当作当前工作树的完整替代。新实验使用有界会话，保存证据后释放资源。

## 2. 冻结的运行和数据配置

V4.1 Flash v2，TP1/DP8/EP8，4 physical chips / 8 TPU devices；FP8 KV/indexer、async、CED、SP/local-Q、Engram CPU offload、fused FP4 EP MoE、DSpark K5、prefix cache 全开。KV 4GiB/rank，max-seqs 1/rank，max-model-len 82944。

LongBench v2 独立前缀，65536 hit + 16384 new extend，输出 1，C8。使用 `X-data-parallel-rank` 固定 cache owner。每波新增 131072 tokens，严格命中总量 524288；preemption 必须为 0。数据 SHA256：`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`。脚本按文档 ID 排序，从 Code Repository Understanding 中取足够长且前缀独立的文档；不是完整 LongBench 任务精度评测。

推荐 **chunk2048/rank，全局 token budget16384，compile buckets [128,256,2048]，PREFILL_ADMISSION_MS=100**。这是有界准入等待：所有 rank ready 即退出，并非固定睡眠 100ms。它保留现有 cache owner，在新 wave 前对齐队列；没有实现全局 cache 目录、KV 迁移或通用负载路由器。

在 pinned Linux runtime、模型挂载和正确编译缓存配置好以后：

```bash
CHUNK=2048 PREFILL_ADMISSION_MS=100 \
  bash scripts/dsv41_bench/serve_cached_prefill.sh
# 另一进程，服务 ready 后运行：
python scripts/dsv41_bench/bench_cached_prefill.py \
  --dataset /tmp/tpu_logs/longbench.json --out /tmp/prefill-c8 \
  --concurrency 8 --rounds 5
```

完整 serve 参数以 [启动脚本](../../scripts/dsv41_bench/serve_cached_prefill.sh) 为准（仓库相对位置为 `scripts/dsv41_bench/serve_cached_prefill.sh`）。编译缓存必须按源码/runtime/拓扑/buckets 分开，候选 kernel 使用新缓存身份。启动和编译不计入测量。

`PREFILL_ADMISSION_MS=off` 是原策略或 C1 延迟优先配置；0 仅 rendezvous/补收队列。C1 实测开启100ms后 TTFT 增加约88.8ms，因此本配置针对 C8 成批流量，不是通用生产默认值。

## 3. 当前最佳性能与度量

排除 warmup、prime、profile，以下为每波指标的中位数。extend 是该波服务端各请求 prefill 时间均值；吞吐为 131072 / 客户端整波完成时间；TTFT 为每波请求均值。不能用 131072 / extend 冒充这里的客户端吞吐。

|配置|轮数|全局 token budget|extend 秒|新增 tokens/s|TTFT 秒|
|---|---:|---:|---:|---:|---:|
|原策略 2k，同服务对照|3|16384|1.303858|69425|1.477213|
|修复后 2k|5|16384|**1.201646**|**73428**|1.345283|
|修复后 4k|5|32768|1.220084|71309|1.349497|
|修复后 8k|5|65536|1.354489|65332|1.507397|

2k 相对原策略 extend -7.84%、新增吞吐 +5.77%。五波 extend 1.19922–1.20363s；20ms 到达抖动测试约1.20098s。2k 是本轮三档中最佳，不是所有并发/流量的普遍最优。

最终2k profile：各rank 8×2048 + 1×256，首forward最大错位6.126ms，device有8个大encoder图；四芯片八core XLA Modules union约1039.7–1041.1ms，span1110.8–1117.7ms。256尾块有实际 replay/尾部工作，不能当作冗余直接删去。4k也已对齐（4×4096+1×256），仍较慢。

生产 admission 单测18项通过，真实2rank CPU Gloo burst/timeout/pause验证通过；O16 uneven C8、idle C1有draft计数增长，覆盖活跃DSpark。4k/8k没有OOM。上述均不能替代完整数值精度验收。

## 4. 数值重复性：已知事实与未知项

历史同服务隔离实验 `exp-usduz1nwhn` / `art-e6moty8stp` 比较 serial-1 / overlap / fresh / serial-2。固定相同8份文档、temperature=0、O1；每组prime/extend各3波，前两波不请求logprobs，第三波请求top5。fresh使用每chunk独立device staging并clone CPU packed输入，仅用于诊断。

前两波16个prime/extend输入中，重复输出变化数分别为 **6 / 2 / 6 / 6**。因此不是单纯bitwise不一致，greedy输出token确实发生变化；原串行和独立staging也存在，不能归因于新overlap的buffer复用。部分输出分布差异较大：extend rank2出现换行/`eps`切换，不同状态下分别观察到换行logprob -2.4375、`eps` logprob 0。这不是同一个token的直接logprob差值。

**尚未证明任务精度下降，也没有排除正确性问题。** 未有可信reference或任务得分对照，不能把它概括成无害舍入误差，也不能认定top-k是根因。最终admission测试没有解决此问题。后续独立跟踪长上下文最后token/logits对齐、CED/DSpark状态和各kernel reference。

证据：[数值说明](overlap/README.md)、[客户端采样设置](overlap/numeric/client.py)、[192个响应及输入证据](overlap/numeric/numeric-results.json)、[汇总](overlap/numeric/summary.json)。优化验收不能通过放宽文本比较来掩盖该问题；既有端到端不稳定也不能作为新kernel不做独立reference验证的理由。

## 5. Top-k 的切入点与证据边界

历史热点来自 `exp-i42cf8kt8e` 的 `idle-ced/2048/profile`，分析 `an-75dhu868qb` / `an-0j5x366k5v`。它发生在最终admission修复前，以下占比不是最新基线的重新归因：

|family|ms/core|历史self占比|
|---|---:|---:|
|fused EP FP4 MoE|318.29|27.14%|
|SparseMLA|147.40|12.57%|
|FP8 matmul|72.51|6.18%|
|indexer thd,cd→thc score matmul|35.41|3.02%|
|SparseCore top-k async-done|18.78|1.60%|

18.78ms是等待事件，不是完整top-k耗时，也不是可直接去掉的收益。历史score矩阵为2048×41472 FP32，约324MiB，观察到dynamic_update_slice、layout copy/reshape再交给top-k。41472是该图的实际宽度，不应等同于原始context长度；先在最终profile核对每层、实际row_lengths、topk和layout。

源码入口（仓库相对路径）：

- `src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py`：`paged_index_scores_jax`（约1680行）、`_sparsecore_top_k`（1930）、`paged_indexer_jax`（2103）；检查 `select_indices_jax` 及 full/candidate 分支。
- `src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py`：`sparsecore_topk`、`_pick_partition`、`_sc_topk_direct`。
- 不要误改同仓库v1/custom_ops indexer后就认为优化了v2实测路径；用HLO/profile确认实际调用。

已有score循环按row_lengths限制trips；SparseCore已支持分区局部候选和二阶段归并。**新工作应先减少producer→consumer的layout转换/复制，再考虑把局部候选选择前移到score生产阶段，避免完整score矩阵落HBM**，而不是重复实现已有分层选择。测量新增launch、candidate写回/merge、VMEM/HBM代价，不能预先承诺收益。

注意一个待核实的契约差异：v2 `_sparsecore_top_k` docstring要求边界同分按较小index；底层SparseCore文档允许边界同分采用不同选择。先通过调用端测试与reference明确实际要求，不要靠修改注释或放宽断言消除差异。

## 6. 建议交付顺序和验收

1. 固定当前源码/runtime复现2k C8至少5波，保留输入SHA、命中、输出、metrics和内存；若偏离基线，先定位环境和调度，勿混入kernel改动。
2. 对最终基线单独profile，分清score计算、布局转换、SparseCore执行/等待和关键路径。最终profile位于artifact根 `/gcs/experiments/exp-lgy33v281e/artifacts/art-algs8mmspm/cache-aware/service2/profile`，通过Falcon分析，不绕过Falcon访问provider。已有host/device分析 `an-0udfkrtezf` / `an-0vs8bric3o`。
3. 先做layout/copy最小改动并独立microbenchmark；再决定是否值得做score生产与候选选择融合。除目标indexer/top-k外保持调度、MoE、SparseMLA、Engram、CED/DSpark和量化配置不变。
4. 对同一保存输入比较score reference、合法索引、top-k集合及边界tie规则。覆盖FP8 scale、负值/±0、重复分数、-inf、row_lengths=0/小于k/非tile整倍数、padding、causal边界、无效/重复candidate、尾块与idle rank。NaN在底层当前契约中不支持，明确处理边界，不宣称已支持。允许的无序输出应比较集合，不误要求排序；同分选择按明确后的调用端契约验收。
5. 参考现有测试 `tests/models/vllm/test_dsv41_v2_attn_kernels_indexer.py`、`tests/models/vllm/test_dsv41_sparsecore_topk_layout_rope.py`、`tests/kernels/test_dsv4_sparsecore_topk.py`。CPU/mock通过不等于真实TPU正确；必须在pinned TPU runtime做reference和实际路径验证。
6. 对照/候选使用相同模型、输入和配置，分别充分warmup，至少5波无profile C8，另跑profile确认关键路径。严格逐请求65536命中、全局524288、preemption0；O>1 uneven C8与idle C1验证活跃draft及collective。报告extend、整波新增吞吐、TTFT、全局budget、CPU/HBM峰值及波动；不混入prime/编译/profile耗时。
7. 保存补丁、源码SHA、命令、runtime、原始结果与失败证据；性能提升和数值验收分别报告。未解决的既有数值问题继续列明，不宣称生产精度已经验证。

## 7. 证据索引与可直接转交的任务

- [最终性能报告](cache-aware/RESULTS.md)、[性能JSON](cache-aware/performance-summary.json)。
- `cache-aware/evidence/cache-aware/`：原始结果、smoke、日志、内存；`wait100`是最终2k数据，`chunk-4096`/`chunk-8192`是修复后重扫。
- `cache-aware/final-source/`：最终admission/启动脚本快照，不是整个DP适配补丁。
- `cache-aware/robust-forwards.json`、`robust-timeline.json`：最终2k的host/device证据；历史热点见 `cache-aware/README.md`、`hotspots.json`、`kernel-families.json`。

可交给下一agent：

> 阅读本handoff及RESULTS，在该分支完整适配基础上，先复现TP1/DP8/EP8、全功能、LongBench独立前缀65536 hit+16384 extend、O1/C8、chunk2048、100ms有界准入基线。通过最终profile确认v2 score→SparseCore top-k关键路径，优先减少布局转换/score写回，再评估生产阶段局部候选融合。冻结其他kernel和调度，独立reference验证索引/分数/tie/mask，保留既有长上下文数值不稳定限制，最后同条件复测性能并交付补丁、命令和原始证据。资源使用Falcon有界实验，4-chip优先2-prod；不建定时任务、不擅自push/开PR。

## 远端保存范围

本分支提交代码、测试、benchmark脚本、关键报告和汇总JSON。大型原始日志、profile和历史诊断中间文件留在原工作目录与文中Falcon artifact；部分历史报告的相对证据链接仅在原工作目录可用。历史报告中的“未提交”描述是测试当时状态。
