# DP cache-aware matching 与 kernel 热点（2026-10-11）

最终结果见 [RESULTS.md](RESULTS.md)：100ms上限，2k/rank，C8基线extend1.202s、73.4k新增tokens/s。以下保留探索过程，40ms不是最终配置。

以下前半部分是实现前的代码/已有profile分析；后半部分记录随后实施和TPU验证。最初分析未改运行逻辑、未提交TPU实验。Falcon离线分析：an-55u0xivlxl（修复版2k各rank forward）、an-75dhu868qb（HLO热点）、an-0j5x366k5v（精细families）；源实验exp-i42cf8kt8e，idle-ced/2048/profile。

## 确认事实与结论边界

benchmark已经用X-data-parallel-rank固定缓存归属，8个独立前缀均命中65536；不是cache miss或路由错误。client线程已有barrier，但不保证HTTP/API处理、engine enqueue、schedule时间一致。
修复版rank3首forward为t=0，其余rank在117.9–119.5ms开始；rank3持续领先一块，最终其kv_len81760的尾部被全局bucket放大成2048，其他rank最后单独跑256。每个请求9次forward；device有9次大encoder图（稳定块约126–127ms），不能把9次全部当作多余。
起点kv_len65408表明命中65536后仍有128 replay；实际kv步进2044，8步后81760，剩160到81920，所以尾块必要。理想对齐可能把9个大bucket+尾块变为8个大bucket+尾块；这是待验证推断，不是已获得约127ms加速。host forward起点差不能等同于request arrival skew，也不能据此设置120ms等待。

## 建议设计：缓存归属 + 有界准入/成本匹配

1. 路由：以真实prefix block hash/cache事件维护rank归属，带epoch/eviction失效。初版用已有显式rank/session affinity保证归属，不能把token内容hash当作真实cache存在证明。查询估计后由owner本地再次校验命中/可用KV；已开始请求不跨rank搬移KV。对无副本prefix，强亲和会和queue等待形成权衡；默认保留owner，在明确成本策略下才考虑冷算或复制。多个API进程需共享目录，避免各自陈旧信息。
2. 本地候选成本：命中后待算tokens（包含必要replay）、当前KV长度、encoder bucket、decoder replay bucket、请求年龄、KV容量。仅按原prompt长度或cache-hit率匹配不够。对当前C8/MAX_SEQS1每rank只有一个候选，主要收益来自准入对齐；多请求队列才有选择匹配空间。
3. 全局协调：在scheduler提交本步计划前交换CPU候选摘要/ready mask；按兼容的phase和bucket选出计划，减少global MAX padding和encoder/decoder阶段错位。runner现有_dp_coordinated_step只做MAX归并，不能在这里等待后指望已生成的scheduler_output自动纳入新请求。需在engine/scheduler层重新poll并计划。
4. 等待有界：只在新prefill wave且其他rank已知有in-flight/ready工作时短暂coalesce；不要求八rank必须满，不阻塞持续decode。统一wave/step编号、超时和取消处理；到期按现有idle collective路径推进。不能在某rank独立sleep或跳collective。避免长请求/冷请求饥饿。
5. 先补观测：API接收、engine enqueue、cache lookup完成、scheduler admission、global step、rank有效tokens、encoder/decoder bucket。再扫0/1/2/5ms等短window（候选值非现有trace证明），测C1/C8、随机到达抖动、冷热混合、不同extend、O16活跃DSpark；同时记录等待/TTFT p99、命中、padding、全局step数、吞吐。不能仅以同步batch client结果决定生产默认值。

本地vllm checkout HEAD与实验pinned e37e51dd不同；本轮未把当前checkout的load-balancer/cadence实现当作该实验精确事实。实验实际路由由benchmark header和命中结果验证，TPU MAX协调来自当前已测runner。

## Kernel 热点

以下为修复版2k profile，rank0 XPlane内两TPU core累计HLO self time除2，单位ms/core；不是8core求和、不是端到端时间，也不是可直接相加的可削减墙钟。family总self约1172.68ms/core；与device union约1170ms接近，但分类包含异步等待，不能把等待再算成独立通信计算。

|family|ms/core|self占比|优化方向|
|---|---:|---:|---|
|fused_ep_moe_v2_fp4w...g48_c128_nb2|318.29|27.14%|优先测tile capacity、expert负载/尾块效率、FP4权重预取和dequant；核对VMEM约束，不能直接把nb2改成nb3|
|SparseMLA-p_512-bz_16-gcz_128|147.40|12.57%|以真实64k–80k KV、2k Q/FP8形状做tile sweep，检查gather布局、KV读取与流水重叠|
|quantized_matmul_jax dot_general|72.51|6.18%|FP8量化/scale reduction/transpose成本，生产者布局与矩阵乘输入融合；此数不含全部其他attention matmul|
|all-gather async-done|55.07|4.70%|暴露等待；先定位对应MoE plan/all-gather与producer，不将55ms视为链路带宽耗时|
|delayed_post_pre_flat_jax|47.80|4.08%|多层重复的mHC post/pre融合、FP32中间张量与布局，需数值验证|
|indexer thd,cd->thc|35.41|3.02%|配套score矩阵写回/转换与top-k一起优化|
|SparseCore top-k async-done|18.78|1.60%|等待不等于完整top-k执行时间；与producer、layout一并看|

Index score形状2048x41472 FP32（约324MiB），trace有dynamic_update_slice、layout copy/reshape再送top-k。代码attention_kernels.py已按row_lengths限制循环trips，不能再次声称简单裁剪有效长度是新优化。可先避免score生成到top-k消费之间的布局转换，再考虑分tile候选top-k归并以减少完整score矩阵落HBM；必须保持top-k/tie与mask语义，且实测SparseCore启动/合并成本。

MoE实际日志capacity128、nb2、prefetch distance1，约55.3MiB/62.7MiB预算；更深预取需减少buffer占用或改tile，不宜硬改常量。当前已知数值重复性问题仍未解决，任何kernel改写需要独立reference/误差与collective测试。

建议顺序：先做可观测的有界prefill准入对照；kernel先MoE，再SparseMLA，indexer layout作为更局部的候选。全部为后续实验建议，未宣称已验证新增加速。

## 2026-10-11 准入修复验证进行中

用户授权先修复prefill cache-aware准入、建立kernel不变的性能基线，暂不处理indexer。实验exp-lgy33v281e / art-algs8mmspm，2-prod，4physical chips/8devices，4h setup后有界holding，完成标志/tmp/cache-aware.done；服务另以/tmp/admission-service.done结束。

最小实现src/vllm_torchtpu/core/dp_prefill_admission.py：在DPEngineCoreProc新wave(step0)且cache+EP时，在SchedulerOutput创建前CPU rendezvous并再次drain本地request队列；默认不启用，additional_config.dp_prefill_admission_ms可配置额外等待，0仅rendezvous；MAX/SUM状态形成共同退出决定。已有cache归属不变，不迁移KV，不修改模型/kernel/bucket。持续decode/运行中batch及pause发出立即退出票。代码通过patch_registry engine生命周期安装。

18项pinned Linux单测通过，覆盖zero-window repoll、partial batch超时、pause/decode、hook gating。实验专用instrument.py/admission_diagnostics.py允许同一空闲服务读原子mode文件切换None/0/2/5ms；该诊断代码不属于生产实现。对照顺序baseline→0→2→5→baseline-repeat，每档warmup+3 measured waves，开启档各跑O16 uneven C8/idle C1。所有kernel与2k/global16k配置不变。cmd-1p6pn8v1d8启动服务。当前无性能结论。

准备了独立验证脚本validate-best.py：按3轮中位extend选候选，随后C1和20ms到达抖动与原策略对照，最后单独3轮C8确认+1轮profile。gloo-regression.py将在服务停止后运行真实2rank CPU collectives，避免与服务争用runtime初始化。

服务实际固定上游源码已通过Falcon cp读取/tmp/dp8-pinned-core.py，确认hook在DPEngineCoreProc._process_engine_step前且upstream首wave会处理START_DP_WAVE；没有使用本地不同HEAD推断实验实现。

## 准入窗口选择与使用

本次修复限定为 **保持已有缓存归属的跨 DP prefill 准入对齐**：固定 `X-data-parallel-rank` 的请求先在 owner 上有界聚合，再产生 SchedulerOutput。不是新建全局缓存目录，也不迁移 KV。通用 engine 默认关闭；本目录 benchmark 启动脚本暂设 C8 候选 40ms（稳定性复测进行中），可用 `PREFILL_ADMISSION_MS=off` 跑原策略对照，或设非负毫秒值；0 表示仅一次 rendezvous/re-poll。没有修改 indexer 或其他 kernel。

```bash
# source pinned runtime and set the source/runtime/topology-specific cache roots first
CHUNK=2048 PREFILL_ADMISSION_MS=40 bash scripts/dsv41_bench/serve_cached_prefill.sh
python scripts/dsv41_bench/bench_cached_prefill.py \
  --dataset /tmp/tpu_logs/longbench.json --out /tmp/cached-prefill-c8 \
  --concurrency 8 --rounds 3
```

配置保持 TP1/DP8/EP8、全功能、LongBench 独立前缀、65536 hit + 16384 extend、O1、C8。每 rank chunk=2048；全局调度 token budget=16384；每波实际新增 tokens=131072。命中计数必须为每请求65536，全局524288；不得把 prime、warmup 或 profiled wave计入正常吞吐。

窗口扫描 None/0/2/5/10/20/30/40/50ms。0–5ms无稳定收益，10/20ms仅部分波次成功对齐；30/50ms 在本轮成批流量扫描中均约1.20s；40ms有一轮1.332s，另外两轮约1.198s。按扫描的中位extend选择40ms，独立3轮未profile确认extend分别1.200149、1.199908、1.200575s。C1控制两轮TTFT约1.469/1.467s，40ms约1.503/1.498s，extend仍约1.395s；这是等待代价，低并发延迟敏感流量应关闭。人为20ms到达抖动的两轮测量中，40ms有一轮未对齐（1.3224s），另一轮1.1980s；不能宣称任意到达模式下都消除错位。窗口约束的是额外轮询预算，collective及请求处理本身可能使总准入时间超过该值。

性能仍不代表之前长上下文数值重复性问题已解决。本次只验证调度、命中及 collective/smoke，不将其提升为模型数值正确性验收。

40ms profile（an-42pq3rdsl8 / an-h603hs2gee）与独立无profile测量不同：rank3首forward比rank0晚113ms，仍有额外全局step；不能用此profile声称9大图降至8大图。其余rank首forward差约0–4ms。正在同一allocation重启服务验证100ms上限，并重复无profile/20ms抖动/profile；新上限不是固定等待，ready全满即退出。

最终18项单测9.49s通过，真实2rank CPU Gloo覆盖burst/partial timeout/pause consensus通过。Gloo测试最初的默认`dist.barrier()`触发torch-tpu默认accelerator初始化冲突，已改为显式CPU tensor的all_reduce作为测试间同步，原始失败日志保留。40ms O16 smoke uneven C8 draft计数增量39，idle C1增量4，均完成；不以O1推断DSpark已工作。memory.events OOM/oom_kill均0，cgroup memory.peak949557506048bytes（包括page cache，不是进程RSS）。
