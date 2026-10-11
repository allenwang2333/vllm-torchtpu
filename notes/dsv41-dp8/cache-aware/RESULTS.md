# Prefill cache-affine admission 基线（2026-10-11）

已实现并验证：保持已有 prefix cache owner，跨 DP rank 在新 wave 产生 SchedulerOutput 前做 CPU rendezvous、补收队列并有界聚合。通用 engine 默认关闭，benchmark 启动脚本默认 `PREFILL_ADMISSION_MS=100`；全部 rank ready 会立即退出，不是固定 sleep 100ms。后续 chunk/持续 decode 不再等待，active batch/pause 有共同退出路径。无 indexer/kernel 改动。

## 最终同服务对照

2-prod，4 physical chips / 8 TPU devices；TP1/DP8/EP8，V4.1 Flash v2，全目标功能开启。LongBench v2 独立前缀，显式 `X-data-parallel-rank`，65536 hit + 16384 extend，O1，C8；每 rank chunk2048，全局 token budget16384，每波新增131072 tokens。KV4GiB/rank，max-seqs1。

数据为排除 warmup、prime 和 profile 的每波指标中位数；extend是服务端各请求 prefill 耗时均值的波次中位数；吞吐为131072/客户端整波墙钟，包含响应完成开销；TTFT是客户端每波请求均值的中位数。

|配置|测量轮数|extend|新增吞吐|TTFT参考|
|---|---:|---:|---:|---:|
|原策略，同服务对照|3|1.30386s|69,425 tokens/s|1.47721s|
|100ms上限|5|1.20165s|73,428 tokens/s|1.34528s|
|100ms上限，20ms到达抖动|3|1.20098s|73,253 tokens/s|1.32675s|

对照到候选：extend降低7.84%，新增吞吐提升5.77%，TTFT降低约8.93%。五轮extend范围1.19922–1.20363s；吞吐范围72,192–74,579 tokens/s。修复后完成了4k/8k各五轮重扫，2k仍是本轮最佳；见下表。有限轮数不代表所有流量模式的长期保证。

所有测量严格验证命中，每请求65536，全局524288；preemption=0。20ms抖动范围1.20008–1.20554s。100ms是尾部容忍上限，40ms曾有1.33s退化，且40ms profile漏了rank3，故未选40ms作为最终基线。

## Profile 证据

100ms profile 的8个rank均为8次2048 bucket + 1次256尾块，必要尾块保留。首forward最大差6.126ms（旧profile约118–120ms）。四芯片八core的XLA Modules活跃union约1039.7–1041.1ms，span约1110.8–1117.7ms；均为8个大encoder图，消除了旧错位多出的大bucket step。这里的device union不等于端到端TTFT；profile轮extend1.22546s，不混入性能表。

Falcon实验 `exp-lgy33v281e` / `art-algs8mmspm`。
- 最终host forward分析 `an-0udfkrtezf`：`robust-forwards.json`
- 最终device分析 `an-0vs8bric3o`：`robust-timeline.json`
- 原40ms诊断分析 `an-42pq3rdsl8` / `an-h603hs2gee` 保留，不作为修复成功证据。

## 使用与代价

在固定runtime和对应编译缓存环境中：

```bash
CHUNK=2048 PREFILL_ADMISSION_MS=100 bash scripts/dsv41_bench/serve_cached_prefill.sh
python scripts/dsv41_bench/bench_cached_prefill.py \
  --dataset /tmp/tpu_logs/longbench.json --out /tmp/prefill-c8 \
  --concurrency 8 --rounds 5
```

设 `PREFILL_ADMISSION_MS=off` 跑原策略/C1延迟优先；0仅rendezvous。C1对照两轮：extend1.35627→1.35350s基本不变，TTFT1.42944→1.51827s，增加88.8ms；吞吐10,592→10,079 tokens/s。该基线针对C8成批流量，并非所有流量默认开启的策略。等待预算不包括collective/handler自身开销，因此不是硬实时端到端100ms承诺。

## 验证与版本

最终生产源码18项pinned Linux单测通过；真实2rank CPU Gloo burst、partial timeout、pause共识通过。100ms与40ms均完成O16不均衡C8/idle C1 smoke（用draft指标验证活跃DSpark）。单测不替代TPU性能实测；本轮未处理既有长上下文数值重复性问题，这仍是性能/调度基线。

Epic ddf9b3766365ecaf00d4ac678990c55b4d4ca5cd；vLLM e37e51dd246cf421c554e7d6f53e178e3fd29085；torch2.13.0+cpu，torch-tpu0.1.1.dev20260924085621，jax0.10.2，libtpu0.0.47。生产源码快照和SHA在`final-source/`；所有结果汇总在`performance-summary.json`；命令、原始结果、smoke、源码与runtime在`evidence/`及Falcon artifact。

同服务A/B使用实验专用mode文件和diagnostic hook切换窗口；生产文件不含该hook。第二次服务使用与最终生产一致的显式CPU tensor版本；首轮版本默认CPU，最终单测/Gloo及第二次服务已验证显式CPU版本。变更未提交、未push、未开PR。

## 修复后的 chunk 重扫

4k使用相同生产准入参数100ms，global token budget32768；五轮extend中位1.220084s（1.214470–1.228735），新增吞吐71,309/s，TTFT1.349497s。host profile各rank均4×4096+1×256，首forward最大差6.433ms；device union1066.1–1067.9ms、span1188.1–1195.1ms，4个大encoder图，排除了“仍因错位所以4k慢”的解释。分析an-sshxo185ab / an-izl7lm7dho，文件chunk4096-forwards.json / chunk4096-timeline.json。O16 uneven C8、idle C1通过，服务exit0。

|每rank chunk|全局token budget|测量轮数|extend中位|新增吞吐中位|TTFT参考|
|---|---:|---:|---:|---:|---:|
|2048|16384|5|1.201646s|73428/s|1.345283s|
|4096|32768|5|1.220084s|71309/s|1.349497s|
|8192|65536|5|1.354489s|65332/s|1.507397s|

8k五轮extend范围1.346229–1.366427s，新增吞吐范围58299–66468/s；客户端整波完成时间存在波动。所有测量严格命中524288、零preemption，O16 uneven C8/idle C1完成。三档保持相同模型/runtime/功能开关和100ms准入；4k/8k通过生产启动脚本直接传additional_config，没有实验动态mode hook。启动/编译时间排除于测量。8k无额外profile，未推断其具体kernel热点。

最终启动脚本继续默认CHUNK=2048、PREFILL_ADMISSION_MS=100。4k/8k不会成为本轮默认配置。Kernel目录151个Python文件哈希与实验快照一致。

4k/8k均exit0，O16 smoke各自draft增量39（uneven C8）/4（idle C1）。主机MemAvailable采样最低分别218.36/158.64GiB，OOM/oom_kill均0。证据归档已下载并校验SHA256：final2=3f6b677d11d3ce01d7761ca447a72140ee3ee5eae6c6862a9d1995b3774881b5，chunks=4d3cd42952514bc35acb6bd7a28a9762afe521d255cfa1b2aa4acf2548614d93。

实验已于2026-10-11 03:03:27 UTC进入SUCCEEDED，holding完成，资源使用结束；状态保存为final-status.json。无定时任务、未提交/推送/开PR。
