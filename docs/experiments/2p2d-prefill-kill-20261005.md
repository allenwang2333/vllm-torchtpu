# Qwen3.5 35B 的 2P2D TP2 Prefill worker kill 测试

## TL;DR

在 `gcp-avtdhasm-qqg7` 上使用 Qwen3.5-35B-A3B-FP8、两个 TP2 Prefill 服务和两个独立 TP2 Decode 服务，完成一次真实 worker `SIGKILL` 和 360 秒观察。

- **复现了健康 P→D 新流量约 300 秒被阻塞。** P0 的 TP0 worker 被 kill 后，健康 P1→D0/D1 的首个请求分别等待 **299.951 秒、299.898 秒后返回 HTTP 500**。D 日志同时记录对已关闭的 P0 controller 连接失败：`Timeout (300s)`。
- **未复现整个 Decode 300 秒完全不处理请求。** 两个 D 在故障窗口内分别完成 **174/174** 个直接模型请求，最大相邻响应完成间隔分别为 **2.671 秒、2.608 秒**。阻塞发生在 Stage3 后台提交线程，主模型线程仍可工作。
- 300 秒阻塞解除后，P1→D0/D1 各完成 **27** 个新请求，均报告 `cached_tokens=1023`。引用已死亡 P0 的两条请求则在客户端等待约 **350.108 秒**后超时，未正常收敛。
- 测试使用宿主机 **venv**，`vllm-torchtpu=0.1.0+nancheck`、`vllm=0.29.0+empty`；不是文档中的 R13 镜像。当前 Python 已包含 Qwen TP 的异步 Stage3 提交路径，native 的慢连接失败仍影响串行提交队列。不能据此判定文档列出的修复 commit 已验证通过。

本轮已结束，实验服务已停止。结论仅覆盖这一版本和故障时序；未测试 DashLLM 路由重试、多机部署或持续高并发负载。

## 测试依据与范围

参考[《abort测试内容》](https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCeQzjerrJGXn6lpz)中“kill Prefill worker 后检查 Decode 是否持续约 300 秒不接流量”的现象，以及仓库已有的 [R13 2P2D 记录](https://github.com/aios-tpu-infra/tpu-misc/blob/1cbde180d6ef29a669fb63af3421104b89f67f00/dashllm/docs/r13-2p2d-abort-test-20260929.md)和[单机 PCP4 到 DP2TP2 记录](https://github.com/aios-tpu-infra/tpu-misc/blob/a9933d52e859b877417141ff978330fb87e8addf/dashllm/docs/local-pcp4-dp2tp2-prefill-abort-20260930.md)。

用户最初要求 PCP2→TP2。本机现有源码在 PCP2/TP1 Producer 启动时拒绝该拓扑，P0/P1 于 10:28:43 启动失败，未进入 kill 实验，见[首次启动证据包](evidence/2p2d-prefill-kill-20261005/run01-evidence.tar.gz)。用户随后明确改为 TP2→TP2，因此本记录的故障结果均来自 TP2→TP2。

独立 TP2 Decode 在 **HND 专用路径**下受到支持；常规路径要求 DP2/4/8/16，不能把常规路径的校验条件解释成所有布局下的限制。本轮日志确认 Producer 为 `pcp1tp2_prefill`，Consumer 为 `tp2dcp1_decode`。

## 实际环境与配置

通过 `yanzecheng.yzc` 连接宿主机；没有使用容器。Python 为 `/ssd/1/wister.zwz/nan-guard/venv/bin/python`，完整模型位于 `/ssd/1/wister.zwz/nan-guard/models/Qwen3.5-35B-A3B-FP8`。

| 组件 | 实际版本 |
| --- | --- |
| vllm | `0.29.0+empty` |
| vllm-torchtpu | `0.1.0+nancheck` |
| torch | `2.13.0` |
| torch-tpu | `0.1.1+release.2026.9.22.20261001190350` |
| tpu-sync-torch | `0.0.1.dev20261001082953` |
| libtpu | `0.0.47` |

四个独立逻辑客户端分配如下。启动前，8 个 rank 的 eager/compiled collective 与设备映射预检全部通过；本轮仍是单机共享物理 TPU session 的实验，并不等价于四台独立主机。

| 服务 | 物理 TPU rank | 并行配置 | API 端口 | Controller 端口 |
| --- | --- | --- | --- | --- |
| P0 | 0、1 | TP2/PCP1/DP1 | 18400 | 27000 |
| P1 | 6、7 | TP2/PCP1/DP1 | 18410 | 27400 |
| D0 | 2、3 | TP2/PCP1/DCP1/DP1 | 19400 | 28000 |
| D1 | 4、5 | TP2/PCP1/DCP1/DP1 | 19410 | 28400 |

均使用 `TPURaidenConnector`、FP8 KV、HND 布局、bfloat16 SSM、APC、async scheduling、EP、block size 2304、128 个缓存块、最大模型长度 8192、最大序列数 8。Prefill 最大 batch token 为 2048、编译尺寸 `[2048]`、MTP 1；Decode 为 128、编译尺寸 `[32,64,128]`、MTP 3。`TPU_RAIDEN_TRANSFER_PARALLELISM=2`、`TPU_RAIDEN_DST_SHARDS=2`、`TPU_P2P_WAIT_PULL_TIMEOUT=120`。完整命令在证据包的四个 `*-argv.json` 中。

OOT 原始目录为 `/ssd/1/wister.zwz/nan-guard/repo/pd_disagg/src/vllm-torchtpu`；运行前复制到本轮 `run03/runtime`，没有修改共享目录。该源码快照没有可用 Git 历史，不能将包版本映射到确定 commit。

为确认布局分支，只在隔离副本 `_raiden_seq_on_lane_layout()` 增加一条日志，打印缓存 getter 与实际环境变量，两者均为 `HND`。这条修改不改变返回值、传输或故障处理。源码差异与散列保存在 `layout-diagnostic.json`，其余 Python 文件散列在 `source-manifest.json`。

| 文件 | SHA256 |
| --- | --- |
| 原始 `tpu_connector.py` | `c545d5d4d132eae4d8151c20060ca17f5a604b5a88e190c9b1446d4cbf17d6b4` |
| 本轮含诊断日志的 `tpu_connector.py` | `1b6c94726ca9325809c75ce7dbbb1a2ffdbaf2f8dea6e83469ab006b1fc2b5d5` |
| 实际 native `_tpu_raiden_torch_2_13_0.so` | `67f9763570dfdf6aefd55b502de33f687d77838ffbf385565c41ed8dd3b84d96` |
| logical-client shim | `03a133078b89f6da496e8f4acd6a8a3cdf8d2338904356460a7bf11410075875` |

## 故障注入与结果

1. 四条 P→D 路径各重复三次，共 12 次。提示词包含唯一请求前缀和合成中文文本，共 1024 token；P 返回真实 KV 参数后，将原始参数交给 D，生成 16 token。12 次全部 HTTP 200，且 `cached_tokens=1023`。
2. 从 P0 为 D0、D1 分别准备一份尚未消费的 KV，避免把一个 Producer 请求复用于两个 Consumer。
3. 2026-10-05 **10:43:31.176（北京时间）**，使用 pidfd 向本轮 P0 TP0 worker（PID 1429249）发送 `SIGKILL`。注入前交叉验证日志 rank、当前进程树、UID 和进程 start ticks，避免误杀其他进程。
4. 确认 P0 controller 已关闭，额外等待两秒，再向两个 D 提交对应的原始 KV 参数。同时分别启动健康 P1→D 流量与直接 D 请求，持续观察 360 秒。直接请求不携带远端 KV，生成 8 token；各流量线程串行发请求，每次完成后间隔两秒。
5. 每约五秒保存健康状态与进程树，并周期性采集 D worker 全线程栈。实际流量成功以模型响应判断，不以 `/health` 单独判断。

| 故障期间的请求 | D0 | D1 |
| --- | --- | --- |
| 健康 P1→D 首个请求 | 299.951 秒，HTTP 500 | 299.898 秒，HTTP 500 |
| 后续健康 P1→D 请求 | 27/27 成功，均使用 1023 个缓存 token | 27/27 成功，均使用 1023 个缓存 token |
| 直接 D 请求 | 174/174 成功 | 174/174 成功 |
| 直接 D 的最大响应完成间隔 | 2.671 秒 | 2.608 秒 |
| 引用已死亡 P0 的请求 | 350.108 秒，客户端 `TimeoutError` | 350.108 秒，客户端 `TimeoutError` |

以下为重复请求的响应耗时，单位毫秒。正常基线只统计 D 段，未包含之前的 P 请求；直接 D 请求与 PD 请求输入不同，不能用它们比较模型性能。故障注入只执行一次；两个 D 的结果来自同一个故障事件，不能视为两次独立故障重复。

| 流量 | count | avg | min | max | stddev | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 正常 P0→D0 | 3 | 114.996 | 98.073 | 145.238 | 21.435 | 136.526 | 144.367 |
| 正常 P0→D1 | 3 | 105.132 | 88.835 | 114.601 | 11.574 | 114.073 | 114.548 |
| 正常 P1→D0 | 3 | 99.118 | 89.069 | 115.029 | 11.380 | 110.674 | 114.594 |
| 正常 P1→D1 | 3 | 105.027 | 90.212 | 113.343 | 10.502 | 112.979 | 113.306 |
| 故障期间直接 D0 | 174 | 76.837 | 71.033 | 670.661 | 45.241 | 75.524 | 87.872 |
| 故障期间直接 D1 | 174 | 76.092 | 71.033 | 607.452 | 40.472 | 74.417 | 83.990 |

健康 P1→D 流量包含一次约 300 秒失败与恢复后的成功，完整混合分布保存在 `summary.json`，不应将其平均值当成稳定性能。故障请求每个 D 仅一个样本，其 350 秒耗时是客户端观测上限。

## 证据与根因判断

**已确认事实：** 在 kill 后约 140.580 秒的 `sample-024.json` 中，两个 D 的 TP0 均有以下后台线程栈：

```text
raiden-stage3-submit-r0-0
  coordinate_transfer
  start_transfer
  _execute_stage3_submit
  _stage3_submit_worker_loop
```

同一窗口内直接 D 请求持续成功。源码 `_ensure_stage3_submit_workers()` 对 Qwen 的 sharded consumer 只创建一个提交线程；`_stage3_submit_worker_loop()` 从队列取出任务后同步执行 `_execute_stage3_submit()`，完成后才取下一项。由此可以解释坏 P0 的 RPC 如何阻止同一 D 上健康 P1 请求的 KV 提交，同时不阻塞模型主线程。

**已确认事实：** 两个 D 在约 300 秒后记录：

```text
Stage-3 controller transfer outcome uncertain; waiting for native terminal state
Timeout (300s) failed to connect to robust endpoint 127.0.0.1:27000:
connect failed to 127.0.0.1:27000: Connection refused
```

紧接着，首次健康 P1 请求报 `Request block registration was cancelled`，返回 HTTP 500；后续新 P1 请求恢复正常。这支持“坏源连接的 300 秒慢失败堵住单个 Stage3 提交队列”的根因判断，置信度高。首次健康请求 registration 被取消的具体原因，本轮没有单独验证，不能直接归因于某个 TTL。

**另一个已确认问题：** 原始故障请求的控制面超时被归为 `outcome uncertain`，继续等待 native terminal state，到客户端 350 秒期限仍未返回。已确认请求未正常结束；为何未获得 terminal state、其取消和资源释放是否完整，还需独立验证。

本轮没有出现 R13 记录中模型主线程卡在 Stage3 并引发整个 Decode 约 300 秒无模型响应的现象。当前 Python 的异步提交隔离了模型计算，但单提交队列仍使 PD 服务流量受坏 Prefill 影响。没有修改 native timeout，也没有测试文档列出的 native 修复 commit，不能声称该修复失效。

## 复现方式与输出

环境前提：本机八张空闲 TPU、上述完整模型、venv、OOT 源码、logical-client helper 与已编译 shim。新环境应提供对应依赖，并调整 harness 中的 `BASE`、`REPO`、`MODEL`、`SHIM` 常量；需要重新做八个 rank 的隔离预检，不能仅替换端口后跳过预检。

本轮 remote 根目录是 `/ssd/1/yanzecheng_yzc/pcp2-tp2-2p2d-kill-20261005`；路径保留首次 PCP2 尝试的命名。下载并解压[原始复现脚本归档](evidence/2p2d-prefill-kill-20261005/harness.tar.gz)即可得到 `harness/run_experiment.py` 和 `harness/inject_prefill.py`。脚本按实测版本原样归档，未为提交改变执行逻辑；在开发机复制到 `$ROOT/harness` 后，用新的输出目录执行：

```bash
PY=/ssd/1/wister.zwz/nan-guard/venv/bin/python
ROOT=/ssd/1/yanzecheng_yzc/pcp2-tp2-2p2d-kill-20261005
RUN=$ROOT/reproduce-tp2

# 在 vllm-torchtpu 仓库根目录运行，解压原始实测脚本。
tar -xzf docs/experiments/evidence/2p2d-prefill-kill-20261005/harness.tar.gz -C "$ROOT"

nohup "$PY" "$ROOT/harness/run_experiment.py" supervise --root "$RUN" \
  > "$ROOT/reproduce-startup.log" 2>&1 < /dev/null &

# 等待 RUN/READY；出现 RUN/FAILED 则先检查日志，不得进行故障注入。
nohup env LD_LIBRARY_PATH=/ssd/1/wister.zwz/nan-guard/miniforge/lib \
  TPU_SKIP_MDS_QUERY=true "$PY" "$ROOT/harness/inject_prefill.py" \
  --root "$RUN" --label kill-tp0-closed --rank 0 --duration 360 --wait-closed \
  > "$ROOT/reproduce-fault.log" 2>&1 < /dev/null &
```

入口要求输出目录尚不存在、各角色端口空闲；注入脚本要求四个服务 READY、四路径重复基线通过，才执行 kill。复现不需要布局诊断日志；本轮该日志的完整单行差异保存在证据包中，所有业务条件未变。

判定方法：检查注入目录 `DONE` 存在且 `FAILED` 不存在；核对 `injection.json`、`controller-closed.json`、真实响应 JSON、`summary.json`、线程栈与 D 日志。对照健康 P1→D 请求的延迟与直接 D 请求的连续完成情况，区分 KV 队列阻塞和整个模型执行停住。请求体与 metadata 为合成输入，可按 UUID 关联日志，未保存账号密码。

- [本地核心证据包](evidence/2p2d-prefill-kill-20261005/run03-core-evidence.tar.gz)：全部请求响应、metadata、events、采样线程栈、日志、配置、harness、connector 源码及验证 JSON；SHA256 为 `f3e8d92f5cb09c6931fa61fd376c36b90f658b3874066d09320f9c84317b048b`。
- 远端完整源码证据包：`$ROOT/run03-evidence.tar.gz`，SHA256 为 `3ac3add8257b7b7af8cdb64ba0a80dbb41596c2a735e300685d887916ae4b962`。完整 OOT 快照还保留在 `$ROOT/run03/runtime`；模型和 native 库采用上述现有路径与散列定位。
- `baseline-kv-validation.json` 核对 12 次正常请求；`recovered-kv-validation.json` 核对 54 次恢复后的 PD 请求；`direct-response-gaps.json` 保存连续响应间隔。
- 故障窗口结束时 P1、D0、D1 健康且模型请求成功。随后按本轮记录的 PID、start ticks 和 UID 停止实验进程树；`cleanup.json` 记录清理范围，四个 API 端口均已关闭，本账号 `fuser` 未见 TPU 占用。

## 分支与材料来源

本分支从 `alibaba/release-0.0.1` 的 `78c6e0d215c5ea8551c0e619e7bf587101cdb442` 创建，仅用于保存实验记录与原始证据。该基线不是本次实测源码的 commit；实测源码无 Git 历史，其身份仍以文中的 SHA256 与证据包为准。

文档与复现材料复制自 `aios-tpu-infra/tpu-misc` 的 `dashllm/docs/local-tp2-tp2-2p2d-prefill-kill-20261005.md` 和相邻 `evidence/local-pcp2-tp2-2p2d-prefill-kill-20261005` 目录。迁移只调整文档链接并增加脚本归档入口，两个原始证据包保持原始字节，展示用 JSON 可补全文件末尾换行但不改变数据。当前分支没有实现故障修复。

归档校验已核对原始证据包 SHA256、脚本语法、脚本归档与实际执行版本的字节一致性、提取 JSON 与原始 JSON 的数据一致性，以及文档本地链接。本次新增材料的 pre-commit 检查通过。全量检查中的 `pyrefly` 在干净基线 `78c6e0d215c5ea8551c0e619e7bf587101cdb442` 上也报相同的 12 处源码错误；本次没有改变 `src/` 或类型检查基线，新增材料检查单独跳过这一已有失败项。迁移阶段没有重跑 TPU 实验，测试结果来自前述实测记录。
