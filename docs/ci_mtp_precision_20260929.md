# Qwen3.5 CI 配置与 MTP 精度检查

## TL;DR

两组 CI 从同一份 `scripts/vllm/integration/configs/qwen35_disagg.sh` 读取模型配置，
默认拓扑为 PCP4→TP2/DP2 和 PCP8→TP2/DP4，分别保留 8/16 devices 的资源规模。
配置以 R11 发布记录为基准；当前源码的 CUSTOM attention 已合并 longctx batched RPA，
无需恢复已删除的 `USE_BATCHED_RPA_LONGCTX`。

本次仅保留部署配置对齐。PR 新增的 Qwen3.5-4B dense 逐 draft logits、K3 和
长上下文测试已撤回，main 原有测试保持不变。PCP8 的新增用例、logits 采集和
验证暂不实施。

下一步只在本机以 `Qwen/Qwen3.5-35B-A3B-FP8` 运行 PCP4→TP2/DP2，记录该轮
真实 logits，再冻结为候选基准。当前会话无法访问 Docker/TPU 和 GitHub，尚未生成
基准，也不能宣称已完成实机精度验收。

## 配置与证据

初始实现基线：`1c25edbb1`；PR 移植到 `origin/main` 的 `250958522`。
移植仅包含本次 CI 和测试差异，保留 main 新增的 MTP logprobs/DSpark 用例，
并保留单机 CI 原有 `MIN_MTP_ACCEPTANCE_RATE=0.70`；新 TP2/DP2 geometry
仍需实机重新测量此阈值。直接读取的 R9 镜像 ID 为 `dc4bd2ba1e61`，配置路径为
`/opt/dashllm-venv/lib/python3.12/site-packages/dashgen/configs/qwen3.7-plus/0522-thinking-re-nvfp4-inputscale/l20g{,_prefill,_decode}.config`。
R11 依据工作区 `tpu-misc/dashllm/docs/docker-native-r11-release-20260928.md`，
本次没有直接读取 R11 镜像；发布记录将 decode token budget 和最大编译档位提高到 768。

| 项目 | Prefill | Decode |
| --- | --- | --- |
| MTP | K1 | K3 |
| TP | 1 | 2 |
| PCP / DP | PCP4 或 PCP8 | DP2 或 DP4 |
| block size | 768 | 2304 |
| max batched tokens | 32768 | 768 |
| max sequences | 128 | 192 |
| memory utilization | 0.75 | 0.9 |
| compile sizes | 512,1024,2048,4096；PCP4 加 8192 | 32,64,128,256,384,512,768 |

共同设置包括 HND、FP8 KV、BF16 SSM state、CUSTOM attention、async scheduling、
prefix caching、align mamba cache 和 unified block pool。MoE 使用 fused EP、
sharded plan、128 requantize block、32768 onehot threshold 和 gather reduce v3。
Prefill 的 Raiden destination shards 为 2。

PCP4 的 8192 档位是本拓扑的必要差异：单请求 32768 tokens、interleave=256 时，
`pcp_local_token_counts` 得到 `[8192] * 4`，仅有 4096 bucket 会无法选择编译档位；
PCP8 则得到 `[4096] * 8`。这不是所有 PCP 输入的通用均分公式。

入口只保留拓扑、端口、connector 和测试场景差异。Pipeline step key 和脚本名保留，
因此旧文件名中的 `p8d8` 不再描述默认 decode DP。角色环境变量可覆盖公共默认值，
显式 `PREFILL_SPECULATIVE_CONFIG=''` / `DECODE_SPECULATIVE_CONFIG=''` 可关闭 MTP。
独立使用 launcher 时保持原有默认值；其他多机消费者（如 DSv4）保留原外层默认值。

## 本机基准的执行范围

使用本 PR 的 PCP4→TP2/DP2 配置，模型为 `Qwen3.5-35B-A3B-FP8`，P K1 / D K3。
测试输入使用自然语言。冻结该轮每个 draft 的全词表 logits，并记录原文、精确
input token IDs、request/逻辑位置/draft step、接受拒绝轨迹、模型 revision、镜像
或运行库版本、代码 commit、实际启动参数和数据 SHA256。

采集必须来自实际 TPU 模型运行；历史接受率、最终输出文本、CPU 参考重算或
合成数据均不能代替该轮 logits。只有真实运行完成且数据完整性检查通过后，
才生成候选基准 manifest；未运行或中断的采集不能标记为基准完成。
冻结一次结果可用于后续回归，但数值容差和基准质量仍需单独验证。

PCP8 因设备不足，本次不增加采集与 logits 检查，不套用 PCP4 的基准。
撤回的 4B 测试结果不能作为 35B MoE 的验证证据。

## 配置复现

在具备本机 Docker/TPU 访问的执行环境中，从本分支仓库根目录运行：

```bash
MODEL_PATH=/mnt/data/yanzecheng_yzc/models/Qwen3.5-35B-A3B-FP8 \
  PREFILL_PCP=4 DECODE_TP=2 DECODE_DP=2 \
  bash scripts/vllm/integration/run_qwen35_p4d2_disagg_correctness.sh
```

上述命令运行已有 CI 正确性测试，本身不会导出完整 draft logits；采集入口需要在
可运行环境中接入并验证后才能用于冻结基准。先核对 `launch_params.txt`、
`prefill_cmd.sh`、`decode_cmd.sh` 和两端日志中的实际拓扑，再记录采集结果。

## 验证状态

配置实现此前通过 6 组 launcher 参数检查和 4 组 Docker/SSH 环境转发检查，
并通过 shellcheck、YAML 和 pre-commit 检查；它们不代表设备模型测试通过。

当前会话的设备视图没有暴露 TPU；Docker socket 连接返回 `operation not permitted`，
GitHub API 不可达。历史本机 PCP4→TP2/DP2 实验记录证明宿主机曾可运行该拓扑，
不能将当前沙箱限制表述成宿主机没有 TPU。本次没有生成真实 logits 基准。
