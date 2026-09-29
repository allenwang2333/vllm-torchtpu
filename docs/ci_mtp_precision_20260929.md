# Qwen3.5 CI 配置与 MTP 精度检查

## TL;DR

两组 CI 从同一份 `scripts/vllm/integration/configs/qwen35_disagg.sh` 读取模型配置，
默认拓扑为 PCP4→TP2/DP2 和 PCP8→TP2/DP4，分别保留 8/16 devices 的资源规模。
配置以 R11 发布记录为基准；当前源码的 CUSTOM attention 已合并 longctx batched RPA，
无需恢复已删除的 `USE_BATCHED_RPA_LONGCTX`。

已有 dense Qwen3.5 MTP 精度测试现在检查每轮每个 draft 的全词表 logits，并增加 K3。
新增三个 nightly 自然语言长度用例，覆盖长 prefill、长 decode 及两者组合。
本次环境无 TPU：CPU 契约和启动参数检查不代表设备数值验收通过。

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

## 精度检查的边界

`tests/entrypoints/llm/mtp_precision.py` 使用 vLLM `worker_extension_cls` 和
`collective_rpc`，在模型启动之后安装采集器，避开 warmup。生产 `src` 不增加测试开关。
采集每个实际调用的真实请求行，包含输入、逻辑位置、step 和全词表 logits；
临时文件逐条重放并删除。编译的 logits head 返回 logits 和原有 argmax 结果。

参考路径复用已加载的 MTP 权重、projection 和 MLP，以独立 CPU causal SDPA
和按 request/逻辑位置维护的 KV 代替 paged attention。FP8 KV 使用实际静态 scale
独立量化、解量化；拒绝分支覆盖后截断历史。每轮首步以 target hidden 为输入，
后续步骤使用参考路径自己的 hidden carry，并以实际 draft token 对齐输入轨迹。
因此不会把 target model logits 当作 MTP logits，也不读取被测 paged KV 来构造参考。

检查要求所有 request-round 都有完整的 `0..K-1` 步骤，缺失记录、非有限值和
任一词表项超过 `abs(error) <= 0.5 + 0.03 * abs(reference)` 都失败。
日志 `MTP_DRAFT_PRECISION` 输出覆盖计数、最大绝对误差和最大容差比例。
上述阈值是尚待 TPU 校准的候选值，不能因失败自动放宽。

适用范围为 dense Qwen3.5、单 MTP 层、无权重量化、关闭 prefix caching、
DP=PP=PCP=DCP=1，支持 TP。
已有 MoE/DP 精度用例继续保留原有输出一致性检查，本次没有声称逐 logits oracle
覆盖这些配置。参考计算复用部分模型算子，因此不能独立验证权重、MLP 或 target hidden。
CPU 采集会同步设备，不能用于证明 async 时序正确，也不能用于性能结论。

## 长度用例

| nightly ID | 精确输入 tokens | 强制输出 tokens |
| --- | --- | --- |
| long-prefill | 65536 | 128 |
| long-decode | 512 | 1024 |
| long-prefill-long-decode | 65536 | 1024 |

每档使用两篇原创河流修复调查材料及不同写作视角，由地点、调查事件和访谈人物
组合成自然语言段落。句式可以复用，但完整段落不重复，不使用重复单 token 或数字串。
直接传 token IDs 保证输入长度，并保留末尾完整写作指令；`ignore_eos` 和实际返回
token 数量断言保证长输出不被提前结束掩盖。

三档均为 Qwen3.5-4B、TP2、K3、HND、FP8 KV、BF16 SSM、CUSTOM attention。
4096 token scheduler budget 强制 64K 输入经历多个 prefill chunk；prefix caching
关闭，避免第二篇材料命中前缀而掩盖 prefill 覆盖。分别运行无 MTP 与有 MTP，
检查最终 greedy 文本一致性，再检查所有已捕获 draft logits。

## 复现与验收

在安装了本分支、匹配版本 vLLM/torch-tpu 的环境中，从仓库根目录运行。
设备测试需可加载 `Qwen/Qwen3.5-4B` 权重和 tokenizer；长度用例至少需要两个
可见 TPU logical devices。两组 disagg CI 使用 `Qwen/Qwen3.5-35B-A3B-FP8`，
可通过 `MODEL_PATH` 指定本地模型目录。

```bash
# CPU 契约：精确长度、完整指令、拒绝回滚、causal mask、逐步覆盖及故障注入。
PYTHONPATH=src:tests python -m pytest tests/test_mtp_prompts.py tests/test_mtp_precision.py -q

# 已有短精度用例，包括 batch/sync/async 和 K2/K3。
PYTHONPATH=src:tests python -m pytest tests/entrypoints/llm/test_speculative_decoding.py \
  -k 'qwen35_mtp_correctness_greedy' -sv

# 三种长度形态；默认 pytest marker 将它们交给已有 speculative nightly lane。
PYTHONPATH=src:tests python -m pytest tests/entrypoints/llm/test_speculative_decoding.py \
  -k 'qwen35_mtp_natural_language_lengths' -sv

# 单机 8-device CI。
bash scripts/vllm/integration/run_qwen35_p4d2_disagg_correctness.sh

# 多机：在 decode/head 两台机器分别执行，设置双方可达地址。
ROLE=decode PREFILL_HOST=<prefill-ip> DECODE_HOST=<decode-ip> \
  bash scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh
ROLE=head PREFILL_HOST=<prefill-ip> DECODE_HOST=<decode-ip> \
  bash scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh
```

逐 logits 临时数据位于 `TMPDIR`（默认系统临时目录），长输出可能占用每 rank 数 GB；
需预留容量。参考 KV 随上下文增长，CPU attention 计算量也随上下文增长；
查询分块避免显式分配完整 64K×64K mask。单条长度测试超时为 4 小时，
实际耗时、设备内存及误差分布需要首次 nightly 运行确认。

CI 启动证据保存在各 `RUN_ROOT/run` 的 `launch_params.txt`、
`prefill_cmd.sh`、`decode_cmd.sh` 和 `logs/`。验收应核对这些实际启动参数，
要求既有正确性断言通过、`MTP_DRAFT_PRECISION` 中 failures 为空、
checked_request_steps 等于 request_rounds×K 且大于零。

## 本机验证记录

- 6 组 launcher 参数捕获：两种拓扑的 CI 默认、显式覆盖/空 MTP、standalone 默认。
- 4 组容器环境传递：本机 Docker 与 SSH 一致、空值/JSON/空格正确、DSv4 默认保留。
- PCP 容量边界及反例、shell 语法、shellcheck 和 YAML 检查通过。
- 本地 Qwen3.5 tokenizer 构造的两组 512/65536 输入严格匹配；65K 文档各含
  712 个完整且唯一段落，末尾指令完整。
- 初始分支测试文件收集得到 57 个用例。首次收集因系统 libstdc++ 缺少 `CXXABI_1.3.15`
  失败，使用 conda 环境自身 `lib` 的 `LD_LIBRARY_PATH` 后通过。
- 12 个 CPU 契约测试通过。pre-commit 的 Pyrefly 项在被忽略的 worktree 中不能
  展开 `src`，改用等价显式文件集合
  `pyrefly check --use-ignore-files=false 'src/**/*.py'` 得到 0 errors；其余 hooks
  使用 `SKIP=pyrefly pre-commit run --files <改动文件>` 验证。
- 未运行 TPU E2E；设备精度、容差校准、长上下文资源与耗时验收仍待执行。

## 基于 main 的 PR 验证

在 `250958522` 上重新执行：12 个 CPU 契约测试通过，61 个设备用例收集成功，
6 组 launcher 参数检查、4 组容器环境转发检查通过，全部适用的 pre-commit
hooks（包括 Pyrefly）通过。移植审查确认 `src` 零修改，main 新增测试和
服务进程清理逻辑保留；oracle 依赖的 draft 方法与初始审查版本一致。
未执行 TPU E2E，不能据此认定设备数值、长上下文资源或 acceptance 阈值已验收。
