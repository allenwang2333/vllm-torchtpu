# Speculative decoding KV 发布前的完成同步

## TL;DR

R9 的 Qwen3.5-35B-A3B-FP8、PCP4/DP4、MTP、async scheduling 场景中，
相同输入的接受率从冷首轮 54.84% 升到暖态 85.71%。冷首轮 D 端的有效历史
MTP KV 中存在 NaN；P 端 MTP forward 后同步 KV 和输出可以消除该异常。

本修复在启用 speculative decoding 的 Raiden producer 发布待发送请求之前，
对模型实际使用的 KV 张量执行 `synchronize_tensors(..., wait=True)`，
不按 MTP、Eagle3、DFlash 等方法名区分。它建立计算写入先于外部读取的关系，
保留 async scheduling，不清零缓存、不改 slot 或量化，不引入 #805。
此前仅启用 MTP 的修订已完成三次独立冷启动验证：每次三轮，共 621 条请求，
九轮均为 252/294（85.71%），有效 KV 采样及真实 draft head 无非有限值。
2026-09-27 扩展启用条件后，MTP 的同步路径保持一致；其他方法的新增覆盖
是发布边界的单测，不代表已完成这些方法的模型端到端验证。

## 问题与证据

基线为 `bd2d7891a0b13e1f246ea05a24c451a25b8aabd6`，R9 镜像为
`dashllm-tpu:q397-20260926-cpu-dashgen-r9`。此前同输入三轮分别为
204/372、252/294、252/294；只清 D 端不能改善首轮，只清 P 端变为 250/294，
而 P 端 forward 后完成同步的三轮均为 252/294。
原版首轮 189 次真实 draft head 非有限，同步组为 0。

Python forward 返回不保证 TPU 的延迟图已提交并完成写入。runner 虽然在
draft 后 finalize，但 Raiden 的 `wait_for_save()` 是空操作；实际源块登记走
`start_load_kv → process_send_load`，还可能发生在零 token 调度步。
Raiden 持有已注册的设备 buffer，源块登记后即可读取，不能依赖后续模型计算
或 sampled token 的 CPU 拷贝间接保证 MTP 写入完成。

以上证据支持计算/写入与传输之间的时序缺陷；尚未将延迟图 materialization
与已提交设备写入的完成等待分别归因。暖态恢复不能证明原版 KV 生命周期正确。
完整前期实验见 [R9 轮次效应记录](https://github.com/aios-tpu-infra/tpu-misc/blob/0d14c0ae68637c792c2e7e02c79d261767153dfd/pd_disagg/docs/r9-mtp-round-drift-20260926.md)。

## 修改范围

- 在 `TPURaidenConnectorWorker.process_send_load` 的 producer 分支中，
  对启用 speculative decoding 且 `reqs_to_send` 非空的调用同步一次，
  再执行已有登记流程。ngram 没有独立 draft KV，仍遵循相同的发布契约。
- 复用 `_flattened_kv_cache_tensors()`，覆盖 runner 使用的 attention、混合状态
  和 draft typed KV 张量；不把初始化时保存的 raw storage 当作唯一计算依赖。
- `wait=True` 同时提交相关延迟计算并等待完成；失败直接传播，源请求不会发布。
- 同时覆盖 Stage-3 与 legacy Raiden 登记入口；D 端、无 spec 和空发送调用不变。
- 不新增传输协议、后台线程或生命周期状态，不替换已登记的物理缓存。

这是阻塞的正确性边界，一次发送调用内的多个请求共用一次同步；它不是每个
draft token 的同步。可能等待其他相关 KV 写入，不保证没有吞吐代价。
异步 event 优化需另行验证事件覆盖、取消和源 block 保留，本提交不扩展该范围。

## 验证与复现

CPU 回归用例模拟延迟写入和外部读取，验证发布时能看到新值、写入失败不发布、
混合 KV 列表覆盖，以及非目标路径不增加同步。它验证调用契约，不能代替 TPU 数值验证。
测试按 MTP、Eagle3、DFlash、ngram 参数化，覆盖两种传输入口和写入失败。

```bash
PYTHONPATH=src python -m pytest -q tests/distributed/test_tpu_connector.py
PYTHONPATH=src python -m pytest -q \
  tests/distributed/test_raiden_pool_manifest.py \
  tests/distributed/test_raiden_layout_fingerprint.py \
  tests/offload/test_offload_connector_register.py
```

此前真实设备验证使用上述 R9 容器，只替换仅启用 MTP 修订的 `tpu_connector.py`，其余 runtime
文件保持不变；权重固定为 HF revision
`9d1823d2dee688a6b25e77009dc727688c44936e`，只读挂载。
P：TP1/PCP4、MTP K1、block 768；D：TP1/DP4、MTP K3、block 2304。
两端 async scheduling 开启，KV FP8，leases=1，CPU offloader 保留，V6D 关闭。

每次独立冷启动后重复固定的 69 条 HTTP 输入，比较空闲稳定边界的服务
accepted/drafted 差分，并用 CPU scheduler trace 对账。设备观察保持与原先
能够复现低接受率的 `trace-cold-kv08` 一致；不安装逐 forward 同步 wrapper。
检查首批长请求有效历史 KV、真实 draft head，以及逐请求输出与计数。

本机复现产物根为
`/mnt/data/yanzecheng_yzc/r9-pcp4dp4-mtp-20260926/mtp-fence-validate`；
容器内为 `/run-e2e/mtp-fence-validate`。`run.sh`、`drive.py`、`harness/`、
`hooks.py`、`cache_action.py` 和结果日志保存在该目录，未入库。
新环境需取得完整 E2E 脚本产物树及原记录中的主机/容器前提，使用空闲 8 TPU。
每次更换唯一结果目录并重启服务，禁止覆盖旧数据：

```bash
sudo docker exec r9-pcp4dp4-mtp \
  bash /run-e2e/mtp-fence-validate/run.sh NEW_UNIQUE_RUN
```

### MTP 修订的历史设备实测结果

2026-09-26，使用 `publish-fence-01/02/03` 三个独立容器启动 session，
每次重新加载模型并使用独立 role cache。三个目录各含三轮结果，均为 252/294；
接受率 count=9，avg/min/max/p90/p99 均为 85.7143%，总体 stddev=0 个百分点。

- 621 条 HTTP 全部成功，逐字输入、完整 choices 和逐请求 accepted/drafted
  与原版暖态 `trace-cold-04/control-2` 一致；CPU trace 与服务计数差分一致。
- 24 条长请求的有效历史 KV 采样无 NaN、无全零 token 行；9 轮共 5,940 次
  D 端真实 draft head 调用均有限。每个 session 的 8 个 worker 均确认 MTP KV
  包含在同步列表中，raw / MTP / manifest 缓存指针在采样间保持不变。
- 未安装逐 forward 同步 wrapper，实际部署文件 SHA256 为
  `ea33ac55cef043baf05395e1825a99d7a359fb6cbe4a88db5c29d77bb2f01b33`。
- 新增回归先在原版得到 4 项失败、4 项通过，再在修复后通过；最终 connector
  374 项、相关布局/offloader 54 项、MTP/connector 生命周期 31 项通过，
  共 459 项定向测试。pre-commit 和独立代码审查通过。

客户端暖态耗时（秒）的描述统计如下。旧对照为 `trace-cold-kv08` 后两轮，
修复为三个 session 各自的后两轮；每轮 69 条请求。总体标准差、线性插值分位数：

| 组 | count | avg | min | max | stddev | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧暖态 | 138 | 0.3187 | 0.2671 | 0.5110 | 0.0517 | 0.3803 | 0.4584 |
| 修复暖态 | 414 | 0.3374 | 0.2660 | 0.5064 | 0.0473 | 0.3816 | 0.4669 |

观测均值增加约 18.7 ms（5.9%）。这是带诊断采集的串行负载、非交错历史对照，
不能将全部差额严格归因于 fence，也不能作为吞吐基准或宣称零性能损失。
并发、抢占、多机和其他模型尚未覆盖；上述实验确认 MTP 修订消除了该部署的冷启动异常。

检查器、单测日志、代码哈希与汇总保存在
`/mnt/data/yanzecheng_yzc/mtp-kv-fence-20260926`：
`verify_e2e.py`、`aggregate.py`、`final-verification.json`、`code-provenance.json`。
每个 E2E 结果目录另存 `verification.json`。复核已有结果不需启动服务：

```bash
CHECKS=/mnt/data/yanzecheng_yzc/mtp-kv-fence-20260926
python3 "$CHECKS/verify_e2e.py" publish-fence-01
python3 "$CHECKS/verify_e2e.py" publish-fence-02
python3 "$CHECKS/verify_e2e.py" publish-fence-03
python3 "$CHECKS/aggregate.py"
```

验证后已恢复容器原始 runtime，四个 TPU 相关 wheel 的 RECORD 校验和
`pip check` 通过；容器停止、8 个 VFIO 设备释放，回环 IPv6 恢复原值。
R9 镜像未修改，收尾证据为 `$CHECKS/cleanup-final.json`。

### 2026-09-27 通用 spec 条件修订

本次保留 `wait=True`，仅移除 MTP 方法限制，不引入事件 API。
扩展后的回归在旧条件下得到 12 项失败、8 项通过，失败来自其他三种方法
没有等待写入。测试默认配置显式设置 `speculative_config=None`，避免
`MagicMock` 的隐式属性被误当成已启用 spec。

修订后以下定向回归共 471 项通过；本次未重跑设备 E2E，也未新增性能结论。
日志位于 `/mnt/data/yanzecheng_yzc/spec-kv-fence-amend-20260927` 的
`red.log`、`green.log`（测试配置修正前）和 `green-final.log`。

```bash
PYTHONPATH=src TPU_SKIP_MDS_QUERY=true python -m pytest -q \
  tests/distributed/test_tpu_connector.py \
  tests/distributed/test_raiden_pool_manifest.py \
  tests/distributed/test_raiden_layout_fingerprint.py \
  tests/offload/test_offload_connector_register.py \
  tests/runner/test_mtp_async_contracts.py \
  tests/runner/test_tpu_runner_kv_finalize.py
```
