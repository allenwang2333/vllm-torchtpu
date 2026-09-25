# 使用上游 Raiden 连接线程回收

## TL;DR

旧 `raiden_thread_fix` 和 `raiden_raw_transport_fix` preload 已由上游
[b38e6f4](https://github.com/google/tpu-sync/commit/b38e6f4e1f5ff0529d2210a6eafbcee11a135e9a)
提供的连接线程生命周期实现替代，不再构建或加载。本分支沿用主线
`daeab1e7a` 的依赖组合：`tpu-sync-torch==0.0.1.dev20260917080754`、
`torch-tpu==0.1.1.dev20260912075415`、`libtpu==0.0.47`；不要恢复旧分支
`tpu-sync-torch==0.0.1.dev20260915225405` 的 pin。

2026-09-17 对旧 0915 wheel 的验证是历史证据，不能作为当前 0917 wheel、
vLLM 0.29 组合已经通过验证的依据。当前组合的运行结论应引用本次测试日志。

## 依赖和迁移

新旧分发共享 `tpu_sync` import 路径。升级前停止使用该环境的服务，卸载两个
分发后安装项目依赖，保留项目所需的 registry/keyring 配置：

```bash
python -m pip uninstall -y tpu-raiden-torch tpu-sync-torch
VLLM_TARGET_DEVICE=empty python -m pip install -e '.[test]'
python -m pip check
```

主线 `daeab1e7a` 记录：0917080754 wheel 保留 torch 2.13 native extension，
其检查过的 0917082735/PyPI wheel 当时只有 torch 2.11 extension。安装必须以
项目 pin 和匹配的 registry 为准，不能把同一天的构建当成可互换版本。
旧 TorchTPU 缺少 `TensorBufferHandle` 符号，不能单独替换 Raiden。

从启动命令移除旧 `with_preload.sh` 包装器，以及 `LD_PRELOAD` 中的
`libraiden_framed_pool.so`、`libraiden_listener_reap.so`、
`libraiden_raw_reap.so`。重启后检查 worker 的 `/proc/<pid>/maps`，确认
实际加载的 native 路径和旧库缺席；仅删除磁盘文件不会更新运行中的进程。

原 `+aios.1` wheel 回移的 RaidenId 修复 `212ff531` 也已进入上游。
bounded staging、请求抢占恢复和 PD completion 仍由 connector 及其独立适配器
负责，不属于线程回收替代的范围。构建镜像仍须遵循构建仓库的 PR805 补丁流程。

## 机制和验证边界

旧实现保留已退出线程的 joinable handle，直到 listener 析构才 join；活动线程数
恢复正常时，线程栈仍可能随历史请求数累积。上游 `DetachedThreadGroup` 在启动前
增加在途计数，线程 detach 后在 task 返回时减少计数；owner 先停止接收并 join
listener，再等待计数归零。共享 State 保证最后一次解锁期间 mutex 仍然存活。

该上游提交的 FramedServer、KVCacheListener 和 RawBufferTransport 使用此机制。
它不再依赖旧 FramedServer preload 的固定线程池或 socket 超时。Framed Stop
等待未完成 RPC，不能保证主动中断永久不发送完整帧的 peer；Raw 析构仍会
shutdown 活跃连接。

历史审查还指出 WeightSynchronizerListener 的 SHUTDOWN worker 读取 listening
FD 与析构 close 之间可能有 FD 复用竞态，尚未完成真实 native 复现。当前 PD KV
connector 不创建它，不能把 PD 验证外推为权重同步全部退出场景已经验证。

## 复现检查

先记录实际分发版本和 native 路径，再在无 TPU 的环境中运行相关 Python 回归：

```bash
python -m pip show vllm torch torch-tpu tpu-sync-torch libtpu
JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true OMP_NUM_THREADS=1 \
python -m pytest -q \
  tests/distributed/test_tpu_connector.py \
  tests/test_raiden_pd_scheduler.py \
  tests/test_raiden_pd_async_scheduler.py \
  tests/test_raiden_pd_decode_scheduler.py \
  tests/runner/test_tpu_runner_preemption.py \
  tests/offload/test_raiden_store_offload.py
```

上游源码测试可在具备其 Bazel/C++ 依赖的环境执行：

```bash
git clone https://github.com/google/tpu-sync.git
cd tpu-sync
git checkout b38e6f4e1f5ff0529d2210a6eafbcee11a135e9a
bazel test --config=oss --test_output=errors \
  //tpu_sync/common:detached_thread_group_test \
  //tpu_sync/kv_cache/reshard:framed_rpc_test
```

源码测试通过只证明被测源码；确认实际 wheel 仍需核对 native 符号或反汇编，并
运行真实连接生命周期探针。探针应重复覆盖串行和并发 RPC、空闲及半截连接析构，
记录 `/proc/self/maps`、线程数和 FD。线程数回落不足以证明线程栈没有泄漏。

历史 0915 wheel 验证记录见旧分支提交 `f4ad7a13d`：包含 Python 源码比对、
native `DetachedThreadGroup` 调用证据和 P4D4 短程验收。其测试计数、性能和
输出一致性结论均限于当时环境；本文件没有声明当前组合已经重复通过这些检查。
