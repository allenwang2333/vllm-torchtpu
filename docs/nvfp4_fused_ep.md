# NVFP4 fused EP MoE 的 W4A8 路径

## TL;DR

NVFP4 的 W4A8 只用于通过 admission 的 `fused_ep_moe_v2` 层，必须同时设置
`USE_MOE_FUSED_EP_KERNEL=1` 和 `MOE_FUSED_EP_ENABLE_W4A8=1`。普通和流水 GMM
始终保持 BF16 activation（W4A16），显式设置 `MOE_REQUANTIZE_BLOCK_SIZE`
只会改变它们的权重量化分块。

上游 #912（`a47d91cba`）新增了独立 W4A8 开关；从只需 fused EP 单开关的旧临时
分支迁移时，需要补上 `MOE_FUSED_EP_ENABLE_W4A8=1`。该提交还将 admission
移到有损 requantization 之前，避免未获准使用 fused EP 的层被自动改写权重。

## 启用与分块契约

在模型加载前设置环境变量，例如：

```bash
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_REQUANTIZE_BLOCK_SIZE=512
# 然后执行部署原有的 vllm serve 命令，并为 MoE 开启 --enable-expert-parallel。
```

若要使用内核推导的最小候选分块，省略显式 block：

```bash
unset MOE_REQUANTIZE_BLOCK_SIZE
```

候选最小块来自内核的 `U32_SUBLANE_TILE * PACK4`，在上述版本中为 `8 * 8 = 64`。
这不是任意硬件和未来内核版本的通用常量。显式 block 优先；fused EP 要求 block
为正整数、是 packed-weight row tile 的倍数，并整除运行时 hidden 与 intermediate
维度。intermediate 可在准备权重时向 block 对齐，hidden 必须已对齐。
权重尺度使用 `[E, K / block_size, 1, N]` 布局。

例如运行时 hidden=4096、intermediate=1024 时，bridge 的分块检查允许
64、128、256、512、1024。通过形状检查不代表已在特定 TPU 上完成编译、精度和性能验证；
最终 admission 还检查 EP、expert placement、路由、activation、mesh、token threshold
以及内存等契约。

## 加载与执行

NVFP4 checkpoint 的 block scale 和全局 scale 先合并。启用 W4A8 后，加载逻辑用
meta tensors 描述候选运行布局，在 requantization 前完成 admission：

- 通过时，按候选 block 准备 K-major FP4 权重和 scales，并在安装到 layer 前验证
  实际布局。候选已通过但准备结果不一致属于实现不变量破坏，会触发 assertion。
- 拒绝时，继续使用原有 GMM 权重策略：没有显式 block 时保留 block-16；显式 block
  仍按原有 requantization 规则处理。GMM activation 保持 BF16。

fused EP kernel 动态将输入和中间 activation 量化为 FP8，在内部完成 routing、EP
交换和 combine，返回本 rank 的 token shard。FP4 权重在内存中压缩存储，计算前在
kernel 内展开为 FP8；这不是原生 FP4 矩阵指令。只有实际获准的层接管通信。

该功能可与 `additional_config.tpu_online_fp8` 共存：后者选择 BF16 Linear 层在线
转成 FP8；这里处理的 routed expert 权重仍为 FP4。

## 复现与验证

以 `c02d6d3b4` 或包含上述 #912 的后续版本为基线，使用支持 native
`torch.float4_e2m1fn_x2` 的 TorchTPU 环境、NVFP4 checkpoint 和已验证的 EP
部署命令。记录模型版本、TPU 型号、并行配置、两个开关和显式 block，保存启动日志：

```bash
# 在原有启动命令末尾追加重定向，例如：
# vllm serve ... > nvfp4-fused-ep.log 2>&1
rg 'Fused EP MoE armed|NVFP4 fused EP W4A8 enabled|not engaged' nvfp4-fused-ep.log
```

block-512 示例的成功日志应包含 `format=fp4`、`rhs_qb=512` 和
`NVFP4 fused EP W4A8 enabled: block-512`。环境开关为 1 不能单独证明 kernel
已经服务某层；应同时检查拒绝日志。

CPU admission 与加载契约可用现有测试复现：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest \
  tests/layers/adapter/test_fused_moe_ep.py -k nvfp4 -q
```

TPU kernel 数值测试位于 `tests/kernels/fused_moe/test_fused_ep_moe_v2.py`。
模型质量应与相同 checkpoint 的 W4A16 基线对比；性能需重复运行，并记录
avg、min、max、count、stddev、p90、p99。此文档和 CPU admission 检查不替代
目标设备上的模型精度或性能验证。
