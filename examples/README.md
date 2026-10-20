# 算子开发示例

本目录包含 pyasc 算子开发端到端示例，覆盖从基础算子到进阶场景的完整开发流程。所有示例均采用 PyTorch 输入输出 Tensor，可直接运行验证。

## 示例总览

| 算子样例 | 功能描述 |
|---------|---------|
| [01_add](./01_add) | 实现手动插入同步流水的Add算子。 |
| [02_add_framework](./02_add_framework) | 实现通过Ascend C框架插入流水同步的Add算子。 |
| [03_matmul_mix](./03_matmul_mix) | 实现MIX模式（包含矩阵计算和矢量计算）下的Matmul算子，计算公式为：C = A * B。 |
| [04_matmul_cube_only](./04_matmul_cube_only) | 实现纯Cube模式（只有矩阵计算）的Matmul算子，计算公式为：C = A * B。 |
| [05_matmul_leakyrelu](./05_matmul_leakyrelu) | 实现MatmulLeakyRelu算子，计算公式为：C = A * B + Bias， C = C > 0 ? C : C * 0.001。 |
| [06_gelu](./06_gelu) | 实现GELU激活函数算子，计算公式为：GELU(x) = x * Φ(x)，采用tanh近似。 |
| [07_swiglu](./07_swiglu) | 实现 SwiGLU 激活函数算子，TPipe/TQue 框架双路 VECIN 输入，五步基础向量算子组合完成计算。 |
| [08_rmsnorm](./08_rmsnorm) | 实现 RMSNorm 归一化算子，TPipe/TQue 框架管理搬运，调用 `asc.adv.rmsnorm` 高阶 API 完成行归一化计算。 |
| [09_linear](./09_linear) | 实现 Linear 全连接算子，使用 Matmul 高阶 API 完成 float16 矩阵乘。 |
| [10_fused_infer_attention](./10_fused_infer_attention) | 实现 Llama 推理中的因果自注意力。 |

## 推荐学习顺序

建议按以下顺序循序渐进地学习：

1. **01_add** — 入门首选。理解手动同步流水、数据搬运（Global↔Local）、多核切分（tiling）的基本概念。
2. **02_add_framework** — 学习 Ascend C 框架提供的 TPipe/TQue/TBuf 机制，体会自动流水同步如何简化开发。
3. **03_matmul_mix** — 进入矩阵计算领域，掌握 Matmul 高阶 API 的 MIX 模式用法。
4. **04_matmul_cube_only** — 深入理解纯 Cube 模式下的矩阵分块与 Workspace 配置。
5. **05_matmul_leakyrelu** — 学习融合算子开发，将矩阵乘与激活函数在单次 Kernel 调用中完成，避免中间数据回传。
6. **06_gelu** — 学习复杂激活函数的实现，掌握九步算子组合、TBuf 复用和 adv API（tanh）的使用。
7. **07_swiglu** — 学习 SwiGLU 激活函数：TPipe/TQue 框架同步、双路 VECIN、TBuf 复用、多步组合计算与 PIPE_V 同步。
8. **08_rmsnorm** — 学习 RMSNorm 归一化：`asc.adv.rmsnorm` 高阶 API 调用、RmsNormTiling 分块参数配置、整块/尾块处理。
9. **09_linear** — 学习 Linear 全连接：Matmul 高阶 API、Cube tiling、多核矩阵乘和工作空间配置。
10. **10_fused_infer_attention** — 综合运用 Matmul、在线 Softmax、多核任务分配和 AIC/AIV 流水，完成 Llama 因果自注意力。

## 运行示例

### 环境准备

请先完成[环境准备](../docs/quick_start.md#envready)，确保已安装 PyTorch 和 torch_npu 插件。

### 通用运行命令

```bash
cd pyasc/examples/<示例目录>
python3 <示例文件>.py -r [RUN_MODE] -v [SOC_VERSION]
```

- `RUN_MODE`：`Model`（仿真器，无需 NPU）或 `NPU`（上板运行）
- `SOC_VERSION`：昇腾 AI 处理器型号，如 `Ascend910B1`

### 示例

```bash
# 仿真器模式运行 Add 示例
python3 examples/01_add/add.py -r Model -v Ascend910B1

# NPU 上板运行 Matmul MIX 示例
python3 examples/03_matmul_mix/matmul_mix.py -r NPU -v Ascend910B1
```
