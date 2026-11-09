# 09 — Linear

## 概述

本样例实现Linear前向计算：$C[M,N] = A[M,K] \times B[N,K]^T$。

## 运行环境要求

| 类别 | 要求 |
|------|------|
| AI处理器 | Ascend 910B / 910C |
| CANN版本 | 社区版9.0.0及以上 |

注意：

- 样例支持NPU上板运行（需要NPU硬件）和仿真器模式（不需要NPU硬件）两种运行方式。仿真器模式运行方式，请参考[运行环境变量配置](../../docs/quick_start.md#envvar-config)完成配置。
- PyTorch和torch_npu的安装，请参考[样例运行验证](../../docs/quick_start.md#example-verification)。

## 目录结构

```text
09_linear/
├── ascendc/
│   ├── build.sh          # 编译脚本，生成build/demo
│   ├── CMakeLists.txt    # CMake构建配置
│   └── linear.asc        # 基于Ascend C实现的Linear算子
├── bench_linear.py       # 基于PyAsc、torch_npu实现的算子测试入口文件
├── linear.py             # 基于PyAsc实现的Linear算子
├── profile_msprof.py     # msprof性能测试脚本，覆盖PyAsc、Ascend C、torch_npu三种方式实现的算子性能测试
└── README.md
```

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| x | 输入 | `[M,K]` | float16 | ND |
| weight | 输入 | `[N,K]` | float16 | ND |
| y | 输出 | `[M,N]` | float16 | ND |

约束：
- 当前实现要求N和K是16的整数倍，M可以是任意正整数。

## 样例实现

### 整体策略

三种计算策略完成的都是同一个矩阵乘，输入输出格式也完全相同。它们的区别不在计算公式，而在于分块配置何时确定，以及Device端的数据搬运和Cube计算由谁组织。

| 计算策略 | 如何组织计算 | 适用场景 |
|---|---|---|
| 运行时Matmul | host根据本次输入shape生成分块配置，Matmul高阶API在Device端管理数据搬运、Cube计算和结果写回 | 默认策略，覆盖所有满足样例约束的M、N、K |
| 编译期Matmul | PyAsc JIT编译时确定基本块和循环边界，Device端不再解析运行时分块配置 | K为512或1024，并且固定基本块既能覆盖足够多的核，有利于降低Scalar控制开销 |
| 显式Cube计算 | 样例直接组织片上搬运、Cube乘加和FixPipe写回，不经过Matmul高阶API的运行时调度 | 未进入编译期Matmul，默认128×128分块产生的任务不超过可用核数一半，且矩阵满足16元素对齐、K可按512分段 |

这里的“基本块”是一次Cube矩阵乘主要处理的输出区域；“运行时”和“编译期”描述的是分块配置的确定时间，不表示两种不同的矩阵乘算法；“显式Cube计算”表示样例直接调用底层搬运和Cube接口。

程序先判断输入是否适合编译期Matmul，再判断是否适合显式Cube计算，其余输入使用运行时Matmul。编译期Matmul内部又覆盖两类常见情况：K方向规约较长而默认任务较少时，使用较小的64×64输出块增加并行度；M很大而N较窄时，使用较高的1024×128输出块复用权重。

### 关键步骤

1. **准备权重**：host把ND权重转换为NZ布局并复制到设备全局内存，这样有助于Cube连续读取。推理过程中同一权重通常会重复使用，因此转换结果会被缓存，后续调用可以直接复用。

2. **确定输出基本块**：程序先用128×128输出块估算任务数量，总任务数 $= \lceil M/128 \rceil \times \lceil N/128 \rceil$。任务太少时，把M方向缩小为64行以增加并行度；任务很多时，让一个核连续计算相邻输出块，以复用已经搬入片上存储的数据。这里的“基本块”是一次矩阵乘主要处理的输出区域，不代表输入必须正好是该尺寸。

3. **分配多核任务**：输出块按轮转方式分给AIC。每个核从与自身核号对应的块开始，完成后按启动核数继续领取后续块，所以逻辑任务数可以大于物理核数，也不会遗漏多出来的任务。

4. **执行K方向乘加**：输入和权重从设备全局内存搬入片上存储，Cube沿K方向进行float32累加。K较大时会分段完成乘加，第一段初始化累加结果，后续分段继续累加。

5. **转换并写回结果**：累加完成后，FixPipe把float32结果转换为float16并写回ND输出。

### 核心接口

| 接口 | 用途 |
|------|------|
| `host.MultiCoreMatmulTiling` | 根据矩阵shape、可用核数和基本块生成运行时分块配置。 |
| `asc.adv.TCubeTiling` | 保存启动核数、单核计算范围和片上缓冲配置。 |
| `asc.adv.Matmul` | 执行矩阵乘法。 |
| `asc.adv.register_matmul` | 将Matmul对象、流水资源、workspace和分块配置关联起来。 |
| `asc.adv.get_mm_config` | 灵活的自定义Matmul模板参数配置。 |
| `Matmul.set_tail` | 在不改变Tiling的情况下，重新设置本次计算的singleCoreM/singleCoreN/singleCoreK，以元素为单位。 |
| `Matmul.iterate_all` | 调用一次iterate_all，会计算出singleCoreM \* singleCoreN大小的C矩阵。 |

### 分块、多核、流水线逻辑

- **分块逻辑**

  - 输出矩阵沿M和N方向切成多个输出块，每个输出块构成一个逻辑任务。
  - 对于输出块很多的大矩阵，运行时Matmul会把相邻基本块合并为较大的单核区域，减少重复读取输入或权重。

- **多核逻辑**

  - 程序根据逻辑任务数决定实际启动的AIC核数：

    ```text
    启动核数 = min(可用AIC核数, 逻辑任务数)
    ```

  - 当逻辑任务多于启动核数时，每个核按固定步长继续领取后续任务。例如10个输出块由4个核计算时：

    ```text
    核0：块0、块4、块8
    核1：块1、块5、块9
    核2：块2、块6
    核3：块3、块7
    ```

  - 这种分配方式称为轮转分配：当前块编号等于“核编号 + 轮次 × 启动核数”。它允许逻辑任务数大于物理核数，并让各核承担的任务数量最多相差一个。

- **流水线逻辑**

  - Device端的数据路径为：

    ```text
    GM →(MTE2)→ L1 →(MTE1)→ L0A/L0B →(Cube)→ L0C(float32)
       →(FixPipe)→ GM(ND输出)
    ```

  - 运行时Matmul和编译期Matmul由Matmul高阶API安排上述搬运与计算阶段。
  - 显式Cube计算按K方向每512个元素处理一段：先把输入和权重搬到L1及L0A/L0B，再由Cube累加到L0C，所有K分段完成后通过FixPipe转换为float16并写回。
  - 三种策略使用相同的多核任务分配原则，但片上流水的组织方式不同。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。

```bash
cd pyasc/examples/09_linear
python3 linear.py -r [RUN_MODE] -v [SOC_VERSION]
```

- RUN_MODE：NPU仿真（Model）或NPU上板（NPU）。
- SOC_VERSION：昇腾AI处理器型号。

```bash
python3 linear.py -r NPU -v Ascend910B4
```

执行成功后输出：

```text
[INFO] start process sample linear.
[INFO] Sample linear run success.
```

## 性能测试

### 概述

- 性能测试目的：对比用PyAsc、Ascend C以及torch_npu分别实现的Linear算子的性能差异。
- 性能测试环境：
  - NPU硬件：Ascend 910B4。
  - CANN版本：社区版9.0.0。
  - 输入数据类型：float16。
  - 采集工具：msprof。
- 统计方式：每个shape通过`msprof op`采集`OpBasicInfo.csv`的Task Duration。每次运行warmup=5（预热）+ iters=10（计时）。
- 复现命令（从 `pyasc` 仓库的父目录执行）：

  ```bash
  # 编译Ascend C演示程序
  cd pyasc/examples/09_linear/ascendc
  bash build.sh

  # 运行性能测试，生成summary.csv和pipeline_detail.csv
  cd ..
  python3 profile_msprof.py --output ./prof_results/full_run
  ```

- 表头说明：
  - x shape：输入矩阵x的shape，格式为`[M,K]`。
  - weight shape：权重矩阵weight的shape，格式为`[N,K]`。
  - scenario：该shape的应用场景。
  - pyasc(us)、ascendc(us)、torch_npu(us)：对应实现的Task Duration，单位为微秒。
  - py/asc：PyAsc耗时与Ascend C耗时的比值，小于1表示PyAsc更快。
  - py/torch：PyAsc耗时与torch_npu耗时的比值，小于1表示PyAsc更快。

- 性能测试结果

  | scenario | x shape `[M,K]` | weight shape `[N,K]` | pyasc(us) | ascendc(us) | torch_npu(us) | py/asc | py/torch |
  |----------|-----------------|----------------------|-----------|-------------|---------------|--------|----------|
  | Hidden短序列 | `[128,512]` | `[512,512]` | 4.48 | **4.24** | 4.70 | 1.057 | 0.953 |
  | QKV短序列 | `[128,512]` | `[1536,512]` | 7.76 | 8.12 | **7.38** | 0.956 | 1.051 |
  | FFN降维短序列 | `[128,1024]` | `[512,1024]` | 5.98 | **5.80** | 5.98 | 1.031 | 1.000 |
  | Hidden中等序列 | `[1024,512]` | `[512,512]` | 10.42 | **9.78** | 11.16 | 1.065 | 0.934 |
  | QKV中等序列 | `[1024,512]` | `[1536,512]` | 16.04 | 16.94 | **14.86** | 0.947 | 1.079 |
  | FFN降维中等序列 | `[1024,1024]` | `[512,1024]` | 14.90 | 14.34 | **13.64** | 1.039 | 1.092 |
  | Hidden长序列 | `[4096,512]` | `[512,512]` | 18.00 | 18.88 | **17.08** | 0.953 | 1.054 |
  | QKV长序列 | `[4096,512]` | `[1536,512]` | 47.28 | **44.58** | 58.66 | 1.061 | 0.806 |
  | FFN降维长序列 | `[4096,1024]` | `[512,1024]` | **31.66** | 32.72 | 34.40 | 0.968 | 0.920 |

  **性能结果分析**

  - PyAsc与Ascend C性能基本持平。9组shape的`py/asc`为0.947x~1.065x，最大差距为6.5%；其中PyAsc在4组shape上更快，Ascend C在5组shape上更快。
  - PyAsc与`torch_npu`整体接近。9组shape的`py/torch`为0.806x~1.092x，其中PyAsc在4组shape上更快、1组持平、4组更慢。差距最大的是QKV长序列，PyAsc快19.4%；PyAsc落后最多的是FFN降维中等序列，慢9.2%。
  - 测试shape覆盖Hidden、QKV和FFN降维三类典型线性投影；M=128、1024和4096分别对应短、中、长序列。