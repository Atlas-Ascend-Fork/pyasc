# 10 - Fused Infer Attention

## 概述

Fused Infer Attention（融合推理注意力）是一类面向神经网络推理场景的FlashAttention算子。它在保持注意力数学定义不变的前提下，减少中间结果的保存和搬运。它以Query、Key和Value为主要输入：Query表示需要查询的信息，Key用于与Query计算相关性，Value保存根据相关性进行汇聚的内容。

计算时，Query与Key的转置相乘得到注意力分数；scale用于控制分数幅度，mask用于屏蔽不应参与计算的位置，Softmax将分数转换为归一化的注意力权重。注意力权重再与Value相乘得到输出：

$$attention\_out = softmax\left(scale \cdot (QK^T) + mask\right)V$$

Fused表示将原本由多个算子完成的矩阵乘、缩放、mask、Softmax和加权求和合并为一个融合算子。融合可以减少中间结果在全局内存中的读写以及多次启动算子的开销，从而提高推理效率。

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
10_fused_infer_attention/
├── ascendc/
│   ├── build.sh                    # 编译脚本，生成build/demo
│   ├── CMakeLists.txt              # CMake构建配置
│   └── fused_infer_attention.asc   # 基于Ascend C实现的Fused Infer Attention算子
├── bench_fused_infer_attention.py  # 基于PyAsc、torch_npu实现的算子测试入口文件
├── fused_infer_attention.py        # 基于PyAsc实现的Fused Infer Attention算子
├── profile_msprof.py               # msprof性能测试脚本，覆盖PyAsc、Ascend C、torch_npu三种方式实现的算子性能测试
└── README.md
```

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | [B, N, S, D] | float16 | BNSD |
| key | 输入 | [B, N, S, D] | float16 | BNSD |
| value | 输入 | [B, N, S, D] | float16 | BNSD |
| atten_mask | 输入 | [1, 1, S, S] | bool | 可广播的causal mask |
| attention_out | 输出 | [B, N, S, D] | float16 | BNSD |

约束：

- 当前样例`D`支持64或128。
- mask必须是上三角causal mask，`True`表示该位置被屏蔽。所有batch和head共用同一份`S × S` mask；当前实现不支持稀疏mask或双向mask（稀疏mask：只让部分token互相看见，跳过大量无关token；双向mask：全部token互相可见，只屏蔽padding）。

## 样例实现

### 整体流程

```text
Query行块
    │
    ├── AIC：Query分块 × Key分块的转置
    │          得到一小块注意力分数
    │
    ├── AIV：缩放分数、加mask、计算在线Softmax
    │          得到当前分块的注意力权重
    │
    ├── AIC：注意力权重 × Value分块
    │          得到当前分块对输出的贡献
    │
    └── AIV：将各分块的结果合并并归一化
               输出attention_out
```

程序先确定Query分块和启动核数，再把矩阵乘交给AIC，把缩放、mask、Softmax、累加和归一化交给AIV。短序列在相邻Query任务之间形成流水，长序列在同一Query任务的多个Key/Value块之间形成流水。

### 关键步骤

输入采用BNSD布局：B是batch数，N是注意力头数，S是序列长度，D是单头维度。每个batch、每个head都要独立完成一次注意力计算。为了避免生成完整的`[S,S]`分数矩阵，样例同时沿两个方向分块：

- **Query行块**：一段连续的Query行，决定本次生成哪一段输出。
- **Key/Value块**：一段连续的Key和Value，决定本轮纳入哪些上下文token。
- **任务**：一个batch、一个head和一个Query行块的组合。一个任务可能依次处理多个Key/Value块。

假设一个任务包含R行Query，当前Key/Value块包含T个token，则数据变化如下：

```text
Query行块 [R,D] × Key块转置 [D,T]
                │
                ▼
     注意力分数 [R,T] → 缩放并叠加mask → 在线Softmax → 概率 [R,T]
                                                     │
                                                     ▼
                                           概率 [R,T] × Value块 [T,D]
                                                     │
                                                     ▼
                                             部分输出 [R,D]
```

1. **切分Query并确定可见范围**

   Query沿序列方向切成连续行块，每个`batch + head + Query行块`构成一个任务。因果注意力要求第`i`个token只能看到第0到第`i`个token，所以靠前的Query行块只访问较少的Key/Value，靠后的行块访问更多。代码先计算当前任务最远可以访问的位置，再用mask屏蔽行块内部仍不可见的元素。

2. **分块计算QK**

   当前任务不会一次读取全部Key，而是每次最多读取512个token。AIC使用Cube计算`Query行块 × Key块^T`，得到`[R,T]`分数块；AIV随后对分数进行缩放并叠加mask。

3. **用在线Softmax合并多个分数块**

   普通Softmax需要先保存完整的一行分数。“在线Softmax”是指按块更新Softmax状态，而不是等所有分数生成后再统一计算。它只为每一行保存当前最大值、按当前最大值计算的指数和、已经累加的输出。处理新分数块时，先比较新旧最大值；若最大值改变，就把旧指数和与旧输出按新基准缩放，再合并新块。这样逐块处理的结果与对完整分数行执行一次Softmax相同，但不需要保存完整`[S,S]`矩阵。

4. **计算PV并累加部分输出**

   AIC使用Cube计算`Probability × Value块`，得到当前Key/Value块贡献的`[R,D]`部分输出。AIV按照在线Softmax给出的缩放比例修正旧输出，再把当前部分输出加进去。

5. **归一化并写回**

   所有Key/Value块处理完成后，AIV用最终指数和归一化累加结果，并把`[R,D]`结果写回对应的Query行。所有任务完成后即可得到完整的`attention_out[B,N,S,D]`。

### 核心接口

| 接口 | 用途 |
| --- | --- |
| `asc.TPipe` | TPipe统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。 |
| `asc.TQue` | 流水任务之间通过队列完成任务间通信和同步。 |
| `asc.TBuf` | 临时变量占用的内存可以使用TBuf数据结构来管理。 |
| `asc.adv.Matmul` | 执行矩阵乘法。 |
| `asc.adv.TCubeTiling` | 描述两次矩阵乘的单核范围、基本块和缓冲配置。 |
| `asc.adv.softmax_flash_v2` | SoftmaxFlash增强版本，对应FlashAttention-2算法。 |
| `asc.data_copy` | 在设备全局内存和片上缓冲之间搬运分数、mask、概率与部分输出。 |
| `asc.cross_core_set_flag` / `asc.cross_core_wait_flag` | 面向分离架构的核间同步控制接口。 |

### 分块、多核、流水线逻辑

下面从分块、多核和流水线三个方面说明调度过程：先把数据切成任务，再把任务分给多个核，最后通过双缓冲和跨核事件衔接AIC与AIV的工作。

```text
完整Attention
    └── 切成多个任务：batch × head × Query行块
            └── 一个任务依次处理多个Key/Value块
                    ├── AIC：QK和PV矩阵乘
                    └── AIV：缩放、mask、在线Softmax、累加和归一化
                               └── 双缓冲流水与跨核事件保证并行且不覆盖数据
```

- **分块逻辑**

  - **Query行方向**：一个任务负责连续R行Query，也只写这R行输出。短序列只包含一个512列Key/Value块，此时使用128行Query块可以降低单任务等待时间；长序列则根据可用AIC数、batch与head数量选择分块，并将Query块限制在当前片上空间可容纳的256行以内，最后一个尾块可以更小。256行对应两个128行Cube基本块，可兼顾任务数量与AIC/AIV流水重叠。
  - **Key/Value方向**：一个任务把自己可见的Key/Value范围按每块最多512个token切分，然后按顺序处理。AIV再把当前`[R,T]`分数块按每次最多8行交给Softmax高阶API，控制临时片上空间的占用。
  - **因果计算裁剪**：以S=1024、R=256为例，一个head产生4个Query任务，Key/Value方向最多有2个块。前两个Query任务只处理第一个Key/Value块，后两个任务才需要处理第二个块。这样既让临时数据能够放入片上存储，也跳过了未来token对应的无效计算。

- **多核逻辑**

  - **任务划分**：每个`batch + head + Query行块`都是一个任务。程序动态查询芯片可用的AIC核数，再把连续任务分配给各核。
  - **负载均衡**：越靠后的Query任务可见范围越大，需要处理的Key/Value块越多。程序以“该任务的Query行数 × Key/Value块数”作为大致计算量，让每个核获得的总工作量尽量接近。这个步骤只改变任务归属，不改变任务内部的计算结果。
  - **AIC和AIV分工**：一个AI Core由一个AIC和两个AIV配合完成任务：

    | 核 | 负责的工作 |
    |----|------------|
    | AIC | 使用Cube执行`Q × K^T`和`Probability × V`两次矩阵乘。 |
    | AIV 0 | 处理Query行块前半部分的缩放、mask、在线Softmax、输出累加和归一化。 |
    | AIV 1 | 处理Query行块后半部分的同类Vector计算。 |

  - 两个AIV处理不同的行，不会重复计算。AIC擅长矩阵乘，AIV擅长逐元素和归约操作，因此这种分工可让Cube和Vector资源同时工作。

- **流水线逻辑**

  - 每个Key/Value块依次经历四个阶段：

    ```text
    QK（AIC）→ 在线Softmax（AIV）→ PV（AIC）→ 输出累加（AIV）
    ```

  - **双缓冲流水**：样例为中间结果准备两份可交替使用的缓冲区。两个缓冲区让当前块和前一块同时处于不同阶段，避免AIC与AIV完全串行执行。处理第N块时：

    - AIC先提交第N块的QK，再处理第N-1块的PV。
    - AIV处理第N块的在线Softmax，并接收、累加第N-1块的PV结果。
    - 两个缓冲区交替使用；只有前一块完成累加后，对应缓冲区才会被下一块复用。

    | 迭代周期 | AIC | AIV | 核心说明 |
    | ---- | ---- | ---- | ---- |
    | 迭代0（填充流水） | QK[0]（缓冲区0） | Softmax[0]（缓冲区0） | 计算第0块KV，中间数据存入缓冲区0；无前置KV块。AIC完成QK计算后，AIV再执行Softmax，只能串行执行，完成流水线预热 |
    | 迭代1（稳态流水） | QK[1]（缓冲区1） + PV[0]（缓冲区0） | Softmax[1]（缓冲区1） + 累加[0]（缓冲区0） | AIC并行执行：计算第1块QK（结果存入缓冲区1）、读取缓冲区0中的数据执行第0块PV；AIV并行执行：第1块进行Softmax、第0块输出累加；前后KV块任务重叠，硬件并发运行 |
    | 迭代2（稳态流水） | QK[2]（缓冲区0） + PV[1]（缓冲区1） | Softmax[2]（缓冲区0） + 累加[1]（缓冲区1） | 第0块KV全部计算完成，缓冲区0可供复用，并存放第2块中间数据；AIC计算第2块QK、读取缓冲区1中的数据执行第1块PV；AIV执行第2块Softmax、第1块输出累加 |
    | 迭代3（稳态流水） | QK[3]（缓冲区1） + PV[2]（缓冲区0） | Softmax[3]（缓冲区1） + 累加[2]（缓冲区0） | 第1块KV全部计算完成，缓冲区1可供复用，并存放第3块中间数据；AIC计算第3块QK、读取缓冲区0中的数据执行第2块PV；AIV执行第3块Softmax、第2块输出累加 |

    - 第一次迭代用于填充流水；从第二次迭代开始形成稳定重叠；主循环结束后，再处理最后一个尚未完成的PV和累加阶段。这里的“填充流水”表示第一个数据块依次进入各阶段，此时还没有前一块可以与它重叠。
  - **跨核同步**：同一个缓冲区会被AIC和AIV交替读写，因此每个缓冲区需要传递四类完成事件：

    | 完成事件 | 发送者 → 接收者 | 含义 |
    |------|-----------------|------------|
    | QK完成 | AIC → AIV | 分数已经写入缓冲区，AIV可以执行Softmax。 |
    | Softmax完成 | AIV → AIC | 概率已经写入缓冲区，AIC可以执行PV。 |
    | PV完成 | AIC → AIV | 部分输出已经写入缓冲区，AIV可以累加。 |
    | 累加完成 | AIV → AIC | AIV已经读完当前结果，AIC可以复用缓冲区。 |

    - AIC复用一个缓冲区前，必须确认上一轮累加已经完成；否则新一轮QK可能覆盖AIV仍在读取的数据。PyAsc使用跨核事件实现这些通知和等待，不需要用户手动轮询内存状态。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。

```bash
cd pyasc/examples/10_fused_infer_attention
python3 fused_infer_attention.py -r [RUN_MODE] -v [SOC_VERSION]
```

- RUN_MODE：NPU仿真（Model）或NPU上板（NPU）。
- SOC_VERSION：昇腾AI处理器型号。

```bash
# NPU上板模式
python3 fused_infer_attention.py -r NPU -v Ascend910B4
```

执行成功后输出：

```
[INFO] start process sample fused_infer_attention.
[INFO] Sample fused_infer_attention run success.
```

## 性能测试

### 概述

- 性能测试目的：对比用PyAsc、Ascend C以及torch_npu分别实现的Fused Infer Attention算子的性能差异。
- 性能测试环境：

  - NPU硬件：Ascend 910B4
  - CANN软件版本：社区版9.0.0
  - PyAsc软件版本：1.1.1
  - 输入数据类型：float16
  - 性能数据采集工具：msprof
- 统计方式：每个shape通过`msprof op`采集`OpBasicInfo.csv`的Task Duration。每次运行warmup=5（预热）+ iters=10（计时）。
- 复现命令（从 `pyasc` 仓库的父目录执行）：

  ```bash
  # 编译Ascend C demo
  cd pyasc/examples/10_fused_infer_attention/ascendc
  bash build.sh

  # 运行性能测试，生成summary.csv和pipeline_detail.csv
  cd ..
  python3 profile_msprof.py --output ./prof_results
  ```

- 表头说明：

  - shape：输入张量shape，格式为`(batch, heads, seq, head_dim)`。
  - scenario：该shape对应的Llama-1模型规模和Prefill场景。
  - elements：注意力分数矩阵的元素数，即`batch × heads × seq × seq`。
  - pyasc(us)、ascendc(us)、torch_npu(us)：对应实现的Task Duration，单位为微秒。
  - py/asc：PyAsc与Ascend C耗时比值。
  - py/torch：PyAsc与torch_npu耗时比值，小于1表示PyAsc更快。

- 性能测试结果

  | shape | scenario | elements | pyasc(us) | ascendc(us) | torch_npu(us) | py/asc | py/torch |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | (1,32,128,128) | Llama-1 7B，短Prompt | 512K | **23.90** | 30.40 | 36.06 | 0.786 | 0.663 |
  | (1,32,512,128) | Llama-1 7B，中等Prompt | 8M | **80.24** | 83.36 | 86.28 | 0.963 | 0.930 |
  | (1,32,1024,128) | Llama-1 7B，长Prompt | 32M | **221.64** | 229.26 | 257.70 | 0.967 | 0.860 |
  | (2,32,512,128) | Llama-1 7B，batch=2 | 16M | 137.16 | 137.46 | **131.74** | 0.998 | 1.041 |
  | (1,40,512,128) | Llama-1 13B，中等Prompt | 10M | **89.72** | 96.42 | 95.62 | 0.931 | 0.938 |
  | (1,64,512,128) | Llama-1 65B，中等Prompt | 16M | 137.60 | 138.06 | **129.12** | 0.997 | 1.066 |

  **性能结果分析**

  - PyAsc与Ascend C性能接近。6组shape的`py/asc`为0.786x~0.998x；短Prompt中PyAsc快21.4%，其余5组差距不超过6.9%。
  - PyAsc在多数场景中达到或超过`torch_npu`。6组shape的`py/torch`为0.663x~1.066x，其中PyAsc在4组shape上更快；在`(2,32,512,128)`和`(1,64,512,128)`上分别慢4.1%和6.6%。
  - 测试shape覆盖Llama-1 7B、13B和65B的head配置，并包含S=128～1024的典型Prefill序列长度。
