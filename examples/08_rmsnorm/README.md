# 08 — RMSNorm 归一化算子

## 概述

本样例实现了 RMSNorm（Root Mean Square Normalization）归一化算子，是 LLaMA 等 Transformer 架构的标准归一化层。

本样例面向推理场景，接口仅输出归一化结果`y`，不额外输出均方根倒数`rstd`。

计算公式：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}} \cdot \gamma$$

## 运行环境要求

| 类别 | 要求 |
|------|------|
| AI 处理器 | Ascend 910B / 910C |
| CANN 版本 | 社区版 8.5.0.alpha001 及以上 |

注意：

- 样例支持NPU上板运行（需要NPU硬件）和仿真器模式（不需要NPU硬件）两种运行方式。仿真器模式运行方式，请参考[运行环境变量配置](../../docs/quick_start.md#envvar-config)完成配置。
- PyTorch和torch_npu的安装，请参考[样例运行验证](../../docs/quick_start.md#example-verification)。

## 目录结构

```text
08_rmsnorm/
├── ascendc/              # Ascend C 手写对标实现
│   ├── build.sh          # 编译脚本，生成 build/demo
│   ├── CMakeLists.txt    # CMake 构建配置
│   └── rmsnorm.asc       # Ascend C kernel 与 ACL 原生 demo 入口
├── bench_rmsnorm.py       # msprof 被测程序
├── profile_msprof.py      # msprof 性能测试脚本
├── README.md
└── rmsnorm.py             # PyAsc 算子实现
```

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| x | 输入 | [B×S×H] | float32 | ND |
| gamma | 输入 | [H] | float32 | ND |
| y | 输出 | [B×S×H] | float32 | ND |

## 样例实现

### 整体流程

```
Global Memory (x_gm, gamma_gm)
    │  gamma: data_copy → TBuf（一次性加载，set_flag/wait_flag 同步）
    │  x: copy_in → data_copy → TQue.enque
    ▼
TQue (in_queue)
    │  compute: deque → asc.adv.rmsnorm(x, gamma, eps, tiling) → enque
    ▼
TQue (out_queue)
    │  copy_out: deque → data_copy
    ▼
Global Memory (y_gm)
```

### 关键步骤

- **gamma 加载** — gamma 权重通过 `TBuf(VECCALC)` 一次性加载，GM→UB 搬运后插入 `asc.set_flag(asc.HardEvent.MTE2_V)` / `asc.wait_flag(asc.HardEvent.MTE2_V)` 显式同步 MTE2 与 Vector 流水线。
- **copy_in** — 将输入数据 x 从 Global Memory 搬运到 VECIN 队列。
- **compute** — 调用 `asc.adv.rmsnorm` 高阶 API。H维度按64对齐且小于2048、处理行数按8对齐时使用基本块路径；其余情况使用通用路径。
- **copy_out** — 计算结果从 out_queue 搬回 Global Memory。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.TPipe` | TPipe 统一管理 Device 端内存等资源，一个 Kernel 函数必须且只能初始化一个 TPipe 对象。 |
| `asc.TQue` | 流水任务之间通过队列完成任务间通信和同步。 |
| `asc.TBuf` | 临时变量占用的内存可以使用 TBuf 数据结构来管理。 |
| `asc.TPipe.init_buffer` | 为 TQue 等队列和 TBuf 分配内存。 |
| `asc.TQue.alloc_tensor` | 从 Que 中分配 Tensor。 |
| `asc.TQue.enque` | 将 Tensor push 到队列。 |
| `asc.TQue.deque` | 将 Tensor 从队列中取出。 |
| `asc.TQue.free_tensor` | 释放 Que 中的指定 Tensor。 |
| `asc.TBuf.get` | 从 TBuf 上获取指定长度的 Tensor。 |
| `asc.data_copy` | 支持 Local Memory 与 Global Memory 之间的数据搬运。 |
| `asc.adv.rmsnorm` | 实现对 shape 大小为 [B, S, H] 的输入数据的 RMSNorm 归一化。 |
| `asc.adv.RmsNormTiling` | 方便用户获取 RMSNorm kernel 计算时所需的 Tiling 参数。 |
| `asc.set_flag` / `asc.wait_flag` | 同一核内不同流水线之间的同步指令。 |
| `asc.get_block_idx` | 获取当前核的 index，用于多核逻辑控制及多核偏移量计算。 |

### 分块、多核、流水线逻辑

- **多核切分**：
  - 通过 `rt.device_info(RT_MODULE_TYPE_VECTOR_CORE, INFO_TYPE_CORE_NUM)` 查询 AIV 核数。
  - `effective_cores = max(1, min(total_rows, max_core_num))` — 总行数少于可用核数时实际核数等于总行数。
  - `rows_per_core = ceil_div(total_rows, effective_cores)` — 每核处理的行数。
  - 各核通过 `asc.get_block_idx() * block_length` 计算在 Global Memory 中的偏移量。
- **分块计算**：
  - `_rows_per_call(hidden_size)` — 核内每次 RmsNorm 调用处理的行数（hidden_size ≤ 512 → 每次处理 8 行、≤ 1024 → 每次处理 4 行、其他→每次处理 2 行）。
  - `full_groups = rows_per_core // max_rows` — 完整的 max_rows 行组数。
  - `rem_rows = rows_per_core % max_rows` — 剩余尾行数。
  - 整块：
    - `chunk_size = max_rows × hidden_size `— 整块元素数。
    - `aligned_rows = ((max_rows + 16 - 1) // 16) × 16` — 按 16 行对齐。
    - `RmsNormTiling` — `b_length=1`（batch 为 1）、`s_length=max_rows`（行数）、`h_length=hidden_size`（隐藏层维度）、`reciprocal_of_h_length=1.0/hidden_size`、`main_bsh_length=chunk_size`、`main_bs_length_align=aligned_rows`、`input_tail_pos=chunk_size`（无 tail）。
  - 尾块：
    - `rem_bsh = rem_rows × hidden_size` — 尾块元素数。
    - `RmsNormTiling` 中 `s_length=rem_rows`、`main_bsh_length=rem_bsh`、`main_bs_length=rem_rows`、`main_bs_length_align=((rem_rows + 16 - 1) // 16) × 16`、`input_tail_pos=rem_bsh`，其余与整块相同。
  - 输入队列和输出队列各分配两个缓冲区（`BUFFER_NUM = 2`）。
- **流水线同步**：TPipe + TQue 管理输入搬运、Vector计算和输出搬运的先后关系。gamma 只在每个核启动时从GM加载一次，并通过`MTE2_V`事件保证Vector计算开始前数据已到达UB。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。

```
cd pyasc/examples/08_rmsnorm
python3 rmsnorm.py -r [RUN_MODE] -v [SOC_VERSION]
```

- RUN_MODE：NPU 仿真（Model）或 NPU 上板（NPU）。
- SOC_VERSION：昇腾 AI 处理器型号。

```bash
# NPU 上板模式
python3 rmsnorm.py -r NPU -v Ascend910B4
```

执行成功后输出：

```
[INFO] start process sample rmsnorm.
[INFO] Sample rmsnorm run success.
```

## 性能测试

### 概述

- 性能测试目的：对比 RMSNorm 的 PyAsc、Ascend C 和 torch_npu 实现。

- 性能测试环境：
  - NPU硬件：Ascend 910B4
  - CANN软件版本：社区版9.0.0
  - PyAsc软件版本：1.1.1
  - 输入数据类型：float32
  - 性能数据采集工具：msprof

- 统计方式：每个 shape 通过 `msprof op` 采集 `OpBasicInfo.csv` 的 Task Duration。每次运行 warmup=5（预热）+ iters=10（计时）。

- 复现命令（从 `pyasc` 仓库的父目录执行）：

  ```bash
  # 编译 Ascend C demo
  cd pyasc/examples/08_rmsnorm/ascendc
  bash build.sh

  # 运行性能测试，输出两份 CSV：
  # summary.csv：每个 shape 的PyAsc、Ascend C、torch_npu 耗时 + py/asc + py/torch 比值
  # pipeline_detail.csv：vec/scalar/mte2/mte3 分项耗时
  cd ..
  python3 profile_msprof.py --output ./prof_results
  ```

- 表头说明
  - shape：输入张量 shape，格式为 `(batch, seq, hidden)`。
  - elements：输入张量元素总数。
  - pyasc(us)、ascendc(us)、torch_npu(us)：对应实现的 Task Duration，单位为微秒。
  - py/asc：PyAsc 与 Ascend C 耗时比值。
  - py/torch：PyAsc 与 torch_npu 耗时比值，小于 1 表示 PyAsc 更快。

- 性能测试结果

  | shape | elements | pyasc(us) | ascendc(us) | torch_npu(us) | py/asc | py/torch |
  | --- | --- | --- | --- | --- | --- | --- |
  | (2,64,256) | 32K | 3.56 | **3.30** | 6.74 | 1.079 | 0.528 |
  | (2,64,512) | 64K | 3.70 | **3.64** | 6.44 | 1.016 | 0.575 |
  | (2,128,512) | 128K | **4.36** | 5.18 | 6.58 | 0.842 | 0.663 |
  | (2,256,512) | 256K | 5.36 | **5.22** | 7.60 | 1.027 | 0.705 |
  | (2,512,512) | 512K | **6.88** | 6.94 | 10.18 | 0.991 | 0.676 |
  | (2,1024,512) | 1M | **10.14** | 10.16 | 12.90 | 0.998 | 0.786 |
  | (2,1024,1024) | 2M | 19.02 | 19.16 | **19.00** | 0.993 | 1.001 |
  | (2,2048,512) | 2M | **16.54** | 16.86 | 17.72 | 0.981 | 0.933 |
  | (2,2048,1024) | 4M | 34.44 | 34.96 | **30.04** | 0.985 | 1.146 |

- 流水耗时明细

  | shape | backend | Task Duration(us) | Vec(us) | Scalar(us) | MTE2(us) | MTE3(us) |
  | --- | --- | --- | --- | --- | --- | --- |
  | (2,64,256) | pyasc | 3.56 | **0.48** | **1.91** | **0.45** | **0.17** |
  | (2,64,256) | ascendc | **3.30** | 0.49 | 1.97 | 0.48 | 0.17 |
  | (2,64,256) | torch_npu | 6.74 | 1.22 | 5.06 | 2.48 | 0.27 |
  | (2,64,512) | pyasc | 3.70 | **0.58** | 1.94 | **0.54** | **0.17** |
  | (2,64,512) | ascendc | **3.64** | 0.59 | **1.78** | 0.74 | 0.20 |
  | (2,64,512) | torch_npu | 6.44 | 1.38 | 4.46 | 2.64 | 0.50 |
  | (2,128,512) | pyasc | **4.36** | **0.90** | **2.20** | **0.74** | **0.22** |
  | (2,128,512) | ascendc | 5.18 | 0.92 | 2.44 | 1.84 | 0.96 |
  | (2,128,512) | torch_npu | 6.58 | 1.52 | 3.37 | 3.01 | 0.99 |
  | (2,256,512) | pyasc | 5.36 | **1.21** | 2.98 | 1.24 | **0.48** |
  | (2,256,512) | ascendc | **5.22** | 1.24 | **2.78** | **1.21** | 0.52 |
  | (2,256,512) | torch_npu | 7.60 | 1.80 | 5.07 | 2.35 | 1.22 |
  | (2,512,512) | pyasc | **6.88** | **1.92** | **3.29** | **2.04** | 1.21 |
  | (2,512,512) | ascendc | 6.94 | 1.99 | 3.43 | 2.48 | **1.06** |
  | (2,512,512) | torch_npu | 10.18 | 2.68 | 5.61 | 4.59 | 1.95 |
  | (2,1024,512) | pyasc | **10.14** | **3.74** | 4.51 | **3.47** | **2.02** |
  | (2,1024,512) | ascendc | 10.16 | 3.81 | **4.37** | 3.71 | 2.07 |
  | (2,1024,512) | torch_npu | 12.90 | 4.14 | 6.67 | 7.08 | 2.84 |
  | (2,1024,1024) | pyasc | 19.02 | 9.96 | **5.87** | 6.28 | 3.58 |
  | (2,1024,1024) | ascendc | 19.16 | 10.07 | 6.03 | **6.05** | **3.50** |
  | (2,1024,1024) | torch_npu | **19.00** | **7.13** | 7.87 | 9.54 | 4.76 |
  | (2,2048,512) | pyasc | **16.54** | 7.23 | 6.43 | **6.05** | 3.42 |
  | (2,2048,512) | ascendc | 16.86 | 7.36 | 6.86 | 6.46 | **3.10** |
  | (2,2048,512) | torch_npu | 17.72 | **7.03** | **5.85** | 8.84 | 5.27 |
  | (2,2048,1024) | pyasc | 34.44 | 19.74 | **8.94** | **11.53** | **5.99** |
  | (2,2048,1024) | ascendc | 34.96 | 19.97 | 9.72 | 11.80 | 6.21 |
  | (2,2048,1024) | torch_npu | **30.04** | **13.33** | 8.95 | 15.15 | 9.23 |

- 结果分析
  - PyAsc与Ascend C性能基本持平。9组shape中，8组耗时差异不超过2.7%；最小shape`(2,64,256)`的绝对差值仅`0.26 us`。
  - PyAsc在9组shape中的7组快于`torch_npu`，优势为6.7%~47.2%。这些shape下PyAsc仅生成推理所需的`y`，Vector、Scalar和数据搬运流水通常也更短；`torch_npu`接口还会生成`rstd`，额外计算和写回均计入Task Duration。
  - `H=1024`时差距缩小：`(2,1024,1024)`两者仅相差0.1%，`(2,2048,1024)`中PyAsc慢14.6%。后一个shape下PyAsc的Vector耗时为`19.74 us`，高于`torch_npu`的`13.33 us`，成为总耗时差异的主要来源；PyAsc较低的MTE2和MTE3耗时不足以抵消Vector计算差距。
