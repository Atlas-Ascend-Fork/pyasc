# 07 — SwiGLU 激活算子

## 概述

本样例实现了 SwiGLU（Swish-Gated Linear Unit）激活函数算子，该算子广泛用于 LLaMA 等现代 Transformer 模型的 FFN 层。

计算公式：

$$\text{SwiGLU}(gate, up) = \frac{gate}{1 + e^{-gate}} \times up$$

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
07_swiglu/
├── ascendc/              # Ascend C 手写对标实现
│   ├── build.sh          # 编译脚本，生成 build/demo
│   ├── CMakeLists.txt    # CMake 构建配置
│   └── swiglu.asc        # Ascend C kernel 与 ACL 原生 demo 入口
├── bench_swiglu.py        # msprof 被测程序
├── profile_msprof.py      # msprof 性能测试脚本
├── README.md
└── swiglu.py              # PyASC 算子实现
```

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| gate | 输入 | [B×S×FFN] | float32 | ND |
| up | 输入 | [B×S×FFN] | float32 | ND |
| y | 输出 | [B×S×FFN] | float32 | ND |

输入范围为 [-2, 2]。

## 样例实现

### 整体流程

```
Global Memory (fused_gm: [gate; up])
    │  copy_in: alloc → data_copy(gate_half) → enque
    │           alloc → data_copy(up_half) → enque
    ▼
TQue VECIN (in_queue_gate)    TQue VECIN (in_queue_up)
    │                                      │
    └────────────┬─────────────────────────┘
                 │  compute: deque → SwiGLU(up, gate) → enque
                 ▼
         TQue VECOUT (out_queue)
                 │  copy_out: deque → data_copy
                 ▼
         Global Memory (y_gm)
```

### 关键步骤

- **copy_in** — gate 和 up 拼接为一个 tensor，从同一块 Global Memory 的两个偏移分别搬运到 VECIN 队列。
- **compute** — 调用高阶 API `asc.adv.swiglu(y, up, gate)` 完成 SwiGLU 计算。
- **copy_out** — 结果从 out_queue 搬回 Global Memory。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.TPipe` | TPipe 统一管理 Device 端内存等资源，一个 Kernel 函数必须且只能初始化一个 TPipe 对象。 |
| `asc.TQue` | 流水任务之间通过队列完成任务间通信和同步。TQue 用来执行队列相关操作、管理相关资源的数据结构。 |
| `asc.TPipe.init_buffer` | 为 TQue 等队列分配内存。 |
| `asc.TQue.alloc_tensor` | 从 Que 中分配 Tensor。 |
| `asc.TQue.enque` | 将 Tensor push 到队列。 |
| `asc.TQue.deque` | 将 Tensor 从队列中取出。 |
| `asc.TQue.free_tensor` | 释放 Que 中的指定 Tensor。 |
| `asc.data_copy` | 支持 Local Memory 与 Global Memory 之间的数据搬运，以及 Local Memory 内部的数据搬运。 |
| `asc.adv.swiglu` | SwiGLU 高阶 API，计算 dst = src0 × Swish(src1)。 |
| `asc.get_block_idx` | 获取当前核的 index，用于多核逻辑控制及多核偏移量计算。 |

### 分块、多核、流水线逻辑

- **多核切分**：
  - `max_core_num = rt.device_info(RT_MODULE_TYPE_VECTOR_CORE, INFO_TYPE_CORE_NUM)` — 查询 AIV 可用核数。
  - `candidates = _gen_core_candidates(max_core_num)` — 动态生成 2 的幂次候选集，末尾追加 `max_core_num` 自身。
  - `effective_cores` 取候选集中首个 `>= needed_cores` 的值。
  - 各核通过 `asc.get_block_idx() * block_length` 计算在 Global Memory 中的偏移量。
- **分块计算**：
  - `total_length = gate.numel()` — 输入张量的总元素数。
  - `dtype_size = gate.element_size()` — 单元素字节数（float32 为 4）。
  - `DATABLOCK_BYTES = 32`，`PREFERRED_COPY_BYTES = 2048`，`FALLBACK_COPY_BYTES = 1024` — 搬运粒度常量。`vec_align_elems = max(1, 32 // dtype_size)` 为最小对齐粒度，首选 tile = `max(vec_align_elems, 2048 // dtype_size)`，降级 tile = `max(vec_align_elems, 1024 // dtype_size)`。当总元素数 >= 首选 tile 时取大 tile，否则取小 tile，这样可以提升带宽利用率。
  - `block_length = ceil_div(ceil_div(total_length, effective_cores), tile_length) × tile_length` — 核内数据量按 tile 向上对齐。
  - 双缓冲机制（`BUFFER_NUM = 2`）。
- **流水线同步**：TPipe/TQue 框架管理搬运同步，Vector 计算间同步由高阶 API 内部管理。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/07_swiglu
python3 swiglu.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 swiglu.py -r Model -v Ascend910B1

# NPU 上板模式
python3 swiglu.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample swiglu.
[INFO] Sample swiglu run success.
```

## 性能测试

### 概述

- 性能测试目的：对比 SwiGLU 的 PyASC、Ascend C 和 torch_npu 实现。
- 性能测试环境：
  - NPU硬件：Ascend 910B4
  - CANN软件版本：社区版8.5.0
  - PyAsc软件版本：1.1.1
  - 输入数据类型：float32
  - 性能数据采集工具：msprof
- 统计方式：每个 shape 通过 `msprof op` 采集 `OpBasicInfo.csv` 的 Task Duration。每次运行 warmup=5（预热）+ iters=10（计时）。
- 复现命令（从 `pyasc` 仓库的父目录执行）：

  ```bash
  # 编译 Ascend C demo
  cd pyasc/examples/07_swiglu/ascendc
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
  | (2,64,256) | 32K | 4.22 | **3.92** | 5.04 | 1.077 | 0.837 |
  | (2,64,512) | 64K | 4.74 | **4.40** | 6.48 | 1.077 | 0.731 |
  | (2,128,512) | 128K | 5.48 | **5.16** | 9.12 | 1.062 | 0.601 |
  | (2,256,512) | 256K | 7.02 | **6.78** | 11.34 | 1.035 | 0.619 |
  | (2,512,512) | 512K | 10.44 | **10.04** | 13.70 | 1.040 | 0.762 |
  | (2,1024,512) | 1M | 17.20 | **16.84** | 20.50 | 1.021 | 0.839 |
  | (2,1024,1024) | 2M | 30.62 | **29.96** | 30.62 | 1.022 | 1.000 |
  | (2,2048,512) | 2M | 31.48 | **30.78** | 30.70 | 1.023 | 1.025 |
  | (2,2048,1024) | 4M | **57.30** | 58.52 | 53.26 | 0.979 | 1.076 |

- 结果分析
  - Pyasc 与 Ascend C 性能基本一致（py/asc 0.979x ~ 1.077x）。
  - elements ≤1M 时 PyAsc 快于 torch_npu（py/torch 0.601x ~ 0.839x）：因为 PyAsc 使用 40 核且 JIT 特化，torch_npu 仅 6 核且为通用实现。2M 时持平（py/torch 1.000x ~ 1.025x），4M 时 torch_npu 略快（py/torch 1.076x）。
