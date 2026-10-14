<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# Ascend C Python算子调试调优指南
本文档介绍了Ascend C Python工程支持的算子调试调优方法使用指导。

## 算子功能调试

为了方便开发者Model仿真、NPU实际运行时，进行功能调试，pyasc提供printf、dump_tensor接口供开发者调用，其中printf主要用于打印标量和字符串信息，dump_tensor用于打印指定Tensor的数据。

### 使用方法
#### printf示例
```python
import asc
@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, BLOCK_LENGTH: asc.ConstExpr[int],
                BUFFER_NUM: asc.ConstExpr[int], TILE_LENGTH: asc.ConstExpr[int], TILE_NUM: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * BLOCK_LENGTH
    ...
    # 使用printf打印字符串
    asc.printf("Before calculating.\\n")
    for i in range(TILE_NUM):
        # 使用printf打印当前循环次数
        asc.printf("current index is %d.\\n", i)
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, TILE_LENGTH)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, TILE_LENGTH)
        copy_out(i, z_gm, out_queue_z, TILE_LENGTH)
```
执行结果样例如下：
```
opType=v, DumpHead: AIV-0, CoreType=AIV, block dim=16, total_block_num=16, block_remain_len=1048024, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
Before calculating.
current index is 0.
current index is 1.
current index is 2.
current index is 3.
current index is 4.
current index is 5.
current index is 6.
current index is 7.
```

- 约束

  - 使用asc.printf若需换行，需对'\n'进行转义；

  - asc.printf接口会对算子实际运行的性能带来一定影响，通常在调测阶段使用。

#### dump_tensor示例
```python
import asc
@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, BLOCK_LENGTH: asc.ConstExpr[int],
                BUFFER_NUM: asc.ConstExpr[int], TILE_LENGTH: asc.ConstExpr[int], TILE_NUM: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * BLOCK_LENGTH
    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    ...
    # 使用dump_tensor打印x_gm输入
    tmp_array = asc.array(asc.uint32, [4, 16])
    tmp_shape_info = asc.ShapeInfo(tmp_array)
    asc.dump_tensor(x_gm, 0, 32, tmp_shape_info)
    for i in range(TILE_NUM):
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, TILE_LENGTH)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, TILE_LENGTH)
        copy_out(i, z_gm, out_queue_z, TILE_LENGTH)

@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, in_queue_x: asc.TQue, in_queue_y: asc.TQue,
            TILE_LENGTH: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[i * TILE_LENGTH:], count=TILE_LENGTH)
    # 使用dump_tensor打印Local Memory的Tensor
    if i == 0:
        asc.dump_tensor(x_local, 1, 32)
    ...
```
执行结果样例如下：
```
opType=v, DumpHead: AIV-0, CoreType=AIV, block dim=16, total_block_num=16, block_remain_len=1048024, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=0, addr=41200000, data_type=float32, position=GM, dump_size=32
[[19.000000, 4.000000, 38.000000, 50.000000, 39.000000, 67.000000, 84.000000, 98.000000, 21.000000, 36.000000, 18.000000, 46.000000, 10.000000, 92.000000, 26.000000, 38.000000],
[39.000000, 9.000000, 82.000000, 37.000000, 35.000000, 65.000000, 97.000000, 59.000000, 89.000000, 63.000000, 70.000000, 57.000000, 35.000000, 3.000000, 16.000000, 42.000000],
[-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-],
[-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-]]
DumpTensor: desc=1, addr=0, data_type=float32, position=UB, dump_size=32
[6.000000, 34.000000, 52.000000, 38.000000, 73.000000, 38.000000, 35.000000, 14.000000, 67.000000, 62.000000, 30.000000, 49.000000, 86.000000, 37.000000, 84.000000, 18.000000, 38.000000, 18.000000, 44.000000, 21.000000, 86.000000, 99.000000, 13.000000, 79.000000, 84.000000, 9.000000, 48.000000, 74.000000, 52.000000, 99.000000, 80.000000, 53.000000]
```

- 约束

  - asc.dump_tensor接口会对算子实际运行的性能带来一定影响，通常在调测阶段使用。

## 算子性能调优

为了帮助开发者快速完成高性能算子开发，pyasc支持通过算子性能调优工具msprof op采集profiling数据，生成性能数据、内存热力图、仿真流水图等。本文档介绍了msprof op在算子开发过程中的应用。纯 PyAsc 算子还可使用项目内置的 [`asc.lib.profiling.Profiler`](python-api/lib/profiling.md)，无需安装 PyTorch / `torch_npu`，详见[使用 pyasc 原生 Profiler 采集性能数据](#使用-pyasc-原生-profiler-采集性能数据)。

msprof op工具用于采集和分析运行在昇腾AI处理器上算子的关键性能指标，开发者根据输出的profiling数据快速定位算子的软硬件性能瓶颈，提升算子性能分析效率。该工具的详细介绍请参考：[《算子开发工具-算子调优》](https://hiascend.com/document/redirect/CannCommunityToolMsProf)。

### 环境准备

工具使用前，需完成环境准备，详细参考[quick_start.md](quick_start.md#envready)。

### 算子性能数据采集

msprof op工具包含msprof op（上板）和msprof op simulator（仿真）两种使用方式，协助用户定位算子内存、算子代码以及算子指令的异常，实现全方位的算子调优。

功能简介：

| 功能名称 | 适用场景       | 展示的图形                      |
|----------|------------------|----------------------------------|
| msprof op   | 适用于实际运行环境中的性能分析   | 计算内存热力图、Roofline瓶颈分析图、Cache热力图、通算流水图、算子代码热点图  |
| msprof op simulator  | 适用于开发和调试阶段，进行详细仿真调优 | 指令流水图、算子代码热点图、内存通路吞吐率波形图 |

本文以add算子为例，介绍如何在上板和仿真两种场景中使用msprof op算子性能调优工具。

add算子实现代码参考[02_add_framework.py](../examples/02_add_framework/add_framework.py),

#### 上板性能数据采集

算子上板信息采集，参考命令如下，详细命令参数请参考msprof op工具文档的说明。
```Shell
msprof op --output=./output python add_framework.py -r NPU
```

成功执行后在output目录下生成如下文件：
```
OPPROF_{timestamp}_XXX
├── dump
├── ArithmeticUtilization.csv
├── L2Cache.csv
├── Memory.csv
├── MemoryL0.csv
├── MemoryUB.csv
├── OpBasicInfo.csv
├── PipeUtilization.csv
├── ResourceConflictRatio.csv
├── visualize_data.bin
```

- 将visualize_data.bin文件导入MindStudio Insight后，将会展示计算内存热力图、Roofline瓶颈分析图、Cache热力图、通算流水图和算子代码热点图。
- 将trace.json文件导入Chrome浏览器或MindStudio Insight后，将会展示通算流水图。

上述展示的图示内容的说明参考[算子调优](https://hiascend.com/document/redirect/CannCommunityToolMsProf)的相关章节。

#### 仿真性能数据采集

算子仿真（以Ascend910B1 simulator为例）运行信息采集，参考命令如下，各参数含义请参考msProf工具文档的说明。
```Shell
msprof op simulator --output=./output python add_framework.py -r Model -v Ascend910B1
```
 成功执行后在output目录下生成如下文件：
```
OPPROF_{timestamp}_XXX
├── dump
└── simulator
    ├── core0.veccore0       // 按照core*.veccore*或core*.cubecore*目录存放各核的数据文件
    │   ├── core0.veccore0_code_exe.csv
    │   ├── core0.veccore0_instr_exe.csv
    │   └── trace.json     // 该核的仿真指令流水图文件
    ├── core0.veccore1
    │   ├── core0.veccore1_code_exe.csv
    │   ├── core0.veccore1_instr_exe.csv
    │   └── trace.json
    ├── core1.veccore0
    │   ├── core1.veccore0_code_exe.csv
    │   ├── core1.veccore0_instr_exe.csv
    │   └── trace.json
    ├── ...
    ├── visualize_data.bin
    └── trace.json      // 全部核的仿真指令流水图文件
```

- 将visualize_data.bin文件导入MindStudio Insight后，将会展示指令流水图、算子代码热点图和内存通路吞吐率波形图。
- 将trace.json文件导入Chrome浏览器或MindStudio Insight后，将会展示指令流水图和内存通路吞吐率波形图。

上述展示的图示内容的说明参考[算子调优](https://hiascend.com/document/redirect/CannCommunityToolMsProf)的相关章节。

## 使用 pyasc 原生 Profiler 采集性能数据

除命令行 `msprof op` 外，PyAsc 在 `asc.lib.profiling` 中提供了基于 CANN `libmsprofiler.so`（`<acl/acl_prof.h>`）的原生 Profiler。它封装了 `aclprofInit`、`aclprofCreateConfig`、`aclprofStart`、`aclprofStop`、`aclprofDestroyConfig`、`aclprofFinalize`，可在不安装 PyTorch、`torch_npu` 的情况下采集纯 PyAsc 算子性能，并解析 CSV 结果。接口说明见 [asc.lib.profiling](python-api/lib/profiling.md)。

### 环境准备

- pyasc 环境准备请参考 [quick_start.md](quick_start.md#envready)。
- 依赖 CANN 运行时中的 `libmsprofiler.so`，以及 `tools/profiler/profiler_tool/analysis/msprof/msprof.py`（用于导出 summary / timeline）。
- 使用前先通过 `asc.runtime.config.set_platform` 选择后端。原生 Profiler 面向 NPU 上板采集。

### 使用上下文管理器采集

下面以 add 算子为例。`vadd_kernel` / `vadd_launch` 的完整实现见 [02_add_framework.py](../examples/02_add_framework/add_framework.py)，也可将其中的 Tensor 换成 NumPy 数组，无需依赖 `torch_npu`。

```python
import numpy as np
import asc
import asc.runtime.config as config
from asc.lib.profiling import AicoreMetrics, Profiler, task_time_median

# vadd_kernel / vadd_launch 实现参见 examples/02_add_framework/add_framework.py

config.set_platform(config.Backend.NPU)
x = np.random.rand(8 * 2048).astype(np.float32)
y = np.random.rand(8 * 2048).astype(np.float32)

profiler = Profiler()
with profiler.profile(metrics=AicoreMetrics.PIPE_UTILIZATION):
    z = vadd_launch(x, y)

result = profiler.last_result
print("median task time (us):", task_time_median(result.tasks, name="vadd_kernel"))
for task in result.tasks:
    print(task.id, task.name, task.type, task.duration)
```

`profile()` 会依次执行 `start` → 运行核函数 → `stop` → `export` → 解析最新一次结果。未指定 `result_path` 时使用临时目录，并在上下文结束时清理。

### 配置 AicoreMetrics 与采集类型

`Profiler.start()` / `profile()` 通过 `metrics` 选择一组 AI Core 指标，通过 `profile_types` 选择采集数据类型。默认指标为 `AicoreMetrics.PIPE_UTILIZATION`，默认类型为任务耗时、AI Core 指标和 L2 Cache。

```python
from asc.lib.profiling import AicoreMetrics, ProfileType, Profiler

profiler = Profiler(result_path="./pyasc_prof")
profiler.start(
    metrics=AicoreMetrics.MEMORY_BANDWIDTH,
    profile_types=[ProfileType.TASK_TIME, ProfileType.AICORE_METRICS, ProfileType.L2CACHE],
)
z = vadd_launch(x, y)
profiler.stop()
profiler.export()
profiler.store_last_result()
print(profiler.last_result.tasks)
```

常用 `AicoreMetrics` 取值：

| 枚举值 | 说明 |
|--------|------|
| AicoreMetrics.ARITHMETIC_UTILIZATION | 算术单元利用率 |
| AicoreMetrics.PIPE_UTILIZATION | 流水线利用率（默认） |
| AicoreMetrics.MEMORY_BANDWIDTH | 内存带宽 |
| AicoreMetrics.L0B_AND_WIDTH | L0B 及相关位宽指标 |
| AicoreMetrics.RESOURCE_CONFLICT_RATIO | 资源冲突比例 |
| AicoreMetrics.MEMORY_UB | UB 内存指标 |
| AicoreMetrics.L2_CACHE | L2 Cache 指标 |
| AicoreMetrics.PIPE_EXECUTE_UTILIZATION | 流水执行利用率 |
| AicoreMetrics.MEMORY_ACCESS | 访存指标 |
| AicoreMetrics.NONE | 不采集 AI Core 指标 |

需要完全自定义 `aclprofCreateConfig` 时，可直接使用 `asc.lib.profiling.MsprofInterface`。

### 解析结果

导出后的 CSV（`op_summary_*.csv` 或 `task_time_*.csv`）会被解析为 `ProfilingResult`，其中每条 `ProfilingTask` 包含任务 ID、kernel 名、任务类型和耗时（微秒）。`task_time_median` 只统计 AI Core 类任务，可用 `name` 过滤 kernel，用 `skip` 丢弃 warmup 样本。

## 使用Ascend PyTorch Profiler采集性能数据

pyasc支持针对PyTorch框架开发的性能分析工具Ascend PyTorch Profiler。PyTorch训练/在线推理场景下，推荐通过Ascend PyTorch Profiler接口采集并解析性能数据，用户可以根据结果自行分析和识别性能瓶颈，请参考官方文档：[Ascend PyTorch Profiler性能数据采集和自动解析](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html)。

### 环境准备

pyasc的环境准备请参考[quick_start.md](quick_start.md#envready)，Ascend PyTorch Profiler的环境准备请参考[Ascend PyTorch Profiler性能数据采集和自动解析](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html)的前提条件小节。

本文以add算子为例，介绍如何通过Ascend PyTorch Profiler接口采集性能数据。

### 示例代码（通过with语句进行采集）

```Python
import torch
import torch_npu

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int):

    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    ......


def vadd_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)

    total_length = z.numel()
    block_length = total_length // USE_CORE_NUM

    vadd_kernel[USE_CORE_NUM, rt.current_stream()](x, y, z, block_length)
    return z


def vadd_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    z = vadd_launch(x, y)
    assert torch.allclose(z, x + y)


experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config) as prof:
    steps = 2
    for step in range(steps):
        vadd_custom(config.Backend.NPU)
        prof.step()

```

### NPU性能数据目录结构

执行后生成数据目录结构示例如下：

```
├── ubuntu_239247_20251101101435_ascend_pt
    ├── ASCEND_PROFILER_OUTPUT
    ├── FRAMEWORK
    ├── logs
    ├── PROF_000001_20230628101435646_FKFLNPEPPRRCFCBA
    ├── profiler_info.json
    ├── profiler_metadata.json
```
