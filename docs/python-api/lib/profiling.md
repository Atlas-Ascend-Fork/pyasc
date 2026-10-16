<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.lib.profiling

基于 CANN `libmsprofiler.so`（`<acl/acl_prof.h>`）的原生性能采集接口。封装 `aclprofInit` / `aclprofCreateConfig` / `aclprofStart` / `aclprofStop` / `aclprofDestroyConfig` / `aclprofFinalize`，可在不安装 PyTorch、`torch_npu` 的情况下采集纯 PyAsc 算子性能。

使用指南见 [调试调优](../../op_debug_prof.md)。

## 接口列表

| 接口 | 说明 |
|------|------|
| [`Profiler`](#profiler) | 开箱即用的性能采集器，支持 `start` / `stop` 与 `profile()` 上下文管理器。 |
| [`AicoreMetrics`](#aicoremetrics) | AI Core 指标组，传入 `aclprofCreateConfig`。 |
| [`ProfileType`](#profiletype) | 采集数据类型。 |
| [`ProfilingResult`](#profilingresult) | 一次导出解析得到的任务列表。 |
| [`ProfilingTask`](#profilingtask) | 单条 kernel / AI Core 任务。 |
| [`task_time_median`](#task_time_median) | 计算 AI Core 任务耗时中位数（微秒）。 |
| [`MsprofInterface`](#msprofinterface) | 对 `aclprof*` C API 的 ctypes 封装，用于自定义采集配置。 |

## Profiler

### *class* asc.lib.profiling.Profiler(result_path: str \| None = None)

**参数说明**

- result_path：CANN 原始 profiling 输出目录。省略时使用临时目录；在 `profile(cleanup=True)` 结束后自动删除。

| 方法 | 说明 |
|------|------|
| `start`(device_id=None, metrics=AicoreMetrics.PIPE_UTILIZATION, profile_types=None) | 在指定设备上启动采集。`profile_types` 默认为 `TASK_TIME`、`AICORE_METRICS`、`L2CACHE`。 |
| `stop`() | 停止采集并调用 `aclprofFinalize`。 |
| `export`() | 调用 CANN `msprof.py` 导出 summary / timeline CSV。 |
| `store_result`(run_id) | 解析指定 run 目录下的 `op_summary_*.csv` 或 `task_time_*.csv`。 |
| `store_last_result`() | 解析结果目录中最新一次 run。 |
| `cleanup`() | 若使用临时目录，则删除原始输出。 |
| `profile`(device_id=None, cleanup=True, metrics=..., profile_types=None) | 上下文管理器：自动 `start` → 执行核函数 → `stop` → `export` → 解析结果。 |
| `last_result` | 最近一次解析得到的 `ProfilingResult`。 |

**约束说明**

- 依赖 CANN 运行时中的 `libmsprofiler.so` 与 `tools/profiler/profiler_tool/analysis/msprof/msprof.py`。
- 面向 NPU 上板采集；使用前需先调用 [`set_platform`](../runtime/config.md#set_platform)。
- `metrics` 同时只能选择一个 [`AicoreMetrics`](#aicoremetrics) 指标组。

**调用示例**

```python
from asc.lib.profiling import Profiler, AicoreMetrics, task_time_median

profiler = Profiler()
with profiler.profile(metrics=AicoreMetrics.PIPE_UTILIZATION):
    kernel[core_num](...)
print(task_time_median(profiler.last_result.tasks, name="kernel"))
```

## AicoreMetrics

### *class* asc.lib.profiling.AicoreMetrics

| 枚举值 | 说明 |
|--------|------|
| AicoreMetrics.ARITHMETIC_UTILIZATION | 算术单元利用率 |
| AicoreMetrics.PIPE_UTILIZATION | 流水线利用率（`Profiler` 默认值） |
| AicoreMetrics.MEMORY_BANDWIDTH | 内存带宽 |
| AicoreMetrics.L0B_AND_WIDTH | L0B 及相关位宽指标 |
| AicoreMetrics.RESOURCE_CONFLICT_RATIO | 资源冲突比例 |
| AicoreMetrics.MEMORY_UB | UB 内存指标 |
| AicoreMetrics.L2_CACHE | L2 Cache 指标 |
| AicoreMetrics.PIPE_EXECUTE_UTILIZATION | 流水执行利用率 |
| AicoreMetrics.MEMORY_ACCESS | 访存指标 |
| AicoreMetrics.NONE | 不采集 AI Core 指标 |

## ProfileType

### *class* asc.lib.profiling.ProfileType

可按位组合后传给 `aclprofCreateConfig`。`Profiler.start()` 默认采集 `TASK_TIME | AICORE_METRICS | L2CACHE`。

| 枚举值 | 说明 |
|--------|------|
| ProfileType.ACL_API | ACL API 轨迹 |
| ProfileType.TASK_TIME | 任务耗时 |
| ProfileType.AICORE_METRICS | AI Core 指标 |
| ProfileType.AICPU | AI CPU |
| ProfileType.L2CACHE | L2 Cache |
| ProfileType.HCCL_TRACE | HCCL 通信轨迹 |
| ProfileType.TRAINING_TRACE | 训练轨迹 |
| ProfileType.MSPROFTX | msprof TX |
| ProfileType.RUNTIME_API | Runtime API |
| ProfileType.TASK_TIME_L0 | L0 任务耗时 |
| ProfileType.TASK_MEMORY | 任务内存 |
| ProfileType.OP_ATTR | 算子属性 |

## ProfilingResult

### *class* asc.lib.profiling.ProfilingResult(tasks, run_id, stored_at)

| 字段 | 说明 |
|------|------|
| tasks | `ProfilingTask` 元组 |
| run_id | 本次导出目录名 |
| stored_at | 解析时间 |

## ProfilingTask

### *class* asc.lib.profiling.ProfilingTask(id, name, type, duration)

| 字段 | 说明 |
|------|------|
| id | 任务 ID |
| name | kernel / 算子名 |
| type | 任务类型，例如 `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC` |
| duration | 耗时，单位微秒 |

## task_time_median

### *function* asc.lib.profiling.task_time_median(tasks, name=None, skip=0) → float

计算 AI Core 类任务的耗时中位数（微秒）。仅统计类型属于 `AI_CORE`、`AIV_SQE`、`AI_VECTOR_CORE`、`MIX_AIC`、`MIX_AIV`、`KERNEL_AIVEC`、`KERNEL_AICORE` 的记录。

**参数说明**

- tasks：`ProfilingResult.tasks`。
- name：只统计该 kernel 名。省略时使用第一条 AI Core 任务的名字。
- skip：丢弃前若干条匹配样本，用于跳过 warmup。必须非负，否则抛出 `ValueError`；丢弃后没有剩余样本时抛出 `RuntimeError`，不使用 warmup 样本计算中位数。

## MsprofInterface

### *class* asc.lib.profiling.MsprofInterface

对 `libmsprofiler.so` 的底层封装。一般使用 `Profiler` 即可；需要完全自定义 `aclprofCreateConfig` 参数时再直接调用。

| 方法 | 对应 C API |
|------|------------|
| `init`(result_path) | `aclprofInit` |
| `create_config`(device_ids, metric, types) | `aclprofCreateConfig` |
| `start`(config) | `aclprofStart` |
| `stop`(config) | `aclprofStop` |
| `destroy_config`(config) | `aclprofDestroyConfig` |
| `finalize`() | `aclprofFinalize` |
