<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.runtime.compiler

`CompileOptions` 控制核函数的 IR 变换与毕昇编译命令。可通过 `@asc.jit(**options)` 按字段名传入，也可构造 `asc.CompileOptions` / `asc.runtime.compiler.CompileOptions`。

## 接口列表

| 接口 | 说明 |
|------|------|
| `CompileOptions`(\*\*fields) | 二进制编译与 IR 变换选项。 |

## CompileOptions

### *class* asc.runtime.compiler.CompileOptions

**参数说明**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| debug | bool | False | 为毕昇编译命令增加 `-g` 等调试选项。 |
| strip_loc | bool | False | 编译 Pass 结束后剥离 IR 中的源码位置调试信息。 |
| verify_sync | bool | False | 运行流水同步校验 Pass。 |
| print_ir_before_all | bool | False | 在每个编译 Pass 前打印 IR，用于排查 Pass 问题。 |
| run_passes | bool | True | 在翻译为 Ascend C 前运行 lowering / 优化 / 后处理 Pass。 |
| kernel_type | KernelType \| None | None | 核类型。为 `None` 时由框架根据 IR 自动推导。取值见 [`KernelType`](config.md#kerneltype)。 |
| opt_level | int | 3 | 毕昇优化级别，合法值为 `1`、`2`、`3`。 |
| auto_sync | bool \| None | True | 是否开启毕昇 `--cce-auto-sync` 自动插入同步。 |
| auto_sync_log | str \| None | `""` | 保存自动同步插入信息的文件路径。`None` 或空字符串表示不落盘。 |
| bisheng_options | tuple[str, ...] \| None | None | 额外追加到毕昇编译命令的参数。 |
| always_compile | bool | False | 跳过 JIT 二进制缓存，每次重新编译。 |
| matmul_cube_only | bool | False | Matmul 是否按纯 Cube 模式编译（不含 Vector 核）。 |
| insert_sync | bool \| None | None | 是否插入 Queue 同步 Pass。`None` 表示由 IR 判断。 |
| vf_vec_len | int \| None | None | C310 架构的向量寄存器长度。其他架构不支持该选项。 |

**约束说明**

- `opt_level` 必须为 1、2 或 3，否则编译器初始化失败。
- `kernel_type` 若显式指定，必须是 [`KernelType`](config.md#kerneltype) 枚举值。
- `vf_vec_len` 仅在 C310（Ascend950PR 系列）上有效；C220 上传入会报错。C310 未指定时默认 256。

**调用示例**

```python
import asc
from asc.runtime.compiler import CompileOptions
from asc.runtime.config import KernelType

@asc.jit(opt_level=2, kernel_type=KernelType.AIV_ONLY, always_compile=True)
def kernel(...):
    ...

options = CompileOptions(debug=True, auto_sync=False)
```
