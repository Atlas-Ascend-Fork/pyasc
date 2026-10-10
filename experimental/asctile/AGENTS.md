<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Agent Guidelines for PyAsc + AscTile

PyAsc is a Python programming model for writing compute kernels that run on Huawei Ascend NPUs.
Two APIs: `asc` (1:1 Ascend C mapping, `@asc.jit`) for low-level control, `asc.experimental.asctile` (tile-based, NumPy-like, `@asctile.jit`) for high-level kernels.
Requires CANN toolkit (Bisheng compiler + NPU runtime).

## External File Loading

CRITICAL: When you encounter a file reference (e.g., @docs/development/codestyle.rst), use your Read tool to load it on a need-to-know basis.

Instructions:
- Do NOT preemptively load all references - use lazy loading based on actual need
- When loaded, treat content as mandatory instructions that override defaults
- Follow references recursively when needed

## Build, Lint, and Test Commands

### Building
- Install in development mode (recommended): `pip install -e .`
- Build wheel: `pip wheel . --no-deps -w dist/`
- Clean build artifacts: `rm -rf build/ dist/`
Other options in @docs/installation/build-from-source.rst.

### Linting and Formatting
- Run ruff linter: `ruff check python/`
- Run ruff with fixes: `ruff check --fix python/`
- Format with yapf (it is better to specify filenames): `yapf -rip python/`
- Format with yapf (diff only, recommended): `git diff | yapf-diff -i`
- Format C++: `clang-format -i <filename>`

### Testing
- Run asctile kernel tests: `pytest python/test/kernels/`
- Run asctile operation tests: `pytest python/test/operations/`
- Run asctile target tests: `pytest python/test/target/`
- Run specific test file: `pytest python/test/kernels/test_vadd.py`
- Run specific test function: `pytest python/test/kernels/test_vadd.py::test_vadd`
- Run tests in parallel: `pytest -n auto python/test/`
- Run with coverage: `pytest -n auto --cov=asc.experimental.asctile.language --cov-report=term-missing --cov-fail-under=90 python/test/unit/`
- Run backend/MLIR tests: `lit -v test/`
- Compile-only mode (no NPU required): `pytest --compile-only python/test/`
- Select backend/platform: `pytest --backend Model --platform Ascend910B1`

## Development Guidelines

- For detailed code style rules (Python/C++/MLIR): @docs/development/codestyle.rst
- For development tools and IDE setup: @docs/development/tools.rst
- For build instructions and dependencies: @docs/installation/build-from-source.rst
- For runtime environment setup: @docs/installation/setup-runtime-env.rst

## Code Style Guidelines

### Python Code
- Maximum line length: 120 characters
- Use yapf for formatting (PEP8-based with custom settings)
- Use isort for import organization (120 char line length)
- Configure yapf: `column_limit = 120`, `disable_split_list_with_comment = true`
- Use `from __future__ import annotations` for modern type hints
- Use type hints extensively: `from typing import Any, Optional, Union, List, Dict, Tuple`
- Use `@overload` decorator for function overloads
- Use `dataclass` for structured data with `@dataclass` decorator
- Naming: functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Every file must include the copyright header (see existing files)
- Use docstrings for classes and public methods
- Use `__all__` to explicitly export public API
- Use exceptions: `RuntimeError`, `ValueError`, `NotImplementedError`

### C++ Code
- Header files: `.h` extension, Implementation files: `.cpp` extension
- Use traditional include guards (`#ifndef`, `#define`, `#endif`) - **Do not use** `#pragma once`
- Use clang-format with `.clang-format` configuration (LLVM-based, 120 char line length)
- Indentation: 4 spaces (no tabs)
- Braces: opening brace on new line for functions, same line for classes/structs/enums/namespaces/control statements
- Pointers/references: aligned left (e.g., `int* ptr`)
- Short functions: can be single-line (empty bodies)
- Short if/loops: not allowed on single line
- Naming: classes/structs/enums `PascalCase`, variables/functions `camelCase`, macros `UPPER_SNAKE_CASE`
- Test filenames: `kebab-case` (e.g., `my-test-case.mlir`)
- Include ordering: local project, empty line, MLIR/LLVM/Clang, empty line, std library (all alphabetical)
- After closing namespace, add comment: `// namespace <name>`
- **Do not use** `using namespace` in header files
- Place file-local functions/classes in anonymous namespace (not `static`)
- Template typename arguments: `PascalCase` (e.g., `typename AttrT`), non-type: `camelCase`
- Always use `typename` instead of `class`

### MLIR Code
- Sort operations, types, attributes, interfaces alphabetically in TableGen files
- Each MLIR pass in separate `.cpp` file under `Transforms/` directory
- Pass filename matches pass name without "Pass" suffix (e.g., `UnrollLoop.cpp`)
- List passes alphabetically in `Passes.td`, `CMakeLists.txt`, and `Passes.h`
- Tests in `test/` directory with `kebab-case` filenames
- IR tests: `test/Dialect/<dialect>/IR/`, Transform tests: `test/Dialect/<dialect>/Transforms/`
- Target/emission tests: `test/Dialect/<dialect>/Target/`, Tool tests: `test/Tools/`

### Testing Guidelines
- Use pytest for Python tests, lit for backend/MLIR tests
- asctile tests: `python/test/asctile/kernels/` (end-to-end), `python/test/asctile/operations/` (op-level), `python/test/asctile/target/` (target-specific)
- asc tests: `python/test/kernels/` (end-to-end), `python/test/unit/` (unit with FileCheck)
- Use descriptive test names, group related tests in test classes
- Use fixtures for common setup/teardown, `pytest.skip()` for conditional test execution
- Unit tests use `FileCheck` fixture to verify generated MLIR (see `@python/test/unit/conftest.py`)

## AscTile Programming Model

### Core Concepts
- **GlobalTensor**: Global memory (HBM) descriptor with pointer and shape (dynamic shapes)
- **LocalTensor** (tile-like): On-chip memory (UB, L0A, L0B, L0C, L1) with fixed static shape (known at JIT time)
- Local tensors use value semantics - each operation produces a new tensor
- Memory hierarchy (Ascend NPU related): `TensorLocation.UB` (default), `TensorLocation.L0A`/`L0B`/`L0C`, `TensorLocation.L1`

### AscTile API Patterns
```python
# Tensor creation and loading
global_tensor = asctile.global_tensor(x_ptr, [size])
local_tensor = asctile.copy_in(global_tensor, shape=[128], offsets=[base])  # explicit element offsets
scalar = asctile.copy_in(global_tensor, offsets=[i])
asctile.copy_out(local_tensor, global_tensor, offsets=[base])

# Arithmetic: add, sub, mul, div, maximum, minimum, left_shift, right_shift
# Comparison: equal, not_equal, greater, greater_equal, less, less_equal
# Unary: abs, ceil, floor, negative, relu, sqrt, rsqrt, exp, log, sin, cos, tanh, erf, softmax
# Operator overloads: +, -, *, /, >, ==, etc.
# Reductions: reduce_sum, reduce_max, reduce_min, reduce_prod (with axes, keep_dims)
# Methods: local_tensor.sum(), local_tensor.max(), etc.
# Shape: reshape, broadcast_to, expand_dims, squeeze
# Creation: full(shape, value), zeros(shape), full_like, zeros_like
# Advanced: matmul(a, b), where(mask, src0, src1)
# Atomics: atomic_add, atomic_max, atomic_min

# Programming model operations
i = asctile.block_idx()    # current NPU block index
n = asctile.block_num()    # total number of blocks

# Loop control
for i in asctile.range(start, stop, step, unroll_factor=4, gm_barrier=False):
    # unroll_factor: how many iterations to unroll
    # gm_barrier=True: prevent parallel load/store optimization
```

### JIT Compilation
```python
@asctile.jit  # or with options: @asctile.jit(always_compile=True, ...)
def kernel(x_ptr, y_ptr, out_ptr, size: int, TILE: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    # ... kernel implementation

x = torch.rand_like(..., device="cpu")  # Initialize tensors with numpy or torch
kernel[8](x, y, out, size, TILE=256)    # Launch with 8 cores
```

### JIT Compile Options
- `run_asctile_passes=True`: Enable AscTile + AscLower pipeline (enabled by default when using `@asctile.jit`)
- `static_alloc=True`: Static vs TPipe-managed UB allocation
- `reuse_alloc=0`: Reuse freed UB regions (`1` to enable)
- `always_compile=False`: Bypass cache, recompile every call
- `opt_level=3`: Bisheng optimization level (1-3)
Other options in @python/asc/runtime/compiler.py (CompileOptions dataclass).

### Architecture Notes
- `../../python/asc/`: Core Python package (codegen, language APIs, runtime, lib bindings)
- `../../python/asc/codegen/`: Python AST → ASC-IR (MLIR) translation (FunctionVisitor)
- `../../python/asc/language/`: Language APIs (`basic/`, `adv/`, `core/`, `fwk/`, `tile/`)
- `../../python/asc/runtime/`: JIT compilation, caching, kernel launching
- `../../python/asc/_C/`: pybind11 bindings to C++ backend (`libpyasc`)
- `../../include/ascir/`: C++ header files and TableGen definitions for the upstream dialects (AscendC, EmitAsc)
- `../../lib/Dialect/Asc/`: Asc dialect IR and optimization passes
- `include/asctile/`: C++ header files and TableGen definitions for the extension dialects (AscTile, AscVF)
- `python/asctile/`: AscTile frontend API and JIT decorator (wrapper over asc)
- `lib/Dialect/AscTile/`: AscTile dialect IR and optimization passes
- `lib/Dialect/AscTile/Transforms/`: AscTile passes (UnrollLoop, PromotePureOps, TransformMathOps, etc.)
- `lib/Dialect/AscVF/`: AscVF dialect (vector fusion)
- `lib/Dialect/EmitAsc/`: EmitAsc dialect (code emission)
- `lib/Conversion/LowerToAsc/`: Lowering passes (AscTile → Asc dialect)
- `lib/Target/AscendC/`: Code emitter (Asc MLIR → Ascend C source)
- `bin/`: CLI tools (`asctile-opt`, etc.)

### Compilation Pipeline
1. Python AST → asctile MLIR (FunctionVisitor in `../../python/asc/codegen/`)
2. AscTile passes (unrolling, loop transforms, math specialization)
3. LowerToAsc conversion passes (asctile → asc dialect)
4. Asc passes (UB allocation, sync insertion, boilerplate generation)
5. CodeEmitter (asc MLIR → Ascend C source)
6. Bisheng compiler (Ascend C → .o binary)

### Key MLIR Passes
- **AscTile**: UnrollLoop, PromotePureOps, TransformMathOps, LegalizeMatmul, FoldCast
- **LowerToAsc**: LowerAscTile, LowerAscTileDataTransfer, LowerArith, LowerMath, LowerSCF, LowerAtomic
- **Asc**: InsertSync, InsertBufIdSync, HoistTensorAllocation, ReuseUBAllocation, ComputeMemoryConsumption

### Hardware Targets
- Ascend 910B series (C220 architecture): Ascend910B1, B2, B2C, B3, B4, B4-1
- Ascend 910_93 series (C220): Ascend910_9362, 9372, 9381, 9382, 9391, 9392
- Ascend 950PR series (C310): Ascend950PR_950z, 9579, 957b, 957c, 957d, 9589, 958b, 9599
- Automatic sync insertion per hardware variant
- BufID-based sync for C310 platforms (InsertBufIdSync)

### Debugging
- Set `PYASC_PRINT_IR=1` to print IR in-between passes to stderr (lots of internal MLIR)
- Use `always_compile=True` to bypass cache
- Set `PYASC_DUMP_PATH=<dir>` to inspect intermediate IR and generated Ascend C
- Set `CAMODEL_LOG_PATH=<dir>` to capture simulator logs when running on Model backend
- Use `--profile-path <dir>` with `--profile` to save NPU profiling CSV reports to a persistent directory
- Check MLIR with `lit` tests
- Use `--compile-only` pytest flag to test compilation without NPU hardware
