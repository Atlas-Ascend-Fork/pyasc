---
name: asctile-create-empty-pass
description: Use when adding, scaffolding, or creating a new empty MLIR pass in AscTile, AscendC, AscVF, AscVF2, or LowerToAsc. Do not use for implementing pass logic or creating pass tests.
---

# AscTile: Create Empty Pass

Create a registered, buildable, no-op MLIR pass. Update exactly these five files, preserving repository conventions:

1. `experimental/asctile/include/asctile/<namespace-path>/Passes.td`
2. `experimental/asctile/include/asctile/<namespace-path>/Passes.h`
3. `experimental/asctile/lib/<namespace-path>/<PassName>.cpp`
4. `experimental/asctile/lib/<namespace-path>/CMakeLists.txt`
5. `experimental/asctile/python/src/Passes.cpp`

Do not implement pass logic, create tests, edit generated files, or add manual registration to an `*-opt` tool.

## Inputs

Ask for missing values before editing. Offer choices where shown.

- **Namespace**: select from the table below. If the directory or pybind function does not exist, stop and ask the user to create the infrastructure first. For a new conversion, the user must first create `include/asctile/Conversion/<Name>/{Passes.td,Passes.h,CMakeLists.txt}`, `lib/Conversion/<Name>/CMakeLists.txt`, and `define<Name>Passes` in `python/src/Passes.cpp`; then rerun this skill.
- **Pass name**: required PascalCase, preferably imperative (`DetectKernelType`, `FuseLoop`). Accept the user's chosen style if they insist. Check that no TableGen def, C++ symbol, or Python binding already uses it.
- **Operation type**: `func::FuncOp` for transformations inside functions, or `ModuleOp` when changing function or module attributes, adding/removing functions, or coordinating multiple functions.
- **Summary**: use the user's one-line summary, optionally rephrased for style. If absent or deferred, use `"TODO: add summary"`. Never invent one.
- **Options**: if requested, collect C++ name, CLI name, type, C++ default literal, and one-line description.

Use these option types only: `bool`, `int`, `unsigned`, `int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`, `double`, `std::string`.

## Namespace Map

| Selection | C++ namespace | CLI prefix | `<namespace-path>` | Pybind function |
|---|---|---|---|---|
| AscendC | `ascendc` | `ascendc-` | `Dialect/AscendC/Transforms` | `defineAscendCPasses` |
| AscTile | `asctile` | `asctile-` | `Dialect/AscTile/Transforms` | `defineAscTilePasses` |
| AscVF | `ascvf` | `ascvf-` | `Dialect/AscVF/Transforms` | `defineAscVFPasses` |
| AscVF2 | `ascvf2` | `ascvf2-` | `Dialect/AscVF2/Transforms` | `defineAscVF2Passes` |
| LowerToAsc | `asclower` | `asclower-` | `Conversion/LowerToAsc` | `defineLowerToAscPasses` |

All paths above are under `experimental/asctile/include/asctile/` and `experimental/asctile/lib/`.

## Naming Contract

For pass `MyOptimization` in namespace `asctile`:

| Artifact | Correct value |
|---|---|
| TableGen def | `MyOptimization` |
| CLI pass | `asctile-my-optimization` |
| Generated macro | `GEN_PASS_DEF_MYOPTIMIZATION` |
| Generated base/options | `MyOptimizationBase`, `MyOptimizationOptions` |
| Concrete class | `MyOptimizationPass` |
| C++ factory | `mlir::asctile::createMyOptimizationPass()` |
| C++ filename | `MyOptimization.cpp` |
| Python function | `add_my_optimization` |

The `Pass` suffix belongs only in the generated concrete class name and factory name. Convert PascalCase to kebab-case for CLI, snake_case for Python, and ALL-CAPS without underscores for the macro.

## Canonical Example

This AscTile `func::FuncOp` skeleton has two options. Replace namespace paths, symbols, operation type, summary, and options as needed. Keep defaults identical in TableGen, the C++ declaration, and Python binding.

```tablegen
def MyOptimization : Pass<"asctile-my-optimization", "func::FuncOp"> {
  let summary = "TODO: add summary";
  let constructor = "mlir::asctile::createMyOptimizationPass()";
  let options = [
    Option<"enableFastMode", "enable-fast-mode", "bool", "false",
           "Enable fast optimization mode">,
    Option<"threshold", "threshold", "int", "10",
           "Optimization threshold">,
  ];
}
```

In `Passes.h`, inside `namespace mlir { namespace asctile { ... } }`, alphabetically among the other declarations:

```cpp
std::unique_ptr<Pass> createMyOptimizationPass(bool enableFastMode = false, int threshold = 10);
```

Create `experimental/asctile/lib/Dialect/AscTile/Transforms/MyOptimization.cpp`:

```cpp
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MYOPTIMIZATION
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct MyOptimizationPass : public asctile::impl::MyOptimizationBase<MyOptimizationPass> {
    MyOptimizationPass(const MyOptimizationOptions& options) : MyOptimizationBase(options) {}

    void runOnOperation() override
    {
        // TODO: Implement pass logic
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMyOptimizationPass(bool enableFastMode, int threshold)
{
    MyOptimizationOptions options;
    options.enableFastMode = enableFastMode;
    options.threshold = threshold;
    return std::make_unique<MyOptimizationPass>(options);
}
```

For no options, remove the constructor and options struct, use an empty `runOnOperation()` body, and define the factory without parameters:

```cpp
std::unique_ptr<Pass> mlir::asctile::createMyOptimizationPass() { return std::make_unique<MyOptimizationPass>(); }
```

For `ModuleOp`, use `"ModuleOp"` in TableGen, omit `func::FuncOps.h`, and use `pm.addPass(...)` in Python instead of `pm.addNestedPass<func::FuncOp>(...)`.

Add `MyOptimization.cpp` alphabetically to the source list in `lib/Dialect/AscTile/Transforms/CMakeLists.txt`. Empty skeletons need no additional link dependencies; add dialect dependencies only later when implementation requires them.

Inside `defineAscTilePasses`, insert alphabetically among Python function names:

```cpp
m.def(
    "add_my_optimization",
    [](PassManager& pm, bool enableFastMode, int threshold) {
        pm.addNestedPass<func::FuncOp>(createMyOptimizationPass(enableFastMode, threshold));
    },
    "pm"_a, "enable_fast_mode"_a = false, "threshold"_a = 10);
```

Without options, use the existing macro:

```cpp
DEFINE_ADD_PASS_ON(func::FuncOp, "add_my_optimization", createMyOptimizationPass);
```

For a `ModuleOp` pass without options, use `DEFINE_ADD_PASS` instead.

## TableGen Defaults

TableGen default fields are quoted C++ literals:

```tablegen
Option<"enableFastMode", "enable-fast-mode", "bool", "false", "Enable fast mode">,
Option<"threshold", "threshold", "int", "10", "Threshold value">,
Option<"ratio", "ratio", "double", "0.5", "Optimization ratio">,
Option<"name", "name", "std::string", "\"default\"", "Pass name">,
```

Do not add `let description = [{...}];` to a skeleton; add it when implementing the pass. Add `dependentDialects` only when implementation creates or depends on operations from another dialect.

## Validation

1. Confirm all five files contain matching names, namespaces, paths, operation types, option types, and defaults.
2. Confirm each insertion is alphabetically ordered by TableGen def name, C++ declaration/factory name, source filename, or Python function name—not by CLI name or description.
3. Run `clang-format -i experimental/asctile/lib/<namespace-path>/<PassName>.cpp`. If unavailable, report it; do not install or change the environment.
4. Run `python3 -m pip install -e .`.
5. If compilation fails in generated/new code, fix the skeleton and rebuild. If failure is infrastructure or environment related (missing pip, compiler, CANN/Bisheng, dependencies), stop and ask the user to fix it.

Report success with the five changed files. Do not claim tests pass; this skill does not create or run pass tests.
