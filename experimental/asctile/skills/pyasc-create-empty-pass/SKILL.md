---
name: pyasc-create-empty-pass
description: Create empty MLIR pass skeleton for further development
---

# PyAsc: Create Empty Pass

This skill creates an empty MLIR pass skeleton with proper structure, following PyAsc conventions. The generated pass is a no-op placeholder ready for implementation.

**CRITICAL**: This skill does NOT create test files or implements pass logic.

## When to Use

- Developer needs to add a new MLIR pass
- AI agent workflow: create pass skeleton → implement pass logic

---

## Step 1: Gather Pass Metadata

**Use the `question` tool with `options` parameter for all selections.**

### 1.1 Select Namespace

Ask user to select from existing namespaces:

```
Options:
- "Asc (ascendc)" - Dialect passes for Ascend C operations
- "AscTile (asctile)" - Dialect passes for AscTile operations
- "AscVF (ascvf)" - Dialect passes for AscVF operations
- "LowerToAsc (asclower)" - Conversion passes from AscTile to Asc
- "New conversion (not listed)" - Create pass in new conversion directory
```

**If user selects "New conversion"**:
- Inform user: "The conversion directory structure does not exist yet. Please create the following manually before re-running this skill:"
  - `include/ascir/Conversion/<ConversionName>/Passes.td`
  - `include/ascir/Conversion/<ConversionName>/Passes.h`
  - `include/ascir/Conversion/<ConversionName>/CMakeLists.txt`
  - `lib/Conversion/<ConversionName>/CMakeLists.txt`
  - Add pybind function `define<ConversionName>Passes` to `python/src/Passes.cpp`
- **TERMINATE** skill with message: "Please create the conversion directory structure manually, then re-run this skill."

**Acceptance Criteria**:
- User selected one of the 4 existing namespaces
- Selected namespace directory exists (verify with `ls include/ascir/<path>`)

### 1.2 Collect Pass Information

Ask user for:

1. **Pass name** (required, PascalCase): e.g., `DetectKernelType`, `FuseLoop`, `AllocateTensor`

   **IMPORTANT**: Suggest imperative form (verb + noun): `DoSomething`, `TransformX`, `OptimizeY`, `DetectZ`, `FuseA`, `AllocateB`. Examples: `DetectKernelType`, `FuseLoop`, `AllocateTensor`, `HoistTensorAllocation`, `InsertSync`.

   If user insists on different style, accept without hesitation.

2. **Operation type** (required, selection):
   - `func::FuncOp` - Nested pass operating on functions
   - `ModuleOp` - Top-level pass operating on modules

   **IMPORTANT**: Inform user:
   - `func::FuncOp` pass can ONLY modify IR inside functions (operations, blocks)
   - `ModuleOp` pass is REQUIRED if you need to:
     - Modify function attributes (e.g., visibility, `ascendc.global`)
     - Add/remove functions
     - Modify module-level attributes
     - Access multiple functions simultaneously

3. **Summary** (required, but user may skip): One-line description of what the pass does.

   **IMPORTANT**: Ask user for summary. If user skips or says "skip"/"none"/"later":
   - Set summary to `"TODO: add summary"` in TableGen
   - **NEVER** guess, imagine, or generate a summary yourself
   - If user provides summary, you may rephrase for clarity and fix typos to match existing pass style (e.g., imperative form: "Hoist X to Y", "Remove redundant Z")

4. **Add options?** (yes/no):
   - If yes, collect for each option:
     - C++ name (camelCase): e.g., `enableOptimization`
     - CLI name (kebab-case): e.g., `enable-optimization`
     - Type: `bool`, `int`, `unsigned`, `int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`, `double`, `std::string`
     - Default value: **MUST be C++ literal in quotes** (see format rules below)
     - Description: One-line description

**Default Value Format Rules**:

**CRITICAL**: Default values in TableGen `Option<>` are string literals that get printed into C++ code. They MUST be quoted C++ literals:

- **bool**: `"false"` or `"true"` (quotes around the literal)
- **Integer types**: `"0"`, `"10"`, `"-1"` (quotes around the number)
- **Float types**: `"0.0"`, `"3.14"`, `"1.5e-10"` (quotes around the number)
- **std::string**: `"\"hello\""` (quotes around escaped-quoted string)

**Example TableGen options**:

```tablegen
let options = [
  Option<"enableFast", "enable-fast", "bool", "false", "Enable fast mode">,
  Option<"threshold", "threshold", "int", "10", "Threshold value">,
  Option<"ratio", "ratio", "double", "0.5", "Optimization ratio">,
  Option<"name", "name", "std::string", "\"default\"", "Pass name">,
];
```

**Acceptance Criteria**:
- Pass name is valid PascalCase (starts with uppercase, alphanumeric)
- All required fields collected
- Summary is either user-provided (possibly rephrased) or `"TODO: add summary"`
- Option names follow naming conventions (C++: camelCase, CLI: kebab-case)
- Default values are quoted C++ literals in correct format

---

## Step 2: Add TableGen Definition to Passes.td

**File**: `include/ascir/<namespace-path>/Passes.td`

**IMPORTANT**: The TableGen `.td` file generates `Passes.h.inc` which provides:
- Base class: `<namespace>::impl::<PassName>Base<Derived>`
- Options struct: `<namespace>::<PassName>Options` (if pass has options)
- `GEN_PASS_DEF_<PASSNAME>` macro

This generated file is included in Step 4's C++ implementation.

### Naming Rules

**CRITICAL**: "Pass" suffix rules:
- **DO NOT** use "Pass" suffix in TableGen def name
- **DO NOT** use "Pass" suffix in CLI name
- **DO** use "Pass" suffix in constructor function name

**Example**: For pass named `MyOptimization`:
- TableGen def: `def MyOptimization` (NOT `MyOptimizationPass`)
- CLI name: `"ascendc-my-optimization"` (NOT `"ascendc-my-optimization-pass"`)
- Constructor: `createMyOptimizationPass()` (WITH "Pass" suffix)

### Templates

**IMPORTANT**: The CLI prefix in the pass name depends on the **namespace**, not the operation type. Use the prefix from the Namespace Reference Table.

**Template 1: Simple pass (no options, func::FuncOp)**

```tablegen
def MyOptimization : Pass<"asctile-my-optimization", "func::FuncOp"> {
  let summary = "Brief description of what the pass does";
  let constructor = "mlir::asctile::createMyOptimizationPass()";
}
```

**Template 2: Simple pass (no options, ModuleOp)**

```tablegen
def MyOptimization : Pass<"asctile-my-optimization", "ModuleOp"> {
  let summary = "Brief description";
  let constructor = "mlir::asctile::createMyOptimizationPass()";
}
```

**Template 3: Pass with options**

```tablegen
def MyOptimization : Pass<"asctile-my-optimization", "func::FuncOp"> {
  let summary = "Brief description";
  let constructor = "mlir::asctile::createMyOptimizationPass()";
  let options = [
    Option<"enableFastMode", "enable-fast-mode", "bool", "false",
           "Enable fast optimization mode">,
    Option<"threshold", "threshold", "int", "10",
           "Minimum threshold for optimization">,
    Option<"maxIterations", "max-iterations", "unsigned", "100",
           "Maximum number of iterations">,
  ];
}
```

**Template 4: Pass with dependentDialects** (optional, use when pass creates ops from other dialects):

```tablegen
def MyOptimization : Pass<"asctile-my-optimization", "func::FuncOp"> {
  let summary = "Brief description";
  let constructor = "mlir::asctile::createMyOptimizationPass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
}
```

**IMPORTANT**: Do NOT add `let description = [{...}];` to skeleton passes. Description is added later when pass is implemented.

**Supported option types**: `bool`, `int`, `unsigned`, `int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`, `double`, `std::string`

### Instructions

1. Read existing `Passes.td` file
2. **Check for duplicate**: If a def with the same name already exists, ask user for a different pass name and restart Step 1.2
3. Find alphabetical position for new pass (sort by def name, NOT by CLI name)
4. Insert new def block at correct position
5. Convert PascalCase pass name to kebab-case for CLI name
6. Add namespace prefix to CLI name (from Namespace Reference Table)

**Acceptance Criteria**:
- New def block is alphabetically sorted among existing defs (by def name)
- CLI name uses kebab-case with correct namespace prefix
- Constructor function name ends with "Pass()"
- Syntax matches existing entries exactly
- No duplicate def names (verified before insertion)

---

## Step 3: Add Function Declaration to Passes.h

**File**: `include/ascir/<namespace-path>/Passes.h`

### Templates

**Without options**:

```cpp
std::unique_ptr<Pass> createMyOptimizationPass();
```

**With options** (default values match TableGen):

```cpp
std::unique_ptr<Pass> createMyOptimizationPass(bool enableFastMode = false, int threshold = 10, unsigned maxIterations = 100);
```

### Instructions

1. Read existing `Passes.h` file
2. Find alphabetical position for new declaration (sort by function name)
3. Insert declaration at correct position
4. Include default parameter values matching TableGen defaults

**Acceptance Criteria**:
- Declaration is alphabetically sorted by function name
- Function signature matches constructor in TableGen
- Default values match TableGen option defaults
- Function name ends with "Pass()"

---

## Step 4: Create C++ Implementation File

**File**: `lib/<namespace-path>/<PassName>.cpp`

**CRITICAL**: Filename does NOT have "Pass" suffix. Use `<PassName>.cpp`, NOT `<PassName>Pass.cpp`.

### Include Path Rules

**IMPORTANT**: Include paths MUST match the directory structure exactly:

- **Dialects**: `ascir/Dialect/<DialectName>/Transforms/Passes.h`
- **Conversions**: `ascir/Conversion/<ConversionName>/Passes.h`

**Examples**:
- Asc: `#include "ascir/Dialect/Asc/Transforms/Passes.h"`
- AscTile: `#include "ascir/Dialect/AscTile/Transforms/Passes.h"`
- AscVF: `#include "ascir/Dialect/AscVF/Transforms/Passes.h"`
- LowerToAsc: `#include "ascir/Conversion/LowerToAsc/Passes.h"`

### Template

```cpp
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MYOPTIMIZATION
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct MyOptimizationPass : public asctile::impl::MyOptimizationBase<MyOptimizationPass> {
    void runOnOperation() override
    {
        // TODO: Implement pass logic
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMyOptimizationPass() { return std::make_unique<MyOptimizationPass>(); }
```

**With options**, modify struct definition and factory function:

```cpp
struct MyOptimizationPass : public asctile::impl::MyOptimizationBase<MyOptimizationPass> {
    MyOptimizationPass(const MyOptimizationOptions& options) : MyOptimizationBase(options) {}

    void runOnOperation() override
    {
        // Access options: this->enableFastMode, this->threshold, etc.
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

**CRITICAL**: Factory function MUST use fully-qualified name `mlir::<namespace>::create<PassName>Pass()` at global scope. Do NOT wrap in `namespace mlir { namespace <namespace> { ... } }` blocks — this can cause hidden linking issues.

### Instructions

1. Create new file at `lib/<namespace-path>/<PassName>.cpp`
2. Replace placeholders:
   - `<namespace-path>`: e.g., `Dialect/AscTile/Transforms`
   - `<namespace>`: e.g., `asctile`
   - `<PassName>`: PascalCase pass name (e.g., `MyOptimization`)
   - `<PASSNAME_UPPER>`: PascalCase name converted to ALL CAPS with **NO underscores** (e.g., `MYOPTIMIZATION`, `UNROLLLOOP`, `HOISTTENSORALLOCATION`)
3. Include copyright header
4. Include necessary headers with correct paths
5. Add `using namespace mlir;` and `using namespace mlir::<namespace>;`
6. Implement empty `runOnOperation()` method
7. **Format file**: Run `clang-format -i lib/<namespace-path>/<PassName>.cpp`
   - If `clang-format` not available, notify user but DO NOT install or modify environment

**CRITICAL**: `GEN_PASS_DEF_` macro uses ALL CAPS with NO underscores. Examples from codebase:
- `MyOptimization` → `GEN_PASS_DEF_MYOPTIMIZATION`
- `UnrollLoop` → `GEN_PASS_DEF_UNROLLLOOP`
- `HoistTensorAllocation` → `GEN_PASS_DEF_HOISTTENSORALLOCATION`
- `DetectKernelType` → `GEN_PASS_DEF_DETECTKERNELTYPE`

**For conversion passes**, adjust include paths and namespace:
- Include: `#include "ascir/Conversion/LowerToAsc/Passes.h"`
- Namespace: `asclower`
- Generated include: `#include "ascir/Conversion/LowerToAsc/Passes.h.inc"`

**Acceptance Criteria**:
- File exists at correct path
- Filename is `<PassName>.cpp` (NO "Pass" suffix)
- Copyright header present
- Correct namespace used
- Include paths match directory structure exactly
- `GEN_PASS_DEF_<NAME>` macro is ALL CAPS with NO underscores
- Struct name is `<PassName>Pass` (WITH "Pass" suffix)
- Factory function uses fully-qualified name: `mlir::<namespace>::create<PassName>Pass()` at global scope (NOT wrapped in namespace blocks)
- For passes with options: explicit Options struct constructor used
- `using namespace mlir;` and `using namespace mlir::<namespace>;` present
- `runOnOperation()` body is empty (skeleton only)
- File formatted with clang-format (or user notified if unavailable)

---

## Step 5: Add to CMakeLists.txt

**File**: `lib/<namespace-path>/CMakeLists.txt`

### Instructions

1. Read existing `CMakeLists.txt`
2. Find source file list (after `add_mlir_dialect_library` or `add_mlir_conversion_library`)
3. Insert `<PassName>.cpp` alphabetically in the list
4. Maintain alphabetical order

**Note**: If the pass requires additional MLIR dialect libraries (e.g., `MLIRSCFDialect`, `MLIRArithDialect`), add them to the `LINK_LIBS PUBLIC` section. For skeleton passes, this is typically not needed — add dependencies when implementing the pass.

**Acceptance Criteria**:
- Filename is alphabetically sorted in source list
- No duplicate entries
- Filename matches created file (NO "Pass" suffix)

---

## Step 6: Add Pybind Definition

**File**: `python/src/Passes.cpp`

### Templates

**For func::FuncOp without options** (use macro):

```cpp
DEFINE_ADD_PASS_ON(func::FuncOp, "add_my_optimization", createMyOptimizationPass);
```

**For ModuleOp without options** (use macro):

```cpp
DEFINE_ADD_PASS("add_my_optimization", createMyOptimizationPass);
```

**For pass with options** (use lambda):

```cpp
m.def(
    "add_my_optimization",
    [](PassManager& pm, bool enableFastMode, int threshold, unsigned maxIterations) {
        pm.addNestedPass<func::FuncOp>(createMyOptimizationPass(enableFastMode, threshold, maxIterations));
    },
    "pm"_a, "enable_fast_mode"_a = false, "threshold"_a = 10, "max_iterations"_a = 100);
```

**For ModuleOp with options**:

```cpp
m.def(
    "add_my_optimization",
    [](PassManager& pm, bool enableFastMode) {
        pm.addPass(createMyOptimizationPass(enableFastMode));
    },
    "pm"_a, "enable_fast_mode"_a = false);
```

### Instructions

1. Read `python/src/Passes.cpp`
2. Identify correct function based on namespace:
   - `Asc` → `defineAscendCPasses`
   - `AscTile` → `defineAscTilePasses`
   - `AscVF` → `defineAscVFPasses`
   - `LowerToAsc` → `defineLowerToAscPasses`
3. Find alphabetical position within function
4. Insert appropriate template
5. Convert pass name to snake_case for Python function name (e.g., `MyOptimization` → `add_my_optimization`)
6. Convert option names to snake_case for Python parameters

**Acceptance Criteria**:
- Entry is alphabetically sorted within function (by Python function name)
- Entry is within the correct pybind function for the namespace (e.g., `defineAscTilePasses` for AscTile, not `defineAscendCPasses`)
- Uses `DEFINE_ADD_PASS` for ModuleOp, `DEFINE_ADD_PASS_ON(func::FuncOp, ...)` for func::FuncOp
- Lambda form used when options exist
- Python function name is snake_case with `add_` prefix
- Option parameter names are snake_case
- Default values match TableGen defaults

---

## Step 7: Build Verification

### Instructions

1. Run: `pip install -e .`
2. Check build output for errors

**If build succeeds**:
- Report success with list of created/modified files
- **TERMINATE** skill

**If build fails with infrastructure errors** (examples):
- "No module named 'pip'"
- "CANN toolkit not found"
- "Bisheng compiler not found"
- "Virtual environment not activated"
- "Missing dependencies"

**Action**: Ask user to fix infrastructure issues. **DO NOT** attempt to fix yourself. **TERMINATE** skill.

**If build fails with compiler errors in generated code**:
- Analyze error messages
- Fix syntax errors, missing includes, incorrect namespaces
- Rebuild and verify

**Acceptance Criteria**:
- Build completes without errors
- No compiler warnings in generated files
- All created files compile successfully

---

## Namespace Reference Table

**How to use**: The `<namespace-path>` placeholder in steps 2-5 is replaced with the "Include path" column value (without the trailing slash). For example, for AscTile: `<namespace-path>` = `Dialect/AscTile/Transforms`.

| Namespace | C++ ns | CLI prefix | Include path (from ascir/) | Lib path | Pybind func |
|-----------|--------|------------|----------------------------|----------|-------------|
| Asc | `ascendc` | `ascendc-` | `Dialect/Asc/Transforms/` | `Dialect/Asc/Transforms/` | `defineAscendCPasses` |
| AscTile | `asctile` | `asctile-` | `Dialect/AscTile/Transforms/` | `Dialect/AscTile/Transforms/` | `defineAscTilePasses` |
| AscVF | `ascvf` | `ascvf-` | `Dialect/AscVF/Transforms/` | `Dialect/AscVF/Transforms/` | `defineAscVFPasses` |
| LowerToAsc | `asclower` | `asclower-` | `Conversion/LowerToAsc/` | `Conversion/LowerToAsc/` | `defineLowerToAscPasses` |

**Include path examples**:
- Header: `#include "ascir/Dialect/AscTile/Transforms/Passes.h"`
- Generated: `#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"`

**Full file path examples** (using `MyOptimization` as pass name):
- TableGen: `include/ascir/Dialect/AscTile/Transforms/Passes.td`
- Header: `include/ascir/Dialect/AscTile/Transforms/Passes.h`
- C++ impl: `lib/Dialect/AscTile/Transforms/MyOptimization.cpp`
- CMakeLists: `lib/Dialect/AscTile/Transforms/CMakeLists.txt`

---

## Generic Dialect Support

**If new dialects are added in the future**, follow the same pattern:

1. Verify directory structure exists:
   - **Dialect**: `include/ascir/Dialect/<NewDialect>/Transforms/` and `lib/Dialect/<NewDialect>/Transforms/`
   - **Conversion**: `include/ascir/Conversion/<NewConversion>/` and `lib/Conversion/<NewConversion>/`
   - Required files: `Passes.td`, `Passes.h`, `CMakeLists.txt`

2. Determine namespace mappings:
   - C++ namespace: typically lowercase dialect name (e.g., `newdialect`)
   - CLI prefix: `<namespace>-` (e.g., `newdialect-`)
   - Include path: `ascir/Dialect/<NewDialect>/Transforms/` or `ascir/Conversion/<NewConversion>/`
   - Pybind function: `define<NewDialect>Passes` (must be added to `Passes.cpp`)

3. Follow same steps 2-7 with appropriate paths and names

**CRITICAL**:
- If Pybind function does not exist in `python/src/Passes.cpp`, ask user to add it manually or create it following existing pattern
- Include paths MUST include the `Transforms/` subdirectory for dialects
- Verify include paths match actual directory structure before proceeding

---

## Critical Rules

**MUST**:
- Maintain alphabetical order in ALL file modifications (note: some existing entries may not be sorted — insert new entries at the correct alphabetical position regardless)
- Use `question` tool with `options` for user input
- Include copyright header in new files
- Verify namespace directory exists before proceeding
- Use exact templates provided
- Format C++ files with `clang-format -i` after creation
- Use correct include paths with `Transforms/` subdirectory for dialects
- Ensure default values in TableGen are quoted C++ literals (e.g., `"10"`, `"false"`, `"\"hello\""`)
- Set summary to `"TODO: add summary"` if user skips or does not provide one
- Use ALL CAPS with NO underscores for `GEN_PASS_DEF_` macros (e.g., `MYOPTIMIZATION`)
- Use fully-qualified factory function name at global scope: `mlir::<namespace>::create<PassName>Pass()`

**MUST NOT**:
- Create test files (use separate skill)
- Fix infrastructure/build-environment issues
- Add comments to generated code (except TODO in runOnOperation)
- Modify files outside the 6 target files
- Install or modify environment if clang-format is unavailable

**NEVER**:
- Use "Pass" suffix in TableGen def name
- Use "Pass" suffix in CLI name
- Use "Pass" suffix in C++ filename for **new** passes (note: some legacy Asc dialect files like `DetectKernelTypePass.cpp` use the suffix — do not follow this pattern for new passes)
- Use "Pass" suffix in Python function name for **new** passes (note: legacy exceptions like `add_noop_pass` exist — do not follow this pattern)
- Proceed if namespace directory does not exist
- Terminate without build verification
- Guess or imagine pass summary — use user-provided text or `"TODO: add summary"`
- Add `let description = [{...}];` to TableGen skeleton (description is added when pass is implemented)
- Use underscores in `GEN_PASS_DEF_` macro (use ALL CAPS only, e.g., `MYOPTIMIZATION` not `MY_OPTIMIZATION`)
- Wrap factory function in `namespace mlir { namespace <namespace> { ... } }` blocks — use fully-qualified name at global scope instead (prevents hidden linking issues)

**IMPORTANT**:
- "Pass" suffix ONLY appears in: constructor function name, struct name
- Alphabetical sorting is by def name/function name/filename, NOT by CLI name
- Option default values must match across TableGen, header, and Pybind
- Pass name should be imperative form (verb + noun), but accept user preference
- `func::FuncOp` pass can only modify IR inside functions; use `ModuleOp` for function/module attributes
- Summary must come from user or be `"TODO: add summary"` — never generate it yourself
- You may rephrase user-provided summary for clarity and to match existing pass style

---

## Troubleshooting

**Build error: "undefined reference to createXxxPass"**:
- Check factory function exists in .cpp file
- Check declaration exists in .h file
- Verify namespace matches

**Build error: "GEN_PASS_DEF_XXX not defined"**:
- Check TableGen def name matches macro name (ALL CAPS, NO underscores)
- Example: `def UnrollLoop` → `GEN_PASS_DEF_UNROLLLOOP`
- Verify `Passes.h.inc` is generated (check build directory)

**Build error: "no matching function for call"**:
- Check option parameter types match across all files
- Verify default values are consistent

**Alphabetical order violation**:
- Re-sort entries by name (def name, function name, or filename)
- Do NOT sort by CLI name or description
