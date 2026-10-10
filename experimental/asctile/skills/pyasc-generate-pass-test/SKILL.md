---
name: pyasc-generate-pass-test
description: Generate tests for MLIR pass (write from scratch or append to existing)
---

## PyAsc: Generate LIT Tests for MLIR Pass

This skill helps AI agents automatically generate comprehensive `lit` tests for MLIR passes in the PyAsc codebase. The agent analyzes the pass implementation and creates test cases covering all transformation scenarios.

## When to Use

Use this skill when:
- A developer provides a link to an MLIR pass implementation file and asks to write tests
- You need to generate lit tests for a new or modified MLIR pass
- Test coverage for a pass needs to be expanded

## Input Requirements

The developer must provide:
1. **Path to MLIR pass file**: e.g., `lib/Dialect/<DialectName>/Transforms/UnrollLoop.cpp`
2. **Pass name**: e.g., `UnrollLoop` (can be extracted from the `.cpp` file or `.td` definition)
3. **Optional**: Specific test scenarios or edge cases to prioritize

## Agent Constraints

**CRITICAL**: The agent MUST NOT modify any files except test files:
- **NEVER** modify MLIR pass implementation files (.cpp, .h)
- **NEVER** modify TableGen definition files (.td)
- **NEVER** modify CMakeLists.txt or build configuration files
- **NEVER** modify test cases that existed before this session (only create new ones)
- **ONLY** create new test files (.mlir) in appropriate test directories

The agent's role is to:
1. **Read** existing code for analysis (pass files, .td files, existing tests)
2. **Generate** new test files based on analysis
3. **Validate** generated tests by running them
4. **Report** test coverage and any issues found

## Agent Pipeline

### Phase 0: Setup Environment and Build Project

**Goal**: Ensure required tools are available.

**IMPORTANT**: Before generating any tests, verify that `ascir-opt`, `FileCheck`, and `lit` are available.

**Steps**:

1. **Check for required binaries**
   ```bash
   which ascir-opt
   which FileCheck
   which lit
   ```
   - If `ascir-opt` or `FileCheck` are not available, proceed to step 2
   - If `lit` is not available, proceed to step 3

2. **Build the project if needed**
   - If `ascir-opt` or `FileCheck` are not available, invoke the `pyasc-check-env-and-build` skill
   - This skill will set up the environment and build the project
   - After building, verify binaries are available

3. **Install lit if needed**
   - If `lit` is not available, ask the user for permission to install it
   - Ask: "The `lit` test runner is not installed. May I install it with `pip3 install lit`?"
   - **ONLY** proceed with installation after explicit user agreement
   - Install command: `pip3 install lit`

4. **Verify setup**
   ```bash
   ascir-opt --help
   FileCheck --version
   lit --version
   ```

### Phase 1: Analyze Pass Implementation

**Goal**: Understand the pass transformation logic and requirements.

**Steps**:

1. **Read the pass's TableGen definition**
   - Search in `include/ascir/Dialect/<DialectName>/Transforms/Passes.td`
   - Find the CLI pass name (the first argument to `Pass<"...", ...>`)
   - Extract pass options and their types
   - Note any dependencies on other passes

2. **Read the pass source file**
   - Identify the main pass class and its `runOnOperation()` method
   - Extract transformation patterns (pattern rewrites, walkers, etc.)
   - Note attributes used (e.g., `unrollFactor`)
   - Identify operation types targeted (e.g., `scf::ForOp`, `asctile::LoadOp`)
   - Check for helper functions and utility classes

3. **Identify transformation categories**
   - **Pattern-based rewrites**: Uses `OpRewritePattern`, `RewritePatternSet`
   - **Walk-based transformations**: Uses `op.walk()` or recursive visitors
   - **Attribute-based logic**: Reads/writes operation attributes
   - **Control flow changes**: Creates/destroys loops, blocks, regions

4. **Extract key information**
   - Pass name and dialect (e.g., `asctile::UnrollLoopPass`)
   - CLI pass name for RUN lines (e.g., `-asctile-unroll-loop`)
   - Input operation types (what ops are transformed)
   - Output operation types (what ops are created)
   - Required attributes and their types
   - Pass options/parameters
   - Dependencies on other passes

5. **Check for pass dependencies**
   - Look at pass options/attributes it reads that other passes set
   - Check operation types it expects that other passes produce
   - Review comments or documentation mentioning dependencies
   - Determine if preceding passes are needed in RUN lines

### Phase 2: Determine Test Coverage Requirements

**Goal**: Identify all test cases needed for comprehensive coverage.

**Heuristic**: Aim for at least one test per transformation pattern and one per edge case category.

**Steps**:

1. **Basic functionality tests**
   - Happy path: minimal working example
   - Multiple operations in sequence
   - Nested structures (loops within loops, etc.)

2. **Edge cases**
   - Empty operations/blocks
   - Single iteration cases
   - Boundary values (0, 1, max values)
   - Missing optional attributes
   - Invalid attribute values

3. **Attribute variations**
   - Different attribute values (e.g., `unrollFactor = 2, 4, 8`)
   - Missing attributes
   - Multiple attributes interacting

4. **Operation type coverage**
   - All supported operation types
   - Mixed operation types

5. **Control flow scenarios**
   - Single basic block
   - Multiple basic blocks
   - Nested regions
   - Dominance relationships

6. **Error handling**
   - Invalid IR structure
   - Type mismatches
   - Missing operands
   - Cyclic dependencies

### Phase 3: Locate Test Directory Structure

**Goal**: Find where tests should be placed.

**Steps**:

1. **Map pass to test directory**
   - AscTile passes → `test/Dialect/AscTile/Transforms/`
   - AscendC passes → `test/Dialect/AscendC/Transforms/`
   - Lowering passes → `test/Conversion/LowerToAsc/`
   - Other dialects follow the same pattern: `test/Dialect/<Name>/Transforms/`

2. **Check existing tests**
   - List existing test files for the pass
   - Analyze naming conventions (kebab-case)
   - Identify test patterns used
   - Check if tests include copyright headers (follow the pattern of the majority)

3. **Determine test filename**
   - Use pass name in kebab-case without dialect name prefix:
   **Example**:
    ```td
    def UnrollLoop : Pass<"asctile-unroll-loop", "func::FuncOp"> {
       let summary = "Unroll loops by unroll_factor";
       let constructor = "mlir::asctile::createUnrollLoopPass()";
    }
    ```
    Filename for test: `unroll-loop.mlir`
    - Add suffix if multiple files: `unroll-loop-basic.mlir`, `unroll-loop-edge-cases.mlir`
    - Prefer a single test file. Only split into multiple files if the single file becomes unmanageably long

### Phase 4: Generate Test Cases

**Goal**: Create lit test files with proper structure.

#### Pre-Generation Validation

**CRITICAL**: Before generating any MLIR operation, the agent MUST:

1. **Read the TableGen definition** for the operation from .td files
   - Search in `include/ascir/Dialect/DialectName/IR/` directory
   - Find the exact operation class definition
   - Extract: operands (with types), attributes (with types), constraints
   - **ALWAYS** study the `assemblyFormat` property to understand how operands and attributes are printed - some attributes may appear among operands in the printed format, not just in `{attrName = value : type}` syntax

2. **Find existing examples** in test files
   - Search in `test/Target/AscendC/` directory
   - Look for similar operations in .mlir files
   - Extract the exact syntax pattern

3. **Validate attribute types** before generating
   - Check if attribute is I32EnumAttr, I64EnumAttr, UnitAttr, etc.
   - Use the correct type (i32 vs i64) based on definition
   - **NEVER** guess attribute types

4. **Distinguish operands from attributes**
   - Operands are in `arguments = (ins ...)` list
   - Attributes have `$attrName` with specific types (UnitAttr, I32EnumAttr, etc.)
   - Count expected operands vs attributes separately

#### Static Validation Rules

1. **NEVER** guess operation signatures - **ALWAYS** read from .td files first
2. **ALWAYS** check enum constraints - enum attributes have specific allowed values
3. **ALWAYS** verify operand count - count operands separately from attributes
4. **ALWAYS** match existing test patterns - use exact syntax from working examples

#### Generation Steps

1. **Create input IR**
   - Use proper MLIR syntax for the dialect
   - Include necessary function signatures
   - Add attributes required by the pass
   - Base operation sequences on existing test files and the pass implementation's expected input patterns
   - Use `snake_case` for test function names, prefixed with a descriptive label (e.g., `@no_fold_unsupported_cast`, `@unroll_dynamic_loop`)
   - Name constants descriptively: `%c0` for `arith.constant 0 : index`, `%c0_i32` for `arith.constant 0 : i32`, `%c256_i32` for `arith.constant 256 : i32`
   - When creating `scf.for` loops, include `scf.yield` as the last operation in the loop body

2. **Run the pass to see actual output**
   - **CRITICAL**: **ALWAYS** run the pass on test input to see actual output before writing CHECK directives
   - Command: `ascir-opt -pass-name <your-test-file>.mlir`
   - This reveals:
     - Exact operation formats (operands, attributes)
     - Intervening operations (constants, etc.)
     - Whether CHECK or CHECK-NEXT is appropriate
   - Use actual output to write CHECK directives, not assumptions

3. **Create expected output IR**
   - Use the actual pass output from step 2 to write CHECK directives
   - Use `CHECK` directives for verification
   - Check operation order and structure
   - Verify attributes are added/removed correctly

4. **Add RUN directives**
   - Primary test: `-pass-name`
   - With pass options: `-pass-name=option_value` or `-pass-name="option1=value1 option2=value2"`
   - With other passes if needed: `-pass1 -pass2`
   - With simplification passes when output is verbose: `-canonicalize` or `-cse`
   - With debug flags if useful: `-debug`
   - For isolated test cases: `-split-input-file` (use `// -----` separator, comment + space + 5 dashes)
   - For multi-variant testing: use `--check-prefixes=CHECK,VARIANT1` and `--check-prefixes=CHECK,VARIANT2`

5. **Group related tests**
   - Basic tests first
   - Edge cases next
   - Complex scenarios last
   - Separate groups with a blank line
   - **NEVER** add descriptive comment headers between test functions

#### Test File Template

```mlir
// RUN: ascir-opt -pass-name %s | FileCheck %s

// CHECK-LABEL: func.func @test_case_name(%arg0: type1, %arg1: type2) -> return_type {
// CHECK: expected_output_line_1
// CHECK-NEXT: expected_output_line_2
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_case_name(%arg0: type1, %arg1: type2) -> return_type {
  %c0 = arith.constant 0 : i32
  // input operations
  return
}
```

#### Complete Working Example

```mlir
// RUN: ascir-opt -asctile-fold-cast %s | FileCheck %s

// CHECK-LABEL: func.func @fold_cast_i8_to_i32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_cast_i8_to_i32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xi16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}
```

#### Critical Formatting Rules

- **ALWAYS** include the **full function signature** (parameters and return types) in CHECK-LABEL
- **ALWAYS** use `%arg0`, `%arg1`, etc. for function arguments (**NEVER** use custom names like `%dst`, `%pipe`, `%cst`)
- **NEVER** add descriptive comments between test functions
- Use full lines in CHECK/CHECK-NEXT (avoid `[[...]]` or `{{...}}` regex patterns)
- **Exception**: Use `[[...]]` or `{{...}}` for very long, repeating type signatures (e.g., matmul types with many attributes)

#### Simplifying Output with -canonicalize

**IMPORTANT**: When a pass generates many constants or similar operations, add `-canonicalize` or `-cse` to simplify output:
```bash
ascir-opt -pass-name -canonicalize input.mlir
ascir-opt -pass-name -canonicalize -cse input.mlir
```
This reduces the number of CHECK lines needed and makes tests more maintainable.

Example:
```mlir
// RUN: ascir-opt -pass-name -canonicalize %s | FileCheck %s
```

### Phase 5: Verify and Iterate

**Goal**: Ensure tests are correct and complete.

**CRITICAL**: You MUST run tests and fix any failures before considering the task complete.

**Steps**:

1. **Check syntax**
   - Verify MLIR syntax is correct
   - Ensure all types are valid
   - Check attribute formats

2. **Verify Test Logic**
   - Each test should have a clear purpose
   - Expected output should match pass behavior
   - CHECK directives should be precise

3. **Run Tests**
   - **ALWAYS** execute: `lit -v /path/to/test.mlir`
   - Run from the project root (where `test/lit.cfg` is discoverable)
   - Or use full path: `lit -v /full/path/to/test/` to run all tests in a directory
   - **NEVER** assume tests pass without running them
   - Fix any failures by comparing with actual pass output
   - Ensure all tests pass

4. **Coverage Check**
   - Verify all transformation paths are tested
   - Check edge cases are covered
   - Ensure error cases are tested (if applicable)

5. **Iterative Improvement**
   - Extract common patterns from errors and create reusable rules
   - Apply fixes to all similar cases in the newly created test file (don't fix one-by-one)
   - Document learned rules for future sessions

## Appendix: Test Structure Reference

### CHECK Directive Rules

**CRITICAL**: After CHECK-LABEL, use CHECK (not CHECK-NEXT) if constants or other operations appear before the first transformed operation.

1. **CHECK-LABEL**: **ALWAYS** include full function signature
2. **CHECK**: Use when there may be intervening lines (constants, other ops)
3. **CHECK-NEXT**: Use **ONLY** for truly consecutive operations
4. **CHECK-NOT**: Verify operations are NOT present
5. **CHECK-SAME**: Match on the same line as previous CHECK - improves readability for long signatures or when a pass modifies function arguments

**Every test function must end with**:
```mlir
// CHECK-NEXT: return
// CHECK-NEXT:}
```

### Testing Different Pass Types

#### Module-Level Passes
For passes that operate on the entire module:
```mlir
// RUN: ascir-opt -pass-name -split-input-file %s | FileCheck %s

// CHECK-LABEL: module attributes {asc.kernel_type = "vector"} {
// CHECK-NEXT: func.func @test(...)
module {
  func.func @test(...) { ... }
}

// -----

// CHECK-LABEL: module {
// Next test case...
```

#### Verification-Only Passes
For passes that verify but don't transform:
```mlir
// RUN: ascir-opt -pass-name -split-input-file -verify-diagnostics %s

func.func @valid_case(...) {
  // Valid IR - should pass
}

// -----

func.func @invalid_case(...) {
  // expected-error@below {{error message}}
  invalid_operation
}
```

**Diagnostic testing syntax**: `expected-error@+N`, `expected-warning@+N`, `expected-remark@+N`, `expected-note@+N` where N is the line offset.
If N is 1, then `below` should be used instead of `+1`.

#### Fold/Canonicalize Tests
For testing fold methods and canonicalize patterns:
```mlir
// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @fold_identity_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_identity_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
```

#### Using CHECK-SAME for Readability
Use CHECK-SAME to improve readability when matching long function signatures or when a pass adds attributes to function arguments:
```mlir
// CHECK-LABEL: func.func @test_kernel(
// CHECK-SAME: %arg0: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK-SAME: %arg1: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
func.func @test_kernel(%arg0: memref<*xf32, 22>, %arg1: memref<*xf32, 22>) { ... }
```

**IMPORTANT**: CHECK-SAME matches on the same line as the previous CHECK, making it easier to verify specific parts of long signatures without repeating the entire line.

### Private Function Declarations
For `func.func private`, parameter names don't appear in output:
```mlir
// CHECK-LABEL: func.func private @test_declaration(i32) -> i32
func.func private @test_declaration(%arg0: i32) -> i32
```
**IMPORTANT**: Use `(i32)` not `(%arg0: i32)` in the CHECK-LABEL.

### Special Patterns for AscendC Buffer Operations
- For `ascendc.get_buf` and `ascendc.rls_buf`, include the full format with pipe type and buffer number: `pipe_<type>, <number>`
- Example: `ascendc.get_buf pipe_v, 0`, `ascendc.rls_buf pipe_mte2, 1`
- Common pipe types: `pipe_v`, `pipe_m`, `pipe_mte2`, `pipe_mte3`, `pipe_mte1`, `pipe_fix`, `pipe_s`
- Buffer numbers are assigned by the pass - verify from actual output

## Common Error Patterns and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CHECK-NEXT: is not on the line after the previous match` | Intervening line (constant, tbuf.get_tensor, etc.) between operations | Change `CHECK-NEXT:` to `CHECK:` before the first affected operation |
| `expected CHECK not found` | Operation format doesn't match actual output | Run pass to see actual format, update CHECK directive |
| Missing buffer number | Pass assigns buffer IDs dynamically | Run pass, extract actual buffer numbers from output |
| Wrong pipe type | Different operations use different pipes | Verify pipe type from actual output (pipe_v, pipe_mte2, etc.) |
| MLIR syntax errors | Invalid type syntax, wrong attribute format, missing operands | Check existing tests, verify against TableGen definitions, check `assemblyFormat` in .td |
| Pass doesn't transform | Missing attributes, wrong operation types, unmet conditions | Check required attributes, verify operation types match pass expectations, ensure pass conditions are met (dominance, purity, location constraints) |
| Private function declaration fails | Parameter names in CHECK-LABEL | Use `(i32)` not `(%arg0: i32)` in CHECK-LABEL |

**After fixing test failures**:
- Update this SKILL.md with new learned patterns if the pattern has complex cases
- Apply the same fix pattern to all similar test functions in the newly created file

## Troubleshooting

**Test fails with "expected CHECK not found"**:
- Run the pass manually to see actual output: `ascir-opt -pass-name test.mlir`
- Check if pass depends on other passes to run first
- Use `-debug` flag to see pass internals
- Verify operation format matches actual output exactly

**Test fails with "CHECK-NEXT: is not on the line after the previous match"**:
- There's an intervening line between operations (e.g., constant declaration)
- Change `CHECK-NEXT:` to `CHECK:` before the first affected operation
- Run the pass to see what operations appear between the expected lines

**MLIR syntax errors**:
- Verify type syntax matches dialect (check existing tests)
- Check attribute formats against TableGen definitions
- Ensure all operands are defined before use
- Verify operation has correct number of operands (check `assemblyFormat` in .td)

**Pass doesn't transform**:
- Check if required attributes are present
- Verify operation types match pass expectations
- Ensure pass conditions are met (e.g., dominance, purity, location constraints)
- Check if operation is in the correct dialect/location

**Private function declaration fails**:
- Parameter names don't appear in output for `func.func private`
- Use `(i32, ...)` not `(%arg0: i32, ...)` in CHECK-LABEL
