/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_ASCENDC_PASSES_TD
#define ASCTILE_ASCENDC_PASSES_TD

include "mlir/Pass/PassBase.td"

def AllocateTensor : Pass<"ascendc-allocate-tensor", "func::FuncOp"> {
  let summary = "Assign memory addresses to LocalTensorAutoOp placeholders for static UB allocation";
  let constructor = "mlir::ascendc::createAllocateTensorPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Converts `LocalTensorAutoOp` placeholders to `LocalTensorV3Op` with concrete memory addresses and tile sizes.
    This pass performs static memory allocation for on-chip buffers by assigning consecutive aligned addresses
    to tensors based on their normalized memory position (A1/L1, A2/L0A, B2/L0B, CO1/L0C, VECCALC/UB).

    Position normalization: B1→A1, VECIN/VECOUT→VECCALC. Addresses are aligned to `ubBlockSize` (256 bytes).
    After allocation, `LocalTensorAutoOp` is replaced with `LocalTensorV3Op` containing position, address, and tile size.
  }];
}

def ComputeMemoryConsumption : Pass<"ascendc-compute-memory-consumption", "ModuleOp"> {
  let summary = "Calculate total memory consumption per buffer location and attach to module";
  let constructor = "mlir::ascendc::createComputeMemoryConsumptionPass()";
  let description = [{
    Analyzes all `LocalTensorV3Op` operations and computes total memory usage for each hardware buffer location:
    L1, L0A, L0B, L0C, and UB. The results are stored as a dictionary attribute `memoryConsumed` on the module,
    with keys like "L1", "L0A", etc. and values in bytes.

    This pass is useful for verifying that kernel memory usage stays within hardware limits.
  }];
}

def ComputeReuseGroup : Pass<"ascendc-compute-reuse-group", "func::FuncOp"> {
  let summary = "Compute reuse group indices for unrolled loop iterations to guide tensor allocation reuse";
  let constructor = "mlir::ascendc::createComputeReuseGroupPass()";
  let description = [{
    Computes `ascendc.reuse_group` indices for operations in unrolled loops to guide downstream tensor
    allocation reuse. The pass analyzes `scf.execute_region` operations with `asctile.unroll_factor` and
    `asctile.unroll_iter` attributes, and remaps iteration indices into reuse groups.

    The reuse group is computed as `(unroll_iter + startIndex) % unrollFactor`, where `startIndex` accounts
    for nesting level and epilogue iterations. An epilogue iteration is detected when `unroll_iter` equals
    `unroll_factor` (i.e., the remainder-handling tail of an unrolled loop). Epilogue iterations shift the
    start index for subsequent loops at the same nesting level, ensuring correct reuse group alignment.

    After processing, `asctile.unroll_iter` attributes are replaced with `ascendc.reuse_group` on all
    operations within unrolled loops.
  }];
}

def DispatchAlloc : Pass<"ascendc-dispatch-alloc", "func::FuncOp"> {
  let summary = "Dispatch tensor allocation to static or TPipe-backed strategy based on target architecture";
  let constructor = "mlir::ascendc::createDispatchAllocPass()";
  let description = [{
    Selects between static allocation and TPipe-backed allocation for on-chip tensor buffers. The decision
    is based on the `asc.static_alloc` module attribute, target architecture, and presence of incompatible
    operations. When targeting C310 architectures without `BroadcastOp`, static allocation is preferred;
    otherwise TPipe-backed allocation is used.
  }];
}

def FillAscOperands : Pass<"ascendc-fill-asc-operands", "func::FuncOp"> {
  let summary = "Fill default cal_count, mask, repeat_times, and repeat_params operands for L0/L2 Ascend C operations";
  let constructor = "mlir::ascendc::createFillAscOperandsPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    Populates default operands for vector operations based on tensor shape:
    - **L0 operations**: Fills `mask`, `repeat_times`, and `repeat_params` (stride values)
    - **L2 operations**: Fills `cal_count` with number of elements

    Default stride values: block_stride=1, repeat_stride=8. Mask is computed based on element type size (1, 2, 4, 8).
    Repeat times calculated as ceil(numElements / numPerRepeat).
  }];
}

def FixupMmadAccParams : Pass<"ascendc-fixup-mmad-acc-params", "func::FuncOp"> {
  let summary = "Fix accumulator initialization for sequential mmad operations using the same destination tensor";
  let constructor = "mlir::ascendc::createFixupMmadAccParamsPass()";
  let dependentDialects = ["emitasc::EmitAscDialect", "memref::MemRefDialect"];
  let description = [{
    Handles matrix multiplication accumulation where multiple `mmad` operations write to the same destination tensor.
    Creates a runtime boolean variable to track whether this is the first mmad to the destination:
    - First mmad: `cmatrixInitVal=true` (initialize accumulator)
    - Subsequent mmads: `cmatrixInitVal=false` (accumulate into existing values)

    Only processes mmad operations that have `cmatrixInitVal` parameter in their mmad_params.

    Example transformation:
    ```mlir
    // Before: mmad with fixed cmatrixInitVal in loop
    %acc = ascendc.local_tensor_auto co1() : <16x16xf32>
    scf.for %i = 0 to %n {
      %params = emitasc.init_struct !ascendc.mmad_params(
        "cmatrixInitVal" = %c1  // always true (reinit each iteration)
      )
      ascendc.mmad %acc, %a, %b, %params
    }

    // After: runtime variable controls initialization
    %acc = ascendc.local_tensor_auto co1() : <16x16xf32>
    %var = emitasc.variable true, memref<1xi1>  // init flag
    scf.for %i = 0 to %n {
      %init_val = memref.load %var[%c0]
      %params = emitasc.init_struct !ascendc.mmad_params(
        "cmatrixInitVal" = %init_val  // true first iter, false after
      )
      ascendc.mmad %acc, %a, %b, %params
      %false = arith.constant false
      memref.store %false, %var[%c0]  // set false for next iter
    }
    ```
  }];
}

def FuseBufIdSync : Pass<"ascendc-fuse-bufid-sync", "func::FuncOp"> {
  let summary = "Remove redundant get_buf/rls_buf synchronization between consecutive operations with matching bufId";
  let constructor = "mlir::ascendc::createFuseBufIdSyncPass()";
  let description = [{
    Optimizes BufId-based synchronization by removing redundant `get_buf`/`rls_buf` operations:
    - For consecutive operations with the same `bufId` attribute and same pipeline type:
      - Remove `get_buf` before middle operations (only first op needs it)
      - Remove `rls_buf` after middle operations (only last op needs it)
    - Pipeline types tracked: PIPE_V (vector), PIPE_MTE2 (GM→UB load), PIPE_MTE3 (UB→GM store)

    After optimization, removes the `bufId` attribute from all operations. Used on C310 architecture.
  }];
}

def InsertBufIdSync : Pass<"ascendc-insert-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization around operations for BufId-based tracking on C310 arch";
  let constructor = "mlir::ascendc::createInsertBufIdSyncPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    Inserts BufId synchronization for Ascend hardware with C310 architecture using buffer ID tracking:
    1. Assigns unique `bufId` to each tensor allocation (`LocalTensorV3Op`, `TBufGetTensorOp`)
    2. For each operation using a tensor, inserts `get_buf` before and `rls_buf` after
    3. Tracks pipeline type per operation: PIPE_MTE1 (L1 load), PIPE_MTE2 (GM load), PIPE_MTE3 (GM store),
       PIPE_FIX (fixpipe), PIPE_S (scalar), PIPE_M (matmul), PIPE_V (vector)
    4. For `scf.for` loops (except `vec_scope_loop` or parallel), inserts MTE3-MTE2/MTE3-MTE1 sync at loop end

    Sets `bufId` array attribute on operations for downstream `FuseBufIdSync` optimization.
  }];
}

def InsertBiasBufIdSync : Pass<"ascendc-insert-bias-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization around MmadOp for bias tensors copied to BT";
  let constructor = "mlir::ascendc::createInsertBiasBufIdSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Inserts buffer synchronization (`get_buf`/`rls_buf`) around `MmadOp` operations that use bias tensors
    in C2 position (bias tensor memory).

    This pass ensures proper synchronization between bias tensor access and matrix multiplication operations
    on C310 architecture. The pass:
    1. Finds `LocalTensorV3Op` in C2 position with a `bufId` attribute
    2. Extracts the `bufId` value from this tensor
    3. For each `MmadOp` with `cmatrixSource` parameter:
       - Inserts `get_buf pipe_m, <bufId>` before the `MmadOp`
       - Inserts `rls_buf pipe_m, <bufId>` after the `MmadOp`

    **Important**: This pass assumes there is only one bias tensor with `bufId` per function. The `UnifyBiasTensor`
    pass consolidates all C2 tensors into a single tensor, ensuring only one `bufId` exists.
  }];
}

def InsertCrossCoreSync : Pass<"ascendc-insert-cross-core-sync", "func::FuncOp"> {
  let summary = "Insert CrossCoreSetFlag and CrossCoreWaitFlag synchronization between if_aic and if_aiv operations";
  let constructor = "mlir::ascendc::createInsertCrossCoreSyncPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    This pass inserts `ascendc.cross_core_set_flag` and `ascendc.cross_core_wait_flag` operations
    to synchronize data transfers between AIV (AI Vector) and AIC (AI Cube) cores. It is a no-op
    unless the function contains at least two `if_aic`/`if_aiv` core groups.

    The pass identifies sync-trigger operations that transfer data between cores
    through an on-chip local tensor:
    - `DataCopyOp` with direction VECCALC -> A1/B1 (UB -> L1, in AIV groups)
    - `FixpipeOp` with direction CO1 -> VECCALC (L0C -> UB, in AIC groups)

    For each sync-trigger op, the pass:
    1. Finds consumers of the trigger's destination tensor on the opposite core
       (only those appearing after the trigger in IR order, via function walk).
       If no consumers are found, no synchronization flags are inserted for this trigger.
    2. Allocates a flag ID (sequential, starting from 0 and wrapping at 16).
    3. Inserts a backward `CrossCoreWaitFlag` before the trigger if this is a second
       trigger for the same tensor (tensor reuse) or if inside a loop. This waits for
       the previous consumer's backward `CrossCoreSetFlag` (loop-carried dependency).
    4. Inserts a producer `CrossCoreSetFlag` after the trigger (forward direction),
       using the trigger op's pipe, only when consumers exist. Signals "data is ready"
       to consumers.
    5. Inserts consumer flags only in the **first** consumer group after the trigger
       (cross-core flags are binary semaphores -- one set, one wait):
       - A `CrossCoreWaitFlag` at the start of the consumer group's body: the consumer
         waits for the producer's "data ready" signal from step 4.
       - A `CrossCoreSetFlag` before the consumer group's terminator (backward direction):
         the consumer signals that the buffer can be safely overwritten.
       Both use the consumer op's pipe (determined by `getOpPipeExt(consumerOp)`).
       Consumer groups after the first receive no flags -- they rely on intra-core
       ordering, since the data is already loaded by the first group's wait.
    6. Seeds backward synchronization to prevent the producer from overwriting a shared
       buffer before the consumer finishes reading. This happens for a first trigger
       inside a loop, or for a second trigger whose enclosing loop differs from the
       one recorded previously:
       - An initial `CrossCoreSetFlag` before the loop (consumer's core, PIPE_S) --
         seeds the backward flag to prevent first-iteration deadlock.
       - A finalization `CrossCoreWaitFlag` after the loop (producer's core, producer's
         pipe) -- consumes the leftover flag.

    When the same destination tensor is reused by a different trigger op (second trigger),
    the pass recognizes it via a destination-to-flag map and reuses the same flag ID:
    the second trigger's `CrossCoreSetFlag` re-sets the forward flag (when it has
    consumers), allowing subsequent consumers to wait on it. This avoids exhausting
    the 16-flag hardware limit.
  }];
}

def InsertCrossCoreSyncGM : Pass<"ascendc-insert-cross-core-sync-gm", "func::FuncOp"> {
  let summary = "Insert cross-core synchronization for CO1->GM and VECCALC->GM data transfers";
  let constructor = "mlir::ascendc::createInsertCrossCoreSyncGMPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let description = [{
    This pass inserts `ascendc.cross_core_set_flag` and `ascendc.cross_core_wait_flag`
    operations to synchronize data transfers from on-chip buffers to Global Memory (GM)
    across AIC (producer) and AIV (consumer) cores. It handles two transfer types:
    `FixpipeOp` from L0C (CO1) to GM, and `DataCopyOp` from UB (VECCALC) to GM.

    GM transfers are synchronized independently of the sync-trigger mechanism handled by
    `InsertCrossCoreSync` and of loop-carried forward/backward sync. Walking the
    `if_aic`/`if_aiv` groups in pre-order, the pass:

    - Producer (the group containing the GM-writing op): inserts `CrossCoreSetFlag`
      operations at the end of the group (before its terminator), on the op's pipe
      (determined by `getOpPipeExt`), and records the root GlobalTensor -- traced back
      through `global_tensor.subindex` chains via `getRootGlobalTensor`:
      - CO1 -> GM (`FixpipeOp`): inserts two flags with ids `id` and `id + 16`.
      - VECCALC -> GM (`DataCopyOp`): inserts a single flag with id `id`.
      Flags are only emitted when a subsequent consumer group exists (a later group
      whose operand traces to the same root); otherwise the `CrossCoreSetFlag` and the
      flag ID allocation are skipped, since no `CrossCoreWaitFlag` would ever consume
      the flag.
    - Consumer (a later group whose operand traces, via `getRootGlobalTensor`, to a
      recorded root): inserts `CrossCoreWaitFlag` operations at the start of the group
      body (before any group operation), on `PIPE_S`:
      - For CO1 -> GM producers: waits on `id` only.
      - For VECCALC -> GM producers: waits on both `id` and `id + 16`.

    The flag pair asymmetry between transfer types reflects the different pipeline
    dependencies of each path. No loop-carried synchronization is inserted for the GM path.

    Flag ID allocation starts from the value stored in the `ascendc.cross_core_flag_id`
    attribute on the function (set by the preceding `InsertCrossCoreSync` pass, or 0
    if absent) and wraps at 16 (`maxTensorId`); the `id + 16` values occupy the
    upper flag bank (16-31). After processing, the pass removes the
    `ascendc.cross_core_flag_id` attribute, as flag ID allocation is complete.
  }];
}

def InsertInitDump : Pass<"ascendc-insert-init-dump", "func::FuncOp"> {
  let summary = "Insert InitDump call and dump_addr kernel argument for kernels containing debug operations";
  let constructor = "mlir::ascendc::createInsertInitDumpPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
  let description = [{
    Inserts AscendC debug-dump initialization for kernels that use debug operations. Walks the function for
    `ascendc.printf` or `ascendc.dump_tensor`; if any are found, appends a `dump_addr` kernel argument
    (`__gm__ uint8_t*`, `emitasc.kernel_arg` = `DumpAddr`) and inserts an `ascendc.init_dump` operation at
    the function entry. The operation is parameterized by the kernel `isMixed` flag (derived from the
    `asc.kernel_type` module attribute), the dump address, and a fixed dump size of 1 MB (1024*1024 bytes)
    per core.
  }];
}

def InsertSubBlockGuard : Pass<"ascendc-insert-subblock-guard", "func::FuncOp"> {
  let summary = "Insert early return guard for non-zero sub-block indices in mixed kernel functions";
  let constructor = "mlir::ascendc::createInsertSubBlockGuardPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Inserts a guard at the beginning of kernel functions that returns early if the current sub-block
    index is not zero. This is useful for multi-core kernels where only sub-block 0 should execute
    the main kernel logic.

    The pass only operates on functions within modules that have `asc.kernel_type = "mixed"` attribute.
    For other kernel types (vector, cube) or modules without the attribute, the pass is a no-op.

    The pass inserts the following C++ code at the function entry:
    ```cpp
    if (AscendC::GetSubBlockIdx() != 0) return;
    ```

    This ensures that only sub-block 0 executes the kernel body, while other sub-blocks return immediately.
    The pass is idempotent and skips functions that already have the guard inserted.
  }];
}

def LowerToL0 : Pass<"ascendc-lower-to-l0", "func::FuncOp"> {
  let summary = "Lower L2 vector operations to L0 intrinsics with default repeat parameters";
  let constructor = "mlir::ascendc::createLowerToL0Pass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
  let description = [{
    Converts L2-level vector operations to L0-level hardware intrinsics:
    - Unary L2→L0: `abs_l2`→`abs_l0`, `exp_l2`→`exp_l0`, etc. with mask, repeat_times=0, empty repeat_params
    - Binary L2→L0: `add_l2`→`add_l0`, `mul_l2`→`mul_l0`, etc. with similar parameters
    - Special ops: `duplicate_l2`→`duplicate_l0`, `cast_l2`→`cast_l0`, `compare_l2`→`compare_l0`
    - Scalar ops: `adds_l2`→`adds_l0`, `muls_l2`→`muls_l0`

    L0 operations provide direct hardware control but require explicit parameters; this pass provides defaults.
    L2 operations are simpler API with automatic parameter calculation (done by `FillAscOperands`).
  }];
}

def PromoteCVBlock : Pass<"ascendc-promote-cv-block", "ModuleOp"> {
  let summary = "Detect kernel type and promote conditional AIC/AIV blocks for pure cube or vector kernels";
  let constructor = "mlir::ascendc::createPromoteCVBlockPass()";
  let description = [{
    Inspects each function for `if_aic` (AI Cube) and `if_aiv` (AI Vector) conditional blocks and sets the
    module-level `asc.kernel_type` attribute to `cube`, `vector`, or `mixed` accordingly. For pure cube or
    vector kernels, unwraps the conditional blocks by moving their body operations into the parent block and
    erasing the conditional wrapper. Mixed kernels retain their conditional structure since operations target
    different compute units.
  }];
}

def RefineCubePosition : Pass<"ascendc-refine-cube-position", "func::FuncOp"> {
  let summary = "Refine A1 cube tensor positions to B1 or C1 based on L0 consumer destinations";
  let constructor = "mlir::ascendc::createRefineCubePositionPass()";
  let description = [{
    Refines the memory position of `LocalTensorAutoOp` tensors initially placed in A1 (L1) by analyzing how
    they are consumed by L0 load and copy operations (`LoadDataL0V2Op`, `DataCopyL0Op`).

    The pass inspects the destination tensor position of each consumer and remaps the source tensor position:
    - Destination in A2 (L0A) -> source stays in A1
    - Destination in B2 (L0B) -> source refined to B1
    - Destination in C2 (bias) -> source refined to C1

    Only tensors with position A1 are processed. This refinement ensures cube tensors are placed in the
    correct hardware buffer (L1, B1, or C1) matching the actual data flow to L0A, L0B, or bias memory.
  }];
}

def RemoveDebugOps : Pass<"ascendc-remove-debug-ops", "ModuleOp"> {
  let summary = "Remove debug operations (PrintfOp, DumpTensorOp, AssertOp) when debug mode is disabled";
  let constructor = "mlir::ascendc::createRemoveDebugOpsPass()";
  let description = [{
    Removes `ascendc.printf`, `asctile.dump_tensor`, and `asctile.assert` operations from the module.
    This pass should run immediately after codegen to remove debug ops when the user has not explicitly
    enabled debug mode.
  }];
}

def ReuseTensorAllocation : Pass<"ascendc-reuse-tensor-allocation", "func::FuncOp"> {
  let summary = "Reuse freed on-chip allocations based on unroll_iter lifetime analysis";
  let constructor = "mlir::ascendc::createReuseTensorAllocationPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Reuses freed on-chip buffer allocations by inserting `LocalTensorReinterpretCastOp`.
    Covers VECCALC (UB), A1 (L1), A2 (L0A), B2 (L0B), and CO1 (L0C) positions.
    Reuse is restricted to tensors of the same position, since each position maps to
    a distinct hardware buffer.

    Reuse decisions are based on `asctile.unroll_iter` attribute: if top tensor's max
    unroll_iter is less than bottom tensor's min unroll_iter, their lifetimes don't
    overlap and reuse is valid.

    Input/output tensors cannot be reused across different unroll_iters because each
    iteration loads/stores different data. Reusing would cause data races. Input/output
    tensors can only be reused with tensors that have the same unroll_iter or no
    unroll_iter at all.

    For in/out tensors: all users share the same unroll_iter (guaranteed by upstream passes),
    so max == min. Two in/out tensors at different iterations can reuse each other.
    Two in/out tensors at the same iteration overlap and cannot reuse.

    For tensors without unroll_iter: fall back to position-based ordering within their scope.

    For cube positions (A1, A2, B2), allocation size is computed with cube-block alignment
    to match the downstream AllocateTensor pass.

    When reusing: larger tensor wraps smaller via reinterpret_cast, or smaller moved before larger and cast.
  }];
}

def ReuseUBAllocation : Pass<"ascendc-reuse-ub-allocation", "func::FuncOp"> {
  let summary = "Reuse freed UB allocations via reinterpret_cast to reduce memory consumption";
  let constructor = "mlir::ascendc::createReuseUBAllocationPass()";
  let options = [
    Option<"reuseInOut", "reuse-in-out", "bool", "false",
           "Allow reuse between input/output tensors even in same loop">,
  ];
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Analyzes tensor lifetimes and reuses freed UB allocations by inserting `LocalTensorReinterpretCastOp`:
    1. For each block, collects `LocalTensorAutoOp` tensors and their last users
    2. Sorts tensors by last user position (earlier-freed tensors can be reused first)
    3. Checks reusability: same block, non-overlapping lifetimes, static shape, VECCALC position
    4. When reusing: larger tensor wraps smaller via reinterpret_cast, or smaller moved before larger and cast

    Special handling for control flow: tensors used across different regions (if/else, while before/after) can be reused.
    Input/output tensors in same loop are not reused unless `reuse-in-out=true`.

    Example: two tensors with non-overlapping lifetimes reused:
    ```mlir
    // Before: separate allocations for sequential tensors
    %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
    ascendc.data_copy_l2 %0, %gm, %c1
    %1 = ascendc.local_tensor_auto veccalc() : <333xi32>
    ascendc.data_copy_l2 %1, %gm, %c1

    // After: single allocation with reinterpret_cast
    %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<333xi32> to !ascendc.local_tensor<333xi32>
    ascendc.data_copy_l2 %0, %gm, %c1
    ascendc.data_copy_l2 %1, %gm, %c1
    ```

    Example: tensors in different loops can be reused:
    ```mlir
    // Before: output tensor in loop 1, input tensor in loop 2
    scf.for { %output = ascendc.local_tensor_auto veccalc() output }
    scf.for { %input = ascendc.local_tensor_auto veccalc() input }

    // After: output reused as input across different loops
    %output = ascendc.local_tensor_auto veccalc() output
    %input = ascendc.reinterpret_cast %output
    scf.for { use %output }
    scf.for { use %input }
    ```
  }];
}

def UnifyBiasTensor : Pass<"ascendc-unify-bias-tensor", "func::FuncOp"> {
  let summary = "Unify C2 (bias) tensor with identical type and size, setting addr to 0";
  let constructor = "mlir::ascendc::createUnifyBiasTensorPass()";
  let description = [{
    Consolidates all `LocalTensorV3Op` operations in C2 position (bias tensor memory) into a single tensor.

    When multiple C2 tensors exist (typically created during loop unrolling or tiling), this pass:
    1. Keeps the first encountered C2 tensor
    2. Replaces all uses of subsequent C2 tensors with the first tensor
    3. Erases the duplicate tensor operations
    4. Sets the address of the unified tensor to 0

    This optimization ensures a single bias tensor allocation per function, which is required for
    correct `bufId`-based synchronization in `InsertBiasBufIdSync` pass.
  }];
}

#endif // ASCTILE_ASCENDC_PASSES_TD
