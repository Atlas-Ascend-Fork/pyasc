/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF_PASSES_TD
#define ASCVF_PASSES_TD

include "mlir/Pass/PassBase.td"

def DispatchVFFusion : Pass<"ascvf-dispatch-vf-fusion", "ModuleOp"> {
  let summary = "Inserts LocalMemBarOp synchronization barriers at required points within VecScopeOp regions";
  let description = [{
    Inserts LocalMemBarOp synchronization barriers at the points within VecScopeOp
    regions where store-to-load dependencies must be preserved. Determines those
    points by marking every operation with a BarrierOp, cloning the function,
    running the vector fusion optimization pipeline and EraseBarrier on the clone,
    and mapping the barriers that survive back to LocalMemBarOp operations in the
    original function before running the same pipeline on it.
  }];
  let constructor = "mlir::ascvf::createDispatchVFFusionPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "ascvf::AscVFDialect"];
}

def EraseBarrier : Pass<"ascvf-erase-barrier", "func::FuncOp"> {
  let summary = "Erases redundant BarrierOp operations within VecScopeOp that protect no pending stores";
  let description = [{
    Erases redundant BarrierOp operations within VecScopeOp regions by sorting
    LoadOp, StoreOp, and BarrierOp operations by their relative position,
    collapsing consecutive barriers to the last one, and removing barriers with
    no preceding active stores. Emits a warning when a LoadOp encounters
    unsynchronized stores.
  }];
  let constructor = "mlir::ascvf::createEraseBarrierPass()";
}

def EliminateCommonMask : Pass<"ascvf-eliminate-common-mask", "func::FuncOp"> {
  let summary = "Eliminates duplicate mask operations within VFGroupOp";
  let description = [{
    Removes redundant UpdateMaskOp and CreateMaskOp operations inside VFGroupOp
    regions by replacing uses of duplicate masks with the first occurrence and
    deleting the duplicates.
  }];
  let constructor = "mlir::ascvf::createEliminateCommonMaskPass()";
}

def EliminateDataTransfer : Pass<"ascvf-eliminate-data-transfer", "func::FuncOp"> {
  let summary = "Eliminates redundant load/store operations in VFGroupOp";
  let description = [{
    Performs three optimizations: eliminates loads after stores by replacing
    the loaded value with the stored value, removes stores that are immediately
    overwritten, and merges repeated loads from the same memory location.
  }];
  let constructor = "mlir::ascvf::createEliminateDataTransferPass()";
}

def FindVFGroup : Pass<"ascvf-find-vf-group", "func::FuncOp"> {
  let summary = "Finds and groups L2 operations into VFGroupOp for vector fusion";
  let description = [{
    Identifies groups of fusible L2 operations (binary, unary, reduce, duplicate)
    that share the same calCount and element type, and wraps them into VFGroupOp
    operations for subsequent vector fusion optimization.
  }];
  let constructor = "mlir::ascvf::createFindVFGroupPass()";
  let dependentDialects = ["ascvf::AscVFDialect"];
}

def FuseVFFor : Pass<"ascvf-fuse-vf-for", "func::FuncOp"> {
  let summary = "Fuses consecutive VFForOp loops within VecScopeOp";
  let description = [{
    Merges adjacent VFForOp loops into a single loop by copying operations from
    subsequent loops into the first loop, reducing loop overhead.
  }];
  let constructor = "mlir::ascvf::createFuseVFForPass()";
}

def InlineVFGroup : Pass<"ascvf-inline-vf-group", "func::FuncOp"> {
  let summary = "Replaces ascvf.vf_group operation with its body";
  let constructor = "mlir::ascvf::createInlineVFGroupPass()";
}

def LowerToReg : Pass<"ascvf-lower-to-reg", "func::FuncOp"> {
  let summary = "Lowers L2 operations to register-level operations with loops";
  let description = [{
    Converts L2 operations (binary, unary, reduce, duplicate) into micro-level
    operations by wrapping them in VecScopeOp, creating VFForOp loops, allocating
    register tensors, and inserting load/store micro operations.
  }];
  let constructor = "mlir::ascvf::createLowerToRegPass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascendc::AscendCDialect", "ascvf::AscVFDialect", "emitasc::EmitAscDialect",
    "scf::SCFDialect"
  ];
}

def MaterializeLoadStore : Pass<"ascvf-materialize-load-store", "func::FuncOp"> {
  let summary = "Materializes abstract load/store micro ops into data copy operations";
  let description = [{
    Converts LoadOp and StoreOp operations into concrete data copy
    operations with physical addresses, creating LocalTensorGetPhyAddrV2Op and
    DataCopyLoadOp/DataCopyStoreOp operations.
  }];
  let constructor = "mlir::ascvf::createMaterializeLoadStorePass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
}

def PromoteCastOps : Pass<"ascvf-hoist-cast", "func::FuncOp"> {
  let summary = "Hoists LocalTensorReinterpretCastOp operations immediately after their defining operations";
  let description = [{
    Relocates each LocalTensorReinterpretCastOp to immediately after the operation
    that defines its input tensor, placing the cast right after its source.
  }];
  let constructor = "mlir::ascvf::createPromoteCastOpsPass()";
}

def ReorderOpsInVecScope : Pass<"ascvf-reorder-ops-in-vec-scope", "func::FuncOp"> {
  let summary = "Hoists and reorders operations within VecScopeOp blocks";
  let description = [{
    Reorders operations inside VecScopeOp blocks by moving constants, register
    tensors, variables, mask operations, and duplicate operations to the
    beginning in a specific order for better code generation.
  }];
  let constructor = "mlir::ascvf::createReorderOpsInVecScopePass()";
}

#endif // ASCVF_PASSES_TD
