/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF2_PASSES_TD
#define ASCVF2_PASSES_TD

include "mlir/Pass/PassBase.td"

def AssignVFGroupTensors : Pass<"ascvf2-assign-vf-group-tensors", "func::FuncOp"> {
  let summary = "Assign local tensors to AscVF2 group operand segments";
  let description = [{
    Assigns local tensors used by a restored, result-free AscVF2 group to its input, output, and temporary operand
    segments. Inputs are read by the group, outputs are written and remain used after it, and temporaries are written
    without external uses. The pass runs after register lowering, inlining, and VF-group legalization.
  }];
  let constructor = "mlir::ascvf2::createAssignVFGroupTensorsPass()";
}

def BufferizeLoopResult : Pass<"ascvf2-bufferize-loop-result", "func::FuncOp"> {
  let summary = "Reuses register-tensor loop-carried buffers where lifetimes permit";
  let description = [{
    Replaces a loop-carried ascendc.reg_tensor destination with its initial register tensor when the initial value is
    not live after the loop and the yielded register tensor has a single destination use tied to the loop iter-arg. The
    rewrite makes the yield equal to the init operand so SCF canonicalization can remove the forwarded result.
  }];
  let constructor = "mlir::ascvf2::createBufferizeLoopResultPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def DropDeadWrite : Pass<"ascvf2-drop-dead-write", "func::FuncOp"> {
  let summary = "Remove exact vector writes overwritten before observation";
  let description = [{
    Removes a vector.transfer_write when a later write with the same tensor root, indices, permutation map, vector type,
    in-bounds information, and mask occurs before any read or memory side effect can observe it. The pass handles exact
    overwrites inside individual blocks, including pure masked regions such as vector.mask, and conservatively preserves
    writes observed by reads or separated by barriers and unknown memory effects.
  }];
  let constructor = "mlir::ascvf2::createDropDeadWritePass()";
}

def ExpandSimpleMaskReg : Pass<"ascvf2-expand-simple-mask-reg", "func::FuncOp"> {
  let summary = "Materialize AscVF2 simple register masks";
  let description = [{
    Replaces ephemeral ascvf2.simple_mask_reg operations with the final AscendC mask materialization. Static counts that
    match a supported hardware pattern become create_mask; other counts become constant-index, variable, and update_mask
    chains. Dynamic counts become variable and update_mask chains.
  }];
  let constructor = "mlir::ascvf2::createExpandSimpleMaskRegPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
}

def FuseVFLoops : Pass<"ascvf2-fuse-vf-loops", "func::FuncOp"> {
  let summary = "Fuse consecutive compatible loops in AscVF2 groups";
  let description = [{
    Merges adjacent affine.for operations with identical iteration domains and compatible loop-nest structure. The pass
    tracks vector transfer memory accesses through AscVF2 tensor aliases, requires equivalent affine indices,
    permutation maps, and masks for shared buffers, and preserves all loop-carried results. It intentionally rejects
    cross-iteration hazards, including writes repeated by an inner loop, and loops whose second body consumes the first
    loop's results.
  }];
  let constructor = "mlir::ascvf2::createFuseVFLoopsPass()";
  let dependentDialects = ["affine::AffineDialect"];
}

def HoistRegTensor : Pass<"ascvf2-hoist-reg-tensor", "func::FuncOp"> {
  let summary = "Hoist register tensors to the start of vector functions";
  let description = [{
    Moves operand-free ascendc.reg_tensor operations to the beginning of a function marked ascvf2.vector_func. This
    includes allocations nested in control flow and is safe because the allocated register tensor has no data
    dependencies.
  }];
  let constructor = "mlir::ascvf2::createHoistRegTensorPass()";
}

def InsertVFSync : Pass<"ascvf2-insert-vf-sync", "func::FuncOp"> {
  let summary = "Insert AscVF2 memory synchronization barriers";
  let description = [{
    Scans each AscVF2 group independently in operation order and inserts a barrier before a vector load that may read a
    tensor root previously written by a vector store. Barriers are placed before the enclosing top-level group operation
    when the hazard crosses loop boundaries, and directly before the load when producer and consumer share a top-level
    operation. Existing barriers make the pass idempotent.
  }];
  let constructor = "mlir::ascvf2::createInsertVFSyncPass()";
  let dependentDialects = ["ascvf2::AscVF2Dialect"];
}

def LegalizeVFGroup : Pass<"ascvf2-legalize-vf-group", "func::FuncOp"> {
  let summary = "Legalizes restored VF-group tensor boundaries";
  let description = [{
    Converts the results of an ascvf2.vf_group restored by inlining to their AscendC local-tensor counterparts.
    Zero-operand local tensor allocations are hoisted before the group in their original order. The pass runs after
    outlined vector-function lowering and inlining, and does not lower the group body.
  }];
  let constructor = "mlir::ascvf2::createLegalizeVFGroupPass()";
}

def LowerToReg2 : Pass<"ascvf2-lower-to-reg", "func::FuncOp"> {
  let summary = "Lower an outlined AscVF2 function to AscendC register operations";
  let description = [{
    Experimental combined register lowering for functions marked ascvf2.vector_func. It installs the AscVF2, vector,
    arith, math, SCF structural conversion patterns in one pattern set and applies them with a single conversion driver.
  }];
  let constructor = "mlir::ascvf2::createLowerToRegPass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascendc::AscendCDialect", "ascvf2::AscVF2Dialect", "emitasc::EmitAscDialect",
    "scf::SCFDialect"
  ];
}

def MergeIntoTransfer : Pass<"ascvf2-merge-into-transfer", "func::FuncOp"> {
  let summary = "Merge scalar transfer reads into broadcast transfers";
  let description = [{
    Folds an adjacent unmasked vector.transfer_read of vector<1> followed by a vector.broadcast into a single
    vector.transfer_read whose affine permutation map broadcasts the loaded element. The rewrite only applies in AscVF2
    vector functions and only when the broadcast destination matches the current VF register width, has no mask, is in
    bounds, and the scalar read has a single use.
  }];
  let constructor = "mlir::ascvf2::createMergeIntoTransferPass()";
  let dependentDialects = ["vector::VectorDialect"];
}

def OutlineVFGroup : Pass<"ascvf2-outline-vf-group", "ModuleOp"> {
  let summary = "Outlines compatible consecutive pure operations into VF groups";
  let description = [{
    Groups runs of supported pure elementwise operations and outlines each run into a private _vfN-suffixed vector
    function marked ascvf2.vector_func. The original function keeps a call-only VF group. External operands become
    function arguments, while arith constants are cloned into the function. Supported reductions may terminate a run
    when their input is produced in the run.
  }];
  let constructor = "mlir::ascvf2::createOutlineVFGroupPass()";
  let dependentDialects = ["ascvf2::AscVF2Dialect", "func::FuncDialect"];
}

def PromoteMemToReg : Pass<"ascvf2-promote-mem-to-reg", "func::FuncOp"> {
  let summary = "Promote exact vector transfer memory chains to register values";
  let description = [{
    Replaces a vector.transfer_read with the SSA vector written by an earlier exact vector.transfer_write to the same
    tensor root, indices, permutation map, vector type, in-bounds information, and mask. The promotion is local to each
    basic block and is invalidated conservatively by barriers, unknown side effects, and transfers that cannot be proven
    exact. It also removes intermediate writes when an exact later write makes them dead.
  }];
  let constructor = "mlir::ascvf2::createPromoteMemToRegPass()";
  let dependentDialects = ["vector::VectorDialect"];
}

def RaiseVFFor : Pass<"ascvf2-raise-vf-for", "func::FuncOp"> {
  let summary = "Raise eligible SCF loops to EmitAsc VF loops";
  let description = [{
    Converts result-free scf.for operations inside emitasc.vec_scope with a constant zero lower bound and constant one
    step to emitasc.vf_for. The loop body is moved without cloning and the upper-bound operand is preserved. Ineligible
    loops and loops outside a vector scope remain unchanged.
  }];
  let constructor = "mlir::ascvf2::createRaiseVFForPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
}

def ReorderGroupedOps : Pass<"ascvf2-reorder-grouped-ops", "func::FuncOp"> {
  let summary = "Reorders pure operations inside AscVF2 groups";
  let description = [{
    Moves pure operations to the earliest valid position within each block of an AscVF2 group. This groups tensor
    materializations before affine loops so later loop fusion does not need to reorder unrelated operations.
  }];
  let constructor = "mlir::ascvf2::createReorderGroupedOpsPass()";
}

def ReuseRegTensor : Pass<"ascvf2-reuse-reg-tensor", "func::FuncOp"> {
  let summary = "Reuse non-overlapping AscVF2 register tensors";
  let description = [{
    Groups register tensors of the same type by loop-aware use interval and rewrites non-overlapping groups to one
    representative. A value read before its first write inside a loop is live for the whole loop, including every
    enclosing loop with the same property. The surviving allocation must dominate every rewritten use. The pass accepts
    only linear structured control flow: single-region loops are allowed, while branches and region-crossing register
    values are rejected.
  }];
  let constructor = "mlir::ascvf2::createReuseRegTensorPass()";
}

def ShrinkVFSignature : Pass<"ascvf2-shrink-vf-signature", "ModuleOp"> {
  let summary = "Removes unused results from outlined AscVF2 functions";
  let description = [{
    Shrinks the result signature of each private ascvf2.vector_func function to the results actually used by its
    VF-group calls. Function symbols and arguments remain unchanged; calls and returns are rewritten to the reduced
    result list.
  }];
  let constructor = "mlir::ascvf2::createShrinkVFSignaturePass()";
  let dependentDialects = ["func::FuncDialect"];
}

def Vectorize : Pass<"ascvf2-vectorize", "func::FuncOp"> {
  let summary = "Converts AscVF2 elementwise groups to affine and vector operations";
  let description = [{
    Converts each supported elementwise arith or math operation inside an AscVF2 group to same-shaped UB memrefs, masked
    affine loops, and vector transfer/compute operations. Supported final-axis reductions accumulate lane-wise and emit\
    one ascvf2.reduce after the chunk loop. The prototype materializes one result and loop per grouped operation.
  }];
  let constructor = "mlir::ascvf2::createVectorizePass()";
  let dependentDialects = [
    "affine::AffineDialect", "arith::ArithDialect", "ascvf2::AscVF2Dialect", "math::MathDialect",
    "memref::MemRefDialect", "vector::VectorDialect"
  ];
}

def WrapVecScope : Pass<"ascvf2-wrap-vec-scope", "func::FuncOp"> {
  let summary = "Replaces result-free AscVF2 groups with vector scopes";
  let description = [{
    Moves the body of each result-free ascvf2.vf_group into an emitasc.vec_scope operation and replaces the AscVF2 yield
    with an EmitAsc yield. Groups with remaining results are rejected because their values still require an SSA boundary
    and are not ready for wrapping.
  }];
  let constructor = "mlir::ascvf2::createWrapVecScopePass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
}

#endif // ASCVF2_PASSES_TD
