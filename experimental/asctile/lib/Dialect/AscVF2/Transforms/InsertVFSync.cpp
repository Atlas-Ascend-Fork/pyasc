/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF2/IR/AscVF2.h"
#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_INSERTVFSYNC
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct TransferEffects {
    DenseSet<Value> reads;
    DenseSet<Value> writes;
};

bool isSupportedScanOperation(Operation* operation)
{
    return isa<affine::AffineForOp, affine::AffineYieldOp, vector::MaskOp, vector::YieldOp, vector::TransferReadOp,
               vector::TransferWriteOp, memref::AllocaOp, ascvf2::BarrierOp, ascvf2::YieldOp>(operation) ||
           isPure(operation);
}

LogicalResult collectTransferEffects(Operation* operation, TransferEffects& effects)
{
    WalkResult walkResult = operation->walk<WalkOrder::PreOrder>([&](Operation* nested) {
        if (!isSupportedScanOperation(nested)) {
            nested->emitOpError("unsupported side-effecting operation in AscVF2 synchronization scan");
            return WalkResult::interrupt();
        }
        if (!isa<vector::TransferReadOp, vector::TransferWriteOp>(nested))
            return WalkResult::advance();

        Value source = dyn_cast<vector::TransferReadOp>(nested) ? cast<vector::TransferReadOp>(nested).getSource() :
                                                                  cast<vector::TransferWriteOp>(nested).getSource();
        std::optional<Value> root = ascvf2::resolveTensorAliasRoot(source);
        if (!root) {
            nested->emitOpError("cannot resolve vector transfer source for synchronization");
            return WalkResult::interrupt();
        }
        if (isa<vector::TransferWriteOp>(nested))
            effects.writes.insert(*root);
        else
            effects.reads.insert(*root);
        return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
        return failure();
    return success();
}

struct InsertVFSyncPass : public ascvf2::impl::InsertVFSyncBase<InsertVFSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        SmallVector<Operation*, 8> topLevelOperations;
        for (Operation& operation : func.getBody().front())
            topLevelOperations.push_back(&operation);

        DenseSet<Value> pendingWrites;
        for (Operation* operation : topLevelOperations) {
            if (isa<ascvf2::BarrierOp>(operation)) {
                pendingWrites.clear();
                continue;
            }

            TransferEffects effects;
            if (failed(collectTransferEffects(operation, effects))) {
                signalPassFailure();
                return;
            }
            if (llvm::any_of(effects.reads, [&](Value root) { return pendingWrites.contains(root); })) {
                OpBuilder builder(operation);
                builder.create<ascvf2::BarrierOp>(operation->getLoc());
                pendingWrites.clear();
            }
            for (Value root : effects.writes)
                pendingWrites.insert(root);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createInsertVFSyncPass() { return std::make_unique<InsertVFSyncPass>(); }
