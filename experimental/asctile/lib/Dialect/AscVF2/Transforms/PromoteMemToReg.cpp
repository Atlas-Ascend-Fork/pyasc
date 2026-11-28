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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_PROMOTEMEMTOREG
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct PendingWrite {
    vector::TransferWriteOp operation;
    ascvf2::TransferDescriptor descriptor;
};

using PendingWrites = DenseMap<Value, SmallVector<PendingWrite, 4>>;

void handleRead(
    vector::TransferReadOp operation, const ascvf2::TransferDescriptor& descriptor, PendingWrites& writes,
    IRRewriter& rewriter)
{
    auto rootWrites = writes.find(descriptor.root);
    if (rootWrites == writes.end())
        return;

    PendingWrite* exactWrite = nullptr;
    for (PendingWrite& write : rootWrites->second) {
        if (write.descriptor.matches(descriptor))
            exactWrite = &write;
    }
    if (!exactWrite) {
        for (PendingWrite& write : rootWrites->second) {
            auto writeType = dyn_cast<VectorType>(write.operation.getVector().getType());
            auto readType = operation.getVectorType();
            if (!writeType || writeType.getRank() != 1 || writeType.getDimSize(0) != 1 || readType.getRank() != 1 ||
                readType.getDimSize(0) <= 1 || writeType.getElementType() != readType.getElementType() ||
                write.descriptor.mask || descriptor.mask || write.descriptor.indices.size() != 1 ||
                descriptor.indices.size() != 1 || write.descriptor.indices.front() != descriptor.indices.front() ||
                !llvm::equal(write.descriptor.inBounds, descriptor.inBounds))
                continue;

            AffineMap readMap = operation.getPermutationMap();
            if (readMap.getNumResults() != 1 || !isa<AffineConstantExpr>(readMap.getResult(0)) ||
                cast<AffineConstantExpr>(readMap.getResult(0)).getValue() != 0 ||
                write.descriptor.permutationMap != AffineMap::getMultiDimIdentityMap(1, operation.getContext()))
                continue;

            rewriter.setInsertionPoint(operation);
            auto broadcast =
                rewriter.create<vector::BroadcastOp>(operation.getLoc(), readType, write.operation.getVector());
            rewriter.replaceOp(operation, broadcast.getResult());
            return;
        }
        return;
    }

    // If masks diverge in the future, inactive-lane values will require a
    // vector select rather than direct SSA forwarding.
    rewriter.replaceOp(operation, exactWrite->operation.getVector());
}

void invalidateAll(PendingWrites& writes) { writes.clear(); }

bool hasNoDirectMemoryEffect(Operation* operation)
{
    if (auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(operation))
        return memoryEffectOp.hasNoEffect();
    return operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
}

void promoteBlock(Block* block, PendingWrites writes, IRRewriter& rewriter)
{
    SmallVector<Operation*, 16> operations;
    for (Operation& operation : *block)
        operations.push_back(&operation);

    for (Operation* operation : operations) {
        if (operation->getBlock() != block)
            continue;

        if (auto read = dyn_cast<vector::TransferReadOp>(operation)) {
            auto descriptor = ascvf2::getTransferDescriptor(operation);
            if (!descriptor) {
                invalidateAll(writes);
                continue;
            }
            handleRead(read, *descriptor, writes, rewriter);
            continue;
        }
        if (auto write = dyn_cast<vector::TransferWriteOp>(operation)) {
            auto descriptor = ascvf2::getTransferDescriptor(operation);
            if (!descriptor) {
                invalidateAll(writes);
                continue;
            }
            writes[descriptor->root].clear();
            writes[descriptor->root].push_back(PendingWrite{write, *descriptor});
            continue;
        }

        if (!operation->getRegions().empty()) {
            PendingWrites regionWrites;
            if (!isa<ascvf2::BarrierOp>(operation) && hasNoDirectMemoryEffect(operation))
                regionWrites = writes;
            for (Region& region : operation->getRegions()) {
                for (Block& nestedBlock : region)
                    promoteBlock(&nestedBlock, regionWrites, rewriter);
            }
            invalidateAll(writes);
            continue;
        }

        if (isa<ascvf2::BarrierOp>(operation) || !isMemoryEffectFree(operation))
            invalidateAll(writes);
    }
}

struct PromoteMemToRegPass : public ascvf2::impl::PromoteMemToRegBase<PromoteMemToRegPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        IRRewriter rewriter(&getContext());
        for (Block& block : func.getBody())
            promoteBlock(&block, PendingWrites{}, rewriter);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createPromoteMemToRegPass() { return std::make_unique<PromoteMemToRegPass>(); }
