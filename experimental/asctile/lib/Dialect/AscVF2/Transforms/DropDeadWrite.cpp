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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_DROPDEADWRITE
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct PendingWrite {
    vector::TransferWriteOp operation;
    ascvf2::TransferDescriptor descriptor;
    bool observed;
};

using PendingWrites = DenseMap<Value, SmallVector<PendingWrite, 4>>;

void markRootObserved(Value root, PendingWrites& writes)
{
    auto rootWrites = writes.find(root);
    if (rootWrites == writes.end())
        return;
    for (PendingWrite& write : rootWrites->second)
        write.observed = true;
}

void handleWrite(
    vector::TransferWriteOp operation, const ascvf2::TransferDescriptor& descriptor, PendingWrites& writes,
    IRRewriter& rewriter)
{
    auto rootWrites = writes.find(descriptor.root);
    if (rootWrites == writes.end()) {
        writes[descriptor.root].push_back(PendingWrite{operation, descriptor, false});
        return;
    }

    for (auto* iterator = rootWrites->second.begin(); iterator != rootWrites->second.end();) {
        if (!iterator->observed && iterator->descriptor.matches(descriptor)) {
            rewriter.eraseOp(iterator->operation);
            iterator = rootWrites->second.erase(iterator);
            continue;
        }
        ++iterator;
    }
    rootWrites->second.push_back(PendingWrite{operation, descriptor, false});
}

void invalidateAll(PendingWrites& writes) { writes.clear(); }

bool hasOnlyMemoryEffectFreeNestedOperations(Operation* operation)
{
    return !operation
                ->walk([&](Operation* nested) {
                    return nested == operation || isMemoryEffectFree(nested) ? WalkResult::advance() :
                                                                               WalkResult::interrupt();
                })
                .wasInterrupted();
}

bool isTransparentBetweenWrites(Operation* operation)
{
    if (isa<ascvf2::BarrierOp>(operation))
        return false;
    if (operation->getRegions().empty())
        return isMemoryEffectFree(operation);
    return operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
           hasOnlyMemoryEffectFreeNestedOperations(operation);
}

void dropDeadWrites(Block* block, PendingWrites writes, IRRewriter& rewriter)
{
    SmallVector<Operation*, 16> operations;
    for (Operation& operation : *block)
        operations.push_back(&operation);

    for (Operation* operation : operations) {
        if (operation->getBlock() != block)
            continue;

        if (isa<vector::TransferReadOp, vector::TransferWriteOp>(operation)) {
            auto descriptor = ascvf2::getTransferDescriptor(operation);
            if (!descriptor) {
                invalidateAll(writes);
                continue;
            }
            if (isa<vector::TransferReadOp>(operation))
                markRootObserved(descriptor->root, writes);
            else
                handleWrite(cast<vector::TransferWriteOp>(operation), *descriptor, writes, rewriter);
            continue;
        }

        if (!operation->getRegions().empty()) {
            if (!isTransparentBetweenWrites(operation)) {
                for (Region& region : operation->getRegions())
                    for (Block& nestedBlock : region)
                        dropDeadWrites(&nestedBlock, PendingWrites{}, rewriter);
                invalidateAll(writes);
            }
            continue;
        }

        if (!isMemoryEffectFree(operation))
            invalidateAll(writes);
    }
}

bool isPrivateTemporaryRoot(Value root)
{
    auto allocation = root.getDefiningOp<memref::AllocaOp>();
    if (!allocation || allocation->getNumOperands() != 0)
        return false;

    SmallVector<Value, 4> values{root};
    DenseSet<Value> visited;
    while (!values.empty()) {
        Value value = values.pop_back_val();
        if (!visited.insert(value).second)
            continue;
        for (OpOperand& use : value.getUses()) {
            Operation* operation = use.getOwner();
            bool allowed =
                llvm::TypeSwitch<Operation*, bool>(operation)
                    .Case<ascvf2::ToBufferOp>([&](auto toBuffer) {
                        if (use.get() != toBuffer.getTensor())
                            return false;
                        values.push_back(toBuffer.getResult());
                        return true;
                    })
                    .Case<ascvf2::ToTensorOp>([&](auto toTensor) {
                        if (use.get() != toTensor.getBuffer())
                            return false;
                        values.push_back(toTensor.getResult());
                        return true;
                    })
                    .Case<memref::ReinterpretCastOp>([&](auto reinterpret) {
                        if (use.get() != reinterpret.getSource())
                            return false;
                        values.push_back(reinterpret.getResult());
                        return true;
                    })
                    .Case<vector::TransferReadOp>([&](auto read) { return use.get() == read.getSource(); })
                    .Case<vector::TransferWriteOp>([&](auto write) { return use.get() == write.getSource(); })
                    .Default([](auto) { return false; });
            if (!allowed)
                return false;
        }
    }
    return true;
}

void eraseDeadTemporaryWrites(func::FuncOp func, IRRewriter& rewriter)
{
    DenseSet<Value> roots;
    func.walk([&](vector::TransferWriteOp write) {
        std::optional<Value> root = ascvf2::resolveTensorAliasRoot(write.getSource());
        if (root && isPrivateTemporaryRoot(*root))
            roots.insert(*root);
    });

    for (Value root : roots) {
        bool hasRead = false;
        func.walk([&](vector::TransferReadOp read) {
            auto readRoot = ascvf2::resolveTensorAliasRoot(read.getSource());
            hasRead |= readRoot == root;
        });
        if (hasRead)
            continue;

        SmallVector<vector::TransferWriteOp, 4> writes;
        func.walk([&](vector::TransferWriteOp write) {
            auto writeRoot = ascvf2::resolveTensorAliasRoot(write.getSource());
            if (writeRoot == root)
                writes.push_back(write);
        });
        for (vector::TransferWriteOp write : writes)
            rewriter.eraseOp(write);
    }
}

struct DropDeadWritePass : public ascvf2::impl::DropDeadWriteBase<DropDeadWritePass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        IRRewriter rewriter(&getContext());
        dropDeadWrites(&func.getBody().front(), PendingWrites{}, rewriter);
        eraseDeadTemporaryWrites(func, rewriter);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createDropDeadWritePass() { return std::make_unique<DropDeadWritePass>(); }
