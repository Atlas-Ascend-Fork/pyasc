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

#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_REORDERGROUPEDOPS
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

class GroupedOpsReorderer {
    DominanceInfo dom;

    static bool canReorder(Operation* op)
    {
        return isPure(op) && !op->mightHaveTrait<OpTrait::IsTerminator>() && !isa<ascvf2::ToTensorOp>(op);
    }

    void reorder(Operation* op, Block& block)
    {
        for (auto& prevOp : llvm::make_range(block.begin(), Block::iterator(op))) {
            if (llvm::all_of(op->getOperands(), [&](Value operand) {
                    if (auto* definingOp = operand.getDefiningOp())
                        return ascendc::opPrecedes(definingOp, &prevOp, dom);
                    return dom.properlyDominates(operand, &prevOp);
                })) {
                op->moveBefore(&prevOp);
                return;
            }
        }
    }

public:
    void runWithinBlock(Block& block)
    {
        SmallVector<Operation*> ops;
        for (auto& op : block) {
            if (canReorder(&op))
                ops.push_back(&op);
        }
        for (auto* op : ops)
            reorder(op, block);
    }

    static void hoistAllocations(Block& block)
    {
        SmallVector<memref::AllocaOp, 4> allocations;
        for (auto& op : block) {
            auto allocation = dyn_cast<memref::AllocaOp>(op);
            if (allocation && allocation->getNumOperands() == 0)
                allocations.push_back(allocation);
        }
        for (memref::AllocaOp allocation : llvm::reverse(allocations))
            allocation->moveBefore(&block, block.begin());
    }

    static void hoistStaticReinterpretCasts(Block& block)
    {
        SmallVector<memref::ReinterpretCastOp, 4> reinterprets;
        for (auto& operation : block) {
            auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(&operation);
            if (reinterpret && reinterpret.getOffsets().empty() && reinterpret.getSizes().empty() &&
                reinterpret.getStrides().empty())
                reinterprets.push_back(reinterpret);
        }
        for (memref::ReinterpretCastOp reinterpret : llvm::reverse(reinterprets)) {
            Operation* source = reinterpret.getSource().getDefiningOp();
            if (source && source->getBlock() == &block)
                reinterpret->moveAfter(source);
            else
                reinterpret->moveBefore(&block, block.begin());
        }
    }

    static void sinkToTensorOps(Block& block)
    {
        SmallVector<ascvf2::ToTensorOp, 4> toTensorOps;
        for (auto& operation : block) {
            if (auto toTensor = dyn_cast<ascvf2::ToTensorOp>(&operation))
                toTensorOps.push_back(toTensor);
        }
        for (ascvf2::ToTensorOp toTensor : toTensorOps) {
            Operation* firstUser = nullptr;
            for (OpOperand& use : toTensor->getUses()) {
                Operation* user = use.getOwner();
                while (user && user->getBlock() != &block)
                    user = user->getParentOp();
                if (!user)
                    continue;
                if (!firstUser || user->isBeforeInBlock(firstUser))
                    firstUser = user;
            }
            if (firstUser)
                toTensor->moveBefore(firstUser);
        }
    }
};

struct ReorderGroupedOpsPass : public ascvf2::impl::ReorderGroupedOpsBase<ReorderGroupedOpsPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        GroupedOpsReorderer reorderer;
        reorderer.runWithinBlock(func.getBody().front());
        GroupedOpsReorderer::sinkToTensorOps(func.getBody().front());
        GroupedOpsReorderer::hoistAllocations(func.getBody().front());
        GroupedOpsReorderer::hoistStaticReinterpretCasts(func.getBody().front());
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createReorderGroupedOpsPass() { return std::make_unique<ReorderGroupedOpsPass>(); }
