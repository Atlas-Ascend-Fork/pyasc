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

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_BUFFERIZELOOPRESULT
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

bool hasConflictingInitUse(Value initArg, scf::ForOp loop, DominanceInfo& dominance)
{
    return llvm::any_of(initArg.getUses(), [&](OpOperand& use) {
        Operation* user = use.getOwner();
        return user != loop && !loop->isAncestor(user) && !dominance.properlyDominates(user, loop);
    });
}

bool hasUseOutsideLoop(Value value, scf::ForOp loop)
{
    return llvm::any_of(value.getUses(), [&](OpOperand& use) {
        Operation* user = use.getOwner();
        return user != loop && !loop->isAncestor(user);
    });
}

bool isRegisterAccumulator(Operation* operation, Value iterArg, Value yieldedValue)
{
    if (!isa<ascendc::BinaryRegOp, ascendc::VecScalarRegOp>(operation))
        return false;
    if (operation->getNumOperands() < 2 || operation->getOpOperand(0).get() != yieldedValue)
        return false;
    return llvm::is_contained(operation->getOperands().drop_front(), iterArg);
}

struct LoopResultBufferization {
    scf::ForOp loop;
    unsigned resultIndex;
    Value initArg;
    Value iterArg;
    Value yieldedValue;
    ascendc::DuplicateRegOp initializer;
};

FailureOr<LoopResultBufferization> canReuseLoopResult(scf::ForOp loop, unsigned resultIndex, DominanceInfo& dominance)
{
    Value initArg = loop.getInitArgs()[resultIndex];
    Value iterArg = loop.getRegionIterArgs()[resultIndex];
    Value yieldedValue = cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(resultIndex);

    if (!initArg.getDefiningOp<ascendc::RegTensorOp>() || !yieldedValue.getDefiningOp<ascendc::RegTensorOp>())
        return failure();
    if (hasConflictingInitUse(initArg, loop, dominance) || hasUseOutsideLoop(yieldedValue, loop))
        return failure();

    ascendc::DuplicateRegOp initializer;
    if (loop->getParentOfType<scf::ForOp>()) {
        for (OpOperand& use : initArg.getUses()) {
            if (use.getOwner() == loop)
                continue;
            auto duplicate = dyn_cast<ascendc::DuplicateRegOp>(use.getOwner());
            if (!duplicate || duplicate.getDstReg() != initArg || !dominance.dominates(duplicate, loop))
                return failure();
            if (initializer)
                return failure();
            initializer = duplicate;
        }
        if (!initializer)
            return failure();
    }

    Operation* accumulator = nullptr;
    for (OpOperand& use : yieldedValue.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()))
            continue;
        if (!loop->isAncestor(use.getOwner()) || !isRegisterAccumulator(use.getOwner(), iterArg, yieldedValue))
            return failure();
        if (accumulator)
            return failure();
        accumulator = use.getOwner();
    }
    if (!accumulator || !iterArg.hasOneUse() || iterArg.use_empty() ||
        iterArg.getUses().begin()->getOwner() != accumulator)
        return failure();

    return LoopResultBufferization{loop, resultIndex, initArg, iterArg, yieldedValue, initializer};
}

struct BufferizeLoopResultPass : public ascvf2::impl::BufferizeLoopResultBase<BufferizeLoopResultPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        SmallVector<scf::ForOp, 4> loops;
        func.walk([&](scf::ForOp loop) { loops.push_back(loop); });

        DominanceInfo dominance(func);
        SmallVector<LoopResultBufferization, 4> replacements;
        for (scf::ForOp loop : loops) {
            for (unsigned index = 0; index < loop.getNumResults(); ++index) {
                FailureOr<LoopResultBufferization> replacement = canReuseLoopResult(loop, index, dominance);
                if (succeeded(replacement))
                    replacements.push_back(*replacement);
            }
        }
        for (auto& [loop, resultIndex, initArg, iterArg, yieldedValue, initializer] : replacements) {
            Value replacement = initArg;
            if (initializer) {
                OpBuilder builder(loop);
                replacement = builder.create<ascendc::RegTensorOp>(loop.getLoc(), initArg.getType());
                builder.create<ascendc::DuplicateRegOp>(
                    initializer.getLoc(), replacement, initializer.getSrcReg(), initializer.getMaskReg());
                loop.getInitArgsMutable()[resultIndex].set(replacement);
            }
            iterArg.replaceAllUsesWith(replacement);
            yieldedValue.replaceAllUsesWith(replacement);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createBufferizeLoopResultPass()
{
    return std::make_unique<BufferizeLoopResultPass>();
}
