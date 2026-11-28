/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscVF2/IR/AscVF2.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace ascendc {

namespace {

void appendImplicitUses(Value value, SmallVectorImpl<OpOperand*>& allUses)
{
    llvm::transform(value.getUses(), std::back_inserter(allUses), [](OpOperand& o) { return &o; });
    for (auto& use : value.getUses()) {
        auto opndNum = use.getOperandNumber();
        auto* user = use.getOwner();
        if (isa<CastOpInterface>(user) || isa<LocalTensorSubIndexOp>(user)) {
            appendImplicitUses(user->getResult(0), allUses);
        } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
            // if value use as init value then this memory use for first iteration and life interval must include
            // lifeInterval iterArg
            appendImplicitUses(forOp.getRegionIterArg(opndNum - forOp.getNumControlOperands()), allUses);
        } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
            // if value return in yield op then it is accumulator and it is used as return value in forOp
            if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
                allUses.push_back(&forOp.getInitArgsMutable()[opndNum]);
                appendImplicitUses(forOp.getRegionIterArg(opndNum), allUses);
                appendImplicitUses(forOp->getResult(opndNum), allUses);
            } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
                appendImplicitUses(ifOp->getResult(opndNum), allUses);
            }
        } else if (auto yieldOp = dyn_cast<ascendc::YieldOp>(user)) {
            Operation* ifCoreOp = yieldOp->getParentOp();
            if (isa<ascendc::IfAIVOp, ascendc::IfAICOp>(ifCoreOp)) {
                appendImplicitUses(ifCoreOp->getResult(opndNum), allUses);
            }
        }
        if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
            appendImplicitUses(selectOp->getResult(0), allUses);
        }
    }
}

} // namespace

SmallVector<OpOperand*> collectAllUses(LocalTensorAutoOp tensorOp)
{
    SmallVector<OpOperand*> uses;
    appendImplicitUses(tensorOp, uses);
    return uses;
}

Pipe getOpPipeExt(Operation* op, Pipe defaultPipe)
{
    if (isa<ascvf::VFGroupOp, ascvf2::VFGroupOp>(op))
        return Pipe::PIPE_V;
    return getOpPipe(op, defaultPipe);
}

} // namespace ascendc
} // namespace mlir
