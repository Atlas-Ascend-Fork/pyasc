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

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_RAISEVFFOR
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

bool isEligibleLoop(scf::ForOp loop)
{
    return loop.getNumResults() == 0 && loop->getParentOfType<ascvf2::VFGroupOp>() &&
           matchPattern(loop.getLowerBound(), m_Zero()) && matchPattern(loop.getStep(), m_One());
}

void raiseLoop(scf::ForOp loop)
{
    OpBuilder builder(loop);
    auto vfLoop = builder.create<emitasc::VFForOp>(loop.getLoc(), loop.getUpperBound());
    loop.getInductionVar().replaceAllUsesWith(vfLoop.getInductionVar());

    Block* vfBody = vfLoop.getBody();
    Operation* vfYield = vfBody->getTerminator();
    Operation* scfYield = loop.getBody()->getTerminator();
    vfBody->getOperations().splice(vfYield->getIterator(), loop.getBody()->getOperations());
    scfYield->erase();
    loop->erase();
}

struct RaiseVFForPass : public ascvf2::impl::RaiseVFForBase<RaiseVFForPass> {
    void runOnOperation() override
    {
        SmallVector<scf::ForOp, 4> loops;
        getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); });

        for (scf::ForOp loop : loops) {
            if (isEligibleLoop(loop))
                raiseLoop(loop);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createRaiseVFForPass() { return std::make_unique<RaiseVFForPass>(); }
