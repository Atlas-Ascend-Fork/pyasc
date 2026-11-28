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

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_WRAPVECSCOPE
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

LogicalResult wrapInVecScope(ascvf2::VFGroupOp group)
{
    if (group.getNumResults() != 0)
        return group.emitOpError("cannot wrap a group with remaining results");
    auto yield = cast<ascvf2::YieldOp>(group.getBody()->getTerminator());

    OpBuilder builder(group);
    auto vecScope = builder.create<emitasc::VecScopeOp>(group.getLoc());
    group.getBody()->moveBefore(&vecScope.getRegion(), vecScope.getRegion().end());
    yield.erase();
    builder.setInsertionPointToEnd(vecScope.getBody());
    builder.create<emitasc::YieldOp>(group.getLoc());

    group->erase();
    return success();
}

struct WrapVecScopePass : public ascvf2::impl::WrapVecScopeBase<WrapVecScopePass> {
    void runOnOperation() override
    {
        SmallVector<ascvf2::VFGroupOp, 4> groups;
        getOperation()->walk([&](ascvf2::VFGroupOp group) { groups.push_back(group); });
        for (ascvf2::VFGroupOp group : groups) {
            if (failed(wrapInVecScope(group))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createWrapVecScopePass() { return std::make_unique<WrapVecScopePass>(); }
