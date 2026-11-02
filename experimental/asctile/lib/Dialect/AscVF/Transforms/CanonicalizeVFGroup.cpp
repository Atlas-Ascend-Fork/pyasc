/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscVF/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_CANONICALIZEVFGROUP
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

SmallVector<Operation*> getUsers(Value value)
{
    SmallVector<Operation*> users;
    if (auto tensorOp = ascendc::getAllocationRoot(value)) {
        auto uses = ascendc::collectAllUses(tensorOp);
        std::transform(
            uses.begin(), uses.end(), std::back_inserter(users), [](OpOperand* op) { return op->getOwner(); });
    } else {
        users = SmallVector<Operation*>{value.getUsers()};
    }
    return users;
}

void canonicalize(ascvf::VFGroupOp groupOp)
{
    DominanceInfo di;
    auto usedAfterGroup = [&](Value value) -> bool {
        return llvm::any_of(getUsers(value), [&](Operation* use) {
            return ascendc::opPrecedes(groupOp, use, di) && !groupOp->isAncestor(use);
        });
    };
    auto usedInsideGroup = [&](Value value) -> bool {
        return llvm::any_of(getUsers(value), [&](Operation* use) { return groupOp->isProperAncestor(use); });
    };
    SmallVector<Value> dstList = groupOp.getDstTensors();
    SmallVector<Value> srcList = groupOp.getSrcTensors();
    SmallVector<Value> src, dst, tmp;
    llvm::copy_if(srcList, std::back_inserter(src), usedInsideGroup);
    llvm::copy_if(
        dstList, std::back_inserter(dst), [&](Value value) { return usedInsideGroup(value) && usedAfterGroup(value); });
    llvm::copy_if(dstList, std::back_inserter(tmp), [&](Value value) {
        return usedInsideGroup(value) && !usedAfterGroup(value);
    });
    groupOp.getInputTensorsMutable().assign(src);
    groupOp.getOutputTensorsMutable().assign(dst);
    groupOp.getTmpTensorsMutable().assign(tmp);
}

struct CanonicalizeVFGroupPass : public ascvf::impl::CanonicalizeVFGroupBase<CanonicalizeVFGroupPass> {
    void runOnOperation() override
    {
        getOperation().walk<WalkOrder::PostOrder, ReverseIterator>(
            [](ascvf::VFGroupOp fusedOp) { canonicalize(fusedOp); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createCanonicalizeVFGroupPass()
{
    return std::make_unique<CanonicalizeVFGroupPass>();
}
