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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "asctile/Dialect/AscVF/IR/AscVFOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

namespace {
bool hasStaticShapes(Operation* op)
{
    return llvm::all_of(op->getOperandTypes(), [](Type type) {
        auto tensorType = dyn_cast<ascendc::LocalTensorType>(type);
        return !tensorType || tensorType.hasStaticShape();
    });
}

void adjustLifetime(Operation*& firstOp, Operation*& lastOp, DominanceInfo& di)
{
    auto* commonBlock = di.findNearestCommonDominator(firstOp->getBlock(), lastOp->getBlock());
    Block* stopBlock = commonBlock;
    Operation* op = lastOp;

    auto process = [&](Operation* op) {
        if (auto vfFor = dyn_cast<emitasc::VFForOp>(op))
            lastOp = vfFor.getBody()->getTerminator();
    };

    process(op);
    while (op->getBlock() != stopBlock) {
        op = op->getParentOp();
        process(op);
    }
}

using Interval = std::pair<Operation*, Operation*>;

SmallVector<Operation*> getUsersInside(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    SmallVector<Operation*> users;
    llvm::copy_if(value.getUsers(), std::back_inserter(users), [&](Operation* op) {
        return di.dominates(groupOp.getBody(), op->getBlock());
    });
    return users;
}

bool isSingleUse(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    return getUsersInside(groupOp, value, di).size() == 1;
}

bool intersect(const Interval& lhs, const Interval& rhs, DominanceInfo& di)
{
    return ascendc::opPrecedes(rhs.first, lhs.second, di) && ascendc::opPrecedes(lhs.first, rhs.second, di);
}

std::optional<Interval> getInterval(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    auto users = getUsersInside(groupOp, value, di);
    if (llvm::is_contained(groupOp.getInputTensors(), value))
        users.emplace_back(groupOp);
    if (llvm::is_contained(groupOp.getOutputTensors(), value))
        users.emplace_back(groupOp.getBody()->getTerminator());
    llvm::stable_sort(users, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
    if (users.empty())
        return std::nullopt;
    Operation* first = users.front();
    Operation* last = users.back();
    return Interval{first, last};
}

} // namespace

//===----------------------------------------------------------------------===//
// VFGroupOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify()
{
    if (auto regType = dyn_cast<ascendc::RegTensorType>(getDstReg().getType())) {
        auto tensorType = cast<ascendc::LocalTensorType>(getSrcTensor().getType());
        if (regType.getElementType() != tensorType.getElementType()) {
            return emitOpError("elem type of reg tensor and local tensor must be the same");
        }
    }
    return success();
}

LogicalResult StoreOp::verify()
{
    if (auto regType = dyn_cast<ascendc::RegTensorType>(getSrcReg().getType())) {
        auto tensorType = cast<ascendc::LocalTensorType>(getDstTensor().getType());
        if (regType.getElementType() != tensorType.getElementType()) {
            return emitOpError("elem type of reg tensor and local tensor must be the same");
        }
    }
    return success();
}

LogicalResult VFGroupOp::verify()
{
    if (getOperation()->getParentOfType<VFGroupOp>()) {
        return emitOpError("cannot be nested in other ascvf.vf_group op");
    }
    auto result = walk([&](Operation* op) {
        if (!hasStaticShapes(op)) {
            op->emitOpError("inside ascvf.vf_group must have static shape");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

bool VFGroupOp::canReuse(OpOperand& lhsOpnd, OpOperand& rhsOpnd)
{
    Value lhs = lhsOpnd.get();
    Value rhs = rhsOpnd.get();
    DominanceInfo di;
    auto groupOp = cast<VFGroupOp>(getOperation());
    auto lhsInfo = getInterval(groupOp, lhs, di);
    auto rhsInfo = getInterval(groupOp, rhs, di);
    // The tensor is not used inside the block
    if (!lhsInfo || !rhsInfo)
        return true;
    if (intersect(lhsInfo.value(), rhsInfo.value(), di))
        return false;
    if (!ascendc::opPrecedes(lhsInfo->second, rhsInfo->first, di)) {
        std::swap(lhs, rhs);
        std::swap(lhsInfo, rhsInfo);
    }
    auto lhsInfoLong = lhsInfo;
    auto rhsInfoLong = rhsInfo;
    adjustLifetime(lhsInfoLong->first, lhsInfoLong->second, di);
    adjustLifetime(rhsInfoLong->first, rhsInfoLong->second, di);
    // All uses have been completed (end of loop)
    if (ascendc::opPrecedes(lhsInfoLong->second, rhsInfoLong->first, di))
        return true;
    // Check inside the loop
    auto loadOp = dyn_cast<ascvf::LoadOp>(lhsInfo->second);
    auto storeOp = dyn_cast<ascvf::StoreOp>(rhsInfo->first);
    if (loadOp && storeOp && ascendc::opPrecedes(loadOp, storeOp, di) && loadOp->getBlock() == storeOp->getBlock() &&
        loadOp.getOffset() == storeOp.getOffset()) {
        if (isSingleUse(groupOp, lhs, di) && isSingleUse(groupOp, rhs, di))
            return true;
        // This is not a nested loop
        auto loop = dyn_cast<emitasc::VFForOp>(loadOp->getParentOp());
        if (loop && !loop->getParentOfType<emitasc::VFForOp>())
            return true;
    }
    return false;
}

SmallVector<Value> VFGroupOp::getDstTensors()
{
    SmallVector<Value> dstTensors;
    dstTensors.append(getOutputTensors().begin(), getOutputTensors().end());
    dstTensors.append(getTmpTensors().begin(), getTmpTensors().end());
    return dstTensors;
}

SmallVector<Value> VFGroupOp::getSrcTensors() { return llvm::to_vector(getInputTensors()); }

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "asctile/Dialect/AscVF/IR/AscVFOps.cpp.inc"
        >();
}
