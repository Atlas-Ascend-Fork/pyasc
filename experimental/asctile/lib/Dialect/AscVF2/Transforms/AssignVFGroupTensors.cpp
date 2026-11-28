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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_ASSIGNVFGROUPTENSORS
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

Value resolveLocalTensor(Value address)
{
    while (Operation* operation = address.getDefiningOp()) {
        if (auto pointer = dyn_cast<emitasc::PtrOffsetOp>(operation)) {
            address = pointer.getBase();
            continue;
        }
        if (auto phyAddr = dyn_cast<ascendc::LocalTensorGetPhyAddrV2Op>(operation))
            return phyAddr.getTensor();
        return {};
    }
    return {};
}

bool isUsedAfterGroup(Value tensor, ascvf2::VFGroupOp group, DominanceInfo& dominance)
{
    return llvm::any_of(tensor.getUses(), [&](OpOperand& use) {
        Operation* user = use.getOwner();
        return !group->isAncestor(user) && ascendc::opPrecedes(group, user, dominance);
    });
}

LogicalResult assignGroupTensors(ascvf2::VFGroupOp group)
{
    if (group.getNumResults() != 0)
        return group.emitOpError("cannot assign tensors to a group with remaining results");

    SetVector<Value> capturedValues;
    getUsedValuesDefinedAbove(group.getRegion(), group.getRegion(), capturedValues);

    DenseSet<Value> readTensors;
    DenseSet<Value> writtenTensors;
    group->walk([&](Operation* operation) {
        if (auto load = dyn_cast<ascendc::DataCopyLoadOp>(operation)) {
            if (Value tensor = resolveLocalTensor(load.getSrc()))
                readTensors.insert(tensor);
            return;
        }
        if (auto store = dyn_cast<ascendc::DataCopyStoreOp>(operation)) {
            if (Value tensor = resolveLocalTensor(store.getDst()))
                writtenTensors.insert(tensor);
        }
    });

    DominanceInfo dominance;
    SmallVector<Value, 4> inputTensors;
    SmallVector<Value, 4> outputTensors;
    SmallVector<Value, 4> temporaryTensors;
    for (Value tensor : capturedValues.getArrayRef()) {
        if (!isa<ascendc::LocalTensorType>(tensor.getType()))
            continue;
        bool isRead = readTensors.contains(tensor);
        bool isWritten = writtenTensors.contains(tensor);
        bool isUsedAfter = isUsedAfterGroup(tensor, group, dominance);
        if (isRead && (!isWritten || isUsedAfter))
            inputTensors.push_back(tensor);
        if (!isWritten)
            continue;
        if (isUsedAfter) {
            outputTensors.push_back(tensor);
        } else {
            temporaryTensors.push_back(tensor);
        }
    }

    group.getInputTensorsMutable().assign(inputTensors);
    group.getOutputTensorsMutable().assign(outputTensors);
    group.getTempTensorsMutable().assign(temporaryTensors);
    return success();
}

struct AssignVFGroupTensorsPass : public ascvf2::impl::AssignVFGroupTensorsBase<AssignVFGroupTensorsPass> {
    void runOnOperation() override
    {
        SmallVector<ascvf2::VFGroupOp, 4> groups;
        getOperation()->walk([&](ascvf2::VFGroupOp group) { groups.push_back(group); });
        for (ascvf2::VFGroupOp group : groups) {
            if (failed(assignGroupTensors(group))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createAssignVFGroupTensorsPass()
{
    return std::make_unique<AssignVFGroupTensorsPass>();
}
