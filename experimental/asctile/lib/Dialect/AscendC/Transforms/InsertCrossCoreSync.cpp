/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Utils/Attributes.h"
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "CrossCoreSyncUtils.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

bool isSyncTriggerOp(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return (src == ascendc::TPosition::VECCALC &&
                (dst == ascendc::TPosition::A1 || dst == ascendc::TPosition::B1)) ||
               (src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::VECCALC);
    });
}

Operation* findGroupAncestor(Operation* op)
{
    for (Operation* p = op->getParentOp(); p; p = p->getParentOp())
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(p))
            return p;
    return nullptr;
}

SmallVector<Operation*> collectUsers(Operation* dstDefiningOp, bool triggerIsAIV, Operation* afterOp)
{
    SmallVector<Operation*> users;
    bool seen = !afterOp;
    dstDefiningOp->getParentOfType<func::FuncOp>()->walk([&](Operation* op) {
        if (op == afterOp) {
            seen = true;
            return;
        }
        if (!seen || isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op))
            return;
        if (!llvm::any_of(op->getOperands(), [&](Value v) { return v.getDefiningOp() == dstDefiningOp; }))
            return;
        if (Operation* group = findGroupAncestor(op); isa_and_present<ascendc::IfAICOp>(group) == triggerIsAIV)
            users.push_back(op);
    });
    return users;
}

ascendc::Pipe processUsers(OpBuilder& builder, ArrayRef<Operation*> users, int32_t flagId, bool dualSync)
{
    Operation* first = users.front();
    Operation* group = findGroupAncestor(first);
    Operation* last = first;
    for (auto* user : users) {
        if (findGroupAncestor(user) != group)
            break;
        last = user;
    }
    bool dualFlags = dualSync && isa_and_present<ascendc::IfAICOp>(group);
    Block& body = group->getRegion(0).front();
    builder.setInsertionPointToStart(&body);
    createWaitFlag(builder, first->getLoc(), flagId, ascendc::getOpPipeExt(first), dualFlags);
    builder.setInsertionPoint(body.getTerminator());
    ascendc::Pipe lastPipe = ascendc::getOpPipeExt(last);
    createSetFlag(builder, last->getLoc(), flagId, lastPipe, dualFlags);
    return lastPipe;
}

template <typename FlagOp>
void insertFlagGroup(OpBuilder& builder, Operation* op, bool isAIV, int32_t flagId, Pipe pipe, bool dualSync)
{
    Location loc = op->getLoc();
    ascir::ConstantOpBuilder consts(builder);
    Operation* group = isAIV ? builder.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{}).getOperation() :
                               builder.create<ascendc::IfAICOp>(loc, TypeRange{}, ValueRange{}).getOperation();
    builder.createBlock(&group->getRegion(0));
    builder.create<FlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
    if (!isAIV && dualSync)
        builder.create<FlagOp>(loc, consts.i32(maxTensorId + flagId), crossCoreMode, pipe);
    builder.create<ascendc::YieldOp>(loc);
}

Operation* getCopyDstRoot(Operation* op)
{
    Value dst = cast<ascendc::DataCopyOp>(op).getDst();
    while (auto subOp = dst.getDefiningOp<ascendc::LocalTensorSubIndexOp>())
        dst = subOp.getTensor();
    return dst.getDefiningOp();
}

struct InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        bool dualSync = false;
        if (auto attr = getModule(funcOp)->getAttrOfType<IntegerAttr>(attr::cvRatio))
            dualSync = attr.getValue().getSExtValue() == 2;
        SmallVector<Operation*> groupOps = collectGroupOps(funcOp);
        if (groupOps.size() < 2)
            return;
        OpBuilder builder(funcOp.getContext());
        int32_t crossCoreFlagId = 0;
        struct FlagInfo {
            int32_t flagId;
            Operation* loopOp;
        };
        struct SyncInfo {
            int flagId;
            bool isAIV;
            ascendc::Pipe pipe;
            Operation* groupOp;
        };
        llvm::DenseMap<Operation*, FlagInfo> dstFlagMap;
        llvm::DenseMap<std::pair<scf::ForOp, int>, SyncInfo> afterLoopSync;
        llvm::DenseMap<std::pair<scf::ForOp, int>, SyncInfo> beforeLoopSync;
        for (auto* groupOp : groupOps) {
            bool isAIV = isa<ascendc::IfAIVOp>(groupOp);
            SmallVector<Operation*> syncOps = collectOps(groupOp, isSyncTriggerOp);
            if (syncOps.empty())
                continue;
            scf::ForOp loopOp = groupOp->getParentOfType<scf::ForOp>();
            for (auto* op : syncOps) {
                Operation* dstDefiningOp = getCopyDstRoot(op);
                if (!dstDefiningOp)
                    continue;
                ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
                auto it = dstFlagMap.find(dstDefiningOp);
                bool isSecondTrigger = it != dstFlagMap.end();
                int32_t flagId = isSecondTrigger ? it->second.flagId : crossCoreFlagId;
                SmallVector<Operation*> users = collectUsers(dstDefiningOp, isAIV, op);
                if (users.empty() && !isSecondTrigger)
                    continue;
                if (!isSecondTrigger) {
                    crossCoreFlagId = (crossCoreFlagId + 1) % maxTensorId;
                    dstFlagMap[dstDefiningOp] = {flagId, loopOp};
                }
                bool loopChanged = isSecondTrigger && (loopOp != it->second.loopOp);
                Block& groupBody = groupOp->getRegion(0).front();
                if (isSecondTrigger || loopOp) {
                    builder.setInsertionPointToStart(&groupBody);
                    createWaitFlag(builder, op->getLoc(), flagId, producerPipe, dualSync && !isAIV);
                    if (loopOp)
                        afterLoopSync.insert(
                            std::pair(std::pair(loopOp, flagId), SyncInfo{flagId, isAIV, producerPipe, groupOp}));
                }
                if (!users.empty()) {
                    builder.setInsertionPoint(groupBody.getTerminator());
                    createSetFlag(builder, op->getLoc(), flagId, producerPipe, dualSync && !isAIV);
                    auto pipe = processUsers(builder, users, flagId, dualSync);
                    if (loopOp || isSecondTrigger)
                        beforeLoopSync.insert(
                            std::pair(std::pair(loopOp, flagId), SyncInfo{flagId, !isAIV, pipe, groupOp}));
                }
                if (isSecondTrigger)
                    it->second.loopOp = loopOp;
            }
        }
        for (auto& [key, sync] : beforeLoopSync) {
            builder.setInsertionPoint(key.first ? key.first : sync.groupOp);
            insertFlagGroup<ascendc::CrossCoreSetFlagOp>(
                builder, sync.groupOp, sync.isAIV, sync.flagId, sync.pipe, dualSync);
        }
        for (auto& [key, sync] : afterLoopSync) {
            builder.setInsertionPointAfter(key.first ? key.first : sync.groupOp);
            insertFlagGroup<ascendc::CrossCoreWaitFlagOp>(
                builder, sync.groupOp, sync.isAIV, sync.flagId, sync.pipe, dualSync);
        }
        funcOp->setAttr(attr::crossCoreFlagId, builder.getI32IntegerAttr(crossCoreFlagId));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
