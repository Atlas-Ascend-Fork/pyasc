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
#include "asctile/Dialect/AscVF/Utils/Utils.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_ELIMINATEDATATRANSFER
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

SmallVector<SmallVector<Operation*>> collectLoadStoreOpsByBlock(ascvf::VFGroupOp groupOp)
{
    SmallVector<SmallVector<Operation*>> loadStoreGroups;
    groupOp.walk([&](Block* block) {
        SmallVector<Operation*> blockOps;
        for (auto& op : *block) {
            if (isa<ascvf::LoadOp, ascvf::StoreOp>(op)) {
                blockOps.emplace_back(&op);
            }
        }
        if (!blockOps.empty()) {
            loadStoreGroups.emplace_back(std::move(blockOps));
        }
    });
    return loadStoreGroups;
}

std::optional<bool> isSubset(Value a, Value b)
{
    if (!isa<ascendc::UpdateMaskOp, ascendc::CreateMaskOp>(a.getDefiningOp()) ||
        !isa<ascendc::UpdateMaskOp, ascendc::CreateMaskOp>(b.getDefiningOp()))
        return std::nullopt;
    auto updateMaskA = a.getDefiningOp<ascendc::UpdateMaskOp>();
    auto updateMaskB = b.getDefiningOp<ascendc::UpdateMaskOp>();
    auto createMaskA = a.getDefiningOp<ascendc::CreateMaskOp>();
    auto createMaskB = b.getDefiningOp<ascendc::CreateMaskOp>();
    if (updateMaskA && updateMaskB)
        return updateMaskA == updateMaskB;
    if (updateMaskA && createMaskB)
        return createMaskB.getMask() == ascendc::MaskPattern::ALL;
    if (createMaskA && createMaskB) {
        auto maskA = createMaskA.getMask();
        auto maskB = createMaskB.getMask();
        if (maskA > ascendc::MaskPattern::VL128 || maskB > ascendc::MaskPattern::VL128)
            return std::nullopt;
        if (maskB == ascendc::MaskPattern::ALL)
            return true;
        if (maskA == ascendc::MaskPattern::ALL)
            return maskB == ascendc::MaskPattern::ALL;
        return maskA <= maskB;
    }
    if (createMaskA && updateMaskB)
        return false;
    return std::nullopt;
}

// Optimization work in the same block
// Before:                           | After:
// Reg r0                            | Reg r0
// Reg r1                            | Reg r1
// for(...) {                        | for(...) {
//   op1(r0)                         |   op1(r0)
//   store local_tensor0[offset], r0 |   store local_tensor0[offset], r0
//   load r1, local_tensor0[offset]  |   op2(r0)
//   op2(r1)                         | }
// }                                 |
void eliminateRedundantLoadsAfterStores(ascvf::VFGroupOp groupOp)
{
    auto loadStoreGroups = collectLoadStoreOpsByBlock(groupOp);
    SmallVector<Operation*> needDelete;
    for (auto& blockOps : loadStoreGroups) {
        ValueMap<SmallVector<Operation*>> tensorToLoadStoreOps;
        for (auto* op : blockOps) {
            if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
                if (auto tensor = ascendc::getAllocationRoot(loadOp.getSrcTensor()))
                    tensorToLoadStoreOps[tensor].emplace_back(loadOp);
            } else if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                if (auto tensor = ascendc::getAllocationRoot(storeOp.getDstTensor()))
                    tensorToLoadStoreOps[tensor].emplace_back(storeOp);
            }
        }
        for (auto& pair : tensorToLoadStoreOps) {
            ascvf::StoreOp lastStore;
            for (auto* op : pair.second) {
                if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                    lastStore = storeOp;
                } else if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
                    // need cse pass before
                    if (lastStore && loadOp.getOffset() == lastStore.getOffset()) {
                        if (auto opt = isSubset(loadOp.getMask(), lastStore.getMask()); opt && opt.value()) {
                            loadOp.getDstReg().replaceAllUsesWith(lastStore.getSrcReg());
                            needDelete.push_back(loadOp);
                        }
                    }
                }
            }
        }
    }
    for (auto* op : needDelete) {
        op->erase();
    }
}

SmallVector<emitasc::VFForOp> getLoops(ascvf::VFGroupOp groupOp)
{
    SmallVector<emitasc::VFForOp> loops;
    groupOp.walk([&](emitasc::VFForOp forOp) { loops.emplace_back(forOp); });
    return loops;
}

ValueMap<Operation*> getDstMap(ascvf::VFGroupOp groupOp)
{
    ValueMap<Operation*> dstMap;
    groupOp.walk([&](Operation* op) {
        for (auto dst : ascvf::getDst(op))
            dstMap[dst] = op;
    });
    return dstMap;
}

// Before: (Ex. duplicate)                 | After:
// Reg r0                                  | Reg r0
// for(i < ub1) {                          | for(i < ub1) {
//   store local_tensor0[offset], r0       |   store local_tensor0[offset], r0
// }                                       | }
// for(i < ub2) {                          | for(i < ub2) {
//   Reg r1 = load local_tensor0[offset]   |   compute(r0)
//   compute(r1)                           | }
// }                                       |
// TODO: We can apply optimize if sure that local_tensor contains repeated value (ub2 >= ub1 && store with linear access
// with full mask)
void replaceIdenticalLoads(ascvf::VFGroupOp groupOp)
{
    auto dstMap = getDstMap(groupOp);
    ValueMap<Value> storesOfScalarValue;
    DominanceInfo di;
    SmallVector<Operation*> needDelete;
    for (auto forOp : getLoops(groupOp)) {
        for (auto& op : llvm::make_early_inc_range(forOp)) {
            if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                auto srcReg = storeOp.getSrcReg();
                if (dstMap.count(srcReg) && ascvf::belong(storeOp->getBlock(), dstMap[srcReg]->getBlock(), di) &&
                    storeOp->getBlock() != dstMap[srcReg]->getBlock()) {
                    storesOfScalarValue[storeOp.getDstTensor()] = srcReg;
                } else {
                    storesOfScalarValue.erase(storeOp.getDstTensor());
                }
            } else if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
                auto dstReg = loadOp.getDstReg();
                if (storesOfScalarValue.count(loadOp.getSrcTensor())) {
                    dstReg.replaceAllUsesWith(storesOfScalarValue[loadOp.getSrcTensor()]);
                    needDelete.push_back(loadOp);
                }
            }
        }
    }
    for (auto* op : needDelete) {
        op->erase();
    }
}

void eliminateOverwrittenStores(ascvf::VFGroupOp groupOp)
{
    ValueSet outputTensors;
    for (auto tensor : groupOp.getOutputTensors()) {
        outputTensors.insert(tensor);
    }
    ValueMap<SmallVector<Operation*>> tensorToLoadStoreOps;
    groupOp.walk([&](Operation* op) {
        if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
            if (auto tensor = ascendc::getAllocationRoot(loadOp.getSrcTensor()))
                tensorToLoadStoreOps[tensor].emplace_back(loadOp);
        } else if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
            if (auto tensor = ascendc::getAllocationRoot(storeOp.getDstTensor()))
                tensorToLoadStoreOps[tensor].emplace_back(storeOp);
        }
    });
    llvm::DenseSet<Operation*> opsToDelete;
    for (auto& pair : tensorToLoadStoreOps) {
        auto& ops = pair.second;
        // Delete the last store if it's to a non-output tensor
        // (the value won't be used outside the VFGroupOp)
        if (auto lastStore = dyn_cast<ascvf::StoreOp>(ops.back())) {
            if (!outputTensors.count(lastStore.getDstTensor())) {
                opsToDelete.insert(lastStore);
            }
        }
        // Erase overwritten stores
        bool hasSeenStore = false;
        // TODO: An analysis is needed to determine whether one or more stores that follow overwrite the stores that
        // precede them.
        for (auto* op : llvm::make_range(ops.rbegin(), ops.rend())) {
            if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                if (hasSeenStore) {
                    opsToDelete.insert(storeOp);
                }
                hasSeenStore = true;
            } else {
                hasSeenStore = false;
            }
        }
    }
    for (auto* op : opsToDelete) {
        op->erase();
    }
}

bool overwrittenBetween(ascvf::LoadOp beginOp, ascvf::LoadOp endOp, DominanceInfo& di)
{
    auto tensor = beginOp.getSrcTensor();
    assert(tensor == endOp.getSrcTensor());
    if (!ascendc::opPrecedes(beginOp, endOp, di))
        std::swap(beginOp, endOp);
    auto* commonOp = di.findNearestCommonDominator(beginOp->getBlock(), endOp->getBlock())->getParentOp();
    bool inRange = false;
    bool overwritten = false;
    commonOp->walk([&](Operation* op) {
        if (op == beginOp)
            inRange = true;
        if (auto storeOp = dyn_cast<ascvf::StoreOp>(op); storeOp && storeOp.getDstTensor() == tensor) {
            overwritten = true;
            return WalkResult::interrupt();
        }
        if (op == endOp)
            return WalkResult::interrupt();
        return WalkResult::advance();
    });
    return overwritten;
}

void mergeDuplicateLoadsFromSameAddress(ascvf::VFGroupOp groupOp)
{
    using Elem = std::tuple<Value, Value, Value>;
    SmallVector<Operation*> needDelete;
    DominanceInfo di;
    groupOp.walk([&](Block* block) {
        llvm::DenseMap<Elem, ascvf::LoadOp> earlyLoad;
        for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
            auto loadOp = dyn_cast<ascvf::LoadOp>(op);
            if (!loadOp)
                continue;
            Elem elem{loadOp.getSrcTensor(), loadOp.getOffset(), loadOp.getMask()};
            auto it = earlyLoad.find(elem);
            if (it != earlyLoad.end()) {
                auto earlyLoadOp = it->second;
                if (overwrittenBetween(earlyLoadOp, loadOp, di)) {
                    earlyLoad[elem] = loadOp;
                } else {
                    loadOp.getDstReg().replaceAllUsesWith(earlyLoadOp.getDstReg());
                    needDelete.push_back(loadOp);
                }
            } else {
                earlyLoad[elem] = loadOp;
            }
        }
    });
    for (auto* op : needDelete) {
        op->erase();
    }
}

struct EliminateDataTransferPass : public ascvf::impl::EliminateDataTransferBase<EliminateDataTransferPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp vfGroupOp) {
            eliminateRedundantLoadsAfterStores(vfGroupOp);
            eliminateOverwrittenStores(vfGroupOp);
            mergeDuplicateLoadsFromSameAddress(vfGroupOp);
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createEliminateDataTransferPass()
{
    return std::make_unique<EliminateDataTransferPass>();
}
