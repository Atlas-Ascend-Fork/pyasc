/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/MapVector.h"

#include <limits>

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_REUSEREGTENSOR
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct RegInterval {
    ascendc::RegTensorOp tensor;
    unsigned first = 0;
    unsigned last = 0;
    SmallVector<Operation*, 4> users;

    RegInterval(ascendc::RegTensorOp tensor) : tensor(tensor) {}
    ~RegInterval() = default;
};

struct ReuseRegTensorPass : public ascvf2::impl::ReuseRegTensorBase<ReuseRegTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        bool unsupportedControlFlow =
            func->walk([&](Operation* op) {
                    bool multiBlock =
                        llvm::any_of(op->getRegions(), [](Region& region) { return region.getBlocks().size() > 1; });
                    if (op->getNumRegions() > 1 || op->getNumSuccessors() != 0 || multiBlock)
                        return WalkResult::interrupt();
                    return WalkResult::advance();
                })
                .wasInterrupted();
        if (unsupportedControlFlow)
            return;

        DenseMap<Operation*, unsigned> order;
        unsigned index = 0;
        func->walk([&](Operation* op) { order[op] = index++; });

        llvm::MapVector<Type, SmallVector<RegInterval, 4>> intervals;
        SmallVector<ascendc::RegTensorOp, 4> unused;
        func.walk([&](ascendc::RegTensorOp tensor) {
            unsigned first = std::numeric_limits<unsigned>::max();
            unsigned last = 0;
            RegInterval interval(tensor);
            for (Operation* user : tensor.getResult().getUsers()) {
                if (user->getNumRegions() != 0 || user->mightHaveTrait<OpTrait::IsTerminator>())
                    return;
                unsigned userOrder = order.lookup(user);
                first = std::min(first, userOrder);
                last = std::max(last, userOrder);
                interval.users.push_back(user);
            }
            if (first == std::numeric_limits<unsigned>::max()) {
                unused.push_back(tensor);
                return;
            }
            interval.first = first;
            interval.last = last;

            DenseSet<Operation*> loops;
            for (Operation* user : interval.users) {
                for (Operation* parent = user->getParentOp(); parent; parent = parent->getParentOp()) {
                    if (isa<LoopLikeOpInterface>(parent))
                        loops.insert(parent);
                }
            }
            for (Operation* loop : loops) {
                Operation* firstUser = nullptr;
                for (Operation* user : interval.users) {
                    if (!loop->isAncestor(user))
                        continue;
                    if (!firstUser || order.lookup(user) < order.lookup(firstUser))
                        firstUser = user;
                }
                bool writtenFirst = isa<ascendc::RegOp>(firstUser) && firstUser->getNumOperands() != 0 &&
                                    firstUser->getOperand(0) == tensor.getResult() &&
                                    llvm::count(firstUser->getOperands(), tensor.getResult()) == 1;
                if (writtenFirst)
                    continue;

                Block& body = loop->getRegion(0).front();
                interval.first = std::min(interval.first, order.lookup(loop));
                interval.last = std::max(interval.last, order.lookup(&body.back()));
            }

            intervals[tensor.getResult().getType()].push_back(interval);
        });

        DominanceInfo dominance(func);
        for (auto& entry : intervals) {
            auto& typeIntervals = entry.second;
            llvm::stable_sort(
                typeIntervals, [](const RegInterval& lhs, const RegInterval& rhs) { return lhs.first < rhs.first; });

            SmallVector<SmallVector<RegInterval, 4>, 4> groups;
            for (const RegInterval& interval : typeIntervals) {
                auto* it = llvm::find_if(groups, [&](const SmallVector<RegInterval, 4>& group) {
                    return group.back().last < interval.first;
                });
                if (it == groups.end()) {
                    groups.emplace_back();
                    groups.back().push_back(interval);
                } else {
                    it->push_back(interval);
                }
            }

            for (auto& group : groups) {
                llvm::stable_sort(group, [&](const RegInterval& lhs, const RegInterval& rhs) {
                    return order.lookup(lhs.tensor) < order.lookup(rhs.tensor);
                });
                auto* representative = llvm::find_if(group, [&](const RegInterval& interval) {
                    return llvm::all_of(group, [&](const RegInterval& other) {
                        return llvm::all_of(
                            other.users, [&](Operation* user) { return dominance.dominates(interval.tensor, user); });
                    });
                });
                if (representative == group.end())
                    continue;

                for (auto& interval : group) {
                    if (interval.tensor == representative->tensor)
                        continue;
                    interval.tensor.getResult().replaceAllUsesWith(representative->tensor.getResult());
                    interval.tensor.erase();
                }
            }
        }

        for (ascendc::RegTensorOp tensor : unused)
            tensor.erase();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createReuseRegTensorPass() { return std::make_unique<ReuseRegTensorPass>(); }
