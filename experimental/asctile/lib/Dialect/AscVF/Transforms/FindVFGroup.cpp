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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_FINDVFGROUP
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

struct OpGroup {
    SmallVector<Operation*> ops;
    Type groupType;
    OpGroup(ArrayRef<Operation*> ops, Type groupType) : ops(ops), groupType(groupType) {}
};

bool isARPattern(ascendc::BroadcastOp bcastOp)
{
    // broadcast by last axis
    // TODO: add case multiple last dim
    // Ex. [12, 1, 1] -> [12, 34, 56] can broadcast
    // Now only support broadcast last dim. Ex: [96, 1] -> [96, 16]
    auto srcType = dyn_cast<ascendc::LocalTensorType>(bcastOp.getSrc().getType());
    auto dstType = dyn_cast<ascendc::LocalTensorType>(bcastOp.getDst().getType());
    if (!srcType || !dstType)
        return false;
    auto shape1 = SmallVector<int64_t>{srcType.getShape()};
    if (shape1.back() != 1)
        return false;
    auto shape2 = SmallVector<int64_t>{dstType.getShape()};
    shape1.pop_back();
    shape2.pop_back();
    return shape1 == shape2;
}

ascendc::LocalTensorType getType(Operation* op)
{
    return llvm::TypeSwitch<Operation*, ascendc::LocalTensorType>(op)
        .Case<ascendc::CompareScalarL2Op>([](ascendc::CompareScalarL2Op op) {
            auto type = dyn_cast<ascendc::LocalTensorType>(op.getSrc0().getType());
            assert(type);
            return type;
        })
        .Case<
            ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::VecScalarL2Op, ascendc::DuplicateL2Op,
            ascendc::BroadcastOp, ascendc::SelectL2Op>([](auto op) {
            auto type = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType());
            assert(type);
            return type;
        })
        .Case<
            ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op, ascendc::ReduceSumOp,
            ascendc::ReduceMaxOp>([](auto op) {
            auto type = dyn_cast<ascendc::LocalTensorType>(op.getSrc().getType());
            assert(type);
            return type;
        })
        .Default([](Operation* op) {
            op->dump();
            llvm_unreachable("was not expected this type");
            return ascendc::LocalTensorType{};
        });
}

bool isFusible(Operation* op)
{
    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case<ascendc::ReduceSumOp, ascendc::ReduceMaxOp, ascendc::ReduceMinOp>(
            [](auto reduceOp) { return reduceOp.getPattern() == ascendc::ReducePattern::AR; })
        .Case<ascendc::BroadcastOp>([](auto bcastOp) { return isARPattern(bcastOp); })
        .Case<
            ascendc::DuplicateL2Op,
            // Vector binary operations (L2)
            ascendc::AddL2Op, ascendc::AndL2Op, ascendc::DivL2Op, ascendc::FusedAbsSubL2Op, ascendc::FusedExpSubL2Op,
            ascendc::SubL2Op, ascendc::MaxL2Op, ascendc::MinL2Op, ascendc::MulL2Op, ascendc::MulAddDstL2Op,
            ascendc::OrL2Op, ascendc::PreluL2Op,
            // Vector unary operations (L2)
            ascendc::AbsL2Op, ascendc::ExpL2Op, ascendc::LnL2Op, ascendc::NegL2Op, ascendc::NotL2Op, ascendc::ReluL2Op,
            ascendc::SqrtL2Op,
            // Vector scalar operations (L2)
            ascendc::AddsL2Op, ascendc::MulsL2Op, ascendc::SubsL2Op, ascendc::DivsL2Op, ascendc::MaxsL2Op,
            ascendc::MinsL2Op, ascendc::LeakyReluL2Op, ascendc::ShiftLeftL2Op, ascendc::ShiftRightL2Op,
            ascendc::CompareScalarL2Op, ascendc::SelectL2Op>([](auto op) {
            if (auto opt = getConstantIntValue(op.getCalCount())) {
                auto type = getType(op);
                return opt.value() == type.getNumElements();
            }
            return false;
        })
        .Case<ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op>([](auto op) {
            if (auto opt = getConstantIntValue(op.getCount())) {
                auto type = getType(op);
                return opt.value() == type.getNumElements();
            }
            return false;
        })
        .Default([](Operation*) { return false; });
}

bool isSameGroup(Operation* firstOp, Operation* secondOp) { return getType(firstOp) == getType(secondOp); }

void findGroupsImpl(Region& region, std::vector<OpGroup>& groups)
{
    auto append = [&groups](SmallVectorImpl<Operation*>& ops) {
        if (!ops.empty()) {
            groups.emplace_back(ops, getType(ops.front()));
            ops.clear();
        }
    };

    SmallVector<Operation*> ops;
    for (auto& op : region.getOps()) {
        for (auto& nestedRegion : op.getRegions()) {
            findGroupsImpl(nestedRegion, groups);
        }
        if (isFusible(&op)) {
            if (!ops.empty() && !isSameGroup(ops.front(), &op)) {
                append(ops);
            }
            ops.emplace_back(&op);
        } else {
            append(ops);
        }
    }
    append(ops);
}

// Find binary_l2, unary_l2 operations that may be executed together
std::vector<OpGroup> findOperationGroups(Region& region)
{
    // 1. The same calCount
    // 2. Between ops absent other operations
    // 3. Contains more than 1 operation
    std::vector<OpGroup> groups;
    findGroupsImpl(region, groups);

    std::vector<OpGroup> filteredGroups;
    llvm::copy_if(
        groups, std::back_inserter(filteredGroups), [](const OpGroup& group) { return group.ops.size() >= 2; });
    return filteredGroups;
}

// Find local tensors that need be copy in RegTensor
ValueVector getInputLocalTensors(ArrayRef<Operation*> group)
{
    // If tensor is input but before it is output then don't insert her
    ValueMap<bool> isInputLocalTensor;
    for (auto* op : group) {
        if (auto opWithSrc = dyn_cast<ascendc::OpWithSrc>(op)) {
            for (auto src : opWithSrc.getSrcTensors()) {
                isInputLocalTensor.try_emplace(src, true);
            }
        } else if (auto duplicateOp = dyn_cast<ascendc::DuplicateL2Op>(op)) {
            auto scalar = duplicateOp.getScalar();
            if (isa<ascendc::LocalTensorType>(scalar.getType()))
                isInputLocalTensor.try_emplace(scalar, true);
        }
        if (auto opWithDst = dyn_cast<ascendc::OpWithDst>(op)) {
            for (auto dst : opWithDst.getDstTensors()) {
                isInputLocalTensor.try_emplace(dst, false);
            }
        }
    }
    ValueVector inputLocalTensors;
    for (const auto& [tensor, isInput] : isInputLocalTensor) {
        if (isInput)
            inputLocalTensors.emplace_back(tensor);
    }
    return ascvf::deduplicate(inputLocalTensors);
}

// Find local tensors that need be copy out from RegTensor
ValueVector getOutputLocalTensors(ArrayRef<Operation*> group)
{
    ValueVector outputLocalTensors;
    for (auto* op : group) {
        llvm::TypeSwitch<Operation*>(op)
            .Case<
                ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::VecScalarL2Op, ascendc::ReduceMaxL2Op,
                ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op, ascendc::ReduceSumOp, ascendc::ReduceMaxOp,
                ascendc::DuplicateL2Op, ascendc::BroadcastOp, ascendc::CompareScalarL2Op, ascendc::SelectL2Op>(
                [&](auto op) { outputLocalTensors.push_back(op.getDst()); });
    }
    return ascvf::deduplicate(outputLocalTensors);
}

SmallVector<int64_t> getSplitPosition(Operation* op)
{
    return llvm::TypeSwitch<Operation*, SmallVector<int64_t>>(op)
        .Case<
            ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::VecScalarL2Op, ascendc::ReduceMaxL2Op,
            ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op, ascendc::DuplicateL2Op, ascendc::CompareScalarL2Op,
            ascendc::SelectL2Op>([](auto) { return SmallVector<int64_t>{}; })
        .Case<ascendc::ReduceSumOp, ascendc::ReduceMaxOp, ascendc::BroadcastOp>(
            [](auto op) { return llvm::to_vector(llvm::seq<int64_t>(1, op.getSrc().getType().getRank())); });
}

// Fuse dimensions of group type
// Ex [12, 34, 56] -> [12 * 34 * 56]
Type fuseGroupType(OpGroup& group)
{
    auto groupType = cast<ascendc::LocalTensorType>(group.groupType);
    auto shape = groupType.getShape();
    std::set<int64_t> splits{0, static_cast<int64_t>(shape.size())};
    for (auto* op : group.ops) {
        auto vec = getSplitPosition(op);
        splits.insert(vec.begin(), vec.end());
    }
    SmallVector<int64_t> newShape;
    for (auto left = splits.begin(), right = std::next(left); right != splits.end();
         left = right, right = std::next(right)) {
        int64_t product = std::accumulate(shape.begin() + *left, shape.begin() + *right, 1L, std::multiplies());
        newShape.push_back(product);
    }
    return ascendc::LocalTensorType::get(newShape, groupType.getElementType());
}

ascvf::VFGroupOp wrapInVFGroupOp(OpGroup& group)
{
    assert(!group.ops.empty());
    auto& ops = group.ops;
    OpBuilder builder(ops.back());
    ValueVector inputs = getInputLocalTensors(ops);
    ValueVector outputs = getOutputLocalTensors(ops);

    auto fusedOp = builder.create<ascvf::VFGroupOp>(builder.getUnknownLoc(), outputs, inputs, fuseGroupType(group));
    auto& block = fusedOp.getRegion().emplaceBlock();

    builder.setInsertionPointToEnd(&block);
    for (auto* op : ops) {
        builder.clone(*op);
        op->erase();
    }
    builder.create<ascvf::YieldOp>(builder.getUnknownLoc());
    return fusedOp;
}

struct FindVFGroupPass : public ascvf::impl::FindVFGroupBase<FindVFGroupPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        for (auto& group : findOperationGroups(funcOp.getRegion())) {
            auto vfGroupOp = wrapInVFGroupOp(group);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createFindVFGroupPass() { return std::make_unique<FindVFGroupPass>(); }
