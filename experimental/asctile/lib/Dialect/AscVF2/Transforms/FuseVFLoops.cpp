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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Config/llvm-config.h"

#include <functional>

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_FUSEVFLOOPS
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;
using affine::AffineForOp;

namespace {

bool hasSupportedBody(AffineForOp loop)
{
    WalkResult walkResult = loop.walk([&](Operation* operation) {
        if (isa<AffineForOp, affine::AffineYieldOp, vector::MaskOp, vector::YieldOp, vector::TransferReadOp,
                vector::TransferWriteOp>(operation))
            return WalkResult::advance();
        if (isPure(operation))
            return WalkResult::advance();
        return WalkResult::interrupt();
    });
    return !walkResult.wasInterrupted();
}

bool haveSameValues(ValueRange first, ValueRange second)
{
    return first.size() == second.size() && llvm::all_of(llvm::zip_equal(first, second), [](auto pair) {
               return std::get<0>(pair) == std::get<1>(pair);
           });
}

bool haveSameDomain(AffineForOp first, AffineForOp second)
{
    return first.getLowerBoundMap() == second.getLowerBoundMap() &&
           haveSameValues(first.getLowerBoundOperands(), second.getLowerBoundOperands()) &&
           first.getUpperBoundMap() == second.getUpperBoundMap() &&
           haveSameValues(first.getUpperBoundOperands(), second.getUpperBoundOperands()) &&
           first.getStep() == second.getStep();
}

using ValueContext = DenseMap<Value, int64_t>;

struct LoopDomain {
    int64_t lower;
    int64_t upper;
    int64_t step;
};

struct IterationFootprints {
    DenseMap<int64_t, DenseSet<int64_t>> reads;
    DenseMap<int64_t, DenseSet<int64_t>> writes;
};

using LoopFootprints = DenseMap<Value, IterationFootprints>;

std::optional<int64_t> evaluateAffineMap(
    AffineMap map, ValueRange operands, const std::function<std::optional<int64_t>(Value)>& evaluateOperand)
{
    if (map.getNumSymbols() != 0 || map.getNumResults() != 1)
        return std::nullopt;
    SmallVector<int64_t, 4> values;
    values.reserve(operands.size());
    for (Value operand : operands) {
        auto value = evaluateOperand(operand);
        if (!value)
            return std::nullopt;
        values.push_back(*value);
    }
    if (map.getNumDims() != values.size())
        return std::nullopt;
    return map.compose(values).front();
}

std::optional<int64_t> evaluateAffineValue(Value value, const ValueContext& context)
{
    if (auto iterator = context.find(value); iterator != context.end())
        return iterator->second;
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
        if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
            return integer.getInt();
    }
    SmallVector<Value, 4> operands{value};
    AffineMap map = AffineMap::getMultiDimIdentityMap(1, value.getContext());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    return evaluateAffineMap(map, operands, [&](Value operand) { return evaluateAffineValue(operand, context); });
}

std::optional<LoopDomain> getLoopDomain(AffineForOp loop)
{
    auto evaluateBound = [](AffineMap map, ValueRange operands) {
        return evaluateAffineMap(map, operands, [](Value operand) {
            if (auto constant = operand.getDefiningOp<arith::ConstantOp>()) {
                if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
                    return std::optional<int64_t>{integer.getInt()};
            }
            return std::optional<int64_t>{};
        });
    };
    auto lower = evaluateBound(loop.getLowerBoundMap(), loop.getLowerBoundOperands());
    auto upper = evaluateBound(loop.getUpperBoundMap(), loop.getUpperBoundOperands());
    if (!lower || !upper || loop.getStep().getSExtValue() <= 0)
        return std::nullopt;
    return LoopDomain{*lower, *upper, loop.getStep().getSExtValue()};
}

std::optional<unsigned> getActiveLanes(Value mask, VectorType vectorType, const ValueContext& context)
{
    if (!mask)
        return vectorType.getDimSize(0);
    if (auto createMask = mask.getDefiningOp<vector::CreateMaskOp>()) {
        if (createMask.getVectorType().getRank() != 1 || createMask.getOperands().size() != 1)
            return std::nullopt;
        auto count = evaluateAffineValue(createMask.getOperands().front(), context);
        if (!count || *count < 0)
            return std::nullopt;
        return std::min<uint64_t>(*count, vectorType.getDimSize(0));
    }
    if (auto constantMask = mask.getDefiningOp<vector::ConstantMaskOp>()) {
        if (constantMask.getVectorType().getRank() != 1 || constantMask.getMaskDimSizes().size() != 1)
            return std::nullopt;
#if LLVM_VERSION_MAJOR >= 20
        int64_t count = constantMask.getMaskDimSizes()[0];
#else
        int64_t count = cast<IntegerAttr>(constantMask.getMaskDimSizes()[0]).getValue().getSExtValue();
#endif
        return std::min<uint64_t>(std::max<int64_t>(count, 0), vectorType.getDimSize(0));
    }
    return vectorType.getDimSize(0);
}

std::optional<DenseSet<int64_t>> getTransferOffsets(Operation* operation, const ValueContext& context)
{
    auto read = dyn_cast<vector::TransferReadOp>(operation);
    auto write = dyn_cast<vector::TransferWriteOp>(operation);
    if (!read && !write)
        return std::nullopt;

    Value source = read ? read.getSource() : write.getSource();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape() || !sourceType.getLayout().isIdentity())
        return std::nullopt;

    ArrayAttr inBounds = read ? read.getInBounds() : write.getInBounds();
    if (!llvm::all_of(inBounds, [](Attribute attr) { return cast<BoolAttr>(attr).getValue(); }))
        return std::nullopt;

    ValueRange indices = read ? read.getIndices() : write.getIndices();
    if (indices.size() != sourceType.getRank())
        return std::nullopt;
    SmallVector<int64_t, 4> indexValues;
    indexValues.reserve(indices.size());
    for (Value index : indices) {
        auto value = evaluateAffineValue(index, context);
        if (!value)
            return std::nullopt;
        indexValues.push_back(*value);
    }

    int64_t baseOffset = 0;
    if (auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>()) {
        ArrayRef<int64_t> offsets = reinterpret.getStaticOffsets();
        if (offsets.size() != 1)
            return std::nullopt;
        baseOffset = offsets.front();
    }
    for (auto [dimension, index] : llvm::enumerate(indexValues)) {
        int64_t stride = 1;
        for (size_t trailing = dimension + 1; trailing < sourceType.getRank(); ++trailing)
            stride *= sourceType.getShape()[trailing];
        baseOffset += index * stride;
    }

    AffineMap permutationMap = read ? read.getPermutationMap() : write.getPermutationMap();
    VectorType vectorType = read ? read.getVectorType() : write.getVector().getType();
    if (permutationMap.getNumResults() != 1 || vectorType.getRank() != 1)
        return std::nullopt;

    int64_t laneStride = 1;
    if (auto dimension = dyn_cast<AffineDimExpr>(permutationMap.getResult(0))) {
        if (dimension.getPosition() >= sourceType.getRank())
            return std::nullopt;
        for (size_t trailing = dimension.getPosition() + 1; trailing < sourceType.getRank(); ++trailing)
            laneStride *= sourceType.getShape()[trailing];
    } else if (!isa<AffineConstantExpr>(permutationMap.getResult(0))) {
        return std::nullopt;
    }

    Value mask = read ? read.getMask() : write.getMask();
    auto activeLanes = getActiveLanes(mask, vectorType, context);
    if (!activeLanes)
        return std::nullopt;
    DenseSet<int64_t> offsets;
    for (unsigned lane = 0; lane < *activeLanes; ++lane)
        offsets.insert(baseOffset + lane * laneStride);
    return offsets;
}

void collectFootprints(
    Operation* operation, int64_t primaryIteration, const ValueContext& context, LoopFootprints& footprints,
    bool& failed)
{
    if (auto loop = dyn_cast<AffineForOp>(operation)) {
        auto domain = getLoopDomain(loop);
        if (!domain)
            failed = true;
        if (!domain)
            return;
        ValueContext nestedContext = context;
        for (int64_t iteration = domain->lower; iteration < domain->upper; iteration += domain->step) {
            nestedContext[loop.getInductionVar()] = iteration;
            for (Operation& nested : loop.getBody()->getOperations())
                collectFootprints(&nested, primaryIteration, nestedContext, footprints, failed);
        }
        return;
    }

    for (Region& region : operation->getRegions())
        for (Block& block : region)
            for (Operation& nested : block.getOperations())
                collectFootprints(&nested, primaryIteration, context, footprints, failed);

    if (!isa<vector::TransferReadOp, vector::TransferWriteOp>(operation))
        return;
    Value source = isa<vector::TransferReadOp>(operation) ? cast<vector::TransferReadOp>(operation).getSource() :
                                                            cast<vector::TransferWriteOp>(operation).getSource();
    std::optional<Value> root = ascvf2::resolveTensorAliasRoot(source);
    auto offsets = getTransferOffsets(operation, context);
    if (!root || !offsets) {
        failed = true;
        return;
    }
    auto& iteration = isa<vector::TransferWriteOp>(operation) ? footprints[*root].writes[primaryIteration] :
                                                                footprints[*root].reads[primaryIteration];
    for (int64_t offset : *offsets)
        iteration.insert(offset);
}

std::optional<LoopFootprints> getLoopFootprints(AffineForOp loop, const ValueContext& context)
{
    auto domain = getLoopDomain(loop);
    if (!domain)
        return std::nullopt;

    LoopFootprints footprints;
    bool failed = false;
    for (int64_t iteration = domain->lower; iteration < domain->upper; iteration += domain->step) {
        ValueContext iterationContext = context;
        iterationContext[loop.getInductionVar()] = iteration;
        for (Operation& nested : loop.getBody()->getOperations())
            collectFootprints(&nested, iteration, iterationContext, footprints, failed);
    }
    if (failed)
        return std::nullopt;
    return footprints;
}

bool intersects(const DenseSet<int64_t>& lhs, const DenseSet<int64_t>& rhs)
{
    return llvm::any_of(lhs, [&](int64_t value) { return rhs.contains(value); });
}

bool haveCompatibleMemoryAccesses(AffineForOp first, AffineForOp second)
{
    SmallVector<AffineForOp, 4> enclosingLoops;
    for (Operation* parent = first->getParentOp(); parent; parent = parent->getParentOp())
        if (auto loop = dyn_cast<AffineForOp>(parent))
            enclosingLoops.push_back(loop);

    SmallVector<LoopDomain, 4> enclosingDomains;
    enclosingDomains.reserve(enclosingLoops.size());
    for (AffineForOp loop : enclosingLoops) {
        auto domain = getLoopDomain(loop);
        if (!domain)
            return false;
        enclosingDomains.push_back(*domain);
    }

    std::function<bool(size_t, const ValueContext&)> visitEnclosing = [&](size_t index, const ValueContext& context) {
        if (index == enclosingLoops.size()) {
            auto firstFootprints = getLoopFootprints(first, context);
            auto secondFootprints = getLoopFootprints(second, context);
            if (!firstFootprints || !secondFootprints)
                return false;

            for (auto& [root, firstRoot] : *firstFootprints) {
                auto secondRoot = secondFootprints->find(root);
                if (secondRoot == secondFootprints->end())
                    continue;
                for (auto& [firstIteration, firstRead] : firstRoot.reads) {
                    for (auto& [secondIteration, secondWrite] : secondRoot->second.writes) {
                        if (firstIteration != secondIteration && intersects(firstRead, secondWrite))
                            return false;
                    }
                }
                for (auto& [firstIteration, firstWrite] : firstRoot.writes) {
                    for (auto& [secondIteration, secondAccess] : secondRoot->second.reads) {
                        if (firstIteration != secondIteration && intersects(firstWrite, secondAccess))
                            return false;
                    }
                    for (auto& [secondIteration, secondWrite] : secondRoot->second.writes) {
                        if (firstIteration != secondIteration && intersects(firstWrite, secondWrite))
                            return false;
                    }
                }
            }
            return true;
        }

        const LoopDomain& domain = enclosingDomains[index];
        for (int64_t iteration = domain.lower; iteration < domain.upper; iteration += domain.step) {
            ValueContext nestedContext = context;
            auto [iterator, inserted] = nestedContext.try_emplace(enclosingLoops[index].getInductionVar(), iteration);
            (void)iterator;
            (void)inserted;
            if (!visitEnclosing(index + 1, nestedContext))
                return false;
        }
        return true;
    };

    return visitEnclosing(0, ValueContext());
}

bool isDefinedIn(Value value, AffineForOp loop)
{
    if (Operation* operation = value.getDefiningOp())
        return loop->isAncestor(operation);
    auto blockArgument = cast<BlockArgument>(value);
    return loop.getRegion().isAncestor(blockArgument.getOwner()->getParent());
}

bool secondUsesFirstResult(AffineForOp first, AffineForOp second)
{
    return second
        ->walk([&](Operation* operation) {
            return llvm::any_of(
                       operation->getOpOperands(),
                       [&](OpOperand& operand) { return isDefinedIn(operand.get(), first); }) ?
                       WalkResult::interrupt() :
                       WalkResult::advance();
        })
        .wasInterrupted();
}

bool canFuse(AffineForOp first, AffineForOp second)
{
    return first->getBlock() == second->getBlock() && std::next(first->getIterator()) == second->getIterator() &&
           haveSameDomain(first, second) && hasSupportedBody(first) && hasSupportedBody(second) &&
           haveCompatibleMemoryAccesses(first, second) && !secondUsesFirstResult(first, second);
}

AffineForOp fuse(AffineForOp first, AffineForOp second)
{
    SmallVector<Value, 4> inits(first.getInits());
    llvm::append_range(inits, second.getInits());

    OpBuilder builder(first);
    auto fused = builder.create<AffineForOp>(
        first.getLoc(), first.getLowerBoundOperands(), first.getLowerBoundMap(), first.getUpperBoundOperands(),
        first.getUpperBoundMap(), first.getStep().getSExtValue(), inits);

    first.getInductionVar().replaceAllUsesWith(fused.getInductionVar());
    for (auto [oldArgument, newArgument] :
         llvm::zip_equal(first.getRegionIterArgs(), fused.getRegionIterArgs().take_front(first.getNumResults())))
        oldArgument.replaceAllUsesWith(newArgument);
    second.getInductionVar().replaceAllUsesWith(fused.getInductionVar());
    for (auto [oldArgument, newArgument] :
         llvm::zip_equal(second.getRegionIterArgs(), fused.getRegionIterArgs().take_back(second.getNumResults())))
        oldArgument.replaceAllUsesWith(newArgument);

    Block* fusedBody = fused.getBody();
    Operation* insertionTerminator = nullptr;
    if (!fusedBody->empty() && isa<affine::AffineYieldOp>(fusedBody->back()))
        insertionTerminator = &fusedBody->back();
    else {
        OpBuilder terminatorBuilder(fusedBody, fusedBody->end());
        insertionTerminator = terminatorBuilder.create<affine::AffineYieldOp>(fused.getLoc(), ValueRange{});
    }

    fusedBody->getOperations().splice(
        insertionTerminator->getIterator(), first.getBody()->getOperations(), first.getBody()->begin(),
        Block::iterator(first.getBody()->getTerminator()));
    fusedBody->getOperations().splice(
        insertionTerminator->getIterator(), second.getBody()->getOperations(), second.getBody()->begin(),
        Block::iterator(second.getBody()->getTerminator()));

    SmallVector<Value, 8> yieldedValues(cast<affine::AffineYieldOp>(first.getBody()->getTerminator()).getOperands());
    llvm::append_range(yieldedValues, cast<affine::AffineYieldOp>(second.getBody()->getTerminator()).getOperands());
    OpBuilder yieldBuilder(insertionTerminator);
    yieldBuilder.create<affine::AffineYieldOp>(fused.getLoc(), yieldedValues);
    insertionTerminator->erase();

    for (auto [oldResult, newResult] :
         llvm::zip_equal(first.getResults(), fused.getResults().take_front(first.getNumResults())))
        oldResult.replaceAllUsesWith(newResult);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(second.getResults(), fused.getResults().take_back(second.getNumResults())))
        oldResult.replaceAllUsesWith(newResult);

    first.erase();
    second.erase();
    return fused;
}

void fuseBlock(Block* block)
{
    for (Block::iterator it = block->begin(); it != block->end();) {
        auto current = dyn_cast<AffineForOp>(&*it);
        if (!current) {
            ++it;
            continue;
        }

        while (std::next(current->getIterator()) != block->end()) {
            auto next = dyn_cast<AffineForOp>(&*std::next(current->getIterator()));
            fuseBlock(current.getBody());
            if (!next || !canFuse(current, next))
                break;
            current = fuse(current, next);
            fuseBlock(current.getBody());
        }
        fuseBlock(current.getBody());
        it = std::next(current->getIterator());
    }
}

struct FuseVFLoopsPass : public ascvf2::impl::FuseVFLoopsBase<FuseVFLoopsPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        fuseBlock(&func.getBody().front());
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createFuseVFLoopsPass() { return std::make_unique<FuseVFLoopsPass>(); }
