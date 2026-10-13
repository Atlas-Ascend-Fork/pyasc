/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscTile/Utils/Attributes.h"
#include "asctile/Dialect/AscTile/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_APPLYCVSTRATEGY
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

std::pair<DistribModeAttr, ArrayRef<int64_t>> getSplitInfo(Operation* op)
{
    if (auto needSplit = op->getAttrOfType<DistribModeAttr>(attr::needSplit))
        return {needSplit, op->getAttrOfType<DenseI64ArrayAttr>(attr::splitShape).asArrayRef()};
    return {DistribModeAttr{}, std::nullopt};
}

SmallVector<Value, 2> splitOffsets(OpBuilder& builder, ValueRange offsets, DistribMode split, ArrayRef<int64_t> shape)
{
    auto rank = shape.size();
    assert(offsets.size() == rank && "offset count must match tensor rank");
    SmallVector<Value, 2> newOffsets;
    auto iType = offsets.front().getType();
    auto loc = builder.getUnknownLoc();
    Value subBlockIdx = builder.create<ascendc::GetSubBlockIdxOp>(loc, iType);
    unsigned axis = getSplitAxis(split, rank);
    for (auto [index, offset] : llvm::enumerate(offsets)) {
        if (index != axis) {
            newOffsets.push_back(offset);
            continue;
        }
        Value halfSize = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(iType, shape[axis]));
        Value addOffset = builder.create<arith::MulIOp>(loc, halfSize, subBlockIdx);
        newOffsets.push_back(builder.create<arith::AddIOp>(loc, offset, addOffset));
    }
    return newOffsets;
}

Value splitTensor(Value opnd, DistribMode split, ArrayRef<int64_t> sizes, ConversionPatternRewriter& rewriter)
{
    auto tensor = dyn_cast<LocalTensorType>(opnd.getType());
    if (!tensor || tensor.getShape() == sizes)
        return opnd;
    assert(tensor.getLoc() == TensorLocation::UB && "tensor.extract_slice must use UB tensor");
    auto newType = tensor.clone(sizes);
    OpFoldResult splat = getSplatValue(opnd);
    if (auto attr = dyn_cast_if_present<Attribute>(splat))
        return rewriter.create<arith::ConstantOp>(opnd.getLoc(), SplatElementsAttr::get(newType, attr));
    if (auto value = dyn_cast_if_present<Value>(splat))
        return rewriter.create<tensor::SplatOp>(opnd.getLoc(), newType, value);
    OpBuilder::InsertionGuard guard(rewriter);
    if (Operation* defOp = opnd.getDefiningOp())
        rewriter.setInsertionPointAfter(defOp);
    ascir::ConstantOpBuilder consts(rewriter);
    SmallVector<Value, 2> zeros(sizes.size(), consts.index(0));
    auto offsets = splitOffsets(rewriter, zeros, split, sizes);
    auto staticOffsets = rewriter.getDenseI64ArrayAttr(SmallVector(sizes.size(), ShapedType::kDynamic));
    SmallVector<int64_t> strides = computeStrides(sizes);
    return rewriter.create<tensor::ExtractSliceOp>(
        opnd.getLoc(), newType, opnd, offsets, /*sizes*/ ValueRange{}, /*strides*/ ValueRange{}, staticOffsets,
        rewriter.getDenseI64ArrayAttr(sizes), rewriter.getDenseI64ArrayAttr(strides));
}

SmallVector<int64_t, 2> getOperandSplitShape(Value operand, DistribMode split, ArrayRef<int64_t> resultShape)
{
    auto shape = llvm::to_vector<2>(cast<LocalTensorType>(operand.getType()).getShape());
    shape[getSplitAxis(split, shape.size())] = resultShape[getSplitAxis(split, resultShape.size())];
    return shape;
}

struct LoopCarriedSplit {
    unsigned index;
    DistribMode split;
    SmallVector<int64_t, 2> shape;
};

bool collectLoopCarriedSplits(scf::ForOp loop, SmallVectorImpl<LoopCarriedSplit>& splits)
{
    ValueRange yieldedValues = loop.getYieldedValues();
    auto loopResults = loop.getLoopResults();
    if (yieldedValues.empty() || !loopResults || yieldedValues.size() != loopResults->size())
        return false;
    for (auto [index, yieldedValue] : llvm::enumerate(yieldedValues)) {
        auto* producer = yieldedValue.getDefiningOp();
        if (!producer)
            continue;
        if (auto cvStrategy = dyn_cast<CVStrategyOp>(producer)) {
            if (cvStrategy->getRegion(0).empty())
                continue;
            auto cvYield = cast<YieldOp>(cvStrategy.getBody()->getTerminator());
            producer = cvYield->getOperand(cast<OpResult>(yieldedValue).getResultNumber()).getDefiningOp();
            if (!producer)
                continue;
        }
        auto [split, shape] = getSplitInfo(producer);
        if (!split)
            continue;
        auto resultType = dyn_cast<LocalTensorType>(yieldedValue.getType());
        if (!resultType || shape.size() != resultType.getRank())
            continue;
        splits.push_back({static_cast<unsigned>(index), split.getValue(), llvm::to_vector<2>(shape)});
    }
    return !splits.empty();
}

struct SplitCopy : OpConversionPattern<CopyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CopyOp op, CopyOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, shape] = getSplitInfo(op);
        auto srcLoc = op.getBase().getType().getLoc();
        auto dstLoc = op.getType().getLoc();
        auto src = rewriter.getRemappedValue(op.getBase());
        if (srcLoc == TensorLocation::L0C && dstLoc == TensorLocation::UB) {
            rewriter.replaceOpWithNewOp<CopyOp>(op, op.getType().clone(shape), src, op.getOffsets(), split);
            return success();
        }
        if (srcLoc == TensorLocation::UB && dstLoc == TensorLocation::L1) {
            auto distrib = split.getValue() == DistribMode::SplitByM ? DistribMode::JoinByM : DistribMode::JoinByN;
            rewriter.replaceOpWithNewOp<CopyOp>(
                op, op.getType(), src, op.getOffsets(), rewriter.getAttr<DistribModeAttr>(distrib));
            return success();
        }
        return op->emitOpError("is not eligible for the CV strategy with splitting");
    }
};

struct SplitStore : OpConversionPattern<StoreOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StoreOp op, StoreOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, shape] = getSplitInfo(op);
        auto offsets = splitOffsets(rewriter, op.getOffsets(), split.getValue(), shape);
        if (auto realShape = op.getRealShape(); !realShape.empty()) {
            // TODO: Implement real shape clamping
            return op->emitOpError("with real shape is not eligible for the CV strategy with splitting");
        }
        rewriter.replaceOpWithNewOp<StoreOp>(
            op, rewriter.getRemappedValue(op.getValue()), op.getBase(), offsets, ValueRange{});
        return success();
    }
};

struct SplitElementwise : OpTraitConversionPattern<OpTrait::Elementwise> {
    using OpTraitConversionPattern::OpTraitConversionPattern;

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const override
    {
        if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
            return op->emitOpError("is not expected inside asctile.cv_strategy body");
        auto [split, shape] = getSplitInfo(op);
        SmallVector<Value, 4> operands;
        for (Value opnd : op->getOperands()) {
            operands.push_back(splitTensor(rewriter.getRemappedValue(opnd), split.getValue(), shape, rewriter));
        }
        SmallVector<Type, 2> results;
        for (Type result : op->getResultTypes()) {
            auto tensor = dyn_cast<LocalTensorType>(result);
            if (!tensor || tensor.getShape() == shape)
                results.push_back(result);
            else
                results.push_back(tensor.clone(shape));
        }
        auto* newOp = rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands, results, op->getAttrs());
        newOp->removeAttr(attr::needSplit);
        newOp->removeAttr(attr::splitShape);
        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct SplitReduce : OpConversionPattern<ReduceOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReduceOp op, ReduceOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getOperand(), split.getValue(), resultShape);
        auto operand =
            splitTensor(rewriter.getRemappedValue(op.getOperand()), split.getValue(), operandShape, rewriter);
        auto resultType = op.getType().clone(resultShape);
        rewriter.replaceOpWithNewOp<ReduceOp>(op, resultType, operand, op.getDims(), op.getKindAttr());
        return success();
    }
};

struct SplitReshape : OpConversionPattern<ReshapeOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReshapeOp op, ReshapeOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getIn(), split.getValue(), resultShape);
        auto operand = splitTensor(rewriter.getRemappedValue(op.getIn()), split.getValue(), operandShape, rewriter);
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType().clone(resultShape), operand);
        return success();
    }
};

struct SplitBroadcast : OpConversionPattern<BroadcastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        BroadcastOp op, BroadcastOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getOperand(), split.getValue(), resultShape);
        auto operand =
            splitTensor(rewriter.getRemappedValue(op.getOperand()), split.getValue(), operandShape, rewriter);
        rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType().clone(resultShape), operand);
        return success();
    }
};

struct SplitFor : OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        scf::ForOp op, scf::ForOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        SmallVector<LoopCarriedSplit, 4> splits;
        if (!collectLoopCarriedSplits(op, splits))
            return failure();
        for (const auto& split : splits) {
            Value init = op.getInitArgs()[split.index];
            Value splitInit = splitTensor(rewriter.getRemappedValue(init), split.split, split.shape, rewriter);
            auto splitType = cast<LocalTensorType>(op.getResult(split.index).getType()).clone(split.shape);
            rewriter.startOpModification(op);
            op->setOperand(op.getNumControlOperands() + split.index, splitInit);
            op.getResult(split.index).setType(splitType);
            op.getRegionIterArg(split.index).setType(splitType);
            rewriter.finalizeOpModification(op);
        }
        return success();
    }
};

template <typename OpT>
struct ConvertOperands : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        SmallVector<Value, 8> operands;
        if (rewriter.getRemappedValues(op->getOperands(), operands).failed())
            return failure();
        auto newOp = rewriter.replaceOpWithNewOp<OpT>(op, op->getResultTypes(), operands, op->getAttrs());
        newOp->removeAttr(attr::needSplit);
        newOp->removeAttr(attr::splitShape);
        return success();
    }
};

struct InlineCVStrategy : OpConversionPattern<CVStrategyOp> {
    using OpConversionPattern<CVStrategyOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        CVStrategyOp op, CVStrategyOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        Block* body = op.getBody();
        Operation* yieldOp = body->getTerminator();
        ValueRange results = yieldOp->getOperands();
        rewriter.inlineBlockBefore(body, op);
        rewriter.replaceOp(op, results);
        rewriter.eraseOp(yieldOp);
        return success();
    }
};

struct ApplyCVStrategyPass : public asctile::impl::ApplyCVStrategyBase<ApplyCVStrategyPass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<CVStrategyOp>();
        target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op) {
            SmallVector<LoopCarriedSplit, 4> splits;
            if (!collectLoopCarriedSplits(op, splits))
                return true;
            for (const auto& split : splits) {
                auto resultType = cast<LocalTensorType>(op.getResult(split.index).getType());
                if (resultType.getShape() != ArrayRef(split.shape))
                    return false;
            }
            return true;
        });
        target.markUnknownOpDynamicallyLegal([](Operation* op) { return !op->hasAttr(attr::needSplit); });
        RewritePatternSet patterns(context);
        patterns.add<
            SplitCopy, SplitStore, SplitElementwise, SplitReduce, SplitReshape, SplitBroadcast, SplitFor,
            InlineCVStrategy, ConvertOperands<DumpTensorOp>, ConvertOperands<InlineOp>, ConvertOperands<YieldOp>,
            ConvertOperands<scf::YieldOp>>(context);
        DenseSet<Operation*> unlegalizedOps;
        ConversionConfig config;
        config.unlegalizedOps = &unlegalizedOps;
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed()) {
            if (unlegalizedOps.empty())
                emitError(funcOp.getLoc(), "CV strategy application failed");
            for (auto* op : unlegalizedOps)
                op->emitOpError("prevents the surrounding CV strategy from application");
            signalPassFailure();
            return;
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createApplyCVStrategyPass() { return std::make_unique<ApplyCVStrategyPass>(); }
