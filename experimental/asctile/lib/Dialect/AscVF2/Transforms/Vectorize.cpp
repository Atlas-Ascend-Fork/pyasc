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
#include "asctile/Dialect/AscVF2/IR/AscVF2.h"
#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_VECTORIZE
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct VFInfo {
    Operation* operation;

    FailureOr<unsigned> getVectorWidth(Type elementType) const
    {
        return ascvf2::getVFVectorWidth(operation, elementType);
    }
};

struct ChunkContext {
    SmallVector<Value> indices;
    AffineMap transferMap;
    ArrayAttr inBounds;
    Value mask;
};

class RewriterAdaptor {
    ConversionPatternRewriter& rewriter;

public:
    explicit RewriterAdaptor(ConversionPatternRewriter& rewriter) : rewriter(rewriter) {}

    ConversionPatternRewriter& operator*() { return rewriter; }
    ConversionPatternRewriter* operator->() { return &rewriter; }

    Value createBuffer(Value tensor)
    {
        auto tensorType = cast<asctile::LocalTensorType>(tensor.getType());
        return rewriter.create<ascvf2::ToBufferOp>(tensor.getLoc(), ascvf2::getUBBufferType(tensorType), tensor);
    }

    Value createTensor(Operation* op, Value buffer, asctile::LocalTensorType type)
    {
        auto result =
            asctile::LocalTensorType::get(type.getShape(), type.getElementType(), asctile::TensorLocation::UB);
        return rewriter.create<ascvf2::ToTensorOp>(op->getLoc(), result, buffer);
    }

    Value createDestination(Operation* op, asctile::LocalTensorType type)
    {
        return rewriter.create<memref::AllocaOp>(op->getLoc(), ascvf2::getUBBufferType(type));
    }

    Value createFlatBuffer(Operation* op, Value buffer)
    {
        auto bufferType = cast<MemRefType>(buffer.getType());
        if (bufferType.getRank() == 1)
            return buffer;

        int64_t count = 1;
        for (int64_t dimension : bufferType.getShape())
            count *= dimension;
        auto flatType = MemRefType::get(
            {count}, bufferType.getElementType(), AffineMap::getMultiDimIdentityMap(1, op->getContext()),
            bufferType.getMemorySpace());
        return rewriter.create<memref::ReinterpretCastOp>(
            op->getLoc(), flatType, buffer, 0, ArrayRef<int64_t>{count}, ArrayRef<int64_t>{1});
    }

    ChunkContext createChunkContext(
        Operation* op, asctile::LocalTensorType resultType, unsigned vecWidth, unsigned hwWidth)
    {
        ChunkContext context;
        for (int64_t dimension : resultType.getShape().drop_back()) {
            auto loop = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, dimension, 1);
            rewriter.setInsertionPointToStart(loop.getBody());
            context.indices.push_back(loop.getInductionVar());
        }
        auto chunkLoop = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, resultType.getShape().back(), vecWidth);
        context.indices.push_back(chunkLoop.getInductionVar());
        rewriter.setInsertionPointToStart(chunkLoop.getBody());
        int64_t chunkSize = resultType.getShape().back();
        if (chunkSize % hwWidth != 0) {
            AffineExpr remainingExpr = chunkSize - rewriter.getAffineDimExpr(0);
            Value remaining = affine::makeComposedAffineApply(
                rewriter, op->getLoc(), AffineMap::get(1, 0, remainingExpr, op->getContext()),
                ArrayRef<OpFoldResult>{chunkLoop.getInductionVar()});
            context.mask = rewriter.create<vector::CreateMaskOp>(
                op->getLoc(), VectorType::get({vecWidth}, rewriter.getI1Type()), ArrayRef<OpFoldResult>{remaining});
        }
        context.transferMap = AffineMap::get(
            resultType.getRank(), 0, rewriter.getAffineDimExpr(resultType.getRank() - 1), op->getContext());
        context.inBounds = rewriter.getBoolArrayAttr({true});
        return context;
    }

    ChunkContext createFlatContext(
        Operation* op, asctile::LocalTensorType resultType, unsigned vecWidth, unsigned hwWidth)
    {
        ChunkContext context;
        int64_t count = resultType.getNumElements();
        auto chunkLoop = rewriter.create<affine::AffineForOp>(op->getLoc(), 0, count, vecWidth);
        context.indices.push_back(chunkLoop.getInductionVar());
        rewriter.setInsertionPointToStart(chunkLoop.getBody());
        if (count % hwWidth != 0) {
            AffineExpr remainingExpr = count - rewriter.getAffineDimExpr(0);
            Value remaining = affine::makeComposedAffineApply(
                rewriter, op->getLoc(), AffineMap::get(1, 0, remainingExpr, op->getContext()),
                ArrayRef<OpFoldResult>{chunkLoop.getInductionVar()});
            context.mask = rewriter.create<vector::CreateMaskOp>(
                op->getLoc(), VectorType::get({vecWidth}, rewriter.getI1Type()), ArrayRef<OpFoldResult>{remaining});
        }
        context.transferMap = AffineMap::getMultiDimIdentityMap(1, op->getContext());
        context.inBounds = rewriter.getBoolArrayAttr({true});
        return context;
    }

    Operation* createMaskedOp(Operation* op, ValueRange operands, VectorType vectorType, Value mask)
    {
        std::string vectorOpName = op->getName().getStringRef().str();
        if (op->getDialect()->getNamespace() == "asctile") {
            StringRef mnemonic = op->getName().getStringRef().drop_front(strlen("asctile."));
            vectorOpName = mnemonic == "bitwise_not" ? "ascvf2.not" : ("ascvf2." + mnemonic.str());
        }
        OperationState state(op->getLoc(), vectorOpName);
        state.addOperands(operands);
        state.addTypes(TypeRange{vectorType});
        state.addAttributes(op->getAttrs());
        Operation* vectorOperation = rewriter.create(state);
        if (mask)
            return vector::maskOperation(rewriter, vectorOperation, mask);
        return vectorOperation;
    }

    void createWrite(Operation* op, Value vectorResult, Value dstBuf, const ChunkContext& context)
    {
        rewriter.create<vector::TransferWriteOp>(
            op->getLoc(), vectorResult, dstBuf, context.indices, AffineMapAttr::get(context.transferMap), context.mask,
            context.inBounds);
    }
};

struct ConvertElementwise : public ConversionPattern {
    ConvertElementwise(TypeConverter& typeConverter, MLIRContext* context, const VFInfo& vfInfo)
        : ConversionPattern(typeConverter, Pattern::MatchAnyOpTypeTag(), PatternBenefit(1), context), vfInfo(vfInfo)
    {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const override
    {
        if (isa<asctile::BroadcastOp>(op))
            return failure();
        if (!ascvf2::isVectorizableGroupOperation(op))
            return failure();
        RewriterAdaptor adaptor(rewriter);
        adaptor->setInsertionPoint(op);
        auto resultType = cast<asctile::LocalTensorType>(op->getResult(0).getType());
        if (resultType.getRank() == 0)
            return op->emitOpError("cannot vectorize a zero-ranked tensor");
        auto hwWidth = vfInfo.getVectorWidth(resultType.getElementType());
        if (failed(hwWidth) || *hwWidth == 0)
            return op->emitError("cannot derive a non-zero vector width for the operation element type");
        int64_t elementCount = resultType.getNumElements();
        bool shapePreserving = hasTrailingBroadcast(op);
        auto vecWidth = std::min(
            *hwWidth, shapePreserving ? static_cast<unsigned>(resultType.getShape().back()) :
                                        static_cast<unsigned>(elementCount));
        auto vectorType = VectorType::get({vecWidth}, resultType.getElementType());
        Value dstBuf = adaptor.createDestination(op, resultType);
        Value flatDestinationBuffer = adaptor.createFlatBuffer(op, dstBuf);
        SmallVector<Value, 4> operandValues;
        operandValues.reserve(op->getNumOperands());
        for (Value operand : op->getOperands()) {
            Value value = rewriter.getRemappedValue(operand);
            if (isa<asctile::LocalTensorType>(value.getType()))
                value = adaptor.createBuffer(value);
            if (isa<MemRefType>(value.getType()) && !shapePreserving)
                value = adaptor.createFlatBuffer(op, value);
            operandValues.push_back(value);
        }
        ChunkContext context = shapePreserving ? adaptor.createChunkContext(op, resultType, vecWidth, *hwWidth) :
                                                 adaptor.createFlatContext(op, resultType, vecWidth, *hwWidth);
        Value padding =
            adaptor->create<arith::ConstantOp>(op->getLoc(), adaptor->getZeroAttr(resultType.getElementType()));
        SmallVector<Value, 4> vectorOperands;
        vectorOperands.reserve(operandValues.size());
        for (Value value : operandValues) {
            if (isa<MemRefType>(value.getType())) {
                value = adaptor->create<vector::TransferReadOp>(
                    op->getLoc(), vectorType, value, context.indices, context.transferMap, padding, context.mask,
                    context.inBounds);
            }
            vectorOperands.push_back(value);
        }
        Operation* maskedOp = adaptor.createMaskedOp(op, vectorOperands, vectorType, context.mask);
        adaptor.createWrite(op, maskedOp->getResult(0), shapePreserving ? dstBuf : flatDestinationBuffer, context);
        adaptor->setInsertionPoint(op);
        Value result = adaptor.createTensor(op, dstBuf, resultType);
        rewriter.replaceOp(op, result);
        return success();
    }

private:
    const VFInfo& vfInfo;

    bool hasTrailingBroadcast(Operation* op) const
    {
        auto resultType = cast<asctile::LocalTensorType>(op->getResult(0).getType());
        return llvm::any_of(op->getOperands(), [&](Value operand) {
            auto broadcastOp = operand.getDefiningOp<asctile::BroadcastOp>();
            if (!broadcastOp)
                return false;
            auto operandType = cast<asctile::LocalTensorType>(operand.getType());
            auto sourceType = cast<asctile::LocalTensorType>(broadcastOp.getOperand().getType());
            return operandType.getRank() == sourceType.getRank() && operandType.getRank() == resultType.getRank() &&
                   operandType.getShape().drop_back() == sourceType.getShape().drop_back() &&
                   operandType.getShape().drop_back() == resultType.getShape().drop_back() &&
                   sourceType.getShape().back() == 1 && operandType.getShape().back() > 1;
        });
    }
};

struct ConvertBroadcast : public OpConversionPattern<asctile::BroadcastOp> {
    using OpConversionPattern<asctile::BroadcastOp>::OpConversionPattern;
    using typename OpConversionPattern<asctile::BroadcastOp>::OpAdaptor;

    ConvertBroadcast(TypeConverter& typeConverter, MLIRContext* context, const VFInfo& vfInfo)
        : OpConversionPattern<asctile::BroadcastOp>(typeConverter, context), vfInfo(vfInfo)
    {}

    LogicalResult matchAndRewrite(
        asctile::BroadcastOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!ascvf2::isVectorizableGroupOperation(op))
            return failure();
        RewriterAdaptor adaptor(rewriter);
        adaptor->setInsertionPoint(op);
        auto resultType = cast<asctile::LocalTensorType>(op.getResult().getType());
        auto sourceType = cast<asctile::LocalTensorType>(op.getOperand().getType());
        auto hwWidth = vfInfo.getVectorWidth(resultType.getElementType());
        if (failed(hwWidth) || *hwWidth == 0)
            return op->emitError("cannot derive a non-zero vector width for the broadcast element type");
        bool scalarSource = sourceType.getNumElements() == 1;
        auto vecWidth = std::min(
            *hwWidth, static_cast<unsigned>(scalarSource ? resultType.getNumElements() : resultType.getShape().back()));

        auto vectorType = VectorType::get({vecWidth}, resultType.getElementType());
        Value dstBuf = adaptor.createDestination(op, resultType);
        Value flatDestinationBuffer = scalarSource ? adaptor.createFlatBuffer(op, dstBuf) : dstBuf;
        Value sourceTensor = rewriter.getRemappedValue(op.getOperand());
        Value srcBuf = adaptor.createBuffer(sourceTensor);
        if (scalarSource)
            srcBuf = adaptor.createFlatBuffer(op, srcBuf);
        ChunkContext context = scalarSource ? adaptor.createFlatContext(op, resultType, vecWidth, *hwWidth) :
                                              adaptor.createChunkContext(op, resultType, vecWidth, *hwWidth);
        Value padding =
            adaptor->create<arith::ConstantOp>(op.getLoc(), adaptor->getZeroAttr(resultType.getElementType()));

        if (scalarSource) {
            AffineMap broadcastMap = AffineMap::get(1, 0, getAffineConstantExpr(0, op.getContext()));
            Value sourceIndex = adaptor->create<arith::ConstantIndexOp>(op->getLoc(), 0);
            Value broadcastVector = rewriter.create<vector::TransferReadOp>(
                op->getLoc(), vectorType, srcBuf, ValueRange{sourceIndex}, AffineMapAttr::get(broadcastMap), padding,
                Value{}, context.inBounds);
            adaptor.createWrite(op, broadcastVector, flatDestinationBuffer, context);

            adaptor->setInsertionPoint(op);
            Value result = adaptor.createTensor(op, dstBuf, resultType);
            rewriter.replaceOp(op, result);
            return success();
        }

        auto scalarVectorType = VectorType::get({1}, resultType.getElementType());
        SmallVector<Value> sourceIndices;
        unsigned resultOffset = resultType.getRank() - sourceType.getRank();
        for (size_t dimension = 0; dimension < sourceType.getRank() - 1; ++dimension)
            sourceIndices.push_back(context.indices[resultOffset + dimension]);
        sourceIndices.push_back(adaptor->create<arith::ConstantIndexOp>(op.getLoc(), 0));
        AffineMap sourceMap = AffineMap::get(
            sourceType.getRank(), 0, rewriter.getAffineDimExpr(sourceType.getRank() - 1), op.getContext());
        Value scalarVector = rewriter.create<vector::TransferReadOp>(
            op.getLoc(), scalarVectorType, srcBuf, sourceIndices, sourceMap, padding, Value{}, context.inBounds);
        Value broadcastVector = rewriter.create<vector::BroadcastOp>(op.getLoc(), vectorType, scalarVector);
        adaptor.createWrite(op, broadcastVector, dstBuf, context);

        adaptor->setInsertionPoint(op);
        Value result = adaptor.createTensor(op, dstBuf, resultType);
        rewriter.replaceOp(op, result);
        return success();
    }

private:
    const VFInfo& vfInfo;
};

Attribute getNeutral(asctile::ReduceKind kind, Type elementType)
{
    if (elementType.isF32() || elementType.isF16()) {
        llvm::APFloat value =
            llvm::APFloat::getZero(elementType.isF32() ? llvm::APFloat::IEEEsingle() : llvm::APFloat::IEEEhalf());
        if (kind == asctile::ReduceKind::Max)
            value = llvm::APFloat::getInf(value.getSemantics(), true);
        else if (kind == asctile::ReduceKind::Min)
            value = llvm::APFloat::getInf(value.getSemantics(), false);
        return FloatAttr::get(elementType, value);
    }
    if (elementType.isInteger(32)) {
        llvm::APInt value = llvm::APInt::getZero(32);
        if (kind == asctile::ReduceKind::Max)
            value = llvm::APInt::getSignedMinValue(32);
        else if (kind == asctile::ReduceKind::Min)
            value = llvm::APInt::getSignedMaxValue(32);
        return IntegerAttr::get(elementType, value);
    }
    return {};
}

Operation* createAcc(OpBuilder& builder, Location loc, asctile::ReduceKind kind, Value acc, Value value)
{
    auto vector = cast<VectorType>(acc.getType());
    auto elementType = vector.getElementType();
    if (elementType.isF32() || elementType.isF16()) {
        if (kind == asctile::ReduceKind::Max)
            return builder.create<arith::MaximumFOp>(loc, vector, acc, value);
        if (kind == asctile::ReduceKind::Min)
            return builder.create<arith::MinimumFOp>(loc, vector, acc, value);
        return builder.create<arith::AddFOp>(loc, vector, acc, value);
    }
    if (elementType.isInteger(32)) {
        if (kind == asctile::ReduceKind::Max)
            return builder.create<arith::MaxSIOp>(loc, vector, acc, value);
        if (kind == asctile::ReduceKind::Min)
            return builder.create<arith::MinSIOp>(loc, vector, acc, value);
        return builder.create<arith::AddIOp>(loc, vector, acc, value);
    }
    return nullptr;
}

Value createChunkAcc(
    Operation* op, const ascvf2::ReductionInfo& reduction, Value srcBuf, Value padding, VectorType vectorType,
    unsigned hwWidth, Value accumulator, ArrayRef<Value> leadingIndices, ConversionPatternRewriter& rewriter)
{
    Location loc = op->getLoc();
    const auto& inputType = reduction.inputType;
    auto chunkLoop = rewriter.create<affine::AffineForOp>(
        loc, 0, inputType.getShape().back(), vectorType.getDimSize(0), ValueRange{accumulator});
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value chunk = chunkLoop.getInductionVar();
    Value mask;
    if (inputType.getShape().back() % hwWidth != 0) {
        AffineExpr remainingExpr = inputType.getShape().back() - rewriter.getAffineDimExpr(0);
        Value remaining = affine::makeComposedAffineApply(
            rewriter, loc, AffineMap::get(1, 0, remainingExpr, op->getContext()), ArrayRef<OpFoldResult>{chunk});
        mask = rewriter.create<vector::CreateMaskOp>(
            loc, VectorType::get({vectorType.getDimSize(0)}, rewriter.getI1Type()), ArrayRef<OpFoldResult>{remaining});
    }
    SmallVector<Value> readIndices(leadingIndices.begin(), leadingIndices.end());
    readIndices.push_back(chunk);
    AffineMap readMap =
        AffineMap::get(inputType.getRank(), 0, rewriter.getAffineDimExpr(inputType.getRank() - 1), op->getContext());
    Value value = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, srcBuf, readIndices, readMap, padding, mask, rewriter.getBoolArrayAttr({true}));
    Operation* accOp = createAcc(rewriter, loc, reduction.kind, chunkLoop.getRegionIterArgs()[0], value);
    Operation* maskedAcc =
        mask ? vector::maskOperation(rewriter, accOp, mask, chunkLoop.getRegionIterArgs()[0]) : accOp;
    rewriter.create<affine::AffineYieldOp>(loc, ValueRange{maskedAcc->getResult(0)});
    rewriter.setInsertionPointAfter(chunkLoop);
    return chunkLoop.getResult(0);
}

Value createReduced(
    Operation* op, const ascvf2::ReductionInfo& reduction, Value acc, ConversionPatternRewriter& rewriter)
{
    auto reducedType = VectorType::get({1}, reduction.inputType.getElementType());
    return rewriter.create<ascvf2::ReduceOp>(
        op->getLoc(), reducedType, acc, rewriter.getI32ArrayAttr({0}),
        asctile::ReduceKindAttr::get(rewriter.getContext(), reduction.kind));
}

Value createFlatAcc(
    Operation* op, const ascvf2::ReductionInfo& reduction, Value srcBuf, Value padding, VectorType vectorType,
    unsigned hwWidth, Value accumulator, ConversionPatternRewriter& rewriter)
{
    Location loc = op->getLoc();
    int64_t elementCount = reduction.inputType.getNumElements();
    auto chunkLoop =
        rewriter.create<affine::AffineForOp>(loc, 0, elementCount, vectorType.getDimSize(0), ValueRange{accumulator});
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value chunk = chunkLoop.getInductionVar();
    Value mask;
    if (elementCount % hwWidth != 0) {
        AffineExpr remainingExpr = elementCount - rewriter.getAffineDimExpr(0);
        Value remaining = affine::makeComposedAffineApply(
            rewriter, loc, AffineMap::get(1, 0, remainingExpr, op->getContext()), ArrayRef<OpFoldResult>{chunk});
        mask = rewriter.create<vector::CreateMaskOp>(
            loc, VectorType::get({vectorType.getDimSize(0)}, rewriter.getI1Type()), ArrayRef<OpFoldResult>{remaining});
    }
    AffineMap readMap = AffineMap::getMultiDimIdentityMap(1, op->getContext());
    Value value = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, srcBuf, ValueRange{chunk}, readMap, padding, mask, rewriter.getBoolArrayAttr({true}));
    Operation* accOp = createAcc(rewriter, loc, reduction.kind, chunkLoop.getRegionIterArgs()[0], value);
    Operation* maskedAcc =
        mask ? vector::maskOperation(rewriter, accOp, mask, chunkLoop.getRegionIterArgs()[0]) : accOp;
    rewriter.create<affine::AffineYieldOp>(loc, ValueRange{maskedAcc->getResult(0)});
    rewriter.setInsertionPointAfter(chunkLoop);
    return chunkLoop.getResult(0);
}

Value vectorizeReduce(
    Operation* op, const ascvf2::ReductionInfo& reduction, const VFInfo& vfInfo, ConversionPatternRewriter& rewriter)
{
    RewriterAdaptor adaptor(rewriter);
    adaptor->setInsertionPoint(op);
    Location loc = op->getLoc();
    const auto& inputType = reduction.inputType;
    auto hwWidth = vfInfo.getVectorWidth(inputType.getElementType());
    Attribute neutral = getNeutral(reduction.kind, inputType.getElementType());
    assert(succeeded(hwWidth) && *hwWidth != 0 && neutral);
    bool flattenInput = reduction.mode == ascvf2::ReductionMode::AllElements;
    int64_t elementCount = inputType.getNumElements();
    auto vecWidth =
        std::min(*hwWidth, static_cast<unsigned>(flattenInput ? elementCount : inputType.getShape().back()));

    auto vectorType = VectorType::get({vecWidth}, inputType.getElementType());
    Value acc = adaptor->create<arith::ConstantOp>(loc, SplatElementsAttr::get(vectorType, neutral));
    Value sourceTensor = rewriter.getRemappedValue(reduction.operand);
    Value srcBuf = adaptor.createBuffer(sourceTensor);
    if (flattenInput)
        srcBuf = adaptor.createFlatBuffer(op, srcBuf);
    Value dstBuf = adaptor.createDestination(op, reduction.resultType);
    Value padding = adaptor->create<arith::ConstantOp>(loc, cast<TypedAttr>(neutral));

    if (reduction.mode == ascvf2::ReductionMode::FinalAxis) {
        SmallVector<Value> leadingIndices;
        for (int64_t dimension : inputType.getShape().drop_back()) {
            auto loop = rewriter.create<affine::AffineForOp>(loc, 0, dimension, 1);
            rewriter.setInsertionPointToStart(loop.getBody());
            leadingIndices.push_back(loop.getInductionVar());
        }
        Value chunkAcc =
            createChunkAcc(op, reduction, srcBuf, padding, vectorType, *hwWidth, acc, leadingIndices, rewriter);
        AffineMap writeMap = AffineMap::get(
            inputType.getRank(), 0, rewriter.getAffineDimExpr(inputType.getRank() - 1), op->getContext());
        SmallVector<Value> writeIndices(leadingIndices.begin(), leadingIndices.end());
        writeIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        Value reduced = createReduced(op, reduction, chunkAcc, rewriter);
        rewriter.create<vector::TransferWriteOp>(
            loc, reduced, dstBuf, writeIndices, AffineMapAttr::get(writeMap), rewriter.getBoolArrayAttr({true}));
    } else {
        Value flatAcc = createFlatAcc(op, reduction, srcBuf, padding, vectorType, *hwWidth, acc, rewriter);

        AffineMap writeMap = AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), op->getContext());
        SmallVector<Value> writeIndices{rewriter.create<arith::ConstantIndexOp>(loc, 0)};
        Value reduced = createReduced(op, reduction, flatAcc, rewriter);
        rewriter.create<vector::TransferWriteOp>(
            loc, reduced, dstBuf, writeIndices, AffineMapAttr::get(writeMap), rewriter.getBoolArrayAttr({true}));
    }

    adaptor->setInsertionPoint(op);
    return adaptor.createTensor(op, dstBuf, reduction.resultType);
}

template <typename OpT>
struct ConvertReduction : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    using typename OpConversionPattern<OpT>::OpAdaptor;

    ConvertReduction(TypeConverter& typeConverter, MLIRContext* context, const VFInfo& vfInfo)
        : OpConversionPattern<OpT>(typeConverter, context), vfInfo(vfInfo)
    {}

    LogicalResult matchAndRewrite(OpT op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto reduction = ascvf2::getSupportedReductionInfo(op);
        if (!reduction)
            return failure();
        Value result = vectorizeReduce(op, *reduction, vfInfo, rewriter);
        rewriter.replaceOp(op, result);
        return success();
    }

private:
    const VFInfo& vfInfo;
};

struct VectorizePass : public ascvf2::impl::VectorizeBase<VectorizePass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return std::optional<Type>{type}; });
        auto materialize = [](OpBuilder& builder, Type type, ValueRange inputs, Location loc) -> Value {
            return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
        };
        typeConverter.addArgumentMaterialization(materialize);
        typeConverter.addSourceMaterialization(materialize);
        typeConverter.addTargetMaterialization(materialize);
        VFInfo vfInfo{func};
        MLIRContext* context = func.getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<
            affine::AffineDialect, memref::MemRefDialect, tensor::TensorDialect, vector::VectorDialect,
            ascvf2::AscVF2Dialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
            [](Operation* op) { return !ascvf2::isVectorizableGroupOperation(op); });
        RewritePatternSet patterns(context);
        patterns.add<ConvertBroadcast>(typeConverter, context, vfInfo);
        patterns.add<ConvertElementwise>(typeConverter, context, vfInfo);
        patterns.add<ConvertReduction<asctile::ReduceOp>, ConvertReduction<asctile::ReduceAs1dOp>>(
            typeConverter, context, vfInfo);
        if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createVectorizePass() { return std::make_unique<VectorizePass>(); }
