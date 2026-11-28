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
#include "asctile/Dialect/AscVF2/Utils/RegTypeConverter.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_LOWERTOREG2
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

enum : unsigned { AnyPatterns = 1, ReductionPatterns = 10 };

inline Operation* getMaskedOperation(vector::MaskOp maskOp)
{
    Block* block = maskOp.getMaskBlock();
    if (block->getOperations().size() <= 1)
        return nullptr;
    return &block->front();
}

inline Value createRegTensor(ConversionPatternRewriter& rewriter, Location loc, Type elementType)
{
    return rewriter.create<ascendc::RegTensorOp>(loc, ascendc::RegTensorType::get(rewriter.getContext(), elementType));
}

FailureOr<Value> createFullMask(ConversionPatternRewriter& rewriter, Operation* operation, VectorType vectorType)
{
    if (vectorType.getRank() != 1 || vectorType.getScalableDims()[0])
        return failure();
    FailureOr<unsigned> hardwareWidth = ascvf2::getVFVectorWidth(operation, vectorType.getElementType());
    if (failed(hardwareWidth) || vectorType.getDimSize(0) > *hardwareWidth)
        return failure();

    return rewriter
        .create<ascvf2::SimpleMaskRegOp>(
            operation->getLoc(), ascendc::MaskRegType::get(rewriter.getContext()),
            TypeAttr::get(vectorType.getElementType()), rewriter.getIndexAttr(vectorType.getDimSize(0)), Value{})
        .getResult();
}

template <typename SourceOp, bool matchUnmasked = true>
struct ConvertMaskedOp : public ConversionPattern {
    ConvertMaskedOp(const TypeConverter& typeConverter, MLIRContext* context, PatternBenefit benefit = 1)
        : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, benefit, context)
    {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const override
    {
        if (auto maskOp = dyn_cast<vector::MaskOp>(op)) {
            auto maskedOp = dyn_cast_if_present<SourceOp>(getMaskedOperation(maskOp));
            if (!maskedOp)
                return failure();
            return matchAndRewrite(maskOp, maskedOp, rewriter);
        }
        if constexpr (!matchUnmasked)
            return failure();
        if (auto fullOp = dyn_cast<SourceOp>(op); fullOp && !isa_and_present<vector::MaskOp>(op->getParentOp())) {
            return matchAndRewrite(vector::MaskOp{}, fullOp, rewriter);
        }
        return failure();
    }

    virtual LogicalResult matchAndRewrite(
        vector::MaskOp maskOp, SourceOp maskedOp, ConversionPatternRewriter& rewriter) const = 0;

    FailureOr<Value> getMask(
        vector::MaskOp maskOp, SourceOp maskedOp, VectorType vectorType, ConversionPatternRewriter& rewriter) const
    {
        if (maskOp)
            return rewriter.getRemappedValue(maskOp.getMask());
        return createFullMask(rewriter, maskedOp.getOperation(), vectorType);
    }

    Operation* getReplacedOperation(vector::MaskOp maskOp, SourceOp maskedOp) const
    {
        return maskOp ? maskOp.getOperation() : maskedOp.getOperation();
    }
};

template <typename SourceOp, typename RegOp>
struct ConvertBinaryOp : public ConvertMaskedOp<SourceOp> {
    using ConvertMaskedOp<SourceOp>::ConvertMaskedOp;

    LogicalResult matchAndRewrite(
        vector::MaskOp maskOp, SourceOp maskedOp, ConversionPatternRewriter& rewriter) const override
    {
        auto vectorType = dyn_cast<VectorType>(maskedOp.getResult().getType());
        if (!vectorType || vectorType.getRank() != 1)
            return failure();

        Value source0 = rewriter.getRemappedValue(maskedOp.getLhs());
        Value source1 = rewriter.getRemappedValue(maskedOp.getRhs());
        FailureOr<Value> mask = this->getMask(maskOp, maskedOp, vectorType, rewriter);
        if (failed(mask))
            return failure();
        if (!isa<ascendc::RegTensorType>(source0.getType()) || !isa<ascendc::RegTensorType>(source1.getType()) ||
            !isa<ascendc::MaskRegType>(mask->getType()))
            return failure();

        Value destination = createRegTensor(rewriter, maskedOp.getLoc(), vectorType.getElementType());
        if (maskOp && maskOp.getPassthru()) {
            Value passthru = rewriter.getRemappedValue(maskOp.getPassthru());
            if (isa<ascendc::RegTensorType>(passthru.getType()))
                destination = passthru;
        }
        rewriter.create<RegOp>(maskedOp.getLoc(), destination, source0, source1, *mask);
        rewriter.replaceOp(this->getReplacedOperation(maskOp, maskedOp), destination);
        return success();
    }
};

template <typename SourceOp, typename RegOp>
struct ConvertUnaryOp : public ConvertMaskedOp<SourceOp> {
    using ConvertMaskedOp<SourceOp>::ConvertMaskedOp;

    LogicalResult matchAndRewrite(
        vector::MaskOp maskOp, SourceOp maskedOp, ConversionPatternRewriter& rewriter) const override
    {
        auto vectorType = dyn_cast<VectorType>(maskedOp.getResult().getType());
        if (!vectorType || vectorType.getRank() != 1)
            return failure();

        Value source = rewriter.getRemappedValue(maskedOp.getOperand());
        FailureOr<Value> mask = this->getMask(maskOp, maskedOp, vectorType, rewriter);
        if (failed(mask) || !isa<ascendc::RegTensorType>(source.getType()) ||
            !isa<ascendc::MaskRegType>(mask->getType()))
            return failure();

        Value destination = createRegTensor(rewriter, maskedOp.getLoc(), vectorType.getElementType());
        rewriter.create<RegOp>(maskedOp.getLoc(), destination, source, *mask);
        rewriter.replaceOp(this->getReplacedOperation(maskOp, maskedOp), destination);
        return success();
    }
};

template <typename SourceOp, typename RegOp>
struct ConvertVecScalarOp : public ConvertMaskedOp<SourceOp> {
    using ConvertMaskedOp<SourceOp>::ConvertMaskedOp;

    LogicalResult matchAndRewrite(
        vector::MaskOp maskOp, SourceOp maskedOp, ConversionPatternRewriter& rewriter) const override
    {
        auto vectorType = dyn_cast<VectorType>(maskedOp.getResult().getType());
        if (!vectorType || vectorType.getRank() != 1)
            return failure();

        Value source = rewriter.getRemappedValue(maskedOp.getBase());
        Value scalar = rewriter.getRemappedValue(maskedOp.getValue());
        FailureOr<Value> mask = this->getMask(maskOp, maskedOp, vectorType, rewriter);
        if (failed(mask))
            return failure();
        if (!isa<ascendc::RegTensorType>(source.getType()) || !scalar.getType().isIntOrFloat() ||
            scalar.getType() != vectorType.getElementType() || !isa<ascendc::MaskRegType>(mask->getType()))
            return failure();

        Value destination = createRegTensor(rewriter, maskedOp.getLoc(), vectorType.getElementType());
        rewriter.create<RegOp>(maskedOp.getLoc(), destination, source, scalar, *mask);
        rewriter.replaceOp(this->getReplacedOperation(maskOp, maskedOp), destination);
        return success();
    }
};

template <typename SourceOp, typename RegOp>
struct ConvertVecScalarDupOp : public ConvertMaskedOp<SourceOp> {
    using ConvertMaskedOp<SourceOp>::ConvertMaskedOp;

    LogicalResult matchAndRewrite(
        vector::MaskOp maskOp, SourceOp maskedOp, ConversionPatternRewriter& rewriter) const override
    {
        auto vectorType = dyn_cast<VectorType>(maskedOp.getResult().getType());
        if (!vectorType || vectorType.getRank() != 1)
            return failure();

        Value source = rewriter.getRemappedValue(maskedOp.getBase());
        Value scalar = rewriter.getRemappedValue(maskedOp.getValue());
        FailureOr<Value> mask = this->getMask(maskOp, maskedOp, vectorType, rewriter);
        if (failed(mask))
            return failure();
        if (!isa<ascendc::RegTensorType>(source.getType()) || !scalar.getType().isIntOrFloat() ||
            scalar.getType() != vectorType.getElementType() || !isa<ascendc::MaskRegType>(mask->getType()))
            return failure();

        Value duplicate = createRegTensor(rewriter, maskedOp.getLoc(), vectorType.getElementType());
        rewriter.create<ascendc::DuplicateRegOp>(maskedOp.getLoc(), duplicate, scalar, *mask);

        Value destination = createRegTensor(rewriter, maskedOp.getLoc(), vectorType.getElementType());
        rewriter.create<RegOp>(maskedOp.getLoc(), destination, source, duplicate, *mask);
        rewriter.replaceOp(this->getReplacedOperation(maskOp, maskedOp), destination);
        return success();
    }
};

bool isSupportedMaskedAscVF2Operation(Operation* op)
{
    return isa<
        ascvf2::AddSOp, ascvf2::SubSOp, ascvf2::MulSOp, ascvf2::DivSOp, ascvf2::MinSOp, ascvf2::MaxSOp, ascvf2::ShLSOp,
        ascvf2::ShRSOp, ascvf2::LeakyReluOp, ascvf2::ReluOp, ascvf2::NotOp>(op);
}

struct ConvertToBuffer : public OpConversionPattern<ascvf2::ToBufferOp> {
    using OpConversionPattern<ascvf2::ToBufferOp>::OpConversionPattern;
    using typename OpConversionPattern<ascvf2::ToBufferOp>::OpAdaptor;

    LogicalResult matchAndRewrite(
        ascvf2::ToBufferOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto tensorType = dyn_cast<ascendc::LocalTensorType>(adaptor.getTensor().getType());
        auto bufferType = dyn_cast<MemRefType>(op.getResult().getType());
        if (!tensorType || !bufferType)
            return failure();
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorGetPhyAddrV2Op>(op, bufferType, adaptor.getTensor());
        return success();
    }
};

struct ConvertAlloca : public OpConversionPattern<memref::AllocaOp> {
    using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;
    using typename OpConversionPattern<memref::AllocaOp>::OpAdaptor;

    LogicalResult matchAndRewrite(memref::AllocaOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto bufferType = dyn_cast<MemRefType>(op.getResult().getType());
        auto memorySpace = bufferType ? dyn_cast<IntegerAttr>(bufferType.getMemorySpace()) : IntegerAttr();
        if (!bufferType || !bufferType.hasStaticShape() || !memorySpace ||
            memorySpace.getValue().getSExtValue() != static_cast<int64_t>(asctile::TensorLocation::UB))
            return failure();

        auto tensorType = ascendc::LocalTensorType::get(bufferType.getShape(), bufferType.getElementType());
        auto tensor = rewriter.create<ascendc::LocalTensorAutoOp>(op.getLoc(), tensorType);
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorGetPhyAddrV2Op>(op, bufferType, tensor);
        return success();
    }
};

struct ConvertToTensor : public OpConversionPattern<ascvf2::ToTensorOp> {
    using OpConversionPattern<ascvf2::ToTensorOp>::OpConversionPattern;
    using typename OpConversionPattern<ascvf2::ToTensorOp>::OpAdaptor;

    LogicalResult matchAndRewrite(
        ascvf2::ToTensorOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        Type resultType = typeConverter->convertType(op.getResult().getType());
        if (auto phyAddr = adaptor.getBuffer().getDefiningOp<ascendc::LocalTensorGetPhyAddrV2Op>()) {
            Value tensor = phyAddr.getTensor();
            if (tensor.getType() == resultType) {
                rewriter.replaceOp(op, tensor);
                return success();
            }
        }
        Value cast =
            rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), resultType, adaptor.getBuffer()).getResult(0);
        rewriter.replaceOp(op, cast);
        return success();
    }
};

struct ConvertReduce : public OpConversionPattern<ascvf2::ReduceOp> {
    using OpConversionPattern<ascvf2::ReduceOp>::OpConversionPattern;
    using typename OpConversionPattern<ascvf2::ReduceOp>::OpAdaptor;

    LogicalResult matchAndRewrite(
        ascvf2::ReduceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto vectorType = dyn_cast<VectorType>(op.getInput().getType());
        Value source = adaptor.getInput();
        if (!isa<ascendc::RegTensorType>(source.getType())) {
            auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>();
            source = castOp && castOp.getInputs().size() == 1 &&
                             isa<ascendc::RegTensorType>(castOp.getInputs().front().getType()) ?
                         castOp.getInputs().front() :
                         Value{};
        }
        if (!vectorType || !source)
            return failure();

        Value destination = createRegTensor(rewriter, op.getLoc(), vectorType.getElementType());
        FailureOr<Value> mask = createFullMask(rewriter, op.getOperation(), vectorType);
        if (failed(mask))
            return failure();
        switch (op.getKind()) {
            case asctile::ReduceKind::Sum:
                rewriter.create<ascendc::ReduceSumRegOp>(op.getLoc(), destination, source, *mask);
                break;
            case asctile::ReduceKind::Max:
                rewriter.create<ascendc::ReduceMaxRegOp>(op.getLoc(), destination, source, *mask);
                break;
            case asctile::ReduceKind::Min:
                rewriter.create<ascendc::ReduceMinRegOp>(op.getLoc(), destination, source, *mask);
                break;
            default:
                return failure();
        }
        rewriter.replaceOp(op, destination);
        return success();
    }
};

struct ConvertBarrier : public OpConversionPattern<ascvf2::BarrierOp> {
    using OpConversionPattern<ascvf2::BarrierOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ascvf2::BarrierOp op, typename ascvf2::BarrierOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ascendc::LocalMemBarOp>(
            op, ascendc::MemType::VEC_STORE, ascendc::MemType::VEC_LOAD);
        return success();
    }
};

FailureOr<Type> inferMaskElementType(Value mask)
{
    Type elementType;
    for (OpOperand& use : mask.getUses()) {
        Operation* owner = use.getOwner();
        VectorType vectorType =
            TypeSwitch<Operation*, VectorType>(owner)
                .Case<vector::MaskOp>([&](auto maskOp) {
                    Operation* masked = getMaskedOperation(maskOp);
                    return masked ? dyn_cast<VectorType>(masked->getResult(0).getType()) : VectorType{};
                })
                .Case<vector::TransferReadOp>([&](auto read) { return read.getVectorType(); })
                .Case<vector::TransferWriteOp>([&](auto write) { return write.getVectorType(); })
                .Default(VectorType{});
        if (!vectorType)
            return failure();
        if (elementType && elementType != vectorType.getElementType())
            return failure();
        elementType = vectorType.getElementType();
    }
    if (!elementType)
        return failure();
    return elementType;
}

struct ConvertCreateMask : public OpConversionPattern<vector::CreateMaskOp> {
    using OpConversionPattern<vector::CreateMaskOp>::OpConversionPattern;
    using typename OpConversionPattern<vector::CreateMaskOp>::OpAdaptor;

    LogicalResult matchAndRewrite(
        vector::CreateMaskOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (op.getVectorType().getRank() != 1)
            return failure();
        auto elementType = inferMaskElementType(op.getResult());
        if (failed(elementType))
            return failure();

        Value mask = rewriter.create<ascvf2::SimpleMaskRegOp>(
            op.getLoc(), ascendc::MaskRegType::get(getContext()), TypeAttr::get(*elementType), IntegerAttr{},
            adaptor.getOperands().front());
        rewriter.replaceOp(op, mask);
        return success();
    }
};

bool isSupportedConstantMask(vector::ConstantMaskOp op)
{
    return op.getVectorType().getRank() == 1 && succeeded(inferMaskElementType(op.getResult()));
}

struct ConvertConstantMask : public OpConversionPattern<vector::ConstantMaskOp> {
    using OpConversionPattern<vector::ConstantMaskOp>::OpConversionPattern;
    using typename OpConversionPattern<vector::ConstantMaskOp>::OpAdaptor;

    LogicalResult matchAndRewrite(
        vector::ConstantMaskOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!isSupportedConstantMask(op))
            return failure();
#if LLVM_VERSION_MAJOR >= 20
        int64_t count = op.getMaskDimSizes()[0];
#else
        int64_t count = cast<IntegerAttr>(op.getMaskDimSizes()[0]).getValue().getSExtValue();
#endif
        auto elementType = inferMaskElementType(op.getResult());
        Value mask = rewriter.create<ascvf2::SimpleMaskRegOp>(
            op.getLoc(), ascendc::MaskRegType::get(getContext()), TypeAttr::get(*elementType),
            rewriter.getIndexAttr(count), Value{});
        rewriter.replaceOp(op, mask);
        return success();
    }
};

bool isContiguousTransfer(
    Value source, VectorType vectorType, ArrayAttr inBounds, AffineMap permutationMap, ValueRange indices)
{
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    return memrefType && memrefType.hasStaticShape() && memrefType.getLayout().isIdentity() &&
           vectorType.getRank() == 1 && inBounds.size() == 1 && indices.size() == memrefType.getRank() &&
           memrefType.getElementType() == vectorType.getElementType() && permutationMap.isMinorIdentity() &&
           llvm::all_of(inBounds, [](Attribute attr) { return cast<BoolAttr>(attr).getValue(); });
}

bool isBroadcastTransfer(
    vector::TransferReadOp op, Value source, VectorType vectorType, ArrayAttr inBounds, AffineMap permutationMap,
    ValueRange indices)
{
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    if (!memrefType || !memrefType.hasStaticShape() || !memrefType.getLayout().isIdentity() ||
        vectorType.getRank() != 1 || vectorType.getScalableDims()[0] || inBounds.size() != 1 ||
        indices.size() != memrefType.getRank() || memrefType.getElementType() != vectorType.getElementType() ||
        permutationMap.getNumResults() != 1 || !isa<AffineConstantExpr>(permutationMap.getResult(0)) ||
        cast<AffineConstantExpr>(permutationMap.getResult(0)).getValue() != 0)
        return false;
    if (!llvm::all_of(inBounds, [](Attribute attr) { return cast<BoolAttr>(attr).getValue(); }))
        return false;
    FailureOr<unsigned> vectorWidth = ascvf2::getVFVectorWidth(op, vectorType.getElementType());
    return succeeded(vectorWidth) && vectorType.getDimSize(0) > 1 && vectorType.getDimSize(0) <= *vectorWidth;
}

Value getFlatSource(Value source)
{
    auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpret || !reinterpret.getOffsets().empty() || !reinterpret.getSizes().empty() ||
        !reinterpret.getStrides().empty())
        return {};

    auto sourceType = dyn_cast<MemRefType>(reinterpret.getSource().getType());
    auto flatType = dyn_cast<MemRefType>(reinterpret.getResult().getType());
    if (!sourceType || !flatType || !sourceType.hasStaticShape() || !sourceType.getLayout().isIdentity() ||
        flatType.getRank() != 1)
        return {};

    ArrayRef<int64_t> offsets = reinterpret.getStaticOffsets();
    ArrayRef<int64_t> sizes = reinterpret.getStaticSizes();
    ArrayRef<int64_t> strides = reinterpret.getStaticStrides();
    if (offsets.size() != 1 || offsets[0] != 0 || sizes.size() != 1 || strides.size() != 1 || strides[0] != 1 ||
        sizes[0] != sourceType.getNumElements() || flatType.getElementType() != sourceType.getElementType() ||
        flatType.getMemorySpace() != sourceType.getMemorySpace())
        return {};
    return reinterpret.getSource();
}

Value createTransferOffset(
    ConversionPatternRewriter& rewriter, Location location, MemRefType sourceType, ValueRange indices,
    bool isFlatSource)
{
    if (isFlatSource)
        return indices.front();

    Value offset = indices.back();
    int64_t stride = 1;
    for (int64_t dimension = sourceType.getRank() - 2; dimension >= 0; --dimension) {
        stride *= sourceType.getShape()[dimension + 1];
        Value dimensionOffset = rewriter.create<arith::MulIOp>(
            location, indices[dimension], rewriter.create<arith::ConstantIndexOp>(location, stride));
        offset = rewriter.create<arith::AddIOp>(location, offset, dimensionOffset);
    }
    return offset;
}

struct ConvertTransferRead : public OpConversionPattern<vector::TransferReadOp> {
    using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        vector::TransferReadOp op, typename vector::TransferReadOp::Adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        bool contiguous = isContiguousTransfer(
            op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(), op.getIndices());
        bool broadcast = isBroadcastTransfer(
            op, op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(), op.getIndices());
        if (!contiguous && !broadcast)
            return failure();
        std::optional<ascendc::LoadDist> loadDist;
        if (broadcast) {
            FailureOr<unsigned> elementSize = ascvf2::getElementSizeInBytes(op.getVectorType().getElementType());
            if (failed(elementSize))
                return failure();
            switch (*elementSize) {
                case 1:
                    loadDist = ascendc::LoadDist::DIST_BRC_B8;
                    break;
                case 2:
                    loadDist = ascendc::LoadDist::DIST_BRC_B16;
                    break;
                case 4:
                    loadDist = ascendc::LoadDist::DIST_BRC_B32;
                    break;
                default:
                    return failure();
            }
        } else {
            loadDist = ascendc::LoadDist::DIST_NORM;
        }

        Value destination = createRegTensor(rewriter, op.getLoc(), op.getVectorType().getElementType());
        Value remappedSource = rewriter.getRemappedValue(op.getSource());
        Value flatSource = getFlatSource(remappedSource);
        bool isFlatSource = static_cast<bool>(flatSource);
        Value sourceMemref = flatSource ? rewriter.getRemappedValue(flatSource) : remappedSource;
        Value sourceOffset = createTransferOffset(
            rewriter, op.getLoc(), cast<MemRefType>(sourceMemref.getType()), op.getIndices(), isFlatSource);
        Value sourceAddress = rewriter.create<emitasc::PtrOffsetOp>(
            op.getLoc(), sourceMemref.getType(), sourceMemref, IntegerAttr{}, sourceOffset);
        rewriter.create<ascendc::DataCopyLoadOp>(op.getLoc(), destination, sourceAddress, *loadDist);
        rewriter.replaceOp(op, destination);
        return success();
    }
};

struct ConvertTransferWrite : public OpConversionPattern<vector::TransferWriteOp> {
    using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        vector::TransferWriteOp op, typename vector::TransferWriteOp::Adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        if (!isContiguousTransfer(
                op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(), op.getIndices()))
            return failure();
        ascendc::StoreDist dist = ascendc::StoreDist::DIST_NORM;
        if (op.getVectorType().getDimSize(0) == 1) {
            FailureOr<unsigned> elementSize = ascvf2::getElementSizeInBytes(op.getVectorType().getElementType());
            if (failed(elementSize))
                return failure();
            switch (*elementSize) {
                case 1:
                    dist = ascendc::StoreDist::DIST_FIRST_ELEMENT_B8;
                    break;
                case 2:
                    dist = ascendc::StoreDist::DIST_FIRST_ELEMENT_B16;
                    break;
                case 4:
                    dist = ascendc::StoreDist::DIST_FIRST_ELEMENT_B32;
                    break;
                default:
                    return failure();
            }
        }
        Value vector = rewriter.getRemappedValue(op.getVector());
        Value remappedSource = rewriter.getRemappedValue(op.getSource());
        Value flatSource = getFlatSource(remappedSource);
        bool isFlatSource = static_cast<bool>(flatSource);
        Value sourceMemref = flatSource ? rewriter.getRemappedValue(flatSource) : remappedSource;
        Value mask;
        if (op.getMask())
            mask = rewriter.getRemappedValue(op.getMask());
        else {
            FailureOr<Value> fullMask = createFullMask(rewriter, op.getOperation(), op.getVectorType());
            if (failed(fullMask))
                return failure();
            mask = *fullMask;
        }
        if (!isa<ascendc::RegTensorType>(vector.getType()) || !isa<ascendc::MaskRegType>(mask.getType()))
            return failure();

        Value destinationOffset = createTransferOffset(
            rewriter, op.getLoc(), cast<MemRefType>(sourceMemref.getType()), op.getIndices(), isFlatSource);
        Value destination = rewriter.create<emitasc::PtrOffsetOp>(
            op.getLoc(), sourceMemref.getType(), sourceMemref, IntegerAttr{}, destinationOffset);
        rewriter.create<ascendc::DataCopyStoreOp>(op.getLoc(), destination, vector, mask, dist);
        rewriter.eraseOp(op);
        return success();
    }
};

bool isSupportedVectorBroadcast(vector::BroadcastOp op)
{
    auto sourceType = dyn_cast<VectorType>(op.getSource().getType());
    auto resultType = op.getResultVectorType();
    return sourceType && sourceType.getRank() == 1 && resultType.getRank() == 1 && sourceType.getDimSize(0) == 1 &&
           sourceType.getElementType() == resultType.getElementType() && !resultType.getElementType().isInteger(1);
}

struct ConvertVectorBroadcast : public OpConversionPattern<vector::BroadcastOp> {
    using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;
    using typename OpConversionPattern<vector::BroadcastOp>::OpAdaptor;

    LogicalResult matchAndRewrite(vector::BroadcastOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!isSupportedVectorBroadcast(op))
            return failure();
        Value source = rewriter.getRemappedValue(op.getSource());
        if (!isa<ascendc::RegTensorType>(source.getType()))
            return failure();

        Value destination = createRegTensor(rewriter, op.getLoc(), op.getResultVectorType().getElementType());
        FailureOr<Value> mask = createFullMask(rewriter, op.getOperation(), op.getResultVectorType());
        if (failed(mask))
            return failure();
        rewriter.create<ascendc::DuplicateRegOp>(op.getLoc(), destination, source, *mask);
        rewriter.replaceOp(op, destination);
        return success();
    }
};

bool isSupportedMaskedArithOperation(Operation* op)
{
    return isa<
        arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::MaximumFOp,
        arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaxSIOp, arith::MinSIOp, arith::DivFOp,
        arith::DivSIOp, arith::AndIOp, arith::OrIOp, arith::NegFOp>(op);
}

bool isSupportedVectorConstant(arith::ConstantOp op)
{
    auto vectorType = dyn_cast<VectorType>(op.getResult().getType());
    auto splatValue = dyn_cast<SplatElementsAttr>(op.getValue());
    return vectorType && vectorType.getRank() == 1 && !vectorType.getElementType().isInteger(1) && splatValue;
}

struct ConvertVectorConstant : public OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
    using typename OpConversionPattern<arith::ConstantOp>::OpAdaptor;

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!isSupportedVectorConstant(op))
            return failure();
        auto vectorType = cast<VectorType>(op.getResult().getType());
        auto scalarValue = cast<SplatElementsAttr>(op.getValue()).getSplatValue<TypedAttr>();
        Value scalar = rewriter.create<arith::ConstantOp>(op.getLoc(), scalarValue);
        Value destination = createRegTensor(rewriter, op.getLoc(), vectorType.getElementType());
        FailureOr<Value> mask = createFullMask(rewriter, op.getOperation(), vectorType);
        if (failed(mask))
            return failure();
        rewriter.create<ascendc::DuplicateRegOp>(op.getLoc(), destination, scalar, *mask);
        rewriter.replaceOp(op, destination);
        return success();
    }
};

bool isSupportedMaskedMathOperation(Operation* op)
{
    return isa<math::AbsFOp, math::AbsIOp, math::ExpOp, math::LogOp, math::SqrtOp>(op);
}

struct LowerToRegPass : public ascvf2::impl::LowerToReg2Base<LowerToRegPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        ascvf2::RegTypeConverter typeConverter;
        ConversionTarget target(getContext());
        target.addLegalDialect<
            ascendc::AscendCDialect, arith::ArithDialect, math::MathDialect, emitasc::EmitAscDialect>();
        target.addDynamicallyLegalOp<
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::MaximumFOp,
            arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaxSIOp, arith::MinSIOp, arith::DivFOp,
            arith::DivSIOp, arith::AndIOp, arith::OrIOp, arith::NegFOp, math::AbsFOp, math::AbsIOp, math::ExpOp,
            math::LogOp, math::SqrtOp, ascvf2::AddSOp, ascvf2::SubSOp, ascvf2::MulSOp, ascvf2::DivSOp, ascvf2::MinSOp,
            ascvf2::MaxSOp, ascvf2::ShLSOp, ascvf2::ShRSOp, ascvf2::LeakyReluOp, ascvf2::ReluOp, ascvf2::NotOp>(
            [](Operation* op) {
                if (isa_and_present<vector::MaskOp>(op->getParentOp()))
                    return true;
                if (!isSupportedMaskedArithOperation(op) && !isSupportedMaskedMathOperation(op) &&
                    !isSupportedMaskedAscVF2Operation(op))
                    return true;
                auto vectorType = dyn_cast<VectorType>(op->getResult(0).getType());
                return !vectorType || vectorType.getRank() != 1;
            });
        target.addLegalOp<ascvf2::SimpleMaskRegOp, ascvf2::VFGroupOp, ascvf2::YieldOp, UnrealizedConversionCastOp>();
        target.addIllegalOp<ascvf2::ToBufferOp, ascvf2::ToTensorOp>();
        target.addIllegalOp<memref::AllocaOp>();
        target.addDynamicallyLegalOp<vector::MaskOp>([](vector::MaskOp op) {
            Operation* maskedOperation = getMaskedOperation(op);
            return !maskedOperation || (!isSupportedMaskedArithOperation(maskedOperation) &&
                                        !isSupportedMaskedMathOperation(maskedOperation) &&
                                        !isSupportedMaskedAscVF2Operation(maskedOperation));
        });
        target.addDynamicallyLegalOp<ascvf2::ReduceOp>([](ascvf2::ReduceOp) { return false; });
        target.addDynamicallyLegalOp<vector::CreateMaskOp>([](vector::CreateMaskOp op) {
            return op.getOperands().size() != 1 || op.getVectorType().getRank() != 1 ||
                   failed(inferMaskElementType(op.getResult()));
        });
        target.addDynamicallyLegalOp<vector::ConstantMaskOp>(
            [](vector::ConstantMaskOp op) { return !isSupportedConstantMask(op); });
        target.addDynamicallyLegalOp<vector::TransferReadOp>([](vector::TransferReadOp op) {
            return !isContiguousTransfer(
                       op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(), op.getIndices()) &&
                   !isBroadcastTransfer(
                       op, op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(),
                       op.getIndices());
        });
        target.addDynamicallyLegalOp<vector::TransferWriteOp>([](vector::TransferWriteOp op) {
            return !isContiguousTransfer(
                op.getSource(), op.getVectorType(), op.getInBounds(), op.getPermutationMap(), op.getIndices());
        });
        target.addDynamicallyLegalOp<vector::BroadcastOp>(
            [](vector::BroadcastOp op) { return !isSupportedVectorBroadcast(op); });
        target.addDynamicallyLegalOp<arith::ConstantOp>(
            [](arith::ConstantOp op) { return !isSupportedVectorConstant(op); });

        RewritePatternSet patterns(&getContext());
        patterns.add<ConvertReduce>(typeConverter, &getContext(), PatternBenefit(ReductionPatterns));
        patterns.add<
            ConvertVecScalarDupOp<ascvf2::AddSOp, ascendc::AddRegOp>,
            ConvertVecScalarDupOp<ascvf2::SubSOp, ascendc::SubRegOp>,
            ConvertVecScalarDupOp<ascvf2::MulSOp, ascendc::MulRegOp>,
            ConvertVecScalarDupOp<ascvf2::DivSOp, ascendc::DivRegOp>,
            ConvertVecScalarDupOp<ascvf2::MinSOp, ascendc::MinRegOp>,
            ConvertVecScalarDupOp<ascvf2::MaxSOp, ascendc::MaxRegOp>,
            ConvertVecScalarOp<ascvf2::ShLSOp, ascendc::ShiftLeftsRegOp>,
            ConvertVecScalarOp<ascvf2::ShRSOp, ascendc::ShiftRightsRegOp>,
            ConvertVecScalarOp<ascvf2::LeakyReluOp, ascendc::LeakyReluRegOp>,
            ConvertUnaryOp<ascvf2::ReluOp, ascendc::ReluRegOp>, ConvertUnaryOp<ascvf2::NotOp, ascendc::NotRegOp>,
            ConvertToBuffer, ConvertToTensor, ConvertAlloca, ConvertBarrier, ConvertCreateMask, ConvertConstantMask,
            ConvertTransferRead, ConvertTransferWrite, ConvertVectorBroadcast, ConvertVectorConstant,
            ConvertBinaryOp<arith::AddFOp, ascendc::AddRegOp>, ConvertBinaryOp<arith::AddIOp, ascendc::AddRegOp>,
            ConvertBinaryOp<arith::SubFOp, ascendc::SubRegOp>, ConvertBinaryOp<arith::SubIOp, ascendc::SubRegOp>,
            ConvertBinaryOp<arith::MulFOp, ascendc::MulRegOp>, ConvertBinaryOp<arith::MulIOp, ascendc::MulRegOp>,
            ConvertBinaryOp<arith::DivFOp, ascendc::DivRegOp>, ConvertBinaryOp<arith::DivSIOp, ascendc::DivRegOp>,
            ConvertBinaryOp<arith::MaximumFOp, ascendc::MaxRegOp>,
            ConvertBinaryOp<arith::MinimumFOp, ascendc::MinRegOp>, ConvertBinaryOp<arith::MaxNumFOp, ascendc::MaxRegOp>,
            ConvertBinaryOp<arith::MinNumFOp, ascendc::MinRegOp>, ConvertBinaryOp<arith::MaxSIOp, ascendc::MaxRegOp>,
            ConvertBinaryOp<arith::MinSIOp, ascendc::MinRegOp>, ConvertBinaryOp<arith::AndIOp, ascendc::AndRegOp>,
            ConvertBinaryOp<arith::OrIOp, ascendc::OrRegOp>, ConvertUnaryOp<arith::NegFOp, ascendc::NegRegOp>,
            ConvertUnaryOp<math::AbsFOp, ascendc::AbsRegOp>, ConvertUnaryOp<math::AbsIOp, ascendc::AbsRegOp>,
            ConvertUnaryOp<math::ExpOp, ascendc::ExpRegOp>, ConvertUnaryOp<math::LogOp, ascendc::LnRegOp>,
            ConvertUnaryOp<math::SqrtOp, ascendc::SqrtRegOp>>(
            typeConverter, &getContext(), PatternBenefit(AnyPatterns));
        scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }

        SmallVector<Operation*, 16> deadReinterprets;
        func.walk([&](memref::ReinterpretCastOp reinterpret) {
            if (reinterpret->use_empty())
                deadReinterprets.push_back(reinterpret);
        });
        for (Operation* reinterpret : deadReinterprets)
            reinterpret->erase();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createLowerToRegPass() { return std::make_unique<LowerToRegPass>(); }
