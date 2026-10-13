/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Conversion/LowerToAsc/Passes.h"
#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Constants.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERTENSOR
#include "asctile/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct ConvertEmpty : public ConvertOp<tensor::EmptyOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(tensor::EmptyOp op, ConvertRewriter& rewriter) const override
    {
        Value tensor = createTensorOp(rewriter, op.getLoc(), op.getType());
        rewriter.replaceOp(op, tensor);
        return success();
    }
};

struct ConvertExtractSlice : public ConvertOp<tensor::ExtractSliceOp> {
    using ConvertOp::ConvertOp;

    std::pair<ascendc::LocalTensorType, SmallVector<int64_t, 2>> convertTypeShape(asctile::LocalTensorType type) const
    {
        auto convertedType = typeConverter->convertType<ascendc::LocalTensorType>(type);
        auto origShape = convertedType.getShape();
        if (origShape.size() == 1) {
            SmallVector<int64_t, 2> shape(2U, 1L);
            shape.back() = origShape.front();
            return {convertedType, shape};
        }
        return {convertedType, llvm::to_vector<2>(origShape)};
    }

    LogicalResult matchAndRewrite(tensor::ExtractSliceOp op, ConvertRewriter& rewriter) const override
    {
        auto srcType = dyn_cast<asctile::LocalTensorType>(op.getSourceType());
        auto dstType = dyn_cast<asctile::LocalTensorType>(op.getResultType());
        if (!srcType || !dstType || srcType.getLoc() != asctile::TensorLocation::UB ||
            dstType.getLoc() != asctile::TensorLocation::UB)
            return op.emitOpError("operand and result must be local tensors located in UB");
        int64_t rank = srcType.getRank();
        if (rank > 2 || rank != dstType.getRank())
            return op.emitOpError("only supports 1D or 2D tensors as operand and result");
        if (!op.getSizes().empty() || !op.getStrides().empty())
            return op.emitOpError("must have fully static sizes and strides");
        if (ArrayRef(computeStrides(op.getStaticSizes())) != op.getStaticStrides())
            return op.emitOpError("must have row-major strides");
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto [dstTypeConv, dstShape] = convertTypeShape(dstType);
        Value blockCount = consts.i16(dstShape[0]);
        int64_t elementsPerBlock = ascendc::ubBlockSize / ascendc::getElementTypeSize(dstTypeConv);
        Value blockLen = consts.i16(dstShape[1] / elementsPerBlock);
        auto [srcTypeConv, srcShape] = convertTypeShape(srcType);
        Value srcGap = consts.i16((srcShape[1] - dstShape[1]) / elementsPerBlock);
        Value dstGap = consts.i16(0);
        Value dst = createTensorOp(rewriter, loc, dstType);
        Value params = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyParamsType>(), ValueRange{blockCount, blockLen, srcGap, dstGap});
        auto offsets = llvm::map_to_vector<2>(op.getMixedOffsets(), [&rewriter, loc](OpFoldResult ofr) {
            return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
        });
        Value offset = offsets[0];
        if (rank == 2) {
            offset = rewriter.createOrFold<arith::MulIOp>(loc, offsets[0], consts.index(srcShape[1]));
            offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, offsets[1]);
        }
        offset = rewriter.createOrFold<arith::IndexCastOp>(loc, rewriter.getI32Type(), offset);
        Value src = rewriter.create<ascendc::LocalTensorSubIndexOp>(
            loc, srcTypeConv, rewriter.getRemappedValue(op.getSource()), offset);
        auto copyOp = rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, params);
        copyOp.setDirection(ascendc::TPosition::VECCALC, ascendc::TPosition::VECCALC);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSplat : public ConvertOp<tensor::SplatOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(tensor::SplatOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<ascendc::DuplicateL2Op>(loc, dst, op.getInput(), consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerTensorPass : public asclower::impl::LowerTensorBase<LowerTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<tensor::EmptyOp, tensor::SplatOp>();
        target.addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertEmpty, ConvertExtractSlice, ConvertSplat>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerTensorPass() { return std::make_unique<LowerTensorPass>(); }
