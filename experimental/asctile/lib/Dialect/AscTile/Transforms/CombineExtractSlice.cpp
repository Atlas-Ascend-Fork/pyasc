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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_COMBINEEXTRACTSLICE
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename OpT>
struct CombineExtractSlice : public OpRewritePattern<tensor::ExtractSliceOp> {
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult combine(tensor::ExtractSliceOp slice, OpT op, PatternRewriter& rewriter) const = 0;

    LogicalResult matchAndRewrite(tensor::ExtractSliceOp op, PatternRewriter& rewriter) const override
    {
        if (!op.getSizes().empty() || !op.getStrides().empty())
            return failure();
        auto def = op.getSource().getDefiningOp<OpT>();
        if (!def)
            return failure();
        return combine(op, def, rewriter);
    }
};

struct CombineWithLoad : public CombineExtractSlice<asctile::LoadOp> {
    using CombineExtractSlice::CombineExtractSlice;

    LogicalResult combine(tensor::ExtractSliceOp slice, asctile::LoadOp op, PatternRewriter& rewriter) const override
    {
        if (!op.getRealShape().empty() || ArrayRef(computeStrides(slice.getStaticSizes())) != slice.getStaticStrides())
            return failure();
        SmallVector<Value, 4> offsets;
        for (auto [loadOffset, sliceOffset] : llvm::zip_equal(op.getOffsets(), slice.getMixedOffsets())) {
            Value addOffset;
            if (auto attr = dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(sliceOffset))) {
                addOffset = rewriter.create<arith::ConstantIntOp>(
                    slice.getLoc(), attr.getValue().getSExtValue(), loadOffset.getType());
            } else {
                auto value = cast<Value>(sliceOffset);
                addOffset = rewriter.createOrFold<arith::IndexCastOp>(value.getLoc(), loadOffset.getType(), value);
            }
            Value offset = rewriter.createOrFold<arith::AddIOp>(slice.getLoc(), loadOffset, addOffset);
            offsets.push_back(offset);
        }
        // TODO: Implement real shape clamping
        rewriter.replaceOpWithNewOp<asctile::LoadOp>(
            slice, slice.getResultType(), op.getBase(), offsets, op.getPadValue(), ValueRange{});
        return success();
    }
};

struct CombineWithSplat : public CombineExtractSlice<tensor::SplatOp> {
    using CombineExtractSlice::CombineExtractSlice;

    LogicalResult combine(tensor::ExtractSliceOp slice, tensor::SplatOp op, PatternRewriter& rewriter) const override
    {
        // This approach is similar to "arith.constant -> tensor.extract_slice" chain handled by the latter's folder
        rewriter.replaceOpWithNewOp<tensor::SplatOp>(slice, slice.getResultType(), op.getInput());
        return success();
    }
};

struct CombineExtractSlicePass : public asctile::impl::CombineExtractSliceBase<CombineExtractSlicePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<CombineWithLoad, CombineWithSplat>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createCombineExtractSlicePass()
{
    return std::make_unique<CombineExtractSlicePass>();
}
