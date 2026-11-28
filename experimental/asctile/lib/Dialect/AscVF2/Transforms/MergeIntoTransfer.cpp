/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF2/IR/AscVF2.h"
#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_MERGEINTOTRANSFER
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct MergeBroadcastIntoTransfer : public OpRewritePattern<vector::BroadcastOp> {
    using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(vector::BroadcastOp op, PatternRewriter& rewriter) const override
    {
        auto read = op.getSource().getDefiningOp<vector::TransferReadOp>();
        if (!read || read->getNextNode() != op)
            return failure();
        if (!op.getSource().hasOneUse())
            return failure();
        if (read.getMask() || read.hasOutOfBoundsDim())
            return failure();

        auto sourceType = dyn_cast<VectorType>(op.getSource().getType());
        auto resultType = op.getResultVectorType();
        if (!sourceType || sourceType.getRank() != 1 || sourceType.getDimSize(0) != 1 || resultType.getRank() != 1 ||
            resultType.getScalableDims()[0] || sourceType.getElementType() != resultType.getElementType())
            return failure();

        FailureOr<unsigned> vectorWidth = ascvf2::getVFVectorWidth(op, resultType.getElementType());
        if (failed(vectorWidth) || resultType.getDimSize(0) <= 1 || resultType.getDimSize(0) > *vectorWidth)
            return failure();

        AffineMap broadcastMap = AffineMap::get(
            cast<ShapedType>(read.getSource().getType()).getRank(), 0, getAffineConstantExpr(0, op.getContext()),
            op.getContext());
        rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
            op, resultType, read.getSource(), read.getIndices(), AffineMapAttr::get(broadcastMap), read.getPadding(),
            Value(), rewriter.getBoolArrayAttr({true}));
        if (read->use_empty())
            rewriter.eraseOp(read);
        return success();
    }
};

struct MergeIntoTransferPass : public ascvf2::impl::MergeIntoTransferBase<MergeIntoTransferPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        RewritePatternSet patterns(&getContext());
        patterns.add<MergeBroadcastIntoTransfer>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createMergeIntoTransferPass() { return std::make_unique<MergeIntoTransferPass>(); }
