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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_FULFILLDATATRANSFER
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

using TL = TensorLocation;

bool isValidCopy(TL src, TL dst)
{
    if (src == TL::L1)
        return dst == TL::L0A || dst == TL::L0B || dst == TL::BT;
    if (src == TL::L0C)
        return dst == TL::L1 || dst == TL::UB;
    if (src == TL::UB)
        return dst == TL::L1;
    return false;
}

std::optional<TL> findIntermediateCopy(TL src, TL dst)
{
    for (TL mid : {TL::L1, TL::UB}) {
        if (isValidCopy(src, mid) && isValidCopy(mid, dst)) {
            return mid;
        }
    }
    return std::nullopt;
}

struct FulfillCopy : OpRewritePattern<CopyOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(CopyOp op, PatternRewriter& rewriter) const override
    {
        auto srcType = op.getBase().getType();
        auto srcLoc = srcType.getLoc();
        auto dstType = op.getType();
        auto dstLoc = dstType.getLoc();
        if (isValidCopy(srcLoc, dstLoc))
            return failure();
        auto midLoc = findIntermediateCopy(srcLoc, dstLoc);
        if (!midLoc)
            return failure();
        auto midType = LocalTensorType::get(dstType.getShape(), dstType.getElementType(), *midLoc);
        DistribModeAttr distribMode = DistribModeAttr{};
        if (*midLoc == TL::L1 && srcLoc == TL::UB)
            distribMode = op.getDistribAttr();
        auto midCopy = rewriter.create<CopyOp>(op.getLoc(), midType, op.getBase(), op.getOffsets(), distribMode);
        if (*midLoc == TL::L1 && dstLoc == TL::BT)
            midCopy->setAttr(attr::isBias, rewriter.getUnitAttr());
        Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0L, 32U);
        SmallVector<Value, 2> offsets{static_cast<size_t>(srcType.getRank()), zero};
        rewriter.replaceOpWithNewOp<CopyOp>(op, dstType, midCopy, offsets, DistribModeAttr{});
        return success();
    }
};

struct FulfillDataTransferPass : public asctile::impl::FulfillDataTransferBase<FulfillDataTransferPass> {
    void runOnOperation() override;
};

void FulfillDataTransferPass::runOnOperation()
{
    func::FuncOp funcOp = getOperation();
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<FulfillCopy>(context);
    if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
        signalPassFailure();
        return;
    }
    funcOp.walk([this](CopyOp op) {
        SmallVector<TL, 3> allowedSrcLocs{TL::L1, TL::L0C, TL::UB};
        SmallVector<TL, 3> allowedDstLocs;
        auto srcType = op.getBase().getType();
        auto srcLoc = srcType.getLoc();
        auto dstLoc = op.getType().getLoc();
        if (srcLoc == TL::L1)
            allowedDstLocs = {TL::L0A, TL::L0B, TL::BT};
        else if (srcLoc == TL::L0C)
            allowedDstLocs = {TL::L1, TL::UB};
        else if (srcLoc == TL::UB)
            allowedDstLocs = {TL::L1};
        if (!llvm::is_contained(allowedSrcLocs, srcLoc) || !llvm::is_contained(allowedDstLocs, dstLoc)) {
            StringRef srcLocStr = stringifyTensorLocation(srcLoc);
            auto diag =
                op.emitError()
                << "Direct data transfer from " << srcLocStr << " to " << stringifyTensorLocation(dstLoc)
                << " is not supported. Please call copy() with explicit locations to fulfill the requested data flow.";
            diag.attachNote(op.getBase().getLoc()) << "source tensor with " << srcLocStr << " location defined here:";
            signalPassFailure();
        }
    });
}

} // namespace

std::unique_ptr<Pass> mlir::asctile::createFulfillDataTransferPass()
{
    return std::make_unique<FulfillDataTransferPass>();
}
