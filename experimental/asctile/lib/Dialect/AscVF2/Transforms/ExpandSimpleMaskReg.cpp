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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_EXPANDSIMPLEMASKREG
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

std::optional<ascendc::MaskPattern> getMaskPattern(int64_t count, unsigned hardwareWidth)
{
    if (count == hardwareWidth)
        return ascendc::MaskPattern::ALL;
    switch (count) {
        case 1:
            return ascendc::MaskPattern::VL1;
        case 2:
            return ascendc::MaskPattern::VL2;
        case 3:
            return ascendc::MaskPattern::VL3;
        case 4:
            return ascendc::MaskPattern::VL4;
        case 8:
            return ascendc::MaskPattern::VL8;
        case 16:
            return ascendc::MaskPattern::VL16;
        case 32:
            return ascendc::MaskPattern::VL32;
        case 64:
            return ascendc::MaskPattern::VL64;
        case 128:
            return ascendc::MaskPattern::VL128;
        default:
            return std::nullopt;
    }
}

struct ExpandSimpleMaskReg : public OpRewritePattern<ascvf2::SimpleMaskRegOp> {
    using OpRewritePattern<ascvf2::SimpleMaskRegOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ascvf2::SimpleMaskRegOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<ascendc::MaskRegType>(op.getResult().getType()))
            return failure();

        Type elementType = op.getElementType();
        FailureOr<unsigned> hardwareWidth = ascvf2::getVFVectorWidth(op, elementType);
        if (failed(hardwareWidth))
            return failure();

        if (auto count = op.getCount()) {
            int64_t staticCount = count->getSExtValue();
            if (auto pattern = getMaskPattern(staticCount, *hardwareWidth)) {
                rewriter.replaceOpWithNewOp<ascendc::CreateMaskOp>(
                    op, ascendc::MaskRegType::get(getContext()), elementType, *pattern);
                return success();
            }

            Value counter = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), staticCount);
            Value variable = rewriter.create<emitasc::VariableOp>(op.getLoc(), counter);
            rewriter.replaceOpWithNewOp<ascendc::UpdateMaskOp>(
                op, ascendc::MaskRegType::get(getContext()), variable, elementType);
            return success();
        }

        Value variable = rewriter.create<emitasc::VariableOp>(op.getLoc(), op.getDynamicCount());
        rewriter.replaceOpWithNewOp<ascendc::UpdateMaskOp>(
            op, ascendc::MaskRegType::get(getContext()), variable, elementType);
        return success();
    }
};

struct ExpandSimpleMaskRegPass : public ascvf2::impl::ExpandSimpleMaskRegBase<ExpandSimpleMaskRegPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        RewritePatternSet patterns(&getContext());
        patterns.add<ExpandSimpleMaskReg>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createExpandSimpleMaskRegPass()
{
    return std::make_unique<ExpandSimpleMaskRegPass>();
}
