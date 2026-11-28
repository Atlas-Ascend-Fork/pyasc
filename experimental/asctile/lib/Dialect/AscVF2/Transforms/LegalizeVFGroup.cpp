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
#include "asctile/Dialect/AscVF2/Utils/RegTypeConverter.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_LEGALIZEVFGROUP
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct ConvertVFGroup : public OpConversionPattern<ascvf2::VFGroupOp> {
    using OpConversionPattern<ascvf2::VFGroupOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ascvf2::VFGroupOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        SmallVector<Type, 4> resultTypes;
        resultTypes.reserve(op.getNumResults());
        for (OpResult result : op.getResults()) {
            Type resultType = typeConverter->convertType(result.getType());
            if (!resultType)
                return failure();
            resultTypes.push_back(resultType);
        }

        auto newOp = cast<ascvf2::VFGroupOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
        rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
        for (auto [result, resultType] : llvm::zip_equal(newOp.getResults(), resultTypes))
            result.setType(resultType);

        auto yieldOp = cast<ascvf2::YieldOp>(newOp.getBody()->getTerminator());
        SmallVector<Value, 4> yieldOperands;
        yieldOperands.reserve(yieldOp.getNumOperands());
        for (Value operand : yieldOp.getOperands()) {
            Value remappedOperand = rewriter.getRemappedValue(operand);
            if (!remappedOperand)
                return failure();
            yieldOperands.push_back(remappedOperand);
        }
        for (auto [result, operand] : llvm::zip_equal(newOp.getResults(), yieldOperands)) {
            if (result.getType() != operand.getType())
                return op.emitError("cannot convert VF group yield operand type")
                       << " " << operand.getType() << " to result type " << result.getType();
        }
        rewriter.modifyOpInPlace(yieldOp, [&] { yieldOp->setOperands(yieldOperands); });
        rewriter.replaceOp(op, newOp.getResults());
        return success();
    }
};

void hoistLocalTensorAutoOps(ascvf2::VFGroupOp group)
{
    SmallVector<ascendc::LocalTensorAutoOp, 4> tensors;
    group.getBody()->walk([&](ascendc::LocalTensorAutoOp tensor) {
        if (tensor->getNumOperands() == 0)
            tensors.push_back(tensor);
    });
    for (ascendc::LocalTensorAutoOp tensor : llvm::reverse(tensors))
        tensor->moveBefore(group);
}

struct LegalizeVFGroupPass : public ascvf2::impl::LegalizeVFGroupBase<LegalizeVFGroupPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        SmallVector<ascvf2::VFGroupOp, 4> groups;
        func.walk([&](ascvf2::VFGroupOp group) { groups.push_back(group); });
        for (ascvf2::VFGroupOp group : groups)
            hoistLocalTensorAutoOps(group);

        ascvf2::RegTypeConverter typeConverter;
        ConversionTarget target(getContext());
        target.addLegalOp<ascvf2::YieldOp, UnrealizedConversionCastOp>();
        target.addDynamicallyLegalOp<ascvf2::VFGroupOp>([](ascvf2::VFGroupOp op) {
            return llvm::none_of(
                       op.getOperands(),
                       [](Value operand) { return isa<asctile::LocalTensorType>(operand.getType()); }) &&
                   llvm::none_of(op.getResults(), [](OpResult result) {
                       return isa<asctile::LocalTensorType>(result.getType());
                   });
        });
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        RewritePatternSet patterns(&getContext());
        patterns.add<ConvertVFGroup>(typeConverter, &getContext());
        if (failed(applyPartialConversion(func, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createLegalizeVFGroupPass() { return std::make_unique<LegalizeVFGroupPass>(); }
