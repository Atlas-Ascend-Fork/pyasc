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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_REPLACEFP8OPS
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

template <typename OpT>
struct CommonReplacer : OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        bool modified = false;
        rewriter.startOpModification(op);
        for (auto& operand : op->getOpOperands()) {
            Value newOp;
            Type operandType = operand.get().getType();
            if (isa<Float8E4M3FNType, Float8E5M2Type>(operandType)) {
                auto castType = rewriter.getI8Type();
                newOp = rewriter.create<arith::BitcastOp>(op->getLoc(), castType, operand.get());
            } else if (auto type = dyn_cast<asctile::GlobalTensorType>(operandType);
                       type && isa<Float8E4M3FNType, Float8E5M2Type>(type.getElementType())) {
                Type castType = asctile::GlobalTensorType::get(type.getShape(), rewriter.getI8Type());
                newOp = rewriter.create<arith::BitcastOp>(op->getLoc(), castType, operand.get());
            } else if (auto type = dyn_cast<asctile::LocalTensorType>(operandType);
                       type && isa<Float8E4M3FNType, Float8E5M2Type>(type.getElementType())) {
                Type castType = type.cloneWith(std::nullopt, rewriter.getI8Type());
                newOp = rewriter.create<arith::BitcastOp>(op->getLoc(), castType, operand.get());
            }
            if (newOp) {
                modified = true;
                operand.set(newOp);
            }
        }
        rewriter.setInsertionPointAfter(op);
        for (auto result : op->getResults()) {
            Type newType;
            if (auto type = dyn_cast<asctile::LocalTensorType>(result.getType());
                type && isa<Float8E4M3FNType, Float8E5M2Type>(type.getElementType())) {
                newType = type.cloneWith(std::nullopt, rewriter.getI8Type());
            }
            if (newType) {
                Type originalType = result.getType();
                result.setType(newType);
                auto newResult = rewriter.create<arith::BitcastOp>(op->getLoc(), originalType, result);
                rewriter.replaceAllUsesExcept(result, newResult.getResult(), newResult);
                modified = true;
            }
        }
        if (modified)
            rewriter.finalizeOpModification(op);
        else
            rewriter.cancelOpModification(op);
        return success(modified);
    }
};

struct ReplaceFp8OpsPass : public asctile::impl::ReplaceFp8OpsBase<ReplaceFp8OpsPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<
            CommonReplacer<asctile::LoadOp>, CommonReplacer<asctile::StoreOp>, CommonReplacer<asctile::BroadcastOp>,
            CommonReplacer<asctile::TransposeOp>, CommonReplacer<arith::ConstantOp>>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createReplaceFp8OpsPass() { return std::make_unique<ReplaceFp8OpsPass>(); }
