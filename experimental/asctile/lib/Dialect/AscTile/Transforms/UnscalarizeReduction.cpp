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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_UNSCALARIZEREDUCTION
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const unscalarizeAttr = "asctile.op_to_unscalarize";

template <typename OpT>
struct ConvertScalar : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    using OpConversionPattern<OpT>::typeConverter;

    LogicalResult matchAndRewrite(
        OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto resultType = typeConverter->convertType(op.getType());
        rewriter.replaceOpWithNewOp<OpT>(op, resultType, adaptor.getOperands(), op->getAttrs())
            ->removeAttr(unscalarizeAttr);
        return success();
    }
};

struct ConvertReduce : OpConversionPattern<asctile::ReduceAs1dOp> {
    using OpConversionPattern::OpConversionPattern;
    using OpConversionPattern::typeConverter;

    LogicalResult matchAndRewrite(
        asctile::ReduceAs1dOp op, asctile::ReduceAs1dOp::Adaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        if (isa<asctile::LocalTensorType>(op.getType()))
            return failure();
        auto resultType = typeConverter->convertType(op.getType());
        rewriter.replaceOpWithNewOp<asctile::ReduceAs1dOp>(op, resultType, op->getOperands(), op->getAttrs())
            ->removeAttr(unscalarizeAttr);
        return success();
    }
};

struct ConvertSplat : OpConversionPattern<tensor::SplatOp> {
    using OpConversionPattern::OpConversionPattern;
    using OpConversionPattern::typeConverter;

    LogicalResult matchAndRewrite(
        tensor::SplatOp op, tensor::SplatOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto resultType = typeConverter->convertType(op.getType());
        rewriter.replaceOpWithNewOp<asctile::BroadcastOp>(op, resultType, adaptor.getOperands(), op->getAttrs())
            ->removeAttr(unscalarizeAttr);
        return success();
    }
};

Value materialize(OpBuilder& builder, Type type, ValueRange inputs, Location loc)
{
    Value input = inputs[0];
    if (input.getType().isIntOrFloat() && isa<asctile::LocalTensorType>(type)) {
        return builder.create<tensor::SplatOp>(loc, type, input);
    }
    if (isa<asctile::LocalTensorType>(input.getType()) && type.isIntOrFloat()) {
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto tensorType = cast<asctile::LocalTensorType>(input.getType());
        SmallVector<Value> indices(tensorType.getRank(), zero);
        return builder.create<tensor::ExtractOp>(loc, input, indices);
    }
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
}

bool isSentinelOp(Operation* op) { return isa<tensor::SplatOp>(op); }

bool isTensorizableOp(Operation* op)
{
    return isa<
        asctile::ReduceAs1dOp, arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp,
        arith::DivFOp, arith::DivSIOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp,
        arith::MaxSIOp, arith::MinSIOp, arith::AndIOp, arith::OrIOp, arith::NegFOp, math::AbsFOp, math::AbsIOp,
        math::ExpOp, math::LogOp, math::SqrtOp>(op);
}

bool supported(Operation* op) { return isTensorizableOp(op) || isSentinelOp(op); }

enum class Status { NotVectorizable = -1, InProgress = 0, Vectorizable = 1 };

// Finds a sequence of scalar computations whose result is used in a vector operation.
bool canTensorize(Operation* op, llvm::DenseMap<Operation*, Status>& info)
{
    if (info.count(op)) {
        const auto& status = info[op];
        return status == Status::InProgress || status == Status::Vectorizable;
    }
    info[op] = Status::InProgress;

    if (!supported(op)) {
        info[op] = Status::NotVectorizable;
        return false;
    }

    if (isTensorizableOp(op)) {
        auto result = op->getResult(0);
        for (auto* use : result.getUsers()) {
            if (!canTensorize(use, info)) {
                info[op] = Status::NotVectorizable;
                return false;
            }
        }
    }
    info[op] = Status::Vectorizable;

    for (Value opnd : op->getOperands()) {
        if (auto* defOp = opnd.getDefiningOp())
            canTensorize(defOp, info);
    }
    return true;
}

void markOps(func::FuncOp root)
{
    root.walk([](asctile::ReduceAs1dOp op) {
        llvm::DenseMap<Operation*, Status> info;
        if (canTensorize(op, info)) {
            for (auto& [op, status] : info) {
                if (status == Status::Vectorizable)
                    op->setAttr(unscalarizeAttr, UnitAttr::get(op->getContext()));
            }
        }
    });
}

struct UnscalarizeReductionPass : public asctile::impl::UnscalarizeReductionBase<UnscalarizeReductionPass> {
    void runOnOperation() override
    {
        func::FuncOp op = getOperation();
        markOps(op);
        TypeConverter converter;
        converter.addConversion([](Type type) { return std::optional<Type>{type}; });
        converter.addConversion([](IntegerType type) { return asctile::LocalTensorType::get(1, type); });
        converter.addConversion([](FloatType type) { return asctile::LocalTensorType::get(1, type); });
        converter.addArgumentMaterialization(materialize);
        converter.addSourceMaterialization(materialize);
        converter.addTargetMaterialization(materialize);
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.markUnknownOpDynamicallyLegal([](Operation* op) { return !op->hasAttr(unscalarizeAttr); });
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertScalar<arith::AddFOp>, ConvertScalar<arith::SubFOp>, ConvertScalar<arith::MulFOp>,
            ConvertScalar<arith::DivFOp>, ConvertScalar<arith::MinimumFOp>, ConvertScalar<arith::MaximumFOp>,
            ConvertScalar<arith::AddIOp>, ConvertScalar<arith::SubIOp>, ConvertScalar<arith::MulIOp>,
            ConvertScalar<arith::DivSIOp>, ConvertScalar<arith::MinSIOp>, ConvertScalar<arith::MaxSIOp>,
            ConvertScalar<math::SqrtOp>,
            //
            ConvertReduce, ConvertSplat
            //
            >(converter, context);
        if (applyFullConversion(op, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createUnscalarizeReductionPass()
{
    return std::make_unique<UnscalarizeReductionPass>();
}
