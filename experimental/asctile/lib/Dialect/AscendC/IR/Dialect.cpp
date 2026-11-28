/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/IR/Interfaces.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

// Inline interface definitions
#include "asctile/Dialect/AscendC/IR/AscTileAscendCOpInterfaces.cpp.inc"

#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {

template <typename ConcreteOp>
struct ReusableSrcModel : public ascendc::OpWithReusableSrc::ExternalModel<ReusableSrcModel<ConcreteOp>, ConcreteOp> {
    SmallVector<Value> getSrcTensors(Operation* op) const { return cast<ascendc::OpWithSrc>(op).getSrcTensors(); }

    SmallVector<Value> getDstTensors(Operation* op) const { return cast<ascendc::OpWithDst>(op).getDstTensors(); }

    bool canReuse(Operation* op, OpOperand& lhs, OpOperand& rhs) const
    {
        auto dstTensors = getDstTensors(op);
        auto srcTensors = getSrcTensors(op);
        return getElementTypeOrSelf(lhs.get()) == getElementTypeOrSelf(rhs.get()) &&
               (llvm::is_contained(dstTensors, lhs.get()) && llvm::is_contained(srcTensors, rhs.get()) ||
                llvm::is_contained(dstTensors, rhs.get()) && llvm::is_contained(srcTensors, lhs.get()));
    }
};

template <>
struct ReusableSrcModel<ascendc::CastL2Op>
    : public ascendc::OpWithReusableSrc::ExternalModel<ReusableSrcModel<ascendc::CastL2Op>, ascendc::CastL2Op> {
    SmallVector<Value> getSrcTensors(Operation* op) const { return cast<ascendc::OpWithSrc>(op).getSrcTensors(); }

    SmallVector<Value> getDstTensors(Operation* op) const { return cast<ascendc::OpWithDst>(op).getDstTensors(); }

    bool canReuse(Operation* op, OpOperand& lhs, OpOperand& rhs) const
    {
        auto dstTensors = getDstTensors(op);
        auto srcTensors = getSrcTensors(op);
        auto check = [&](OpOperand& lhs, OpOperand& rhs) {
            return llvm::is_contained(dstTensors, lhs.get()) && llvm::is_contained(srcTensors, rhs.get()) &&
                   getElementTypeOrSelf(lhs.get()).getIntOrFloatBitWidth() <=
                       getElementTypeOrSelf(rhs.get()).getIntOrFloatBitWidth();
        };
        return check(lhs, rhs) || check(rhs, lhs);
    }
};

template <typename... ConcreteOps>
void attachReusableSrcModels(MLIRContext& context)
{
    (ConcreteOps::template attachInterface<ReusableSrcModel<ConcreteOps>>(context), ...);
}

} // namespace

void ascendc::registerReusableSrcModels(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext* context, ascendc::AscendCDialect* dialect) {
        attachReusableSrcModels<
            ascendc::MmadOp, ascendc::MmadWithBiasOp, ascendc::CompareL0Op, ascendc::CompareL2Op,
            ascendc::CompareScalarL0Op, ascendc::CompareScalarL2Op, ascendc::SelectL0Op, ascendc::SelectL2Op,
            ascendc::SelectScalarL0Op, ascendc::SelectScalarL2Op, ascendc::CastL2Op>(*context);
        for (RegisteredOperationName opName : context->getRegisteredOperations()) {
            if (&opName.getDialect() != dialect)
                continue;
            if (opName.hasInterface<ascendc::UnaryOp>() || opName.hasInterface<ascendc::BinaryOp>() ||
                opName.hasInterface<ascendc::VecScalarOp>())
                opName.attachInterface<ReusableSrcModel<Operation>>();
        }
    });
}
