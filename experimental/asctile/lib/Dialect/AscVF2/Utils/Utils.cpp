/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscVF2/IR/AscVF2.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include <climits>

namespace mlir {
namespace ascvf2 {

namespace {

std::optional<ReductionInfo> getSupportedReductionInfo(
    asctile::ReduceKind kind, Value operand, asctile::LocalTensorType inputType, asctile::LocalTensorType resultType,
    ReductionMode mode)
{
    if ((kind != asctile::ReduceKind::Sum && kind != asctile::ReduceKind::Max && kind != asctile::ReduceKind::Min) ||
        inputType.getRank() == 0 || inputType.getLoc() != asctile::TensorLocation::UB ||
        resultType.getLoc() != asctile::TensorLocation::UB || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getElementType() != resultType.getElementType())
        return std::nullopt;
    return ReductionInfo{kind, operand, inputType, resultType, mode};
}

bool isSupportedBroadcastOperation(Operation* op)
{
    auto broadcastOp = dyn_cast<asctile::BroadcastOp>(op);
    if (!broadcastOp)
        return false;
    auto sourceType = dyn_cast<asctile::LocalTensorType>(broadcastOp.getOperand().getType());
    auto resultType = dyn_cast<asctile::LocalTensorType>(broadcastOp.getResult().getType());
    if (!sourceType || !resultType || sourceType.getRank() == 0 || sourceType.getRank() > resultType.getRank() ||
        sourceType.getLoc() != asctile::TensorLocation::UB || resultType.getLoc() != asctile::TensorLocation::UB ||
        !sourceType.hasStaticShape() || !resultType.hasStaticShape() ||
        sourceType.getElementType() != resultType.getElementType())
        return false;
    if (sourceType.getShape().back() != 1)
        return false;
    unsigned resultOffset = resultType.getRank() - sourceType.getRank();
    return llvm::all_of(
        llvm::zip(sourceType.getShape().drop_back(), resultType.getShape().drop_front(resultOffset)),
        [](auto dimensions) { return std::get<0>(dimensions) == std::get<1>(dimensions); });
}

} // namespace

FailureOr<unsigned> getElementSizeInBytes(Type type)
{
    if (!type.isIntOrFloat())
        return failure();
    unsigned bitWidth = type.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % CHAR_BIT != 0)
        return failure();
    return bitWidth / CHAR_BIT;
}

FailureOr<unsigned> getVFVectorWidth(Operation* operation, Type elementType)
{
    constexpr unsigned defaultVectorBytes = 256;
    unsigned vectorBytes = defaultVectorBytes;
    if (auto module = operation->getParentOfType<ModuleOp>()) {
        auto attribute = module->getAttrOfType<IntegerAttr>("asc.vf_vec_len");
        if (attribute && attribute.getValue().isStrictlyPositive())
            vectorBytes = static_cast<unsigned>(attribute.getValue().getZExtValue());
    }
    auto elementSize = getElementSizeInBytes(elementType);
    if (failed(elementSize) || *elementSize == 0 || vectorBytes % *elementSize != 0)
        return failure();
    return vectorBytes / *elementSize;
}

bool isSupportedElementwiseOperation(Operation* op)
{
    return llvm::TypeSwitch<Operation*, bool>(op)
        .Case<
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp,
            arith::DivSIOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaxSIOp,
            arith::MinSIOp, arith::AndIOp, arith::OrIOp, arith::NegFOp, math::AbsFOp, math::AbsIOp, math::ExpOp,
            math::LogOp, math::SqrtOp, asctile::AddSOp, asctile::SubSOp, asctile::MulSOp, asctile::DivSOp,
            asctile::MinSOp, asctile::MaxSOp, asctile::ShLSOp, asctile::ShRSOp, asctile::LeakyReluOp, asctile::ReluOp,
            asctile::BitwiseNotOp>([](auto) { return true; })
        .Default([](auto) { return false; });
}

bool isVectorizableGroupOperation(Operation* op)
{
    if (isSupportedBroadcastOperation(op))
        return isPure(op);
    if (!isSupportedElementwiseOperation(op) || !isPure(op) || op->getNumResults() != 1 || op->getNumOperands() == 0)
        return false;
    auto resultType = dyn_cast<asctile::LocalTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getLoc() != asctile::TensorLocation::UB || !resultType.hasStaticShape())
        return false;
    return llvm::all_of(op->getOperands(), [&](Value operand) {
        if (auto operandType = dyn_cast<asctile::LocalTensorType>(operand.getType()))
            return operandType == resultType;
        return operand.getType().isIntOrFloat() && operand.getType() == resultType.getElementType();
    });
}

MemRefType getUBBufferType(asctile::LocalTensorType tensorType)
{
    auto layout = AffineMap::getMultiDimIdentityMap(tensorType.getRank(), tensorType.getContext());
    auto memorySpace =
        Builder(tensorType.getContext()).getI64IntegerAttr(static_cast<int64_t>(asctile::TensorLocation::UB));
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(), layout, memorySpace);
}

std::optional<Value> resolveTensorAliasRoot(Value value)
{
    while (Operation* operation = value.getDefiningOp()) {
        if (auto toBuffer = dyn_cast<ToBufferOp>(operation)) {
            value = toBuffer.getTensor();
            continue;
        }
        if (auto toTensor = dyn_cast<ToTensorOp>(operation)) {
            value = toTensor.getBuffer();
            continue;
        }
        if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(operation)) {
            value = reinterpret.getSource();
            continue;
        }
        if (isa<memref::AllocaOp>(operation) || isa<asctile::LocalTensorType>(value.getType()))
            return value;
        return std::nullopt;
    }
    return value;
}

std::optional<TransferDescriptor> getTransferDescriptor(Operation* operation)
{
    auto read = dyn_cast<vector::TransferReadOp>(operation);
    auto write = dyn_cast<vector::TransferWriteOp>(operation);
    if (!read && !write)
        return std::nullopt;
    if ((read && read.hasOutOfBoundsDim()) || (write && write.hasOutOfBoundsDim()))
        return std::nullopt;

    std::optional<Value> root = resolveTensorAliasRoot(read ? read.getSource() : write.getSource());
    if (!root)
        return std::nullopt;

    TransferDescriptor descriptor;
    descriptor.root = *root;
    descriptor.vectorType = read ? read.getVectorType() : write.getVectorType();
    ValueRange indices = read ? read.getIndices() : write.getIndices();
    descriptor.indices.append(indices.begin(), indices.end());
    descriptor.permutationMap = read ? read.getPermutationMap() : write.getPermutationMap();
    ArrayAttr inBounds = read ? read.getInBounds() : write.getInBounds();
    descriptor.inBounds.append(inBounds.begin(), inBounds.end());
    descriptor.mask = read ? read.getMask() : write.getMask();
    return descriptor;
}

std::optional<ReductionInfo> getSupportedReductionInfo(Operation* operation)
{
    if (auto reduceOp = dyn_cast<asctile::ReduceOp>(operation)) {
        auto inputType = dyn_cast<asctile::LocalTensorType>(reduceOp.getOperand().getType());
        auto resultType = dyn_cast<asctile::LocalTensorType>(reduceOp.getResult().getType());
        if (!inputType || !resultType || reduceOp.getDims().size() != 1)
            return std::nullopt;
        auto dimension = dyn_cast<IntegerAttr>(reduceOp.getDims()[0]);
        if (!dimension || dimension.getInt() != inputType.getRank() - 1)
            return std::nullopt;

        SmallVector<int64_t> expectedShape{inputType.getShape()};
        expectedShape.back() = 1;
        if (resultType.getShape() != ArrayRef<int64_t>{expectedShape})
            return std::nullopt;
        return getSupportedReductionInfo(
            reduceOp.getKind(), reduceOp.getOperand(), inputType, resultType, ReductionMode::FinalAxis);
    }
    if (auto reduceOp = dyn_cast<asctile::ReduceAs1dOp>(operation)) {
        auto inputType = dyn_cast<asctile::LocalTensorType>(reduceOp.getOperand().getType());
        auto resultType = dyn_cast<asctile::LocalTensorType>(reduceOp.getResult().getType());
        if (!inputType || !resultType || resultType.getShape() != ArrayRef<int64_t>{1})
            return std::nullopt;
        return getSupportedReductionInfo(
            reduceOp.getKind(), reduceOp.getOperand(), inputType, resultType, ReductionMode::AllElements);
    }
    return std::nullopt;
}

} // namespace ascvf2
} // namespace mlir
