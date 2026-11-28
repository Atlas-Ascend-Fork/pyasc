/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCVF2_UTILS_UTILS_H
#define ASCTILE_DIALECT_ASCVF2_UTILS_UTILS_H

#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace ascvf2 {

inline constexpr llvm::StringRef vectorFuncAttr = "ascvf2.vector_func";

enum class ReductionMode {
    FinalAxis,
    AllElements,
};

struct ReductionInfo {
    asctile::ReduceKind kind;
    Value operand;
    asctile::LocalTensorType inputType;
    asctile::LocalTensorType resultType;
    ReductionMode mode;
};

struct TransferDescriptor {
    Value root;
    VectorType vectorType;
    SmallVector<Value, 4> indices;
    AffineMap permutationMap;
    SmallVector<bool, 2> inBounds;
    Value mask;

    bool matches(const TransferDescriptor& other) const
    {
        return root == other.root && vectorType == other.vectorType && permutationMap == other.permutationMap &&
               llvm::equal(inBounds, other.inBounds) && mask == other.mask && indices.size() == other.indices.size() &&
               llvm::equal(indices, other.indices);
    }
};

FailureOr<unsigned> getElementSizeInBytes(Type type);

FailureOr<unsigned> getVFVectorWidth(Operation* operation, Type elementType);

bool isSupportedElementwiseOperation(Operation* operation);

bool isVectorizableGroupOperation(Operation* operation);

MemRefType getUBBufferType(asctile::LocalTensorType tensorType);

std::optional<Value> resolveTensorAliasRoot(Value value);

std::optional<TransferDescriptor> getTransferDescriptor(Operation* operation);

std::optional<ReductionInfo> getSupportedReductionInfo(Operation* operation);

} // namespace ascvf2
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCVF2_UTILS_UTILS_H
