/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCTILE_UTILS_UTILS_H
#define ASCTILE_DIALECT_ASCTILE_UTILS_UTILS_H

#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace asctile {

TypedAttr getSplatAttr(arith::ConstantOp cstOp);

OpFoldResult getSplatValue(Value cstTile);

unsigned getSplitAxis(DistribMode mode, size_t rank = 2U);

SmallVector<Value> getTensorShape(OpBuilder& builder, asctile::TensorOp tensorOp);

Value materializeSplatValue(OpBuilder& builder, Value cstTile);

} // namespace asctile
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCTILE_UTILS_UTILS_H
