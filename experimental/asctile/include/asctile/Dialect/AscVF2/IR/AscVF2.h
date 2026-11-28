/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCVF2_IR_ASCVF2_H
#define ASCTILE_DIALECT_ASCVF2_IR_ASCVF2_H

#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// Inline dialect definitions
#include "asctile/Dialect/AscVF2/IR/AscVF2Dialect.h.inc"

// Inline enum definitions
#include "asctile/Dialect/AscVF2/IR/AscVF2Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "asctile/Dialect/AscVF2/IR/AscVF2Attributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "asctile/Dialect/AscVF2/IR/AscVF2Types.h.inc"

#define GET_OP_CLASSES
#include "asctile/Dialect/AscVF2/IR/AscVF2Ops.h.inc"

namespace mlir {
namespace ascvf2 {

void registerExternalModels(DialectRegistry& registry);

} // namespace ascvf2
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCVF2_IR_ASCVF2_H
