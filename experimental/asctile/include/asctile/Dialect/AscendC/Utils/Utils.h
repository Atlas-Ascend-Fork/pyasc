/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCENDC_UTILS_UTILS_H
#define ASCTILE_DIALECT_ASCENDC_UTILS_UTILS_H

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace ascendc {

SmallVector<OpOperand*> collectAllUses(ascendc::LocalTensorAutoOp tensorOp);

Pipe getOpPipeExt(Operation* op, Pipe defaultPipe = Pipe::PIPE_S);

} // namespace ascendc
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCENDC_UTILS_UTILS_H
