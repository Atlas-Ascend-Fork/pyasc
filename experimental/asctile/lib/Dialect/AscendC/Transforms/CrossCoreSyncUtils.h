/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCENDC_TRANSFORMS_CROSSCORESYNC_H
#define ASCTILE_DIALECT_ASCENDC_TRANSFORMS_CROSSCORESYNC_H

#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include <cstdint>

namespace mlir {
namespace ascendc {

inline constexpr uint8_t crossCoreMode = 4;
inline constexpr int32_t maxTensorId = 16;

inline bool isDataCopyOp(Operation* op, llvm::function_ref<bool(TPosition, TPosition)> pred)
{
    auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
    if (!copyOp)
        return false;
    auto direction = copyOp.getDirection();
    if (!direction)
        return false;
    auto [src, dst] = *direction;
    return pred(src, dst);
}

inline void createSetFlag(OpBuilder& builder, Location loc, int32_t flagId, Pipe pipe, bool dualSync = false)
{
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::CrossCoreSetFlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
    if (dualSync)
        builder.create<ascendc::CrossCoreSetFlagOp>(loc, consts.i32(maxTensorId + flagId), crossCoreMode, pipe);
}

inline void createWaitFlag(OpBuilder& builder, Location loc, int32_t flagId, Pipe pipe, bool dualSync = false)
{
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::CrossCoreWaitFlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
    if (dualSync)
        builder.create<ascendc::CrossCoreWaitFlagOp>(loc, consts.i32(maxTensorId + flagId), crossCoreMode, pipe);
}

inline SmallVector<Operation*> collectGroupOps(func::FuncOp funcOp)
{
    SmallVector<Operation*> groups;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op)) {
            groups.push_back(op);
            return WalkResult::skip();
        }
        return WalkResult::advance();
    });
    return groups;
}

inline SmallVector<Operation*> collectOps(Operation* groupOp, llvm::function_ref<bool(Operation*)> pred)
{
    SmallVector<Operation*> ops;
    groupOp->walk([&](Operation* op) {
        if (pred(op))
            ops.push_back(op);
    });
    return ops;
}

} // namespace ascendc
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCENDC_TRANSFORMS_CROSSCORESYNC_H
