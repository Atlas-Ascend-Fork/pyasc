/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF/IR/AscVF.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "asctile/Dialect/AscVF/IR/AscVFOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

namespace {
bool hasStaticShapes(Operation* op)
{
    return llvm::all_of(op->getOperandTypes(), [](Type type) {
        auto tensorType = dyn_cast<ascendc::LocalTensorType>(type);
        return !tensorType || tensorType.hasStaticShape();
    });
}
} // namespace

//===----------------------------------------------------------------------===//
// VFGroupOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify()
{
    if (auto regType = dyn_cast<ascendc::RegTensorType>(getDstReg().getType())) {
        auto tensorType = cast<ascendc::LocalTensorType>(getSrcTensor().getType());
        if (regType.getElementType() != tensorType.getElementType()) {
            return emitOpError("elem type of reg tensor and local tensor must be the same");
        }
    }
    return success();
}

LogicalResult StoreOp::verify()
{
    if (auto regType = dyn_cast<ascendc::RegTensorType>(getSrcReg().getType())) {
        auto tensorType = cast<ascendc::LocalTensorType>(getDstTensor().getType());
        if (regType.getElementType() != tensorType.getElementType()) {
            return emitOpError("elem type of reg tensor and local tensor must be the same");
        }
    }
    return success();
}

LogicalResult VFGroupOp::verify()
{
    auto result = walk([&](Operation* op) {
        if (!hasStaticShapes(op)) {
            op->emitOpError("inside ascvf.vf_group must have static shape");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "asctile/Dialect/AscVF/IR/AscVFOps.cpp.inc"
        >();
}
