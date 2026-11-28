/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF2/IR/AscVF2.h"
#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_HOISTREGTENSOR
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct HoistRegTensorPass : public ascvf2::impl::HoistRegTensorBase<HoistRegTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp func = getOperation();
        if (!func->getAttr(ascvf2::vectorFuncAttr))
            return;

        SmallVector<ascendc::RegTensorOp, 16> tensors;
        func.walk([&](ascendc::RegTensorOp tensor) { tensors.push_back(tensor); });

        Block* entry = &func.getBody().front();
        for (ascendc::RegTensorOp tensor : llvm::reverse(tensors))
            tensor->moveBefore(entry, entry->begin());
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createHoistRegTensorPass() { return std::make_unique<HoistRegTensorPass>(); }
