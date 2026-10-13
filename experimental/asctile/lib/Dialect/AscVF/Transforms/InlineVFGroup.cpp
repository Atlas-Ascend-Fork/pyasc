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
#include "asctile/Dialect/AscVF/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_INLINEVFGROUP
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

struct InlineVFGroupPass : public ascvf::impl::InlineVFGroupBase<InlineVFGroupPass> {
    void runOnOperation() override
    {
        getOperation().walk([](ascvf::VFGroupOp op) {
            Block* body = op.getBody();
            auto* yieldOp = body->getTerminator();
            op->getBlock()->getOperations().splice(op->getIterator(), body->getOperations());
            yieldOp->erase();
            op->erase();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createInlineVFGroupPass() { return std::make_unique<InlineVFGroupPass>(); }
