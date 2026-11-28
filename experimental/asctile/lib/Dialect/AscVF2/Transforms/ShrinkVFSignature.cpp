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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/BitVector.h"

#include <cassert>

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_SHRINKVFSIGNATURE
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

struct ShrinkVFSignaturePass : public ascvf2::impl::ShrinkVFSignatureBase<ShrinkVFSignaturePass> {
    void runOnOperation() override
    {
        DenseMap<func::FuncOp, func::CallOp> calls;

        getOperation()->walk([&](ascvf2::VFGroupOp group) {
            group->walk([&](func::CallOp call) {
                auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
                if (!callee || !callee->getAttr(ascvf2::vectorFuncAttr))
                    return WalkResult::advance();

                assert(!calls.contains(callee) && "expected one call per vector function");
                calls[callee] = call;
                return WalkResult::advance();
            });
        });

        OpBuilder builder(&getContext());
        for (auto& [function, call] : calls) {
            SmallVector<std::optional<unsigned>, 4> resultMapping(function.getNumResults());
            unsigned numKeptResults = 0;
            for (auto [index, result] : llvm::enumerate(call.getResults())) {
                if (result.use_empty())
                    continue;
                resultMapping[index] = numKeptResults++;
            }
            if (numKeptResults == function.getNumResults())
                continue;

            BitVector erasedResults(function.getNumResults(), true);
            for (auto [index, mapping] : llvm::enumerate(resultMapping)) {
                if (mapping)
                    erasedResults.reset(index);
            }
            function.eraseResults(erasedResults);

            SmallVector<func::ReturnOp, 4> returns;
            function.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
            for (func::ReturnOp returnOp : returns) {
                SmallVector<Value, 4> operands;
                for (auto [index, mapping] : llvm::enumerate(resultMapping)) {
                    if (mapping)
                        operands.push_back(returnOp.getOperand(index));
                }
                returnOp.getOperandsMutable().assign(operands);
            }

            builder.setInsertionPoint(call);
            auto newCall = builder.create<func::CallOp>(call.getLoc(), function, call.getOperands());
            for (auto [index, mapping] : llvm::enumerate(resultMapping)) {
                if (!mapping)
                    continue;
                call.getResult(index).replaceAllUsesWith(newCall.getResult(*mapping));
            }
            call.erase();
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createShrinkVFSignaturePass() { return std::make_unique<ShrinkVFSignaturePass>(); }
