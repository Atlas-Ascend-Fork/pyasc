/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_VERIFYTENSORLOCATION
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

using TL = TensorLocation;
using LocVector = SmallVector<TL, 8>;

bool badLoc(Operation* op, LocalTensorType type, StringRef name, ArrayRef<TL> allow)
{
    auto loc = type.getLoc();
    if (loc != TL::Auto && !llvm::is_contained(allow, loc)) {
        op->emitError() << name << " tensor location must be "
                        << llvm::join(llvm::map_range(allow, stringifyTensorLocation), ", ") << ", got "
                        << stringifyTensorLocation(loc);
        return true;
    }
    return false;
}

struct VerifyTensorLocationPass : public asctile::impl::VerifyTensorLocationBase<VerifyTensorLocationPass> {
    void runOnOperation() override;
};

void VerifyTensorLocationPass::runOnOperation()
{
    func::FuncOp funcOp = getOperation();
    funcOp.walk([this](Operation* op) {
        if (op->mightHaveTrait<OpTrait::IsTerminator>())
            return;
        auto isAutoTensor = [](Type type) {
            auto tensor = dyn_cast<LocalTensorType>(type);
            return tensor && tensor.getLoc() == TL::Auto;
        };
        if (llvm::any_of(op->getOperandTypes(), isAutoTensor)) {
            op->emitError(
                "Unable to resolve location for input tensor(s) of the current operation. "
                "Please provide explicit tensor locations in related creation or memory operations.");
            signalPassFailure();
        }
        if (llvm::any_of(op->getResultTypes(), isAutoTensor)) {
            op->emitError(
                "Unable to resolve location for result tensor(s) of the current operation. "
                "Please provide explicit tensor locations in related memory operations.");
            signalPassFailure();
        }
    });
    // Further checks are meaningless if tensor locations are not fully resolved
    if (getPassState().irAndPassFailed.getInt())
        return;
    funcOp.walk([this](LoadOp op) {
        if (badLoc(op, op.getType(), "result", {TL::UB, TL::L1, TL::L0A, TL::L0B, TL::BT}))
            signalPassFailure();
    });
    funcOp.walk([this](StoreOp op) {
        if (badLoc(op, op.getValue().getType(), "src", {TL::UB, TL::L0C}))
            signalPassFailure();
    });
    funcOp.walk([this](AccumulatorOp op) {
        if (op.getBias() && badLoc(op, op.getBias().getType(), "bias", TL::BT))
            signalPassFailure();
    });
    funcOp.walk([this](MatmulOp op) {
        if (badLoc(op, op.getMatrixA().getType(), "input", TL::L0A) ||
            badLoc(op, op.getMatrixB().getType(), "other", TL::L0B) || badLoc(op, op.getType(), "result", TL::L0C) ||
            op.getBias() && badLoc(op, op.getBias().getType(), "bias", TL::BT))
            signalPassFailure();
    });
    funcOp.walk([this](MatmulAccOp op) {
        if (badLoc(op, op.getMatrixA().getType(), "input", TL::L0A) ||
            badLoc(op, op.getMatrixB().getType(), "other", TL::L0B) ||
            badLoc(op, op.getAcc().getType(), "acc", TL::L0C))
            signalPassFailure();
    });
    funcOp.walk([this](TransposeOp op) {
        LocVector allow = {TL::UB};
        if (op.getType().getRank() == 2)
            allow.append({TL::L1, TL::L0A, TL::L0B});
        if (badLoc(op, op.getOperand().getType(), "input", allow))
            signalPassFailure();
    });
    funcOp.walk([this](DumpTensorOp op) {
        auto type = dyn_cast<LocalTensorType>(op.getTensor().getType());
        if (!type)
            return;
        if (badLoc(op, type, "tensor", {TL::L1, TL::UB}))
            signalPassFailure();
    });
}

} // namespace

std::unique_ptr<Pass> mlir::asctile::createVerifyTensorLocationPass()
{
    return std::make_unique<VerifyTensorLocationPass>();
}
