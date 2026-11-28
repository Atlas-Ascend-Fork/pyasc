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
#include "asctile/Dialect/AscTile/Utils/Attributes.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_CUBETRANSPOSETOLOAD
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

constexpr const char* const fusibleAttr = "asctile.fusible";

bool isL0Location(TensorLocation loc) { return loc == TensorLocation::L0A || loc == TensorLocation::L0B; }

RankedTensorType getTransposedType(LocalTensorType type)
{
    ArrayRef<int64_t> shape = type.getShape();
    return LocalTensorType::get({shape[1], shape[0]}, type.getElementType(), type.getLoc());
}

const char* getTransposeAttrName(TensorLocation loc, bool isL1)
{
    if (loc == TensorLocation::L0A)
        return isL1 ? asctile::attr::transposeAL1 : asctile::attr::transposeAL0;
    return isL1 ? asctile::attr::transposeBL1 : asctile::attr::transposeBL0;
}

void setUnitAttr(PatternRewriter& rewriter, Operation* op, const char* attr)
{
    rewriter.startOpModification(op);
    op->setAttr(attr, rewriter.getUnitAttr());
    rewriter.finalizeOpModification(op);
}

WalkResult failWithDiag(Operation* op, const char* msg, Operation* noteOp = nullptr, const char* noteMsg = nullptr)
{
    auto diag = op->emitError() << msg;
    if (noteOp != nullptr)
        diag.attachNote(noteOp->getLoc()) << noteMsg;
    return WalkResult::interrupt();
}

WalkResult markTransposeCopy(asctile::TransposeOp op)
{
    auto tileLoc = op.getType().getLoc();
    if (!isL0Location(tileLoc))
        return WalkResult::advance();

    auto copyOp = op.getOperand().getDefiningOp<asctile::CopyOp>();
    if (!copyOp) {
        return failWithDiag(
            op, "Transpose on L0A/B is supported for copy operation results only.", op.getOperand().getDefiningOp(),
            "unexpected source operation");
    }
    if (copyOp.getType().getShape().size() != 2) {
        return failWithDiag(op, "Transpose op on Cube supports 2-dim tensors only.");
    }
    auto elemSize = ascendc::getElementTypeSize(copyOp.getType());
    auto* loadOp = copyOp.getBase().getDefiningOp();
    if (!loadOp || !isa<asctile::LoadOp>(loadOp)) {
        return failWithDiag(
            copyOp, "Copy operation on Cube expects load operation results only.", loadOp,
            "unexpected source operation");
    }
    for (auto* user : loadOp->getUsers()) {
        // TODO: Add support for copyOp with several transposeOp users
        auto copyOp = dyn_cast<asctile::CopyOp>(user);
        if (!copyOp || !copyOp->hasOneUse()) {
            return failWithDiag(
                op, "Load op on L1 should be used by copy operations only.", user, "unexpected user op");
        }
        if (elemSize != 2 &&
            !llvm::all_of(copyOp->getUsers(), [](auto* useCopy) { return isa<asctile::TransposeOp>(useCopy); })) {
            return failWithDiag(
                op, "All copy operations should be used by transpose op only.", copyOp, "op without transpose op user");
        }
    }

    op->setAttr(fusibleAttr, UnitAttr::get(op.getContext()));
    return WalkResult::advance();
}

WalkResult markTransposeLoad(asctile::TransposeOp op)
{
    auto tileLoc = op.getType().getLoc();
    if (tileLoc != TensorLocation::L1)
        return WalkResult::advance();

    auto loadOp = op.getOperand().getDefiningOp<asctile::LoadOp>();
    if (!loadOp) {
        return failWithDiag(
            op, "Transpose on L1 is supported for load operation results only.", op.getOperand().getDefiningOp(),
            "unexpected source operation");
    }
    if (loadOp.getType().getShape().size() != 2) {
        return failWithDiag(op, "Transpose op on Cube supports 2-dim tensors only.");
    }
    if (!loadOp->hasOneUse()) {
        // TODO: Add support for loadOp with several transposeOp users
        return failWithDiag(
            op, "Transpose should be the only user of load to L1 operation.", loadOp, "load op with many uses");
    }
    std::optional<TensorLocation> copiesLoc;
    for (auto* user : op->getUsers()) {
        auto copyOp = dyn_cast<asctile::CopyOp>(user);
        if (!copyOp) {
            return failWithDiag(
                op, "Transpose op on L1 should be used by copy operations only.", user, "unexpected user op");
        }
        auto copyLoc = copyOp.getType().getLoc();
        if (!isL0Location(copyLoc)) {
            return failWithDiag(
                op, "Transpose op on L1 should be used by copy to L0A/L0B operations only.", copyOp,
                "unsupported copy op");
        }
        if (copiesLoc.has_value()) {
            if (copyLoc != copiesLoc.value()) {
                return failWithDiag(
                    op, "All users of transpose op on L1 should have same location.", copyOp,
                    "copy op with different location");
            }
        } else {
            copiesLoc = copyLoc;
        }
    }

    op->setAttr(fusibleAttr, UnitAttr::get(op.getContext()));
    return WalkResult::advance();
}

struct CubeTransposeToLoad : OpRewritePattern<asctile::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::TransposeOp op, PatternRewriter& rewriter) const override
    {
        if (!op->hasAttrOfType<UnitAttr>(fusibleAttr))
            return failure();
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        Operation *oldOp, *newOp;
        const char* attr{};
        if (isL0Location(tileLoc)) {
            auto copyOp = op.getOperand().getDefiningOp<asctile::CopyOp>();
            attr = getTransposeAttrName(tileLoc, /*isL1=*/false);
            setUnitAttr(rewriter, copyOp.getBase().getDefiningOp(), attr);
            Type newType = getTransposedType(copyOp.getType());
            newOp = rewriter.create<asctile::CopyOp>(
                op.getLoc(), newType, copyOp.getBase(), copyOp.getOffsets(), copyOp.getDistribAttr());
            oldOp = copyOp;
        } else if (tileLoc == TensorLocation::L1) {
            auto loadOp = op.getOperand().getDefiningOp<asctile::LoadOp>();
            if (!op->use_empty()) {
                auto copyOp = cast<asctile::CopyOp>(*op->user_begin());
                attr = getTransposeAttrName(copyOp.getType().getLoc(), /*isL1=*/true);
                for (auto* user : op->getUsers())
                    setUnitAttr(rewriter, user, attr);
            }
            Type newType = getTransposedType(loadOp.getType());
            newOp = rewriter.create<asctile::LoadOp>(
                op.getLoc(), newType, loadOp.getBase(), loadOp.getOffsets(), loadOp.getPadValue(),
                loadOp.getRealShape());
            oldOp = loadOp;
        }
        newOp->setAttrs(oldOp->getAttrs());
        if (attr) {
            newOp->setAttr(attr, rewriter.getUnitAttr());
        }
        rewriter.replaceOp(oldOp, newOp);
        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct CubeTransposeToLoadPass : public asctile::impl::CubeTransposeToLoadBase<CubeTransposeToLoadPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp->walk(markTransposeCopy).wasInterrupted() || funcOp->walk(markTransposeLoad).wasInterrupted()) {
            signalPassFailure();
            return;
        }
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<CubeTransposeToLoad>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
        funcOp->walk([this](asctile::TransposeOp op) {
            if (isL0Location(op.getType().getLoc())) {
                op.emitOpError() << "not supported on cube";
                signalPassFailure();
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createCubeTransposeToLoadPass()
{
    return std::make_unique<CubeTransposeToLoadPass>();
}
