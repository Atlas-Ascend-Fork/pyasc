/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===----------------------------------------------------------------------===//
// ReuseTensorAllocation Pass
//===----------------------------------------------------------------------===//
//
// Reduces on-chip memory by reusing freed tensor allocations. Covers VECCALC (UB),
// A1 (L1), A2 (L0A), B2 (L0B), and CO1 (L0C) positions. Reuse is restricted to
// tensors of the same position, since each position maps to a distinct hardware buffer.
//
// Reuse decision:
// 1. Input/output tensor restriction: if both tensors are I/O and both have unroll_iter
//    attributes with different values, they cannot be reused (each iteration loads/stores
//    different data). If either tensor doesn't have unroll_iter, no restriction applies.
// 2. Lifetime overlap: top's endLife must precede bottom's beginLife in program order.
//    Uses opPrecedes with DominanceInfo to correctly handle operations in different blocks.
// 3. Same-op reuse: when top's endLife and bottom's beginLife are the same compute op,
//    hardware reads all inputs before writing the output — safe to share one allocation.
//
// User collection traverses CastOpInterface and LocalTensorSubIndexOp chains to find
// indirect users (ops that use a view of the allocation). View-producing ops themselves
// are skipped during read/write classification.
//
// Loop-carried values (scf.yield in ForOp/WhileOp, scf.condition in WhileOp) are excluded
// from the reuse list — their data persists across all iterations.
//
//===----------------------------------------------------------------------===//

#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscendC/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Utils/Attributes.h"
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "ascendc-reuse-tensor-allocation"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REUSETENSORALLOCATION
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const eraseMeAttr = "asc.erase_me";

using TensorOp = ascendc::LocalTensorAutoOp;
using Color = int64_t;
using ColorMap = std::unordered_map<TensorOp, Color, PointerLikeTypeHash<TensorOp>>;
using ColorSet = std::unordered_set<Color>;
using TensorSet = std::unordered_set<TensorOp, PointerLikeTypeHash<TensorOp>>;

class LifetimeInfo {
    TensorOp tensor;
    OpOperand* firstUse = nullptr;
    OpOperand* lastUse = nullptr;
    Operation* beginLifeOp = nullptr;
    Operation* endLifeOp = nullptr;
    DominanceInfo* di;

    // Examples:
    // common uses:
    // {
    //  write(a) <- first use       ┐
    //  ...                         | lifetime
    //  read(a) <- last use         ┘
    // }
    //
    // for-op yield value uses:
    // b = for(...) {             ┐
    //   ...                      |
    //   write(a) <- first use    |
    //   ...                      |
    //   read(a)                  | lifetime (because recursive effect memory)
    //   ...                      |
    //   yield a                  |
    // }                          |
    // ...                        |
    // read(b) <- last use        ┘
    OpOperand* findFirstUse(ArrayRef<OpOperand*> uses) const
    {
        OpOperand* operand = uses.front();
        if (auto forOp = dyn_cast<scf::ForOp>(operand->getOwner())) {
            // A case when a first use of a tensor is setting of yielded value of a for-loop.
            // In this case the tensor leaves from this setting operation till the end of the loop
            // and it's first use was selected as init value for corresponding iteration argument
            // of the loop.
            for (auto* use : uses) {
                Operation* user = use->getOwner();
                auto parentForOp = user->getParentOfType<scf::ForOp>();
                if (parentForOp == forOp && isWriteToAllocation(user, tensor))
                    return use;
            }
        }
        return operand;
    }

    // Examples:
    // simple lifetime:
    // {
    //  write(a) <- first use       ┐
    // ...                          | lifetime
    //  read(a) <- last use         ┘
    // }
    // nested recursive lifetime:
    // write(a) <- first use      ┐
    // for(...) {                 |
    //   read(a) <- last use      | lifetime (because recursive effect memory)
    //   ...                      |
    //   yield                    ┘
    // }
    Operation* findEndLifeOp() const
    {
        if (!getBeginLifeOp() || !lastUse)
            return nullptr;

        auto* lastOp = lastUse->getOwner();
        Block* commonBlock = di->findNearestCommonDominator(getBeginLifeOp()->getBlock(), lastOp->getBlock());

        Operation* lastUseParentOp;
        if (auto blockArg = llvm::dyn_cast<BlockArgument>(lastUse->get()))
            lastUseParentOp = blockArg.getOwner()->getParentOp();

        auto process = [&](Operation* op) {
            if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                if (lastUseParentOp != forOp)
                    lastOp = forOp.getBody()->getTerminator();
            } else if (auto whileOp = dyn_cast<scf::WhileOp>(op))
                lastOp = whileOp.getBody()->getTerminator();
        };

        auto* op = lastOp;
        process(op);
        while (op->getBlock() != commonBlock) {
            op = op->getParentOp();
            process(op);
        }
        return lastOp;
    }

    // Special case when current tensor is last used as initial value for a for-loop op
    // and other tensor is used as yielded value in the same for-loop op, i.e. it is a
    // setting value for the loop's iteration argument after first iteration.
    // In this case init value buffer may be reused for yielded value.
    bool isLastUseInitValueFor(const LifetimeInfo& lti) const
    {
        auto blockArg = llvm::dyn_cast<BlockArgument>(lastUse->get());
        bool noUseAfterTheLoop = lastUse->getOwner() == getEndLifeOp();
        bool isOtherTensorYielded = lti.firstUse->getOwner() != lti.getBeginLifeOp();
        return blockArg && noUseAfterTheLoop && isOtherTensorYielded &&
               blockArg.getOwner()->getParentOp() == lti.getBeginLifeOp();
    }

public:
    LifetimeInfo(TensorOp tensorOp, DominanceInfo& dinfo) : tensor(tensorOp), di(&dinfo)
    {
        SmallVector<OpOperand*> uses = collectAllUses(tensor);
        llvm::stable_sort(uses, [&](OpOperand* lhso, OpOperand* rhso) {
            return ascendc::opPrecedes(lhso->getOwner(), rhso->getOwner(), *di);
        });
        llvm::erase_if(uses, [](OpOperand* opnd) { return opnd->getOwner()->getParentOfType<ascvf::VFGroupOp>(); });
        beginLifeOp = uses.front()->getOwner();
        firstUse = findFirstUse(uses);
        lastUse = uses.back();
        endLifeOp = findEndLifeOp();
    }

    TensorOp getTensorOp() const { return tensor; }
    Operation* getBeginLifeOp() const { return beginLifeOp; }
    Operation* getEndLifeOp() const { return endLifeOp; }

    bool comesAfter(const LifetimeInfo& lti) const
    {
        return ascendc::opPrecedes(lti.getBeginLifeOp(), getBeginLifeOp(), *di);
    }

    bool diesAfter(const LifetimeInfo& lti) const
    {
        return ascendc::opPrecedes(lti.getEndLifeOp(), getEndLifeOp(), *di);
    }

    bool livesAfter(const LifetimeInfo& lti) const
    {
        if (lti.isLastUseInitValueFor(*this))
            return ascendc::opPrecedes(lti.lastUse->getOwner(), firstUse->getOwner(), *di);
        return ascendc::opPrecedes(lti.getEndLifeOp(), getBeginLifeOp(), *di);
    }

    Operation* getEndBorderOp(const LifetimeInfo& lti) const
    {
        if (isLastUseInitValueFor(lti))
            return lastUse->getOwner() == lti.firstUse->getOwner() ? lastUse->getOwner() : nullptr;
        return getEndLifeOp() == lti.getBeginLifeOp() ? getEndLifeOp() : nullptr;
    }

    bool intersect(const LifetimeInfo& lti) const
    {
        return !getEndBorderOp(lti) && !lti.getEndBorderOp(*this) && !livesAfter(lti) && !lti.livesAfter(*this);
    }

    void dump() const
    {
        llvm::dbgs() << "tensor: ";
        getTensorOp()->dump();
        llvm::dbgs() << "beginLife: ";
        if (getBeginLifeOp())
            getBeginLifeOp()->dump();
        else
            llvm::dbgs() << "-";
        llvm::dbgs() << "firstUse: ";
        if (firstUse && firstUse->getOwner())
            firstUse->getOwner()->dump();
        else
            llvm::dbgs() << "-";
        llvm::dbgs() << "lastUse: ";
        if (lastUse && lastUse->getOwner())
            lastUse->getOwner()->dump();
        else
            llvm::dbgs() << "-";
        llvm::dbgs() << "endLife: ";
        if (getEndLifeOp())
            getEndLifeOp()->dump();
        else
            llvm::dbgs() << "-";
    }
};

using TensorLifeTimes = SmallVector<LifetimeInfo>;

// Returns the actual allocation size matching AllocateTensor's sizing rules:
// cube-block-aligned for A1/A2/B2, raw byte size for VECCALC/CO1.
int64_t getAllocationSize(TensorOp op)
{
    auto shapedType = cast<ShapedType>(op.getType());
    auto pos = op.getPosition();
    if (pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::B1 || pos == ascendc::TPosition::A2 ||
        pos == ascendc::TPosition::B2 || pos == ascendc::TPosition::CO1)
        return ascendc::getTypeSizeCubeBlockAlign(shapedType, pos);
    return ascendc::getTypeSize(op.getType());
}

bool isTensorGreaterOrEqual(TensorOp lhs, TensorOp rhs) { return getAllocationSize(lhs) >= getAllocationSize(rhs); }

bool isReusablePosition(ascendc::TPosition pos)
{
    // VECIN, VECOUT also reusable but for the same unroll factor
    return pos == ascendc::TPosition::VECCALC || pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::B1 ||
           pos == ascendc::TPosition::C1 || pos == ascendc::TPosition::A2 || pos == ascendc::TPosition::B2 ||
           pos == ascendc::TPosition::CO1;
}

bool isReusable(TensorOp op) { return isReusablePosition(op.getPosition()) && op.getType().hasStaticShape(); }

TensorSet collectCrossGroupTensors(func::FuncOp funcOp)
{
    TensorSet crossGroup;
    funcOp.walk([&](ascendc::DataCopyOp copyOp) {
        auto direction = copyOp.getDirection();
        if (!direction)
            return;
        auto [src, dst] = *direction;
        if (src == ascendc::TPosition::VECCALC && (dst == ascendc::TPosition::A1 || dst == ascendc::TPosition::B1))
            if (auto root = ascendc::getAllocationRoot(copyOp.getSrc()))
                crossGroup.insert(root);
        if (src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::VECCALC)
            if (auto root = ascendc::getAllocationRoot(copyOp.getDst()))
                crossGroup.insert(root);
    });
    return crossGroup;
}

std::optional<int64_t> getReuseGroup(Operation* op)
{
    if (auto iter = op->getAttrOfType<IntegerAttr>(ascendc::attr::reuseGroup))
        return iter.getValue().getSExtValue();
    return std::nullopt;
}

void markForErase(TensorOp op) { op->setAttr(eraseMeAttr, UnitAttr::get(op.getContext())); }

bool hasShareLoopAncestor(Operation* lhs, Operation* rhs)
{
    constexpr int nestedLevel = 2;
    std::unordered_set<Operation*> lhsLoops, rhsLoops;
    std::vector<Operation*> results;
    scf::ForOp forOp = lhs->getParentOfType<scf::ForOp>();
    for (int i = nestedLevel; i > 0 && forOp; --i) {
        lhsLoops.insert(forOp);
        forOp = forOp->getParentOfType<scf::ForOp>();
    }
    forOp = rhs->getParentOfType<scf::ForOp>();
    for (int i = nestedLevel; i > 0 && forOp; --i) {
        rhsLoops.insert(forOp);
        forOp = forOp->getParentOfType<scf::ForOp>();
    }
    std::set_intersection(
        lhsLoops.begin(), lhsLoops.end(), rhsLoops.begin(), rhsLoops.end(), std::back_inserter(results));
    return !results.empty();
}

bool isReadFromAllocation(Operation* op, TensorOp root)
{
    if (auto srcOp = dyn_cast<ascendc::OpWithSrc>(op))
        for (Value src : srcOp.getSrcTensors()) {
            if (auto blockArg = llvm::dyn_cast<BlockArgument>(src))
                if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()))
                    src = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
            auto srcRoot = ascendc::getAllocationRoot(src);
            if (srcRoot && srcRoot == root)
                return true;
        }
    return false;
}

bool canReuse(const LifetimeInfo& topTensorLTI, const LifetimeInfo& bottomTensorLTI)
{
    LLVM_DEBUG(llvm::dbgs() << "Can reuse?: " << bottomTensorLTI.getTensorOp() << "\n");
    const auto* topInfo = &topTensorLTI;
    const auto* bottomInfo = &bottomTensorLTI;
    if (topTensorLTI.comesAfter(bottomTensorLTI) ||
        !bottomTensorLTI.comesAfter(topTensorLTI) && topTensorLTI.diesAfter(bottomTensorLTI)) {
        std::swap(topInfo, bottomInfo);
    }
    TensorOp top = topInfo->getTensorOp();
    TensorOp bottom = bottomInfo->getTensorOp();
    bool isTopInOut = top.getInput() || top.getOutput();
    auto iter1 = getReuseGroup(top);
    bool isBottomInOut = bottom.getInput() || bottom.getOutput();
    auto iter2 = getReuseGroup(bottom);
    if (hasShareLoopAncestor(topInfo->getBeginLifeOp(), bottomInfo->getBeginLifeOp()) &&
        (iter1 && iter2 && iter1 != iter2 && (isTopInOut || isBottomInOut))) {
        LLVM_DEBUG(llvm::dbgs() << "reject: in or out" << "\n");
        return false;
    }
    if (auto* borderOp = topInfo->getEndBorderOp(*bottomInfo)) {
        if (top.getType().getElementType() != bottom.getType().getElementType()) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: same-op element type mismatch\n");
            return false;
        }
        // Only allow same-op reuse for operations marked with OpWithReusableSrc.
        // Operations like BroadcastOp that read sources multiple times are not marked
        // and will be rejected here.
        auto reuse = dyn_cast<ascendc::OpWithReusableSrc>(borderOp);
        if (!reuse) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: operation not marked with OpWithReusableSrcInterface\n");
            return false;
        }
        SmallVector<OpOperand*> topOpnds, bottomOpnds;
        for (OpOperand& operand : borderOp->getOpOperands()) {
            auto root = ascendc::getAllocationRoot(operand.get());
            if (root == top)
                topOpnds.push_back(&operand);
            else if (root == bottom)
                bottomOpnds.push_back(&operand);
        }
        for (auto* topOpnd : topOpnds) {
            for (auto* bottomOpnd : bottomOpnds) {
                if (!reuse.canReuse(*topOpnd, *bottomOpnd)) {
                    LLVM_DEBUG(llvm::dbgs() << "  reject: cannot reuse inside op\n");
                    return false;
                }
            }
        }
    }
    // Strict intersection without intersect borders
    if (topInfo->intersect(*bottomInfo)) {
        LLVM_DEBUG(llvm::dbgs() << "  reject: strict lifetime overlap\n");
        return false;
    }
    return true;
}

auto getConflictedTensors(const LifetimeInfo& tensorLTI, ArrayRef<LifetimeInfo>& tensorsLifeTimes)
{
    SmallVector<TensorOp> conflicted;
    TensorOp tensor = tensorLTI.getTensorOp();
    for (const auto& otherTensorLTI : tensorsLifeTimes) {
        TensorOp otherTensor = otherTensorLTI.getTensorOp();
        if (tensor != otherTensor && !canReuse(tensorLTI, otherTensorLTI))
            conflicted.emplace_back(otherTensor);
    }
    return conflicted;
};

auto getColorsSet(const ColorMap& colors, ArrayRef<TensorOp> allTensors)
{
    ColorSet neighbors;
    for (TensorOp tensor : allTensors) {
        if (auto it = colors.find(tensor); it != colors.end())
            neighbors.insert(it->second);
    }
    return neighbors;
}

Color getFirstFree(const ColorSet& colors)
{
    int freeColor = 0;
    while (colors.count(freeColor)) {
        ++freeColor;
    }
    return freeColor;
}

void processTensorList(ArrayRef<LifetimeInfo> tensorsLifeTimes, DominanceInfo& di)
{
    ColorMap colors;
    LLVM_DEBUG(llvm::dbgs() << "Processing tensor list with " << tensorsLifeTimes.size() << " tensors\n");
    for (const auto& tensorLTI : tensorsLifeTimes) {
        TensorOp topTensor = tensorLTI.getTensorOp();
        LLVM_DEBUG(llvm::dbgs() << "Trying to color: " << topTensor << "\n");
        Color freeColor = getFirstFree(getColorsSet(colors, getConflictedTensors(tensorLTI, tensorsLifeTimes)));
        LLVM_DEBUG(llvm::dbgs() << "Set Color: " << freeColor << "\n");
        colors.insert({topTensor, freeColor});
    }

    SmallVector<std::pair<TensorOp, Color>> colorVec;
    int64_t colorSize = -1;
    for (auto pa : colors) {
        colorVec.push_back(pa);
        colorSize = std::max(colorSize, pa.second + 1);
    }

    llvm::sort(colorVec, [&](const std::pair<TensorOp, Color>& x, const std::pair<TensorOp, Color>& y) {
        return ascendc::opPrecedes(x.first, y.first, di);
    });

    SmallVector<SmallVector<TensorOp>> tensorGroups; // for stability
    tensorGroups.resize(colorSize);
    for (auto& [tensor, color] : colorVec) {
        tensorGroups[color].push_back(tensor);
    }

    for (int color = 0; color < colorSize; ++color) {
        auto& tensors = tensorGroups[color];
        TensorOp bigger = tensors[0];
        Block* block = bigger->getBlock();
        for (auto tensor : tensors) {
            block = di.findNearestCommonDominator(block, tensor->getBlock());
            if (isTensorGreaterOrEqual(tensor, bigger)) {
                bigger = tensor;
            }
        }
        bigger->moveBefore(block, block->begin());
        for (auto reused : tensors) {
            if (reused == bigger)
                continue;
            OpBuilder builder(reused);
            builder.setInsertionPointAfter(reused);
            auto castOp =
                builder.create<ascendc::LocalTensorReinterpretCastOp>(reused->getLoc(), reused.getType(), bigger);
            reused->replaceAllUsesWith(castOp);
            markForErase(reused);
        }
    }
}

struct ReuseTensorAllocationPass : public ascendc::impl::ReuseTensorAllocationBase<ReuseTensorAllocationPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        DominanceInfo di(funcOp);

        std::map<ascendc::TPosition, TensorLifeTimes> tensorsByPos;
        TensorSet crossGroup = collectCrossGroupTensors(funcOp);
        funcOp.walk<WalkOrder::PreOrder>([&](TensorOp tensor) {
            if (crossGroup.count(tensor)) {
                LLVM_DEBUG(llvm::dbgs() << "tensor: ");
                LLVM_DEBUG(tensor->dump());
                LLVM_DEBUG(llvm::dbgs() << "  reject: cross-group transfer tensor\n");
                return;
            }
            if (!tensor->getUsers().empty() && isReusable(tensor))
                tensorsByPos[tensor.getPosition()].emplace_back(tensor, di);
        });

        for (auto& [position, tensorLifeTimes] : tensorsByPos) {
            // Tensors are sorted in bigger first order to exclude situations like one of two non-intersecting big
            // tensors are colored differently when only one of them intersects with small tensor and small tensor
            // is colored before both big ones.
            // Second sorting criteria is not significant and is used only for stable sorting.
            llvm::stable_sort(tensorLifeTimes, [&](const LifetimeInfo& lhs, const LifetimeInfo& rhs) {
                auto lhsSize = getAllocationSize(lhs.getTensorOp());
                auto rhsSize = getAllocationSize(rhs.getTensorOp());
                if (lhsSize != rhsSize)
                    return lhsSize > rhsSize;
                return lhs.diesAfter(rhs);
            });

            LLVM_DEBUG(llvm::dbgs() << "Lifetimes:\n");
            for (auto& lt : tensorLifeTimes) {
                LLVM_DEBUG(lt.dump());
            }

            processTensorList(tensorLifeTimes, di);
        }
        funcOp.walk([](TensorOp op) {
            if (op->hasAttr(eraseMeAttr))
                op.erase();
        });
        funcOp.walk([](Operation* op) { op->removeAttr(ascendc::attr::reuseGroup); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createReuseTensorAllocationPass()
{
    return std::make_unique<ReuseTensorAllocationPass>();
}
