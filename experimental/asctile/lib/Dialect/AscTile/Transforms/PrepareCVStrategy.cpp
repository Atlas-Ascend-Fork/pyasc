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
#include "asctile/Dialect/AscTile/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PREPARECVSTRATEGY
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

class CVStrategyModel {
    enum struct CopyVerdict { Accept, Fail, Skip };

    enum struct UserKind : bool { Split, Unsplit };

    using SplitShape = SmallVector<int64_t, 2>;

    struct SplitState {
        Value value;
        SplitShape shape;
        unsigned axis;
        UserKind kind = UserKind::Split;
    };

    struct SplitOutput {
        SplitShape shape;
        unsigned axis = 0;
        bool propagate = true;
    };

    struct LoopCarriedEdge {
        scf::ForOp loop;
        BlockArgument iterArg;
        unsigned index;
        SplitShape shape;
        unsigned axis;
    };

    CVStrategyOp root;
    DistribMode split;
    SmallVector<SplitState, 4> worklist;
    DenseMap<Operation*, SplitShape> annotations;
    DenseMap<Operation*, SmallVector<SplitShape, 4>> operandStates;
    SmallVector<LoopCarriedEdge, 2> loopCarriedEdges;

    CopyVerdict getResultShape(CopyOp op, SmallVectorImpl<int64_t>& shape)
    {
        if (op.getDistrib()) {
            op.emitOpError() << "'distrib' argument must be omitted inside the CV strategy";
            return CopyVerdict::Fail;
        }
        auto srcType = op.getBase().getType();
        auto dstType = op.getType();
        if (srcType.getLoc() != TensorLocation::L0C || dstType.getLoc() != TensorLocation::UB)
            return CopyVerdict::Skip;
        assert(srcType.getRank() == dstType.getRank());
        if (srcType.getRank() != 1 && srcType.getRank() != 2) {
            op.emitOpError() << "requires 1D or 2D tensors, got " << srcType.getRank() << "D";
            return CopyVerdict::Fail;
        }
        auto srcShape = srcType.getShape();
        auto dstShape = dstType.getShape();
        if (srcShape != dstShape) {
            op.emitOpError() << "requires the full source shape " << srcShape << ", got " << dstShape;
            return CopyVerdict::Fail;
        }
        shape.append(dstShape.begin(), dstShape.end());
        auto axis = getSplitAxis(split, shape.size());
        auto divisor = split == DistribMode::SplitByM ? 2 : 32;
        if (shape[axis] % divisor != 0) {
            op.emitOpError() << "requires the active split dimension divisible by " << divisor << ", got "
                             << shape[axis];
            return CopyVerdict::Fail;
        }
        shape[axis] /= 2;
        return CopyVerdict::Accept;
    }

    LogicalResult getTensorShape(Operation* op, Value value, SmallVectorImpl<int64_t>& shape)
    {
        auto type = dyn_cast<LocalTensorType>(value.getType());
        if (!type || (type.getRank() != 1 && type.getRank() != 2))
            return op->emitOpError() << "requires 1D or 2D local tensors, got " << value.getType();
        shape.append(type.getShape().begin(), type.getShape().end());
        return success();
    }

    LogicalResult planAnnotation(ArrayRef<int64_t> shape, Operation* op)
    {
        if (auto existingSplit = op->getAttrOfType<DistribModeAttr>(attr::needSplit);
            existingSplit && existingSplit.getValue() != split)
            return op->emitOpError("has conflicting CV split mode requests");
        if (auto existingShape = op->getAttrOfType<DenseI64ArrayAttr>(attr::splitShape);
            existingShape && existingShape.asArrayRef() != shape)
            return op->emitOpError("has conflicting CV split shape requests");
        auto [it, inserted] = annotations.try_emplace(op, shape.begin(), shape.end());
        if (!inserted && it->second != shape)
            return op->emitOpError("has conflicting CV split requests");
        return success();
    }

    LogicalResult recordOperandState(const SplitState& state, Operation* user)
    {
        auto& shapes = operandStates[user];
        if (shapes.empty())
            shapes.resize(user->getNumOperands());
        for (auto [index, operand] : llvm::enumerate(user->getOperands())) {
            if (operand != state.value)
                continue;
            if (!shapes[index].empty() && shapes[index] != state.shape)
                return user->emitOpError() << "has conflicting split shapes for operand " << index;
            shapes[index] = state.shape;
        }
        return success();
    }

    LogicalResult getCopySplitShape(const SplitState& state, CopyOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape, resultShape;
        if (getTensorShape(op, state.value, inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        if (inputShape != resultShape)
            return op.emitOpError("must preserve the tensor shape");
        shape.append(state.shape.begin(), state.shape.end());
        return success();
    }

    LogicalResult getCopyJoinShape(const SplitState& state, CopyOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape, resultShape;
        if (getTensorShape(op, state.value, inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        if (resultShape.size() != state.shape.size())
            return op.emitOpError("must preserve the tensor rank");
        for (auto [index, size] : llvm::enumerate(resultShape)) {
            auto expectedSize = index == state.axis ? state.shape[index] * 2 : state.shape[index];
            if (size != expectedSize)
                return op.emitOpError("must reconstruct the full tensor shape");
        }
        shape.append(state.shape.begin(), state.shape.end());
        return success();
    }

    LogicalResult getElementwiseOutputShape(const SplitState& state, Operation* op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        if (getTensorShape(op, state.value, inputShape).failed())
            return failure();
        for (Value operand : op->getOperands()) {
            if (operand == state.value || !isa<LocalTensorType>(operand.getType()))
                continue;
            SplitShape operandShape;
            if (getTensorShape(op, operand, operandShape).failed() || operandShape != inputShape)
                return op->emitOpError("must use tensors with the same shape");
        }
        for (Value result : op->getResults()) {
            SplitShape resultShape;
            if (getTensorShape(op, result, resultShape).failed() || resultShape != inputShape)
                return op->emitOpError("must preserve the input tensor shape");
        }
        shape.append(state.shape.begin(), state.shape.end());
        return success();
    }

    LogicalResult getReduceOutput(const SplitState& state, ReduceOp op, SplitOutput& output)
    {
        SplitShape inputShape, resultShape;
        if (getTensorShape(op, op.getOperand(), inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        SmallVector<bool, 2> reducedDims(inputShape.size(), false);
        unsigned reducedCount = 0;
        for (Attribute attr : op.getDims()) {
            auto dimension = cast<IntegerAttr>(attr).getInt();
            if (dimension < 0)
                dimension += static_cast<int64_t>(inputShape.size());
            if (dimension < 0 || dimension >= static_cast<int64_t>(inputShape.size()))
                return op.emitOpError() << "has reduction dimension " << dimension << " out of range";
            if (reducedDims[dimension])
                return op.emitOpError() << "has duplicate reduction dimension " << dimension;
            reducedDims[dimension] = true;
            ++reducedCount;
            if (dimension == static_cast<int64_t>(state.axis))
                return op.emitOpError() << "cannot reduce the active split axis " << state.axis;
        }
        SplitShape expectedShape;
        unsigned outputAxis = state.axis;
        if (resultShape.size() == inputShape.size()) {
            for (auto [index, size] : llvm::enumerate(inputShape))
                expectedShape.push_back(reducedDims[index] ? 1 : size);
        } else if (resultShape.size() + reducedCount == inputShape.size()) {
            for (auto [index, size] : llvm::enumerate(inputShape)) {
                if (reducedDims[index]) {
                    if (index < state.axis)
                        --outputAxis;
                    continue;
                }
                expectedShape.push_back(size);
            }
        } else {
            return op.emitOpError("must use a canonical keep-dims or squeezed result shape");
        }
        if (resultShape != expectedShape)
            return op.emitOpError() << "must use the canonical reduction result shape " << expectedShape;
        output.shape.swap(resultShape);
        output.axis = outputAxis;
        output.shape[output.axis] = state.shape[state.axis];
        return success();
    }

    LogicalResult getReshapeOutput(const SplitState& state, ReshapeOp op, SplitOutput& output)
    {
        SplitShape inputShape, resultShape;
        if (getTensorShape(op, op.getIn(), inputShape).failed() ||
            getTensorShape(op, op.getOut(), resultShape).failed())
            return failure();
        unsigned outputAxis = state.axis;
        if (resultShape.size() == inputShape.size()) {
            if (resultShape[state.axis] != inputShape[state.axis])
                return op.emitOpError("must preserve the active split axis");
        } else if (inputShape.size() == 1 && resultShape.size() == 2) {
            outputAxis = getSplitAxis(split, resultShape.size());
            auto otherAxis = 1 - outputAxis;
            if (resultShape[otherAxis] != 1 || resultShape[outputAxis] != inputShape.front())
                return op.emitOpError("must only expand the non-active split axis");
        } else if (inputShape.size() == 2 && resultShape.size() == 1) {
            outputAxis = getSplitAxis(split, resultShape.size());
            auto otherAxis = 1 - state.axis;
            if (inputShape[otherAxis] != 1 || resultShape.front() != inputShape[state.axis])
                return op.emitOpError("must only squeeze the non-active split axis");
        } else {
            return op.emitOpError("must preserve the tensor rank or only expand/squeeze the non-active split axis");
        }
        output.shape.swap(resultShape);
        output.axis = outputAxis;
        output.shape[output.axis] = state.shape[state.axis];
        return success();
    }

    LogicalResult getBroadcastOutput(const SplitState& state, BroadcastOp op, SplitOutput& output)
    {
        SplitShape inputShape, resultShape;
        if (getTensorShape(op, op.getOperand(), inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        unsigned outputAxis = state.axis;
        if (inputShape.size() == resultShape.size()) {
            if (inputShape[outputAxis] != resultShape[outputAxis])
                return op.emitOpError("cannot broadcast the active split axis");
        } else if (inputShape.size() == 1 && resultShape.size() == 2) {
            outputAxis = 1;
            auto expectedAxis = getSplitAxis(split, resultShape.size());
            if (outputAxis != expectedAxis)
                return op.emitOpError() << "cannot map active split axis " << state.axis << " to result axis "
                                        << outputAxis;
            if (inputShape.front() != resultShape[outputAxis])
                return op.emitOpError("cannot broadcast the active split axis");
        } else {
            return op.emitOpError("must preserve the tensor rank or only prepend a non-active broadcast dimension");
        }
        output.shape.swap(resultShape);
        output.axis = outputAxis;
        output.shape[output.axis] = state.shape[state.axis];
        if (output.axis != getSplitAxis(split, output.shape.size())) {
            return op.emitOpError() << "would move the active split axis to unsupported result axis " << output.axis;
        }
        return success();
    }

    LogicalResult getOutput(const SplitState& state, Operation* op, SplitOutput& output)
    {
        if (isa<StoreOp>(op)) {
            output.shape = state.shape;
            output.axis = state.axis;
            output.propagate = false;
            return success();
        }
        if (auto copyOp = dyn_cast<CopyOp>(op)) {
            auto srcLoc = copyOp.getBase().getType().getLoc();
            auto dstLoc = copyOp.getType().getLoc();
            if (srcLoc == TensorLocation::L0C && dstLoc == TensorLocation::UB) {
                output.propagate = true;
                output.axis = state.axis;
                return getCopySplitShape(state, copyOp, output.shape);
            }
            if (srcLoc == TensorLocation::UB && dstLoc == TensorLocation::L1) {
                output.propagate = false;
                output.axis = state.axis;
                return getCopyJoinShape(state, copyOp, output.shape);
            }
            return op->emitOpError("only L0C->UB and UB->L1 copies are supported by CV strategy propagation");
        }
        if (isa<DumpTensorOp, InlineOp>(op)) {
            output.propagate = false;
            return success();
        }
        if (op->hasTrait<OpTrait::Elementwise>()) {
            output.axis = state.axis;
            return getElementwiseOutputShape(state, op, output.shape);
        }
        if (auto reduceOp = dyn_cast<ReduceOp>(op))
            return getReduceOutput(state, reduceOp, output);
        if (auto reshapeOp = dyn_cast<ReshapeOp>(op))
            return getReshapeOutput(state, reshapeOp, output);
        if (auto broadcastOp = dyn_cast<BroadcastOp>(op))
            return getBroadcastOutput(state, broadcastOp, output);
        return op->emitOpError("is not supported by CV strategy propagation");
    }

    LogicalResult propagateUser(const SplitState& state, Operation* user)
    {
        SplitOutput output;
        if (recordOperandState(state, user).failed())
            return failure();
        if (user->getNumRegions() != 0 || user->getNumSuccessors() != 0)
            return user->emitOpError("has regions or successors and is not supported by CV strategy propagation");
        if (getOutput(state, user, output).failed())
            return failure();
        if (annotations.contains(user)) {
            if (annotations[user] != output.shape) {
                return user->emitOpError() << "has conflicting CV split requests, the surrounding CV strategy is "
                                           << stringifyDistribMode(split) << " splitting with shape " << output.shape;
            }
        } else if (planAnnotation(output.shape, user).failed()) {
            return failure();
        }
        if (output.propagate) {
            for (Value result : user->getResults())
                worklist.push_back({result, output.shape, output.axis, state.kind});
        }
        return success();
    }

    LogicalResult propagateLoopYield(const SplitState& state, scf::YieldOp yieldOp)
    {
        auto loop = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!loop)
            return yieldOp.emitOpError("must be terminated by scf.yield to propagate a CV strategy result");
        ValueRange yieldedValues = loop.getYieldedValues();
        auto loopResults = loop.getLoopResults();
        if (!loopResults || yieldedValues.size() != loopResults->size())
            return yieldOp.emitOpError("has mismatched yielded values and loop results");
        if (planAnnotation({}, yieldOp).failed())
            return failure();
        for (auto [index, operand] : llvm::enumerate(yieldedValues)) {
            if (operand != state.value)
                continue;
            auto result = (*loopResults)[index];
            auto iterArg = loop.getTiedLoopRegionIterArg(result);
            if (!iterArg)
                return yieldOp.emitOpError() << "has no region argument for yielded value " << index;
            loopCarriedEdges.push_back({loop, iterArg, static_cast<unsigned>(index), state.shape, state.axis});
            worklist.push_back({result, state.shape, state.axis, UserKind::Unsplit});
        }
        return success();
    }

    LogicalResult propagateYield(const SplitState& state, YieldOp yieldOp, CVStrategyOp parent)
    {
        if (planAnnotation({}, yieldOp).failed())
            return failure();
        for (auto [result, operand] : llvm::zip_equal(parent.getResults(), yieldOp.getOperands())) {
            if (operand == state.value)
                worklist.push_back({result, state.shape, state.axis, UserKind::Unsplit});
        }
        return success();
    }

    LogicalResult validateLoopCarriedEdge(const LoopCarriedEdge& edge)
    {
        auto iterArgType = dyn_cast<LocalTensorType>(edge.iterArg.getType());
        if (!iterArgType || iterArgType.getLoc() != TensorLocation::UB)
            return edge.loop->emitOpError() << "requires a UB region argument for CV strategy result " << edge.index;
        if (iterArgType.getRank() != edge.shape.size())
            return edge.loop->emitOpError("has a rank-changing CV strategy loop-carried value");
        for (auto [index, size] : llvm::enumerate(iterArgType.getShape())) {
            auto expectedSize = index == edge.axis ? edge.shape[index] * 2 : edge.shape[index];
            if (size != expectedSize)
                return edge.loop->emitOpError()
                       << "has an incompatible shape for CV strategy loop-carried value " << edge.index;
        }
        for (auto* user : edge.iterArg.getUsers()) {
            if (user->getParentOfType<CVStrategyOp>() != root)
                return user->emitOpError("must use the loop-carried value only inside the corresponding CV strategy");
            if (!annotations.contains(user))
                return user->emitOpError("must participate in the corresponding CV strategy split");
        }
        return success();
    }

public:
    explicit CVStrategyModel(CVStrategyOp root) : root(root), split(root.getSplit()) {};
    ~CVStrategyModel() = default;

    LogicalResult ensure()
    {
        for (auto op : root.getOps<CopyOp>()) {
            SplitShape shape;
            auto verdict = getResultShape(op, shape);
            if (verdict == CopyVerdict::Fail) {
                return failure();
            }
            if (verdict == CopyVerdict::Skip)
                continue;
            assert(verdict == CopyVerdict::Accept);
            if (planAnnotation(shape, op).failed())
                return failure();
            worklist.push_back({op.getResult(), shape, getSplitAxis(split, shape.size())});
        }
        while (!worklist.empty()) {
            auto state = worklist.pop_back_val();
            for (auto* user : state.value.getUsers()) {
                if (state.kind == UserKind::Split) {
                    if (user->getParentOfType<CVStrategyOp>() != root)
                        continue;
                    if (auto yieldOp = dyn_cast<YieldOp>(user)) {
                        if (propagateYield(state, yieldOp, root).failed())
                            return failure();
                        continue;
                    }
                }
                if (state.kind == UserKind::Unsplit) {
                    if (auto parent = user->getParentOfType<CVStrategyOp>()) {
                        if (parent.getSplit() != split)
                            return user->emitError()
                                   << "tensor defined inside a CV strategy (" << stringifyDistribMode(split)
                                   << ") cannot be used inside another CV strategy with different 'split' argument ("
                                   << stringifyDistribMode(parent.getSplit()) << ")";
                        if (auto yieldOp = dyn_cast<YieldOp>(user)) {
                            if (propagateYield(state, yieldOp, parent).failed())
                                return failure();
                            continue;
                        }
                    }
                }
                if (auto yieldOp = dyn_cast<scf::YieldOp>(user); yieldOp && state.kind == UserKind::Unsplit) {
                    if (propagateLoopYield(state, yieldOp).failed())
                        return failure();
                    continue;
                }
                if (propagateUser(state, user).failed())
                    return failure();
            }
        }
        for (const auto& edge : loopCarriedEdges) {
            if (validateLoopCarriedEdge(edge).failed())
                return failure();
        }
        OpBuilder builder(root);
        auto needSplitAttr = builder.getAttr<DistribModeAttr>(split);
        for (auto& [op, shape] : annotations) {
            op->setAttr(attr::needSplit, needSplitAttr);
            op->setAttr(attr::splitShape, builder.getDenseI64ArrayAttr(shape));
        }
        return success();
    }
};

struct PrepareCVStrategyPass : public asctile::impl::PrepareCVStrategyBase<PrepareCVStrategyPass> {
    void runOnOperation() override
    {
        getOperation().walk([this](CVStrategyOp op) {
            CVStrategyModel model(op);
            if (model.ensure().failed())
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPrepareCVStrategyPass() { return std::make_unique<PrepareCVStrategyPass>(); }
