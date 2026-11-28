/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_UNROLLLOOP
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

// TODO: Use mlir::loopUnrollByFactor instead of customized implementation

Value ceilDivPositive(OpBuilder& builder, Location loc, Value dividend, int64_t divisor)
{
    assert(divisor > 0 && "expected positive divisor");
    assert(dividend.getType().isIntOrIndex() && "expected integer or index-typed value");

    Value divisorMinusOneCst =
        builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(dividend.getType(), divisor - 1));
    Value divisorCst = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(dividend.getType(), divisor));
    Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOneCst);
    return builder.create<arith::DivUIOp>(loc, sum, divisorCst);
}

Value ceilDivPositive(OpBuilder& builder, Location loc, Value dividend, Value divisor)
{
    assert(dividend.getType().isIntOrIndex() && "expected integer or index-typed value");
    Value cstOne = builder.create<arith::ConstantOp>(loc, builder.getOneAttr(dividend.getType()));
    Value divisorMinusOne = builder.create<arith::SubIOp>(loc, divisor, cstOne);
    Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOne);
    return builder.create<arith::DivUIOp>(loc, sum, divisor);
}

std::optional<int64_t> getConstantTripCount(scf::ForOp forOp)
{
    std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
    if (!lbCstOp.has_value() || !ubCstOp.has_value() || !stepCstOp.has_value())
        return {};
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst > 0 && "expected non-negative loop bounds and positive step");
    return llvm::divideCeilSigned(ubCst - lbCst, stepCst);
}

void generateUnrolledLoop(
    Block* loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn, ValueRange iterArgs, ValueRange yieldedValues)
{
    auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);
    if (!annotateFn)
        annotateFn = [](unsigned, Operation*, OpBuilder) {};
    Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);
    SmallVector<Value, 4> lastYielded(yieldedValues);
    for (unsigned i = 1; i < unrollFactor; i++) {
        IRMapping operandMap;
        operandMap.map(iterArgs, lastYielded);
        if (!forOpIV.use_empty()) {
            Value ivUnroll = ivRemapFn(i, forOpIV, builder);
            operandMap.map(forOpIV, ivUnroll);
        }
        for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
            Operation* clonedOp = builder.clone(*it, operandMap);
            annotateFn(i, clonedOp, builder);
        }
        for (unsigned i = 0, e = lastYielded.size(); i < e; i++)
            lastYielded[i] = operandMap.lookupOrDefault(yieldedValues[i]);
    }
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++)
        annotateFn(0, &*it, builder);
    loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

LogicalResult loopUnrollByFactor(
    scf::ForOp forOp, int64_t unrollFactor, function_ref<void(unsigned, Operation*, OpBuilder)> annotateFn)
{
    assert(unrollFactor > 0 && "expected positive unroll factor");
    if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
        return success();
    OpBuilder boundsBuilder(forOp);
    IRRewriter rewriter(forOp.getContext());
    auto loc = forOp.getLoc();
    Value step = forOp.getStep();
    Value upperBoundUnrolled;
    Value stepUnrolled;
    bool generateEpilogueLoop = true;
    std::optional<int64_t> constTripCount = getConstantTripCount(forOp);
    if (constTripCount) {
        int64_t lbCst = getConstantIntValue(forOp.getLowerBound()).value();
        int64_t ubCst = getConstantIntValue(forOp.getUpperBound()).value();
        int64_t stepCst = getConstantIntValue(forOp.getStep()).value();
        if (unrollFactor == 1) {
            if (*constTripCount == 1 && failed(forOp.promoteIfSingleIteration(rewriter)))
                return failure();
            return success();
        }
        int64_t tripCountEvenMultiple = *constTripCount - (*constTripCount % unrollFactor);
        int64_t upperBoundUnrolledCst = lbCst + tripCountEvenMultiple * stepCst;
        int64_t stepUnrolledCst = stepCst * unrollFactor;
        generateEpilogueLoop = upperBoundUnrolledCst < ubCst;
        if (generateEpilogueLoop)
            upperBoundUnrolled = boundsBuilder.create<arith::ConstantOp>(
                loc, boundsBuilder.getIntegerAttr(forOp.getUpperBound().getType(), upperBoundUnrolledCst));
        else
            upperBoundUnrolled = forOp.getUpperBound();
        stepUnrolled = stepCst == stepUnrolledCst ?
                           step :
                           boundsBuilder.create<arith::ConstantOp>(
                               loc, boundsBuilder.getIntegerAttr(step.getType(), stepUnrolledCst));
    } else {
        auto lowerBound = forOp.getLowerBound();
        auto upperBound = forOp.getUpperBound();
        Value diff = boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
        Value tripCount = ceilDivPositive(boundsBuilder, loc, diff, step);
        Value unrollFactorCst = boundsBuilder.create<arith::ConstantOp>(
            loc, boundsBuilder.getIntegerAttr(tripCount.getType(), unrollFactor));
        Value tripCountRem = boundsBuilder.create<arith::RemSIOp>(loc, tripCount, unrollFactorCst);
        Value tripCountEvenMultiple = boundsBuilder.create<arith::SubIOp>(loc, tripCount, tripCountRem);
        upperBoundUnrolled = boundsBuilder.create<arith::AddIOp>(
            loc, lowerBound, boundsBuilder.create<arith::MulIOp>(loc, tripCountEvenMultiple, step));
        stepUnrolled = boundsBuilder.create<arith::MulIOp>(loc, step, unrollFactorCst);
    }
    if (generateEpilogueLoop) {
        OpBuilder epilogueBuilder(forOp->getContext());
        epilogueBuilder.setInsertionPointAfter(forOp);
        auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
        epilogueForOp.setLowerBound(upperBoundUnrolled);
        auto results = forOp.getResults();
        auto epilogueResults = epilogueForOp.getResults();
        for (auto e : llvm::zip(results, epilogueResults)) {
            std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
        }
        epilogueForOp->setOperands(epilogueForOp.getNumControlOperands(), epilogueForOp.getInitArgs().size(), results);
        (void)epilogueForOp.promoteIfSingleIteration(rewriter);
    }
    forOp.setUpperBound(upperBoundUnrolled);
    forOp.setStep(stepUnrolled);
    auto iterArgs = ValueRange(forOp.getRegionIterArgs());
    auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();
    generateUnrolledLoop(
        forOp.getBody(), forOp.getInductionVar(), unrollFactor,
        [&](unsigned i, Value iv, OpBuilder b) {
            auto stride =
                b.create<arith::MulIOp>(loc, step, b.create<arith::ConstantOp>(loc, b.getIntegerAttr(iv.getType(), i)));
            return b.create<arith::AddIOp>(loc, iv, stride);
        },
        annotateFn, iterArgs, yieldedValues);
    (void)forOp.promoteIfSingleIteration(rewriter);
    return success();
}

class Annotator {
    bool enable;
    int64_t loopId = 0;

    void setI64Attr(StringRef name, int64_t value, Operation* op, Builder builder) const
    {
        if (enable)
            op->setAttr(name, builder.getI64IntegerAttr(value));
    }

public:
    explicit Annotator(bool enable) : enable(enable) {}
    ~Annotator() = default;

    void applyLoopId(Operation* op) { setI64Attr(attr::unrolledLoop, loopId++, op, Builder(op)); }

    void operator()(unsigned iter, Operation* op, Builder builder) const
    {
        setI64Attr(attr::unrollIter, static_cast<int64_t>(iter), op, builder);
    }
};

int64_t getUnrollFactor(scf::ForOp loop)
{
    if (auto a = loop->getAttrOfType<IntegerAttr>(attr::unrollFactor))
        return std::max(1L, a.getValue().getSExtValue());
    return 1L;
}

void wrapLoopBeforeUnroll(scf::ForOp loop)
{
    int64_t unrollFactor = getUnrollFactor(loop);
    if (unrollFactor <= 1)
        return;
    OpBuilder builder(loop);
    auto exec = builder.create<scf::ExecuteRegionOp>(loop.getLoc(), loop.getResultTypes());
    exec->setAttr(attr::unrollFactor, builder.getI64IntegerAttr(unrollFactor));
    auto* body = &exec.getRegion().emplaceBlock();
    loop->moveBefore(body, body->end());
    loop->replaceAllUsesWith(exec.getResults());
    builder.setInsertionPointToEnd(body);
    builder.create<scf::YieldOp>(loop.getLoc(), loop.getResults());
}

struct UnrollLoopPass : public asctile::impl::UnrollLoopBase<UnrollLoopPass> {
    UnrollLoopPass(const UnrollLoopOptions& options) : UnrollLoopBase(options) {}

    void runOnOperation() override
    {
        Annotator annotator(annotate);
        auto op = getOperation();
        op.walk(wrapLoopBeforeUnroll);
        op.walk([&](scf::ForOp loop) {
            int64_t unrollFactor = getUnrollFactor(loop);
            loop->removeAttr(attr::unrollFactor);
            if (unrollFactor <= 1)
                return;
            Builder builder(loop);
            for (auto& op : loop.getBody()->without_terminator())
                annotator(unrollFactor, &op, builder);
            auto result = loopUnrollByFactor(loop, unrollFactor, annotator);
            if (failed(result))
                signalPassFailure();
        });
        op.walk([&](scf::ExecuteRegionOp exec) { annotator.applyLoopId(exec); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createUnrollLoopPass(bool annotate)
{
    UnrollLoopOptions options;
    options.annotate = annotate;
    return std::make_unique<UnrollLoopPass>(options);
}
