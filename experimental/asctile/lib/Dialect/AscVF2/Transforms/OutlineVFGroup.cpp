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
#include "asctile/Dialect/AscVF2/IR/AscVF2.h"
#include "asctile/Dialect/AscVF2/Transforms/Passes.h"
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace ascvf2 {
#define GEN_PASS_DEF_OUTLINEVFGROUP
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"
} // namespace ascvf2
} // namespace mlir

using namespace mlir;

namespace {

bool isCompatibleReductionSink(Operation* op, ArrayRef<Operation*> producers)
{
    auto reduction = ascvf2::getSupportedReductionInfo(op);
    if (!reduction)
        return false;

    Operation* producer = reduction->operand.getDefiningOp();
    return producer && llvm::is_contained(producers, producer);
}

bool isCompatibleReductionPrefix(Operation* op)
{
    if (!ascvf2::getSupportedReductionInfo(op))
        return false;
    return llvm::any_of(op->getResult(0).getUses(), [&](OpOperand& use) {
        auto broadcastOp = dyn_cast<asctile::BroadcastOp>(use.getOwner());
        return broadcastOp && ascvf2::isVectorizableGroupOperation(broadcastOp);
    });
}

SmallVector<SmallVector<Operation*>> findConsecutiveOperations(Block* block)
{
    SmallVector<SmallVector<Operation*>> chains;
    SmallVector<Operation*> chain;
    auto flush = [&] {
        if (chain.size() >= 2)
            chains.push_back(chain);
        chain.clear();
    };

    for (auto& op : *block) {
        if (ascvf2::isVectorizableGroupOperation(&op)) {
            chain.push_back(&op);
            continue;
        }
        if (chain.empty() && isCompatibleReductionPrefix(&op)) {
            chain.push_back(&op);
            continue;
        }
        if (!chain.empty() && isCompatibleReductionSink(&op, chain)) {
            chain.push_back(&op);
            continue;
        }
        flush();
    }
    flush();
    return chains;
}

LogicalResult outlineOperations(
    ArrayRef<Operation*> operations, SmallVector<Value>& outputs, DenseMap<Operation*, unsigned>& functionCounters)
{
    Operation* lastOp = operations.back();
    auto parentFunction = lastOp->getParentOfType<func::FuncOp>();
    auto module = lastOp->getParentOfType<ModuleOp>();
    if (!parentFunction || !module)
        return lastOp->emitError("VF group must be contained in a module function");

    DenseSet<Operation*> operationSet(operations.begin(), operations.end());
    SetVector<Value> inputs;
    SetVector<Operation*> constantOperations;
    for (auto* op : operations) {
        for (Value operand : op->getOperands()) {
            auto* definingOp = operand.getDefiningOp();
            if (definingOp && operationSet.contains(definingOp))
                continue;
            if (isa_and_nonnull<arith::ConstantOp>(definingOp)) {
                constantOperations.insert(definingOp);
                continue;
            }
            inputs.insert(operand);
        }
    }

    SmallVector<Type> resultTypes;
    for (Value output : outputs)
        resultTypes.push_back(output.getType());

    unsigned& functionCounter = functionCounters[parentFunction.getOperation()];
    std::string functionName;
    do {
        functionName = (parentFunction.getName() + "_vf" + Twine(++functionCounter)).str();
    } while (SymbolTable::lookupSymbolIn(module, functionName));

    auto functionType = FunctionType::get(
        lastOp->getContext(), ValueRange(inputs.getArrayRef()).getTypes(), ValueRange(outputs).getTypes());
    OpBuilder moduleBuilder(module.getBody(), module.getBody()->end());
    auto vectorFunction = moduleBuilder.create<func::FuncOp>(lastOp->getLoc(), functionName, functionType);
    vectorFunction.setPrivate();
    vectorFunction->setAttr(ascvf2::vectorFuncAttr, UnitAttr::get(lastOp->getContext()));

    Block* functionBody = vectorFunction.addEntryBlock();
    OpBuilder functionBuilder(functionBody, functionBody->end());

    IRMapping mapping;
    for (auto [input, argument] : llvm::zip_equal(inputs.getArrayRef(), vectorFunction.getArguments()))
        mapping.map(input, argument);
    for (Operation* constantOperation : constantOperations) {
        Operation* clonedConstant = functionBuilder.clone(*constantOperation);
        for (auto [result, clonedResult] :
             llvm::zip_equal(constantOperation->getResults(), clonedConstant->getResults()))
            mapping.map(result, clonedResult);
    }

    for (auto* op : operations)
        functionBuilder.clone(*op, mapping);

    SmallVector<Value> returnValues;
    for (Value output : outputs)
        returnValues.push_back(mapping.lookup(output));
    functionBuilder.create<func::ReturnOp>(lastOp->getLoc(), returnValues);

    OpBuilder builder(lastOp);
    auto groupOp =
        builder.create<ascvf2::VFGroupOp>(lastOp->getLoc(), resultTypes, ValueRange{}, ValueRange{}, ValueRange{});
    Block& groupBlock = groupOp.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&groupBlock);
    auto callOp = builder.create<func::CallOp>(lastOp->getLoc(), vectorFunction, inputs.getArrayRef());

    builder.create<ascvf2::YieldOp>(lastOp->getLoc(), callOp.getResults());

    for (auto [index, output] : llvm::enumerate(outputs)) {
        output.replaceUsesWithIf(
            groupOp.getResult(index), [&](OpOperand& use) { return !operationSet.contains(use.getOwner()); });
    }
    for (auto* op : llvm::reverse(operations))
        op->erase();
    return success();
}

LogicalResult processBlock(Block* block, DenseMap<Operation*, unsigned>& functionCounters)
{
    Operation* parentOp = block->getParentOp();
    if (isa_and_nonnull<ascvf2::VFGroupOp>(parentOp) || parentOp->getParentOfType<ascvf2::VFGroupOp>())
        return success();

    for (auto& operations : findConsecutiveOperations(block)) {
        SmallVector<Value> outputs;
        DenseSet<Operation*> operationSet(operations.begin(), operations.end());
        for (auto* op : operations) {
            for (Value result : op->getResults()) {
                if (llvm::any_of(
                        result.getUses(), [&](OpOperand& use) { return !operationSet.contains(use.getOwner()); }))
                    outputs.push_back(result);
            }
        }
        if (outputs.empty())
            continue;
        if (failed(outlineOperations(operations, outputs, functionCounters)))
            return failure();
    }
    return success();
}

struct OutlineVFGroupPass : public ascvf2::impl::OutlineVFGroupBase<OutlineVFGroupPass> {
    void runOnOperation() override
    {
        DenseMap<Operation*, unsigned> functionCounters;
        SmallVector<Block*> blocks;
        getOperation()->walk([&](Block* block) { blocks.push_back(block); });
        for (auto* block : blocks) {
            if (failed(processBlock(block, functionCounters))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf2::createOutlineVFGroupPass() { return std::make_unique<OutlineVFGroupPass>(); }
