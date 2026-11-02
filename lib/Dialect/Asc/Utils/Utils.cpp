/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Constants.h"
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

template <typename... T>
using AllowInline = ascir::AllowlistInlinerInterface<T...>;

namespace ascendc {

int64_t getTypeSize(Type type)
{
    if (auto shaped = dyn_cast<ShapedType>(type))
        return shaped.getNumElements() * getTypeSize(shaped.getElementType());
    return type.getIntOrFloatBitWidth() / CHAR_BIT;
}

int64_t getTypeSizeCubeBlockAlign(ShapedType type, TPosition position)
{
    auto shape = type.getShape();
    int64_t elemSize = getElementTypeSize(type);
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        int64_t align = ((position == TPosition::A1 || position == TPosition::A2) && i == 1) ?
                            cubeKBlockBytes / elemSize :
                            cubeBlockSize;
        size *= static_cast<int64_t>(llvm::alignTo(shape[i], align));
    }
    return size * elemSize;
}

int64_t getElementTypeSize(ShapedType type) { return getTypeSize(type.getElementType()); }

bool opPrecedes(Operation* lhs, Operation* rhs) { return lhs != rhs && lhs->isBeforeInBlock(rhs); }

bool opPrecedes(Operation* lhs, Operation* rhs, DominanceInfo& di)
{
    if (lhs == rhs) {
        return false;
    }
    Block* lhsBlk = lhs->getBlock();
    Block* rhsBlk = rhs->getBlock();
    if (lhsBlk == rhsBlk) {
        return lhs->isBeforeInBlock(rhs);
    }
    Block* dtr = di.findNearestCommonDominator(lhsBlk, rhsBlk);
    Operation* lhsAnc = dtr->findAncestorOpInBlock(*lhs);
    Operation* rhsAnc = dtr->findAncestorOpInBlock(*rhs);
    if (lhsAnc != rhsAnc)
        return lhsAnc->isBeforeInBlock(rhsAnc);
    if (lhs->isAncestor(rhs))
        return true;
    if (rhs->isAncestor(lhs))
        return false;
    if (auto ifOp = dyn_cast<scf::IfOp>(lhsAnc))
        return ifOp.thenBlock()->findAncestorOpInBlock(*lhs);
    if (auto whileOp = dyn_cast<scf::WhileOp>(lhsAnc))
        return whileOp.getBeforeBody()->findAncestorOpInBlock(*lhs);
    return di.properlyDominates(lhs, rhs);
}

void registerInlinerInterfaces(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext*, BuiltinDialect* dialect) {
        dialect->addInterface<AllowInline<UnrealizedConversionCastOp>>();
    });
    registry.addExtension(+[](MLIRContext*, emitc::EmitCDialect* dialect) {
        dialect->addInterface<AllowInline<emitc::CastOp, emitc::ConstantOp>>();
    });
}

ModuleOp getModule(Operation* op)
{
    if (isa<ModuleOp>(op))
        return cast<ModuleOp>(op);
    auto mod = op->getParentOfType<ModuleOp>();
    assert(mod && "operation must be within a module");
    return mod;
}

StringRef getCompilationArch(Operation* op)
{
    if (auto attr = getModule(op)->getAttrOfType<StringAttr>(attr::compilationArch))
        return attr.getValue();
    return {};
}

StringRef getSocVersion(Operation* op)
{
    if (auto attr = getModule(op)->getAttrOfType<StringAttr>(attr::socVersion))
        return attr.getValue();
    return {};
}

std::optional<int64_t> getVecLen(Operation* op)
{
    if (auto attr = getModule(op)->getAttrOfType<IntegerAttr>(attr::vfVecLen))
        return attr.getValue().getSExtValue();
    return std::nullopt;
}

bool isTargetArchC310(Operation* op) { return getCompilationArch(op) == "c310"; }

LocalTensorAutoOp getAllocationRoot(Value v)
{
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
            Value src = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
            return getAllocationRoot(src);
        }
    }
    auto* defOp = v.getDefiningOp();
    if (!defOp)
        return {};
    if (auto op = dyn_cast<LocalTensorAutoOp>(defOp))
        return op;
    if (auto op = dyn_cast<LocalTensorReinterpretCastOp>(defOp))
        return getAllocationRoot(op.getIn());
    if (auto op = dyn_cast<LocalTensorSubIndexOp>(defOp))
        return getAllocationRoot(op.getTensor());
    if (auto aiv = dyn_cast<ascendc::IfAIVOp>(defOp)) {
        if (auto result = dyn_cast<OpResult>(v)) {
            unsigned idx = result.getResultNumber();
            return getAllocationRoot(aiv.getBody()->getTerminator()->getOperand(idx));
        }
    }
    if (auto aiv = dyn_cast<ascendc::IfAICOp>(defOp)) {
        if (auto result = dyn_cast<OpResult>(v)) {
            unsigned idx = result.getResultNumber();
            return getAllocationRoot(aiv.getBody()->getTerminator()->getOperand(idx));
        }
    }
    return {};
}

Pipe getOpPipe(Operation* op, Pipe defaultPipe)
{
    return llvm::TypeSwitch<Operation*, Pipe>(op)
        .Case<VectorOp>([](auto) { return Pipe::PIPE_V; })
        .Case<MmadOp, MmadWithBiasOp>([](auto) { return Pipe::PIPE_M; })
        .Case([](FixpipeOp) { return Pipe::PIPE_FIX; })
        .Case([](CopyToL0Op) { return Pipe::PIPE_MTE1; })
        .Case([](FillOp) { return Pipe::PIPE_MTE2; })
        .Case([defaultPipe](DataCopyOp copyOp) {
            if (auto direction = copyOp.getDirection()) {
                auto [src, dst] = *direction;
                if (src == TPosition::A1 && dst == TPosition::VECCALC ||
                    src == TPosition::A1 && (dst == TPosition::A2 || dst == TPosition::B2 || dst == TPosition::CO1))
                    return Pipe::PIPE_MTE1;
                if (src == TPosition::GM)
                    return Pipe::PIPE_MTE2;
                if (dst == TPosition::GM || src == TPosition::VECCALC && dst == TPosition::A1)
                    return Pipe::PIPE_MTE3;
                if (src == TPosition::VECCALC && dst == TPosition::VECCALC)
                    return Pipe::PIPE_V;
            }
            return defaultPipe;
        })
        .Case<LocalTensorGetValueOp, LocalTensorSetValueOp>([](auto) { return Pipe::PIPE_S; })
        .Default(defaultPipe);
}

bool isWriteToAllocation(Operation* op, ascendc::LocalTensorAutoOp root)
{
    if (auto dstOp = dyn_cast<ascendc::OpWithDst>(op)) {
        for (Value dst : dstOp.getDstTensors()) {
            auto dstRoot = ascendc::getAllocationRoot(dst);
            if (dstRoot && dstRoot == root)
                return true;
        }
    }
    return false;
}

} // namespace ascendc
} // namespace mlir
