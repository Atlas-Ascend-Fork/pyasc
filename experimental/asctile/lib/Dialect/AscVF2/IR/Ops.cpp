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
#include "asctile/Dialect/AscVF2/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/CVGroupCanonicalization.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "asctile/Dialect/AscVF2/IR/AscVF2Ops.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf2;

namespace {

template <typename OpT>
LogicalResult verifyVectorUnaryOp(OpT op)
{
    auto inputType = dyn_cast<VectorType>(op.getOperand().getType());
    auto resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!inputType || inputType.getRank() != 1 || !resultType || resultType != inputType)
        return op.emitOpError("operand and result must be matching rank-one vectors");
    if (isa<NotOp>(op) && !inputType.getElementType().isInteger())
        return op.emitOpError("operand element type must be an integer");
    return success();
}

template <typename OpT>
LogicalResult verifyVectorScalarOp(OpT op)
{
    auto inputType = dyn_cast<VectorType>(op.getBase().getType());
    auto resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!inputType || inputType.getRank() != 1 || !resultType || resultType != inputType)
        return op.emitOpError("base and result must be matching rank-one vectors");
    if (!op.getValue().getType().isIntOrFloat() || op.getValue().getType() != inputType.getElementType())
        return op.emitOpError("scalar must match the vector element type");
    if ((isa<ShLSOp>(op) || isa<ShRSOp>(op)) && !inputType.getElementType().isInteger())
        return op.emitOpError("base element type must be an integer");
    return success();
}

} // namespace

namespace {

void adjustLifetime(Operation*& firstOp, Operation*& lastOp, DominanceInfo& di)
{
    auto* commonBlock = di.findNearestCommonDominator(firstOp->getBlock(), lastOp->getBlock());
    Block* stopBlock = commonBlock;
    Operation* op = lastOp;

    auto process = [&](Operation* op) {
        if (auto vfFor = dyn_cast<emitasc::VFForOp>(op))
            lastOp = vfFor.getBody()->getTerminator();
    };

    process(op);
    while (op->getBlock() != stopBlock) {
        op = op->getParentOp();
        process(op);
    }
}

using Interval = std::pair<Operation*, Operation*>;

SmallVector<Operation*> getUsersInside(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    SmallVector<Operation*> users;
    llvm::copy_if(value.getUsers(), std::back_inserter(users), [&](Operation* op) {
        return di.dominates(groupOp.getBody(), op->getBlock());
    });
    return users;
}

bool isSingleUse(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    return getUsersInside(groupOp, value, di).size() == 1;
}

bool intersect(const Interval& lhs, const Interval& rhs, DominanceInfo& di)
{
    return ascendc::opPrecedes(rhs.first, lhs.second, di) && ascendc::opPrecedes(lhs.first, rhs.second, di);
}

Value getLocalTensor(Value memref)
{
    while (auto offset = memref.getDefiningOp<emitasc::PtrOffsetOp>())
        memref = offset.getBase();
    if (auto address = memref.getDefiningOp<ascendc::LocalTensorGetPhyAddrV2Op>())
        return address.getTensor();
    return {};
}

bool hasSameOffset(Value lhs, Value rhs)
{
    auto lhsOffset = lhs.getDefiningOp<emitasc::PtrOffsetOp>();
    auto rhsOffset = rhs.getDefiningOp<emitasc::PtrOffsetOp>();
    if (!lhsOffset || !rhsOffset)
        return lhs == rhs;

    Value lhsDynamicOffset = lhsOffset.getDynamicOffset();
    Value rhsDynamicOffset = rhsOffset.getDynamicOffset();
    if (lhsDynamicOffset && rhsDynamicOffset)
        return lhsDynamicOffset == rhsDynamicOffset;
    if (!lhsDynamicOffset && !rhsDynamicOffset)
        return lhsOffset.getStaticOffsetAttr() == rhsOffset.getStaticOffsetAttr();
    return false;
}

std::optional<Interval> getInterval(VFGroupOp groupOp, Value value, DominanceInfo& di)
{
    auto users = getUsersInside(groupOp, value, di);
    if (llvm::is_contained(groupOp.getInputTensors(), value))
        users.emplace_back(groupOp);
    if (llvm::is_contained(groupOp.getOutputTensors(), value))
        users.emplace_back(groupOp.getBody()->getTerminator());
    llvm::stable_sort(users, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
    if (users.empty())
        return std::nullopt;
    Operation* first = users.front();
    Operation* last = users.back();
    return Interval{first, last};
}

} // namespace

bool VFGroupOp::canReuse(OpOperand& lhsOpnd, OpOperand& rhsOpnd)
{
    Value lhs = lhsOpnd.get();
    Value rhs = rhsOpnd.get();
    DominanceInfo di;
    auto groupOp = cast<VFGroupOp>(getOperation());
    auto lhsInfo = getInterval(groupOp, lhs, di);
    auto rhsInfo = getInterval(groupOp, rhs, di);
    if (!lhsInfo || !rhsInfo)
        return true;
    if (intersect(lhsInfo.value(), rhsInfo.value(), di))
        return false;
    if (!ascendc::opPrecedes(lhsInfo->second, rhsInfo->first, di)) {
        std::swap(lhs, rhs);
        std::swap(lhsInfo, rhsInfo);
    }
    auto lhsInfoLong = lhsInfo;
    auto rhsInfoLong = rhsInfo;
    adjustLifetime(lhsInfoLong->first, lhsInfoLong->second, di);
    adjustLifetime(rhsInfoLong->first, rhsInfoLong->second, di);
    if (ascendc::opPrecedes(lhsInfoLong->second, rhsInfoLong->first, di))
        return true;
    auto loadOp = dyn_cast<ascendc::DataCopyLoadOp>(lhsInfo->second);
    auto storeOp = dyn_cast<ascendc::DataCopyStoreOp>(rhsInfo->first);
    Value loadTensor = loadOp ? getLocalTensor(loadOp.getSrc()) : Value();
    Value storeTensor = storeOp ? getLocalTensor(storeOp.getDst()) : Value();
    if (loadOp && storeOp && loadTensor == lhs && storeTensor == rhs && ascendc::opPrecedes(loadOp, storeOp, di) &&
        loadOp->getBlock() == storeOp->getBlock() && hasSameOffset(loadOp.getSrc(), storeOp.getDst())) {
        if (isSingleUse(groupOp, lhs, di) && isSingleUse(groupOp, rhs, di))
            return true;
        auto loop = dyn_cast<emitasc::VFForOp>(loadOp->getParentOp());
        if (loop && !loop->getParentOfType<emitasc::VFForOp>())
            return true;
    }
    return false;
}

void VFGroupOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context)
{
    results.add<
        ascir::EraseEmptyGroup<VFGroupOp, YieldOp>, ascir::EraseUnusedOperands<VFGroupOp, YieldOp>,
        ascir::EraseUnusedResults<VFGroupOp, YieldOp>>(context);
}

LogicalResult VFGroupOp::verify()
{
    if (getOperation()->getParentOfType<VFGroupOp>())
        return emitOpError("cannot be nested in another ascvf2.vf_group");

    auto yieldOp = cast<YieldOp>(getBody()->getTerminator());
    if (yieldOp.getNumOperands() != getNumResults()) {
        return emitOpError() << "number of yield operands (" << yieldOp.getNumOperands()
                             << ") must match number of results (" << getNumResults() << ")";
    }
    for (auto [result, operand] : llvm::zip_equal(getResults(), yieldOp.getOperands())) {
        if (result.getType() != operand.getType()) {
            return emitOpError() << "yield operand #" << result.getResultNumber() << " type " << operand.getType()
                                 << " does not match result type " << result.getType();
        }
    }

    auto walkResult = walk([&](Operation* op) {
        if (op != getOperation() && isa<VFGroupOp>(op)) {
            op->emitOpError("cannot be nested in another ascvf2.vf_group");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(walkResult.wasInterrupted());
}

LogicalResult ToBufferOp::verify()
{
    auto tensorType = dyn_cast<asctile::LocalTensorType>(getTensor().getType());
    if (!tensorType || !tensorType.hasStaticShape() || tensorType.getLoc() != asctile::TensorLocation::UB)
        return emitOpError("source must be a static UB local tensor");

    auto memrefType = dyn_cast<MemRefType>(getResult().getType());
    if (!memrefType || !memrefType.hasStaticShape() || memrefType.getRank() != tensorType.getRank() ||
        memrefType.getShape() != tensorType.getShape() || memrefType.getElementType() != tensorType.getElementType() ||
        !memrefType.getMemorySpace())
        return emitOpError("result must be an N-D UB memref with the source shape and element type");

    auto memorySpace = dyn_cast<IntegerAttr>(memrefType.getMemorySpace());
    if (!memorySpace || memorySpace.getValue().getSExtValue() != static_cast<int64_t>(asctile::TensorLocation::UB))
        return emitOpError("result memory space must be UB");

    if (memrefType.getNumElements() != tensorType.getNumElements())
        return emitOpError() << "result element count (" << memrefType.getNumElements()
                             << ") must match source element count (" << tensorType.getNumElements() << ")";
    return success();
}

OpFoldResult ToBufferOp::fold(FoldAdaptor)
{
    auto source = getTensor().getDefiningOp<ToTensorOp>();
    if (source && source.getBuffer().getType() == getResult().getType())
        return source.getBuffer();
    return {};
}

LogicalResult ToTensorOp::verify()
{
    auto memrefType = dyn_cast<MemRefType>(getBuffer().getType());
    if (!memrefType || !memrefType.hasStaticShape() || !memrefType.getMemorySpace())
        return emitOpError("source must be a UB memref");
    auto memorySpace = dyn_cast<IntegerAttr>(memrefType.getMemorySpace());
    if (!memorySpace || memorySpace.getValue().getSExtValue() != static_cast<int64_t>(asctile::TensorLocation::UB))
        return emitOpError("source memory space must be UB");

    auto tensorType = dyn_cast<asctile::LocalTensorType>(getResult().getType());
    if (!tensorType || !tensorType.hasStaticShape() || tensorType.getLoc() != asctile::TensorLocation::UB ||
        memrefType.getRank() != tensorType.getRank() || memrefType.getShape() != tensorType.getShape() ||
        memrefType.getElementType() != tensorType.getElementType())
        return emitOpError("result must be a UB local tensor with the source shape and element type");
    if (memrefType.getNumElements() != tensorType.getNumElements())
        return emitOpError() << "result element count (" << tensorType.getNumElements()
                             << ") must match source element count (" << memrefType.getNumElements() << ")";
    return success();
}

OpFoldResult ToTensorOp::fold(FoldAdaptor)
{
    auto source = getBuffer().getDefiningOp<ToBufferOp>();
    if (source && source.getTensor().getType() == getResult().getType())
        return source.getTensor();
    return {};
}

LogicalResult ReduceOp::verify()
{
    auto inputType = dyn_cast<VectorType>(getInput().getType());
    auto resultType = dyn_cast<VectorType>(getResult().getType());
    if (!inputType || inputType.getRank() != 1 || !resultType || resultType.getRank() != 1 ||
        resultType.getShape() != ArrayRef<int64_t>{1} || inputType.getElementType() != resultType.getElementType())
        return emitOpError("input must be vector<VxT> and result must be vector<1xT>");
    if (getDims().size() != 1 || cast<IntegerAttr>(getDims()[0]).getInt() != 0)
        return emitOpError("only reduction over vector dimension 0 is supported");
    if (getKind() != asctile::ReduceKind::Sum && getKind() != asctile::ReduceKind::Max &&
        getKind() != asctile::ReduceKind::Min)
        return emitOpError("only sum, max, and min reductions are supported");
    return success();
}

LogicalResult SimpleMaskRegOp::verify()
{
    if (getCountAttr() && getDynamicCount())
        return emitOpError("cannot specify both static and dynamic count");
    if (!getCountAttr() && !getDynamicCount())
        return emitOpError("must specify either static or dynamic count");
    return success();
}

LogicalResult AddSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult SubSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult MulSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult DivSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult MinSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult MaxSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult ShLSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult ShRSOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult LeakyReluOp::verify() { return verifyVectorScalarOp(*this); }
LogicalResult ReluOp::verify() { return verifyVectorUnaryOp(*this); }
LogicalResult NotOp::verify() { return verifyVectorUnaryOp(*this); }

//===----------------------------------------------------------------------===//
// AscVF2Dialect
//===----------------------------------------------------------------------===//

void AscVF2Dialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "asctile/Dialect/AscVF2/IR/AscVF2Ops.cpp.inc"
        >();
}
