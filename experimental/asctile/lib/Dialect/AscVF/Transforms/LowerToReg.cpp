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
#include "asctile/Dialect/AscVF/Utils/Attributes.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_LOWERTOREG
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;
namespace {

void setAlignmentAttr(Operation* op) { op->setAttr(ascvf::attr::alignment, UnitAttr::get(op->getContext())); }

SmallVector<int64_t> getStrides(ArrayRef<int64_t> shape)
{
    SmallVector<int64_t> strides;
    std::exclusive_scan(shape.rbegin(), shape.rend(), std::back_inserter(strides), 1, std::multiplies<>{});
    std::reverse(strides.begin(), strides.end());
    return strides;
}

struct VFInfo {
    Value oneRepeatSize, vecLen;
    // contains strides for loop variables for linear access
    // Ex. addr = i * stride0 + j * stride1 + ... -> strides = [stride0, stride1, ...]
    SmallVector<int64_t> strides;
    Type elemType;
    SmallVector<int64_t> shape;

    static Value createVecLen(ImplicitLocOpBuilder& builder, Operation* op)
    {
        if (auto vecLen = ascendc::getVecLen(op))
            return builder.create<arith::ConstantIndexOp>(*vecLen);
        return builder.create<ascendc::GetVecLenOp>(builder.getIndexType());
    }

    explicit VFInfo(emitasc::VecScopeOp op)
    {
        auto groupOp = op->getParentOfType<ascvf::VFGroupOp>();
        assert(groupOp && "emitasc.vec_scope op must be inside ascvf.vf_group op");
        auto builder = ImplicitLocOpBuilder::atBlockBegin(UnknownLoc::get(op.getContext()), op.getBody());
        vecLen = createVecLen(builder, op);
        ascendc::LocalTensorType groupType = cast<ascendc::LocalTensorType>(groupOp.getGroupType());
        elemType = getElementTypeOrSelf(groupType);
        Value sizeIndex = builder.create<arith::ConstantIndexOp>(ascendc::getTypeSize(elemType));
        oneRepeatSize = builder.create<arith::DivSIOp>(vecLen, sizeIndex);
        shape = SmallVector<int64_t>{groupType.getShape()};
        // TODO: Add memory pattern finder. Conservative we believe it is linear access memory
        // shape = [96, 16] -> strides = [16, 1]. For softmax by columns strides is [1, 16]
        // TODO: merge unimportant shapes [123, 456, 16] -> [123 * 456, 16]
        strides = getStrides(shape);
    }
};

class RewriterAdaptor {
    using Rewriter = ConversionPatternRewriter;

    Rewriter& rewriter;

    template <size_t... indices>
    auto createRegTensorsImpl(Type elemType, std::index_sequence<indices...>)
    {
        return std::make_tuple((static_cast<void>(indices), createRegTensor(elemType))...);
    }

public:
    RewriterAdaptor(Rewriter& rewriter) : rewriter(rewriter) {}
    ~RewriterAdaptor() = default;

    Rewriter& operator*() { return rewriter; }
    Rewriter* operator->() { return &rewriter; }

    template <typename OpT, typename... Args>
    OpT create(Args&&... args)
    {
        return rewriter.create<OpT>(rewriter.getUnknownLoc(), args...);
    }

    ascendc::RegTensorOp createRegTensor(Type elemType)
    {
        return create<ascendc::RegTensorOp>(rewriter.getType<ascendc::RegTensorType>(elemType));
    }

    template <size_t count>
    auto createRegTensors(Type elemType)
    {
        return createRegTensorsImpl(elemType, std::make_index_sequence<count>{});
    }

    Value createUI32Variable(Value initValue)
    {
        return create<emitasc::VariableOp>(MemRefType::get(1, rewriter.getIntegerType(32U, false)), initValue);
    }

    Value createMaskOp(Type elemType, ascendc::MaskPattern pattern = ascendc::MaskPattern::ALL)
    {
        return create<ascendc::CreateMaskOp>(rewriter.getType<ascendc::MaskRegType>(), elemType, pattern);
    }

    Value updateMaskOp(Value calCount, Type elemType)
    {
        return create<ascendc::UpdateMaskOp>(rewriter.getType<ascendc::MaskRegType>(), calCount, elemType);
    }

    // bias + lhs * rhs
    Value createFMA(Value bias, Value lhs, Value rhs)
    {
        auto mulOp = create<arith::MulIOp>(lhs, rhs);
        return create<arith::AddIOp>(bias, mulOp);
    }
};

template <typename OpT>
struct ConvertOp : public OpConversionPattern<OpT> {
    ConvertOp(MLIRContext* context, const VFInfo& vfInfo, PatternBenefit benefit = 1)
        : OpConversionPattern<OpT>::OpConversionPattern(context, benefit), vfInfo(vfInfo)
    {}

    virtual LogicalResult matchAndRewrite(OpT op, RewriterAdaptor& adaptor) const = 0;

    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        RewriterAdaptor adaptor(rewriter);
        return matchAndRewrite(op, adaptor);
    }

protected:
    const VFInfo& vfInfo;
};

Value makeNestedLoop(ArrayRef<int64_t> dims, ArrayRef<int64_t> strides, RewriterAdaptor& adaptor)
{
    assert(dims.size() == strides.size());
    ascir::ConstantOpBuilder consts(*adaptor);
    Value offset = consts.index(0);
    for (int i = 0; i < dims.size(); ++i) {
        auto loop = adaptor.create<emitasc::VFForOp>(consts.index(dims[i]));
        adaptor->setInsertionPointToStart(loop.getBody());
        offset = adaptor.createFMA(offset, loop.getInductionVar(), consts.index(strides[i]));
    }
    return offset;
}

template <typename L2Op, typename RegOp>
struct ConvertBinaryL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);

        auto [src0Reg, src1Reg, dstReg] = adaptor.createRegTensors<3>(vfInfo.elemType);
        auto countVal = consts.index(dims.back());
        Value calCount = adaptor.createUI32Variable(countVal);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        offset = adaptor.create<arith::AddIOp>(offset, mulOp);
        auto load1Op = adaptor.create<ascvf::LoadOp>(src0Reg, op.getSrc0(), offset, updateMask);
        auto load2Op = adaptor.create<ascvf::LoadOp>(src1Reg, op.getSrc1(), offset, updateMask);
        adaptor.create<RegOp>(dstReg, src0Reg, src1Reg, maskAll);
        auto storeOp = adaptor.create<ascvf::StoreOp>(op.getDst(), offset, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename RegOp>
struct ConvertUnaryL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);
        auto [srcReg, dstReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        Value countVal = consts.index(dims.back());
        Value calCount = adaptor.createUI32Variable(countVal);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        offset = adaptor.createFMA(offset, loop.getInductionVar(), vfInfo.oneRepeatSize);
        auto loadOp = adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), offset, updateMask);
        adaptor.create<RegOp>(dstReg, srcReg, maskAll);
        auto storeOp = adaptor.create<ascvf::StoreOp>(op.getDst(), offset, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename RegOp>
struct ConvertVecScalarL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dstReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& shape = vfInfo.shape;
        int64_t count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        Value countVal = consts.index(count);
        Value calCount = adaptor.createUI32Variable(countVal);
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp, updateMask);
        adaptor.create<RegOp>(dstReg, srcReg, op.getScalar(), updateMask);
        adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename L2Op, typename BinRegOp>
struct ConvertVecScalarWithDuplicateL2 : public ConvertOp<L2Op> {
    using ConvertOp<L2Op>::ConvertOp;
    using ConvertOp<L2Op>::vfInfo;

    LogicalResult matchAndRewrite(L2Op op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);
        auto [srcReg, dupReg, dstReg] = adaptor.createRegTensors<3>(vfInfo.elemType);
        Value countVal = consts.index(dims.back());
        Value calCount = adaptor.createUI32Variable(countVal);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        offset = adaptor.createFMA(offset, loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), offset, updateMask);
        adaptor.create<ascendc::DuplicateRegOp>(dupReg, op.getScalar(), maskAll);
        adaptor.create<BinRegOp>(dstReg, srcReg, dupReg, maskAll);
        adaptor.create<ascvf::StoreOp>(op.getDst(), offset, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename ReduceL2Op, typename AccumulateRegOp, typename ReduceRegOp>
struct ConvertReduceL2 : public ConvertOp<ReduceL2Op> {
    using ConvertOp<ReduceL2Op>::ConvertOp;
    using ConvertOp<ReduceL2Op>::vfInfo;

    template <typename ReduceOpType>
    static Value getNeutralElement(ascir::ConstantOpBuilder& consts, Type elemType)
    {
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMaxL2Op>) {
            if (elemType.isF32())
                return consts.f32(-std::numeric_limits<float>::infinity());
            if (elemType.isF16())
                return consts.f16(-std::numeric_limits<float>::infinity());
            if (elemType.isInteger(32))
                return consts.i32(-std::numeric_limits<int>::infinity());
        }
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceSumL2Op>) {
            if (elemType.isF32())
                return consts.f32(0);
            if (elemType.isF16())
                return consts.f16(0);
            if (elemType.isInteger(32))
                return consts.i32(0);
        }
        if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMinL2Op>) {
            if (elemType.isF32())
                return consts.f32(std::numeric_limits<float>::infinity());
            if (elemType.isF16())
                return consts.f16(std::numeric_limits<float>::infinity());
            if (elemType.isInteger(32))
                return consts.i32(std::numeric_limits<int>::infinity());
        }
        llvm_unreachable("unknown neutral element");
    }

    LogicalResult matchAndRewrite(ReduceL2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, dstReg, accReg, acc0Reg] = adaptor.createRegTensors<4>(vfInfo.elemType);
        ascir::ConstantOpBuilder consts(*adaptor);
        Value neutral = getNeutralElement<ReduceL2Op>(consts, vfInfo.elemType);
        Value count = consts.index(op.getSrc().getType().getNumElements());
        adaptor.create<ascendc::DuplicateScalarRegOp>(accReg, neutral);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes = adaptor.create<arith::DivSIOp>(adaptor->getIndexType(), count, vfInfo.oneRepeatSize);
        Value calCount = adaptor.createUI32Variable(count);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), mulOp, updateMask);
        adaptor.create<AccumulateRegOp>(accReg, accReg, srcReg, maskAll);
        adaptor->setInsertionPointAfter(loop);
        Value remOp = adaptor.create<arith::RemSIOp>(count, vfInfo.oneRepeatSize);
        Value cmpOp = adaptor.create<arith::CmpIOp>(arith::CmpIPredicate::ne, remOp, consts.index(0));
        auto ifOp = adaptor->create<scf::IfOp>(adaptor->getUnknownLoc(), cmpOp, false);
        adaptor->setInsertionPointToStart(ifOp.getBody());
        adaptor.create<ascendc::DuplicateScalarRegOp>(acc0Reg, neutral);
        Value lastIter = adaptor.create<arith::MulIOp>(repeatTimes, vfInfo.oneRepeatSize);
        Value tailMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc(), lastIter, tailMask);
        adaptor.create<ascendc::SelectRegOp>(acc0Reg, srcReg, acc0Reg, tailMask);
        adaptor.create<AccumulateRegOp>(accReg, accReg, acc0Reg, maskAll);
        adaptor->setInsertionPointAfter(ifOp);
        adaptor.create<ReduceRegOp>(dstReg, accReg, maskAll);
        Value maskOne = adaptor.createMaskOp(vfInfo.elemType, ascendc::MaskPattern::VL1);
        adaptor.create<ascvf::StoreOp>(op.getDst(), consts.index(0), dstReg, maskOne);
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertReduceSum : public ConvertOp<ascendc::ReduceSumOp> {
    using ConvertOp<ascendc::ReduceSumOp>::ConvertOp;
    using ConvertOp<ascendc::ReduceSumOp>::vfInfo;

    LogicalResult matchAndRewrite(ascendc::ReduceSumOp op, RewriterAdaptor& adaptor) const override
    {
        if (op.getPattern() != ascendc::ReducePattern::AR) {
            return failure();
        }
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        SmallVector<Value> indexes;
        Value offsetSrc = consts.index(0);
        Value offsetDst = consts.index(0);
        assert(!dims.empty());
        for (size_t i = 0; i < dims.size() - 1; ++i) {
            auto loop = adaptor.create<emitasc::VFForOp>(consts.index(dims[i]));
            adaptor->setInsertionPointToStart(loop.getBody());
            int64_t strideSrc = vfInfo.strides[i];
            int64_t strideDst = vfInfo.strides[i] / dims.back();
            auto mulSrc = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideSrc));
            offsetSrc = adaptor.create<arith::AddIOp>(offsetSrc, mulSrc);
            auto mulDst = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideDst));
            offsetDst = adaptor.create<arith::AddIOp>(offsetDst, mulDst);
        }
        auto type = ascendc::LocalTensorType::get(SmallVector<int64_t>{dims.back()}, vfInfo.elemType);
        auto dstView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getDst(), offsetDst);
        auto srcView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getSrc(), offsetSrc);
        adaptor.create<ascendc::ReduceSumL2Op>(dstView, srcView, op.getSharedTmpBuffer(), consts.index(dims.back()));
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertCompareScalar : public ConvertOp<ascendc::CompareScalarL2Op> {
    using ConvertOp<ascendc::CompareScalarL2Op>::ConvertOp;
    using ConvertOp<ascendc::CompareScalarL2Op>::vfInfo;

    LogicalResult matchAndRewrite(ascendc::CompareScalarL2Op op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);
        auto srcReg = adaptor.createRegTensor(vfInfo.elemType);
        auto dstReg = adaptor.create<ascendc::MaskRegOp>(ascendc::MaskRegType::get(op.getContext()));
        auto scalarVal = op.getSrc1Scalar();
        auto srcType = dyn_cast<ascendc::LocalTensorType>(op.getSrc0().getType());
        auto dstType = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType());
        assert(srcType && "CompareScalarL2Op must have LocalTensorType");
        auto calCount = consts.index(srcType.getNumElements());
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), calCount, vfInfo.oneRepeatSize);
        Value sreg = adaptor.createUI32Variable(calCount);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());

        Value updateMask = adaptor.updateMaskOp(sreg, vfInfo.elemType);
        Value loadOffset = adaptor.createFMA(offset, loop.getInductionVar(), vfInfo.oneRepeatSize);
        auto loadOp = adaptor.create<ascvf::LoadOp>(srcReg, op.getSrc0(), loadOffset, updateMask);
        adaptor.create<ascendc::CompareScalarRegOp>(dstReg, srcReg, op.getSrc1Scalar(), updateMask, op.getCmpMode());

        auto sizeOfElem = consts.index(vfInfo.elemType.getIntOrFloatBitWidth() / CHAR_BIT);
        auto maskElemCount = adaptor.create<arith::DivSIOp>(vfInfo.vecLen, sizeOfElem);
        auto dstStrideBytes = adaptor.create<arith::DivSIOp>(maskElemCount, consts.index(CHAR_BIT));
        auto dstStrideInElemType = adaptor.create<arith::DivSIOp>(dstStrideBytes, sizeOfElem);
        Value storeOffset = adaptor.createFMA(offset, loop.getInductionVar(), dstStrideBytes);
        adaptor.create<ascvf::StoreOp>(op.getDst(), storeOffset, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertSelectL2 : public ConvertOp<ascendc::SelectL2Op> {
    using ConvertOp<ascendc::SelectL2Op>::ConvertOp;
    using ConvertOp<ascendc::SelectL2Op>::vfInfo;

    LogicalResult matchAndRewrite(ascendc::SelectL2Op op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);
        auto [src0Reg, src1Reg, dstReg] = adaptor.createRegTensors<3>(vfInfo.elemType);
        auto maskReg = adaptor.create<ascendc::MaskRegOp>(ascendc::MaskRegType::get(op.getContext()));
        auto selReg = adaptor.createRegTensor(adaptor->getIntegerType(8, false));
        auto countVal = consts.index(dims.back());
        Value calCount = adaptor.createUI32Variable(countVal);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
        auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
        adaptor->setInsertionPointToStart(loop.getBody());
        Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
        Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
        auto divOp = adaptor.create<arith::DivSIOp>(vfInfo.oneRepeatSize, consts.index(CHAR_BIT));
        Value mul2Op = adaptor.create<arith::MulIOp>(loop.getInductionVar(), divOp);
        Value offsetAct = adaptor.create<arith::AddIOp>(offset, mulOp);
        Value offsetSel = adaptor.create<arith::AddIOp>(offset, mul2Op);
        auto loadMaskOp = adaptor.create<ascvf::LoadOp>(maskReg, op.getSelMask(), offsetSel, updateMask);
        auto load1Op = adaptor.create<ascvf::LoadOp>(src0Reg, op.getSrc0(), offsetAct, updateMask);
        auto load2Op = adaptor.create<ascvf::LoadOp>(src1Reg, op.getSrc1(), offsetAct, updateMask);
        adaptor.create<ascendc::SelectRegOp>(dstReg, src0Reg, src1Reg, maskReg);
        auto storeOp = adaptor.create<ascvf::StoreOp>(op.getDst(), offsetAct, dstReg, updateMask);
        adaptor->eraseOp(op);
        return success();
    }
};

template <typename ReduceHL, typename ReduceL2>
struct ConvertHLReduceWithIndex : public ConvertOp<ReduceHL> {
    using ConvertOp<ReduceHL>::ConvertOp;
    using ConvertOp<ReduceHL>::vfInfo;

    LogicalResult matchAndRewrite(ReduceHL op, RewriterAdaptor& adaptor) const override
    {
        if (op.getPattern() != ascendc::ReducePattern::AR) {
            return failure();
        }
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        SmallVector<Value> indexes;
        Value offsetSrc = consts.index(0);
        Value offsetDst = consts.index(0);
        assert(!dims.empty());
        for (size_t i = 0; i < dims.size() - 1; ++i) {
            auto loop = adaptor.create<emitasc::VFForOp>(consts.index(dims[i]));
            adaptor->setInsertionPointToStart(loop.getBody());
            int64_t strideSrc = vfInfo.strides[i];
            int64_t strideDst = vfInfo.strides[i] / dims.back();
            auto mulSrc = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideSrc));
            offsetSrc = adaptor.create<arith::AddIOp>(offsetSrc, mulSrc);
            auto mulDst = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideDst));
            offsetDst = adaptor.create<arith::AddIOp>(offsetDst, mulDst);
        }
        auto type = ascendc::LocalTensorType::get(SmallVector<int64_t>{dims.back()}, vfInfo.elemType);
        auto dstView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getDst(), offsetDst);
        auto srcView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getSrc(), offsetSrc);
        adaptor.create<ReduceL2>(dstView, srcView, op.getSharedTmpBuffer(), consts.index(dims.back()), consts.index(0));
        adaptor->eraseOp(op);
        return success();
    }
};

LogicalResult create1DDuplicateScalar(ascendc::DuplicateL2Op op, RewriterAdaptor& adaptor, const VFInfo& vfInfo)
{
    auto [srcReg, tmpReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
    Value zero = adaptor.create<arith::ConstantIndexOp>(0);
    ascir::ConstantOpBuilder consts(*adaptor);

    auto scalarVal = op.getScalar();
    if (isa<ascendc::LocalTensorType>(scalarVal.getType())) {
        Value maskOne = adaptor.createMaskOp(vfInfo.elemType, ascendc::MaskPattern::VL1);
        adaptor.create<ascvf::LoadOp>(srcReg, op.getScalar(), consts.index(0), maskOne);
        Value maskAll = adaptor.createMaskOp(vfInfo.elemType);
        adaptor.create<ascendc::DuplicateRegOp>(tmpReg, srcReg, maskAll);
    } else {
        adaptor.create<ascendc::DuplicateScalarRegOp>(tmpReg, scalarVal);
    }
    auto countVal = consts.index(op.getDst().getType().getNumElements());
    Value calCount = adaptor.createUI32Variable(countVal);
    Value repeatTimes = adaptor.create<arith::CeilDivSIOp>(adaptor->getIndexType(), countVal, vfInfo.oneRepeatSize);
    auto loop = adaptor.create<emitasc::VFForOp>(repeatTimes);
    adaptor->setInsertionPoint(loop.getBody()->getTerminator());
    Value updateMask = adaptor.updateMaskOp(calCount, vfInfo.elemType);
    Value mulOp = adaptor.create<arith::MulIOp>(loop.getInductionVar(), vfInfo.oneRepeatSize);
    auto storeOp = adaptor.create<ascvf::StoreOp>(op.getDst(), mulOp, tmpReg, updateMask);
    adaptor->eraseOp(op);
    return success();
}

// duplicate_l2
/*
for(...) {
  [other nested loops]
    reg0 = load %src[0]
    for(i) {
      store(%dst[i * stride + offset], reg0, mask)
*/
struct ConvertDuplicateL2 : public ConvertOp<ascendc::DuplicateL2Op> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(ascendc::DuplicateL2Op op, RewriterAdaptor& adaptor) const override
    {
        auto [srcReg, tmpReg] = adaptor.createRegTensors<2>(vfInfo.elemType);
        ascir::ConstantOpBuilder consts(*adaptor);
        auto scalarVal = op.getScalar();
        const auto& dims = vfInfo.shape;
        Value offset =
            makeNestedLoop(ArrayRef<int64_t>{dims}.drop_back(), ArrayRef<int64_t>{vfInfo.strides}.drop_back(), adaptor);
        auto type = ascendc::LocalTensorType::get(SmallVector<int64_t>{dims.back()}, vfInfo.elemType);
        auto dstView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getDst(), offset);
        auto duplicateOp = adaptor.create<ascendc::DuplicateL2Op>(dstView, scalarVal, consts.index(dims.back()));
        if (create1DDuplicateScalar(duplicateOp, adaptor, vfInfo).failed())
            return failure();
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertBroadcast : public ConvertOp<ascendc::BroadcastOp> {
    using ConvertOp::ConvertOp;
    LogicalResult matchAndRewrite(ascendc::BroadcastOp op, RewriterAdaptor& adaptor) const override
    {
        ascir::ConstantOpBuilder consts(*adaptor);
        const auto& dims = vfInfo.shape;
        SmallVector<Value> indexes;
        Value offsetSrc = consts.index(0);
        Value offsetDst = consts.index(0);
        assert(!dims.empty());
        for (size_t i = 0; i < dims.size() - 1; ++i) {
            auto loop = adaptor.create<emitasc::VFForOp>(consts.index(dims[i]));
            adaptor->setInsertionPointToStart(loop.getBody());
            int64_t strideSrc = vfInfo.strides[i];
            int64_t strideDst = vfInfo.strides[i] / dims.back();
            auto mulSrc = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideSrc));
            offsetSrc = adaptor.create<arith::AddIOp>(offsetSrc, mulSrc);
            auto mulDst = adaptor.create<arith::MulIOp>(loop.getInductionVar(), consts.index(strideDst));
            offsetDst = adaptor.create<arith::AddIOp>(offsetDst, mulDst);
        }
        auto type = ascendc::LocalTensorType::get(SmallVector<int64_t>{dims.back()}, vfInfo.elemType);
        auto dstView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getSrc(), offsetDst);
        auto srcView = adaptor.create<ascendc::LocalTensorSubIndexOp>(type, op.getDst(), offsetSrc);

        auto dupOp = adaptor.create<ascendc::DuplicateL2Op>(srcView, dstView, consts.index(dims.back()));
        if (create1DDuplicateScalar(dupOp, adaptor, vfInfo).failed())
            return failure();
        adaptor->eraseOp(op);
        return success();
    }
};

struct ConvertLoadOp : public ConvertOp<ascvf::LoadOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(ascvf::LoadOp loadOp, RewriterAdaptor& adaptor) const override
    {
        auto subIndex = loadOp.getSrcTensor().getDefiningOp<ascendc::LocalTensorSubIndexOp>();
        if (!subIndex)
            return failure();
        auto offset = adaptor.create<arith::AddIOp>(subIndex.getIndex(), loadOp.getOffset());
        adaptor->modifyOpInPlace(loadOp, [&] {
            loadOp.getSrcTensorMutable().assign(subIndex.getTensor());
            loadOp.getOffsetMutable().assign(offset);
        });
        return success();
    }
};

struct ConvertStoreOp : public ConvertOp<ascvf::StoreOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(ascvf::StoreOp storeOp, RewriterAdaptor& adaptor) const override
    {
        auto subIndex = storeOp.getDstTensor().getDefiningOp<ascendc::LocalTensorSubIndexOp>();
        if (!subIndex)
            return failure();
        auto offset = adaptor.create<arith::AddIOp>(subIndex.getIndex(), storeOp.getOffset());
        adaptor->modifyOpInPlace(storeOp, [&] {
            storeOp.getDstTensorMutable().assign(subIndex.getTensor());
            storeOp.getOffsetMutable().assign(offset);
        });
        return success();
    }
};

struct ConvertSubIndex : public ConvertOp<ascendc::LocalTensorSubIndexOp> {
    using ConvertOp::ConvertOp;

    LogicalResult matchAndRewrite(ascendc::LocalTensorSubIndexOp subIndex, RewriterAdaptor& adaptor) const override
    {
        if (subIndex.getResult().getUsers().empty()) {
            adaptor->eraseOp(subIndex);
            return success();
        }
        return failure();
    }
};

LogicalResult convertToReg(emitasc::VecScopeOp vecScopeOp)
{
    VFInfo vfInfo(vecScopeOp);
    MLIRContext* context = vecScopeOp.getContext();
    ConversionTarget target(*context);
    target.addDynamicallyLegalDialect<ascendc::AscendCDialect>([](Operation* op) {
        return llvm::none_of(op->getOperandTypes(), [](Type type) { return isa<ascendc::LocalTensorType>(type); });
    });
    target.addLegalOp<ascendc::LocalTensorSubIndexOp>();
    target.addDynamicallyLegalOp<ascvf::LoadOp>([](ascvf::LoadOp loadOp) {
        auto* defOp = loadOp.getSrcTensor().getDefiningOp();
        if (defOp)
            if (auto subIndex = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp)) {
                return subIndex->getParentOfType<emitasc::VecScopeOp>() !=
                       loadOp->getParentOfType<emitasc::VecScopeOp>();
            }
        return true;
    });
    target.addDynamicallyLegalOp<ascvf::StoreOp>([](ascvf::StoreOp storeOp) {
        auto* defOp = storeOp.getDstTensor().getDefiningOp();
        if (defOp)
            if (auto subIndex = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp)) {
                return subIndex->getParentOfType<emitasc::VecScopeOp>() !=
                       storeOp->getParentOfType<emitasc::VecScopeOp>();
            }
        return true;
    });
    target.addLegalDialect<arith::ArithDialect, ascvf::AscVFDialect, emitasc::EmitAscDialect, scf::SCFDialect>();
    RewritePatternSet patterns(context);
    patterns.add<
        // BinaryOp
        ConvertBinaryL2<ascendc::AddL2Op, ascendc::AddRegOp>, ConvertBinaryL2<ascendc::AndL2Op, ascendc::AndRegOp>,
        ConvertBinaryL2<ascendc::DivL2Op, ascendc::DivRegOp>, ConvertBinaryL2<ascendc::MaxL2Op, ascendc::MaxRegOp>,
        ConvertBinaryL2<ascendc::MinL2Op, ascendc::MinRegOp>, ConvertBinaryL2<ascendc::MulL2Op, ascendc::MulRegOp>,
        ConvertBinaryL2<ascendc::MulAddDstL2Op, ascendc::MulAddDstRegOp>,
        ConvertBinaryL2<ascendc::OrL2Op, ascendc::OrRegOp>, ConvertBinaryL2<ascendc::PreluL2Op, ascendc::PreluRegOp>,
        ConvertBinaryL2<ascendc::SubL2Op, ascendc::SubRegOp>,
        // UnaryOp
        ConvertUnaryL2<ascendc::AbsL2Op, ascendc::AbsRegOp>, ConvertUnaryL2<ascendc::ExpL2Op, ascendc::ExpRegOp>,
        ConvertUnaryL2<ascendc::LnL2Op, ascendc::LnRegOp>, ConvertUnaryL2<ascendc::NegL2Op, ascendc::NegRegOp>,
        ConvertUnaryL2<ascendc::NotL2Op, ascendc::NotRegOp>, ConvertUnaryL2<ascendc::ReluL2Op, ascendc::ReluRegOp>,
        ConvertUnaryL2<ascendc::SqrtL2Op, ascendc::SqrtRegOp>,
        // VecScalarOp
        ConvertVecScalarL2<ascendc::LeakyReluL2Op, ascendc::LeakyReluRegOp>,
        ConvertVecScalarL2<ascendc::ShiftLeftL2Op, ascendc::ShiftLeftsRegOp>,
        ConvertVecScalarL2<ascendc::ShiftRightL2Op, ascendc::ShiftRightsRegOp>,
        // VecScalarWithDuplicate
        ConvertVecScalarWithDuplicateL2<ascendc::AddsL2Op, ascendc::AddRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::DivsL2Op, ascendc::DivRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MaxsL2Op, ascendc::MaxRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MinsL2Op, ascendc::MinRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::MulsL2Op, ascendc::MulRegOp>,
        ConvertVecScalarWithDuplicateL2<ascendc::SubsL2Op, ascendc::SubRegOp>,
        // Reduce
        ConvertReduceL2<ascendc::ReduceMaxL2Op, ascendc::MaxRegOp, ascendc::ReduceMaxRegOp>,
        ConvertReduceL2<ascendc::ReduceSumL2Op, ascendc::AddRegOp, ascendc::ReduceSumRegOp>,
        ConvertReduceL2<ascendc::ReduceMinL2Op, ascendc::MinRegOp, ascendc::ReduceMinRegOp>,
        // Highlevel reduce
        ConvertReduceSum, ConvertHLReduceWithIndex<ascendc::ReduceMaxOp, ascendc::ReduceMaxL2Op>,
        ConvertHLReduceWithIndex<ascendc::ReduceMinOp, ascendc::ReduceMinL2Op>, ConvertLoadOp, ConvertStoreOp,
        ConvertSubIndex,
        // Shape Manipulation Ops
        ConvertDuplicateL2, ConvertBroadcast, ConvertCompareScalar, ConvertSelectL2>(context, vfInfo);
    return applyPartialConversion(vecScopeOp, target, std::move(patterns));
}

emitasc::VecScopeOp wrapInVecScope(ascvf::VFGroupOp op)
{
    OpBuilder builder(op);
    auto vecScope = builder.create<emitasc::VecScopeOp>(builder.getUnknownLoc());
    op.getBody()->moveBefore(&vecScope.getRegion(), vecScope.getRegion().end());
    builder.createBlock(&op.getRegion(), op.getRegion().end());
    auto yield = builder.create<ascvf::YieldOp>(builder.getUnknownLoc());
    vecScope.getBody()->getTerminator()->erase(); // replace ascvf.yield with emitasc.yield
    builder.setInsertionPointToEnd(vecScope.getBody());
    builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
    vecScope->moveBefore(yield);
    return vecScope;
}

struct LowerToRegPass : public ascvf::impl::LowerToRegBase<LowerToRegPass> {
    void runOnOperation() override
    {
        getOperation().walk([this](ascvf::VFGroupOp op) {
            auto vecScope = wrapInVecScope(op);
            if (convertToReg(vecScope).failed())
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createLowerToRegPass() { return std::make_unique<LowerToRegPass>(); }
