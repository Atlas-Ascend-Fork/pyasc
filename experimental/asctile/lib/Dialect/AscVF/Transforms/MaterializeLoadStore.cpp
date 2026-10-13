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
#include "asctile/Dialect/AscVF/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_MATERIALIZELOADSTORE
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

// Find local tensors that are used as dst or src
ValueVector getUsedLocalTensors(emitasc::VecScopeOp vecScope)
{
    ValueVector usedLocalTensors;
    vecScope.walk([&](Operation* op) {
        if (auto load = dyn_cast<ascvf::LoadOp>(op)) {
            usedLocalTensors.emplace_back(load.getSrcTensor());
        } else if (auto store = dyn_cast<ascvf::StoreOp>(op)) {
            usedLocalTensors.emplace_back(store.getDstTensor());
        }
    });
    return ascvf::deduplicate(usedLocalTensors);
}

ValueMap<Value> setAddress(emitasc::VecScopeOp vecScopeOp, ArrayRef<Value> usedTensors)
{
    ValueMap<Value> addrTensors;
    OpBuilder builder(vecScopeOp);
    for (auto value : usedTensors) {
        auto tensorType = cast<ascendc::LocalTensorType>(value.getType());
        auto shape = tensorType.getShape();
        auto elemType = tensorType.getElementType();
        auto type = MemRefType::get(shape, elemType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto getPhyAddrOp = builder.create<ascendc::LocalTensorGetPhyAddrV2Op>(builder.getUnknownLoc(), type, value);
        addrTensors[value] = getPhyAddrOp.getResult();
    }
    return addrTensors;
}

void materialize(emitasc::VecScopeOp vecScopeOp, Type groupType)
{
    ValueMap<Value> addrTensors;
    // materialize getPhyAddr
    SmallVector<Value> usedTensors = getUsedLocalTensors(vecScopeOp);
    addrTensors = setAddress(vecScopeOp, usedTensors);

    // materialize DataCopy from LoadMicro, StoreMicro
    vecScopeOp->walk([&](Operation* op) {
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        if (auto load = dyn_cast<ascvf::LoadOp>(op)) {
            auto tensor = load.getSrcTensor();
            auto tensorType = cast<ascendc::LocalTensorType>(tensor.getType());
            auto shape = tensorType.getShape();
            auto elemType = tensorType.getElementType();
            auto dstReg = load.getDstReg();
            if (isa<ascendc::MaskRegType>(dstReg.getType())) {
                auto maskReg = load.getDstReg();
                auto ureg = builder.create<ascendc::UnalignRegOp>(
                    builder.getUnknownLoc(), ascendc::UnalignRegType::get(load.getContext()));
                auto dstAddrWithOffset = builder.create<emitasc::PtrOffsetOp>(
                    builder.getUnknownLoc(),
                    MemRefType::get(shape, elemType, {}, static_cast<int>(ascendc::AddressSpace::ubuf)),
                    addrTensors[tensor], IntegerAttr{}, load.getOffset());
                builder.create<ascendc::LoadUnAlignPreOp>(builder.getUnknownLoc(), ureg, dstAddrWithOffset);
                auto selReg = builder.create<ascendc::RegTensorOp>(
                    builder.getUnknownLoc(),
                    ascendc::RegTensorType::get(load.getContext(), builder.getIntegerType(8, false)));
                builder.create<ascendc::LoadUnAlignOp>(builder.getUnknownLoc(), selReg, ureg, dstAddrWithOffset);
                builder.create<ascendc::MaskGenWithRegTensorOp>(
                    builder.getUnknownLoc(), maskReg, selReg,
                    builder.getIntegerType(groupType.getIntOrFloatBitWidth(), false));
            } else {
                bool process = false;
                auto elemType = cast<ascendc::RegTensorType>(load.getDstReg().getType()).getElementType();
                Value tensor = load.getSrcTensor();
                auto resultType = MemRefType::get(shape, elemType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
                auto srcAddr = builder.create<emitasc::PtrOffsetOp>(
                    builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, load.getOffset());

                if (auto maskOp = load.getMask().getDefiningOp<ascendc::CreateMaskOp>()) {
                    // WA: if load one elem then make broadcast (need LoadUnalign and if has next duplicate then load
                    // with broadcast)
                    if (maskOp.getMask() == ascendc::MaskPattern::VL1) {
                        process = true;
                        builder.create<ascendc::DataCopyLoadOp>(
                            builder.getUnknownLoc(), load.getDstReg(), srcAddr,
                            ascendc::LoadDist::DIST_BRC_B32); // TODO: Change on loadUnalign
                    }
                }
                if (!process) {
                    process = true;
                    builder.create<ascendc::DataCopyLoadOp>(
                        builder.getUnknownLoc(), load.getDstReg(), srcAddr, ascendc::LoadDist::DIST_NORM);
                }
                if (!process) {
                    llvm_unreachable("add load support");
                }
            }
            load.erase();
        } else if (auto store = dyn_cast<ascvf::StoreOp>(op)) {
            bool isAlignment = store->hasAttr(ascvf::attr::alignment);
            Value tensor = store.getDstTensor();
            auto shape = cast<ascendc::LocalTensorType>(tensor.getType()).getShape();
            auto resultType = MemRefType::get(shape, groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
            auto dstAddr = builder.create<emitasc::PtrOffsetOp>(
                builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, store.getOffset());
            bool process = false;
            if (auto maskOp = store.getMask().getDefiningOp<ascendc::CreateMaskOp>()) {
                if (maskOp.getMask() == ascendc::MaskPattern::VL1) {
                    process = true;
                    builder.create<ascendc::DataCopyStoreOp>(
                        builder.getUnknownLoc(), dstAddr, store.getSrcReg(), store.getMask(),
                        ascendc::StoreDist::DIST_FIRST_ELEMENT_B32); // TODO: choose for other types
                }
            }
            if (!process) {
                if (isa<ascendc::MaskRegType>(store.getSrcReg().getType())) {
                    auto elemType = builder.getIntegerType(8, false);
                    auto tensor = store.getDstTensor();
                    auto dstAddr = addrTensors[tensor];
                    auto offset = store.getOffset();
                    auto resultType =
                        MemRefType::get(shape, elemType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
                    auto dstAddrWithOffset = builder.create<emitasc::PtrOffsetOp>(
                        builder.getUnknownLoc(), resultType, dstAddr, IntegerAttr{}, offset);
                    auto ureg = builder.create<ascendc::UnalignRegOp>(
                        builder.getUnknownLoc(), ascendc::UnalignRegType::get(store.getContext()));
                    Type intType;
                    if (groupType.getIntOrFloatBitWidth() == 32) {
                        intType = builder.getIntegerType(32, false);
                    } else {
                        intType = builder.getIntegerType(16, false);
                    }
                    auto castedOffset = builder.create<emitasc::ReinterpretCastOp>(
                        builder.getUnknownLoc(),
                        MemRefType::get(shape, intType, {}, static_cast<int>(ascendc::AddressSpace::ubuf)),
                        dstAddrWithOffset);
                    builder.create<ascendc::StoreUnAlignOp>(
                        builder.getUnknownLoc(), castedOffset, store.getSrcReg(), ureg);
                    auto sizeOfType = groupType.getIntOrFloatBitWidth() / 8;
                    auto maskElemCount = 256 / sizeOfType;
                    auto sizeInUi8 = maskElemCount / 8;
                    auto step = sizeInUi8;
                    auto tailOffset =
                        builder.create<arith::AddIOp>(builder.getUnknownLoc(), offset, consts.index(step));
                    auto dstAddrWithTailOffset = builder.create<emitasc::PtrOffsetOp>(
                        builder.getUnknownLoc(), resultType, dstAddr, IntegerAttr{}, tailOffset);
                    builder.create<ascendc::StoreUnAlignPostOp>(builder.getUnknownLoc(), dstAddrWithTailOffset, ureg);
                } else {
                    builder.create<ascendc::DataCopyStoreOp>(
                        builder.getUnknownLoc(), dstAddr, store.getSrcReg(), store.getMask(),
                        ascendc::StoreDist::DIST_NORM);
                    process = true;
                }
            }
            store.erase();
        }
    });
}

struct MaterializeLoadStorePass : public ascvf::impl::MaterializeLoadStoreBase<MaterializeLoadStorePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp fusedOp) {
            auto elemType = getElementTypeOrSelf(fusedOp.getGroupType());
            fusedOp.walk([&](emitasc::VecScopeOp vecScope) { materialize(vecScope, elemType); });
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createMaterializeLoadStorePass()
{
    return std::make_unique<MaterializeLoadStorePass>();
}
