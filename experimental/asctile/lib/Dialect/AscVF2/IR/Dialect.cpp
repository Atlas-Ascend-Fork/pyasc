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
#include "asctile/Dialect/AscendC/IR/Interfaces.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/IR/DialectImplementation.h"

// Inline dialect definitions
#include "asctile/Dialect/AscVF2/IR/AscVF2Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf2;

//===----------------------------------------------------------------------===//
// AscVF2Dialect
//===----------------------------------------------------------------------===//

void AscVF2Dialect::initialize()
{
    registerAttributes();
    registerTypes();
    registerOps();
    addInterface<ascir::PermissiveInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

namespace {

struct VFGroupSrcModel : public ascendc::OpWithSrc::ExternalModel<VFGroupSrcModel, ascvf2::VFGroupOp> {
    SmallVector<Value> getSrcTensors(Operation* op) const
    {
        return llvm::to_vector(cast<ascvf2::VFGroupOp>(op).getInputTensors());
    }
};

struct VFGroupReusableSrcModel
    : public ascendc::OpWithReusableSrc::ExternalModel<VFGroupReusableSrcModel, ascvf2::VFGroupOp> {
    SmallVector<Value> getSrcTensors(Operation* op) const { return cast<ascendc::OpWithSrc>(op).getSrcTensors(); }

    SmallVector<Value> getDstTensors(Operation* op) const { return cast<ascendc::OpWithDst>(op).getDstTensors(); }

    bool canReuse(Operation* op, OpOperand& src, OpOperand& dst) const
    {
        return cast<ascvf2::VFGroupOp>(op).canReuse(src, dst);
    }
};

struct VFGroupDstModel : public ascendc::OpWithDst::ExternalModel<VFGroupDstModel, ascvf2::VFGroupOp> {
    SmallVector<Value> getDstTensors(Operation* op) const
    {
        auto group = cast<ascvf2::VFGroupOp>(op);
        SmallVector<Value> tensors;
        tensors.append(group.getOutputTensors().begin(), group.getOutputTensors().end());
        tensors.append(group.getTempTensors().begin(), group.getTempTensors().end());
        return tensors;
    }
};

} // namespace

void ascvf2::registerExternalModels(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext* context, AscVF2Dialect*) {
        ascvf2::VFGroupOp::attachInterface<VFGroupSrcModel>(*context);
        ascvf2::VFGroupOp::attachInterface<VFGroupDstModel>(*context);
        ascvf2::VFGroupOp::attachInterface<VFGroupReusableSrcModel>(*context);
    });
}
