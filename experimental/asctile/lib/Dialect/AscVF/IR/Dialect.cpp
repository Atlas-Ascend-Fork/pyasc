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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/IR/DialectImplementation.h"

// Inline dialect definitions
#include "asctile/Dialect/AscVF/IR/AscVFDialect.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::initialize()
{
    registerAttributes();
    registerTypes();
    registerOps();
}

//===----------------------------------------------------------------------===//
// External models
//===----------------------------------------------------------------------===//

struct VFGroupSrcModel : public ascendc::OpWithSrc::ExternalModel<VFGroupSrcModel, ascvf::VFGroupOp> {
    SmallVector<Value> getSrcTensors(Operation* op) const { return cast<ascvf::VFGroupOp>(op).getSrcTensors(); }
};

struct VFGroupDstModel : public ascendc::OpWithDst::ExternalModel<VFGroupDstModel, ascvf::VFGroupOp> {
    SmallVector<Value> getDstTensors(Operation* op) const { return cast<ascvf::VFGroupOp>(op).getDstTensors(); }
};

struct VFGroupReusableSrcModel
    : public ascendc::OpWithReusableSrc::ExternalModel<VFGroupReusableSrcModel, ascvf::VFGroupOp> {
    bool canReuse(Operation* op, OpOperand& src, OpOperand& dst) const
    {
        return cast<ascvf::VFGroupOp>(op).canReuse(src, dst);
    }
};

void ascvf::registerExternalModels(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext* ctx, AscVFDialect* dialect) {
        dialect->addInterface<ascir::PermissiveInlinerInterface>();
        ascvf::VFGroupOp::attachInterface<VFGroupSrcModel>(*ctx);
        ascvf::VFGroupOp::attachInterface<VFGroupDstModel>(*ctx);
        ascvf::VFGroupOp::attachInterface<VFGroupReusableSrcModel>(*ctx);
    });
}
