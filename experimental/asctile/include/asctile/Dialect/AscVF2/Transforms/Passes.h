/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCVF2_TRANSFORMS_PASSES_H
#define ASCTILE_DIALECT_ASCVF2_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascvf2 {

#define GEN_PASS_DECL
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAssignVFGroupTensorsPass();
std::unique_ptr<Pass> createBufferizeLoopResultPass();
std::unique_ptr<Pass> createDropDeadWritePass();
std::unique_ptr<Pass> createExpandSimpleMaskRegPass();
std::unique_ptr<Pass> createFuseVFLoopsPass();
std::unique_ptr<Pass> createHoistRegTensorPass();
std::unique_ptr<Pass> createInsertVFSyncPass();
std::unique_ptr<Pass> createLegalizeVFGroupPass();
std::unique_ptr<Pass> createLowerToRegPass();
std::unique_ptr<Pass> createMergeIntoTransferPass();
std::unique_ptr<Pass> createOutlineVFGroupPass();
std::unique_ptr<Pass> createPromoteMemToRegPass();
std::unique_ptr<Pass> createRaiseVFForPass();
std::unique_ptr<Pass> createReorderGroupedOpsPass();
std::unique_ptr<Pass> createReuseRegTensorPass();
std::unique_ptr<Pass> createShrinkVFSignaturePass();
std::unique_ptr<Pass> createVectorizePass();
std::unique_ptr<Pass> createWrapVecScopePass();

} // namespace ascvf2

#define GEN_PASS_REGISTRATION
#include "asctile/Dialect/AscVF2/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ASCTILE_DIALECT_ASCVF2_TRANSFORMS_PASSES_H
