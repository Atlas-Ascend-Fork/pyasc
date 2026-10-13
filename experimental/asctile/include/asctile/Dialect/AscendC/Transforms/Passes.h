/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCENDC_TRANSFORMS_PASSES_H
#define ASCTILE_DIALECT_ASCENDC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascendc {

#define GEN_PASS_DECL
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAllocateTensorPass();
std::unique_ptr<Pass> createComputeMemoryConsumptionPass();
std::unique_ptr<Pass> createComputeReuseGroupPass();
std::unique_ptr<Pass> createDispatchAllocPass();
std::unique_ptr<Pass> createFillAscOperandsPass();
std::unique_ptr<Pass> createFixupMmadAccParamsPass();
std::unique_ptr<Pass> createFuseBufIdSyncPass();
std::unique_ptr<Pass> createInsertBiasBufIdSyncPass();
std::unique_ptr<Pass> createInsertBufIdSyncPass();
std::unique_ptr<Pass> createInsertCrossCoreSyncPass();
std::unique_ptr<Pass> createInsertCrossCoreSyncGMPass();
std::unique_ptr<Pass> createInsertInitDumpPass();
std::unique_ptr<Pass> createInsertSubBlockGuardPass();
std::unique_ptr<Pass> createLowerToL0Pass();
std::unique_ptr<Pass> createPromoteCVBlockPass();
std::unique_ptr<Pass> createRefineCubePositionPass();
std::unique_ptr<Pass> createReuseTensorAllocationPass();
std::unique_ptr<Pass> createReuseUBAllocationPass(bool reuseInOut = false);
std::unique_ptr<Pass> createRemoveDebugOpsPass();
std::unique_ptr<Pass> createUnifyBiasTensorPass();

void populateLowerToL0Patterns(RewritePatternSet& patterns);

} // namespace ascendc

#define GEN_PASS_REGISTRATION
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ASCTILE_DIALECT_ASCENDC_TRANSFORMS_PASSES_H
