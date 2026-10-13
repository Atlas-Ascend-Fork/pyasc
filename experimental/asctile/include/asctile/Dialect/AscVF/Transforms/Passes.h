/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCVF_TRANSFORMS_PASSES_H
#define ASCTILE_DIALECT_ASCVF_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascvf {

#define GEN_PASS_DECL
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDispatchVFFusionPass();
std::unique_ptr<Pass> createEliminateCommonMaskPass();
std::unique_ptr<Pass> createEliminateDataTransferPass();
std::unique_ptr<Pass> createEraseBarrierPass();
std::unique_ptr<Pass> createFindVFGroupPass();
std::unique_ptr<Pass> createFuseVFForPass();
std::unique_ptr<Pass> createInlineVFGroupPass();
std::unique_ptr<Pass> createLowerToRegPass();
std::unique_ptr<Pass> createMaterializeLoadStorePass();
std::unique_ptr<Pass> createPromoteCastOpsPass();
std::unique_ptr<Pass> createReorderOpsInVecScopePass();

} // namespace ascvf

#define GEN_PASS_REGISTRATION
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // ASCTILE_DIALECT_ASCVF_TRANSFORMS_PASSES_H
