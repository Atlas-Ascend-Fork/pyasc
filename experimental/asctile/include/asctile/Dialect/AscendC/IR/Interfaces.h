/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_ASCENDC_IR_INTERFACES_H
#define ASCTILE_ASCENDC_IR_INTERFACES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"

// Interface declarations
#include "asctile/Dialect/AscendC/IR/AscTileAscendCOpInterfaces.h.inc"

namespace mlir {
namespace ascendc {

void registerReusableSrcModels(DialectRegistry& registry);

} // namespace ascendc
} // namespace mlir

#endif // ASCTILE_ASCENDC_IR_INTERFACES_H
