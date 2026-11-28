/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF2_TYPES_TD
#define ASCVF2_TYPES_TD

include "Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "asctile/Dialect/AscTile/IR/Dialect.td"
include "asctile/Dialect/AscTile/IR/Types.td"

class AscVF2_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AscVF2_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AscVF2_UBLocalTensor : Type<
  And<[
    CPred<"::llvm::isa<::mlir::asctile::LocalTensorType>($_self)">,
    CPred<"::llvm::cast<::mlir::asctile::LocalTensorType>($_self).getLoc() == ::mlir::asctile::TensorLocation::UB">,
    CPred<"::llvm::cast<::mlir::asctile::LocalTensorType>($_self).hasStaticShape()">
  ]>,
  "static UB local tensor",
  "::mlir::asctile::LocalTensorType">;

#endif // ASCVF2_TYPES_TD
