/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_DIALECT_ASCVF2_UTILS_REG_TYPE_CONVERTER_H
#define ASCTILE_DIALECT_ASCVF2_UTILS_REG_TYPE_CONVERTER_H

#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace mlir {
namespace ascvf2 {

struct RegTypeConverter : TypeConverter {
    RegTypeConverter()
    {
        addConversion([](Type type) { return std::optional<Type>{type}; });
        addConversion([](asctile::LocalTensorType type) {
            return std::optional<Type>{ascendc::LocalTensorType::get(type.getShape(), type.getElementType())};
        });
        addConversion([](VectorType type) {
            if (type.getElementType().isInteger(1))
                return std::optional<Type>{ascendc::MaskRegType::get(type.getContext())};
            return std::optional<Type>{ascendc::RegTensorType::get(type.getContext(), type.getElementType())};
        });
        addArgumentMaterialization(addUnrealizedCast);
        addSourceMaterialization(addUnrealizedCast);
        addTargetMaterialization(addUnrealizedCast);
    }

private:
    static Value addUnrealizedCast(OpBuilder& builder, Type type, ValueRange inputs, Location loc)
    {
        return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
    }
};

} // namespace ascvf2
} // namespace mlir

#endif // ASCTILE_DIALECT_ASCVF2_UTILS_REG_TYPE_CONVERTER_H
