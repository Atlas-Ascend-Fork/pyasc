/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"

#include "ascir/Extension/PyOpBuilder.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "InitFuncDef.h"

namespace py = pybind11;
using namespace mlir;
using mlir::ascir::PyOpBuilder;

namespace {

void bindCreateAscTileOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_asctile_CountMaskOp",
            [](PyOpBuilder& self, Value& count, std::optional<Value> other) -> asctile::CountMaskOp {
                Value otherVal = other.has_value() ? *other : Value();
                return self.create<asctile::CountMaskOp>(count, otherVal);
            },
            py::arg("count"), py::arg("other") = py::none())
        .def(
            "create_asctile_BitwiseMaskOp",
            [](PyOpBuilder& self, Value& highBits, Value& lowBits,
               std::optional<Value> other) -> asctile::BitwiseMaskOp {
                Value otherVal = other.has_value() ? *other : Value();
                return self.create<asctile::BitwiseMaskOp>(highBits, lowBits, otherVal);
            },
            py::arg("highBits"), py::arg("lowBits"), py::arg("other") = py::none())
        .def(
            "create_asctile_InlineVFOp",
            [](PyOpBuilder& self, Type result, const std::vector<Value>& inputs, const std::string& code) -> Value {
                return self.create<asctile::InlineVFOp>(result, ValueRange{inputs}, StringRef(code));
            })
        .def(
            "create_asctile_AssertOp",
            [](PyOpBuilder& self, const Value& cond, const std::string& msg) {
                self.create<asctile::AssertOp>(cond, self->getStringAttr(msg));
            })
        .def(
            "create_asctile_InlineOp",
            [](PyOpBuilder& self, const std::string& value, const std::optional<std::vector<Value>>& args) {
                static std::vector<Value> noValues;
                self.create<asctile::InlineOp>(self->getStringAttr(value), args.value_or(noValues));
            },
            "value"_a, "args"_a = py::none())
#include "asctile/Dialect/AscTile/IR/AscTileOpBindings.h.inc"
        ;
}

void bindCreateBuiltinOperations(py::class_<PyOpBuilder>& clss)
{
    clss.def("create_UnrealizedConversionCastOp", [](PyOpBuilder& self, Type result, Value operand) -> Value {
        return self.create<UnrealizedConversionCastOp>(result, operand)->getResult(0);
    });
}

void bindCreateTensorOperations(py::class_<PyOpBuilder>& clss)
{
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    clss.def(
            "create_tensor_CastOp",
            [](PyOpBuilder& self, Type result, Value operand) -> Value {
                return self.create<tensor::CastOp>(result, operand);
            })
        .def(
            "create_tensor_ConcatOp",
            [](PyOpBuilder& self, Type result, uint64_t dim, const std::vector<Value>& inputs) -> Value {
                return self.create<tensor::ConcatOp>(result, dim, inputs);
            })
        .def(
            "create_tensor_EmptyOp",
            [](PyOpBuilder& self, Type result) -> Value {
                return self.create<tensor::EmptyOp>(result, /*dynamicSizes*/ ValueRange{});
            })
        .def(
            "create_tensor_SplatOp",
            [](PyOpBuilder& self, Type result, Value input) -> Value {
                return self.create<tensor::SplatOp>(result, input);
            })
        .def("cast_tensor_location", [](PyOpBuilder& self, asctile::TensorLocation loc, Value tensor) -> Value {
            auto type = dyn_cast<asctile::LocalTensorType>(tensor.getType());
            if (!type)
                throw std::runtime_error("cast_tensor_location(): value must have LocalTensorType");
            return self.create<tensor::CastOp>(
                asctile::LocalTensorType::get(type.getShape(), type.getElementType(), loc), tensor);
        });
}

} // namespace

void mlir::asctile::initAsctileBuilder(pybind11::class_<PyOpBuilder>& clss)
{
    bindCreateAscTileOperations(clss);
    bindCreateBuiltinOperations(clss);
    bindCreateTensorOperations(clss);
}
