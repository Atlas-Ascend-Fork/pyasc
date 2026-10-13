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
#include "asctile/Dialect/AscTile/Utils/Attributes.h"
#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscendC/Utils/Attributes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "InitFuncDef.h"

namespace py = pybind11;
using namespace mlir;

namespace {

void bindAttrs(py::module& m)
{
    auto modAttr = m.def_submodule("attr");
    modAttr.attr("cv_ratio") = py::str(ascendc::attr::cvRatio);
    modAttr.attr("gm_barrier") = py::str(asctile::attr::gmBarrier);
    modAttr.attr("static_alloc") = py::str(ascendc::attr::staticAlloc);
    modAttr.attr("unroll_factor") = py::str(asctile::attr::unrollFactor);
}

void bindEnums(py::module& m)
{
    using ret = py::return_value_policy;

    py::enum_<asctile::TensorLocation>(
        m, "asctile_TensorLocation", py::module_local(), "A memory location of a :py:class:`LocalTensor`.")
        .value("Auto", asctile::TensorLocation::Auto, "Automatic resolution (if available)")
        .value("BT", asctile::TensorLocation::BT, "Bias Table buffer (Cube)")
        .value("L0A", asctile::TensorLocation::L0A, "L0A buffer (Cube)")
        .value("L0B", asctile::TensorLocation::L0B, "L0B buffer (Cube)")
        .value("L0C", asctile::TensorLocation::L0C, "L0C buffer (Cube)")
        .value("L1", asctile::TensorLocation::L1, "L1 buffer (Cube)")
        .value("UB", asctile::TensorLocation::UB, "Unified buffer (Vector)")
        .value("FIX", asctile::TensorLocation::FIX, "FixPipe buffer (Not supported)")
        .def(py::init([](const std::string& name) {
            auto normName = StringRef(name).upper();
            if (auto loc = asctile::symbolizeTensorLocation(normName))
                return *loc;
            throw py::value_error(name + " does not match a valid TensorLocation name");
        }))
        .def_static("symbolize", [](int32_t loc) -> asctile::TensorLocation {
            return static_cast<asctile::TensorLocation>(loc);
        });

    py::enum_<asctile::AtomicKind>(m, "AtomicKind", py::module_local())
        .value("Add", asctile::AtomicKind::Add)
        .value("Max", asctile::AtomicKind::Max)
        .value("Min", asctile::AtomicKind::Min)
        .def_static(
            "symbolize", [](int32_t kind) -> asctile::AtomicKind { return static_cast<asctile::AtomicKind>(kind); });

    py::enum_<asctile::CompareMode>(m, "CompareMode", py::module_local())
        .value("LT", asctile::CompareMode::LT)
        .value("GT", asctile::CompareMode::GT)
        .value("EQ", asctile::CompareMode::EQ)
        .value("LE", asctile::CompareMode::LE)
        .value("GE", asctile::CompareMode::GE)
        .value("NE", asctile::CompareMode::NE)
        .def_static(
            "symbolize", [](uint8_t mode) -> asctile::CompareMode { return static_cast<asctile::CompareMode>(mode); });

    py::enum_<asctile::ReduceKind>(m, "ReduceKind", py::module_local())
        .value("Sum", asctile::ReduceKind::Sum)
        .value("Max", asctile::ReduceKind::Max)
        .value("Min", asctile::ReduceKind::Min)
        .value("Prod", asctile::ReduceKind::Prod)
        .value("Mean", asctile::ReduceKind::Mean)
        .value("All", asctile::ReduceKind::All)
        .value("Any", asctile::ReduceKind::Any)
        .value("XorSum", asctile::ReduceKind::XorSum)
        .def_static("symbolize", [](int32_t kind) { return static_cast<asctile::ReduceKind>(kind); });

    py::enum_<asctile::RoundMode>(m, "asctile_RoundMode", py::module_local())
        .value("Default", asctile::RoundMode::Default)
        .value("NoRound", asctile::RoundMode::NoRound)
        .value("Rint", asctile::RoundMode::Rint)
        .value("Floor", asctile::RoundMode::Floor)
        .value("Ceil", asctile::RoundMode::Ceil)
        .value("Round", asctile::RoundMode::Round)
        .value("Trunc", asctile::RoundMode::Trunc)
        .value("Odd", asctile::RoundMode::Odd)
        .def_static("symbolize", [](int32_t mode) { return static_cast<asctile::RoundMode>(mode); });

    py::enum_<asctile::DistribMode>(m, "asctile_DistribMode", py::module_local())
        .value("FullVec0", asctile::DistribMode::FullVec0)
        .value("FullVec1", asctile::DistribMode::FullVec1)
        .value("SplitByM", asctile::DistribMode::SplitByM)
        .value("SplitByN", asctile::DistribMode::SplitByN)
        .value("JoinByM", asctile::DistribMode::JoinByM)
        .value("JoinByN", asctile::DistribMode::JoinByN)
        .def_static("symbolize", [](int32_t mode) { return static_cast<asctile::DistribMode>(mode); });
}

void bindAscTileType(py::module& m)
{
    using namespace pybind11::literals;

    m.def(
        "get_asctile_GlobalTensorType",
        [](const std::vector<int64_t>& shape, Type elementType) -> Type {
            return asctile::GlobalTensorType::get(shape, elementType);
        },
        "shape"_a, "element_type"_a);
    m.def(
        "get_asctile_LocalTensorType",
        [](const std::vector<int64_t>& shape, Type elementType, asctile::TensorLocation loc) -> Type {
            return asctile::LocalTensorType::get(shape, elementType, loc);
        },
        "shape"_a, "element_type"_a, "loc"_a = asctile::TensorLocation::UB);
    m.def(
        "get_tensor_location",
        [](Type type) -> asctile::TensorLocation {
            auto tileType = llvm::dyn_cast_if_present<asctile::LocalTensorType>(type);
            if (!tileType)
                throw std::runtime_error("get_tensor_location(): must be LocalTensorType");
            return tileType.getLoc();
        },
        "type"_a);
}

void bindAscTile(py::module& m)
{
    using ret = py::return_value_policy;
    py::class_<asctile::CountMaskOp, OpState>(m, "CountMaskOp", py::module_local())
        .def("get_region", &asctile::CountMaskOp::getRegion, ret::reference);
    py::class_<asctile::BitwiseMaskOp, OpState>(m, "BitwiseMaskOp", py::module_local())
        .def("get_region", &asctile::BitwiseMaskOp::getRegion, ret::reference);
}

} // namespace

void mlir::asctile::initIRModule(py::module&& m)
{
    m.def("load_dialects", [](MLIRContext& context) {
        DialectRegistry registry;
        registry.insert<asctile::AscTileDialect, ascvf::AscVFDialect, tensor::TensorDialect>();
        asctile::registerExternalModels(registry);
        ascvf::registerExternalModels(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    });

    bindAttrs(m);
    bindEnums(m);
    bindAscTileType(m);
    bindAscTile(m);
}
