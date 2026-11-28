# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(TARGET_NAME AscTilePythonExt)

add_library(${TARGET_NAME} OBJECT
  IR.cpp
  Module.cpp
  OpBuilder.cpp
  Passes.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
  AscTileDialect
  AscTileTransforms
  AscVFDialect
  AscVFTransforms
  AscVF2Dialect
  AscVF2Transforms
  AscTileLowerToAsc
  AscTileAscendCTransforms
  MLIRAffineToStandard
  MLIRAffineTransforms
)

add_dependencies(${TARGET_NAME} AscTilePybindGen)

# pybind11==2.10.3 issue
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(${TARGET_NAME} PRIVATE -Wno-deprecated-literal-operator -Wno-unknown-warning-option)
endif()

add_ascir_extension(${TARGET_NAME})
