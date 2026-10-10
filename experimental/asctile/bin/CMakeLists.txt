# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# asctile-lsp

set(TARGET_NAME asctile-lsp)

add_mlir_tool(${TARGET_NAME} asctile-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  AscTileDialect
  AscVFDialect

  MLIRIR
  MLIRLspServerLib
  MLIRAsc
  MLIREmitAsc
)

mlir_check_all_link_libraries(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# asctile-opt

set(TARGET_NAME asctile-opt)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

add_mlir_tool(${TARGET_NAME} asctile-opt.cpp PARTIAL_SOURCES_INTENDED SUPPORT_PLUGINS)

target_link_libraries(${TARGET_NAME} PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  MLIRAscTransforms
  MLIREmitAsc
  MLIRMlirOptMain

  AscTileAscendCTransforms
  AscTileLowerToAsc
  AscTileTransforms
  AscVFTransforms
)

llvm_update_compile_flags(${TARGET_NAME})

mlir_check_all_link_libraries(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
