# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(AscVF2Dialect.h.inc -gen-dialect-decls -dialect=ascvf2)
mlir_tablegen(AscVF2Dialect.cpp.inc -gen-dialect-defs -dialect=ascvf2)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(AscVF2Ops.h.inc -gen-op-decls)
mlir_tablegen(AscVF2Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(AscVF2Types.h.inc -gen-typedef-decls -dialect=ascvf2 -typedefs-dialect=ascvf2)
mlir_tablegen(AscVF2Types.cpp.inc -gen-typedef-defs -dialect=ascvf2 -typedefs-dialect=ascvf2)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(AscVF2Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ascvf2)
mlir_tablegen(AscVF2Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ascvf2)
mlir_tablegen(AscVF2Enums.h.inc -gen-enum-decls)
mlir_tablegen(AscVF2Enums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(AscVF2DialectIncGen)

add_mlir_doc(Doc AscVF2 Dialects/ -gen-dialect-doc -dialect ascvf2)
