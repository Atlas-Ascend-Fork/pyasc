# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_dialect_library(AscVF2Transforms
  OutlineVFGroup.cpp
  ShrinkVFSignature.cpp
  FuseVFLoops.cpp
  HoistRegTensor.cpp
  DropDeadWrite.cpp
  MergeIntoTransfer.cpp
  InsertVFSync.cpp
  LegalizeVFGroup.cpp
  AssignVFGroupTensors.cpp
  LowerToReg.cpp
  BufferizeLoopResult.cpp
  ExpandSimpleMaskReg.cpp
  PromoteMemToReg.cpp
  RaiseVFFor.cpp
  ReorderGroupedOps.cpp
  ReuseRegTensor.cpp
  WrapVecScope.cpp
  Vectorize.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASCIR_INCLUDE_DIR}/asctile/Dialect/AscVF2

  DEPENDS
  AscVF2PassIncGen

LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRAsc
  MLIREmitAsc
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRVectorDialect
  AscTileDialect
  AscVF2Dialect
  AscVF2Utils
)
