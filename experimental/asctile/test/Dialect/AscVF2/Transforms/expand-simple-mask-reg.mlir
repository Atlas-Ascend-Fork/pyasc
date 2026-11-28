// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-expand-simple-mask-reg %s | FileCheck %s

// CHECK-LABEL: func.func @expand_static_and_dynamic_masks(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<f32>, %arg2: index) attributes {ascvf2.vector_func}
// CHECK: %c5 = arith.constant 5 : index
// CHECK: %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK: %1 = ascendc.create_mask f32, VL32 : !ascendc.mask_reg
// CHECK: %2 = emitasc.variable %c5 : index, memref<1xindex>
// CHECK: %3 = ascendc.update_mask f32, %2 : memref<1xindex>
// CHECK: %4 = emitasc.variable %arg2 : index, memref<1xindex>
// CHECK: %5 = ascendc.update_mask f32, %4 : memref<1xindex>
// CHECK: ascendc.add_reg {{.*}}, %0
// CHECK: ascendc.add_reg {{.*}}, %1
// CHECK: ascendc.add_reg {{.*}}, %3
// CHECK: ascendc.add_reg {{.*}}, %5
// CHECK-NOT: ascvf2.simple_mask_reg
func.func @expand_static_and_dynamic_masks(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<f32>, %arg2: index) attributes {ascvf2.vector_func} {
  %mask64 = ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
  %mask32 = ascvf2.simple_mask_reg f32 static 32 : !ascendc.mask_reg
  %mask5 = ascvf2.simple_mask_reg f32 static 5 : !ascendc.mask_reg
  %dynamic = ascvf2.simple_mask_reg f32 dynamic %arg2 : !ascendc.mask_reg
  %dst64 = ascendc.reg_tensor : <f32>
  %dst32 = ascendc.reg_tensor : <f32>
  %dst5 = ascendc.reg_tensor : <f32>
  %dynamicDst = ascendc.reg_tensor : <f32>
  ascendc.add_reg %dst64, %arg0, %arg1, %mask64 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dst32, %arg0, %arg1, %mask32 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dst5, %arg0, %arg1, %mask5 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dynamicDst, %arg0, %arg1, %dynamic : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @skip_unmarked_function(%arg0: !ascendc.reg_tensor<f32>, %arg1: index)
// CHECK: ascvf2.simple_mask_reg
// CHECK: ascendc.add_reg
func.func @skip_unmarked_function(%arg0: !ascendc.reg_tensor<f32>, %arg1: index) {
  %mask = ascvf2.simple_mask_reg f32 dynamic %arg1 : !ascendc.mask_reg
  %dst = ascendc.reg_tensor : <f32>
  ascendc.add_reg %dst, %arg0, %arg0, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}
