// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-hoist-reg-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_reg_tensors(%arg0: index, %arg1: index, %arg2: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT: %[[FIRST:.*]] = ascendc.reg_tensor
// CHECK-NEXT: %[[SECOND:.*]] = ascendc.reg_tensor
// CHECK-NEXT: %[[THIRD:.*]] = ascendc.reg_tensor
// CHECK: %[[MASK:.*]] = ascendc.create_mask f32, ALL
// CHECK-NEXT: ascendc.abs_reg %[[FIRST]], %[[SECOND]], %[[MASK]]
// CHECK: scf.for
// CHECK: ascendc.abs_reg %[[THIRD]], %[[FIRST]], %[[MASK]]
func.func @hoist_reg_tensors(%upper: index, %lower: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  ascendc.abs_reg %first, %second, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  scf.for %index = %lower to %upper step %step {
    %third = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %third, %first, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: index, %arg1: index, %arg2: index) {
// CHECK: scf.for
// CHECK: %[[TENSOR:.*]] = ascendc.reg_tensor
func.func @ignore_non_vector_function(%upper: index, %lower: index, %step: index) {
  scf.for %index = %lower to %upper step %step {
    %tensor = ascendc.reg_tensor : <f32>
  }
  return
}
