// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -ascvf-lower-to-reg %s | FileCheck %s

// CHECK-LABEL: func.func @translate_unary_l2
// CHECK: ascendc.abs_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.exp_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.ln_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.neg_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.not_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.relu_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @translate_unary_l2(%arg0: i32, %arg1: !ascendc.local_tensor<1024xf32>) {
  ascvf.vf_group dst(%arg1 : !ascendc.local_tensor<1024xf32>) src(%arg1 : !ascendc.local_tensor<1024xf32>) {
    ascendc.abs_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.exp_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.ln_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.neg_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.not_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.relu_l2 %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  } {groupType = !ascendc.local_tensor<1024xf32>}
  return
}

// CHECK-LABEL: func.func @translate_binary_l2
// CHECK: ascendc.add_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.and_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.div_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.sub_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.max_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.min_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.mul_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.mul_add_dst_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.or_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.prelu_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @translate_binary_l2(%arg0: i32, %arg1: !ascendc.local_tensor<1024xf32>) {
  ascvf.vf_group dst(%arg1 : !ascendc.local_tensor<1024xf32>) src(%arg1 : !ascendc.local_tensor<1024xf32>) {
    ascendc.add_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.and_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.div_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.sub_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.max_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.min_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.mul_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.mul_add_dst_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.or_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
    ascendc.prelu_l2 %arg1, %arg1, %arg1, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  } {groupType = !ascendc.local_tensor<1024xf32>}
  return
}

// CHECK-LABEL: func.func @translate_vec_scalar_l2
// CHECK: ascendc.add_reg
// CHECK: ascendc.leaky_relu_reg
// CHECK: ascendc.leaky_relu_reg
// CHECK: ascendc.mul_reg
// CHECK: ascendc.max_reg
// CHECK: ascendc.min_reg
// CHECK: ascendc.shift_lefts_reg
// CHECK: ascendc.shift_rights_reg
func.func @translate_vec_scalar_l2(%arg0: i32, %arg1: !ascendc.local_tensor<1024xf32>, %arg2: f32) {
  ascvf.vf_group dst(%arg1 : !ascendc.local_tensor<1024xf32>) src(%arg1 : !ascendc.local_tensor<1024xf32>) {
    ascendc.adds_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.leaky_relu_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.leaky_relu_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.muls_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.maxs_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.mins_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.shift_left_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
    ascendc.shift_right_l2 %arg1, %arg1, %arg2, %arg0 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, f32, i32
  } {groupType = !ascendc.local_tensor<1024xf32>}
  return
}

// CHECK-LABEL: func.func @translate_mask_ops
// CHECK: ascendc.compare_scalar_reg
// CHECK: ascendc.select_reg
func.func @translate_mask_ops(%arg0: i64, %arg1: !ascendc.local_tensor<96xui8>, %arg2: !ascendc.local_tensor<768xf16>, %arg3: f16) {
  ascvf.vf_group dst(%arg1, %arg2 : !ascendc.local_tensor<96xui8>, !ascendc.local_tensor<768xf16>) src(%arg2 : !ascendc.local_tensor<768xf16>) {
    ascendc.compare_scalar_l2 %arg1, %arg2, %arg3, %arg0 {asc.reuse_group = 0 : i64, cmpMode = 5 : i64} : !ascendc.local_tensor<96xui8>, !ascendc.local_tensor<768xf16>, f16, i64
    ascendc.select_l2 %arg2, %arg1, %arg2, %arg2, %arg0 {asc.reuse_group = 0 : i64, selMode = 2 : i32} : !ascendc.local_tensor<768xf16>, !ascendc.local_tensor<96xui8>, !ascendc.local_tensor<768xf16>, !ascendc.local_tensor<768xf16>, i64
  } {groupType = !ascendc.local_tensor<768xf16>}
  return
}

// -----

// CHECK-LABEL: func.func @known_vec_len
// CHECK: ascvf.vf_group
// CHECK-NOT: ascendc.get_vec_len
module attributes {asc.vf_vec_len = 256 : i32} {
  func.func @known_vec_len(%arg5: !ascendc.local_tensor<1024xf32>, %arg6: i64) {
    ascvf.vf_group dst(%arg5 : !ascendc.local_tensor<1024xf32>) src(%arg5 : !ascendc.local_tensor<1024xf32>) {
      ascendc.exp_l2 %arg5, %arg5, %arg6 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i64
      ascendc.exp_l2 %arg5, %arg5, %arg6 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i64
    } {groupType = !ascendc.local_tensor<1024xf32>}
    return
  }
}
