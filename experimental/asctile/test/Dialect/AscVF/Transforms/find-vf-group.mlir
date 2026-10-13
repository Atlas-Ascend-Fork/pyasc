// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf-find-vf-group %s | FileCheck %s

// CHECK-LABEL: func.func @not_convert_if_one_operation
// CHECK-NOT: ascvf.vf_group
func.func @not_convert_if_one_operation(%vec: !ascendc.local_tensor<256xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  return
}

// CHECK-LABEL: func.func @same_group_if_cal_count_and_types_equals
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
func.func @same_group_if_cal_count_and_types_equals(%vec: !ascendc.local_tensor<256xf32>) {
  %c256_0 = arith.constant 256 : i32
  %c256_1 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256_0 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256_1 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_cal_count_not_equals
// CHECK: ascvf.vf_group
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
func.func @different_group_if_cal_count_not_equals(%vec0: !ascendc.local_tensor<256xf32>, %vec1: !ascendc.local_tensor<128xf32>) {
  %c256 = arith.constant 256 : i32
  %c128 = arith.constant 128 : i32
  ascendc.add_l2 %vec0, %vec0, %vec0, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %vec0, %vec0, %vec0, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.add_l2 %vec1, %vec1, %vec1, %c128 : !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, i32
  ascendc.mul_l2 %vec1, %vec1, %vec1, %c128 : !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_types_not_equals
// CHECK-NOT: ascvf.vf_group
func.func @different_group_if_types_not_equals(%vecf32: !ascendc.local_tensor<256xf32>, %vecf16: !ascendc.local_tensor<256xf16>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vecf32, %vecf32, %vecf32, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %vecf16, %vecf16, %vecf16, %c256 : !ascendc.local_tensor<256xf16>, !ascendc.local_tensor<256xf16>, !ascendc.local_tensor<256xf16>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_between_op_exist_other_op
// CHECK: ascvf.vf_group
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
func.func @different_group_if_between_op_exist_other_op(%vec: !ascendc.local_tensor<256xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  call @blank() : () -> ()
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  return
}

// CHECK-LABEL: func.func private @blank()
func.func private @blank()

// CHECK-LABEL: func.func @create_nested_vec_scope
func.func @create_nested_vec_scope(%dst0 : !ascendc.local_tensor<256xf32>, %dst1 : !ascendc.local_tensor<256xf32>, %src : !ascendc.local_tensor<256xf32>) {
  %c256_i32 = arith.constant 256 : i32
  %c1_idx = arith.constant 1 : index
// CHECK: scf.for
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
  scf.for %arg0 = %c1_idx to %c1_idx step %c1_idx {
    ascendc.add_l2 %dst0, %src, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
    ascendc.mul_l2 %dst1, %dst0, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
    %true = arith.constant true
// CHECK: scf.if
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
    scf.if %true {
      ascendc.add_l2 %dst0, %src, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
// CHECK: } else {
// CHECK: ascvf.vf_group
// CHECK-NOT: ascvf.vf_group
    } else {
      ascendc.add_l2 %dst0, %src, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
    }
  }
  return
}
