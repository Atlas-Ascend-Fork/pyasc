// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-fulfill-data-transfer %s | FileCheck %s

// CHECK-LABEL: func.func @valid_copy_l1_to_l0a(%arg0: tensor<16x16xf32, #asctile.local<L1>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
// CHECK-NEXT:  %0 = asctile.copy %arg0[%arg1, %arg2] : tensor<16x16xf32, #asctile.local<L1>>, tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:  return %0 : tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @valid_copy_l1_to_l0a(%arg0: tensor<16x16xf32, #asctile.local<L1>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%arg1, %arg2] : tensor<16x16xf32, #asctile.local<L1>>, tensor<16x16xf32, #asctile.local<L0A>>
  return %0 : tensor<16x16xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @invalid_copy_ub_to_l0a(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
// CHECK:       %0 = asctile.copy %arg0[%arg1, %arg2] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L1>>, tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:  return %1 : tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @invalid_copy_ub_to_l0a(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<L0A>>
  return %0 : tensor<16x16xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @invalid_copy_ub_to_bt(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<BT>> {
// CHECK:       %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.is_bias} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L1>>, tensor<16x16xf32, #asctile.local<BT>>
// CHECK-NEXT:  return %1 : tensor<16x16xf32, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @invalid_copy_ub_to_bt(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<BT>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<BT>>
  return %0 : tensor<16x16xf32, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @invalid_copy_ub_to_bt_with_distrib(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xf32, #asctile.local<BT>> {
// CHECK:       %0 = asctile.copy %arg0[%arg1] {asctile.is_bias, distrib = #asctile.distrib_mode<full_vec0>} : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.copy %0[%c0_i32] : tensor<16xf32, #asctile.local<L1>>, tensor<16xf32, #asctile.local<BT>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @invalid_copy_ub_to_bt_with_distrib(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xf32, #asctile.local<BT>> {
  %0 = asctile.copy %arg0[%arg1] {distrib = #asctile.distrib_mode<full_vec0>} : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<BT>>
  return %0 : tensor<16xf32, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @invalid_copy_ub_to_l0a_with_distrib(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
// CHECK:       %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<full_vec1>} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L1>>, tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:  return %1 : tensor<16x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @invalid_copy_ub_to_l0a_with_distrib(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<L0A>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<full_vec1>} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<L0A>>
  return %0 : tensor<16x16xf32, #asctile.local<L0A>>
}
