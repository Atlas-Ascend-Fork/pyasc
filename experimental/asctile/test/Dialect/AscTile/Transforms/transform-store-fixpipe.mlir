// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt --asctile-transform-store-fixpipe %s | FileCheck %s

// CHECK-LABEL: func.func @copy_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @copy_from_l0c_with_distrib(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) -> tensor<16x64xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_from_l0c_with_distrib(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) -> tensor<16x64xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
  return %0 : tensor<16x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @store_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  asctile.store %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @store_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %relu = asctile.relu %arg0 : tensor<16x16xf32, #asctile.local<L0C>>
  asctile.store %relu, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @store_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] {quantize} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %cast = asctile.cast <default> %arg0 : tensor<16x16xf32, #asctile.local<L0C>> to tensor<16x16xi8, #asctile.local<L0C>>
  asctile.store %cast, %arg1[%c0, %c0] : tensor<16x16xi8, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
  return
}

// CHECK-LABEL: func.func @copy_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %relu = asctile.relu %arg0 : tensor<16x16xf32, #asctile.local<L0C>>
  %0 = asctile.copy %relu[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @copy_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] {quantize} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xi8, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %cast = asctile.cast <default> %arg0 : tensor<16x16xf32, #asctile.local<L0C>> to tensor<16x16xi8, #asctile.local<L0C>>
  %0 = asctile.copy %cast[%c0, %c0] : tensor<16x16xi8, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
  return %0 : tensor<16x16xi8, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @non_l0c_copy(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @non_l0c_copy(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @non_l0c_store(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store %arg0, %arg1[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @non_l0c_store(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  asctile.store %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.global>
  return
}
