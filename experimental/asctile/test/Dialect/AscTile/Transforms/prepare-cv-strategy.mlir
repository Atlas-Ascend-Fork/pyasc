// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-prepare-cv-strategy %s | FileCheck %s

// CHECK-LABEL: func.func @prepare_split_by_m_flow(
// CHECK-SAME:   %arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>,
// CHECK-SAME:   %arg2: tensor<32x128xf32, #asctile.local<UB>>, %arg3: tensor<32x128xf32, #asctile.global>,
// CHECK-SAME:   %arg4: tensor<32x128xf32, #asctile.global>, %arg5: tensor<32xf32, #asctile.global>) {
// CHECK:       %0 = asctile.cv_strategy <split_by_m> -> tensor<32x128xf32, #asctile.local<UB>> {
// CHECK-NEXT:    %1 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %2 = arith.mulf %1, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %3 = asctile.reduce <sum> %2 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>, dims = [1 : i32]}
// CHECK-NEXT:    %4 = asctile.broadcast %3 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    %5 = arith.addf %4, %arg2 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    asctile.dump_tensor %5 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
// CHECK-NEXT:    asctile.store %5, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    %6 = asctile.reshape %3 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16>}
// CHECK-NEXT:    asctile.store %6, %arg5[%c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16>}
// CHECK-NEXT:    %7 = asctile.reshape %6 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>}
// CHECK-NEXT:    %8 = asctile.broadcast %7 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    asctile.store %8, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    asctile.yield %5 : tensor<32x128xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
// CHECK-NEXT:  }
// CHECK-NEXT:  asctile.store %0, %arg4[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
func.func @prepare_split_by_m_flow(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x128xf32, #asctile.local<UB>>, %arg3: tensor<32x128xf32, #asctile.global>, %arg4: tensor<32x128xf32, #asctile.global>, %arg5: tensor<32xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %5 = asctile.cv_strategy <split_by_m> -> tensor<32x128xf32, #asctile.local<UB>> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.mulf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    %2 = asctile.reduce <sum> %1 {dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
    %3 = asctile.broadcast %2 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
    %4 = arith.addf %3, %arg2 : tensor<32x128xf32, #asctile.local<UB>>
    asctile.dump_tensor %4 : tensor<32x128xf32, #asctile.local<UB>>
    asctile.store %4, %arg3[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
    %6 = asctile.reshape %2 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
    asctile.store %6, %arg5[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    %7 = asctile.reshape %6 : tensor<32xf32, #asctile.local<UB>> to tensor<32x1xf32, #asctile.local<UB>>
    %8 = asctile.broadcast %7 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
    asctile.store %8, %arg3[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
    asctile.yield %4 : tensor<32x128xf32, #asctile.local<UB>>
  }
  asctile.store %5, %arg4[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @prepare_split_by_n_flow(
// CHECK-SAME:   %arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>,
// CHECK-SAME:   %arg2: tensor<32x64xf32, #asctile.local<UB>>, %arg3: tensor<32x64xf32, #asctile.global>,
// CHECK-SAME:   %arg4: tensor<64xf32, #asctile.global>) {
// CHECK:       %0 = asctile.cv_strategy <split_by_n> -> tensor<32x64xf32, #asctile.local<UB>> {
// CHECK-NEXT:    %2 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %3 = arith.addf %2, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %4 = asctile.reduce <sum> %3 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>, dims = [0 : i32]}
// CHECK-NEXT:    %5 = asctile.broadcast %4 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %6 = arith.mulf %5, %arg2 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    asctile.store %6, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %7 = asctile.reduce <sum> %3 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32>, dims = [0 : i32]}
// CHECK-NEXT:    asctile.store %7, %arg4[%c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32>}
// CHECK-NEXT:    %8 = asctile.reshape %7 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>}
// CHECK-NEXT:    %9 = asctile.broadcast %8 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    asctile.store %9, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    asctile.yield %6 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64>}
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = asctile.copy %0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:  asctile.dump_tensor %1 : tensor<32x64xf32, #asctile.local<L1>>
func.func @prepare_split_by_n_flow(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.local<UB>>, %arg3: tensor<32x64xf32, #asctile.global>, %arg4: tensor<64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %5 = asctile.cv_strategy <split_by_n> -> tensor<32x64xf32, #asctile.local<UB>> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.addf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    %2 = asctile.reduce <sum> %1 {dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.local<UB>>
    %3 = asctile.broadcast %2 : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
    %4 = arith.mulf %3, %arg2 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.store %4, %arg3[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
    %7 = asctile.reduce <sum> %1 {dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<64xf32, #asctile.local<UB>>
    asctile.store %7, %arg4[%c0_i32] : tensor<64xf32, #asctile.local<UB>>, tensor<64xf32, #asctile.global>
    %8 = asctile.reshape %7 : tensor<64xf32, #asctile.local<UB>> to tensor<1x64xf32, #asctile.local<UB>>
    %9 = asctile.broadcast %8 : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
    asctile.store %9, %arg3[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
    asctile.yield %4 : tensor<32x64xf32, #asctile.local<UB>>
  }
  %6 = asctile.copy %5[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
  asctile.dump_tensor %6 : tensor<32x64xf32, #asctile.local<L1>>
  return
}

// CHECK-LABEL: func.func @prepare_split_over_yield(
// CHECK:       %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
// CHECK-NEXT:    %5 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %5 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.reduce <sum> %1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>, dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.broadcast %2 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
// CHECK-NEXT:    %5 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    %6 = arith.addf %3, %5 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %6 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
// CHECK-NEXT:  }
// CHECK-NEXT:  asctile.store %4, %arg2[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
func.func @prepare_split_over_yield(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    %5 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %5 : tensor<32x64xf32, #asctile.local<UB>>
  }
  %1 = arith.addf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
  %2 = asctile.reduce <sum> %1 {dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
  %3 = asctile.broadcast %2 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
  %4 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    %6 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %7 = arith.addf %3, %6 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %7 : tensor<32x64xf32, #asctile.local<UB>>
  }
  asctile.store %4, %arg2[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @prepare_split_over_loop(
// CHECK:       %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg1) -> (tensor<64xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %3 = asctile.cv_strategy <split_by_n> -> tensor<64xf32, #asctile.local<UB>> {
// CHECK-NEXT:      %4 = asctile.copy %arg0[%c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32>} : tensor<64xf32, #asctile.local<L0C>>, tensor<64xf32, #asctile.local<UB>>
// CHECK-NEXT:      %5 = arith.mulf %4, %arg4 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32>} : tensor<64xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.yield %5 : tensor<64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64>}
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64>} %3 : tensor<64xf32, #asctile.local<UB>>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = asctile.reshape %0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>} : tensor<64xf32, #asctile.local<UB>> to tensor<1x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.broadcast %1 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  asctile.store %2, %arg2[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
func.func @prepare_split_over_loop(%arg0: tensor<64xf32, #asctile.local<L0C>>, %arg1: tensor<64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> tensor<64xf32, #asctile.local<UB>> {
    %1 = asctile.cv_strategy <split_by_n> -> tensor<64xf32, #asctile.local<UB>> {
      %2 = asctile.copy %arg0[%c0_i32] : tensor<64xf32, #asctile.local<L0C>>, tensor<64xf32, #asctile.local<UB>>
      %3 = arith.mulf %2, %arg3 : tensor<64xf32, #asctile.local<UB>>
      asctile.yield %3 : tensor<64xf32, #asctile.local<UB>>
    }
    scf.yield %1 : tensor<64xf32, #asctile.local<UB>>
  }
  %4 = asctile.reshape %0 : tensor<64xf32, #asctile.local<UB>> to tensor<1x64xf32, #asctile.local<UB>>
  %5 = asctile.broadcast %4 : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %5, %arg2[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @prepare_split_over_different_yields(
// CHECK:       %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg1) -> (tensor<32x128xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %2 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
// CHECK-NEXT:      %5 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:      %6 = math.absf %5 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:      asctile.yield %6 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = math.absf %2 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %4 = asctile.broadcast %3 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    scf.yield {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>} %4
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:  asctile.store %1, %arg2[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
func.func @prepare_split_over_different_yields(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x128xf32, #asctile.local<UB>>, %arg2: tensor<32x128xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> tensor<32x128xf32, #asctile.local<UB>> {
    %1 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
      %2 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
      %3 = math.absf %2 : tensor<32x64xf32, #asctile.local<UB>>
      asctile.yield %3 : tensor<32x64xf32, #asctile.local<UB>>
    }
    %4 = math.absf %1 : tensor<32x64xf32, #asctile.local<UB>>
    %5 = asctile.broadcast %4 : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
    scf.yield %5 : tensor<32x128xf32, #asctile.local<UB>>
  }
  %6 = arith.addf %0, %arg1 : tensor<32x128xf32, #asctile.local<UB>>
  asctile.store %6, %arg2[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
  return
}
