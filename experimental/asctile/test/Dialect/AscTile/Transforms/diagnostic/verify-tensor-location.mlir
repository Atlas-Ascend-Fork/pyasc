// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-verify-tensor-location -split-input-file -verify-diagnostics %s

func.func @valid_transpose_2d_ub(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<32x16xf32, #asctile.local<UB>> {
  %0 = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<UB>>
  return %0 : tensor<32x16xf32, #asctile.local<UB>>
}

// -----

func.func @invalid_transpose_l0c(%arg0: tensor<16x32xf32, #asctile.local<L0C>>) -> tensor<32x16xf32, #asctile.local<L0C>> {
  // expected-error@+1 {{input tensor location must be UB, L1, L0A, L0B, got L0C}}
  %0 = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<L0C>> to tensor<32x16xf32, #asctile.local<L0C>>
  return %0 : tensor<32x16xf32, #asctile.local<L0C>>
}

// -----

func.func @valid_load_ub(%arg0: tensor<128xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<128xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// -----

func.func @invalid_load_l0c(%arg0: tensor<128xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<L0C>> {
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{result tensor location must be UB, L1, L0A, L0B, BT, got L0C}}
  %0 = asctile.load %arg0[%c0] : tensor<128xf32, #asctile.global>, tensor<32xf32, #asctile.local<L0C>>
  return %0 : tensor<32xf32, #asctile.local<L0C>>
}

// -----

func.func @valid_store_ub(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<UB>>, %arg2: i32) {
  asctile.store %arg1, %arg0[%arg2] : tensor<32xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.global>
  return
}

// -----

func.func @invalid_store_l0a(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<L0A>>, %arg2: i32) {
  // expected-error@+1 {{src tensor location must be UB, L0C, got L0A}}
  asctile.store %arg1, %arg0[%arg2] : tensor<32xf32, #asctile.local<L0A>>, tensor<128xf32, #asctile.global>
  return
}

// -----

func.func @valid_matmul(%arg0: tensor<16x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>, %arg2: tensor<16xf32, #asctile.local<BT>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %0 = asctile.matmul %arg0, %arg1, %arg2 : tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>, tensor<16xf32, #asctile.local<BT>> -> tensor<16x16xf32, #asctile.local<L0C>>
  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
}

// -----

func.func @invalid_matmul_a(%arg0: tensor<16x16xf16, #asctile.local<UB>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  // expected-error@+1 {{input tensor location must be L0A, got UB}}
  %0 = asctile.matmul %arg0, %arg1 : tensor<16x16xf16, #asctile.local<UB>>, tensor<16x16xf16, #asctile.local<L0B>> -> tensor<16x16xf32, #asctile.local<L0C>>
  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
}

// -----

func.func @valid_matmul_acc(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf16, #asctile.local<L0A>>, %arg2: tensor<16x16xf16, #asctile.local<L0B>>) {
  asctile.matmul_acc %arg0, %arg1, %arg2 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
  return
}

// -----

func.func @invalid_matmul_acc(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<16x16xf16, #asctile.local<L0A>>, %arg2: tensor<16x16xf16, #asctile.local<L0B>>) {
  // expected-error@+1 {{acc tensor location must be L0C, got UB}}
  asctile.matmul_acc %arg0, %arg1, %arg2 : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
  return
}

// -----

func.func @invalid_accumulator_bias(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  // expected-error@+1 {{bias tensor location must be BT, got UB}}
  %0 = asctile.accumulator %arg0 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
}

// -----

func.func @valid_transpose_1d_ub(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.transpose %arg0, [0 : i32] : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// -----

func.func @invalid_transpose_1d_l1(%arg0: tensor<32xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<L1>> {
  // expected-error@+1 {{input tensor location must be UB, got L1}}
  %0 = asctile.transpose %arg0, [0 : i32] : tensor<32xf32, #asctile.local<L1>> to tensor<32xf32, #asctile.local<L1>>
  return %0 : tensor<32xf32, #asctile.local<L1>>
}

// -----

func.func @invalid_auto_result(%arg0: tensor<128xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<auto>> {
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{Unable to resolve location for result tensor(s) of the current operation}}
  %0 = asctile.load %arg0[%c0] : tensor<128xf32, #asctile.global>, tensor<32xf32, #asctile.local<auto>>
  return %0 : tensor<32xf32, #asctile.local<auto>>
}

// -----

func.func @valid_copy_with_distrib(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) -> tensor<16x64xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
  return %0 : tensor<16x64xf32, #asctile.local<UB>>
}

// -----

func.func @valid_copy_ub_l1_with_distrib(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] {distrib = #asctile.distrib_mode<join_by_n>} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<L1>>
  return %0 : tensor<16x128xf32, #asctile.local<L1>>
}

// -----

func.func @valid_dump_l1_tensor(%arg0: tensor<32xf32, #asctile.local<L1>>) {
  asctile.dump_tensor %arg0 : tensor<32xf32, #asctile.local<L1>>
  return
}

// -----

func.func @invalid_dump_l0a_tensor(%arg0: tensor<32xf32, #asctile.local<L0A>>) {
  // expected-error@+1 {{tensor tensor location must be L1, UB, got L0A}}
  asctile.dump_tensor %arg0 : tensor<32xf32, #asctile.local<L0A>>
  return
}
