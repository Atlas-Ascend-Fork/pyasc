// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-cube-transpose-to-load -split-input-file -verify-diagnostics %s

func.func @transpose_l1_not_from_load(%arg0: tensor<32x64xf32, #asctile.global>) -> tensor<64x32xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-note@+1 {{unexpected source operation}}
  %copy_l1 = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-error@+1 {{Transpose on L1 is supported for load operation results only.}}
  %transpose = asctile.transpose %copy_l1, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  return %copy : tensor<64x32xf32, #asctile.local<L0A>>
}

// -----

func.func @transpose_l1_load_many_uses(%arg0: tensor<32x64xf32, #asctile.global>) -> (tensor<64x32xf32, #asctile.local<L0A>>, tensor<32x64xf32, #asctile.local<L0A>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  // expected-note@+1 {{load op with many uses}}
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-error@+1 {{Transpose should be the only user of load to L1 operation.}}
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  %copy0 = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  %copy1 = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0A>>
  return %copy0, %copy1 : tensor<64x32xf32, #asctile.local<L0A>>, tensor<32x64xf32, #asctile.local<L0A>>
}

// -----

func.func @transpose_l1_3d(%arg0: tensor<4x8x16xf32, #asctile.global>) -> tensor<16x8x4xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32, %c0_i32], %cst : tensor<4x8x16xf32, #asctile.global>, tensor<4x8x16xf32, #asctile.local<L1>>
  // expected-error@+1 {{Transpose op on Cube supports 2-dim tensors only.}}
  %transpose = asctile.transpose %load, [2 : i32, 1 : i32, 0 : i32] : tensor<4x8x16xf32, #asctile.local<L1>> to tensor<16x8x4xf32, #asctile.local<L1>>
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32, %c0_i32] : tensor<16x8x4xf32, #asctile.local<L1>>, tensor<16x8x4xf32, #asctile.local<L0A>>
  return %copy : tensor<16x8x4xf32, #asctile.local<L0A>>
}

// -----

func.func @transpose_l1_non_copy_user(%arg0: tensor<32x64xf32, #asctile.global>) -> (tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L1>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-error@+1 {{Transpose op on L1 should be used by copy operations only.}}
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  // expected-note@+1 {{unexpected user op}}
  return %copy, %transpose : tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L1>>
}

// -----

func.func @transpose_l1_copy_to_ub(%arg0: tensor<32x64xf32, #asctile.global>) -> tensor<64x32xf32, #asctile.local<UB>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-error@+1 {{Transpose op on L1 should be used by copy to L0A/L0B operations only.}}
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  // expected-note@+1 {{unsupported copy op}}
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<UB>>
  return %copy : tensor<64x32xf32, #asctile.local<UB>>
}

// -----

func.func @transpose_l1_mixed_copy_loc(%arg0: tensor<32x64xf32, #asctile.global>) -> (tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L0B>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  // expected-error@+1 {{All users of transpose op on L1 should have same location.}}
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  // expected-note@+1 {{copy op with different location}}
  %copy_a = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  %copy_b = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0B>>
  return %copy_a, %copy_b : tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L0B>>
}

// -----

func.func @transpose_l0a_not_from_copy(%arg0: tensor<32x128xf16, #asctile.global>) -> tensor<128x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  // expected-note@+1 {{unexpected source operation}}
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
  // expected-error@+1 {{Transpose on L0A/B is supported for copy operation results only.}}
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L1>> to tensor<128x32xf16, #asctile.local<L0A>>
  return %transpose : tensor<128x32xf16, #asctile.local<L0A>>
}

// -----

func.func @transpose_l0a_3d(%arg0: tensor<4x8x16xf16, #asctile.global>) -> tensor<16x8x4xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32, %c0_i32], %cst : tensor<4x8x16xf16, #asctile.global>, tensor<4x8x16xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32, %c0_i32] : tensor<4x8x16xf16, #asctile.local<L1>>, tensor<4x8x16xf16, #asctile.local<L0A>>
  // expected-error@+1 {{Transpose op on Cube supports 2-dim tensors only.}}
  %transpose = asctile.transpose %copy, [2 : i32, 1 : i32, 0 : i32] : tensor<4x8x16xf16, #asctile.local<L0A>> to tensor<16x8x4xf16, #asctile.local<L0A>>
  return %transpose : tensor<16x8x4xf16, #asctile.local<L0A>>
}

// -----

func.func @transpose_l0a_copy_not_from_load(%arg0: tensor<32x128xf16, #asctile.local<L1>>) -> tensor<128x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  // expected-error@+1 {{Copy operation on Cube expects load operation results only.}}
  %copy = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L0A>> to tensor<128x32xf16, #asctile.local<L0A>>
  return %transpose : tensor<128x32xf16, #asctile.local<L0A>>
}

// -----

func.func @transpose_l0a_load_non_copy_user(%arg0: tensor<32x128xf16, #asctile.global>) -> (tensor<128x32xf16, #asctile.local<L0A>>, tensor<32x128xf16, #asctile.local<L1>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  // expected-error@+1 {{Load op on L1 should be used by copy operations only.}}
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L0A>> to tensor<128x32xf16, #asctile.local<L0A>>
  // expected-note@+1 {{unexpected user op}}
  return %transpose, %load : tensor<128x32xf16, #asctile.local<L0A>>, tensor<32x128xf16, #asctile.local<L1>>
}

// -----

func.func @transpose_l0a_f32_partial(%arg0: tensor<32x128xf32, #asctile.global>) -> (tensor<128x32xf32, #asctile.local<L0A>>, tensor<32x128xf32, #asctile.local<L0A>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf32, #asctile.global>, tensor<32x128xf32, #asctile.local<L1>>
  %copy_t = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<L1>>, tensor<32x128xf32, #asctile.local<L0A>>
  // expected-error@+1 {{All copy operations should be used by transpose op only.}}
  %transpose = asctile.transpose %copy_t, [1 : i32, 0 : i32] : tensor<32x128xf32, #asctile.local<L0A>> to tensor<128x32xf32, #asctile.local<L0A>>
  // expected-note@+1 {{op without transpose op user}}
  %copy_n = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<L1>>, tensor<32x128xf32, #asctile.local<L0A>>
  return %transpose, %copy_n : tensor<128x32xf32, #asctile.local<L0A>>, tensor<32x128xf32, #asctile.local<L0A>>
}
