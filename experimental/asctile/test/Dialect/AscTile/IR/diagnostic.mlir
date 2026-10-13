// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -verify-diagnostics %s

// expected-error@below {{global tensor must have at least one dimension}}
func.func private @global_tensor_0d(tensor<f32, #asctile.global>)

// -----

// expected-error@below {{local tensor must have at least one dimension}}
func.func private @local_tensor_0d(tensor<f32, #asctile.local<UB>>)

// -----

// expected-error@below {{local tensor must have static shape}}
func.func private @local_tensor_dynamic_shape(tensor<32x?xf32, #asctile.local<L1>>)

// -----

func.func @tensor_wrong_sizes(%arg0: memref<*xf32, 22>, %arg1: i32) -> tensor<?x?xf32, #asctile.global> {
  // expected-error@below {{must have value in 'sizes' for each dynamic dimension}}
  %0 = asctile.tensor %arg0(%arg1) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  return %0 : tensor<?x?xf32, #asctile.global>
}

// -----

func.func @dim_index_exceeds_rank(%arg0: memref<*xf32, 22>) -> i32 {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x64xf32, #asctile.global>
  // expected-error@below {{'index' must not exceed the tensor rank}}
  %d = asctile.dim %0, 2 : tensor<32x64xf32, #asctile.global>
  return %d : i32
}

// -----

func.func @load_real_shape_non_ub(%arg0: tensor<64xf16, #asctile.global>, %arg1: f16, %arg2: i32) -> tensor<64xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{only supports 'real_shape' with UB location, got L1}}
  %0 = asctile.load %arg0[%c0], %arg1, (%arg2) : tensor<64xf16, #asctile.global>, tensor<64xf16, #asctile.local<L1>>
  return %0 : tensor<64xf16, #asctile.local<L1>>
}

// -----

func.func @load_real_shape_size_mismatch(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: f16, %arg2: i32) -> tensor<64x64xf16, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{real_shape must have same size as tensor shape}}
  %0 = asctile.load %arg0[%c0, %c0], %arg1, (%arg2) : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<UB>>
  return %0 : tensor<64x64xf16, #asctile.local<UB>>
}

// -----

func.func @load_real_shape_exceeds_tile(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: f16) -> tensor<64x64xf16, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  %0 = asctile.load %arg0[%c0, %c0], %arg1, (%c128, %c128) : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<UB>>
  return %0 : tensor<64x64xf16, #asctile.local<UB>>
}

// -----

func.func @store_real_shape_size_mismatch(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<?x?xf32, #asctile.global>, %arg2: i32) {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{real_shape must have same size as tensor shape}}
  asctile.store %arg0, %arg1[%c0, %c0], (%arg2) : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @store_real_shape_exceeds_tile(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<?x?xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  asctile.store %arg0, %arg1[%c0, %c0], (%c32, %c32) : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @store_fixpipe_type_mismatch(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{failed to verify that all of {base, value} have same element type}}
  asctile.store_fixpipe %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
  return
}

// -----

func.func @store_fixpipe_real_shape_exceeds(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<?x?xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  asctile.store_fixpipe %arg0, %arg1[%c0, %c0], (%c32, %c32) : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @copy_fixpipe_type_mismatch(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{failed to verify that all of {base, result} have same element type}}
  %0 = asctile.copy_fixpipe %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
  return %0 : tensor<16x16xi8, #asctile.local<UB>>
}

// -----

func.func @copy_split_by_n_ub_to_l1(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x64xf32, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{'distrib' mode split_by_n is only supported when copying from L0C to UB}}
  %0 = asctile.copy %arg0[%c0, %c0] {distrib = #asctile.distrib_mode<split_by_n>} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x64xf32, #asctile.local<L1>>
  return %0 : tensor<16x64xf32, #asctile.local<L1>>
}

// -----

func.func @copy_join_by_m_l0c_to_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) -> tensor<32x64xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{'distrib' mode join_by_m is only supported when copying from UB to L1}}
  %0 = asctile.copy %arg0[%c0, %c0] {distrib = #asctile.distrib_mode<join_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  return %0 : tensor<32x64xf32, #asctile.local<UB>>
}

// -----

func.func @accumulator_bias_shape_mismatch(%arg0: tensor<16xf32, #asctile.local<BT>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{bias shape must match result's second dimension}}
  %0 = asctile.accumulator %arg0 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<16xf32, #asctile.local<BT>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @cube_group_yield_type_mismatch(%arg0: tensor<32xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<UB>> {
  // expected-error@below {{yield operand type at index 0 ('tensor<32xf32, #asctile.local<L1>>') does not match result type ('tensor<32xf32, #asctile.local<UB>>')}}
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<L1>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<L1>>
    asctile.yield %1 : tensor<32xf32, #asctile.local<L1>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// -----

func.func @vector_group_yield_count_mismatch(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  // expected-error@below {{number of yield operands (2) must match number of results (1)}}
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1, %1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// -----

func.func @cv_strategy_yield_count_mismatch(%arg0: tensor<32x64xf32, #asctile.local<UB>>) -> tensor<32x64xf32, #asctile.local<UB>> {
  // expected-error@below {{number of yield operands (2) must match number of results (1)}}
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    asctile.yield %arg0, %arg0 : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<UB>>
  }
  return %0 : tensor<32x64xf32, #asctile.local<UB>>
}

// -----

func.func @cv_strategy_yield_type_mismatch(%arg0: tensor<32x64xf32, #asctile.local<UB>>) -> tensor<32x64xf32, #asctile.local<L1>> {
  // expected-error@below {{yield operand #0 type 'tensor<32x64xf32, #asctile.local<UB>>' does not match result type 'tensor<32x64xf32, #asctile.local<L1>>'}}
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<L1>> {
    asctile.yield %arg0 : tensor<32x64xf32, #asctile.local<UB>>
  }
  return %0 : tensor<32x64xf32, #asctile.local<L1>>
}

// -----

func.func @cv_strategy_yield_rank_mismatch(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<1x32xf32, #asctile.local<UB>> {
  // expected-error@below {{yield operand #0 type 'tensor<32xf32, #asctile.local<UB>>' does not match result type 'tensor<1x32xf32, #asctile.local<UB>>'}}
  %0 = asctile.cv_strategy <split_by_m> -> tensor<1x32xf32, #asctile.local<UB>> {
    asctile.yield %arg0 : tensor<32xf32, #asctile.local<UB>>
  }
  return %0 : tensor<1x32xf32, #asctile.local<UB>>
}
