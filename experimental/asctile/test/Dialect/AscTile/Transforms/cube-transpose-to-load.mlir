// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-cube-transpose-to-load %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_l0a_basic(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l0} : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l0} : tensor<32x128xf16, #asctile.local<L1>>, tensor<128x32xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY]] : tensor<128x32xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l0a_basic(%arg0: tensor<32x128xf16, #asctile.global>) -> tensor<128x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L0A>> to tensor<128x32xf16, #asctile.local<L0A>>
  return %transpose : tensor<128x32xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @transpose_l0b_basic(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b_l0} : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l0} : tensor<64x32xf16, #asctile.local<L1>>, tensor<32x64xf16, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY]] : tensor<32x64xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @transpose_l0b_basic(%arg0: tensor<64x32xf16, #asctile.global>) -> tensor<32x64xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xf16, #asctile.local<L1>>, tensor<64x32xf16, #asctile.local<L0B>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<64x32xf16, #asctile.local<L0B>> to tensor<32x64xf16, #asctile.local<L0B>>
  return %transpose : tensor<32x64xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @transpose_ub_not_transformed(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<UB>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<32x128xf16, #asctile.local<UB>>, tensor<32x128xf16, #asctile.local<UB>>
// CHECK-NEXT: %[[TRANSPOSE:.*]] = asctile.transpose %[[COPY]], [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<UB>> to tensor<128x32xf16, #asctile.local<UB>>
// CHECK-NEXT: return %[[TRANSPOSE]] : tensor<128x32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @transpose_ub_not_transformed(%arg0: tensor<32x128xf16, #asctile.global>) -> tensor<128x32xf16, #asctile.local<UB>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<UB>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<UB>>, tensor<32x128xf16, #asctile.local<UB>>
  %transpose = asctile.transpose %copy, [1: i32, 0 : i32] : tensor<32x128xf16, #asctile.local<UB>> to tensor<128x32xf16, #asctile.local<UB>>
  return %transpose : tensor<128x32xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @transpose_l0a_f32(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<16x32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l0} : tensor<16x32xf32, #asctile.global>, tensor<16x32xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l0} : tensor<16x32xf32, #asctile.local<L1>>, tensor<32x16xf32, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY]] : tensor<32x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l0a_f32(%arg0: tensor<16x32xf32, #asctile.global>) -> tensor<32x16xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<16x32xf32, #asctile.global>, tensor<16x32xf32, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<16x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<L0A>> to tensor<32x16xf32, #asctile.local<L0A>>
  return %transpose : tensor<32x16xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @transpose_l1_a_fp32(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x64xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l1} : tensor<32x64xf32, #asctile.global>, tensor<64x32xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l1} : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY]] : tensor<64x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l1_a_fp32(%arg0: tensor<32x64xf32, #asctile.global>) -> tensor<64x32xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  return %copy : tensor<64x32xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @transpose_l1_b_fp32(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b_l1} : tensor<64x32xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l1} : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY]] : tensor<32x64xf32, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @transpose_l1_b_fp32(%arg0: tensor<64x32xf32, #asctile.global>) -> tensor<32x64xf32, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xf32, #asctile.global>, tensor<64x32xf32, #asctile.local<L1>>
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<64x32xf32, #asctile.local<L1>> to tensor<32x64xf32, #asctile.local<L1>>
  %copy = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
  return %copy : tensor<32x64xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @transpose_l1_a_multi_copy(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x64xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l1} : tensor<32x64xf32, #asctile.global>, tensor<64x32xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY0:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l1} : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
// CHECK-NEXT: %[[COPY1:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l1} : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY0]], %[[COPY1]] : tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l1_a_multi_copy(%arg0: tensor<32x64xf32, #asctile.global>) -> (tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L0A>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x64xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<32x64xf32, #asctile.local<L1>> to tensor<64x32xf32, #asctile.local<L1>>
  %copy0 = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  %copy1 = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<64x32xf32, #asctile.local<L1>>, tensor<64x32xf32, #asctile.local<L0A>>
  return %copy0, %copy1 : tensor<64x32xf32, #asctile.local<L0A>>, tensor<64x32xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @transpose_l1_b_multi_copy(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b_l1} : tensor<64x32xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY0:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l1} : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
// CHECK-NEXT: %[[COPY1:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l1} : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY0]], %[[COPY1]] : tensor<32x64xf32, #asctile.local<L0B>>, tensor<32x64xf32, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @transpose_l1_b_multi_copy(%arg0: tensor<64x32xf32, #asctile.global>) -> (tensor<32x64xf32, #asctile.local<L0B>>, tensor<32x64xf32, #asctile.local<L0B>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xf32, #asctile.global>, tensor<64x32xf32, #asctile.local<L1>>
  %transpose = asctile.transpose %load, [1 : i32, 0 : i32] : tensor<64x32xf32, #asctile.local<L1>> to tensor<32x64xf32, #asctile.local<L1>>
  %copy0 = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
  %copy1 = asctile.copy %transpose[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L1>>, tensor<32x64xf32, #asctile.local<L0B>>
  return %copy0, %copy1 : tensor<32x64xf32, #asctile.local<L0B>>, tensor<32x64xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @partial_transpose_l0a_f16(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l0} : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY_T:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l0} : tensor<32x128xf16, #asctile.local<L1>>, tensor<128x32xf16, #asctile.local<L0A>>
// CHECK-NEXT: %[[COPY_N:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY_T]], %[[COPY_N]] : tensor<128x32xf16, #asctile.local<L0A>>, tensor<32x128xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @partial_transpose_l0a_f16(%arg0: tensor<32x128xf16, #asctile.global>) -> (tensor<128x32xf16, #asctile.local<L0A>>, tensor<32x128xf16, #asctile.local<L0A>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
  %copy_t = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy_t, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L0A>> to tensor<128x32xf16, #asctile.local<L0A>>
  %copy_n = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  return %transpose, %copy_n : tensor<128x32xf16, #asctile.local<L0A>>, tensor<32x128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @partial_transpose_l0b_f16(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b_l0} : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY_T:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l0} : tensor<64x32xf16, #asctile.local<L1>>, tensor<32x64xf16, #asctile.local<L0B>>
// CHECK-NEXT: %[[COPY_N:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<64x32xf16, #asctile.local<L1>>, tensor<64x32xf16, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY_T]], %[[COPY_N]] : tensor<32x64xf16, #asctile.local<L0B>>, tensor<64x32xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @partial_transpose_l0b_f16(%arg0: tensor<64x32xf16, #asctile.global>) -> (tensor<32x64xf16, #asctile.local<L0B>>, tensor<64x32xf16, #asctile.local<L0B>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
  %copy_t = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xf16, #asctile.local<L1>>, tensor<64x32xf16, #asctile.local<L0B>>
  %transpose = asctile.transpose %copy_t, [1 : i32, 0 : i32] : tensor<64x32xf16, #asctile.local<L0B>> to tensor<32x64xf16, #asctile.local<L0B>>
  %copy_n = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xf16, #asctile.local<L1>>, tensor<64x32xf16, #asctile.local<L0B>>
  return %transpose, %copy_n : tensor<32x64xf16, #asctile.local<L0B>>, tensor<64x32xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @partial_transpose_l0a_bf16(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xbf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a_l0} : tensor<32x128xbf16, #asctile.global>, tensor<32x128xbf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY_T:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a_l0} : tensor<32x128xbf16, #asctile.local<L1>>, tensor<128x32xbf16, #asctile.local<L0A>>
// CHECK-NEXT: %[[COPY_N:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<32x128xbf16, #asctile.local<L1>>, tensor<32x128xbf16, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY_T]], %[[COPY_N]] : tensor<128x32xbf16, #asctile.local<L0A>>, tensor<32x128xbf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @partial_transpose_l0a_bf16(%arg0: tensor<32x128xbf16, #asctile.global>) -> (tensor<128x32xbf16, #asctile.local<L0A>>, tensor<32x128xbf16, #asctile.local<L0A>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : bf16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xbf16, #asctile.global>, tensor<32x128xbf16, #asctile.local<L1>>
  %copy_t = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xbf16, #asctile.local<L1>>, tensor<32x128xbf16, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy_t, [1 : i32, 0 : i32] : tensor<32x128xbf16, #asctile.local<L0A>> to tensor<128x32xbf16, #asctile.local<L0A>>
  %copy_n = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xbf16, #asctile.local<L1>>, tensor<32x128xbf16, #asctile.local<L0A>>
  return %transpose, %copy_n : tensor<128x32xbf16, #asctile.local<L0A>>, tensor<32x128xbf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @partial_transpose_l0b_bf16(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xbf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b_l0} : tensor<64x32xbf16, #asctile.global>, tensor<64x32xbf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY_T:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b_l0} : tensor<64x32xbf16, #asctile.local<L1>>, tensor<32x64xbf16, #asctile.local<L0B>>
// CHECK-NEXT: %[[COPY_N:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<64x32xbf16, #asctile.local<L1>>, tensor<64x32xbf16, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY_T]], %[[COPY_N]] : tensor<32x64xbf16, #asctile.local<L0B>>, tensor<64x32xbf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @partial_transpose_l0b_bf16(%arg0: tensor<64x32xbf16, #asctile.global>) -> (tensor<32x64xbf16, #asctile.local<L0B>>, tensor<64x32xbf16, #asctile.local<L0B>>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : bf16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xbf16, #asctile.global>, tensor<64x32xbf16, #asctile.local<L1>>
  %copy_t = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xbf16, #asctile.local<L1>>, tensor<64x32xbf16, #asctile.local<L0B>>
  %transpose = asctile.transpose %copy_t, [1 : i32, 0 : i32] : tensor<64x32xbf16, #asctile.local<L0B>> to tensor<32x64xbf16, #asctile.local<L0B>>
  %copy_n = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xbf16, #asctile.local<L1>>, tensor<64x32xbf16, #asctile.local<L0B>>
  return %transpose, %copy_n : tensor<32x64xbf16, #asctile.local<L0B>>, tensor<64x32xbf16, #asctile.local<L0B>>
}
