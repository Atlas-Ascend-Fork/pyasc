// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-unscalarize-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @skip_reduce_standalone(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
// CHECK-NEXT:  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:  return %0 : f32
// CHECK-NEXT:}
func.func @skip_reduce_standalone(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @vectorize_without_scalar_block(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<8xf32, #asctile.local<UB>> {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<8xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %1 : tensor<8xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @vectorize_without_scalar_block(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<8xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %splat = tensor.splat %0 : tensor<8xf32, #asctile.local<UB>>
  return %splat : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @unscalarize_multiple_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %2 = arith.addf %arg0, %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %3 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %4 = arith.subf %arg0, %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %2, %4 : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @unscalarize_multiple_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  %splat0 = tensor.splat %0 : tensor<16xf32, #asctile.local<UB>>
  %1 = arith.addf %arg0, %splat0 : tensor<16xf32, #asctile.local<UB>>
  %splat1 = tensor.splat %0 : tensor<16xf32, #asctile.local<UB>>
  %2 = arith.subf %arg0, %splat1 : tensor<16xf32, #asctile.local<UB>>
  return %1, %2 : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @unscalarize_chain_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %2 = arith.addf %arg0, %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %3 = asctile.reduce_as_1d <min> %2 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %4 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %5 = arith.subf %arg0, %4 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %5 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @unscalarize_chain_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  %splat1 = tensor.splat %0 : tensor<16xf32, #asctile.local<UB>>
  %1 = arith.addf %arg0, %splat1: tensor<16xf32, #asctile.local<UB>>
  %2 = asctile.reduce_as_1d <min> %1 : tensor<16xf32, #asctile.local<UB>>, f32
  %splat2 = tensor.splat %0 : tensor<16xf32, #asctile.local<UB>>
  %3 = arith.subf %arg0, %splat2 : tensor<16xf32, #asctile.local<UB>>
  return %3 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_with_scalar_block(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
// CHECK-NEXT:   %cst = arith.constant 1.280000e+02 : f32
// CHECK-NEXT:   %splat = tensor.splat %cst : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = arith.divf %0, %splat : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %2 = arith.addf %1, %splat : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %3 = math.sqrt %2 : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %4 = asctile.broadcast %3 : tensor<1xf32, #asctile.local<UB>> to tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %4 : tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @vectorize_with_scalar_block(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %cst = arith.constant 1.280000e+02 : f32
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %1 = arith.divf %0, %cst : f32
  %2 = arith.addf %1, %cst : f32
  %3 = math.sqrt %2 : f32
  %splat = tensor.splat %3 : tensor<128xf32, #asctile.local<UB>>
  return %splat : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_different_reduce(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = asctile.reduce_as_1d <sum> %arg1 : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %2 = arith.addf %0, %1 : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %3 = asctile.broadcast %2 : tensor<1xf32, #asctile.local<UB>> to tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %3 : tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @vectorize_different_reduce(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %1 = asctile.reduce_as_1d <sum> %arg1 : tensor<128xf32, #asctile.local<UB>>, f32
  %2 = arith.addf %0, %1 : f32
  %splat = tensor.splat %2 : tensor<128xf32, #asctile.local<UB>>
  return %splat : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @dont_vectorize_if_not_splat_ends(%arg0: tensor<128xf32, #asctile.local<UB>>) -> f32 {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:   %1 = arith.addf %0, %0 : f32
// CHECK-NEXT:   return %1 : f32
// CHECK-NEXT: }
func.func @dont_vectorize_if_not_splat_ends(%arg0: tensor<128xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %1 = arith.addf %0, %0 : f32
  return %1 : f32
}

// CHECK-LABEL: func.func @partial_convert_vectorized_ops(%arg0: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, f32) {
// CHECK-NEXT:   %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %1 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:   %splat = tensor.splat %1 : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %2 = arith.addf %0, %splat : tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:   %3 = asctile.broadcast %2 : tensor<1xf32, #asctile.local<UB>> to tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %3, %1 : tensor<128xf32, #asctile.local<UB>>, f32
// CHECK-NEXT: }
func.func @partial_convert_vectorized_ops(%arg0: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, f32) {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %1 = asctile.reduce_as_1d <sum> %arg0 : tensor<128xf32, #asctile.local<UB>>, f32
  %2 = arith.addf %0, %1 : f32
  %splat = tensor.splat %2 : tensor<128xf32, #asctile.local<UB>>
  return %splat, %1 : tensor<128xf32, #asctile.local<UB>>, f32
}
