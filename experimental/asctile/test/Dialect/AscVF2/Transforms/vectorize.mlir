// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-vectorize -canonicalize -cse %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @elementwise_flatten_chain(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> (tensor<2x128xf32, #asctile.local<UB>>, tensor<2x128xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
// CHECK: %[[FIRST:.*]] = memref.alloca() : memref<2x128xf32, 26>
// CHECK: memref.reinterpret_cast %[[FIRST]]
// CHECK: affine.for %arg2 = 0 to 256 step 64 {
// CHECK: vector.transfer_read
// CHECK: arith.subf
// CHECK: vector.transfer_write
// CHECK: }
// CHECK: ascvf2.to_tensor
// CHECK: %[[SECOND:.*]] = memref.alloca() : memref<2x128xf32, 26>
// CHECK: memref.reinterpret_cast %[[SECOND]]
// CHECK: affine.for %arg2 = 0 to 256 step 64 {
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: vector.transfer_write
// CHECK: }
// CHECK: ascvf2.to_tensor
// CHECK: return
func.func @elementwise_flatten_chain(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> (tensor<2x128xf32, #asctile.local<UB>>, tensor<2x128xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
  %0 = arith.subf %arg0, %arg1 : tensor<2x128xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<2x128xf32, #asctile.local<UB>>
  return %0, %1 : tensor<2x128xf32, #asctile.local<UB>>, tensor<2x128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @integer_tail_and_mask(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> tensor<129xi32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 129 step 64 {
// CHECK: affine.apply #map
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.mask
// CHECK: arith.subi
// CHECK: vector.transfer_write
// CHECK: }
// CHECK: ascvf2.to_tensor
func.func @integer_tail_and_mask(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> tensor<129xi32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = arith.subi %arg0, %arg1 : tensor<129xi32, #asctile.local<UB>>
  return %0 : tensor<129xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @scalar_broadcast_flatten(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK-NOT: vector.broadcast
// CHECK: affine.for %arg2 = 0 to 1024 step 64 {
// CHECK: vector.transfer_read %{{.+}}[%c0], %{{.+}} {in_bounds = [true], permutation_map = #map1} : memref<1xf32, 26>, vector<64xf32>
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK: }
func.func @scalar_broadcast_flatten(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @row_broadcast_keeps_shape(%arg0: tensor<32x1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 32 {
// CHECK: affine.for %arg3 = 0 to 32 step 32 {
// CHECK: vector.broadcast %{{.+}} : vector<1xf32> to vector<32xf32>
// CHECK: arith.addf
// CHECK: }
// CHECK: }
func.func @row_broadcast_keeps_shape(%arg0: tensor<32x1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = asctile.broadcast %arg0 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @final_axis_reduction(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<2x1xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 2 {
// CHECK: affine.for %arg3 = 0 to 129 step 64 iter_args(%arg4 = %{{.+}}) -> (vector<64xf32>) {
// CHECK: affine.apply #map
// CHECK: vector.create_mask
// CHECK: arith.addf
// CHECK: }
// CHECK: }
// CHECK: ascvf2.reduce <sum>
// CHECK: vector.transfer_write
func.func @final_axis_reduction(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<2x1xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = arith.addf %arg0, %arg1 : tensor<2x129xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [1 : i32]} : tensor<2x129xf32, #asctile.local<UB>>, tensor<2x1xf32, #asctile.local<UB>>
  return %1 : tensor<2x1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @reduce_as_1d_flatten(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 258 step 64 {
// CHECK: affine.apply #map
// CHECK: vector.create_mask
// CHECK: arith.mulf
// CHECK: }
// CHECK: ascvf2.reduce <min>
// CHECK: vector.transfer_write
func.func @reduce_as_1d_flatten(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = arith.mulf %arg0, %arg1 : tensor<2x129xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <min> %0 : tensor<2x129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @asctile_scalar_and_unary_operations(%arg0: tensor<70xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<70xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 70 step 64 {
// CHECK: vector.mask %{{.+}} { ascvf2.muls %{{.+}}, %arg1 : vector<64xf32>, f32 }
// CHECK: vector.mask %{{.+}} { ascvf2.relu %{{.+}} : vector<64xf32> }
// CHECK: }
func.func @asctile_scalar_and_unary_operations(%arg0: tensor<70xf32, #asctile.local<UB>>, %scalar: f32) -> tensor<70xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = asctile.muls %arg0, %scalar : tensor<70xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<70xf32, #asctile.local<UB>>
  return %1 : tensor<70xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @asctile_integer_operations(%arg0: tensor<70xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<70xi32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
// CHECK: affine.for %arg2 = 0 to 70 step 64 {
// CHECK: vector.mask %{{.+}} { ascvf2.shls %{{.+}}, %arg1 : vector<64xi32>, i32 }
// CHECK: vector.mask %{{.+}} { ascvf2.not %{{.+}} : vector<64xi32> }
// CHECK: }
func.func @asctile_integer_operations(%arg0: tensor<70xi32, #asctile.local<UB>>, %shift: i32) -> tensor<70xi32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = asctile.shls %arg0, %shift : tensor<70xi32, #asctile.local<UB>>
  %1 = asctile.bitwise_not %0 : tensor<70xi32, #asctile.local<UB>>
  return %1 : tensor<70xi32, #asctile.local<UB>>
}

}
