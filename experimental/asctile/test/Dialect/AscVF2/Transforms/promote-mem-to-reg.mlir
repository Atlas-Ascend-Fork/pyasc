// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-promote-mem-to-reg %s | FileCheck %s

#map = affine_map<(d0) -> (0)>

// CHECK-LABEL: func.func @forward_exact_write(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = ascvf2.to_buffer %{{.+}} : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
// CHECK-NEXT: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK-NOT: vector.transfer_read
// CHECK: %{{.+}} = arith.addf %arg0, %arg0 : vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @forward_exact_write(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return %sum : vector<4xf32>
}

// CHECK-LABEL: func.func @forward_scalar_broadcast(%arg0: vector<1xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<1xf32>, memref<8xf32, 26>
// CHECK-NEXT: %{{.+}} = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @forward_scalar_broadcast(%arg0: vector<1xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<1xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%arg2], %arg3 {in_bounds = [true], permutation_map = #map} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @forward_through_alias(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK-NOT: vector.transfer_read
// CHECK: return %arg0 : vector<4xf32>
func.func @forward_through_alias(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %aliasedTensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %aliasedBuffer = ascvf2.to_buffer %aliasedTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %aliasedBuffer[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_read_after_overlapping_write(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: tensor<8xf32, #asctile.local<UB>>, %arg3: index, %arg4: index, %arg5: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK-COUNT-2: vector.transfer_write
// CHECK: %{{.+}} = vector.transfer_read %{{.+}}[%arg3], %arg5 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @keep_read_after_overlapping_write(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: tensor<8xf32, #asctile.local<UB>>, %arg3: index, %arg4: index, %arg5: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg2 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg3] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %arg1, %buffer[%arg4] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%arg3], %arg5 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_read_after_barrier(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: ascvf2.barrier
// CHECK: %{{.+}} = vector.transfer_read %{{.+}}[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
func.func @keep_read_after_barrier(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  ascvf2.barrier
  %loaded = vector.transfer_read %buffer[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @forward_inside_loop(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = affine.for %{{.+}} = 0 to 8 step 4 iter_args(%{{.+}} = %arg0) -> (vector<4xf32>) {
// CHECK-NEXT: vector.transfer_write %arg0, %{{.+}}[%{{.+}}] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK-NEXT: affine.yield %arg0 : vector<4xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: %{{.+}} = arith.addf %{{.+}}, %{{.+}} : vector<4xf32>
func.func @forward_inside_loop(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %result = affine.for %index = 0 to 8 step 4 iter_args(%acc = %arg0) -> vector<4xf32> {
    vector.transfer_write %arg0, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %loaded = vector.transfer_read %buffer[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    affine.yield %loaded : vector<4xf32>
  }
  %sum = arith.addf %result, %result : vector<4xf32>
  return %sum : vector<4xf32>
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32>
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: %{{.+}} = vector.transfer_read %{{.+}}[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
func.func @ignore_non_vector_function(%arg0: vector<4xf32>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: index, %arg3: f32) -> vector<4xf32> {
  %buffer = ascvf2.to_buffer %arg1 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}
