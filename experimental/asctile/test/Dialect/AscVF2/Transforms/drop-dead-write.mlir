// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-drop-dead-write -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @remove_exact_overwrite(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = memref.alloca() : memref<8xf32, 26>
// CHECK: %{{.+}} = ascvf2.to_tensor %{{.+}} : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
// CHECK: vector.transfer_write %arg1, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: return %{{.+}} : tensor<8xf32, #asctile.local<UB>>
func.func @remove_exact_overwrite(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %arg1, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @remove_overwrite_in_loop(%{{.+}}: vector<4xf32>, %{{.+}}: vector<4xf32>) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func}
// CHECK: affine.for %{{.+}} = 0 to 8 step 4 {
// CHECK-NEXT: vector.transfer_write %arg1, %{{.+}}[%{{.+}}] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK-NOT: vector.transfer_write
func.func @remove_overwrite_in_loop(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %arg0, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    vector.transfer_write %arg1, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @keep_observed_write(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: %{{.+}} = vector.transfer_read %{{.+}}[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK: vector.transfer_write %arg1, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @keep_observed_write(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %buffer = memref.alloca() : memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%arg2], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  vector.transfer_write %arg1, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_write_across_barrier(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func}
// CHECK: vector.transfer_write %arg0, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: ascvf2.barrier
// CHECK: vector.transfer_write %arg1, %{{.+}}[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK: return %{{.+}} : tensor<8xf32, #asctile.local<UB>>
func.func @keep_write_across_barrier(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  ascvf2.barrier
  vector.transfer_write %arg1, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @remove_unread_private_temporary_writes(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index, %arg3: index) attributes {ascvf2.vector_func}
// CHECK-NOT: vector.transfer_write
// CHECK: return
func.func @remove_unread_private_temporary_writes(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
  %buffer = memref.alloca() : memref<8xf32, 26>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %arg1, %buffer[%arg3] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>>
// CHECK-COUNT-2: vector.transfer_write
// CHECK: return %{{.+}} : tensor<8xf32, #asctile.local<UB>>
func.func @ignore_non_vector_function(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: index) -> tensor<8xf32, #asctile.local<UB>> {
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  vector.transfer_write %arg0, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %arg1, %buffer[%arg2] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}
