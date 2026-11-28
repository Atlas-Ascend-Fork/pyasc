// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-insert-vf-sync %s | FileCheck %s

// CHECK-LABEL: func.func @insert_barrier_between_dependent_loops(%arg0: tensor<8xf32, #asctile.local<UB>>, %arg1: vector<4xf32>, %arg2: f32) attributes {ascvf2.vector_func} {
// CHECK: %[[BUFFER:.*]] = ascvf2.to_buffer %arg0
// CHECK: affine.for
// CHECK-NEXT: vector.transfer_write
// CHECK: ascvf2.barrier
// CHECK: affine.for
// CHECK: vector.transfer_read
func.func @insert_barrier_between_dependent_loops(%tensor: tensor<8xf32, #asctile.local<UB>>, %value: vector<4xf32>, %padding: f32) attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @no_barrier_for_internal_dependency(%arg0: tensor<8xf32, #asctile.local<UB>>, %arg1: vector<4xf32>, %arg2: f32) attributes {ascvf2.vector_func} {
// CHECK: affine.for
// CHECK-NEXT: vector.transfer_write
// CHECK-NEXT: vector.transfer_read
// CHECK-NOT: ascvf2.barrier
func.func @no_barrier_for_internal_dependency(%tensor: tensor<8xf32, #asctile.local<UB>>, %value: vector<4xf32>, %padding: f32) attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @no_barrier_for_independent_roots(%arg0: tensor<8xf32, #asctile.local<UB>>, %arg1: tensor<8xf32, #asctile.local<UB>>, %arg2: vector<4xf32>, %arg3: f32) attributes {ascvf2.vector_func} {
// CHECK-NOT: ascvf2.barrier
func.func @no_barrier_for_independent_roots(%first: tensor<8xf32, #asctile.local<UB>>, %second: tensor<8xf32, #asctile.local<UB>>, %value: vector<4xf32>, %padding: f32) attributes {ascvf2.vector_func} {
  %firstBuffer = ascvf2.to_buffer %first : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %secondBuffer = ascvf2.to_buffer %second : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %firstBuffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %secondBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @reuse_barrier_for_alias_chain(%arg0: tensor<8xf32, #asctile.local<UB>>, %arg1: vector<4xf32>, %arg2: f32) attributes {ascvf2.vector_func} {
// CHECK: %[[BUFFER:.*]] = ascvf2.to_buffer %arg0
// CHECK: %[[ALIASED_TENSOR:.*]] = ascvf2.to_tensor %[[BUFFER]]
// CHECK: %[[ALIASED_BUFFER:.*]] = ascvf2.to_buffer %[[ALIASED_TENSOR]]
// CHECK: affine.for
// CHECK-NEXT: vector.transfer_write
// CHECK: ascvf2.barrier
// CHECK: affine.for
// CHECK: vector.transfer_read
// CHECK-NOT: ascvf2.barrier
func.func @reuse_barrier_for_alias_chain(%tensor: tensor<8xf32, #asctile.local<UB>>, %value: vector<4xf32>, %padding: f32) attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %aliasedTensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %aliasedBuffer = ascvf2.to_buffer %aliasedTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  ascvf2.barrier
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %aliasedBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @sync_after_reduction_store(%arg0: tensor<8xf32, #asctile.local<UB>>, %arg1: vector<4xf32>, %arg2: f32, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK: %[[REDUCED:.*]] = ascvf2.reduce <max> %arg1
// CHECK: %[[BUFFER:.*]] = ascvf2.to_buffer %arg0
// CHECK: vector.transfer_write %[[REDUCED]]
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: vector.transfer_read
func.func @sync_after_reduction_store(%tensor: tensor<8xf32, #asctile.local<UB>>, %vector: vector<4xf32>, %padding: f32, %index: index) attributes {ascvf2.vector_func} {
  %reduced = ascvf2.reduce <max> %vector [0 : i32] : vector<4xf32> to vector<1xf32>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  vector.transfer_write %reduced, %buffer[%index] {in_bounds = [true]} : vector<1xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<1xf32>
  arith.addf %loaded, %loaded : vector<1xf32>
  return
}
