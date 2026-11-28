// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-reorder-grouped-ops %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_allocations(%arg0: memref<4xf32, 26>, %arg1: f32, %arg2: index) -> memref<4xf32, 26> attributes {ascvf2.vector_func}
// CHECK: %[[FIRST:.*]] = memref.alloca() : memref<4xf32, 26>
// CHECK: %[[SECOND:.*]] = memref.alloca() : memref<4xf32, 26>
// CHECK: memref.store %arg1, %arg0[%arg2] : memref<4xf32, 26>
// CHECK: memref.store %arg1, %[[FIRST]][%arg2] : memref<4xf32, 26>
// CHECK: return %[[SECOND]]
func.func @hoist_allocations(%arg0: memref<4xf32, 26>, %arg1: f32, %arg2: index) -> memref<4xf32, 26> attributes {ascvf2.vector_func} {
  memref.store %arg1, %arg0[%arg2] : memref<4xf32, 26>
  %first = memref.alloca() : memref<4xf32, 26>
  memref.store %arg1, %first[%arg2] : memref<4xf32, 26>
  %second = memref.alloca() : memref<4xf32, 26>
  return %second : memref<4xf32, 26>
}

// CHECK-LABEL: func.func @sink_to_tensor_before_first_user(%arg0: memref<4xf32, 26>, %arg1: memref<4xf32, 26>, %arg2: f32, %arg3: index) -> tensor<4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func}
// CHECK: memref.store %arg2, %arg1[%arg3] : memref<4xf32, 26>
// CHECK: %[[TENSOR:.*]] = ascvf2.to_tensor %arg0 : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
// CHECK: return %[[TENSOR]]
func.func @sink_to_tensor_before_first_user(%arg0: memref<4xf32, 26>, %arg1: memref<4xf32, 26>, %arg2: f32, %arg3: index) -> tensor<4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %tensor = ascvf2.to_tensor %arg0 : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  memref.store %arg2, %arg1[%arg3] : memref<4xf32, 26>
  return %tensor : tensor<4xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @move_pure_op_before_side_effect(%arg0: memref<4xf32, 26>, %arg1: index, %arg2: f32) -> f32 attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = arith.addf %arg2, %arg2 : f32
// CHECK-NEXT: memref.store %arg2, %arg0[%arg1] : memref<4xf32, 26>
// CHECK: return %{{.+}}
func.func @move_pure_op_before_side_effect(%arg0: memref<4xf32, 26>, %arg1: index, %arg2: f32) -> f32 attributes {ascvf2.vector_func} {
  memref.store %arg2, %arg0[%arg1] : memref<4xf32, 26>
  %result = arith.addf %arg2, %arg2 : f32
  return %result : f32
}

// CHECK-LABEL: func.func @hoist_static_reinterpret_cast(%arg0: memref<f32, 26>, %arg1: f32) -> memref<f32, 26> attributes {ascvf2.vector_func}
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [], strides: [] : memref<f32, 26> to memref<f32, 26>
// CHECK: memref.store %arg1, %arg0[] : memref<f32, 26>
// CHECK: return %[[CAST]]
func.func @hoist_static_reinterpret_cast(%arg0: memref<f32, 26>, %arg1: f32) -> memref<f32, 26> attributes {ascvf2.vector_func} {
  memref.store %arg1, %arg0[] : memref<f32, 26>
  %cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [], strides: [] : memref<f32, 26> to memref<f32, 26>
  return %cast : memref<f32, 26>
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: memref<4xf32, 26>, %arg1: f32, %arg2: index) -> memref<4xf32, 26> {
// CHECK: memref.store
// CHECK: %[[ALLOC:.*]] = memref.alloca()
// CHECK: return %[[ALLOC]]
func.func @ignore_non_vector_function(%arg0: memref<4xf32, 26>, %arg1: f32, %arg2: index) -> memref<4xf32, 26> {
  memref.store %arg1, %arg0[%arg2] : memref<4xf32, 26>
  %alloc = memref.alloca() : memref<4xf32, 26>
  return %alloc : memref<4xf32, 26>
}
