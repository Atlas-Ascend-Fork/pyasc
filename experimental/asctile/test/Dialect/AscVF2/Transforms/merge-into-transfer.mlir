// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-merge-into-transfer %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (0)>
// CHECK: #map1 = affine_map<(d0, d1) -> (0)>

module attributes {asc.vf_vec_len = 16 : i32} {

// CHECK-LABEL: func.func @merge_1d_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true], permutation_map = #map} : memref<1xf32>, vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @merge_1d_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %scalar = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<4xf32>
  return %vector : vector<4xf32>
}

// CHECK-LABEL: func.func @merge_2d_read(%arg0: memref<8x1xf32>, %arg1: index, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1, %arg2], %arg3 {in_bounds = [true], permutation_map = #map1} : memref<8x1xf32>, vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @merge_2d_read(%arg0: memref<8x1xf32>, %arg1: index, %arg2: index, %arg3: f32) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %scalar = vector.transfer_read %arg0[%arg1, %arg2], %arg3 {in_bounds = [true]} : memref<8x1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<4xf32>
  return %vector : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_masked_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32, %arg3: vector<1xi1>) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1], %arg2, %arg3 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
// CHECK: %{{.+}} = vector.broadcast %{{.+}} : vector<1xf32> to vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @keep_masked_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32, %arg3: vector<1xi1>) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %scalar = vector.transfer_read %arg0[%arg1], %arg2, %arg3 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<4xf32>
  return %vector : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_shared_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> (vector<4xf32>, vector<1xf32>) attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
// CHECK: %{{.+}} = vector.broadcast %{{.+}} : vector<1xf32> to vector<4xf32>
// CHECK: return %{{.+}}, %{{.+}} : vector<4xf32>, vector<1xf32>
func.func @keep_shared_read(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> (vector<4xf32>, vector<1xf32>) attributes {ascvf2.vector_func} {
  %scalar = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<4xf32>
  return %vector, %scalar : vector<4xf32>, vector<1xf32>
}

// CHECK-LABEL: func.func @keep_non_vf_width(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<8xf32> attributes {ascvf2.vector_func}
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
// CHECK: %{{.+}} = vector.broadcast %{{.+}} : vector<1xf32> to vector<8xf32>
// CHECK: return %{{.+}} : vector<8xf32>
func.func @keep_non_vf_width(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<8xf32> attributes {ascvf2.vector_func} {
  %scalar = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<8xf32>
  return %vector : vector<8xf32>
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<4xf32>
// CHECK: %{{.+}} = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
// CHECK: %{{.+}} = vector.broadcast %{{.+}} : vector<1xf32> to vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
func.func @ignore_non_vector_function(%arg0: memref<1xf32>, %arg1: index, %arg2: f32) -> vector<4xf32> {
  %scalar = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
  %vector = vector.broadcast %scalar : vector<1xf32> to vector<4xf32>
  return %vector : vector<4xf32>
}

}
