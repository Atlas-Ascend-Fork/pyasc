// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-fuse-vf-loops %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (-d0 + 6)>
// CHECK: #map1 = affine_map<(d0) -> (d0 + 4)>
#mask = affine_map<(d0) -> (6 - d0)>
#shift = affine_map<(d0) -> (d0 + 4)>

// CHECK-LABEL: func.func @fuse_elementwise_producer_consumer(%arg0: memref<8xf32, 26>, %arg1: memref<8xf32, 26>, %arg2: memref<8xf32, 26>, %arg3: f32) attributes {ascvf2.vector_func}
// CHECK: affine.for %{{.+}} = 0 to 8 step 4 {
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg0[%{{.+}}], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %arg2[%{{.+}}] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg2[%{{.+}}], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %{{.+}} : vector<4xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %arg1[%{{.+}}] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
func.func @fuse_elementwise_producer_consumer(%arg0: memref<8xf32, 26>, %arg1: memref<8xf32, 26>, %arg2: memref<8xf32, 26>, %arg3: f32) attributes {ascvf2.vector_func} {
  affine.for %index = 0 to 8 step 4 {
    %input = vector.transfer_read %arg0[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %input, %arg2[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %value = vector.transfer_read %arg2[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %result = arith.addf %value, %value : vector<4xf32>
    vector.transfer_write %result, %arg1[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return
}

// CHECK-LABEL: func.func @fuse_loops_with_iter_args(%arg0: memref<8xf32, 26>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: f32) -> (vector<4xf32>, vector<4xf32>) attributes {ascvf2.vector_func}
// CHECK: %{{.+}}:2 = affine.for %{{.+}} = 0 to 8 step 4 iter_args(%{{.+}} = %arg1, %{{.+}} = %arg2) -> (vector<4xf32>, vector<4xf32>) {
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg0[%{{.+}}], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %{{.+}} : vector<4xf32>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg0[%{{.+}}], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
// CHECK-NEXT: %{{.+}} = arith.mulf %{{.+}}, %{{.+}} : vector<4xf32>
// CHECK-NEXT: affine.yield %{{.+}}, %{{.+}} : vector<4xf32>, vector<4xf32>
func.func @fuse_loops_with_iter_args(%arg0: memref<8xf32, 26>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: f32) -> (vector<4xf32>, vector<4xf32>) attributes {ascvf2.vector_func} {
  %first = affine.for %index = 0 to 8 step 4 iter_args(%acc = %arg1) -> vector<4xf32> {
    %value = vector.transfer_read %arg0[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.addf %acc, %value : vector<4xf32>
    affine.yield %sum : vector<4xf32>
  }
  %second = affine.for %index = 0 to 8 step 4 iter_args(%acc = %arg2) -> vector<4xf32> {
    %value = vector.transfer_read %arg0[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.mulf %acc, %value : vector<4xf32>
    affine.yield %sum : vector<4xf32>
  }
  return %first, %second : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL: func.func @fuse_nested_masked_loops(%arg0: memref<2x6xf32, 26>, %arg1: memref<2x6xf32, 26>, %arg2: f32) attributes {ascvf2.vector_func}
// CHECK: affine.for %{{.+}} = 0 to 2 {
// CHECK-NEXT: affine.for %{{.+}} = 0 to 6 step 4 {
// CHECK-NEXT: %{{.+}} = affine.apply #map(%{{.+}})
// CHECK-NEXT: %{{.+}} = vector.create_mask %{{.+}} : vector<4xi1>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg0[%{{.+}}, %{{.+}}], %arg2, %{{.+}} {in_bounds = [true]} : memref<2x6xf32, 26>, vector<4xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %arg1[%{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x6xf32, 26>
// CHECK-NEXT: %{{.+}} = affine.apply #map(%{{.+}})
// CHECK-NEXT: %{{.+}} = vector.create_mask %{{.+}} : vector<4xi1>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %arg1[%{{.+}}, %{{.+}}], %arg2, %{{.+}} {in_bounds = [true]} : memref<2x6xf32, 26>, vector<4xf32>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %{{.+}} : vector<4xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %arg1[%{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x6xf32, 26>
func.func @fuse_nested_masked_loops(%arg0: memref<2x6xf32, 26>, %arg1: memref<2x6xf32, 26>, %arg2: f32) attributes {ascvf2.vector_func} {
  affine.for %row = 0 to 2 {
    affine.for %column = 0 to 6 step 4 {
      %remaining = affine.apply #mask(%column)
      %mask = vector.create_mask %remaining : vector<4xi1>
      %value = vector.transfer_read %arg0[%row, %column], %arg2, %mask {in_bounds = [true]} : memref<2x6xf32, 26>, vector<4xf32>
      vector.transfer_write %value, %arg1[%row, %column], %mask {in_bounds = [true]} : vector<4xf32>, memref<2x6xf32, 26>
    }
  }
  affine.for %row = 0 to 2 {
    affine.for %column = 0 to 6 step 4 {
      %remaining = affine.apply #mask(%column)
      %mask = vector.create_mask %remaining : vector<4xi1>
      %value = vector.transfer_read %arg1[%row, %column], %arg2, %mask {in_bounds = [true]} : memref<2x6xf32, 26>, vector<4xf32>
      %result = arith.addf %value, %value : vector<4xf32>
      vector.transfer_write %result, %arg1[%row, %column], %mask {in_bounds = [true]} : vector<4xf32>, memref<2x6xf32, 26>
    }
  }
  return
}

// CHECK-LABEL: func.func @do_not_fuse_shifted_access(%arg0: memref<8xf32, 26>, %arg1: memref<8xf32, 26>, %arg2: memref<8xf32, 26>, %arg3: f32) attributes {ascvf2.vector_func}
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to 8 step 4
func.func @do_not_fuse_shifted_access(%arg0: memref<8xf32, 26>, %arg1: memref<8xf32, 26>, %arg2: memref<8xf32, 26>, %arg3: f32) attributes {ascvf2.vector_func} {
  affine.for %index = 0 to 8 step 4 {
    %input = vector.transfer_read %arg0[%index], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %input, %arg2[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %shifted = affine.apply #shift(%index)
    %value = vector.transfer_read %arg2[%shifted], %arg3 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %result = arith.addf %value, %value : vector<4xf32>
    vector.transfer_write %result, %arg1[%shifted] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return
}

// CHECK-LABEL: func.func @do_not_fuse_result_dependence(%arg0: memref<8xf32, 26>, %arg1: f32, %arg2: vector<4xf32>) -> vector<4xf32> attributes {ascvf2.vector_func}
// CHECK-COUNT-2: affine.for
func.func @do_not_fuse_result_dependence(%arg0: memref<8xf32, 26>, %arg1: f32, %arg2: vector<4xf32>) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %first = affine.for %index = 0 to 8 step 4 iter_args(%acc = %arg2) -> vector<4xf32> {
    %value = vector.transfer_read %arg0[%index], %arg1 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.addf %acc, %value : vector<4xf32>
    affine.yield %sum : vector<4xf32>
  }
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %first, %arg0[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return %first : vector<4xf32>
}

// CHECK-LABEL: func.func @do_not_fuse_unknown_domain(%arg0: memref<8xf32, 26>, %arg1: index, %arg2: f32) attributes {ascvf2.vector_func}
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to %arg1 step 4
func.func @do_not_fuse_unknown_domain(%arg0: memref<8xf32, 26>, %arg1: index, %arg2: f32) attributes {ascvf2.vector_func} {
  affine.for %index = 0 to %arg1 step 4 {
    %value = vector.transfer_read %arg0[%index], %arg2 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %arg0[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to %arg1 step 4 {
    %value = vector.transfer_read %arg0[%index], %arg2 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %arg0[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: memref<8xf32, 26>, %arg1: index, %arg2: f32)
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to %arg1 step 4
func.func @ignore_non_vector_function(%arg0: memref<8xf32, 26>, %arg1: index, %arg2: f32) {
  affine.for %index = 0 to %arg1 step 4 {
    %value = vector.transfer_read %arg0[%index], %arg2 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %arg0[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to %arg1 step 4 {
    %value = vector.transfer_read %arg0[%index], %arg2 {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %arg0[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return
}
