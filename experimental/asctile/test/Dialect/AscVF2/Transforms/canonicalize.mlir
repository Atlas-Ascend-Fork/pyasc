// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @erase_empty_group(%arg0: vector<4xf32>) -> vector<4xf32> {
// CHECK-NOT: ascvf2.vf_group
// CHECK-NOT: ascvf2.yield
// CHECK: return %arg0 : vector<4xf32>
func.func @erase_empty_group(%arg0: vector<4xf32>) -> vector<4xf32> {
  %result = ascvf2.vf_group {
    ascvf2.yield %arg0 : vector<4xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @erase_unused_result(%arg0: vector<4xf32>) -> vector<4xf32> {
// CHECK-COUNT-1: %{{.+}} = ascvf2.vf_group {
// CHECK: %{{.+}} = arith.addf %arg0, %arg0 : vector<4xf32>
// CHECK: ascvf2.yield %{{.+}} : vector<4xf32>
// CHECK: return %{{.+}} : vector<4xf32>
// CHECK-NOT: ascvf2.vf_group
// CHECK-NOT: arith.constant dense<0.000000e+00> : vector<4xf32>
func.func @erase_unused_result(%arg0: vector<4xf32>) -> vector<4xf32> {
  %unused = ascvf2.vf_group {
    %zero = arith.constant dense<0.000000e+00> : vector<4xf32>
    ascvf2.yield %zero : vector<4xf32>
  } : vector<4xf32>
  %result = ascvf2.vf_group {
    %sum = arith.addf %arg0, %arg0 : vector<4xf32>
    ascvf2.yield %sum : vector<4xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @forward_group_result(%arg0: vector<4xf32>) -> vector<4xf32> {
// CHECK-NOT: ascvf2.vf_group
// CHECK-NOT: arith.addf
// CHECK: return %arg0 : vector<4xf32>
func.func @forward_group_result(%arg0: vector<4xf32>) -> vector<4xf32> {
  %result = ascvf2.vf_group {
    %sum = arith.addf %arg0, %arg0 : vector<4xf32>
    ascvf2.yield %arg0 : vector<4xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @fold_to_tensor_to_buffer(%arg0: memref<4xf32, 26>) -> memref<4xf32, 26> {
// CHECK-NOT: ascvf2.to_tensor
// CHECK-NOT: ascvf2.to_buffer
// CHECK: return %arg0 : memref<4xf32, 26>
func.func @fold_to_tensor_to_buffer(%arg0: memref<4xf32, 26>) -> memref<4xf32, 26> {
  %tensor = ascvf2.to_tensor %arg0 : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  return %buffer : memref<4xf32, 26>
}

// CHECK-LABEL: func.func @fold_to_buffer_to_tensor(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>> {
// CHECK-NOT: ascvf2.to_buffer
// CHECK-NOT: ascvf2.to_tensor
// CHECK: return %arg0 : tensor<4xf32, #asctile.local<UB>>
func.func @fold_to_buffer_to_tensor(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>> {
  %buffer = ascvf2.to_buffer %arg0 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  return %tensor : tensor<4xf32, #asctile.local<UB>>
}
