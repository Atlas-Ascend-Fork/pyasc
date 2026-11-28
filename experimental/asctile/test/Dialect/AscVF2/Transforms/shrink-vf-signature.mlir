// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-shrink-vf-signature %s | FileCheck %s

// CHECK-LABEL: func.func @remove_unused_leading_middle_and_trailing_results(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
// CHECK: %0:3 = ascvf2.vf_group {
// CHECK: %1:3 = func.call @shrink_vf(%arg0, %arg1) : (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
// CHECK: ascvf2.yield %1#0, %1#1, %1#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK: } : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK: return %0#0, %0#1, %0#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
func.func @remove_unused_leading_middle_and_trailing_results(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
  %group:3 = ascvf2.vf_group {
    %call:5 = func.call @shrink_vf(%arg0, %arg1) : (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
    ascvf2.yield %call#0, %call#3, %call#4 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  } : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  return %group#0, %group#1, %group#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @keep_all_used_results(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
// CHECK: %0:3 = ascvf2.vf_group {
// CHECK: %1:3 = func.call @keep_all(%arg0, %arg1) : (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
// CHECK: ascvf2.yield %1#0, %1#1, %1#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK: } : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK: return %0#0, %0#1, %0#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
func.func @keep_all_used_results(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
  %group:3 = ascvf2.vf_group {
    %call:3 = func.call @keep_all(%arg0, %arg1) : (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
    ascvf2.yield %call#0, %call#1, %call#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  } : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  return %group#0, %group#1, %group#2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @ignore_non_vector_function_call(%arg0: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
// CHECK: %0 = ascvf2.vf_group {
// CHECK: %1:2 = func.call @non_vector(%arg0) : (tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
// CHECK: ascvf2.yield %1#0 : tensor<32x32xf32, #asctile.local<UB>>
// CHECK: } : tensor<32x32xf32, #asctile.local<UB>>
// CHECK: return %0 : tensor<32x32xf32, #asctile.local<UB>>
func.func @ignore_non_vector_function_call(%arg0: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %group = ascvf2.vf_group {
    %call:2 = func.call @non_vector(%arg0) : (tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
    ascvf2.yield %call#0 : tensor<32x32xf32, #asctile.local<UB>>
  } : tensor<32x32xf32, #asctile.local<UB>>
  return %group : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func private @shrink_vf(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
// CHECK: %0 = arith.addf %arg0, %arg1
// CHECK: %1 = arith.subf %arg0, %arg1
// CHECK: %2 = arith.mulf %arg0, %arg1
// CHECK: return %0, %1, %2
func.func private @shrink_vf(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
  %0 = arith.addf %arg0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.subf %arg0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %2 = arith.mulf %arg0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %0, %1, %2, %1, %2 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func private @keep_all(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
// CHECK: return %arg0, %arg1, %arg0
func.func private @keep_all(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
  return %arg0, %arg1, %arg0 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func private @non_vector(%arg0: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
// CHECK: return %arg0, %arg0
func.func private @non_vector(%arg0: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
  return %arg0, %arg0 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}
