// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-combine-extract-slice %s | FileCheck %s

// CHECK-LABEL: func.func @combine_with_constant(%arg0: index, %arg1: index) -> tensor<32x32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %cst : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @combine_with_constant(%arg0: index, %arg1: index) -> tensor<32x32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #asctile.local<UB>>
  %extracted_slice = tensor.extract_slice %cst[%arg0, %arg1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  return %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @combine_with_splat(%arg0: index, %arg1: index, %arg2: f32) -> tensor<32x32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %splat = tensor.splat %arg2 : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %splat : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @combine_with_splat(%arg0: index, %arg1: index, %arg2: f32) -> tensor<32x32xf32, #asctile.local<UB>> {
  %splat = tensor.splat %arg2 : tensor<32x64xf32, #asctile.local<UB>>
  %extracted_slice = tensor.extract_slice %splat[%arg0, %arg1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  return %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @combine_with_load(%arg0: tensor<?x?xf32, #asctile.global>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: index) -> tensor<32x32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:  %0 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:  %1 = arith.addi %arg1, %0 : i32
// CHECK-NEXT:  %2 = arith.addi %arg2, %c8_i32 : i32
// CHECK-NEXT:  %3 = asctile.load %arg0[%1, %2], %arg3 : tensor<?x?xf32, #asctile.global>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @combine_with_load(%arg0: tensor<?x?xf32, #asctile.global>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: index) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = asctile.load %arg0[%arg1, %arg2], %arg3 : tensor<?x?xf32, #asctile.global>, tensor<32x64xf32, #asctile.local<UB>>
  %extracted_slice = tensor.extract_slice %0[%arg4, 8] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  return %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
}
