// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-replace-fp8-ops %s | FileCheck %s

// CHECK-LABEL: func.func @test_load(%arg0: memref<*xf8E5M2, 22>) -> tensor<32xf8E5M2, #asctile.local<UB>> {
// CHECK-DAG: %[[PAD:.*]] = arith.constant 60 : i8
// CHECK:         %0 = asctile.tensor %arg0() : memref<*xf8E5M2, 22>, tensor<128xf8E5M2, #asctile.global>
// CHECK-NEXT:    %1 = arith.bitcast %0 : tensor<128xf8E5M2, #asctile.global> to tensor<128xi8, #asctile.global>
// CHECK-NEXT:    %2 = asctile.load %1[%c0_i32], %[[PAD]] : tensor<128xi8, #asctile.global>, tensor<32xi8, #asctile.local<UB>>
// CHECK-NEXT:    %3 = arith.bitcast %2 : tensor<32xi8, #asctile.local<UB>> to tensor<32xf8E5M2, #asctile.local<UB>>
// CHECK-NEXT:    return %3 : tensor<32xf8E5M2, #asctile.local<UB>>
func.func @test_load(%arg0: memref<*xf8E5M2, 22>) -> tensor<32xf8E5M2, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf8E5M2, 22>, tensor<128xf8E5M2, #asctile.global>
  %c0 = arith.constant 0 : i32
  %pad = arith.constant 1.0 : f8E5M2
  %1 = asctile.load %0[%c0], %pad : tensor<128xf8E5M2, #asctile.global>, tensor<32xf8E5M2, #asctile.local<UB>>
  return %1 : tensor<32xf8E5M2, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @test_load_l1(%arg0: memref<*xf8E5M2, 22>) -> tensor<32xf8E5M2, #asctile.local<L1>> {
// CHECK-DAG: %[[PAD:.*]] = arith.constant 60 : i8
// CHECK:         %0 = asctile.tensor %arg0() : memref<*xf8E5M2, 22>, tensor<128xf8E5M2, #asctile.global>
// CHECK-NEXT:    %1 = arith.bitcast %0 : tensor<128xf8E5M2, #asctile.global> to tensor<128xi8, #asctile.global>
// CHECK-NEXT:    %2 = asctile.load %1[%c0_i32], %[[PAD]] : tensor<128xi8, #asctile.global>, tensor<32xi8, #asctile.local<L1>>
// CHECK-NEXT:    %3 = arith.bitcast %2 : tensor<32xi8, #asctile.local<L1>> to tensor<32xf8E5M2, #asctile.local<L1>>
// CHECK-NEXT:    return %3 : tensor<32xf8E5M2, #asctile.local<L1>>
func.func @test_load_l1(%arg0: memref<*xf8E5M2, 22>) -> tensor<32xf8E5M2, #asctile.local<L1>> {
  %0 = asctile.tensor %arg0() : memref<*xf8E5M2, 22>, tensor<128xf8E5M2, #asctile.global>
  %c0 = arith.constant 0 : i32
  %pad = arith.constant 1.0 : f8E5M2
  %1 = asctile.load %0[%c0], %pad : tensor<128xf8E5M2, #asctile.global>, tensor<32xf8E5M2, #asctile.local<L1>>
  return %1 : tensor<32xf8E5M2, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @test_transpose(%arg0: tensor<128x128xf8E5M2, #asctile.local<UB>>) -> tensor<128x128xf8E5M2, #asctile.local<UB>> {
// CHECK:         %0 = arith.bitcast %arg0 : tensor<128x128xf8E5M2, #asctile.local<UB>> to tensor<128x128xi8, #asctile.local<UB>>
// CHECK-NEXT:    %1 = asctile.transpose %0, [1 : i32, 0 : i32] : tensor<128x128xi8, #asctile.local<UB>> to tensor<128x128xi8, #asctile.local<UB>>
// CHECK-NEXT:    %2 = arith.bitcast %1 : tensor<128x128xi8, #asctile.local<UB>> to tensor<128x128xf8E5M2, #asctile.local<UB>>
// CHECK-NEXT:    return %2 : tensor<128x128xf8E5M2, #asctile.local<UB>>
func.func @test_transpose(%arg0: tensor<128x128xf8E5M2, #asctile.local<UB>>) -> tensor<128x128xf8E5M2, #asctile.local<UB>> {
  %1 = asctile.transpose %arg0, [1: i32, 0: i32] : tensor<128x128xf8E5M2, #asctile.local<UB>> to tensor<128x128xf8E5M2, #asctile.local<UB>>
  return %1 : tensor<128x128xf8E5M2, #asctile.local<UB>>
}
