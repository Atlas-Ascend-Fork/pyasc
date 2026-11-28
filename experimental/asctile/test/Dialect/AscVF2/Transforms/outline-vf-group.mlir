// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group %s | FileCheck %s

// CHECK-LABEL: func.func @outline_constant_and_external_users(%arg0: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>) {
// CHECK: %cst = arith.constant 2.000000e+00 : f32
// CHECK: %[[GROUP:.*]]:2 = ascvf2.vf_group {
// CHECK-NEXT: %[[CALL:.*]]:2 = func.call @outline_constant_and_external_users_vf1(%arg0) : (tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>)
// CHECK-NEXT: ascvf2.yield %[[CALL]]#0, %[[CALL]]#1 : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT: } : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT: return %[[GROUP]]#0, %[[GROUP]]#1 : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @outline_constant_and_external_users(%arg0: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>) {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.muls %0, %cst : tensor<128xf32, #asctile.local<UB>>
  return %0, %1 : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @outline_groups_split_by_call(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
// CHECK: ascvf2.vf_group {
// CHECK: func.call @outline_groups_split_by_call_vf1(
// CHECK: ascvf2.yield
// CHECK: %[[SPLIT:.*]] = call @split
// CHECK: ascvf2.vf_group {
// CHECK: func.call @outline_groups_split_by_call_vf2(
// CHECK: ascvf2.yield
func.func @outline_groups_split_by_call(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = arith.subf %0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %2 = func.call @split(%1) : (tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>>
  %3 = arith.mulf %2, %2 : tensor<128xf32, #asctile.local<UB>>
  %4 = arith.divf %3, %2 : tensor<128xf32, #asctile.local<UB>>
  return %4 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @outline_elementwise_with_reduction_sink(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
// CHECK: ascvf2.vf_group {
// CHECK: func.call @outline_elementwise_with_reduction_sink_vf1(
// CHECK: ascvf2.yield
func.func @outline_elementwise_with_reduction_sink(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @outline_reduction_with_broadcast_prefix(%arg0: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
// CHECK: ascvf2.vf_group {
// CHECK: func.call @outline_reduction_with_broadcast_prefix_vf1(
// CHECK: ascvf2.yield
func.func @outline_reduction_with_broadcast_prefix(%arg0: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <max> %arg0 : tensor<1x128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<1x128xf32, #asctile.local<UB>>
  return %1 : tensor<1x128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_outline_non_ub_tensors(%arg0: tensor<32xf32, #asctile.local<L1>>, %arg1: tensor<32xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<L1>> {
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_outline_non_ub_tensors(%arg0: tensor<32xf32, #asctile.local<L1>>, %arg1: tensor<32xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<L1>> {
  %0 = arith.subf %arg0, %arg1 : tensor<32xf32, #asctile.local<L1>>
  %1 = arith.mulf %0, %arg0 : tensor<32xf32, #asctile.local<L1>>
  return %1 : tensor<32xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @do_not_outline_unsupported_reduction(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_outline_unsupported_reduction(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <prod> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_outline_single_operation(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_outline_single_operation(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  return %0 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_outline_reduction_without_producer(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_outline_reduction_without_producer(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <sum> %arg0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %0 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func private @split(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
func.func private @split(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  return %arg0 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @outline_elementwise_with_reduce_min_sink(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
// CHECK: ascvf2.vf_group
// CHECK: func.call @outline_elementwise_with_reduce_min_sink_vf1(
// CHECK: ascvf2.yield
func.func @outline_elementwise_with_reduce_min_sink(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <min> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @outline_reduction_min_with_broadcast_prefix(%arg0: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
// CHECK: ascvf2.vf_group
// CHECK: func.call @outline_reduction_min_with_broadcast_prefix_vf1(
// CHECK: ascvf2.yield
func.func @outline_reduction_min_with_broadcast_prefix(%arg0: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <min> %arg0 : tensor<1x128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<1x128xf32, #asctile.local<UB>>
  return %1 : tensor<1x128xf32, #asctile.local<UB>>
}
