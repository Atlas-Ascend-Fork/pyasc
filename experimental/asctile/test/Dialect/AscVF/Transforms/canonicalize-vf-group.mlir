// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -canonicalize-vf-group %s | FileCheck %s

// CHECK-LABEL:func.func @dont_create_new_opnds() {
// CHECK:       ascvf.vf_group {
func.func @dont_create_new_opnds() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i64 = arith.constant 32 : i64
  ascvf.vf_group {
    ascendc.adds_l2 %2, %1, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  } {groupType = !ascendc.local_tensor<32xf32>}
  return
}

// CHECK-LABEL:func.func @erase_if_unused() {
// CHECK:       ascvf.vf_group ins(%1 : !ascendc.local_tensor<32xf32>) {
func.func @erase_if_unused() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i64 = arith.constant 32 : i64
  ascvf.vf_group outs(%0 : !ascendc.local_tensor<32xf32>) ins(%1 : !ascendc.local_tensor<32xf32>) tmps(%2 : !ascendc.local_tensor<32xf32>) {
    ascendc.adds_l2 %1, %1, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  } {groupType = !ascendc.local_tensor<32xf32>}
  return
}

// CHECK-LABEL:func.func @dst_to_tmp_if_unused_outside() {
// CHECK:       ascvf.vf_group ins(%1 : !ascendc.local_tensor<32xf32>) tmps(%2 : !ascendc.local_tensor<32xf32>) {
func.func @dst_to_tmp_if_unused_outside() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i64 = arith.constant 32 : i64
  ascvf.vf_group outs(%2 : !ascendc.local_tensor<32xf32>) ins(%1 : !ascendc.local_tensor<32xf32>) {
    ascendc.adds_l2 %2, %1, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  } {groupType = !ascendc.local_tensor<32xf32>}
  return
}

// CHECK-LABEL:func.func @tmp_to_dst_if_used_outside() {
// CHECK:       ascvf.vf_group outs(%2 : !ascendc.local_tensor<32xf32>) ins(%1 : !ascendc.local_tensor<32xf32>) {
func.func @tmp_to_dst_if_used_outside() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c32_i64 = arith.constant 32 : i64
  ascvf.vf_group ins(%1 : !ascendc.local_tensor<32xf32>) tmps(%2 : !ascendc.local_tensor<32xf32>) {
    ascendc.adds_l2 %2, %1, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  } {groupType = !ascendc.local_tensor<32xf32>}
  ascendc.adds_l2 %2, %2, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  return
}

// CHECK-LABEL:func.func @dependency_throught_other_op() {
// CHECK:       ascvf.vf_group outs(%2 : !ascendc.local_tensor<32xf32>) ins(%1 : !ascendc.local_tensor<32xf32>) {
func.func @dependency_throught_other_op() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c32_i64 = arith.constant 32 : i64
  ascvf.vf_group outs(%2 : !ascendc.local_tensor<32xf32>) ins(%1 : !ascendc.local_tensor<32xf32>) {
    ascendc.adds_l2 %2, %1, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, f32, i64
  } {groupType = !ascendc.local_tensor<32xf32>}
  %3 = ascendc.local_tensor.subindex %2[%c0_i64] : !ascendc.local_tensor<32xf32>, i64, !ascendc.local_tensor<32xf32>
  ascendc.duplicate_l2 %3, %cst, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
  return
}
