// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-apply-cv-strategy -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func.func @m_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
// CHECK:       %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %2 = arith.muli %1, %c16_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg1, %2 : i32
// CHECK-NEXT:  asctile.store %0, %arg3[%3, %arg2] : tensor<16x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg3[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
// CHECK:       %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<split_by_n>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %2 = arith.muli %1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  asctile.store %0, %arg3[%arg1, %3] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg3[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @m_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
// CHECK-NEXT:  %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<join_by_m>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %0 : tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @m_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
  return %0 : tensor<32x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @n_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
// CHECK-NEXT:  %0 = asctile.copy %arg0[%arg1, %arg2] {distrib = #asctile.distrib_mode<join_by_n>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %0 : tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @n_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
  return %0 : tensor<32x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @m_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c16 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %extracted_slice_0 = tensor.extract_slice %arg1[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.addf %extracted_slice, %extracted_slice_0 : tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c16_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg2[%5, %arg4] : tensor<16x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
  %0 = arith.addf %arg0, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg2[%arg3, %arg4] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[0, %1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = math.absf %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c32_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg4, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg2[%arg3, %5] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
  %0 = math.absf %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg2[%arg3, %arg4] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @m_reduce_broadcast(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c16 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.reduce <sum> %extracted_slice {dims = [1 : i32]} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.broadcast %2 : tensor<16x1xf32, #asctile.local<UB>> to tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %5 = arith.muli %4, %c16_i32 : i32
// CHECK-NEXT:  %6 = arith.addi %arg2, %5 : i32
// CHECK-NEXT:  asctile.store %3, %arg1[%6, %arg3] : tensor<16x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_reduce_broadcast(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x128xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.reduce <sum> %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>, dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %0 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
  asctile.store %1, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_reduce(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[0, %1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.reduce <sum> %extracted_slice {dims = [0 : i32]} : tensor<32x32xf32, #asctile.local<UB>>, tensor<1x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c32_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg1[%arg2, %5] : tensor<1x32xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_reduce(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<1x64xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.reduce <sum> %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>, dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>} : tensor<1x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_broadcast(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[0, %1] [1, 32] [32, 1] : tensor<1x64xf32, #asctile.local<UB>> to tensor<1x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.broadcast %extracted_slice : tensor<1x32xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c32_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg1[%arg2, %5] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_broadcast(%arg0: tensor<1x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.broadcast %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @m_yield_store(
// CHECK:       %0 = asctile.copy %arg0[%arg2, %arg3] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %2 = arith.muli %1, %c16_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  asctile.store %0, %arg1[%3, %arg3] : tensor<16x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_yield_store(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    %1 = asctile.copy %arg0[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
  }
  asctile.store %0, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_1d_broadcast(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[%1] [32] [1] : tensor<64xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.broadcast %extracted_slice : tensor<32xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c32_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg1[%arg2, %5] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_1d_broadcast(%arg0: tensor<64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.broadcast %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_squeezed_reduce_broadcast(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[0, %1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.reduce <sum> %extracted_slice {dims = [0 : i32]} : tensor<32x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.broadcast %2 : tensor<32xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %5 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:  %6 = arith.addi %arg3, %5 : i32
// CHECK-NEXT:  asctile.store %3, %arg1[%arg2, %6] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_squeezed_reduce_broadcast(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.global>, %arg2: i32, %arg3: i32) {
  %0 = asctile.reduce <sum> %arg0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32>, dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<64xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %0 {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %1, %arg1[%arg2, %arg3] {asctile.need_split = #asctile.distrib_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @m_loop_carried(
// CHECK:       %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c16 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg1[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %extracted_slice_0 = tensor.extract_slice %arg2[%1, 0] [16, 128] [128, 1] : tensor<32x128xf32, #asctile.local<UB>> to tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2:2 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %extracted_slice, %arg6 = %extracted_slice_0) -> (tensor<16x64xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %8 = asctile.copy %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    %9 = arith.mulf %8, %arg5 : tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    %10 = math.absf %9 : tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:    %11 = asctile.broadcast %10 : tensor<16x64xf32, #asctile.local<UB>> to tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:    scf.yield %9, %11 : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = asctile.reduce <sum> %2#0 {dims = [1 : i32]} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = asctile.broadcast %3 : tensor<16x1xf32, #asctile.local<UB>> to tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  %5 = arith.addf %4, %2#1 : tensor<16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  %6 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %7 = arith.muli %6, %c16_i32 : i32
// CHECK-NEXT:  asctile.store %5, %arg3[%7, %c0_i32] : tensor<16x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_loop_carried(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x128xf32, #asctile.local<UB>>, %arg3: tensor<32x128xf32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %0:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<32x64xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.local<UB>>) {
    %1 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
      %2 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
      %3 = arith.mulf %2, %arg4 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
      asctile.yield %3 : tensor<32x64xf32, #asctile.local<UB>> {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>}
    }
    %4 = math.absf %1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
    %5 = asctile.broadcast %4 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
    scf.yield {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64>} %1, %5 : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.local<UB>>
  }
  %6 = asctile.reduce <sum> %0#0 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>, dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
  %7 = asctile.broadcast %6 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
  %8 = arith.addf %7, %0#1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x128xf32, #asctile.local<UB>>
  asctile.store %8, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>} : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
  return
}
