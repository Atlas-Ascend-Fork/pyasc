// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asclower-tensor -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_empty() -> (tensor<8x16xi16, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L0C>>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8x16xi16>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x16xi16> to tensor<8x16xi16, #asctile.local<UB>>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto co1() : <16x32xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L0C>>
// CHECK-NEXT:  return %1, %3 : tensor<8x16xi16, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_empty() -> (tensor<8x16xi16, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L0C>>) {
  %0 = tensor.empty() : tensor<8x16xi16, #asctile.local<UB>>
  %1 = tensor.empty() : tensor<16x32xf32, #asctile.local<L0C>>
  return %0, %1 : tensor<8x16xi16, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_extract_slice_1d(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: index) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c1_i16, %c2_i16, %c2_i16, %c0_i16) : i16, i16, i16, i16
// CHECK-NEXT:  %3 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %0[%3] : !ascendc.local_tensor<32xf32>, i32, !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %4, %2 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<32xf32>, !ascendc.data_copy_params
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_extract_slice_1d(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: index) -> tensor<16xf32, #asctile.local<UB>> {
  %extracted_slice = tensor.extract_slice %arg0[%arg1] [16] [1] : tensor<32xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
  return %extracted_slice : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_extract_slice_2d(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: index, %arg2: index) -> tensor<32x32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x64xf32, #asctile.local<UB>> to !ascendc.local_tensor<32x64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32x32xf32>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c32_i16, %c4_i16, %c4_i16, %c0_i16) : i16, i16, i16, i16
// CHECK-NEXT:  %3 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:  %4 = arith.addi %3, %arg2 : index
// CHECK-NEXT:  %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:  %6 = ascendc.local_tensor.subindex %0[%5] : !ascendc.local_tensor<32x64xf32>, i32, !ascendc.local_tensor<32x64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %6, %2 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<32x32xf32>, !ascendc.local_tensor<32x64xf32>, !ascendc.data_copy_params
// CHECK-NEXT:  %7 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x32xf32> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %7 : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_extract_slice_2d(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: index, %arg2: index) -> tensor<32x32xf32, #asctile.local<UB>> {
  %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg2] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  return %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
