// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -asclower-asctile-data-transfer -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %c32_i32, %9 : i32
// CHECK-NEXT:  %12 = arith.remsi %10, %c32_i32 : i32
// CHECK-NEXT:  %13 = arith.cmpi eq, %12, %c0_i32 : i32
// CHECK-NEXT:  %14 = arith.subi %c32_i32, %12 : i32
// CHECK-NEXT:  %15 = arith.select %13, %c0_i32, %14 : i32
// CHECK-NEXT:  %16 = arith.addi %10, %15 : i32
// CHECK-NEXT:  %17 = arith.addi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %18 = arith.minsi %17, %c32_i32 : i32
// CHECK-NEXT:  %19 = arith.subi %18, %arg1 : i32
// CHECK-NEXT:  %20 = arith.maxsi %19, %c0_i32 : i32
// CHECK-NEXT:  %21 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %22 = arith.divsi %15, %c4_i32 : i32
// CHECK-NEXT:  %23 = arith.subi %c64_i32, %16 : i32
// CHECK-NEXT:  %24 = arith.divsi %23, %c32_i32 : i32
// CHECK-NEXT:  %25 = ascendc.construct !ascendc.data_copy_ext_params(%20, %10, %21, %24, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %26 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %22, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %25, %26 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  %27 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %27 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2], %cst : tensor<32x32xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<2x2x3x8xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<3x4x5x16xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<3x4x5x16xf32, #asctile.global> to !ascendc.global_tensor<3x4x5x16xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %c16_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = arith.muli %arg2, %c80_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %3, %4 : i32
// CHECK-NEXT:  %6 = arith.muli %arg1, %c320_i32 : i32
// CHECK-NEXT:  %7 = arith.addi %5, %6 : i32
// CHECK-NEXT:  %8 = ascendc.global_tensor.subindex %1[%7] : !ascendc.global_tensor<3x4x5x16xf32>, i32, !ascendc.global_tensor<3x4x5x16xf32>
// CHECK-NEXT:  %9 = ascendc.local_tensor_auto veccalc() : <2x2x3x8xf32>
// CHECK-NEXT:  %10 = arith.subi %c16_i32, %arg4 : i32
// CHECK-NEXT:  %11 = arith.cmpi slt, %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.select %11, %c0_i32, %10 : i32
// CHECK-NEXT:  %13 = arith.minsi %12, %c8_i32 : i32
// CHECK-NEXT:  %14 = arith.muli %13, %c4_i32 : i32
// CHECK-NEXT:  %15 = arith.subi %c16_i32, %13 : i32
// CHECK-NEXT:  %16 = arith.remsi %14, %c32_i32 : i32
// CHECK-NEXT:  %17 = arith.cmpi eq, %16, %c0_i32 : i32
// CHECK-NEXT:  %18 = arith.subi %c32_i32, %16 : i32
// CHECK-NEXT:  %19 = arith.select %17, %c0_i32, %18 : i32
// CHECK-NEXT:  %20 = arith.addi %14, %19 : i32
// CHECK-NEXT:  %21 = arith.addi %arg1, %c2_i32 : i32
// CHECK-NEXT:  %22 = arith.minsi %21, %c3_i32 : i32
// CHECK-NEXT:  %23 = arith.subi %22, %arg1 : i32
// CHECK-NEXT:  %24 = arith.maxsi %23, %c0_i32 : i32
// CHECK-NEXT:  %25 = arith.muli %15, %c4_i32 : i32
// CHECK-NEXT:  %26 = arith.divsi %19, %c4_i32 : i32
// CHECK-NEXT:  %27 = arith.subi %c32_i32, %20 : i32
// CHECK-NEXT:  %28 = arith.divsi %27, %c32_i32 : i32
// CHECK-NEXT:  %29 = ascendc.construct !ascendc.data_copy_ext_params(%24, %14, %25, %28, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %30 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %26, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %9, %8, %29, %30 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<2x2x3x8xf32>, !ascendc.global_tensor<3x4x5x16xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  %31 = builtin.unrealized_conversion_cast %9 : !ascendc.local_tensor<2x2x3x8xf32> to tensor<2x2x3x8xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %31 : tensor<2x2x3x8xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<2x2x3x8xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<3x4x5x16xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2, %arg3, %arg4], %cst : tensor<3x4x5x16xf32, #asctile.global>, tensor<2x2x3x8xf32, #asctile.local<UB>>
  return %1: tensor<2x2x3x8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %arg2, %9 : i32
// CHECK-NEXT:  %12 = arith.remsi %10, %c32_i32 : i32
// CHECK-NEXT:  %13 = arith.cmpi eq, %12, %c0_i32 : i32
// CHECK-NEXT:  %14 = arith.subi %c32_i32, %12 : i32
// CHECK-NEXT:  %15 = arith.select %13, %c0_i32, %14 : i32
// CHECK-NEXT:  %16 = arith.addi %10, %15 : i32
// CHECK-NEXT:  %17 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:  %18 = arith.minsi %arg1, %17 : i32
// CHECK-NEXT:  %19 = arith.subi %18, %arg3 : i32
// CHECK-NEXT:  %20 = arith.maxsi %19, %c0_i32 : i32
// CHECK-NEXT:  %21 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %22 = arith.divsi %15, %c4_i32 : i32
// CHECK-NEXT:  %23 = arith.subi %c64_i32, %16 : i32
// CHECK-NEXT:  %24 = arith.divsi %23, %c32_i32 : i32
// CHECK-NEXT:  %25 = ascendc.construct !ascendc.data_copy_ext_params(%20, %10, %21, %24, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %26 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %22, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %25, %26 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  %27 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %27 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst : tensor<?x?xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %arg6, %8 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %arg2, %9 : i32
// CHECK-NEXT:  %12 = arith.remsi %10, %c32_i32 : i32
// CHECK-NEXT:  %13 = arith.cmpi eq, %12, %c0_i32 : i32
// CHECK-NEXT:  %14 = arith.subi %c32_i32, %12 : i32
// CHECK-NEXT:  %15 = arith.select %13, %c0_i32, %14 : i32
// CHECK-NEXT:  %16 = arith.addi %10, %15 : i32
// CHECK-NEXT:  %17 = arith.cmpi slt, %16, %c64_i32 : i32
// CHECK-NEXT:  scf.if %17 {
// CHECK-NEXT:    ascendc.duplicate_l2 %5, %cst, %c0_i32 : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %18 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:  %19 = arith.minsi %arg1, %18 : i32
// CHECK-NEXT:  %20 = arith.subi %19, %arg3 : i32
// CHECK-NEXT:  %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:  %22 = arith.minsi %21, %arg5 : i32
// CHECK-NEXT:  %23 = arith.subi %c16_i32, %22 : i32
// CHECK-NEXT:  %24 = arith.cmpi sgt, %23, %c0_i32 : i32
// CHECK-NEXT:  scf.if %24 {
// CHECK-NEXT:    %32 = arith.muli %22, %c16_i32 : i32
// CHECK-NEXT:    %33 = ascendc.local_tensor.subindex %5[%32] : !ascendc.local_tensor<16x16xf32>, i32, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %34 = arith.muli %23, %c16_i32 : i32
// CHECK-NEXT:    ascendc.duplicate_l2 %33, %cst, %34 {asc.cal_count_set} : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %25 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %26 = arith.divsi %15, %c4_i32 : i32
// CHECK-NEXT:  %27 = arith.subi %c64_i32, %16 : i32
// CHECK-NEXT:  %28 = arith.divsi %27, %c32_i32 : i32
// CHECK-NEXT:  %29 = ascendc.construct !ascendc.data_copy_ext_params(%22, %10, %25, %28, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %30 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %26, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %29, %30 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  %31 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %31 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst, (%arg5, %arg6) : tensor<?x?xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %c16_i32, %9 : i32
// CHECK-NEXT:  %12 = arith.subi %c32_i32, %9 : i32
// CHECK-NEXT:  %13 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %14 = arith.maxsi %13, %c0_i32 : i32
// CHECK-NEXT:  %15 = arith.minsi %14, %c16_i32 : i32
// CHECK-NEXT:  %16 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %19 = ascendc.construct !ascendc.data_copy_ext_params(%15, %10, %17, %18, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %19 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %c16_i32, %9 : i32
// CHECK-NEXT:  %12 = arith.subi %arg5, %9 : i32
// CHECK-NEXT:  %13 = arith.subi %arg4, %arg2 : i32
// CHECK-NEXT:  %14 = arith.maxsi %13, %c0_i32 : i32
// CHECK-NEXT:  %15 = arith.minsi %14, %c16_i32 : i32
// CHECK-NEXT:  %16 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %19 = ascendc.construct !ascendc.data_copy_ext_params(%15, %10, %17, %18, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %19 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32)
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.fixpipe_config(%14) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %13, %15 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.fixpipe_config(%14) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %13, %15 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf16, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf16, #asctile.global> to !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf16>, i32, !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %14 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %13 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.fixpipe_config(%15) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %14, %16 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf16, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf16, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf16, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf16, #asctile.global> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg4, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %14 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %13 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.fixpipe_config(%15) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %14, %16 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<?x?xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf16, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf16, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.minsi %9, %arg7 : i32
// CHECK-NEXT:  %11 = arith.minsi %arg6, %c16_i32 : i32
// CHECK-NEXT:  %12 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %10 : i32, "mSize" = %11 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32)
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.fixpipe_config(%13) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %12, %14 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
func.func @lower_store_fixpipe_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], (%arg6, %arg7) : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<2x8xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<2x8xf32, #asctile.local<UB>> to !ascendc.local_tensor<2x8xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = arith.cmpi slt, %arg3, %c0_i32 : i32
// CHECK-NEXT:  %4 = arith.select %3, %c0_i32, %arg3 : i32
// CHECK-NEXT:  %5 = arith.minsi %arg5, %4 : i32
// CHECK-NEXT:  %6 = arith.muli %5, %c4_i32 : i32
// CHECK-NEXT:  %7 = arith.subi %c8_i32, %5 : i32
// CHECK-NEXT:  %8 = arith.subi %arg3, %5 : i32
// CHECK-NEXT:  %9 = arith.maxsi %arg2, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.minsi %arg4, %9 : i32
// CHECK-NEXT:  %11 = arith.muli %7, %c4_i32 : i32
// CHECK-NEXT:  %12 = arith.divsi %11, %c32_i32 : i32
// CHECK-NEXT:  %13 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.data_copy_ext_params(%10, %6, %12, %13, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %2, %0, %14 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<2x8xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<2x8xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store %arg1, %0 [%c0_i32, %c0_i32], (%arg4, %arg5) : tensor<2x8xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<16x64xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<16x128xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<16x128xf16, #asctile.global> to !ascendc.global_tensor<16x128xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <16x64xf16>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c16_i32, %c64_i32, %c0_i32, %c128_i32, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %3 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<16x128xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x64xf16> to tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %4 : tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<16x64xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<16x128xf16, #asctile.global>
  %1 = asctile.load %0[%c0_i32, %c0_i32], %cst : tensor<16x128xf16, #asctile.global>, tensor<16x64xf16, #asctile.local<L1>>
  return %1 : tensor<16x64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<16x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x128xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<16x128xf32, #asctile.global> to !ascendc.global_tensor<16x128xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <16x64xf32>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c16_i32, %c64_i32, %c0_i32, %c128_i32, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %3 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<16x128xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %4 : tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<16x64xf32, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x128xf32, #asctile.global>
  %1 = asctile.load %0[%c0_i32, %c0_i32], %cst : tensor<16x128xf32, #asctile.global>, tensor<16x64xf32, #asctile.local<L1>>
  return %1 : tensor<16x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp16_dynamic(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf16, #asctile.global> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <16x64xf16>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %12, %21 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:      %25 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.muli %12, %c128_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c2048_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
// CHECK-NEXT:  scf.if %18 {
// CHECK-NEXT:    %20 = arith.muli %12, %c16_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c16_i32, %12 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c4_i32, %21, %12, %c0_i32) [ui16, ui16, ui16, f16] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<16x64xf16>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %19 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x64xf16> to tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %19 : tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp16_dynamic(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf16, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst : tensor<?x?xf16, #asctile.global>, tensor<16x64xf16, #asctile.local<L1>>
  return %1 : tensor<16x64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp32_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<56x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <56x64xf32>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c56_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %12, %21 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:      %25 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<56x64xf32>, i32, !ascendc.local_tensor<56x64xf32>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c56_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<56x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c64_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<56x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.muli %12, %c256_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c14336_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
// CHECK-NEXT:  scf.if %18 {
// CHECK-NEXT:    %20 = arith.muli %12, %c8_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c64_i32, %12 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<56x64xf32>, i32, !ascendc.local_tensor<56x64xf32>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c8_i32, %21, %12, %c0_i32) [ui16, ui16, ui16, f32] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<56x64xf32>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %19 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<56x64xf32> to tensor<56x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %19 : tensor<56x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp32_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<56x64xf32, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst : tensor<?x?xf32, #asctile.global>, tensor<56x64xf32, #asctile.local<L1>>
  return %1 : tensor<56x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp16_transpose(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <16x64xf32>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %19 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %20 = arith.maxsi %19, %c0_i32 : i32
// CHECK-NEXT:    %21 = arith.minsi %12, %20 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %21 step %c1_i32  : i32 {
// CHECK-NEXT:      %22 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %23 = ascendc.global_tensor.subindex %4[%22] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:      %24 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %25 = ascendc.local_tensor.subindex %5[%24] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
// CHECK-NEXT:      %26 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %25, %23, %26 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %19 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %19 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.ceildivsi %13, %c8_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c8_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.cmpi sgt, %16, %c0_i32 : i32
// CHECK-NEXT:  scf.if %17 {
// CHECK-NEXT:    %19 = arith.muli %15, %c128_i32 : i32
// CHECK-NEXT:    %20 = arith.muli %16, %c16_i32 : i32
// CHECK-NEXT:    %21 = ascendc.local_tensor.subindex %5[%19] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
// CHECK-NEXT:    %22 = ascendc.construct !ascendc.init_const_value_params(%c1_i32, %20, %c0_i32, %c0_i32) [ui16, ui16, ui16, f32] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %21, %22 : !ascendc.local_tensor<16x64xf32>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %18 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %18 : tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp16_transpose(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf32, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst {asctile.transpose_b_l0} : tensor<?x?xf32, #asctile.global>, tensor<16x64xf32, #asctile.local<L1>>
  return %1 : tensor<16x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp32_transpose(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<50x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <50x64xf32>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c50_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %19 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %20 = arith.maxsi %19, %c0_i32 : i32
// CHECK-NEXT:    %21 = arith.minsi %12, %20 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %21 step %c1_i32  : i32 {
// CHECK-NEXT:      %22 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %23 = ascendc.global_tensor.subindex %4[%22] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:      %24 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %25 = ascendc.local_tensor.subindex %5[%24] : !ascendc.local_tensor<50x64xf32>, i32, !ascendc.local_tensor<50x64xf32>
// CHECK-NEXT:      %26 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c50_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %25, %23, %26 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<50x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %19 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c64_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %19 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<50x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.ceildivsi %13, %c8_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c8_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.cmpi sgt, %16, %c0_i32 : i32
// CHECK-NEXT:  scf.if %17 {
// CHECK-NEXT:    %19 = arith.muli %15, %c512_i32 : i32
// CHECK-NEXT:    %20 = arith.muli %16, %c64_i32 : i32
// CHECK-NEXT:    %21 = ascendc.local_tensor.subindex %5[%19] : !ascendc.local_tensor<50x64xf32>, i32, !ascendc.local_tensor<50x64xf32>
// CHECK-NEXT:    %22 = ascendc.construct !ascendc.init_const_value_params(%c1_i32, %20, %c0_i32, %c0_i32) [ui16, ui16, ui16, f32] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %21, %22 : !ascendc.local_tensor<50x64xf32>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %18 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<50x64xf32> to tensor<50x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %18 : tensor<50x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_fp32_transpose(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<50x64xf32, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst {asctile.transpose_b_l0} : tensor<?x?xf32, #asctile.global>, tensor<50x64xf32, #asctile.local<L1>>
  return %1 : tensor<50x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <32x16xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf16> to tensor<32x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<32x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<32x16xf16, #asctile.local<L0A>>
  return %0: tensor<32x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x32xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<16x32xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<16x32xf16, #asctile.local<L0A>>
  return %0: tensor<16x32xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: tensor<50x64xf32, #asctile.local<L1>>) -> tensor<50x32xf32, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<50x64xf32, #asctile.local<L1>> to !ascendc.local_tensor<50x64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <50x32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c1024_i32] : !ascendc.local_tensor<50x64xf32>, i32, !ascendc.local_tensor<50x64xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c4_i32 : i32, "kStep" = %c4_i32 : i32, "srcStride" = %c4_i32 : i32, "dstStride" = %c4_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<50x32xf32>, !ascendc.local_tensor<50x64xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<50x32xf32> to tensor<50x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<50x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: tensor<50x64xf32, #asctile.local<L1>>) -> tensor<50x32xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<50x64xf32, #asctile.local<L1>>, tensor<50x32xf32, #asctile.local<L0A>>
  return %0: tensor<50x32xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf32, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c128_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c4_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<16x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0A>>
  return %0: tensor<16x32xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0B>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() : <32x16xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, b2>} : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf16> to tensor<32x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:  return %4 : tensor<32x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0b_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<32x16xf16, #asctile.local<L0B>>
  return %0: tensor<32x16xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0B>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() : <16x32xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, b2>} : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L0B>>
// CHECK-NEXT:  return %4 : tensor<16x32xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0b_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<16x32xf16, #asctile.local<L0B>>
  return %0: tensor<16x32xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_b_trans_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0B>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf32, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() : <16x32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, b2>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L0B>>
// CHECK-NEXT:  return %4 : tensor<16x32xf32, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0b_b_trans_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] {asctile.transpose_b_l0} : tensor<32x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0B>>
  return %0: tensor<16x32xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_diff_src_dst_f32(%arg0: tensor<96x128xf32, #asctile.local<L1>>) -> tensor<16x64xf32, #asctile.local<L0B>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<96x128xf32, #asctile.local<L1>> to !ascendc.local_tensor<96x128xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() : <16x64xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c6272_i32] : !ascendc.local_tensor<96x128xf32>, i32, !ascendc.local_tensor<96x128xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c8_i32 : i32, "srcStride" = %c6_i32 : i32, "dstStride" = %c4_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, b2>} : !ascendc.local_tensor<16x64xf32>, !ascendc.local_tensor<96x128xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L0B>>
// CHECK-NEXT:  return %4 : tensor<16x64xf32, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0b_diff_src_dst_f32(%arg0: tensor<96x128xf32, #asctile.local<L1>>) -> tensor<16x64xf32, #asctile.local<L0B>> {
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c64_i32] : tensor<96x128xf32, #asctile.local<L1>>, tensor<16x64xf32, #asctile.local<L0B>>
  return %0: tensor<16x64xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_diff_src_dst(%arg0: tensor<160x96xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<160x96xf16, #asctile.local<L1>> to !ascendc.local_tensor<160x96xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x16xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c2560_i32] : !ascendc.local_tensor<160x96xf16>, i32, !ascendc.local_tensor<160x96xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c10_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<160x96xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x16xf16> to tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_diff_src_dst(%arg0: tensor<160x96xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<160x96xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  return %0: tensor<16x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_trans_diff_src_dst(%arg0: tensor<96x160xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<96x160xf16, #asctile.local<L1>> to !ascendc.local_tensor<96x160xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x16xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c3328_i32] : !ascendc.local_tensor<96x160xf16>, i32, !ascendc.local_tensor<96x160xf16>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c6_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %3 {direction = #ascendc.copy_direction<gm, a2>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<96x160xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x16xf16> to tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:  return %4 : tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @lower_copy_l1_l0a_trans_diff_src_dst(%arg0: tensor<96x160xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c32_i32] {asctile.transpose_a_l0} : tensor<96x160xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  return %0: tensor<16x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_load_bias_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<64xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<64xf32, #asctile.global> to !ascendc.global_tensor<64xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %c64_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<64xf32>, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<64xf32> to tensor<64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_bias_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<64xf32, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<64xf32, #asctile.global>
  %1 = asctile.load %0[%c0_i32], %cst {asctile.is_bias} : tensor<64xf32, #asctile.global>, tensor<64xf32, #asctile.local<L1>>
  return %1 : tensor<64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_bias_l1_bt_fp32(%arg0: tensor<64xf32, #asctile.local<L1>>) -> tensor<64xf32, #asctile.local<BT>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64xf32, #asctile.local<L1>> to !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto c2() : <64xf32>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c1_i32, %c8_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l0 %1, %0, %2 {direction = #ascendc.copy_direction<gm, c2>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.data_copy_params
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<64xf32> to tensor<64xf32, #asctile.local<BT>>
// CHECK-NEXT:  return %3 : tensor<64xf32, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @lower_copy_bias_l1_bt_fp32(%arg0: tensor<64xf32, #asctile.local<L1>>) -> tensor<64xf32, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32] : tensor<64xf32, #asctile.local<L1>>, tensor<64xf32, #asctile.local<BT>>
  return %0 : tensor<64xf32, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @lower_load_bias_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<64xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<64xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<64xf16, #asctile.global> to !ascendc.global_tensor<64xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <64xf16>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %c64_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<64xf16>, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<64xf16> to tensor<64xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<64xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_bias_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<64xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<64xf16, #asctile.global>
  %1 = asctile.load %0[%c0_i32], %cst {asctile.is_bias} : tensor<64xf16, #asctile.global>, tensor<64xf16, #asctile.local<L1>>
  return %1 : tensor<64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_bias_l1_bt_fp16(%arg0: tensor<64xf16, #asctile.local<L1>>) -> tensor<64xf16, #asctile.local<BT>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64xf16, #asctile.local<L1>> to !ascendc.local_tensor<64xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto c2() : <64xf32>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c1_i32, %c4_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l0 %1, %0, %2 {direction = #ascendc.copy_direction<gm, c2>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<64xf32> to tensor<64xf16, #asctile.local<BT>>
// CHECK-NEXT:  return %3 : tensor<64xf16, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @lower_copy_bias_l1_bt_fp16(%arg0: tensor<64xf16, #asctile.local<L1>>) -> tensor<64xf16, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32] : tensor<64xf16, #asctile.local<L1>>, tensor<64xf16, #asctile.local<BT>>
  return %0 : tensor<64xf16, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @lower_store_3d_tensor
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[ONE]], %[[STRIDE1]], %[[STRIDE1]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK-SAME: direction = #ascendc.copy_direction<veccalc, gm>
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}
func.func @lower_store_3d_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32xf32, #asctile.local<UB>>) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32xf32, #asctile.global>
  %c0_i32 = arith.constant 0 : i32
  asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32] : tensor<32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_4d_tensor
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[STRIDE2:.*]] = arith.constant 131072 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[COUNT1:.*]], %[[STRIDE1]], %[[STRIDE1]], %[[STRIDE2]], %[[STRIDE2]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK-SAME: direction = #ascendc.copy_direction<veccalc, gm>
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}
func.func @lower_store_4d_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32x32xf32, #asctile.local<UB>>) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32x32xf32, #asctile.global>
  %c0_i32 = arith.constant 0 : i32
  asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<32x32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_4d_with_transpose
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[STRIDE1a:.*]] = arith.constant 128 : i32
// CHECK-DAG: %[[STRIDE2:.*]] = arith.constant 131072 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[COUNT1:.*]], %[[STRIDE1]], %[[STRIDE1a]], %[[STRIDE2]], %[[STRIDE2]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}
func.func @lower_store_4d_with_transpose(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32x32xf32, #asctile.local<UB>>) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32x32xf32, #asctile.global>
  %c0_i32 = arith.constant 0 : i32
  asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] {asctile.transpose_dims = array<i32: 0, 2, 1, 3>}: tensor<32x32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_3d_with_transpose
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[STRIDE2:.*]] = arith.constant 128 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[ONE:.*]], %[[STRIDE1]], %[[STRIDE2]], %[[ANY1:.*]], %[[ANY2:.*]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}
func.func @lower_store_3d_with_transpose(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32xf32, #asctile.local<UB>>) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32xf32, #asctile.global>
  %c0_i32 = arith.constant 0 : i32
  asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32] {asctile.transpose_dims = array<i32: 1, 0, 2>}: tensor<32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32xf32, #asctile.global>
  return
}


// CHECK-LABEL: func.func @lower_copy_fixpipe_relu_quantize(%arg0: tensor<16x32xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<16x32xbf16, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xbf16>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %0[%3] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.quant_mode_t(%c16_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.fixpipe_params_c310<0 : i32>("nSize" = %c32_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c256_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %5 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.co2_layout(%c0_i32) constexpr static : i32
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.fixpipe_config(%7, %false) constexpr static : !ascendc.co2_layout, i1
// CHECK-NEXT:  ascendc.fixpipe %1, %4, %6, %8 {direction = #ascendc.copy_direction<co1, a1>} : !ascendc.local_tensor<16x32xbf16>, !ascendc.local_tensor<16x32xf32>, !ascendc.fixpipe_params_c310<0 : i32>, !ascendc.fixpipe_config
// CHECK-NEXT:  %9 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xbf16> to tensor<16x32xbf16, #asctile.local<L1>>
// CHECK-NEXT:  return %9 : tensor<16x32xbf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_fixpipe_relu_quantize(%arg0: tensor<16x32xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<16x32xbf16, #asctile.local<L1>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {quantize, relu} : tensor<16x32xf32, #asctile.local<L0C>>, tensor<16x32xbf16, #asctile.local<L1>>
  return %0 : tensor<16x32xbf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_fixpipe_split_by_m
// CHECK: emitasc.init_struct !ascendc.fixpipe_params_c310<1 : i32>("nSize" = %c64_i32 : i32, "mSize" = %c32_i32 : i32, "srcStride" = %c32_i32 : i32, "dstStride" = %c64_i32 : i32, "dualDstCtl" = %c1_i8 : i8)
func.func @lower_copy_fixpipe_split_by_m(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<16x64xf32, #asctile.local<UB>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {distrib = #asctile.distrib_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
  return %0 : tensor<16x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_copy_fixpipe_split_by_n
// CHECK: emitasc.init_struct !ascendc.fixpipe_params_c310<1 : i32>("nSize" = %c64_i32 : i32, "mSize" = %c32_i32 : i32, "srcStride" = %c32_i32 : i32, "dstStride" = %c32_i32 : i32, "dualDstCtl" = %c2_i8 : i8)
func.func @lower_copy_fixpipe_split_by_n(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {distrib = #asctile.distrib_mode<split_by_n>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.local<UB>>
  return %0 : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_copy_fixpipe_full_vec0
// CHECK: emitasc.init_struct !ascendc.fixpipe_params_c310<1 : i32>("nSize" = %c64_i32 : i32, "mSize" = %c32_i32 : i32, "srcStride" = %c32_i32 : i32, "dstStride" = %c64_i32 : i32, "dualDstCtl" = %c0_i8 : i8, "subBlockId" = %false : i1)
func.func @lower_copy_fixpipe_full_vec0(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<UB>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {distrib = #asctile.distrib_mode<full_vec0>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  return %0 : tensor<32x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_copy_fixpipe_full_vec1
// CHECK: emitasc.init_struct !ascendc.fixpipe_params_c310<1 : i32>("nSize" = %c64_i32 : i32, "mSize" = %c32_i32 : i32, "srcStride" = %c32_i32 : i32, "dstStride" = %c64_i32 : i32, "dualDstCtl" = %c0_i8 : i8, "subBlockId" = %true : i1)
func.func @lower_copy_fixpipe_full_vec1(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<UB>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {distrib = #asctile.distrib_mode<full_vec1>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  return %0 : tensor<32x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @load_ub_transpose_3d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> tensor<32x16x128xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<128x16x32xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<128x16x32xf32, #asctile.global> to !ascendc.global_tensor<128x16x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = arith.muli %arg1, %c512_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %3, %4 : i32
// CHECK-NEXT:  %6 = ascendc.global_tensor.subindex %1[%5] : !ascendc.global_tensor<128x16x32xf32>, i32, !ascendc.global_tensor<128x16x32xf32>
// CHECK-NEXT:  %7 = ascendc.local_tensor_auto veccalc() : <32x16x128xf32>
// CHECK-NEXT:  %8 = arith.subi %c128_i32, %arg1 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.minsi %arg4, %9 : i32
// CHECK-NEXT:  %11 = arith.subi %c16_i32, %arg2 : i32
// CHECK-NEXT:  %12 = arith.maxsi %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = arith.minsi %arg4, %12 : i32
// CHECK-NEXT:  %14 = arith.subi %c32_i32, %arg4 : i32
// CHECK-NEXT:  %15 = arith.maxsi %14, %c0_i32 : i32
// CHECK-NEXT:  %16 = arith.minsi %arg4, %15 : i32
// CHECK-NEXT:  %17 = arith.subi %c128_i32, %10 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
// CHECK-NEXT:  %19 = arith.subi %c16_i32, %13 : i32
// CHECK-NEXT:  %20 = arith.cmpi sgt, %19, %c0_i32 : i32
// CHECK-NEXT:  %21 = arith.ori %18, %20 : i1
// CHECK-NEXT:  %22 = arith.subi %c32_i32, %16 : i32
// CHECK-NEXT:  %23 = arith.cmpi sgt, %22, %c0_i32 : i32
// CHECK-NEXT:  scf.if %21 {
// CHECK-NEXT:    ascendc.duplicate_l2 %7, %arg3, %c0_i32 : !ascendc.local_tensor<32x16x128xf32>, f32, i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.if %23 {
// CHECK-NEXT:      %26 = arith.muli %16, %c2048_i32 : i32
// CHECK-NEXT:      %27 = ascendc.local_tensor.subindex %7[%26] : !ascendc.local_tensor<32x16x128xf32>, i32, !ascendc.local_tensor<32x16x128xf32>
// CHECK-NEXT:      %28 = arith.muli %22, %c2048_i32 : i32
// CHECK-NEXT:      ascendc.duplicate_l2 %27, %arg3, %28 {asc.cal_count_set} : !ascendc.local_tensor<32x16x128xf32>, f32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %24 = ascendc.nd_dma_params 3, %arg3 : f32, %10, %13, %16, %c512_i32, %c32_i32, %c1_i32, [1 : i32, 128 : i32, 2048 : i32], [0 : i32, 0 : i32, 0 : i32], [0 : i32, 0 : i32, 0 : i32], : <f32, 3>
// CHECK-NEXT:  ascendc.data_copy_nd_dma %7, %6, %24 {dims = 3 : i32, direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<32x16x128xf32>, !ascendc.global_tensor<128x16x32xf32>, !ascendc.nd_dma_params<f32, 3>
// CHECK-NEXT:  %25 = builtin.unrealized_conversion_cast %7 : !ascendc.local_tensor<32x16x128xf32> to tensor<32x16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %25 : tensor<32x16x128xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @load_ub_transpose_3d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> tensor<32x16x128xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<128x16x32xf32, #asctile.global>
  %1 = asctile.load %0[%arg1, %arg2, %arg4], %arg3, (%arg4, %arg4, %arg4) {asctile.transpose_dims = array<i32: 2, 1, 0>} : tensor<128x16x32xf32, #asctile.global>, tensor<32x16x128xf32, #asctile.local<UB>>
  return %1 : tensor<32x16x128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_get_value(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> f32 {
// CHECK-NEXT:  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.get_value %1, %3 : !ascendc.global_tensor<?x?xf32>, i32, f32
// CHECK-NEXT:  return %4 : f32
// CHECK-NEXT:}
func.func @lower_get_value(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> f32 {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.get_value %0[%arg3, %arg4] : tensor<?x?xf32, #asctile.global>
  return %1 : f32
}

// CHECK-LABEL: func.func @lower_set_value_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<1xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : tensor<1xf32, #asctile.local<UB>> to !ascendc.local_tensor<1xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %0, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  %4 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  %6 = emitc.cast %5 : i32 to ui64
// CHECK-NEXT:  ascendc.global_tensor.set_value %2, %6, %3 : !ascendc.global_tensor<?x?xf32>, ui64, f32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_set_value_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<1xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.set_value %arg1, %0[%arg2, %arg3] : tensor<1xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_set_value_scalar(%arg0: memref<*xf32, 22>, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK-NEXT:  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:  %4 = emitc.cast %3 : i32 to ui64
// CHECK-NEXT:  ascendc.global_tensor.set_value %1, %4, %arg1 : !ascendc.global_tensor<?x?xf32>, ui64, f32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_set_value_scalar(%arg0: memref<*xf32, 22>, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.set_value %arg1, %0[%arg2, %arg3] : f32, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_load_gm_l1_transpose_a_l1(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<64x16xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf16, #asctile.global> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <64x16xf16>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %12, %21 : i32
// CHECK-NEXT:    scf.for %arg5 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg5, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:      %25 = arith.muli %arg5, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<64x16xf16>, i32, !ascendc.local_tensor<64x16xf16>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c64_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<64x16xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c64_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<64x16xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.muli %12, %c128_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c2048_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
// CHECK-NEXT:  scf.if %18 {
// CHECK-NEXT:    %20 = arith.muli %12, %c16_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c16_i32, %12 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<64x16xf16>, i32, !ascendc.local_tensor<64x16xf16>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c4_i32, %21, %12, %c0_i32) [ui16, ui16, ui16, f16] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<64x16xf16>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %19 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<64x16xf16> to tensor<64x16xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %19 : tensor<64x16xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_load_gm_l1_transpose_a_l1(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<64x16xf16, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst {asctile.transpose_a_l1} : tensor<?x?xf16, #asctile.global>, tensor<64x16xf16, #asctile.local<L1>>
  return %1 : tensor<64x16xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_fixpipe_to_ub_relu(%arg0: tensor<16x32xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<16x32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %0[%3] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.fixpipe_params_c310<1 : i32>("nSize" = %c32_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.fixpipe_config(%6, %true) constexpr static : !ascendc.co2_layout, i1
// CHECK-NEXT:  ascendc.fixpipe %1, %4, %5, %7 {direction = #ascendc.copy_direction<co1, veccalc>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.fixpipe_params_c310<1 : i32>, !ascendc.fixpipe_config
// CHECK-NEXT:  %8 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %8 : tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_copy_fixpipe_to_ub_relu(%arg0: tensor<16x32xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32) -> tensor<16x32xf32, #asctile.local<UB>> {
  %0 = asctile.copy_fixpipe %arg0 [%arg1, %arg2] {relu} : tensor<16x32xf32, #asctile.local<L0C>>, tensor<16x32xf32, #asctile.local<UB>>
  return %0 : tensor<16x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_copy_ub_l1_fp16(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf16, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xf16>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %4 = arith.muli %arg1, %c16_i32 : i32
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:    %6 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %1[%6] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %7, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<16x32xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<16x32xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_ub_l1_fp16(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<16x32xf16, #asctile.local<UB>>, tensor<16x32xf16, #asctile.local<L1>>
  return %0 : tensor<16x32xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_ub_l1_fp32(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xf32>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c3_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %4 = arith.muli %arg1, %c8_i32 : i32
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:    %6 = arith.muli %arg1, %c128_i32 : i32
// CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %1[%6] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %7, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.data_copy_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<16x32xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_ub_l1_fp32(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<16x32xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L1>>
  return %0 : tensor<16x32xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_ub_l1_with_offsets(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf16, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c160_i32] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %5 = arith.muli %arg1, %c16_i32 : i32
// CHECK-NEXT:    %6 = ascendc.local_tensor.subindex %2[%5] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:    %7 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:    %8 = ascendc.local_tensor.subindex %1[%7] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %8, %6, %3 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<16x32xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %4 : tensor<16x32xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_ub_l1_with_offsets(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %0 = asctile.copy %arg0[%c5_i32, %c0_i32] : tensor<16x32xf16, #asctile.local<UB>>, tensor<16x32xf16, #asctile.local<L1>>
  return %0 : tensor<16x32xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_ub_l1_single_row_fp16(%arg0: tensor<1x32xf16, #asctile.local<UB>>) -> tensor<1x32xf16, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x32xf16, #asctile.local<UB>> to !ascendc.local_tensor<1x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <1x32xf16>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c1_i32, %c1_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %4 = arith.muli %arg1, %c16_i32 : i32
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<1x32xf16>, i32, !ascendc.local_tensor<1x32xf16>
// CHECK-NEXT:    %6 = arith.muli %arg1, %c16_i32 : i32
// CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %1[%6] : !ascendc.local_tensor<1x32xf16>, i32, !ascendc.local_tensor<1x32xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %7, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<1x32xf16>, !ascendc.local_tensor<1x32xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<1x32xf16> to tensor<1x32xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<1x32xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_ub_l1_single_row_fp16(%arg0: tensor<1x32xf16, #asctile.local<UB>>) -> tensor<1x32xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<1x32xf16, #asctile.local<UB>>, tensor<1x32xf16, #asctile.local<L1>>
  return %0 : tensor<1x32xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_ub_l1_unaligned_height_fp16(%arg0: tensor<4x32xf16, #asctile.local<UB>>) -> tensor<4x32xf16, #asctile.local<L1>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<4x32xf16, #asctile.local<UB>> to !ascendc.local_tensor<4x32xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <4x32xf16>
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c4_i32, %c1_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %4 = arith.muli %arg1, %c16_i32 : i32
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<4x32xf16>, i32, !ascendc.local_tensor<4x32xf16>
// CHECK-NEXT:    %6 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %1[%6] : !ascendc.local_tensor<4x32xf16>, i32, !ascendc.local_tensor<4x32xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %7, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<4x32xf16>, !ascendc.local_tensor<4x32xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<4x32xf16> to tensor<4x32xf16, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<4x32xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @lower_copy_ub_l1_unaligned_height_fp16(%arg0: tensor<4x32xf16, #asctile.local<UB>>) -> tensor<4x32xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<4x32xf16, #asctile.local<UB>>, tensor<4x32xf16, #asctile.local<L1>>
  return %0 : tensor<4x32xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_b_unaligned_k_f32(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<56x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <56x64xf32>
// CHECK-NEXT:  %6 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %7 = arith.minsi %arg1, %c56_i32 : i32
// CHECK-NEXT:  %8 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %9 = arith.maxsi %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %7, %9 : i32
// CHECK-NEXT:  %13 = arith.minsi %6, %11 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %12, %21 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:      %25 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<56x64xf32>, i32, !ascendc.local_tensor<56x64xf32>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %13, %c0_i32, %13, %c56_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<56x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %12, %13, %c0_i32, %arg2, %c64_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<56x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = arith.muli %12, %c256_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c14336_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.divsi %16, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
// CHECK-NEXT:  scf.if %18 {
// CHECK-NEXT:    %20 = arith.muli %12, %c8_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c64_i32, %12 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<56x64xf32>, i32, !ascendc.local_tensor<56x64xf32>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c8_i32, %21, %12, %c0_i32) [ui16, ui16, ui16, f32] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<56x64xf32>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %19 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<56x64xf32> to tensor<56x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %19 : tensor<56x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  }

func.func @lower_load_gm_l1_b_unaligned_k_f32(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<56x64xf32, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst : tensor<?x?xf32, #asctile.global>, tensor<56x64xf32, #asctile.local<L1>>
  return %1 : tensor<56x64xf32, #asctile.local<L1>>
}

// -----

module attributes {ascendc.cv_ratio = 2 : i64} {
  // CHECK-LABEL: func.func @lower_copy_ub_l1_dual_vector_fp16(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf16, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf16>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xf16>
  // CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %4 = arith.muli %arg1, %c16_i32 : i32
  // CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c256_i32 : i32
  // CHECK-NEXT:    %7 = arith.addi %6, %c80_i32 : i32
  // CHECK-NEXT:    %8 = ascendc.local_tensor.subindex %1[%7] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
  // CHECK-NEXT:    ascendc.data_copy_l2 %8, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<16x32xf16>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L1>>
  // CHECK-NEXT:  return %3 : tensor<16x32xf16, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_dual_vector_fp16(%arg0: tensor<16x32xf16, #asctile.local<UB>>) -> tensor<16x32xf16, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = asctile.copy %arg0[%c5_i32, %c0_i32] : tensor<16x32xf16, #asctile.local<UB>>, tensor<16x32xf16, #asctile.local<L1>>
    return %0 : tensor<16x32xf16, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_dual_vector_fp32(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x32xf32>
  // CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c3_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %4 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:    %7 = arith.addi %6, %c24_i32 : i32
  // CHECK-NEXT:    %8 = ascendc.local_tensor.subindex %1[%7] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
  // CHECK-NEXT:    ascendc.data_copy_l2 %8, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %3 : tensor<16x32xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_dual_vector_fp32(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = asctile.copy %arg0[%c3_i32, %c0_i32] : tensor<16x32xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<L1>>
    return %0 : tensor<16x32xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_dual_vector_split_by_m_fp32(%arg0: tensor<8x128xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<8x128xf32, #asctile.local<UB>> to !ascendc.local_tensor<8x128xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x128xf32>
  // CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c8_i32, %c1_i32, %c15_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %4 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<8x128xf32>, i32, !ascendc.local_tensor<8x128xf32>
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:    %7 = arith.addi %6, %c64_i32 : i32
  // CHECK-NEXT:    %8 = ascendc.local_tensor.subindex %1[%7] : !ascendc.local_tensor<16x128xf32>, i32, !ascendc.local_tensor<16x128xf32>
  // CHECK-NEXT:    ascendc.data_copy_l2 %8, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x128xf32>, !ascendc.local_tensor<8x128xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x128xf32> to tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %3 : tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_dual_vector_split_by_m_fp32(%arg0: tensor<8x128xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = asctile.copy %arg0[%c8_i32, %c0_i32] : tensor<8x128xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<L1>>
    return %0 : tensor<16x128xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_dual_vector_split_by_n_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x64xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x128xf32>
  // CHECK-NEXT:  %2 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c7_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %4 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %0[%4] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:    %7 = arith.addi %6, %c1024_i32 : i32
  // CHECK-NEXT:    %8 = ascendc.local_tensor.subindex %1[%7] : !ascendc.local_tensor<16x128xf32>, i32, !ascendc.local_tensor<16x128xf32>
  // CHECK-NEXT:    ascendc.data_copy_l2 %8, %5, %2 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x128xf32>, !ascendc.local_tensor<16x64xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x128xf32> to tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %3 : tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_dual_vector_split_by_n_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = asctile.copy %arg0[%c0_i32, %c64_i32] : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<L1>>
    return %0 : tensor<16x128xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_join_by_m_fp32(%arg0: tensor<8x128xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<8x128xf32, #asctile.local<UB>> to !ascendc.local_tensor<8x128xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x128xf32>
  // CHECK-NEXT:  %2 = ascendc.get_sub_block_idx : i32
  // CHECK-NEXT:  %3 = arith.muli %2, %c64_i32 : i32
  // CHECK-NEXT:  %4 = ascendc.construct !ascendc.data_copy_params(%c8_i32, %c1_i32, %c15_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %0[%6] : !ascendc.local_tensor<8x128xf32>, i32, !ascendc.local_tensor<8x128xf32>
  // CHECK-NEXT:    %8 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:    %9 = arith.addi %8, %3 : i32
  // CHECK-NEXT:    %10 = ascendc.local_tensor.subindex %1[%9] : !ascendc.local_tensor<16x128xf32>, i32, !ascendc.local_tensor<16x128xf32>
  // CHECK-NEXT:    ascendc.data_copy_l2 %10, %7, %4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x128xf32>, !ascendc.local_tensor<8x128xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x128xf32> to tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %5 : tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_join_by_m_fp32(%arg0: tensor<8x128xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<join_by_m>} : tensor<8x128xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<L1>>
    return %0 : tensor<16x128xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_join_by_n_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x64xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x128xf32>
  // CHECK-NEXT:  %2 = ascendc.get_sub_block_idx : i32
  // CHECK-NEXT:  %3 = arith.muli %2, %c1024_i32 : i32
  // CHECK-NEXT:  %4 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c7_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:    %6 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:    %7 = ascendc.local_tensor.subindex %0[%6] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:    %8 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:    %9 = arith.addi %8, %3 : i32
  // CHECK-NEXT:    %10 = ascendc.local_tensor.subindex %1[%9] : !ascendc.local_tensor<16x128xf32>, i32, !ascendc.local_tensor<16x128xf32>
  // CHECK-NEXT:    ascendc.data_copy_l2 %10, %7, %4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x128xf32>, !ascendc.local_tensor<16x64xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x128xf32> to tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %5 : tensor<16x128xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_join_by_n_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x128xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<join_by_n>} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x128xf32, #asctile.local<L1>>
    return %0 : tensor<16x128xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_full_vec0_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x64xf32, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x64xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x64xf32>
  // CHECK-NEXT:  %2 = ascendc.get_sub_block_idx : i32
  // CHECK-NEXT:  %3 = arith.cmpi eq, %2, %c0_i32 : i32
  // CHECK-NEXT:  %4 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c7_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.if %3 {
  // CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:      %6 = arith.muli %arg1, %c8_i32 : i32
  // CHECK-NEXT:      %7 = ascendc.local_tensor.subindex %0[%6] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:      %8 = arith.muli %arg1, %c128_i32 : i32
  // CHECK-NEXT:      %9 = ascendc.local_tensor.subindex %1[%8] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
  // CHECK-NEXT:      ascendc.data_copy_l2 %9, %7, %4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x64xf32>, !ascendc.local_tensor<16x64xf32>, !ascendc.data_copy_params
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L1>>
  // CHECK-NEXT:  return %5 : tensor<16x64xf32, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_full_vec0_fp32(%arg0: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<16x64xf32, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<full_vec0>} : tensor<16x64xf32, #asctile.local<UB>>, tensor<16x64xf32, #asctile.local<L1>>
    return %0 : tensor<16x64xf32, #asctile.local<L1>>
  }

  // CHECK-LABEL: func.func @lower_copy_ub_l1_full_vec1_fp16(%arg0: tensor<16x64xf16, #asctile.local<UB>>) -> tensor<16x64xf16, #asctile.local<L1>> {
  // CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x64xf16, #asctile.local<UB>> to !ascendc.local_tensor<16x64xf16>
  // CHECK-NEXT:  %1 = ascendc.local_tensor_auto a1() : <16x64xf16>
  // CHECK-NEXT:  %2 = ascendc.get_sub_block_idx : i32
  // CHECK-NEXT:  %3 = arith.cmpi eq, %2, %c1_i32 : i32
  // CHECK-NEXT:  %4 = ascendc.construct !ascendc.data_copy_params(%c16_i32, %c1_i32, %c3_i32, %c0_i32) : i32, i32, i32, i32
  // CHECK-NEXT:  scf.if %3 {
  // CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
  // CHECK-NEXT:      %6 = arith.muli %arg1, %c16_i32 : i32
  // CHECK-NEXT:      %7 = ascendc.local_tensor.subindex %0[%6] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
  // CHECK-NEXT:      %8 = arith.muli %arg1, %c256_i32 : i32
  // CHECK-NEXT:      %9 = ascendc.local_tensor.subindex %1[%8] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
  // CHECK-NEXT:      ascendc.data_copy_l2 %9, %7, %4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x64xf16>, !ascendc.local_tensor<16x64xf16>, !ascendc.data_copy_params
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x64xf16> to tensor<16x64xf16, #asctile.local<L1>>
  // CHECK-NEXT:  return %5 : tensor<16x64xf16, #asctile.local<L1>>
  // CHECK-NEXT:}
  func.func @lower_copy_ub_l1_full_vec1_fp16(%arg0: tensor<16x64xf16, #asctile.local<UB>>) -> tensor<16x64xf16, #asctile.local<L1>> {
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {distrib = #asctile.distrib_mode<full_vec1>} : tensor<16x64xf16, #asctile.local<UB>>, tensor<16x64xf16, #asctile.local<L1>>
    return %0 : tensor<16x64xf16, #asctile.local<L1>>
  }
}
