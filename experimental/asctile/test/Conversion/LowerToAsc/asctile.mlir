// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asclower-asctile -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @lower_tensor_static(%arg0: memref<*xf32, 22>) -> tensor<16x8xf32, #asctile.global> {
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<16x8xf32>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<16x8xf32>, memref<*xf32, 22>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.global_tensor<16x8xf32> to tensor<16x8xf32, #asctile.global>
// CHECK-NEXT:  return %1 : tensor<16x8xf32, #asctile.global>
// CHECK-NEXT:}
func.func @lower_tensor_static(%arg0: memref<*xf32, 22>) -> tensor<16x8xf32, #asctile.global> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x8xf32, #asctile.global>
  return %0 : tensor<16x8xf32, #asctile.global>
}

// CHECK-LABEL: func.func @lower_tensor_dynamic(%arg0: memref<*xi32, 22>, %arg1: i32, %arg2: i32) -> tensor<?x8x?xi32, #asctile.global> {
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x8x?xi32>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x8x?xi32>, memref<*xi32, 22>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.global_tensor<?x8x?xi32> to tensor<?x8x?xi32, #asctile.global>
// CHECK-NEXT:  return %1 : tensor<?x8x?xi32, #asctile.global>
// CHECK-NEXT:}
func.func @lower_tensor_dynamic(%arg0: memref<*xi32, 22>, %arg1: i32, %arg2: i32) -> tensor<?x8x?xi32, #asctile.global> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xi32, 22>, tensor<?x8x?xi32, #asctile.global>
  return %0 : tensor<?x8x?xi32, #asctile.global>
}

// CHECK-LABEL: func.func @lower_tensor_l2_cache_disabled(%arg0: memref<*xf32, 22>) -> tensor<16x8xf32, #asctile.global> {
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<16x8xf32>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<16x8xf32>, memref<*xf32, 22>
// CHECK-NEXT:  ascendc.global_tensor.set_l2_cache_hint %0 {mode = 0 : i32, rwMode = 3 : i32} : !ascendc.global_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.global_tensor<16x8xf32> to tensor<16x8xf32, #asctile.global>
// CHECK-NEXT:  return %1 : tensor<16x8xf32, #asctile.global>
// CHECK-NEXT:}
func.func @lower_tensor_l2_cache_disabled(%arg0: memref<*xf32, 22>) -> tensor<16x8xf32, #asctile.global> {
  %0 = asctile.tensor %arg0() {disableL2Cache} : memref<*xf32, 22>, tensor<16x8xf32, #asctile.global>
  return %0 : tensor<16x8xf32, #asctile.global>
}

// CHECK-LABEL: func.func @lower_accumulator() -> tensor<64x256xf32, #asctile.local<L0C>> {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() : <64x256xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<64x256xf32> to tensor<64x256xf32, #asctile.local<L0C>>
// CHECK-NEXT:  return %1 : tensor<64x256xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_accumulator() -> tensor<64x256xf32, #asctile.local<L0C>> {
  %0 = asctile.accumulator : tensor<64x256xf32, #asctile.local<L0C>>
  return %0 : tensor<64x256xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_accumulator_with_bias(%arg0: tensor<256xf32, #asctile.local<BT>>) -> tensor<64x256xf32, #asctile.local<L0C>> {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() : <64x256xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<64x256xf32> to tensor<64x256xf32, #asctile.local<L0C>>
// CHECK-NEXT:  return %1 : tensor<64x256xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_accumulator_with_bias(%arg0: tensor<256xf32, #asctile.local<BT>>) -> tensor<64x256xf32, #asctile.local<L0C>> {
  %0 = asctile.accumulator %arg0 : tensor<64x256xf32, #asctile.local<L0C>>, tensor<256xf32, #asctile.local<BT>>
  return %0 : tensor<64x256xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<8x32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<8x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<8x32xf32> to tensor<8x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %2 : tensor<8x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<8x32xf32, #asctile.local<UB>> {
  %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<8x32xf32, #asctile.local<UB>>
  return %0 : tensor<8x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<16x1xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x1xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x1xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<1xf32, #asctile.local<UB>> to !ascendc.local_tensor<1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  ascendc.duplicate_l2 %2, %3, %c0_i64 : !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  ascendc.broadcast %5, %0, %c1_i32, %c16_i32, %c32_i32, %c1_i32, %c16_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x1xf32>, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %4, %6 : tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<16x1xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>) {
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %arg1 : tensor<16x1xf32, #asctile.local<UB>> to tensor<16x32xf32, #asctile.local<UB>>
  return %0, %1 : tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_softmax(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <128xui8>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c1_i32 : i32, "srcK" = %c16_i32 : i32, "oriSrcM" = %c1_i32 : i32, "oriSrcK" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %0, %2, %3, %4 {operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 1, 1>} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<128xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_softmax(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.softmax %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_softmax_2D(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <2176xui8>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c16_i32 : i32, "srcK" = %c32_i32 : i32, "oriSrcM" = %c16_i32 : i32, "oriSrcK" = %c32_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %0, %2, %3, %4 {operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 1, 1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<2176xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_softmax_2D(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<UB>> {
  %0 = asctile.softmax %arg0 : tensor<16x32xf32, #asctile.local<UB>>
  return %0 : tensor<16x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_sum(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_sum %1, %0, %2, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_sum(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <sum> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_min(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_min %1, %0, %2, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_min(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <min> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_max(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_max %1, %0, %2, %c64_i32, %c32_i32 {pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_max(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <max> %arg0 {dims = [1 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_with_reuse(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <0xui8>
// CHECK-NEXT:  ascendc.reduce_max %1, %0, %2, %c64_i32, %c32_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<0xui8>, i32, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_with_reuse(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <max> %arg0 {asctile.reuse_source, dims = [1 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_prod(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_prod %1, %0, %2, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_prod(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <prod> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_inline_vf(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> (tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>) {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : tensor<32xi32, #asctile.local<UB>> to !ascendc.local_tensor<32xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  ascvf.vf_group dst(%2 : !ascendc.local_tensor<32xf16>) {
// CHECK-NEXT:    emitasc.vec_scope {
// CHECK-NEXT:      emitasc.verbatim ";;; // $0" %2 : !ascendc.local_tensor<32xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {groupType = !ascendc.local_tensor<32xf16>}
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <32xi16>
// CHECK-NEXT:  ascvf.vf_group dst(%4 : !ascendc.local_tensor<32xi16>) src(%1, %0 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xi32>) {
// CHECK-NEXT:    emitasc.vec_scope {
// CHECK-NEXT:      emitasc.verbatim ";;; // $0 $1 $2" %4, %1, %0 : !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {groupType = !ascendc.local_tensor<32xi16>}
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<32xi16> to tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:  return %3, %5 : tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_inline_vf(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> (tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>) {
  %0 = asctile.inline_vf() ";;; // $0" : () -> tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.inline_vf(%arg0, %arg1) ";;; // $0 $1 $2" : (tensor<32xf32, #asctile.local<UB>>, tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi16, #asctile.local<UB>>
  return %0, %1 : tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_layer_norm(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: tensor<256xf32, #asctile.local<UB>>, %arg2: tensor<256xf32, #asctile.local<UB>>) -> (tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg2 : tensor<256xf32, #asctile.local<UB>> to !ascendc.local_tensor<256xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg1 : tensor<256xf32, #asctile.local<UB>> to !ascendc.local_tensor<256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <4xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <4xf32>
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.layer_norm_separate_tiling("aLength" = %c4_i32 : i32, "rLength" = %c256_i32 : i32, "halfAddRepeatTimes" = %c0_i32 : i32, "rHeadLength" = %c256_i32 : i32, "k2Rec" = %cst_0 : f32, "k2RRec" = %cst : f32, "inputXSize" = %c1024_i32 : i32, "meanVarSize" = %c4_i32 : i32, "numberOfTmpBuf" = %c3_i32 : i32, "varianceTmpTensorPos" = %c3072_i32 : i32, "varianceTmpTensorSize" = %c4_i32 : i32, "tmpBufSize" = %c3076_i32 : i32, "oneTmpSize" = %c1024_i32 : i32, "firstTmpStartPos" = %c0_i32 : i32, "secondTmpStartPos" = %c1024_i32 : i32, "thirdTmpStartPos" = %c2048_i32 : i32, "loopRound" = %c1_i32 : i32, "inputRoundSize" = %c1024_i32 : i32, "inputTailSize" = %c0_i32 : i32, "inputTailPos" = %c1024_i32 : i32, "meanVarRoundSize" = %c4_i32 : i32, "meanVarTailSize" = %c0_i32 : i32, "meanVarTailPos" = %c4_i32 : i32, "arCurLength" = %c1024_i32 : i32, "aCurLength" = %c4_i32 : i32, "rValueBack" = %cst_0 : f32)
// CHECK-NEXT:  %7 = emitasc.init_struct !ascendc.layer_norm_para("aLength" = %c4_i32 : i32, "rLength" = %c256_i32 : i32, "rLengthWithPadding" = %c256_i32 : i32)
// CHECK-NEXT:  %8 = ascendc.local_tensor_auto veccalc() : <12304xui8>
// CHECK-NEXT:  ascendc.layer_norm %3, %4, %5, %2, %1, %0, %cst_1, %6, %7, %8 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, f32, !ascendc.layer_norm_separate_tiling, !ascendc.layer_norm_para, !ascendc.local_tensor<12304xui8>
// CHECK-NEXT:  %9 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  %10 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<4xf32> to tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %11 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<4xf32> to tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %9, %10, %11 : tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_layer_norm(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: tensor<256xf32, #asctile.local<UB>>, %arg2: tensor<256xf32, #asctile.local<UB>>) -> (tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>) {
  %cst = arith.constant 1.000000e-05 : f32
  %output, %mean, %outputVarRstd = asctile.layer_norm %arg0, %arg1, %arg2, %cst : tensor<4x256xf32, #asctile.local<UB>>, tensor<256xf32, #asctile.local<UB>>, tensor<256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
  return %output, %mean, %outputVarRstd : tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_rms_norm(%arg0: tensor<16x32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>, %arg2: f32) -> tensor<16x32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <4096xui8>
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.rmsnorm_tiling("bLength" = %c16_i32 : i32, "sLength" = %c1_i32 : i32, "hLength" = %c32_i32 : i32, "originalHLength" = %c32_i32 : i32, "reciprocalOfHLength" = %cst : f32, "mainBshLength" = %c512_i32 : i32, "mainBsLength" = %c16_i32 : i32, "mainBsLengthAlign" = %c16_i32 : i32, "loopRound" = %c1_i32 : i32, "tailBshLength" = %c0_i32 : i32, "inputTailPos" = %c512_i32 : i32, "tailBsLength" = %c0_i32 : i32)
// CHECK-NEXT:  ascendc.rms_norm %2, %1, %0, %arg2, %4, %3 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32xf32>, f32, !ascendc.rmsnorm_tiling, !ascendc.local_tensor<4096xui8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_rms_norm(%arg0: tensor<16x32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>, %arg2: f32) -> tensor<16x32xf32, #asctile.local<UB>> {
  %0 = asctile.rms_norm %arg0, %arg1, %arg2 : tensor<16x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<16x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cube_group(%arg0: tensor<16x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x16xf32, #asctile.local<L0B>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
// CHECK-NEXT:  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %2 = asctile.matmul %arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0B>> -> tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:  return %1 : tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_cube_group(%arg0: tensor<16x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x16xf32, #asctile.local<L0B>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %0 = asctile.cube_group(%arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0B>>) {
    %1 = asctile.matmul %arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0B>> -> tensor<16x16xf32, #asctile.local<L0C>>
    asctile.yield %1 : tensor<16x16xf32, #asctile.local<L0C>>
  } : tensor<16x16xf32, #asctile.local<L0C>>
  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_vector_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = ascendc.if_aiv -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT:    %2 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    ascendc.yield %3 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_vector_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_power_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <1536xui8>
// CHECK-NEXT:  ascendc.power %2, %1, %0, %3, %c16_i32, %false {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<1536xui8>, i32, i1
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %4 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_power_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = asctile.power %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_power_f16(%arg0: tensor<32xf16, #asctile.local<UB>>, %arg1: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<32xf16, #asctile.local<UB>> to !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf16, #asctile.local<UB>> to !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2048xui8>
// CHECK-NEXT:  ascendc.power %2, %1, %0, %3, %c32_i32, %false {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<2048xui8>, i32, i1
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  return %4 : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_power_f16(%arg0: tensor<32xf16, #asctile.local<UB>>, %arg1: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
  %0 = asctile.power %arg0, %arg1 : tensor<32xf16, #asctile.local<UB>>
  return %0 : tensor<32xf16, #asctile.local<UB>>
}

// -----

module attributes {asc.compilation_arch = "c310"} {
// CHECK-LABEL: func.func @lower_power_i32_c310(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <0xui8>
// CHECK-NEXT:  ascendc.power %2, %1, %0, %3, %c16_i32, %false {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<0xui8>, i32, i1
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %4 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
  func.func @lower_power_i32_c310(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
    %0 = asctile.power %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
    return %0 : tensor<16xi32, #asctile.local<UB>>
  }

// CHECK-LABEL: func.func @lower_power_f16_c310(%arg0: tensor<32xf16, #asctile.local<UB>>, %arg1: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<32xf16, #asctile.local<UB>> to !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf16, #asctile.local<UB>> to !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <384xui8>
// CHECK-NEXT:  ascendc.power %2, %1, %0, %3, %c32_i32, %false {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<384xui8>, i32, i1
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  return %4 : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
  func.func @lower_power_f16_c310(%arg0: tensor<32xf16, #asctile.local<UB>>, %arg1: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
    %0 = asctile.power %arg0, %arg1 : tensor<32xf16, #asctile.local<UB>>
    return %0 : tensor<32xf16, #asctile.local<UB>>
  }
}

// -----

// CHECK-LABEL: func.func @lower_assert(%arg0: i1) {
// CHECK:        %0 = arith.xori %arg0, %true : i1
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     ascendc.printf  {desc = "Assertion failed at {{.*}}:{{[0-9]+}}:{{[0-9]+}}: assertion message\0A"} :
// CHECK-NEXT:     ascendc.trap
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_assert(%cond: i1) {
  asctile.assert %cond, "assertion message" : i1
  return
}

// CHECK-LABEL: func.func @lower_dump_local_tensor(%arg0: tensor<32xf32, #asctile.local<UB>>) {
// CHECK:        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:   %1 = ascendc.local_tensor.get_shape_info %0 : !ascendc.local_tensor<32xf32>, !ascendc.shape_info
// CHECK-NEXT:   ascendc.dump_tensor %0, %c0_i32, %c32_i32, %1 : !ascendc.local_tensor<32xf32>, i32, i32, !ascendc.shape_info
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_dump_local_tensor(%arg0: tensor<32xf32, #asctile.local<UB>>) {
  asctile.dump_tensor %arg0 : tensor<32xf32, #asctile.local<UB>>
  return
}

// CHECK-LABEL: func.func @lower_dump_global_tensor_static(%arg0: memref<*xf32, 22>) {
// CHECK:        %1 = ascendc.global_tensor : !ascendc.global_tensor<4x8x16xf32>
// CHECK-NEXT:   ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<4x8x16xf32>, memref<*xf32, 22>
// CHECK-NEXT:   %2 = ascendc.global_tensor.get_phy_addr %1, %0 : !ascendc.global_tensor<4x8x16xf32>, memref<*xf32, 22>, ui64
// CHECK-NEXT:   ascendc.printf %2, %c4_i32, %c8_i32, %c16_i32 {desc = "GlobalTensor(addr=%p, dtype=f32, shape=[%d, %d, %d])\0A"} : memref<*xf32, 22>, i32, i32, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_dump_global_tensor_static(%arg0: memref<*xf32, 22>) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<4x8x16xf32, #asctile.global>
  asctile.dump_tensor %0 : tensor<4x8x16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_dump_global_tensor_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32) {
// CHECK:        %1 = ascendc.global_tensor : !ascendc.global_tensor<?x?x?xf32>
// CHECK-NEXT:   ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<?x?x?xf32>, memref<*xf32, 22>
// CHECK-NEXT:   %2 = ascendc.global_tensor.get_phy_addr %1, %0 : !ascendc.global_tensor<?x?x?xf32>, memref<*xf32, 22>, ui64
// CHECK-NEXT:   ascendc.printf %2, %arg1, %arg2, %arg3 {desc = "GlobalTensor(addr=%p, dtype=f32, shape=[%d, %d, %d])\0A"} : memref<*xf32, 22>, i32, i32, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_dump_global_tensor_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0(%arg1, %arg2, %arg3) : memref<*xf32, 22>, tensor<?x?x?xf32, #asctile.global>
  asctile.dump_tensor %0 : tensor<?x?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_dump_global_tensor_mixed(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) {
// CHECK:        %1 = ascendc.global_tensor : !ascendc.global_tensor<?x?x32xf32>
// CHECK-NEXT:   ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<?x?x32xf32>, memref<*xf32, 22>
// CHECK-NEXT:   %2 = ascendc.global_tensor.get_phy_addr %1, %0 : !ascendc.global_tensor<?x?x32xf32>, memref<*xf32, 22>, ui64
// CHECK-NEXT:   ascendc.printf %2, %arg1, %arg2, %c32_i32 {desc = "GlobalTensor(addr=%p, dtype=f32, shape=[%d, %d, %d])\0A"} : memref<*xf32, 22>, i32, i32, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_dump_global_tensor_mixed(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?x32xf32, #asctile.global>
  asctile.dump_tensor %0 : tensor<?x?x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_inline(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: tensor<?x?x32xf32, #asctile.global>, %arg3: tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg3 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg2 : tensor<?x?x32xf32, #asctile.global> to !ascendc.global_tensor<?x?x32xf32>
// CHECK-NEXT:  emitasc.verbatim "123 $0;" %arg0, %arg1, %1, %0 : memref<*xf32, 22>, i32, !ascendc.global_tensor<?x?x32xf32>, !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_inline(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: tensor<?x?x32xf32, #asctile.global>, %arg3: tensor<32xf32, #asctile.local<UB>>) {
  asctile.inline "123 $0;" %arg0, %arg1, %arg2, %arg3 : memref<*xf32, 22>, i32, tensor<?x?x32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  return
}
