// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -ascvf-find-vf-group -ascvf-lower-to-reg -canonicalize -cse -ascvf-dispatch-vf-fusion -canonicalize -cse -ascvf-materialize-load-store %s | FileCheck %s

// CHECK-LABEL: func.func @general_test(%arg0: !ascendc.que_bind<gm, vecin, 1>) {
// CHECK:   ascvf.vf_group dst(%0 : !ascendc.local_tensor<256xf32>) src(%2, %1 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>) {
// CHECK:     %3 = ascendc.local_tensor.get_phy_addr_v2 %1 : !ascendc.local_tensor<256xf32>, memref<256xf32, 26>
// CHECK:     %4 = ascendc.local_tensor.get_phy_addr_v2 %2 : !ascendc.local_tensor<256xf32>, memref<256xf32, 26>
// CHECK:     %5 = ascendc.local_tensor.get_phy_addr_v2 %0 : !ascendc.local_tensor<256xf32>, memref<256xf32, 26>
// CHECK:     emitasc.vec_scope {
// CHECK:       %[[MASK_ALL:[0-9]+]] = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK:       %[[VEC_LEN:[0-9]+]] = ascendc.get_vec_len : index
// CHECK:       %[[ONE_REPEAT_SIZE:[0-9]+]] = arith.divsi %[[VEC_LEN]], %c4 : index
// CHECK:       %[[UB:[0-9]+]] = arith.ceildivsi %c256, %[[ONE_REPEAT_SIZE]] : index
// CHECK:       emitasc.vf_for %[[UB]] : index {
// CHECK:       ^bb0(%arg1: index):
// CHECK:         %[[OFFSET:[0-9]+]] = arith.muli %arg1, %[[ONE_REPEAT_SIZE]] : index
// CHECK:         %[[UPDATE_MASK:[0-9]+]] = ascendc.update_mask f32, %[[COUNT:[0-9]+]] : memref<1xui32>
// CHECK:         %[[ADDR0:[0-9]+]] = emitasc.ptr_offset %3[%[[OFFSET]]] : memref<256xf32, 26>, memref<256xf32, 26>
// CHECK:         ascendc.data_copy_vld_reg %[[REG0:[0-9]+]], %[[ADDR0]] {dist = 0 : i32} : !ascendc.reg_tensor<f32>, memref<256xf32, 26>
// CHECK:         %[[ADDR1:[0-9]+]] = emitasc.ptr_offset %4[%[[OFFSET]]] : memref<256xf32, 26>, memref<256xf32, 26>
// CHECK:         ascendc.data_copy_vld_reg %[[REG1:[0-9]+]], %[[ADDR1]] {dist = 0 : i32} : !ascendc.reg_tensor<f32>, memref<256xf32, 26>
// CHECK:         ascendc.add_reg %[[REG2:[0-9]+]], %[[REG0]], %[[REG1]], %[[MASK_ALL]] : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:         ascendc.mul_reg %[[REG3:[0-9]+]], %[[REG2]], %[[REG1]], %[[MASK_ALL]] : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:         %[[ADDR2:[0-9]+]] = emitasc.ptr_offset %5[%[[OFFSET]]] : memref<256xf32, 26>, memref<256xf32, 26>
// CHECK:         ascendc.data_copy_vst_reg %[[ADDR2]], %[[REG3]], %[[UPDATE_MASK]] {dist = 16 : i32} : memref<256xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       }
// CHECK:     }
// CHECK:   } {groupType = !ascendc.local_tensor<256xf32>}
// CHECK: }
func.func @general_test(%que_bind: !ascendc.que_bind<gm, vecin, 1>) {
  %c256_i32 = arith.constant 256 : i32
  %dst =  ascendc.local_tensor_auto veccalc() : <256xf32>
  %src0 = ascendc.local_tensor_auto veccalc() : <256xf32>
  %src1 = ascendc.local_tensor_auto veccalc() : <256xf32>
  ascendc.add_l2 %dst, %src0, %src1, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  ascendc.mul_l2 %dst, %dst, %src1, %c256_i32 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, i32
  return
}

// CHECK-LABEL: func.func @softmax_kernel
// CHECK: ascvf.vf_group
// CHECK:   emitasc.vec_scope {
// CHECK:     ascendc.duplicate
// CHECK:     ascendc.duplicate
// CHECK:     emitasc.vf_for
// CHECK:       ascendc.data_copy_vld_reg
// CHECK:       ascendc.max_reg
// CHECK:     }
// CHECK:     ascendc.reduce_max_reg
// CHECK:     ascendc.duplicate_reg
// CHECK:     emitasc.vf_for
// CHECK:       ascendc.data_copy_vld_reg
// CHECK:       ascendc.sub_reg
// CHECK:       ascendc.exp_reg
// CHECK:       ascendc.data_copy_vst_reg
// CHECK:       ascendc.add_reg
// CHECK:     }
// CHECK:     ascendc.reduce_sum_reg
// CHECK:     ascendc.duplicate_reg
// CHECK:     emitasc.vf_for
// CHECK:       ascendc.data_copy_vld_reg
// CHECK:       ascendc.div_reg
// CHECK:       ascendc.data_copy_vst_reg
// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: ascendc.data_copy_pad_l2_ext
func.func @softmax_kernel(%arg0: memref<*xf32, 22>) {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
  %4 = ascendc.local_tensor_auto veccalc() output : <1024xf32>
  %5 = ascendc.local_tensor_auto veccalc() : <1024xf32>
  %6 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %7 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %f = ascendc.local_tensor_auto veccalc() : <1024xf32>
  %z = ascendc.local_tensor_auto veccalc() : <1024xf32>
  %y = ascendc.local_tensor_auto veccalc() : <1024xf32>
  %11 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %12 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %x = ascendc.local_tensor_auto veccalc() input : <1024xf32>

  %ext_params = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascendc.reduce_max_l2 %12, %x, %11, %c1024_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<16xf32>, i64, i64
  ascendc.duplicate_l2 %y, %12, %c1024_i64 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.sub_l2 %z, %x, %y, %c1024_i64 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i64
  ascendc.exp_l2 %f, %z, %c1024_i64 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i64
  ascendc.reduce_sum_l2 %7, %f, %6, %c1024_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<16xf32>, i64
  ascendc.duplicate_l2 %5, %7, %c1024_i64 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.div_l2 %4, %f, %5, %c1024_i64 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %0, %4, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.data_copy_ext_params
  return
}

// -----

// CHECK-LABEL: func.func @layernorm_v4_kernel
// CHECK: ascvf.vf_group
// CHECK:   emitasc.vec_scope {
// CHECK:     emitasc.vf_for %c2 : index {
// CHECK:     ^bb0(%arg1: index):
// CHECK:       emitasc.vf_for %c16 : index {
// CHECK:       ^bb0(%arg2: index):
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.add_reg
// CHECK:       }
// CHECK:       ascendc.reduce_sum_reg
// CHECK:       ascendc.data_copy_vst_reg
// CHECK:       emitasc.vf_for %c16 : index {
// CHECK:       ^bb0(%arg2: index):
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.duplicate_reg
// CHECK:         ascendc.div_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.mul_reg
// CHECK:         ascendc.add_reg
// CHECK:       }
// CHECK:       ascendc.reduce_sum_reg
// CHECK:       ascendc.data_copy_vst_reg
// CHECK:       emitasc.vf_for %c16 : index {
// CHECK:       ^bb0(%arg2: index):
// CHECK:         ascendc.div_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:       }
// CHECK:     }
// CHECK:     ascendc.local_mem_bar VEC_STORE, VEC_LOAD
// CHECK:     emitasc.vf_for %c2 : index {
// CHECK:     ^bb0(%arg1: index):
// CHECK:       emitasc.vf_for %c16 : index {
// CHECK:       ^bb0(%arg2: index):
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.mul_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.sub_reg
// CHECK:         ascendc.add_reg
// CHECK:         ascendc.sqrt_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.div_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.sub_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.mul_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.mul_reg
// CHECK:         ascendc.data_copy_vst_reg
// CHECK:         ascendc.data_copy_vld_reg
// CHECK:         ascendc.add_reg
// CHECK:         }
// CHECK:     }
// CHECK:   }
// CHECK: } {groupType = !ascendc.local_tensor<2x1024xf32>}
module attributes {asc.vf_vec_len = 256 : i32} {
  func.func @layernorm_v4_kernel(%arg0: memref<*xf32, 22>) {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.024000e+03 : f32
    %cst_0 = arith.constant 5.120000e+02 : f32
    %c2_i32 = arith.constant 2 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
    %1 = ascendc.local_tensor_auto veccalc() : <2x1xf32>
    %2 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
    %4 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %6 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %7 = ascendc.local_tensor_auto veccalc() : <2x1xf32>
    %8 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %9 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %10 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %11 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %12 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %13 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %14 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %15 = ascendc.local_tensor_auto veccalc() : <2x1024xf32>
    %16 = ascendc.local_tensor_auto veccalc() : <0xui8>
    %17 = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
    ascendc.reduce_sum %1, %2, %3, %c2_i32, %c1024_i32 {pattern = 1 : i32} : !ascendc.local_tensor<2x1xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
    ascendc.broadcast %4, %1, %c1_i32, %c2_i32, %c1024_i32, %c1_i32, %c2_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1xf32>, i32, i32, i32, i32, i32, i32
    ascendc.divs_l2 %5, %4, %cst, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, f32, i64
    ascendc.mul_l2 %6, %2, %2, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.reduce_sum %7, %6, %16, %c2_i32, %c1024_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<2x1xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<0xui8>, i32, i32
    ascendc.broadcast %6, %7, %c1_i32, %c2_i32, %c1024_i32, %c1_i32, %c2_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1xf32>, i32, i32, i32, i32, i32, i32
    ascendc.divs_l2 %8, %6, %cst, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, f32, i64
    ascendc.mul_l2 %6, %5, %5, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.sub_l2 %6, %8, %6, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.adds_l2 %6, %6, %cst_0, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, f32, i64
    ascendc.sqrt_l2 %6, %6, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.div_l2 %6, %9, %6, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.sub_l2 %10, %2, %5, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.mul_l2 %13, %10, %6, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.mul_l2 %14, %13, %12, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.add_l2 %15, %14, %11, %c2048_i64 : !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.local_tensor<2x1024xf32>, i64
    ascendc.data_copy_pad_l2_ext %0, %15, %17 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<2x1024xf32>, !ascendc.data_copy_ext_params
    return
  }
}
