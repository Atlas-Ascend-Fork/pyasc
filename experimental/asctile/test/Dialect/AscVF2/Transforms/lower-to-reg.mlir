// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

// CHECK-LABEL: func.func @skip_unmarked_function() -> vector<64xf32>
// CHECK-NEXT: %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK-NEXT: return %cst : vector<64xf32>
// CHECK-NEXT: }
func.func @skip_unmarked_function() -> vector<64xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  return %cst : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_marked_function() -> vector<64xf32>
// CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: %1 = ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
// CHECK-NEXT: ascendc.duplicate_reg %0, %cst, %1 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %0 : !ascendc.reg_tensor<f32> to vector<64xf32>
// CHECK-NEXT: return %2 : vector<64xf32>
// CHECK-NEXT: }
func.func @lower_marked_function() -> vector<64xf32> attributes {ascvf2.vector_func} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  return %cst : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_scalar_ops(%arg0: vector<64xi32>, %arg1: vector<64xi32>, %arg2: i32) -> vector<64xi32>
// CHECK: ascvf2.simple_mask_reg i32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.shift_lefts_reg
// CHECK-NOT: ascvf2.shls
func.func @lower_scalar_ops(%arg0: vector<64xi32>, %arg1: vector<64xi32>, %arg2: i32) -> vector<64xi32> attributes {ascvf2.vector_func} {
  %result = ascvf2.shls %arg0, %arg2 : vector<64xi32>, i32
  return %result : vector<64xi32>
}

// CHECK-LABEL: func.func @lower_math_and_scalar_duplicate(%arg0: vector<64xf32>, %arg1: f32) -> vector<64xf32>
// CHECK: ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.exp_reg
// CHECK: ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.duplicate_reg
// CHECK: ascendc.add_reg
// CHECK-NOT: math.exp
// CHECK-NOT: ascvf2.adds
func.func @lower_math_and_scalar_duplicate(%arg0: vector<64xf32>, %arg1: f32) -> vector<64xf32> attributes {ascvf2.vector_func} {
  %exp = math.exp %arg0 : vector<64xf32>
  %result = ascvf2.adds %exp, %arg1 : vector<64xf32>, f32
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_partial_width_masked_arithmetic(%arg0: vector<32xf32>, %arg1: vector<32xf32>, %arg2: vector<32xf32>) -> vector<32xf32>
// CHECK: ascvf2.simple_mask_reg f32 static 32 : !ascendc.mask_reg
// CHECK: ascvf2.simple_mask_reg f32 static 5 : !ascendc.mask_reg
// CHECK: ascendc.add_reg
// CHECK-NOT: vector.mask
func.func @lower_partial_width_masked_arithmetic(%arg0: vector<32xf32>, %arg1: vector<32xf32>, %arg2: vector<32xf32>) -> vector<32xf32> attributes {ascvf2.vector_func} {
  %full = arith.addf %arg0, %arg1 : vector<32xf32>
  %mask = vector.constant_mask [5] : vector<32xi1>
  %result = vector.mask %mask { arith.addf %full, %arg2 : vector<32xf32> } : vector<32xi1> -> vector<32xf32>
  return %result : vector<32xf32>
}

// CHECK-LABEL: func.func @lower_dynamic_masked_arithmetic(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: index) -> vector<4xi32>
// CHECK: ascvf2.simple_mask_reg i32 dynamic %arg2 : !ascendc.mask_reg
// CHECK: ascendc.add_reg {{.*}} : !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.mask_reg
// CHECK-NOT: vector.create_mask
// CHECK-NOT: arith.index_cast
// CHECK-NOT: vector.mask
func.func @lower_dynamic_masked_arithmetic(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: index) -> vector<4xi32> attributes {ascvf2.vector_func} {
  %mask = vector.create_mask %arg2 : vector<4xi1>
  %result = vector.mask %mask { arith.addi %arg0, %arg1 : vector<4xi32> } : vector<4xi1> -> vector<4xi32>
  return %result : vector<4xi32>
}

// CHECK-LABEL: func.func @lower_buffer_boundaries(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>>
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<4xf32, #asctile.local<UB>> to !ascendc.local_tensor<4xf32>
// CHECK: ascendc.local_tensor.get_phy_addr_v2 {{.*}} : !ascendc.local_tensor<4xf32>, memref<4xf32, 26>
// CHECK: ascendc.local_tensor_auto veccalc() : <4xf32>
// CHECK: ascendc.local_tensor.get_phy_addr_v2 {{.*}} : !ascendc.local_tensor<4xf32>, memref<4xf32, 26>
// CHECK-NOT: ascvf2.to_buffer
// CHECK-NOT: ascvf2.to_tensor
// CHECK-NOT: memref.alloca
func.func @lower_buffer_boundaries(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg0 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  %temporary = memref.alloca() : memref<4xf32, 26>
  %result = ascvf2.to_tensor %temporary : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  return %result : tensor<4xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_barrier() attributes {ascvf2.vector_func}
// CHECK: ascendc.local_mem_bar VEC_STORE, VEC_LOAD
// CHECK-NOT: ascvf2.barrier
func.func @lower_barrier() attributes {ascvf2.vector_func} {
  ascvf2.vf_group {
    ascvf2.barrier
    ascvf2.yield
  }
  return
}

// CHECK-LABEL: func.func @lower_contiguous_transfers(%arg0: memref<4xi32, 26>, %arg1: memref<4xi32, 26>, %arg2: memref<4xi32, 26>, %arg3: index) attributes {ascvf2.vector_func}
// CHECK: ascvf2.simple_mask_reg i32 dynamic %arg3 : !ascendc.mask_reg
// CHECK: emitasc.ptr_offset
// CHECK: ascendc.data_copy_vld_reg {{.*}} : !ascendc.reg_tensor<i32>, memref<4xi32, 26>
// CHECK: emitasc.ptr_offset
// CHECK: ascendc.data_copy_vld_reg {{.*}} : !ascendc.reg_tensor<i32>, memref<4xi32, 26>
// CHECK: ascendc.add_reg {{.*}} : !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.mask_reg
// CHECK: emitasc.ptr_offset
// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 16 : i32} : memref<4xi32, 26>, !ascendc.reg_tensor<i32>, !ascendc.mask_reg
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
func.func @lower_contiguous_transfers(%arg0: memref<4xi32, 26>, %arg1: memref<4xi32, 26>, %arg2: memref<4xi32, 26>, %arg3: index) attributes {ascvf2.vector_func} {
  %pad = arith.constant 0 : i32
  %mask = vector.create_mask %arg3 : vector<4xi1>
  %lhs = vector.transfer_read %arg0[%arg3], %pad, %mask {in_bounds = [true]} : memref<4xi32, 26>, vector<4xi32>
  %rhs = vector.transfer_read %arg1[%arg3], %pad, %mask {in_bounds = [true]} : memref<4xi32, 26>, vector<4xi32>
  %result = vector.mask %mask { arith.addi %lhs, %rhs : vector<4xi32> } : vector<4xi1> -> vector<4xi32>
  vector.transfer_write %result, %arg2[%arg3], %mask {in_bounds = [true]} : vector<4xi32>, memref<4xi32, 26>
  return
}

// CHECK-LABEL: func.func @lower_broadcast_transfer_read(%arg0: memref<1xf32, 26>, %arg1: index) -> vector<32xf32>
// CHECK: emitasc.ptr_offset
// CHECK: ascendc.data_copy_vld_reg {{.*}} {dist = 3 : i32} : !ascendc.reg_tensor<f32>, memref<1xf32, 26>
// CHECK-NOT: vector.transfer_read
func.func @lower_broadcast_transfer_read(%arg0: memref<1xf32, 26>, %arg1: index) -> vector<32xf32> attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f32
  %result = vector.transfer_read %arg0[%arg1], %pad {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : memref<1xf32, 26>, vector<32xf32>
  return %result : vector<32xf32>
}

// CHECK-LABEL: func.func @lower_vector_broadcast(%arg0: vector<1xf32>, %arg1: memref<32xf32, 26>, %arg2: index) attributes {ascvf2.vector_func}
// CHECK: ascvf2.simple_mask_reg f32 static 32 : !ascendc.mask_reg
// CHECK: ascendc.duplicate_reg {{.*}} : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascvf2.simple_mask_reg f32 static 32 : !ascendc.mask_reg
// CHECK: emitasc.ptr_offset
// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 16 : i32} : memref<32xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NOT: vector.broadcast
func.func @lower_vector_broadcast(%arg0: vector<1xf32>, %arg1: memref<32xf32, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %result = vector.broadcast %arg0 : vector<1xf32> to vector<32xf32>
  vector.transfer_write %result, %arg1[%arg2] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, 26>
  return
}

// CHECK-LABEL: func.func @lower_first_element_store_b8(%arg0: vector<1xi8>, %arg1: memref<1xi8, 26>, %arg2: index) attributes {ascvf2.vector_func}
// CHECK: ascvf2.simple_mask_reg i8 static 1 : !ascendc.mask_reg
// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 3 : i32} : memref<1xi8, 26>, !ascendc.reg_tensor<i8>, !ascendc.mask_reg
func.func @lower_first_element_store_b8(%arg0: vector<1xi8>, %arg1: memref<1xi8, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  vector.transfer_write %arg0, %arg1[%arg2] {in_bounds = [true]} : vector<1xi8>, memref<1xi8, 26>
  return
}

// CHECK-LABEL: func.func @lower_first_element_store_b16(%arg0: vector<1xf16>, %arg1: memref<1xf16, 26>, %arg2: index) attributes {ascvf2.vector_func}
// CHECK: ascvf2.simple_mask_reg f16 static 1 : !ascendc.mask_reg
// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 4 : i32} : memref<1xf16, 26>, !ascendc.reg_tensor<f16>, !ascendc.mask_reg
func.func @lower_first_element_store_b16(%arg0: vector<1xf16>, %arg1: memref<1xf16, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  vector.transfer_write %arg0, %arg1[%arg2] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, 26>
  return
}

// CHECK-LABEL: func.func @lower_first_element_store_b32(%arg0: vector<1xf32>, %arg1: memref<1xf32, 26>, %arg2: index) attributes {ascvf2.vector_func}
// CHECK: ascvf2.simple_mask_reg f32 static 1 : !ascendc.mask_reg
// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 5 : i32} : memref<1xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @lower_first_element_store_b32(%arg0: vector<1xf32>, %arg1: memref<1xf32, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  vector.transfer_write %arg0, %arg1[%arg2] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, 26>
  return
}

// CHECK-LABEL: func.func @lower_reductions(%arg0: vector<64xf32>, %arg1: vector<64xi32>) -> (vector<1xf32>, vector<1xf32>, vector<1xi32>)
// CHECK: ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.reduce_sum_reg
// CHECK: ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.reduce_max_reg
// CHECK: ascvf2.simple_mask_reg i32 static 64 : !ascendc.mask_reg
// CHECK: ascendc.reduce_min_reg
// CHECK-NOT: ascvf2.reduce
func.func @lower_reductions(%arg0: vector<64xf32>, %arg1: vector<64xi32>) -> (vector<1xf32>, vector<1xf32>, vector<1xi32>) attributes {ascvf2.vector_func} {
  %sum = ascvf2.reduce <sum> %arg0 [0 : i32] : vector<64xf32> to vector<1xf32>
  %max = ascvf2.reduce <max> %arg0 [0 : i32] : vector<64xf32> to vector<1xf32>
  %min = ascvf2.reduce <min> %arg1 [0 : i32] : vector<64xi32> to vector<1xi32>
  return %sum, %max, %min : vector<1xf32>, vector<1xf32>, vector<1xi32>
}

// CHECK-LABEL: func.func @convert_for_edge(%arg0: index, %arg1: index, %arg2: index) -> vector<4xf32>
// CHECK: %cst = arith.constant 0.000000e+00 : f32
// CHECK: %0 = ascendc.reg_tensor : <f32>
// CHECK: ascvf2.simple_mask_reg f32 static 4 : !ascendc.mask_reg
// CHECK: ascendc.duplicate_reg %0, %cst
// CHECK: %2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %0) -> (!ascendc.reg_tensor<f32>) {
// CHECK: builtin.unrealized_conversion_cast %arg4 : !ascendc.reg_tensor<f32> to vector<4xf32>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<4xf32> to !ascendc.reg_tensor<f32>
// CHECK: scf.yield {{.*}} : !ascendc.reg_tensor<f32>
// CHECK: builtin.unrealized_conversion_cast %2 : !ascendc.reg_tensor<f32> to vector<4xf32>
// CHECK: return
func.func @convert_for_edge(%arg0: index, %arg1: index, %arg2: index) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %loop = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst) -> (vector<4xf32>) {
    %cast = builtin.unrealized_conversion_cast %arg4 : vector<4xf32> to !ascendc.reg_tensor<f32>
    %result = builtin.unrealized_conversion_cast %cast : !ascendc.reg_tensor<f32> to vector<4xf32>
    scf.yield %result : vector<4xf32>
  }
  return %loop : vector<4xf32>
}
