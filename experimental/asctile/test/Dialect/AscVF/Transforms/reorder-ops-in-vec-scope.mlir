// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf-reorder-ops-in-vec-scope %s | FileCheck %s

// CHECK-LABEL: func.func @reorder_supported_ops(%arg0: !ascendc.local_tensor<1xf32>) {
// CHECK:     emitasc.vec_scope {
// CHECK-NEXT:  %cst = arith.constant 0xFF800000 : f32
// CHECK-NEXT:  %c64 = arith.constant 64 : index
// CHECK-NEXT:  %c5 = arith.constant 5 : index
// CHECK-NEXT:  %c7_i64 = arith.constant 7 : i64
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %3 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
// CHECK-NEXT:  %4 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
// CHECK-NEXT:  ascendc.duplicate %0, %cst : <f32>, f32
// CHECK-NEXT:  ascendc.duplicate %1, %cst : <f32>, f32
// CHECK-NEXT:  %5 = emitasc.variable %c64 : index, memref<1xui32>
// CHECK-NEXT:  %6 = arith.index_cast %c7_i64 : i64 to index
// CHECK-NEXT:  %7 = arith.remsi %c64, %c5 : index
// CHECK-NEXT:  %8 = arith.muli %7, %c5 : index
// CHECK-NEXT:  %9 = emitasc.variable %8 : index, memref<1xui32>
// CHECK-NEXT:  %10 = arith.ceildivsi %8, %6 : index
// CHECK-NEXT:  %11 = arith.cmpi eq, %6, %7 : index
// CHECK-NEXT:  %12 = arith.divsi %10, %c5 : index
// CHECK-NEXT:  %13 = arith.subi %12, %c5 : index
// CHECK-NEXT:  %14 = arith.addi %13, %c5 : index
// CHECK-NEXT:  ascvf.load %2, %arg0[%c64], %3 : !ascendc.reg_tensor<f32>, <1xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  %15 = ascendc.update_mask f32, %5 : memref<1xui32>
// CHECK-NEXT:  ascendc.select_reg %1, %2, %0, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg0[%14], %1, %4 : <1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:}
func.func @reorder_supported_ops(%arg0: !ascendc.local_tensor<1xf32>) {
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %cst = arith.constant 0xFF800000 : f32
    ascendc.duplicate %0, %cst : <f32>, f32
    %c64 = arith.constant 64 : index
    %1 = emitasc.variable %c64 : index, memref<1xui32>
    %2 = ascendc.reg_tensor : <f32>
    %3 = ascendc.reg_tensor : <f32>
    %c5 = arith.constant 5 : index
    %c7_i64 = arith.constant 7 : i64
    %4 = arith.index_cast %c7_i64 : i64 to index
    %5 = arith.remsi %c64, %c5 : index
    %6 = arith.muli %5, %c5 : index
    %7 = emitasc.variable %6 : index, memref<1xui32>
    %8 = arith.ceildivsi %6, %4 : index
    ascendc.duplicate %2, %cst : <f32>, f32
    %9 = arith.cmpi eq, %4, %5 : index
    %mask = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
    ascvf.load %3, %arg0[%c64], %mask : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index, !ascendc.mask_reg
    %10 = arith.divsi %8, %c5 : index
    %11 = ascendc.update_mask f32, %1 : memref<1xui32>
    %12 = arith.subi %10, %c5 : index
    ascendc.select_reg %2, %3, %0, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    %13 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
    %14 = arith.addi %12, %c5 : index
    ascvf.store %arg0[%14], %2, %13 : !ascendc.local_tensor<1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @reorder_inside_block(%arg0: !ascendc.local_tensor<1xf32>) {
// CHECK:     emitasc.vec_scope {
// CHECK-NEXT: %c64 = arith.constant 64 : index
// CHECK-NEXT: %cst = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: ascendc.duplicate %0, %cst : <f32>, f32
// CHECK-NEXT: %2 = emitasc.variable %c64 : index, memref<1xui32>
// CHECK-NEXT: emitasc.vf_for %c64 : index {
// CHECK-NEXT: ^bb0(%arg1: index):
// CHECK-NEXT:   %3 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
// CHECK-NEXT:   %4 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:   %5 = arith.divsi %4, %c64 : index
// CHECK-NEXT:   %6 = arith.subi %5, %c64 : index
// CHECK-NEXT:   %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:   ascvf.load %1, %arg0[%6], %3 : !ascendc.reg_tensor<f32>, <1xf32>, index, !ascendc.mask_reg
// CHECK-NEXT: }
// CHECK-NEXT:}
func.func @reorder_inside_block(%arg0: !ascendc.local_tensor<1xf32>) {
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %c64 = arith.constant 64 : index
    %1 = emitasc.variable %c64 : index, memref<1xui32>
    %cst = arith.constant 0xFF800000 : f32
    ascendc.duplicate %0, %cst : <f32>, f32
    %2 = ascendc.reg_tensor : <f32>
    emitasc.vf_for %c64 : index {
    ^bb0(%arg1: index):
      %3 = ascendc.update_mask f32, %1 : memref<1xui32>
      %4 = arith.muli %arg1, %c64 : index
      %5 = arith.divsi %4, %c64 : index
      %6 = arith.subi %5, %c64 : index
      %mask = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
      ascvf.load %2, %arg0[%6], %mask : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index, !ascendc.mask_reg
    }
  }
  return
}
