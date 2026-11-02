// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascendc-reuse-tensor-allocation %s | FileCheck %s

// CHECK-LABEL: func.func @works_without_crash() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @works_without_crash() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_different_tensor_types_and_shapes(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>, %arg3: !ascendc.global_tensor<?xf32>, %arg4: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  scf.for %arg5 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <1000xi32>
// CHECK-NEXT:    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg1, %c8_i64 : !ascendc.local_tensor<777xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<8xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg2, %c8_i64 : !ascendc.local_tensor<8xf16>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg2, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf16>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg4, %c8_i64 : !ascendc.local_tensor<1000xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg4, %0, %c8_i64 : !ascendc.global_tensor<?xi32>, !ascendc.local_tensor<1000xi32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_different_tensor_types_and_shapes(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>, %arg3: !ascendc.global_tensor<?xf32>, %arg4: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  scf.for %arg5 = %c0 to %c32 step %c1 {
    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
    ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<777xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<777xf32>, i64
    %2 = ascendc.local_tensor_auto veccalc() : <8xf16>
    ascendc.data_copy_l2 %2, %arg2, %c8_i64 : !ascendc.local_tensor<8xf16>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf16>, i64
    %3 = ascendc.local_tensor_auto veccalc() : <1000xi32>
    ascendc.data_copy_l2 %3, %arg4, %c8_i64 : !ascendc.local_tensor<1000xi32>, !ascendc.global_tensor<?xi32>, i64
    ascendc.data_copy_l2 %arg4, %3, %c8_i64 : !ascendc.global_tensor<?xi32>, !ascendc.local_tensor<1000xi32>, i64
  }
  return
}

// CHECK-LABEL: func.func @reuse_tensor_with_different_attributes_no_loop(%arg0: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() output : <8xi32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_tensor_with_different_attributes_no_loop(%arg0: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %1 = ascendc.local_tensor_auto veccalc() input : <8xi32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %2 = ascendc.local_tensor_auto veccalc() output : <8xi32>
  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_input_output_in_same_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %1, %0, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_input_output_in_same_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() output : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @reuse_input_output_in_different_loops(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %0, %2, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %3, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_input_output_in_different_loops(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() output : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %2, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @noreuse_tensor_because_op_does_not_create_new_tensor(%arg0: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xindex>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_tensor_because_op_does_not_create_new_tensor(%arg0: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xi32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xf32>
  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %5 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xindex>
  ascendc.data_copy_l2 %4, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_two_temporary_tensors(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %1, %4, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_two_temporary_tensors(%arg0: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.add_l3 %4, %0, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.data_copy_l2 %arg0, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_tensor_used_in_other_block(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_tensor_used_in_other_block(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg2, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_loop(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @hoist_and_reuse_inside_loop(%arg0 : !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.for %arg1 = %c0 to %c32 step %c1 {
    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
    scf.for %arg2 = %c0 to %c32 step %c1 {
      %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
      ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.for %arg2 = %c0 to %c32 step %c1 {
      %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
      ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
    %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.for %arg2 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_condition(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %4, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %6, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @hoist_and_reuse_inside_condition(%arg0 : !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.if %true {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_while_loop(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %4, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %6, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    scf.condition(%true)
// CHECK-NEXT:  } do {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @hoist_and_reuse_inside_while_loop(%arg0 : !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.while : () -> () {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %true = arith.constant true
    scf.condition(%true)
  } do {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_users_in_then_and_else_regions(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_users_in_then_and_else_regions(%arg0: !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.if %true {
    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_deeply_nested_blocks(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_deeply_nested_blocks(%arg0: !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.if %true {
    scf.for %arg1 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  } else {
    scf.for %arg1 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_users_in_while_before_and_after(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    scf.condition(%true)
// CHECK-NEXT:  } do {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_users_in_while_before_and_after(%arg0: !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.while : () -> () {
    ascendc.add_l3 %1, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %true = arith.constant true
    scf.condition(%true)
  } do {
    ascendc.add_l3 %2, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_multiple_tensors_nested_blocks(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %5, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_multiple_tensors_nested_blocks(%arg0: !ascendc.global_tensor<?xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %4, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.for %arg1 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %1, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.if %true {
    ascendc.add_l3 %2, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %3, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %5, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_tensors_from_the_same_memory(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto a1() : <16x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_tensors_from_the_same_memory(%arg0: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto a1() : <16x16xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  %1 = ascendc.local_tensor_auto a1() : <16x16xf32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_different_positions(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto b2() : <16x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_different_positions(%arg0: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto a2() : <16x16xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  %1 = ascendc.local_tensor_auto b2() : <16x16xf32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_matmul_pattern_no_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() output : <64x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <16x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a2() input : <64x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<64x16xf32> to !ascendc.local_tensor<64x16xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %4, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %0, %3, %4 : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %0, %2, %1 : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<64x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_matmul_pattern_no_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %co1 = ascendc.local_tensor_auto co1() output : <64x16xf32>
  %a2_0 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <64x16xf32>
  %a2_1 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <64x16xf32>
  %b2_0 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <16x16xf32>
  %b2_1 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <16x16xf32>
  ascendc.data_copy_l2 %a2_0, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %b2_0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %co1, %a2_0, %b2_0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
  ascendc.data_copy_l2 %a2_1, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %b2_1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %co1, %a2_1, %b2_1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
  ascendc.data_copy_l2 %arg1, %co1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<64x16xf32>, i64
  return
}

// CHECK-LABEL: func.func @matmul_v3_kernel(%arg0: !ascendc.mmad_params, %arg1: !ascendc.load_data_2d_params_v2, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.fixpipe_params_v220, %arg4: !ascendc.fixpipe_config) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() output : <256x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <64x256xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto b2() input : <64x256xf16>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto b1() input output : <256x256xf16>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto b1() input output : <256x256xf16>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a2() input : <256x64xf16>
// CHECK-NEXT:  %6 = ascendc.local_tensor_auto a2() input : <256x64xf16>
// CHECK-NEXT:  %7 = ascendc.local_tensor_auto a1() input output : <256x256xf16>
// CHECK-NEXT:  %8 = ascendc.local_tensor_auto a1() input output : <256x256xf16>
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %9 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
// CHECK-NEXT:  %10 = ascendc.global_tensor : !ascendc.global_tensor<7680x4608xf16>
// CHECK-NEXT:  %11 = ascendc.global_tensor : !ascendc.global_tensor<1152x4608xf16>
// CHECK-NEXT:  %12 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<64x256xf16> to !ascendc.local_tensor<64x256xf16>
// CHECK-NEXT:  %13 = ascendc.reinterpret_cast %6 : !ascendc.local_tensor<256x64xf16> to !ascendc.local_tensor<256x64xf16>
// CHECK-NEXT:  %14 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<64x256xf16> to !ascendc.local_tensor<64x256xf16>
// CHECK-NEXT:  %15 = ascendc.reinterpret_cast %5 : !ascendc.local_tensor<256x64xf16> to !ascendc.local_tensor<256x64xf16>
// CHECK-NEXT:  scf.for %arg5 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    scf.for %arg6 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      ascendc.data_copy_l2 %7, %9, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      ascendc.data_copy_l2 %3, %10, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:        ascendc.load_data_l0_v2 %5, %7, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %1, %3, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %5, %1, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:        ascendc.load_data_l0_v2 %6, %7, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %2, %3, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %6, %2, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:      ascendc.data_copy_l2 %8, %9, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      ascendc.data_copy_l2 %4, %10, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:        ascendc.load_data_l0_v2 %15, %8, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %14, %4, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %15, %14, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:        ascendc.load_data_l0_v2 %13, %8, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %12, %4, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %13, %12, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.fixpipe %11, %0, %arg3, %arg4 : !ascendc.global_tensor<1152x4608xf16>, !ascendc.local_tensor<256x256xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @matmul_v3_kernel(%arg0: !ascendc.mmad_params, %arg1: !ascendc.load_data_2d_params_v2, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.fixpipe_params_v220, %arg4: !ascendc.fixpipe_config) {
  %c1_i32 = arith.constant 1 : i32
  %c256_i32 = arith.constant 256 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
  %1 = ascendc.global_tensor : !ascendc.global_tensor<7680x4608xf16>
  %2 = ascendc.global_tensor : !ascendc.global_tensor<1152x4608xf16>
  %3 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <64x256xf16>
  %4 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <256x64xf16>
  %5 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <64x256xf16>
  %6 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <256x64xf16>
  %7 = ascendc.local_tensor_auto b1() input output {asc.reuse_group = 1 : i64} : <256x256xf16>
  %8 = ascendc.local_tensor_auto a1() input output {asc.reuse_group = 1 : i64} : <256x256xf16>
  %9 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <64x256xf16>
  %10 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <256x64xf16>
  %11 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <64x256xf16>
  %12 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <256x64xf16>
  %13 = ascendc.local_tensor_auto b1() input output {asc.reuse_group = 0 : i64} : <256x256xf16>
  %14 = ascendc.local_tensor_auto a1() input output {asc.reuse_group = 0 : i64} : <256x256xf16>
  %15 = ascendc.local_tensor_auto co1() output : <256x256xf32>
  scf.for %arg5 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
    scf.for %arg6 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
      ascendc.data_copy_l2 %14, %0, %arg2 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
      ascendc.data_copy_l2 %13, %1, %arg2 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
        ascendc.load_data_l0_v2 %12, %14, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %11, %13, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %12, %11, %arg0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
        ascendc.load_data_l0_v2 %10, %14, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %9, %13, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %10, %9, %arg0 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
      }
      ascendc.data_copy_l2 %8, %0, %arg2 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
      ascendc.data_copy_l2 %7, %1, %arg2 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
        ascendc.load_data_l0_v2 %6, %8, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %5, %7, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %6, %5, %arg0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
        ascendc.load_data_l0_v2 %4, %8, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %3, %7, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %4, %3, %arg0 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
      }
    }
    ascendc.fixpipe %2, %15, %arg3, %arg4 : !ascendc.global_tensor<1152x4608xf16>, !ascendc.local_tensor<256x256xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
  }
  return
}

// CHECK-LABEL: func.func @reuse_cv_block(%arg0: !ascendc.data_copy_ext_params, %arg1: !ascendc.data_copy_pad_ext_params<f32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() input : <32x64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32x64xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<32x64xf32> to !ascendc.local_tensor<32x64xf32>
// CHECK-NEXT:  %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:  %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %3 = ascendc.global_tensor : !ascendc.global_tensor<128x64xf32>
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %2, %3, %arg0, %arg1 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<32x64xf32>, !ascendc.global_tensor<128x64xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  %4 = scf.for %arg2 = %c1_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %2) -> (!ascendc.local_tensor<32x64xf32>)  : i32 {
// CHECK-NEXT:    %5 = ascendc.if_aiv(%arg3 : !ascendc.local_tensor<32x64xf32>) -> !ascendc.local_tensor<32x64xf32> {
// CHECK-NEXT:      ascendc.data_copy_pad_l0_ext %0, %3, %arg0, %arg1 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<32x64xf32>, !ascendc.global_tensor<128x64xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:      ascendc.add_l2 %1, %arg3, %0, %c2048_i64 : !ascendc.local_tensor<32x64xf32>, !ascendc.local_tensor<32x64xf32>, !ascendc.local_tensor<32x64xf32>, i64
// CHECK-NEXT:      ascendc.yield %1 : !ascendc.local_tensor<32x64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %5 : !ascendc.local_tensor<32x64xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_cv_block(%arg0: !ascendc.data_copy_ext_params, %arg1: !ascendc.data_copy_pad_ext_params<f32>) {
  %0 = ascendc.local_tensor_auto veccalc() input : <32x64xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32x64xf32>
  %2 = ascendc.local_tensor_auto veccalc() input : <32x64xf32>
  %c2048_i64 = arith.constant 2048 : i64
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %3 = ascendc.global_tensor : !ascendc.global_tensor<128x64xf32>
  ascendc.data_copy_pad_l0_ext %0, %3, %arg0, %arg1 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<32x64xf32>, !ascendc.global_tensor<128x64xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
  %4 = scf.for %arg2 = %c1_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ascendc.local_tensor<32x64xf32>)  : i32 {
    %5 = ascendc.if_aiv(%arg3 : !ascendc.local_tensor<32x64xf32>) -> !ascendc.local_tensor<32x64xf32> {
      ascendc.data_copy_pad_l0_ext %2, %3, %arg0, %arg1 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<32x64xf32>, !ascendc.global_tensor<128x64xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
      ascendc.add_l2 %1, %arg3, %2, %c2048_i64 : !ascendc.local_tensor<32x64xf32>, !ascendc.local_tensor<32x64xf32>, !ascendc.local_tensor<32x64xf32>, i64
      ascendc.yield %1 : !ascendc.local_tensor<32x64xf32>
    }
    scf.yield %5 : !ascendc.local_tensor<32x64xf32>
  }
  return
}

// CHECK-LABEL: func.func @cross_group_in_cv_blocks_with_reg_after(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.mmad_params, %arg2: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto co1() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <8xf32>
// CHECK-NEXT:  %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %3, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aic {
// CHECK-NEXT:    ascendc.mmad %1, %2, %2, %arg1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg2, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @cross_group_in_cv_blocks_with_reg_after(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.mmad_params, %arg2: !ascendc.global_tensor<?xf32>) {
  %c256_i32 = arith.constant 256 : i32
  %c8_i64 = arith.constant 8 : i64
  %ub_cross = ascendc.local_tensor_auto veccalc() : <8xf32>
  %a1_dst = ascendc.local_tensor_auto a1() : <8xf32>
  %co1_out = ascendc.local_tensor_auto co1() : <8xf32>
  ascendc.if_aiv {
    ascendc.data_copy_l2 %ub_cross, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %a1_dst, %ub_cross, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, i32
    ascendc.yield
  }
  ascendc.if_aic {
    ascendc.mmad %co1_out, %a1_dst, %a1_dst, %arg1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.mmad_params
    ascendc.yield
  }
  %ub_reg = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %ub_reg, %arg2, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg2, %ub_reg, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_cross_group_ub_to_b1(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b1() : <8xf32>
// CHECK-NEXT:  %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %2, %c256_i32 {direction = #ascendc.copy_direction<veccalc, b1>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, i32
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_cross_group_ub_to_b1(%arg0: !ascendc.global_tensor<?xf32>) {
  %c256_i32 = arith.constant 256 : i32
  %c8_i64 = arith.constant 8 : i64
  %ub_cross = ascendc.local_tensor_auto veccalc() : <8xf32>
  %b1_dst = ascendc.local_tensor_auto b1() : <8xf32>
  ascendc.data_copy_l2 %ub_cross, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %b1_dst, %ub_cross, %c256_i32 {direction = #ascendc.copy_direction<veccalc, b1>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, i32
  %ub_reg = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %ub_reg, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_same_op_cross_group(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.fixpipe_params_v220, %arg2: !ascendc.fixpipe_config) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto co1() : <8xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.if_aic {
// CHECK-NEXT:    ascendc.fixpipe %2, %1, %arg1, %arg2 {direction = #ascendc.copy_direction<co1, veccalc>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.add_l3 %0, %2, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_same_op_cross_group(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.fixpipe_params_v220, %arg2: !ascendc.fixpipe_config) {
  %c8_i64 = arith.constant 8 : i64
  %co1_src = ascendc.local_tensor_auto co1() : <8xf32>
  %ub_cross = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.if_aic {
    ascendc.fixpipe %ub_cross, %co1_src, %arg1, %arg2 {direction = #ascendc.copy_direction<co1, veccalc>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    ascendc.yield
  }
  %ub_reg = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.if_aiv {
    ascendc.add_l3 %ub_reg, %ub_cross, %ub_cross : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @reuse_two_temps_inside_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:         %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c4 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %2, %1, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.mul_l3 %0, %2, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @reuse_two_temps_inside_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %a = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <8xf32>
  %c = ascendc.local_tensor_auto veccalc() output : <8xf32>
  ascendc.data_copy_l2 %a, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.for %arg2 = %c0 to %c4 step %c1 {
    ascendc.add_l3 %b, %a, %a : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.mul_l3 %c, %b, %b : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %c, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @noreuse_a_and_yield_set_before_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:         ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    %4 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %1) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:      ascendc.add_l3 %0, %3, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.mul_l3 %2, %0, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:      scf.yield %2 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @noreuse_a_and_yield_set_before_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %a = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <8xf32>
  %c = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %a, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.add_l3 %b, %a, %a : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.mul_l3 %c, %b, %b : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    scf.yield %c : !ascendc.local_tensor<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @noreuse_a_and_yielded_in_same_op_with_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:         %3 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    %4 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %3) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:      ascendc.add_l3 %0, %2, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.mul_l3 %1, %0, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:      scf.yield %1 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @noreuse_a_and_yielded_in_same_op_with_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %a = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <8xf32>
  %c = ascendc.local_tensor_auto veccalc() output : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %a, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.add_l3 %b, %a, %a : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.mul_l3 %c, %b, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %c, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    scf.yield %c : !ascendc.local_tensor<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @noreuse_a_and_yield_def_after_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() input output : <8xf32>
// CHECK:         ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    %4 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %3) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:      ascendc.add_l3 %0, %2, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.mul_l3 %1, %0, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      scf.yield %1 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @noreuse_a_and_yield_def_after_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %a = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <8xf32>
  %c = ascendc.local_tensor_auto veccalc() : <8xf32>
  %init = ascendc.local_tensor_auto veccalc() input output : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %a, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.add_l3 %b, %a, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.mul_l3 %c, %b, %a : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield %c : !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %init, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_init_and_b_temp(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:       %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %4 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %2) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:    ascendc.add_l3 %0, %1, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.mul_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield %3 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_init_and_b_temp(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %a = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <8xf32>
  %c = ascendc.local_tensor_auto veccalc() : <8xf32>
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %a, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.add_l3 %b, %a, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.mul_l3 %c, %b, %a : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield %c : !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %r, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_init_value_and_yielded_if_set_before_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:       ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    scf.yield %2 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_init_value_and_yielded_if_set_before_last_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    %b = ascendc.local_tensor_auto veccalc() input : <8xf32>
    ascendc.data_copy_l2 %b, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    scf.yield %b : !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %r, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_yield_with_init_if_def_in_same_op_with_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:       %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:    ascendc.add_l3 %1, %arg3, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield %1 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_yield_with_init_if_def_in_same_op_with_iter_arg_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %a = ascendc.local_tensor_auto veccalc() : <8xf32>
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.add_l3 %a, %arg3, %arg3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield %a : !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %r, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_init_value_and_yield_set_after_iter_arg_last_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c4 = arith.constant 4 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %1) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    scf.yield %0 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_init_value_and_yield_set_after_iter_arg_last_use(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init) -> (!ascendc.local_tensor<8xf32>) {
    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    %b = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %b, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    scf.yield %b : !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %r, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_init_value_and_yield_for_iter_arg_lower_shape(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK:         %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<16xf32> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    %4:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %3, %arg4 = %0) -> (!ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:      ascendc.data_copy_l2 %1, %arg0, %c16_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:      ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:      scf.yield %2, %1 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %4#0, %c16_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %4#1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @noreuse_init_value_and_yield_for_iter_arg_lower_shape(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %init1 = ascendc.local_tensor_auto veccalc() input : <16xf32>
  %init2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %a = ascendc.local_tensor_auto veccalc() : <8xf32>
  %b = ascendc.local_tensor_auto veccalc() : <16xf32>
  ascendc.data_copy_l2 %init1, %arg0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %init2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %r1, %r2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %init1, %arg4 = %init2) -> (!ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>) {
    ascendc.data_copy_l2 %arg1, %arg3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16xf32>, i64
    ascendc.data_copy_l2 %a, %arg0, %c16_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %b, %arg0, %c8_i64 : !ascendc.local_tensor<16xf32>, !ascendc.global_tensor<?xf32>, i64
    scf.yield %b, %a : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg1, %r1, %c16_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16xf32>, i64
  ascendc.data_copy_l2 %arg1, %r2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_nested_init_value_and_yield(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK:         ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c4 step %c1 {
// CHECK-NEXT:      %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (!ascendc.local_tensor<8xf32>) {
// CHECK-NEXT:        %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:        ascendc.data_copy_l2 %arg1, %arg4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:        ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:        scf.yield %2 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @noreuse_nested_init_value_and_yield(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8_i64 = arith.constant 8 : i64
  %init = ascendc.local_tensor_auto veccalc() input : <8xf32>
  ascendc.data_copy_l2 %init, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  scf.for %arg2 = %c0 to %c4 step %c1 {
    %ri = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %init) -> (!ascendc.local_tensor<8xf32>) {
      ascendc.data_copy_l2 %arg1, %arg4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
      %b = ascendc.local_tensor_auto veccalc() : <8xf32>
      ascendc.data_copy_l2 %b, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
      scf.yield %b : !ascendc.local_tensor<8xf32>
    }
    ascendc.data_copy_l2 %arg1, %ri, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}
