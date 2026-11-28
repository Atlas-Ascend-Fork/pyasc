// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-reuse-reg-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @reuse_between_loops(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.abs_reg %1, %2, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_between_loops(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  %third = ascendc.reg_tensor : <f32>
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %first, %second, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %third, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @reuse_sequential_registers_in_loop(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_sequential_registers_in_loop(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  %third = ascendc.reg_tensor : <f32>
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %first, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %second, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %third, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @keep_overlapping_registers(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.abs_reg %1, %2, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %2, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_overlapping_registers(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %first, %second, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %second, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @keep_different_types(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<i32>, %arg2: index, %arg3: index, %arg4: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.create_mask i32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %3 = ascendc.reg_tensor : <i32>
// CHECK-NEXT:  scf.for %arg5 = %arg2 to %arg3 step %arg4 {
// CHECK-NEXT:    ascendc.abs_reg %2, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %3, %arg1, %1 : !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_different_types(%sourceFloat: !ascendc.reg_tensor<f32>, %sourceInt: !ascendc.reg_tensor<i32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %floatMask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %intMask = ascendc.create_mask i32, ALL : !ascendc.mask_reg
  %float = ascendc.reg_tensor : <f32>
  %int = ascendc.reg_tensor : <i32>
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %float, %sourceFloat, %floatMask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %int, %sourceInt, %intMask : !ascendc.reg_tensor<i32>, !ascendc.reg_tensor<i32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @ignore_non_vector_function(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  scf.for %index = %lower to %upper step %step {
    ascendc.abs_reg %first, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @keep_loop_carried_register(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = scf.for %arg4 = %arg1 to %arg2 step %arg3 iter_args(%arg5 = %1) -> (!ascendc.reg_tensor<f32>) {
// CHECK-NEXT:    %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:    ascendc.abs_reg %3, %arg5, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    scf.yield %3 : !ascendc.reg_tensor<f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_loop_carried_register(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %init = ascendc.reg_tensor : <f32>
  %result = scf.for %index = %lower to %upper step %step iter_args(%accumulator = %init) -> !ascendc.reg_tensor<f32> {
    %next = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %next, %accumulator, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    scf.yield %next : !ascendc.reg_tensor<f32>
  }
  return
}

// CHECK-LABEL: func.func @keep_loop_invariant_register(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  ascendc.duplicate_reg %1, %cst, %0 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.add_reg %2, %arg0, %1, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %3, %2, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_loop_invariant_register(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %cst = arith.constant 0.0 : f32
  %zero = ascendc.reg_tensor : <f32>
  %sum = ascendc.reg_tensor : <f32>
  %result = ascendc.reg_tensor : <f32>
  ascendc.duplicate_reg %zero, %cst, %mask : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  scf.for %index = %lower to %upper step %step {
    ascendc.add_reg %sum, %source, %zero, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %result, %sum, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @keep_inplace_loop_accumulator(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  ascendc.duplicate_reg %1, %cst, %0 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    ascendc.add_reg %1, %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %2, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_inplace_loop_accumulator(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %cst = arith.constant 0.0 : f32
  %acc = ascendc.reg_tensor : <f32>
  %tmp = ascendc.reg_tensor : <f32>
  ascendc.duplicate_reg %acc, %cst, %mask : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  scf.for %index = %lower to %upper step %step {
    ascendc.add_reg %acc, %acc, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %tmp, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @keep_registers_from_different_loop_scopes(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @keep_registers_from_different_loop_scopes(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  scf.for %first = %lower to %upper step %step {
    %firstRegister = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %firstRegister, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  scf.for %second = %lower to %upper step %step {
    %secondRegister = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %secondRegister, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @reuse_registers_defined_in_loop(%arg0: !ascendc.reg_tensor<f32>, %arg1: index, %arg2: index, %arg3: index) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
// CHECK-NEXT:    %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_registers_defined_in_loop(%source: !ascendc.reg_tensor<f32>, %lower: index, %upper: index, %step: index) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  scf.for %index = %lower to %upper step %step {
    %first = ascendc.reg_tensor : <f32>
    %second = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %first, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascendc.abs_reg %second, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @erase_unused_register(%arg0: !ascendc.reg_tensor<f32>) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @erase_unused_register(%source: !ascendc.reg_tensor<f32>) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %used = ascendc.reg_tensor : <f32>
  %unused = ascendc.reg_tensor : <f32>
  ascendc.abs_reg %used, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @ignore_branching_function(%arg0: !ascendc.reg_tensor<f32>, %arg1: i1) attributes {ascvf2.vector_func} {
// CHECK-NEXT:  %0 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  scf.if %arg1 {
// CHECK-NEXT:    ascendc.abs_reg %1, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.abs_reg %2, %arg0, %0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @ignore_branching_function(%source: !ascendc.reg_tensor<f32>, %condition: i1) attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  scf.if %condition {
    ascendc.abs_reg %first, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  } else {
    ascendc.abs_reg %second, %source, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}
