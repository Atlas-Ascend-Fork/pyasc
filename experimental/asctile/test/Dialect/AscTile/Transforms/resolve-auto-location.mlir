// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-resolve-auto-location %s | FileCheck %s

// CHECK-LABEL: func.func @resolve_load_auto(%arg0: tensor<128xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.load %arg0[%c0_i32] : tensor<128xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_load_auto(%arg0: tensor<128xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<128xf32, #asctile.global>, tensor<32xf32, #asctile.local<auto>>
  %1 = tensor.cast %0 : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_copy_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.copy %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_copy_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %auto_in = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  %auto_out = asctile.copy %auto_in[%c0] : tensor<32xf32, #asctile.local<auto>>, tensor<32xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_store_auto(%arg0: tensor<32xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<UB>>) {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: asctile.store %arg1, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @resolve_store_auto(%arg0: tensor<32xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<UB>>) {
  %c0 = arith.constant 0 : i32
  %auto = tensor.cast %arg1 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  asctile.store %auto, %arg0[%c0] : tensor<32xf32, #asctile.local<auto>>, tensor<32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @resolve_relu_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_relu_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %auto_in = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  %auto_out = asctile.relu %auto_in : tensor<32xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_asctile_cast_auto(%arg0: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.cast <default> %arg0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_asctile_cast_auto(%arg0: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %auto_in = tensor.cast %arg0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xf16, #asctile.local<auto>>
  %auto_out = asctile.cast <default> %auto_in : tensor<32xf16, #asctile.local<auto>> to tensor<32xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_for_auto(%arg0: tensor<32xf32, #asctile.local<auto>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK: %cast = tensor.cast %arg0 : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: %0 = scf.for
// CHECK-SAME: iter_args(%arg2 = %cast) -> (tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT: %1 = asctile.relu %arg2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: scf.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_for_auto(%arg0: tensor<32xf32, #asctile.local<auto>>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %arg0) -> tensor<32xf32, #asctile.local<auto>> {
    %casted = tensor.cast %iter : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
    %relued = asctile.relu %casted : tensor<32xf32, #asctile.local<UB>>
    %back = tensor.cast %relued : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
    scf.yield %back : tensor<32xf32, #asctile.local<auto>>
  }
  %final = tensor.cast %result : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %final : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_if_auto(%arg0: i1, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = scf.if %arg0 -> (tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT: scf.yield %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:} else {
// CHECK-NEXT: scf.yield %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_if_auto(%arg0: i1, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %result = scf.if %arg0 -> tensor<32xf32, #asctile.local<auto>> {
    %then_val = tensor.cast %arg1 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
    scf.yield %then_val : tensor<32xf32, #asctile.local<auto>>
  } else {
    %else_val = tensor.cast %arg1 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
    scf.yield %else_val : tensor<32xf32, #asctile.local<auto>>
  }
  %final = tensor.cast %result : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %final : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @reconcile_cast_chain(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @reconcile_cast_chain(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L1>>
  %1 = tensor.cast %0 : tensor<32xf32, #asctile.local<L1>> to tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @reconcile_cast_chain_diff(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<L1>> {
// CHECK-NEXT: %cast = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L1>>
// CHECK-NEXT: return %cast : tensor<32xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @reconcile_cast_chain_diff(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<L1>> {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L0C>>
  %1 = tensor.cast %0 : tensor<32xf32, #asctile.local<L0C>> to tensor<32xf32, #asctile.local<L1>>
  return %1 : tensor<32xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @resolve_setvalue_auto(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<1xf32, #asctile.local<UB>>) {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: asctile.set_value %arg1, %arg0[%c0_i32] : tensor<1xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @resolve_setvalue_auto(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<1xf32, #asctile.local<UB>>) {
  %c0 = arith.constant 0 : i32
  %auto = tensor.cast %arg1 : tensor<1xf32, #asctile.local<UB>> to tensor<1xf32, #asctile.local<auto>>
  asctile.set_value %auto, %arg0[%c0] : tensor<1xf32, #asctile.local<auto>>, tensor<128xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @resolve_reshape_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32x1xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.reshape %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32x1xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32x1xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_reshape_auto(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32x1xf32, #asctile.local<UB>> {
  %auto_in = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  %auto_out = asctile.reshape %auto_in : tensor<32xf32, #asctile.local<auto>> to tensor<32x1xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32x1xf32, #asctile.local<auto>> to tensor<32x1xf32, #asctile.local<UB>>
  return %result : tensor<32x1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @resolve_transpose_auto_l1(%arg0: tensor<16x32xf32, #asctile.local<L1>>) -> tensor<32x16xf32, #asctile.local<L1>> {
// CHECK-NEXT: %0 = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<L1>> to tensor<32x16xf32, #asctile.local<L1>>
// CHECK-NEXT: return %0 : tensor<32x16xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @resolve_transpose_auto_l1(%arg0: tensor<16x32xf32, #asctile.local<L1>>) -> tensor<32x16xf32, #asctile.local<L1>> {
  %auto_in = tensor.cast %arg0 : tensor<16x32xf32, #asctile.local<L1>> to tensor<16x32xf32, #asctile.local<auto>>
  %auto_out = asctile.transpose %auto_in, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<auto>> to tensor<32x16xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32x16xf32, #asctile.local<auto>> to tensor<32x16xf32, #asctile.local<L1>>
  return %result : tensor<32x16xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @resolve_relu_l0c(%arg0: tensor<32xf32, #asctile.local<L0C>>) -> tensor<32xf32, #asctile.local<L0C>> {
// CHECK-NEXT: %0 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<L0C>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @resolve_relu_l0c(%arg0: tensor<32xf32, #asctile.local<L0C>>) -> tensor<32xf32, #asctile.local<L0C>> {
  %auto_in = tensor.cast %arg0 : tensor<32xf32, #asctile.local<L0C>> to tensor<32xf32, #asctile.local<auto>>
  %auto_out = asctile.relu %auto_in : tensor<32xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<L0C>>
  return %result : tensor<32xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @resolve_transpose_concrete_operand(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<32x16xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @resolve_transpose_concrete_operand(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<32x16xf32, #asctile.local<UB>> {
  %auto_out = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<auto>>
  %result = tensor.cast %auto_out : tensor<32x16xf32, #asctile.local<auto>> to tensor<32x16xf32, #asctile.local<UB>>
  return %result : tensor<32x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @second_stage_relu(%arg0: tensor<32xf32, #asctile.local<auto>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %cast = tensor.cast %arg0 : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: %0 = asctile.relu %cast : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @second_stage_relu(%arg0: tensor<32xf32, #asctile.local<auto>>) -> tensor<32xf32, #asctile.local<UB>> {
  %t = asctile.relu %arg0 : tensor<32xf32, #asctile.local<auto>>
  %r = tensor.cast %t : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
  return %r : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @second_stage_store(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<auto>>, %arg2: i32) {
// CHECK-NEXT: %cast = tensor.cast %arg1 : tensor<32xf32, #asctile.local<auto>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: asctile.store %cast, %arg0[%arg2] : tensor<32xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @second_stage_store(%arg0: tensor<128xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.local<auto>>, %arg2: i32) {
  asctile.store %arg1, %arg0[%arg2] : tensor<32xf32, #asctile.local<auto>>, tensor<128xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @resolve_dump_tensor(%arg0: tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:  asctile.dump_tensor %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @resolve_dump_tensor(%arg0: tensor<32xf32, #asctile.local<UB>>) {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  asctile.dump_tensor %0 : tensor<32xf32, #asctile.local<auto>>
  return
}

// CHECK-LABEL: func.func @resolve_inline(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: i32, %arg2: tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:  asctile.inline "123 $0;" %arg0, %arg1, %arg2 : tensor<32xf32, #asctile.local<UB>>, i32, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @resolve_inline(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: i32, %arg2: tensor<32xf32, #asctile.local<UB>>) {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  asctile.inline "123 $0;" %0, %arg1, %arg2 : tensor<32xf32, #asctile.local<auto>>, i32, tensor<32xf32, #asctile.local<UB>>
  return
}

// CHECK-LABEL: func.func @resolve_unrealized_conversion_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> !ascendc.local_tensor<32xf32> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  return %0 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT:}
func.func @resolve_unrealized_conversion_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> !ascendc.local_tensor<32xf32> {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<auto>>
  %1 = builtin.unrealized_conversion_cast %0 : tensor<32xf32, #asctile.local<auto>> to !ascendc.local_tensor<32xf32>
  return %1 : !ascendc.local_tensor<32xf32>
}
