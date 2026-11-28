// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-unroll-loop -cse %s | FileCheck %s --check-prefixes=CHECK,NOANN
// RUN: asctile-opt -asctile-unroll-loop=annotate -cse %s | FileCheck %s --check-prefixes=CHECK,ANNOT

// CHECK-LABEL: func.func @unroll_static_loop(%arg0: tensor<32xi32, #asctile.global>) {
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  scf.execute_region {
// CHECK-NEXT:    %c32 = arith.constant 32 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c2 {
// ANNOT-NEXT:      %1 = arith.index_cast %arg1 {asctile.unroll_iter = 0 : i64} : index to i32
// NOANN-NEXT:      %1 = arith.index_cast %arg1 : index to i32
// ANNOT-NEXT:      asctile.set_value %1, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// NOANN-NEXT:      asctile.set_value %1, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:      %2 = arith.muli %c1, %c1 : index
// CHECK-NEXT:      %3 = arith.addi %arg1, %2 : index
// ANNOT-NEXT:      %4 = arith.index_cast %3 {asctile.unroll_iter = 1 : i64} : index to i32
// NOANN-NEXT:      %4 = arith.index_cast %3 : index to i32
// ANNOT-NEXT:      asctile.set_value %4, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// NOANN-NEXT:      asctile.set_value %4, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:    }
// ANNOT-NEXT:    %0 = arith.index_cast %c32 {asctile.unroll_iter = 2 : i64} : index to i32
// NOANN-NEXT:    %0 = arith.index_cast %c32 : index to i32
// ANNOT-NEXT:    asctile.set_value %0, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// NOANN-NEXT:    asctile.set_value %0, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:    scf.yield
// ANNOT-NEXT:  } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 0 : i64}
// NOANN-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_static_loop(%arg0: tensor<32xi32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  scf.for %arg1 = %c0 to %c33 step %c1 {
    %3 = arith.index_cast %arg1 : index to i32
    asctile.set_value %3, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
    scf.yield
  } {asctile.unroll_factor = 2}
  return
}

// ANNOT-LABEL: func.func @unroll_static_loop_with_result(%arg0: tensor<32xi32, #asctile.global>) -> i32 {
// ANNOT-NEXT:  %c0_i32 = arith.constant 0 : i32
// ANNOT-NEXT:  %c1_i32 = arith.constant 1 : i32
// ANNOT-NEXT:  %c32_i32 = arith.constant 32 : i32
// ANNOT-NEXT:  %0 = scf.execute_region -> i32 {
// ANNOT-NEXT:    %c2_i32 = arith.constant 2 : i32
// ANNOT-NEXT:    %1 = scf.for %arg1 = %c0_i32 to %c32_i32 step %c2_i32 iter_args(%arg2 = %c0_i32) -> (i32)  : i32 {
// ANNOT-NEXT:      asctile.set_value %arg1, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      %2 = arith.muli %c1_i32, %c1_i32 : i32
// ANNOT-NEXT:      %3 = arith.addi %arg1, %2 : i32
// ANNOT-NEXT:      asctile.set_value %3, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      scf.yield %3 : i32
// ANNOT-NEXT:    }
// ANNOT-NEXT:    scf.yield %1 : i32
// ANNOT-NEXT:  } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 0 : i64}
// ANNOT-NEXT:  return %0 : i32
// ANNOT-NEXT:}
func.func @unroll_static_loop_with_result(%arg0: tensor<32xi32, #asctile.global>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg2 = %c0_i32) -> (i32) : i32 {
    asctile.set_value %arg1, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
    scf.yield %arg1 : i32
  } {asctile.unroll_factor = 2}
  return %0 : i32
}

// ANNOT-LABEL: func.func @unroll_dynamic_loop(%arg0: tensor<32xi32, #asctile.global>, %arg1: i32) {
// ANNOT-NEXT:  %c0_i32 = arith.constant 0 : i32
// ANNOT-NEXT:  %c1_i32 = arith.constant 1 : i32
// ANNOT-NEXT:  scf.execute_region {
// ANNOT-NEXT:    %0 = arith.subi %arg1, %c0_i32 : i32
// ANNOT-NEXT:    %1 = arith.subi %c1_i32, %c1_i32 : i32
// ANNOT-NEXT:    %2 = arith.addi %0, %1 : i32
// ANNOT-NEXT:    %3 = arith.divui %2, %c1_i32 : i32
// ANNOT-NEXT:    %c3_i32 = arith.constant 3 : i32
// ANNOT-NEXT:    %4 = arith.remsi %3, %c3_i32 : i32
// ANNOT-NEXT:    %5 = arith.subi %3, %4 : i32
// ANNOT-NEXT:    %6 = arith.muli %5, %c1_i32 : i32
// ANNOT-NEXT:    %7 = arith.addi %c0_i32, %6 : i32
// ANNOT-NEXT:    %8 = arith.muli %c1_i32, %c3_i32 : i32
// ANNOT-NEXT:    scf.for %arg2 = %c0_i32 to %7 step %8  : i32 {
// ANNOT-NEXT:      asctile.set_value %arg2, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      %9 = arith.muli %c1_i32, %c1_i32 : i32
// ANNOT-NEXT:      %10 = arith.addi %arg2, %9 : i32
// ANNOT-NEXT:      asctile.set_value %10, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      %c2_i32 = arith.constant 2 : i32
// ANNOT-NEXT:      %11 = arith.muli %c1_i32, %c2_i32 : i32
// ANNOT-NEXT:      %12 = arith.addi %arg2, %11 : i32
// ANNOT-NEXT:      asctile.set_value %12, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    }
// ANNOT-NEXT:    scf.for %arg2 = %7 to %arg1 step %c1_i32  : i32 {
// ANNOT-NEXT:      asctile.set_value %arg2, %arg0[%c0_i32] {asctile.unroll_iter = 3 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    }
// ANNOT-NEXT:    scf.yield
// ANNOT-NEXT:  } {asctile.unroll_factor = 3 : i64, asctile.unrolled_loop = 0 : i64}
// ANNOT-NEXT:  return
// ANNOT-NEXT:}
func.func @unroll_dynamic_loop(%arg0: tensor<32xi32, #asctile.global>, %arg1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32 : i32 {
    asctile.set_value %arg2, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
    scf.yield
  } {asctile.unroll_factor = 3}
  return
}

// ANNOT-LABEL: func.func @unroll_nested_loop(%arg0: tensor<32xi32, #asctile.global>) {
// ANNOT-NEXT:  %c0_i32 = arith.constant 0 : i32
// ANNOT-NEXT:  %c0 = arith.constant 0 : index
// ANNOT-NEXT:  %c4 = arith.constant 4 : index
// ANNOT-NEXT:  scf.execute_region {
// ANNOT-NEXT:    %c2 = arith.constant 2 : index
// ANNOT-NEXT:    scf.for %arg1 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:      scf.execute_region {
// ANNOT-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        }
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        scf.yield
// ANNOT-NEXT:      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 0 : i64}
// ANNOT-NEXT:      scf.execute_region {
// ANNOT-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        }
// ANNOT-NEXT:        scf.yield
// ANNOT-NEXT:      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 1 : i64}
// ANNOT-NEXT:      scf.execute_region {
// ANNOT-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        }
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        scf.yield
// ANNOT-NEXT:      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 2 : i64}
// ANNOT-NEXT:      scf.execute_region {
// ANNOT-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        }
// ANNOT-NEXT:        scf.yield
// ANNOT-NEXT:      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 3 : i64}
// ANNOT-NEXT:    }
// ANNOT-NEXT:    scf.execute_region {
// ANNOT-NEXT:      scf.for %arg1 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      }
// ANNOT-NEXT:      asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      scf.yield
// ANNOT-NEXT:    } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 4 : i64}
// ANNOT-NEXT:    scf.execute_region {
// ANNOT-NEXT:      scf.for %arg1 = %c0 to %c4 step %c2 {
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:      }
// ANNOT-NEXT:      scf.yield
// ANNOT-NEXT:    } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 5 : i64}
// ANNOT-NEXT:    scf.yield
// ANNOT-NEXT:  } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 6 : i64}
// ANNOT-NEXT:  return
// ANNOT-NEXT:}
func.func @unroll_nested_loop(%arg0: tensor<32xi32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %c0_i1 = arith.constant 0 : i1
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  scf.for %arg1 = %c0 to %c5 step %c1 {
    scf.for %arg2 = %c0 to %c5 step %c1 {
      asctile.set_value %c0_i32, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
      scf.yield
    } {asctile.unroll_factor = 2}
    scf.for %arg2 = %c0 to %c4 step %c1 {
      asctile.set_value %c0_i32, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
      scf.yield
    } {asctile.unroll_factor = 2}
  } {asctile.unroll_factor = 2}
  return
}

// CHECK-LABEL: func.func @unroll_loop_yield_global(%arg0: i32) -> (i32, i32) {
// CHECK:       %0:2 = scf.execute_region -> (i32, i32) {
// CHECK-NEXT:      %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:      %1:2 = scf.for %arg1 = %c0_i32 to %c32_i32 step %c2_i32 iter_args(%arg2 = %c0_i32, %arg3 = %c1_i32) -> (i32, i32) : i32 {
// ANNOT-NEXT:        %2 = arith.addi %c1_i32, %arg2 {asctile.unroll_iter = 0 : i64} : i32
// NOANN-NEXT:        %2 = arith.addi %c1_i32, %arg2 : i32
// ANNOT-NEXT:        %3 = arith.addi %c1_i32, %2 {asctile.unroll_iter = 1 : i64} : i32
// NOANN-NEXT:        %3 = arith.addi %c1_i32, %2 : i32
// CHECK-NEXT:        scf.yield %3, %arg0 : i32, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %1#0, %1#1 : i32, i32
// ANNOT-NEXT:    } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 0 : i64}
// NOANN-NEXT:    }
// CHECK-NEXT:    return %0#0, %0#1 : i32, i32
// CHECK-NEXT:  }
func.func @unroll_loop_yield_global(%arg0: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0:2 = scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg2 = %c0_i32, %arg3 = %c1_i32) -> (i32, i32) : i32{
    %1 = arith.addi %c1_i32, %arg2: i32
    scf.yield %1, %arg0 : i32, i32
  } {asctile.unroll_factor = 2}
  return %0#0, %0#1 : i32, i32
}
