// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-raise-vf-for %s | FileCheck %s

// CHECK-LABEL: func.func @raise_eligible_loops(%arg0: index, %arg1: index) {
// CHECK: ascvf2.vf_group
// CHECK: emitasc.vf_for %arg0 : index
// CHECK: emitasc.vf_for %arg1 : index
// CHECK: arith.addi
// CHECK-NOT: scf.for
func.func @raise_eligible_loops(%upper: index, %innerUpper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ascvf2.vf_group {
    scf.for %index = %c0 to %upper step %c1 {
      scf.for %nested = %c0 to %innerUpper step %c1 {
        arith.addi %index, %nested : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @keep_ineligible_loops(%arg0: index) {
// CHECK: ascvf2.vf_group
// CHECK: scf.for %{{.*}} = %{{.*}} to %arg0 step
// CHECK: scf.for %{{.*}} = %{{.*}} to %arg0 step
// CHECK: scf.for %{{.*}} = %{{.*}} to %arg0 step {{.*}} iter_args
// CHECK: scf.yield
// CHECK-NOT: emitasc.vf_for
func.func @keep_ineligible_loops(%upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ascvf2.vf_group {
    scf.for %stepped = %c1 to %upper step %c2 {
      arith.addi %stepped, %c1 : index
    }
    scf.for %nonzero = %c2 to %upper step %c1 {
      arith.addi %nonzero, %c1 : index
    }
    %result = scf.for %carrier = %c0 to %upper step %c1 iter_args(%accumulator = %c0) -> (index) {
      %next = arith.addi %accumulator, %carrier : index
      scf.yield %next : index
    }
    arith.addi %result, %c1 : index
  }
  return
}

// CHECK-LABEL: func.func @keep_loop_outside_group(%arg0: index) {
// CHECK: scf.for
// CHECK-NOT: emitasc.vf_for
func.func @keep_loop_outside_group(%upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %index = %c0 to %upper step %c1 {
    arith.addi %index, %c1 : index
  }
  return
}
