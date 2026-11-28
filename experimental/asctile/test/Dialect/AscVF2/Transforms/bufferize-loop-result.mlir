// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-bufferize-loop-result -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @reuse_loop_result(%arg0: index, %arg1: f32) attributes {ascvf2.vector_func} {
// CHECK: %[[INIT:.*]] = ascendc.reg_tensor
// CHECK: scf.for {{.*}} to %arg0 step
// CHECK: ascendc.max_reg %[[INIT]], %[[INIT]], %{{.*}}, %{{.*}}
// CHECK-NOT: iter_args
// CHECK-NOT: scf.yield
// CHECK: ascendc.reduce_max_reg
func.func @reuse_loop_result(%upper: index, %value: f32) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %input = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %output = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %mask = ascendc.mask_reg : !ascendc.mask_reg
  ascendc.duplicate_reg %init, %value, %mask : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  %result = scf.for %index = %c0 to %upper step %c1 iter_args(%acc = %init) -> !ascendc.reg_tensor<f32> {
    %destination = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
    ascendc.max_reg %destination, %acc, %input, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    scf.yield %destination : !ascendc.reg_tensor<f32>
  }
  ascendc.reduce_max_reg %output, %result, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @keep_live_init(%arg0: index) attributes {ascvf2.vector_func} {
// CHECK: scf.for {{.*}} to %arg0 step {{.*}} iter_args
// CHECK: scf.yield %{{.*}}
func.func @keep_live_init(%upper: index) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %input = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %mask = ascendc.mask_reg : !ascendc.mask_reg
  %result = scf.for %index = %c0 to %upper step %c1 iter_args(%acc = %init) -> !ascendc.reg_tensor<f32> {
    %destination = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
    ascendc.max_reg %destination, %acc, %input, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    scf.yield %destination : !ascendc.reg_tensor<f32>
  }
  ascendc.reduce_max_reg %init, %init, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_max_reg %init, %result, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @reuse_nested_loop_result(%arg0: index, %arg1: f32) attributes {ascvf2.vector_func} {
// CHECK: scf.for
// CHECK: %[[ACC:.*]] = ascendc.reg_tensor
// CHECK: ascendc.duplicate_reg %[[ACC]], %arg1, %{{.*}}
// CHECK: scf.for
// CHECK: ascendc.max_reg %[[ACC]], %[[ACC]], %{{.*}}, %{{.*}}
// CHECK: ascendc.reduce_max_reg
func.func @reuse_nested_loop_result(%upper: index, %value: f32) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %input = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %output = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %mask = ascendc.mask_reg : !ascendc.mask_reg
  ascendc.duplicate_reg %init, %value, %mask : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  scf.for %row = %c0 to %upper step %c1 {
    %result = scf.for %index = %c0 to %upper step %c1 iter_args(%acc = %init) -> !ascendc.reg_tensor<f32> {
      %destination = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
      ascendc.max_reg %destination, %acc, %input, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      scf.yield %destination : !ascendc.reg_tensor<f32>
    }
    ascendc.reduce_max_reg %output, %result, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function(%arg0: index) {
// CHECK: scf.for {{.*}} iter_args
func.func @ignore_non_vector_function(%upper: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %input = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  %mask = ascendc.mask_reg : !ascendc.mask_reg
  %result = scf.for %index = %c0 to %upper step %c1 iter_args(%acc = %init) -> !ascendc.reg_tensor<f32> {
    %destination = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
    ascendc.max_reg %destination, %acc, %input, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    scf.yield %destination : !ascendc.reg_tensor<f32>
  }
  return
}
