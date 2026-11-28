// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-wrap-vec-scope %s | FileCheck %s

// CHECK-LABEL: func.func @wrap_result_free_group(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
// CHECK: emitasc.vec_scope {
// CHECK: arith.addf %arg0, %arg1
// CHECK-NOT: ascvf2.vf_group
// CHECK-NOT: ascvf2.yield
func.func @wrap_result_free_group(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  ascvf2.vf_group {
    %sum = arith.addf %lhs, %rhs : vector<4xf32>
    ascvf2.yield
  }
  return
}

// CHECK-LABEL: func.func @wrap_multiple_groups(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
// CHECK: emitasc.vec_scope {
// CHECK: arith.addf %arg0, %arg1
// CHECK: emitasc.vec_scope {
// CHECK: arith.subf %arg0, %arg1
func.func @wrap_multiple_groups(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  ascvf2.vf_group {
    %sum = arith.addf %lhs, %rhs : vector<4xf32>
    ascvf2.yield
  }
  ascvf2.vf_group {
    %difference = arith.subf %lhs, %rhs : vector<4xf32>
    ascvf2.yield
  }
  return
}
