// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-legalize-vf-group %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_static_local_tensor_auto() -> !ascendc.local_tensor<32xf32> {
// CHECK: %[[TENSOR:.*]] = ascendc.local_tensor_auto veccalc()
// CHECK: %[[GROUP:.*]] = ascvf2.vf_group {
// CHECK-NEXT: ascvf2.yield %[[TENSOR]]
// CHECK-NEXT: } : !ascendc.local_tensor<32xf32>
// CHECK: return %[[GROUP]]
func.func @hoist_static_local_tensor_auto() -> !ascendc.local_tensor<32xf32> {
  %result = ascvf2.vf_group {
    %tensor = ascendc.local_tensor_auto veccalc() : <32xf32>
    ascvf2.yield %tensor : !ascendc.local_tensor<32xf32>
  } : !ascendc.local_tensor<32xf32>
  return %result : !ascendc.local_tensor<32xf32>
}

// CHECK-LABEL: func.func @keep_dynamic_local_tensor_auto(%arg0: i64) -> !ascendc.local_tensor<32xf32> {
// CHECK: %[[GROUP:.*]] = ascvf2.vf_group {
// CHECK-NEXT: %[[TENSOR:.*]] = ascendc.local_tensor_auto veccalc(%arg0)
// CHECK-NEXT: ascvf2.yield %[[TENSOR]]
// CHECK-NEXT: } : !ascendc.local_tensor<32xf32>
// CHECK: return %[[GROUP]]
func.func @keep_dynamic_local_tensor_auto(%shape: i64) -> !ascendc.local_tensor<32xf32> {
  %result = ascvf2.vf_group {
    %tensor = ascendc.local_tensor_auto veccalc(%shape) : <32xf32>
    ascvf2.yield %tensor : !ascendc.local_tensor<32xf32>
  } : !ascendc.local_tensor<32xf32>
  return %result : !ascendc.local_tensor<32xf32>
}

// CHECK-LABEL: func.func @convert_local_tensor_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK: %[[INPUT:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK: %[[GROUP:.*]] = ascvf2.vf_group {
// CHECK-NEXT: ascvf2.yield %[[INPUT]]
// CHECK-NEXT: } : !ascendc.local_tensor<32xf32>
// CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[GROUP]] : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK: return %[[RESULT]]
func.func @convert_local_tensor_group(%tensor: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %carrier = ascendc.local_tensor_auto veccalc() : <32xf32>
  %result = ascvf2.vf_group {
    ascvf2.yield %tensor : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}
