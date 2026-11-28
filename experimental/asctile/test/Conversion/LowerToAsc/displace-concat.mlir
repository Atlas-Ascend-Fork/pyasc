// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asclower-displace-concat -canonicalize -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @fold_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  return %arg0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = tensor.concat dim(0) %arg0 : (tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fully_displace_concat_ub() -> tensor<8x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%c0_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c48_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<5x16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x16xf32> to tensor<8x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tensor_user"(%1) : (!ascendc.local_tensor<3x16xf32>) -> ()
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<3x16xf32> to tensor<3x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tile_user"(%4) : (tensor<3x16xf32, #asctile.local<UB>>) -> ()
// CHECK-NEXT:  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to tensor<5x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tile_user"(%5) : (tensor<5x16xf32, #asctile.local<UB>>) -> ()
// CHECK-NEXT:  return %3 : tensor<8x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fully_displace_concat_ub() -> tensor<8x16xf32, #asctile.local<UB>> {
  %0 = ascendc.local_tensor_auto veccalc() : <3x16xf32>
  "tensor_user"(%0) : (!ascendc.local_tensor<3x16xf32>) -> ()
  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3x16xf32> to tensor<3x16xf32, #asctile.local<UB>>
  "tile_user"(%1) : (tensor<3x16xf32, #asctile.local<UB>>) -> ()
  %2 = ascendc.local_tensor_auto veccalc() : <5x16xf32>
  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to tensor<5x16xf32, #asctile.local<UB>>
  "tile_user"(%3) : (tensor<5x16xf32, #asctile.local<UB>>) -> ()
  %4 = tensor.concat dim(0) %1, %3 : (tensor<3x16xf32, #asctile.local<UB>>, tensor<5x16xf32, #asctile.local<UB>>) -> tensor<8x16xf32, #asctile.local<UB>>
  return %4 : tensor<8x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fully_displace_concat_l0c() -> tensor<8x16xf32, #asctile.local<L0C>> {
// CHECK:       %0 = ascendc.local_tensor_auto co1() : <8x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%c0_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<8x10xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c80_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<8x6xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x16xf32> to tensor<8x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:  "tensor_user"(%1) : (!ascendc.local_tensor<8x10xf32>) -> ()
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<8x10xf32> to tensor<8x10xf32, #asctile.local<L0C>>
// CHECK-NEXT:  "tile_user"(%4) : (tensor<8x10xf32, #asctile.local<L0C>>) -> ()
// CHECK-NEXT:  "tensor_user"(%2) : (!ascendc.local_tensor<8x6xf32>) -> ()
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x6xf32> to tensor<8x6xf32, #asctile.local<L0C>>
// CHECK-NEXT:  "tile_user"(%5) : (tensor<8x6xf32, #asctile.local<L0C>>) -> ()
// CHECK-NEXT:  return %3 : tensor<8x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @fully_displace_concat_l0c() -> tensor<8x16xf32, #asctile.local<L0C>> {
  %0 = ascendc.local_tensor_auto co1() : <8x10xf32>
  "tensor_user"(%0) : (!ascendc.local_tensor<8x10xf32>) -> ()
  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x10xf32> to tensor<8x10xf32, #asctile.local<L0C>>
  "tile_user"(%1) : (tensor<8x10xf32, #asctile.local<L0C>>) -> ()
  %2 = ascendc.local_tensor_auto co1() : <8x6xf32>
  "tensor_user"(%2) : (!ascendc.local_tensor<8x6xf32>) -> ()
  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x6xf32> to tensor<8x6xf32, #asctile.local<L0C>>
  "tile_user"(%3) : (tensor<8x6xf32, #asctile.local<L0C>>) -> ()
  %4 = tensor.concat dim(1) %1, %3 : (tensor<8x10xf32, #asctile.local<L0C>>, tensor<8x6xf32, #asctile.local<L0C>>) -> tensor<8x16xf32, #asctile.local<L0C>>
  return %4 : tensor<8x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @convert_concat_fallback(%arg0: tensor<10xf32, #asctile.local<UB>>, %arg1: tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<6xf32, #asctile.local<UB>> to !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<10xf32, #asctile.local<UB>> to !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor.subindex %2[%c0_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %1, %c10_i64 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<10xf32>, !ascendc.local_tensor<10xf32>, i64
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %2[%c10_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %4, %0, %c6_i64 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<6xf32>, !ascendc.local_tensor<6xf32>, i64
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @convert_concat_fallback(%arg0: tensor<10xf32, #asctile.local<UB>>, %arg1: tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<10xf32, #asctile.local<UB>>, tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}
