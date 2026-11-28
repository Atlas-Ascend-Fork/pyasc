// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-assign-vf-group-tensors %s | FileCheck %s

// CHECK-LABEL: func.func @classify_operand_lists(%arg0: !ascendc.local_tensor<8xf32>, %arg1: !ascendc.local_tensor<8xf32>, %arg2: !ascendc.local_tensor<8xf32>, %arg3: index) {
// CHECK: ascvf2.vf_group ins(%arg0 : !ascendc.local_tensor<8xf32>) outs(%arg1 : !ascendc.local_tensor<8xf32>) tmps(%arg2 : !ascendc.local_tensor<8xf32>) {
func.func @classify_operand_lists(%input: !ascendc.local_tensor<8xf32>, %output: !ascendc.local_tensor<8xf32>, %temporary: !ascendc.local_tensor<8xf32>, %offset: index) {
  ascvf2.vf_group {
    %value = ascendc.reg_tensor : <f32>
    %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    %inputAddress = ascendc.local_tensor.get_phy_addr_v2 %input : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %inputPointer = emitasc.ptr_offset %inputAddress[%offset] : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vld_reg %value, %inputPointer {dist = 0 : i32} : !ascendc.reg_tensor<f32>, memref<8xf32, 26>
    %temporaryAddress = ascendc.local_tensor.get_phy_addr_v2 %temporary : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %temporaryPointer = emitasc.ptr_offset %temporaryAddress[%offset] : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vst_reg %temporaryPointer, %value, %mask {dist = 16 : i32} : memref<8xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    %outputAddress = ascendc.local_tensor.get_phy_addr_v2 %output : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %outputPointer = emitasc.ptr_offset %outputAddress[%offset] : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vst_reg %outputPointer, %value, %mask {dist = 16 : i32} : memref<8xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascvf2.yield
  }
  %outputAddress = ascendc.local_tensor.get_phy_addr_v2 %output : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
  return
}

// CHECK-LABEL: func.func @classify_in_place_tensor(%arg0: !ascendc.local_tensor<8xf32>, %arg1: index) {
// CHECK: ascvf2.vf_group ins(%arg0 : !ascendc.local_tensor<8xf32>) outs(%arg0 : !ascendc.local_tensor<8xf32>) {
func.func @classify_in_place_tensor(%tensor: !ascendc.local_tensor<8xf32>, %offset: index) {
  ascvf2.vf_group {
    %value = ascendc.reg_tensor : <f32>
    %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    %address = ascendc.local_tensor.get_phy_addr_v2 %tensor : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %pointer = emitasc.ptr_offset %address[%offset] : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vld_reg %value, %pointer {dist = 0 : i32} : !ascendc.reg_tensor<f32>, memref<8xf32, 26>
    ascendc.data_copy_vst_reg %pointer, %value, %mask {dist = 16 : i32} : memref<8xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascvf2.yield
  }
  %address = ascendc.local_tensor.get_phy_addr_v2 %tensor : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
  return
}

// CHECK-LABEL: func.func @skip_tensor_without_memory_access(%arg0: !ascendc.local_tensor<8xf32>) {
// CHECK: ascvf2.vf_group {
func.func @skip_tensor_without_memory_access(%tensor: !ascendc.local_tensor<8xf32>) {
  ascvf2.vf_group {
    %address = ascendc.local_tensor.get_phy_addr_v2 %tensor : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    ascvf2.yield
  }
  return
}
