// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf-eliminate-data-transfer -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL:func.func @dont_create_load_for_loaded_reg_tensor
// CHECK:     emitasc.vf_for %10 : index {
// CHECK-NEXT:^bb0(%arg6: index):
// CHECK-NEXT:  %11 = arith.muli %arg6, %8 : index
// CHECK-NEXT:  %12 = ascendc.update_mask f32, %9 : memref<1xui32>
// CHECK-NEXT:  ascvf.load %0, %arg2[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.load %1, %arg3[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascendc.add_reg %2, %0, %1, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg4[%11], %2, %12 : <1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.load %3, %arg4[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.load %4, %arg2[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascendc.add_reg %5, %3, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg5[%11], %5, %12 : <1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:}
func.func @dont_create_load_for_loaded_reg_tensor(%arg0: !ascendc.global_tensor<?x?xf32>, %arg1: !ascendc.data_copy_ext_params, %arg2: !ascendc.local_tensor<1x1024xf32>, %arg3: !ascendc.local_tensor<1x1024xf32>, %arg4: !ascendc.local_tensor<1x1024xf32>, %arg5: !ascendc.local_tensor<1x1024xf32>) {
  ascvf.vf_group dst(%arg4, %arg5 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>) src(%arg3, %arg2 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>) {
    emitasc.vec_scope {
      %c4 = arith.constant 4 : index
      %0 = ascendc.reg_tensor : <f32>
      %1 = ascendc.reg_tensor : <f32>
      %2 = ascendc.reg_tensor : <f32>
      %3 = ascendc.reg_tensor : <f32>
      %4 = ascendc.reg_tensor : <f32>
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
      %7 = ascendc.get_vec_len : index
      %8 = arith.divsi %7, %c4 : index
      %9 = arith.constant 1024 : index
      %10 = emitasc.variable %9 : index, memref<1xui32>
      %11 = arith.ceildivsi %9, %8 : index
      %12 = emitasc.variable %9 : index, memref<1xui32>
      emitasc.vf_for %11 : index {
      ^bb0(%arg7: index):
        %13 = arith.muli %arg7, %8 : index
        %14 = ascendc.update_mask f32, %10 : memref<1xui32>
        ascvf.load %0, %arg2[%13], %14 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascvf.load %1, %arg3[%13], %14 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascendc.add_reg %2, %0, %1, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg4[%13], %2, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %15 = arith.muli %arg7, %8 : index
        ascvf.load %3, %arg4[%15], %14 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascvf.load %4, %arg2[%15], %14 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascendc.add_reg %5, %3, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%15], %5, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {groupType = !ascendc.local_tensor<1024xf32>}
  ascendc.data_copy_pad_l2_ext %arg0, %arg5, %arg1 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}

// CHECK-LABEL: func.func @dont_rewrite_memory
// CHECK:     emitasc.vf_for %10 : index {
// CHECK-NEXT:^bb0(%arg6: index):
// CHECK-NEXT:  %11 = arith.muli %arg6, %8 : index
// CHECK-NEXT:  %12 = ascendc.update_mask f32, %9 : memref<1xui32>
// CHECK-NEXT:  ascvf.load %0, %arg5[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascendc.exp_reg %1, %0, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg5[%11], %1, %12 : <1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.load %2, %arg5[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascendc.exp_reg %3, %2, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg5[%11], %3, %12 : <1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.load %4, %arg5[%11], %12 : !ascendc.reg_tensor<f32>, <1x1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  ascendc.exp_reg %5, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg5[%11], %5, %12 : <1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:}
func.func @dont_rewrite_memory(%arg0: !ascendc.global_tensor<?x?xf32>, %arg1: !ascendc.data_copy_ext_params, %arg2: !ascendc.local_tensor<1x1024xf32>, %arg3: !ascendc.local_tensor<1x1024xf32>, %arg4: !ascendc.local_tensor<1x1024xf32>, %arg5: !ascendc.local_tensor<1x1024xf32>) {
  ascvf.vf_group dst(%arg5 : !ascendc.local_tensor<1x1024xf32>) src(%arg5 : !ascendc.local_tensor<1x1024xf32>) {
    emitasc.vec_scope {
      %c4 = arith.constant 4 : index
      %0 = ascendc.reg_tensor : <f32>
      %1 = ascendc.reg_tensor : <f32>
      %2 = ascendc.reg_tensor : <f32>
      %3 = ascendc.reg_tensor : <f32>
      %4 = ascendc.reg_tensor : <f32>
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
      %7 = ascendc.get_vec_len : index
      %8 = arith.divsi %7, %c4 : index
      %9 = arith.constant 1024 : index
      %10 = emitasc.variable %9 : index, memref<1xui32>
      %11 = arith.ceildivsi %9, %8 : index
      %12 = emitasc.variable %9 : index, memref<1xui32>
      %13 = emitasc.variable %9 : index, memref<1xui32>
      emitasc.vf_for %11 : index {
      ^bb0(%arg7: index):
        %14 = arith.muli %arg7, %8 : index
        %15 = ascendc.update_mask f32, %10 : memref<1xui32>
        ascvf.load %0, %arg5[%14], %15 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascendc.exp_reg %1, %0, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%14], %1, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %16 = arith.muli %arg7, %8 : index
        ascvf.load %2, %arg5[%16], %15 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascendc.exp_reg %3, %2, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%16], %3, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %17 = arith.muli %arg7, %8 : index
        ascvf.load %4, %arg5[%17], %15 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index, !ascendc.mask_reg
        ascendc.exp_reg %5, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%17], %5, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {groupType = !ascendc.local_tensor<1x1024xf32>}
  ascendc.data_copy_pad_l2_ext %arg0, %arg5, %arg1 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}
