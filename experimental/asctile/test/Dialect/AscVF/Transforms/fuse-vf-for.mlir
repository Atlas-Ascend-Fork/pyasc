// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf-fuse-vf-for %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_sequential_same_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     emitasc.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  emitasc.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, <1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, <1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, <1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @fuse_sequential_same_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index, !ascendc.mask_reg
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index, !ascendc.mask_reg
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1], %1 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index, !ascendc.mask_reg
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}

// CHECK-LABEL: func.func @dont_fuse_non_sequential(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     emitasc.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  emitasc.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  ascvf.load %0, %arg0[%c0], %1 : !ascendc.reg_tensor<f32>, <1024xf32>, index, !ascendc.mask_reg
// CHECK-NEXT:  emitasc.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @dont_fuse_non_sequential(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.load %0, %arg0[%c0], %1 : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index, !ascendc.mask_reg
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}

// CHECK-LABEL: func.func @dont_fuse_different_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     emitasc.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  emitasc.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  emitasc.vf_for %c1 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @dont_fuse_different_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    emitasc.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    emitasc.vf_for %c1 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}

// CHECK-LABEL:func.func @fuse_barrier(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:       emitasc.vec_scope {
// CHECK-NEXT:    %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:    emitasc.vf_for %c1 : index {
// CHECK-NEXT:    ^bb0(%arg1: index):
// CHECK-NEXT:      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:      ascvf.barrier
// CHECK-NEXT:      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fuse_barrier(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c1 = arith.constant 1 : index
  emitasc.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    emitasc.vf_for %c1 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.barrier
    emitasc.vf_for %c1 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}
