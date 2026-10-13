// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -ascendc-insert-cross-core-sync %s | FileCheck %s

// CHECK-LABEL: func.func @aiv_trigger_no_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @aiv_trigger_no_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    %c256 = arith.constant 256 : i32
    ascendc.data_copy_l2 %1, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aic(%0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @same_core_no_trigger(%arg0: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @same_core_no_trigger(%arg0: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c32 = arith.constant 32 : i64
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.relu_l2 %1, %arg0, %c32 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i64
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aic(%0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    ascendc.yield %0 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @loop_forward_backward_sync(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:  %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %arg1) -> (!ascendc.local_tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:      ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_mte3 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @loop_forward_backward_sync(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg1) -> !ascendc.local_tensor<16x16xf32> : i32 {
    ascendc.if_aiv {
      ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
      ascendc.yield
    }
    %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    scf.yield %0 : !ascendc.local_tensor<16x16xf32>
  }
  return %result : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_two_aic_consumer_groups(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %0, %c256_i32 {direction = #ascendc.copy_direction<a1, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %5 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %5 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_two_aic_consumer_groups(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %co2 = ascendc.local_tensor_v3 co1, 1024, 1024 : !ascendc.local_tensor<16x16xf32>
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  ascendc.if_aiv {
    ascendc.data_copy_l2 %ub, %dst, %c256 {direction = #ascendc.copy_direction<a1, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %1 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co2, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @same_core_triggers(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 0 : i32} {
// CHECK-NOT: cross_core
func.func @same_core_triggers(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c256 = arith.constant 256 : i32
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.data_copy_l2 %1, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  %1 = ascendc.if_aiv(%arg1 : !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %2 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    ascendc.data_copy_l2 %2, %arg1, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
  }
  return %1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aiv_trigger_two_consumers_same_group(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.mmad %1, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.mmad %2, %0, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @aiv_trigger_two_consumers_same_group(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %co2 = ascendc.local_tensor_v3 co1, 1024, 1024 : !ascendc.local_tensor<16x16xf32>
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.mmad %co2, %dst, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @consumer_uses_tensor_in_nested_loop({{.*}}) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @consumer_uses_tensor_in_nested_loop(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf16>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
    }
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @second_trigger_in_different_loop(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %arg1) -> (!ascendc.local_tensor<16x16xf16>)  : i32 {
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:      ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_mte3 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %arg1 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aic {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %arg1) -> (!ascendc.local_tensor<16x16xf16>)  : i32 {
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:      ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_mte3 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:      ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %arg1 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @second_trigger_in_different_loop(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf16>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  %result1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg1) -> !ascendc.local_tensor<16x16xf16> : i32 {
    ascendc.if_aiv {
      ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
      ascendc.yield
    }
    %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    scf.yield %arg1 : !ascendc.local_tensor<16x16xf16>
  }
  %result2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg1) -> !ascendc.local_tensor<16x16xf16> : i32 {
    ascendc.if_aiv {
      ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
      ascendc.yield
    }
    %1 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    scf.yield %arg1 : !ascendc.local_tensor<16x16xf16>
  }
  return %co1 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @scf_if_inside_scf_for
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      scf.if %true {
// CHECK-NEXT:        ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @scf_if_inside_scf_for(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf16>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  %true = arith.constant 1 : i1
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      scf.if %true {
        ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      }
    }
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @scf_for_inside_scf_if
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_m : i32
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:        ascendc.mmad %1, %0, %0, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_1, 4, pipe_m : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @scf_for_inside_scf_if(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf16>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c256 = arith.constant 256 : i32
  %true = arith.constant 1 : i1
  ascendc.if_aiv {
    ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    scf.if %true {
      scf.for %i = %c0 to %c2 step %c1 : i32 {
        ascendc.mmad %co1, %dst, %dst, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      }
    }
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @cv_ratio_2_dual_sync(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) -> !ascendc.local_tensor<8xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       %0 = ascendc.local_tensor_v3 co1, 0, 64 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.if_aic -> !ascendc.local_tensor<8xf32> {
// CHECK-NEXT:    ascendc.fixpipe %1, %0, %arg0, %arg1 {direction = #ascendc.copy_direction<co1, veccalc>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_fix : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32, 4, pipe_fix : i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aiv -> !ascendc.local_tensor<8xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_v : i32
// CHECK-NEXT:    ascendc.add_l3 %2, %1, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_v : i32
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %4 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT:}
module attributes {ascendc.cv_ratio = 2 : i64} {
  func.func @cv_ratio_2_dual_sync(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) -> !ascendc.local_tensor<8xf32> {
    %co1 = ascendc.local_tensor_v3 co1, 0, 64 : !ascendc.local_tensor<8xf32>
    %ub = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<8xf32>
    %result = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<8xf32>
    %0 = ascendc.if_aic -> !ascendc.local_tensor<8xf32> {
      ascendc.fixpipe %ub, %co1, %arg0, %arg1 {direction = #ascendc.copy_direction<co1, veccalc>} : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
      ascendc.yield %ub : !ascendc.local_tensor<8xf32>
    }
    %1 = ascendc.if_aiv -> !ascendc.local_tensor<8xf32> {
      ascendc.add_l3 %result, %ub, %ub : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
      ascendc.yield %result : !ascendc.local_tensor<8xf32>
    }
    return %1 : !ascendc.local_tensor<8xf32>
  }
}

// -----
 
// CHECK-LABEL: func.func @cv_ratio_2_aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params, %arg3: !ascendc.load_data_2d_params_v2) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_mte1 : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c16_i32, 4, pipe_mte1 : i32
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %0, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    ascendc.load_data_l0_v2 %2, %0, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    ascendc.mmad %3, %1, %2, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte1 : i32
// CHECK-NEXT:    %c16_i32_1 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32_1, 4, pipe_mte1 : i32
// CHECK-NEXT:    ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %4 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
module attributes {ascendc.cv_ratio = 2 : i64} {
  func.func @cv_ratio_2_aiv_trigger_aic_consumer(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params, %arg3: !ascendc.load_data_2d_params_v2) -> !ascendc.local_tensor<16x16xf32> {
    %dst = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
    %l0a = ascendc.local_tensor_v3 a2, 0, 256 : !ascendc.local_tensor<16x16xf32>
    %l0b = ascendc.local_tensor_v3 b2, 0, 256 : !ascendc.local_tensor<16x16xf32>
    %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    %c256 = arith.constant 256 : i32
    ascendc.if_aiv {
      ascendc.data_copy_l2 %dst, %arg0, %c256 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, i32
      ascendc.yield
    }
    %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.load_data_l0_v2 %l0a, %dst, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.load_data_2d_params_v2
      ascendc.load_data_l0_v2 %l0b, %dst, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.load_data_2d_params_v2
      ascendc.mmad %co1, %l0a, %l0b, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    return %0 : !ascendc.local_tensor<16x16xf32>
  }
}
 
// -----
 
// CHECK-LABEL: func.func @cv_ratio_2_inner_loop_trigger(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params, %arg3: !ascendc.load_data_2d_params_v2, %arg4: i32) -> !ascendc.local_tensor<16x16xf32> attributes {ascendc.cross_core_flag_id = 1 : i32} {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    scf.for %arg5 = %arg4 to %arg4 step %arg4  : i32 {
// CHECK-NEXT:      %5 = arith.muli %arg5, %arg4 : i32
// CHECK-NEXT:      %6 = ascendc.local_tensor.subindex %0[%5] : !ascendc.local_tensor<16x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:      ascendc.data_copy_l2 %6, %arg0, %arg4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32, 4, pipe_mte1 : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c16_i32, 4, pipe_mte1 : i32
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %0, %arg3 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    ascendc.load_data_l0_v2 %2, %0, %arg3 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    ascendc.mmad %3, %1, %2, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte1 : i32
// CHECK-NEXT:    %c16_i32_1 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32_1, 4, pipe_mte1 : i32
// CHECK-NEXT:    ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %4 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
module attributes {ascendc.cv_ratio = 2 : i64} {
  func.func @cv_ratio_2_inner_loop_trigger(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.mmad_params, %arg3: !ascendc.load_data_2d_params_v2, %arg4: i32) -> !ascendc.local_tensor<16x16xf32> {
    %l1 = ascendc.local_tensor_v3 a1, 0, 1024 : !ascendc.local_tensor<16x16xf16>
    %l0a = ascendc.local_tensor_v3 a2, 0, 256 : !ascendc.local_tensor<16x16xf16>
    %l0b = ascendc.local_tensor_v3 b2, 0, 256 : !ascendc.local_tensor<16x16xf16>
    %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.if_aiv {
      scf.for %i = %arg4 to %arg4 step %arg4 : i32 {
        %offset = arith.muli %i, %arg4 : i32
        %dst = ascendc.local_tensor.subindex %l1[%offset] : !ascendc.local_tensor<16x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
        ascendc.data_copy_l2 %dst, %arg0, %arg4 {direction = #ascendc.copy_direction<veccalc, a1>} : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, i32
      }
      ascendc.yield
    }
    %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      ascendc.load_data_l0_v2 %l0a, %l1, %arg3 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.load_data_2d_params_v2
      ascendc.load_data_l0_v2 %l0b, %l1, %arg3 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.load_data_2d_params_v2
      ascendc.mmad %co1, %l0a, %l0b, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
    }
    return %0 : !ascendc.local_tensor<16x16xf32>
  }
}
