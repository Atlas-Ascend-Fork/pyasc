// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascendc-insert-cross-core-sync-gm %s | FileCheck %s

// CHECK-LABEL: func.func @loop_fixpipe_to_gm(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.local_tensor<16x16xf32>, %arg4: !ascendc.mmad_params, %arg5: !ascendc.fixpipe_params_v220, %arg6: !ascendc.fixpipe_config) -> !ascendc.local_tensor<16x16xf32> {
// CHECK:       %3 = scf.for
// CHECK-NEXT:    ascendc.if_aic {
// CHECK-NEXT:      ascendc.mmad %0, %arg1, %arg2, %arg4 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      ascendc.fixpipe %1, %0, %arg5, %arg6 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_fix : i32
// CHECK-NEXT:      %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:      ascendc.cross_core_set_flag %c16_i32, 4, pipe_fix : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:      ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:      ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %arg8 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @loop_fixpipe_to_gm(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.local_tensor<16x16xf32>, %arg4: !ascendc.mmad_params, %arg5: !ascendc.fixpipe_params_v220, %arg6: !ascendc.fixpipe_config) -> !ascendc.local_tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c256_i32 = arith.constant 256 : i32
  %0 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %1 = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  %2 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %result = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %arg3) -> !ascendc.local_tensor<16x16xf32> : i32 {
    ascendc.if_aic {
      ascendc.mmad %0, %arg1, %arg2, %arg4 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
      ascendc.fixpipe %1, %0, %arg5, %arg6 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
      ascendc.yield
    }
    ascendc.if_aiv {
      ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
      ascendc.yield
    }
    scf.yield %acc : !ascendc.local_tensor<16x16xf32>
  }
  return %result : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @aic_fixpipe_to_gm(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.mmad_params, %arg4: !ascendc.fixpipe_params_v220, %arg5: !ascendc.fixpipe_config) {
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:    ascendc.mmad %0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.fixpipe %1, %0, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_fix : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c16_i32, 4, pipe_fix : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:  }
func.func @aic_fixpipe_to_gm(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.mmad_params, %arg4: !ascendc.fixpipe_params_v220, %arg5: !ascendc.fixpipe_config) {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %0 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %1 = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  %2 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  ascendc.if_aic {
    ascendc.mmad %0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
    ascendc.fixpipe %1, %0, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    ascendc.yield
  }
  ascendc.if_aiv {
    ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @ub_to_gm_sync(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 {{%.*}}, {{%.*}}, {{%.*}} {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c16_i32, 4, pipe_s : i32
// CHECK-NEXT:    ascendc.data_copy_l2 {{%.*}}, %arg0, {{%.*}} {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:    ascendc.mmad {{%.*}}, {{%.*}}, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.yield {{%.*}} : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return {{%.*}} : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @ub_to_gm_sync(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf32>, %arg2: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %a1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aiv {
    ascendc.data_copy_l2 %gm_sub, %ub, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.data_copy_l2 %a1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.mmad %co1, %a1, %arg1, %arg2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @ub_to_gm_no_consumer
// CHECK: ascendc.data_copy_l2 {{%.*}}, {{%.*}}, {{%.*}} {direction = #ascendc.copy_direction<veccalc, gm>}
// CHECK-NOT: cross_core
func.func @ub_to_gm_no_consumer(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.mmad_params) -> !ascendc.local_tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aiv {
    ascendc.data_copy_l2 %gm_sub, %ub, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.mmad %co1, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
    ascendc.yield %co1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @fixpipe_to_gm_no_consumer
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:    ascendc.mmad %0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.fixpipe %2, %0, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.duplicate_l2 %1, %c0_i32, %c256_i32 : !ascendc.local_tensor<16x16xf32>, i32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fixpipe_to_gm_no_consumer(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.mmad_params, %arg4: !ascendc.fixpipe_params_v220, %arg5: !ascendc.fixpipe_config) {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aic {
    ascendc.mmad %co1, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
    ascendc.fixpipe %gm_sub, %co1, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    ascendc.yield
  }
  ascendc.if_aiv {
    ascendc.duplicate_l2 %ub, %c0_i32, %c256_i32 : !ascendc.local_tensor<16x16xf32>, i32, i32
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @ub_to_gm_same_core_type
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = ascendc.if_aiv -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %3 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @ub_to_gm_same_core_type(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %ub2 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aiv {
    ascendc.data_copy_l2 %gm_sub, %ub, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aiv -> !ascendc.local_tensor<16x16xf32> {
    ascendc.data_copy_l2 %ub2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.yield %ub2 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @fixpipe_to_gm_same_core_type
// CHECK:       ascendc.if_aic {
// CHECK-NEXT:    ascendc.mmad %0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    ascendc.fixpipe %2, %0, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aic {
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fixpipe_to_gm_same_core_type(%arg0: !ascendc.global_tensor<32x32xf32>, %arg1: !ascendc.local_tensor<16x16xf16>, %arg2: !ascendc.local_tensor<16x16xf16>, %arg3: !ascendc.mmad_params, %arg4: !ascendc.fixpipe_params_v220, %arg5: !ascendc.fixpipe_config) {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %co1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
  %a1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aic {
    ascendc.mmad %co1, %arg1, %arg2, %arg3 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
    ascendc.fixpipe %gm_sub, %co1, %arg4, %arg5 {direction = #ascendc.copy_direction<co1, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    ascendc.yield
  }
  ascendc.if_aic {
    ascendc.data_copy_l2 %a1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @ub_to_gm_mixed_consumers
// CHECK:       ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %0, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_set_flag %c0_i32_0, 4, pipe_mte3 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.if_aiv {
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c0_i32_0, 4, pipe_s : i32
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    ascendc.cross_core_wait_flag %c16_i32, 4, pipe_s : i32
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
// CHECK-NEXT:    ascendc.yield %2 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %4 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @ub_to_gm_mixed_consumers(%arg0: !ascendc.global_tensor<32x32xf32>) -> !ascendc.local_tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %ub = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %ub2 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %a1 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<16x16xf32>
  %gm_sub = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
  ascendc.if_aiv {
    ascendc.data_copy_l2 %gm_sub, %ub, %c256_i32 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, i32
    ascendc.yield
  }
  ascendc.if_aiv {
    ascendc.data_copy_l2 %ub2, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.yield
  }
  %0 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
    ascendc.data_copy_l2 %a1, %arg0, %c256_i32 {direction = #ascendc.copy_direction<gm, a1>} : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, i32
    ascendc.yield %a1 : !ascendc.local_tensor<16x16xf32>
  }
  return %0 : !ascendc.local_tensor<16x16xf32>
}
