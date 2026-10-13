// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-input-output-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @input_output_tensor_ub(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg3 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg1, %arg3 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %arg3 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %3, %arg3 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @input_output_tensor_ub(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg3 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %arg1, %arg3 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %2 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %arg2, %2, %arg3 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %2, %3, %arg3 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @scf_for_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:  %2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:    scf.yield %arg2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  }
// CHECK:       ascendc.data_copy_l2 %1, %2, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @scf_for_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<64xf32>
    }
    ascendc.data_copy_l2 %arg0, %result, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    return
}

// CHECK-LABEL: func.func @scf_for_yield_tensor_align_copy_count(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <1xf32>
// CHECK-NEXT:  %2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<1xf32>) {
// CHECK-NEXT:    scf.yield %arg2 : !ascendc.local_tensor<1xf32>
// CHECK-NEXT:  }
// CHECK:       ascendc.data_copy_l2 %1, %2, %c8_i64 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c1_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<1xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @scf_for_yield_tensor_align_copy_count(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <1xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<1xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<1xf32>
    }    
    ascendc.data_copy_l2 %arg0, %result, %c1_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<1xf32>, i64
    return
}

// CHECK-LABEL: func.func @scf_if_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>, %arg1: i1) {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:  %2 = scf.if %arg1 -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:    scf.yield %0 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %0 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  }
// CHECK:       ascendc.data_copy_l2 %1, %2, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @scf_if_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>, %cond: i1) {
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.if %cond -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %0 : !ascendc.local_tensor<64xf32>
    } else {
        scf.yield %0 : !ascendc.local_tensor<64xf32>
    }
    ascendc.data_copy_l2 %arg0, %result, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    return
}

// CHECK-LABEL: func.func @subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input output : <128xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %1, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <128xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
    ascendc.data_copy_l2 %arg1, %1, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @copy_to_l0_mark_input_output(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.load_data_2d_params) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto b1() input output : <64xf16>
// CHECK-NEXT:  ascendc.load_data_g2l %0, %arg0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<*xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <64xf16>
// CHECK-NEXT:  ascendc.load_data_g2l %1, %0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @copy_to_l0_mark_input_output(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.load_data_2d_params) {
    %0 = ascendc.local_tensor_auto b1() : <64xf16>
    ascendc.load_data_g2l %0, %arg0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<*xf16>, !ascendc.load_data_2d_params
    %1 = ascendc.local_tensor_auto b2() : <64xf16>
    ascendc.load_data_g2l %1, %0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, !ascendc.load_data_2d_params
    return
}

// CHECK-LABEL: func.func @fixpipe_mark_input_output(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto a1() output : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto co1() input : <64xf32>
// CHECK-NEXT:  ascendc.fixpipe %1, %0, %arg0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fixpipe_mark_input_output(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) {
    %0 = ascendc.local_tensor_auto a1() : <64xf32>
    %1 = ascendc.local_tensor_auto co1() : <64xf32>
    ascendc.fixpipe %1, %0, %arg0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    return
}

// CHECK-LABEL: func.func @reinterpret_and_subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input output : <128xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<128xf32> to !ascendc.local_tensor<64xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<64xf32> to !ascendc.local_tensor<64xf16>
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %3[%arg3] : !ascendc.local_tensor<64xf16>, i32, !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<32xf16> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %5, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<32xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reinterpret_and_subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <128xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
    %cast1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<128xf32> to !ascendc.local_tensor<64xf16>
    %sub = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
    %cast2 = ascendc.reinterpret_cast %sub : !ascendc.local_tensor<64xf32> to !ascendc.local_tensor<64xf16>
    %sub2 = ascendc.local_tensor.subindex %cast2[%arg3] : !ascendc.local_tensor<64xf16>, i32, !ascendc.local_tensor<32xf16>
    %cast3 = ascendc.reinterpret_cast %sub2 : !ascendc.local_tensor<32xf16> to !ascendc.local_tensor<32xf32>
    ascendc.data_copy_l2 %arg1, %cast3, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<32xf32>, i32
    return
}

// CHECK-LABEL: func.func @scf_for_no_init_args() {
// CHECK:         scf.for %arg0 = %c0 to %c10 step %c1 {
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_no_init_args() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c10 step %c1 {
    }
    return
}

// CHECK-LABEL: func.func @scf_for_no_init_args_use_inside(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c10 step %c1 {
// CHECK-NEXT:      ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_no_init_args_use_inside(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    scf.for %i = %c0 to %c10 step %c1 {
        ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    }
    return
}

// CHECK-LABEL: func.func @scf_for_init_dead_after_loop() {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      scf.yield %arg1 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_init_dead_after_loop() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_iter_arg_not_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.muls_l2 %2, %arg2, %cst, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:      scf.yield %2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_iter_arg_not_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f32
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %inner = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.muls_l2 %inner, %tensor, %c2, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
        scf.yield %inner : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_iter_arg_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:      ascendc.data_copy_l2 %2, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
// CHECK-NEXT:      %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.sub_l2 %3, %2, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      %4 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.if_aiv {
// CHECK-NEXT:        %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:        ascendc.data_copy_l2 %4, %2, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %4 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_iter_arg_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.data_copy_l2 %1, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
        %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.sub_l2 %2, %1, %tensor, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
        scf.yield %1 : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_yielded_from_if_aiv(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.if_aiv(%arg2 : !ascendc.local_tensor<64xf32>) -> !ascendc.local_tensor<64xf32> {
// CHECK-NEXT:        %3 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:        ascendc.data_copy_l2 %3, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
// CHECK-NEXT:        %4 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:        ascendc.sub_l2 %4, %3, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:        %5 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:        ascendc.if_aiv {
// CHECK-NEXT:          %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:          ascendc.data_copy_l2 %5, %3, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:        }
// CHECK-NEXT:        ascendc.yield %5 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_yielded_from_if_aiv(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %inner = ascendc.if_aiv(%tensor : !ascendc.local_tensor<64xf32>) -> !ascendc.local_tensor<64xf32> {
            %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
            ascendc.data_copy_l2 %1, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
            %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
            ascendc.sub_l2 %2, %1, %tensor, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
            ascendc.yield %1 : !ascendc.local_tensor<64xf32>
        }
        scf.yield %inner : !ascendc.local_tensor<64xf32>
    }
    return
}
