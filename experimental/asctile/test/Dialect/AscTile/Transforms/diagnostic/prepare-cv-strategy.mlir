// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-prepare-cv-strategy -split-input-file -verify-diagnostics %s

func.func @reduce_active_axis(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{cannot reduce the active split axis 0}}
    %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @reshape_active_axis(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{must preserve the active split axis}}
    %1 = asctile.reshape %0 : tensor<32x64xf32, #asctile.local<UB>> to tensor<1x2048xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @unsupported_consumer(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{is not supported by CV strategy propagation}}
    %1 = asctile.reduce_as_1d <sum> %0 : tensor<32x64xf32, #asctile.local<UB>>, f32
  }
  return
}

// -----

func.func @conflicting_split_shape(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{has conflicting CV split shape requests}}
    %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.distrib_mode<split_by_m>, asctile.split_shape = array<i64: 8, 64>} : tensor<32x64xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @unsupported_cv_strategy_result_user(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    %1 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32x64xf32, #asctile.local<UB>>
  }
  // expected-error@+1 {{is not supported by CV strategy propagation}}
  %2 = asctile.reduce_as_1d <sum> %0 : tensor<32x64xf32, #asctile.local<UB>>, f32
  return
}

// -----

func.func @external_result_used_inside_cv(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
    %1 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32x64xf32, #asctile.local<UB>>
  }
  %2 = asctile.cv_strategy <split_by_n> -> tensor<32x64xf32, #asctile.local<UB>> {
    %3 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{tensor defined inside a CV strategy (split_by_m) cannot be used inside another CV strategy with different 'split' argument (split_by_n)}}
    %4 = arith.addf %3, %0 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.yield %4 : tensor<32x64xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @split_by_m_1d_broadcast(%arg0: tensor<32xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32] : tensor<32xf32, #asctile.local<L0C>>, tensor<32xf32, #asctile.local<UB>>
    // expected-error@+1 {{cannot map active split axis 0 to result axis 1}}
    %1 = asctile.broadcast %0 : tensor<32xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @reshape_moves_active_axis(%arg0: tensor<64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_n> {
    %0 = asctile.copy %arg0[%c0_i32] : tensor<64xf32, #asctile.local<L0C>>, tensor<64xf32, #asctile.local<UB>>
    // expected-error@+1 {{must only expand the non-active split axis}}
    %1 = asctile.reshape %0 : tensor<64xf32, #asctile.local<UB>> to tensor<64x1xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @loop_iter_arg_used_outside_cv(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> tensor<32x64xf32, #asctile.local<UB>> {
    %1 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
      %2 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
      %3 = arith.mulf %2, %arg3 : tensor<32x64xf32, #asctile.local<UB>>
      asctile.yield %3 : tensor<32x64xf32, #asctile.local<UB>>
    }
    // expected-error@+1 {{must use the loop-carried value only inside the corresponding CV strategy}}
    asctile.dump_tensor %arg3 : tensor<32x64xf32, #asctile.local<UB>>
    scf.yield %1 : tensor<32x64xf32, #asctile.local<UB>>
  }
  asctile.store %0, %arg2[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// -----

func.func @unsupported_loop_result_user(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> tensor<32x64xf32, #asctile.local<UB>> {
    %1 = asctile.cv_strategy <split_by_m> -> tensor<32x64xf32, #asctile.local<UB>> {
      %2 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
      %3 = arith.mulf %2, %arg3 : tensor<32x64xf32, #asctile.local<UB>>
      asctile.yield %3 : tensor<32x64xf32, #asctile.local<UB>>
    }
    scf.yield %1 : tensor<32x64xf32, #asctile.local<UB>>
  }
  // expected-error@+1 {{is not supported by CV strategy propagation}}
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<32x64xf32, #asctile.local<UB>>, f32
  return
}
