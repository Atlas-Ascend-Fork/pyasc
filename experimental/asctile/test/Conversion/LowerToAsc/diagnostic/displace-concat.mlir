// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asclower-displace-concat -split-input-file -verify-diagnostics %s

func.func @concat_ub_invalid_dim(%arg0: tensor<8x16xf16, #asctile.local<UB>>, %arg1: tensor<8x8xf16, #asctile.local<UB>>) -> tensor<8x24xf16, #asctile.local<UB>> {
  // expected-error@below {{only supports dimension 0 for UB tensor, got 1}}
  // expected-error@below {{failed to legalize operation 'tensor.concat'}}
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<8x16xf16, #asctile.local<UB>>, tensor<8x8xf16, #asctile.local<UB>>) -> tensor<8x24xf16, #asctile.local<UB>>
  return %0 : tensor<8x24xf16, #asctile.local<UB>>
}

// -----

func.func @concat_l0c_invalid_dim(%arg0: tensor<8x16xf16, #asctile.local<L0C>>, %arg1: tensor<10x16xf16, #asctile.local<L0C>>) -> tensor<18x16xf16, #asctile.local<L0C>> {
  // expected-error@below {{only supports dimension 1 for L0C tensor, got 0}}
  // expected-error@below {{failed to legalize operation 'tensor.concat'}}
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<8x16xf16, #asctile.local<L0C>>, tensor<10x16xf16, #asctile.local<L0C>>) -> tensor<18x16xf16, #asctile.local<L0C>>
  return %0 : tensor<18x16xf16, #asctile.local<L0C>>
}

// -----

func.func @concat_invalid_loc(%arg0: tensor<8x16xf16, #asctile.local<L1>>, %arg1: tensor<8x8xf16, #asctile.local<L1>>) -> tensor<8x24xf16, #asctile.local<L1>> {
  // expected-error@below {{only supports tensors located in UB or L0C, got L1}}
  // expected-error@below {{failed to legalize operation 'tensor.concat'}}
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<8x16xf16, #asctile.local<L1>>, tensor<8x8xf16, #asctile.local<L1>>) -> tensor<8x24xf16, #asctile.local<L1>>
  return %0 : tensor<8x24xf16, #asctile.local<L1>>
}
