// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-wrap-vec-scope -verify-diagnostics %s
func.func @reject_group_with_results(%vector: vector<4xf32>) -> vector<4xf32> {
  // expected-error@+1 {{'ascvf2.vf_group' op cannot wrap a group with remaining results}}
  %result = ascvf2.vf_group {
    ascvf2.yield %vector : vector<4xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}
