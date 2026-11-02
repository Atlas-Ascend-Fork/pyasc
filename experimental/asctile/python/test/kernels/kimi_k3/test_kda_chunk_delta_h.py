# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference implementation: https://github.com/sgl-project/sgl-kernel-npu/blob/87153b4f6e19d68fc11ab1bc705cc35f1d1eca3e/python/sgl_kernel_npu/sgl_kernel_npu/fla/kda_chunk_delta_h.py

import pytest
import torch

from ...target.kimi_k3.test_kda_chunk_delta_h import run_kda_chunk_delta_h_test


@pytest.mark.parametrize(
    "b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state, save_new_value, is_varlen, use_exp2, "
    "base_k, unroll_factor_k, reuse_alloc", [
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, True, False, False, False, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, True, True, False, False, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, True, False, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, True, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, True, 64, 1, 0),
        (1, 64, 1, 1, 128, 32, 64, 32, torch.float16, False, True, False, False, False, False, 64, 1, 0),
        (1, 128, 1, 1, 64, 32, 64, 32, torch.float16, True, True, False, False, False, False, 64, 1, 0),
        (1, 96, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, True, False, 64, 1, 0),
        (4, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, False, 64, 1, 0),
        (1, 64, 8, 1, 64, 32, 64, 32, torch.float16, True, False, False, False, False, False, 64, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, False, 32, 1, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, False, 16, 2, 0),
        (1, 64, 1, 1, 64, 32, 64, 32, torch.float16, False, True, False, False, False, False, 16, 4, 0),
    ])
def test_kda_chunk_delta_h(profiler, runs, b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state,
                           save_new_value, is_varlen, use_exp2, base_k, unroll_factor_k, reuse_alloc):
    run_kda_chunk_delta_h_test(profiler, runs, b, t, h, hg, k, v, bt, bv, dtype, use_g, use_gk, use_initial_state,
                               save_new_value, is_varlen, use_exp2, base_k, unroll_factor_k, reuse_alloc,
                               always_compile=True)
