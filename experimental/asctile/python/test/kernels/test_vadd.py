# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from ..target.test_vadd import add as vadd_kernel


def vadd_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    core_num = 16
    vadd_kernel[core_num](x, y, out, out.size, tile_length=128, unroll_factor=2, l2_cache=(False, False, True),
                          always_compile=True)
    return out


def test_vadd(torch_seed: int):
    rng = np.random.default_rng(torch_seed)
    size = 8192
    x = rng.random(size, dtype=np.float32) * 10
    y = rng.random(size, dtype=np.float32) * 10
    out = vadd_launch(x, y)
    np.testing.assert_allclose(out, x + y)
