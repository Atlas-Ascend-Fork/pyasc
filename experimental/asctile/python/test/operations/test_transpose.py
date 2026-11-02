# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

from asc.experimental import asctile
import pytest
import torch
from .helpers import assert_close


@pytest.fixture(autouse=True)
def require_c310_auto(require_c310):
    require_c310()


@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32,
                                   torch.float8_e5m2, torch.float8_e4m3fn))
@pytest.mark.parametrize("permute, input_shape, real_shape, offsets, pad_value", [
    # 2d cases
    [[1, 0], [16, 32], [16, 32], [0, 0], 0],
    # 3d cases
    [[0, 2, 1], [2, 16, 32], [2, 16, 32], [0, 0, 0], 0],
    [[1, 2, 0], [8, 2, 32], [8, 2, 32], [0, 0, 0], 0],
    [[2, 0, 1], [2, 16, 32], [2, 16, 32], [0, 0, 0], 0],
    [[2, 1, 0], [8, 2, 32], [8, 2, 32], [0, 0, 0], 0],
    # 4d cases
    [[0, 1, 3, 2], [2, 3, 16, 16], [2, 3, 16, 16], [0, 0, 0, 0], 0],
    [[3, 2, 1, 0], [16, 2, 3, 16], [16, 2, 3, 16], [0, 0, 0, 0], 0],
    [[3, 1, 2, 0], [32, 2, 3, 16], [32, 2, 3, 16], [0, 0, 0, 0], 0],
    # test with empty permutation
    [[], [64, 32], [64, 32], [0, 0], 0],
    [[], [2, 32, 16], [2, 32, 16], [0, 0, 0], 0],
    [[], [3, 2, 32, 16], [3, 2, 32, 16], [0, 0, 0, 0], 0],
    # test when real_shape less than target buffer
    [[2, 0, 1], [3, 32, 64], [3, 30, 50], [0, 0, 0], 0],
    [[0, 2, 1], [2, 64, 32], [1, 50, 10], [0, 0, 0], 0],
    [[3, 2, 0, 1], [16, 2, 16, 32], [10, 2, 10, 30], [0, 0, 0, 0], 0],
    [[1, 0], [32, 64], [30, 60], [0, 0], 0],
    # test when offset too large and tile is cutted
    [[1, 0], [32, 32], [24, 24], [16, 0], 17],
    [[0, 2, 1], [2, 32, 32], [2, 24, 24], [0, 0, 16], 17],
    [[0, 1, 3, 2], [2, 2, 32, 32], [1, 1, 24, 24], [1, 0, 16, 0], 127],
    # padding outer dimension only
    [[0, 1, 3, 2], [4, 2, 32, 32], [4, 2, 32, 32], [3, 0, 0, 0], 17],
])
def test_transpose_onload(permute, input_shape, real_shape, offsets, pad_value, dtype):
    dim_order = permute
    if len(permute) == 0:
        dim_order = list(range(0, len(input_shape)))
        dim_order[-1], dim_order[-2] = dim_order[-2], dim_order[-1]

    output_shape = [input_shape[i] for i in dim_order]
    write_real_shape = [real_shape[i] for i in dim_order]

    verify_copy_count = [min(max(0, input_shape[i] - offsets[i]), real_shape[i]) for i in range(0, len(dim_order))]
    verify_pad_count = [input_shape[i] - verify_copy_count[i] for i in range(0, len(dim_order))]
    input = torch.rand(input_shape).mul(100).to(dtype=dtype)
    result = torch.zeros(output_shape, dtype=dtype)
    items_align = 32 // input.element_size()
    if output_shape[-1] % items_align != 0 or input_shape[-1] % items_align != 0:
        pytest.skip("data is not 32 byte aligned")

    golden = input
    for i in range(0, len(real_shape)):
        golden = golden.narrow(i, offsets[i], verify_copy_count[i])
        if verify_pad_count[i] > 0:
            shape = list(golden.shape)
            shape[i] = verify_pad_count[i]
            golden = torch.cat((golden, torch.full(shape, pad_value, dtype=golden.dtype)), dim=i)
    golden = golden.permute(dim_order)

    @asctile.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, input_shape: asctile.ConstExpr, real_shape: asctile.ConstExpr,
               output_shape: asctile.ConstExpr, write_real_shape: asctile.ConstExpr, offsets: asctile.ConstExpr,
               pad_value: asctile.ConstExpr, permute: asctile.ConstExpr):
        g_input = asctile.global_tensor(input_ptr, input_shape)
        tile = asctile.copy_in(g_input, offsets, input_shape, real_shape=real_shape, pad_value=pad_value)
        g_output = asctile.global_tensor(result_ptr, output_shape)
        asctile.copy_out(tile.transpose(*permute), g_output, [0] * len(output_shape))

    kernel[1](input, result, input_shape, real_shape, output_shape, write_real_shape, offsets, pad_value, permute)
    assert_close(result, golden)


@pytest.mark.parametrize("dtype", (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
))
@pytest.mark.parametrize("permute, input_shape, output_shape, offsets", [
    # 3d cases
    [[1, 0, 2], [2, 4, 128], [4, 2, 128], [0, 0, 0]],
    [[1, 0, 2], [2, 4, 128], [5, 3, 160], [1, 1, 4]],
    # 4d cases
    [[0, 2, 1, 3], [2, 3, 4, 128], [2, 4, 3, 128], [0, 0, 0, 0]],
    [[2, 0, 1, 3], [2, 3, 4, 128], [4, 4, 4, 128], [0, 2, 0, 0]],
    [[1, 2, 0, 3], [2, 3, 4, 128], [4, 4, 4, 128], [0, 0, 2, 0]],
])
def test_transpose_onstore(permute, input_shape, output_shape, offsets, dtype):
    dim_order = permute
    if len(permute) == 0:
        dim_order = list(range(0, len(input_shape)))
        dim_order[-1], dim_order[-2] = dim_order[-2], dim_order[-1]

    input = torch.rand(input_shape).mul(100).to(dtype=dtype)
    result = torch.zeros(output_shape, dtype=dtype)
    items_align = 32 // input.element_size()
    if output_shape[-1] % items_align != 0 or input_shape[-1] % items_align != 0:
        pytest.skip("data is not 32 byte aligned")

    golden = input.permute(dim_order)
    read_shape = [math.prod(input_shape[:-1]), input_shape[-1]]

    @asctile.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, input_shape: asctile.ConstExpr, read_shape: asctile.ConstExpr,
               output_shape: asctile.ConstExpr, offsets: asctile.ConstExpr, permute: asctile.ConstExpr):

        g_input = asctile.global_tensor(input_ptr, read_shape)
        tile = asctile.copy_in(g_input, [0] * len(read_shape), read_shape)
        tile = tile.reshape(input_shape)
        g_output = asctile.global_tensor(result_ptr, output_shape)
        asctile.copy_out(tile.transpose(*permute), g_output, offsets)

    kernel[1](input, result, input_shape, read_shape, output_shape, offsets, permute)

    for i in permute:
        result = result.narrow(i, offsets[i], input_shape[permute[i]])

    assert_close(result, golden)


@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32,
                                   torch.float8_e5m2, torch.float8_e4m3fn))
@pytest.mark.parametrize("input_shape", [
    [128, 64],
    [64, 128],
])
def test_transpose_in_ub(input_shape, dtype):
    input = torch.rand(input_shape).mul(100).to(dtype=dtype)
    element_size = input.element_size()
    element_align = 32 // element_size
    if input.shape[0] % element_align != 0 or input.shape[1] % element_align != 0:
        pytest.skip("Shape is not 32 byte aligned")
    result = torch.zeros(input_shape[::-1], dtype=dtype)
    copy = torch.zeros_like(input)

    @asctile.jit(always_compile=True)
    def kernel(input_ptr, result_ptr, copy_ptr, input_shape: asctile.ConstExpr):
        g_input = asctile.global_tensor(input_ptr, input_shape)
        # Regular load here
        tile = asctile.copy_in(g_input, offsets=[0] * len(input_shape), shape=input_shape)
        # Transpose in ub
        result = tile.transpose()
        g_output = asctile.global_tensor(result_ptr, [input_shape[1], input_shape[0]])
        g_copy = asctile.global_tensor(copy_ptr, input_shape)
        asctile.copy_out(result, g_output, offsets=[0] * len(input_shape))
        # 'copy' needs here to disable optimizing load+transpose in single op
        asctile.copy_out(tile, g_copy, offsets=[0] * len(input_shape))

    kernel[1](input, result, copy, input_shape)
    assert_close(input.T, result)
