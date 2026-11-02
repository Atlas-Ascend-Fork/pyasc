# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch

from .test_softmax_fused import test_cases


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def softmax(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, input_shape: asctile.ConstExpr,
            output_shape: asctile.ConstExpr, total_rows: asctile.ConstExpr, rows_per_iter: asctile.ConstExpr,
            rows_per_core: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [input_shape[0], input_shape[1]])
    out_gm = asctile.global_tensor(out_ptr, [output_shape[0], output_shape[1]])

    start_row = asctile.block_idx() * rows_per_core
    tile_iters = asctile.ceildiv(rows_per_core, rows_per_iter)

    for i in range(tile_iters, unroll_factor=unroll_factor):
        offset_row = start_row + i * rows_per_iter
        row = asctile.copy_in(x_gm, [offset_row, 0], [rows_per_iter, input_shape[1]])
        row_minus_max = row - asctile.reduce_max(row, 1, keep_dims=True)
        numerator = asctile.exp(row_minus_max)
        denominator = asctile.reduce_sum(numerator, 1, keep_dims=True)
        out = numerator / denominator
        asctile.copy_out(out, out_gm, [offset_row, 0])


@pytest.mark.parametrize(
    "test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params",
    test_cases)
def test_softmax(profiler, backend, platform, device_id, runs, test_name, block_num, input_shapes, input_dtypes,
                 output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):
    unroll_factor = runtime_params[0]
    input_shape, input_dtype = input_shapes[0], input_dtypes[0]
    output_dtype = output_dtypes[0]

    total_rows, rows_per_iter, rows_per_core = None, None, None
    if tiling_key == 500:
        total_rows, _, _, tilePerCore, rows_per_iter, _, _, _ = tiling_params
        rows_per_core = tilePerCore * rows_per_iter
    if tiling_key == 1000:
        total_rows, _, _, rows_per_iter, rows_per_core, _ = tiling_params

    # Convert any shape to 2D
    if len(input_shape) == 1:
        input_shape_2d = [1, input_shape[0]]
    else:
        input_shape_2d = [torch.prod(torch.tensor(input_shape[:-1])).item(), input_shape[-1]]

    # Alignment
    num_rows, num_cols = input_shape_2d
    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    num_cols_padded = asctile.ceildiv(num_cols, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS
    padded_input_shape = [num_rows, num_cols_padded]
    padded_output_shape = padded_input_shape

    in_tensor = torch.full(padded_input_shape, dtype=input_dtype, fill_value=float('-inf'))
    in_tensor[:num_rows, :num_cols] = torch.randn(input_shape_2d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_output_shape, dtype=output_dtype)

    with profiler.profile():
        for _ in range(runs):
            softmax[block_num](in_tensor, out_tensor, padded_input_shape, padded_output_shape, total_rows,
                               rows_per_iter, rows_per_core, unroll_factor)

    expected = torch.softmax(in_tensor, dim=1)
    torch.testing.assert_close(out_tensor[:num_rows, :num_cols], expected[:num_rows, :num_cols], atol=1e-3, rtol=1e-3)
