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


@asctile.jit(always_compile=1)
def gather_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, index_ptr: asctile.GlobalAddress,
                  data_count: int, index_count: int, read_count: asctile.ConstExpr, offset: int,
                  check_bounds: asctile.ConstExpr, row_len: asctile.ConstExpr, pad_value: asctile.ConstExpr) -> None:
    input_tensor = asctile.global_tensor(input_ptr, [data_count, row_len])
    index_tensor = asctile.global_tensor(index_ptr, [index_count])
    index = asctile.copy_in(index_tensor, [0], [read_count])
    tile = asctile.gather(input_tensor, [offset], 0, index, pad_value=pad_value, check_bounds=check_bounds)
    output_tensor = asctile.global_tensor(output_ptr, [index_count, tile.shape[1]])
    asctile.copy_out(tile, output_tensor, [0, 0])


@asctile.jit(always_compile=1)
def gather_kernel_partial(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                          index_ptr: asctile.GlobalAddress, data_count: int, index_count: int,
                          read_count: asctile.ConstExpr, offset: int, check_bounds: asctile.ConstExpr,
                          row_len: asctile.ConstExpr, num_indices: int, pad_value: asctile.ConstExpr) -> None:
    input_tensor = asctile.global_tensor(input_ptr, [data_count, row_len])
    index_tensor = asctile.global_tensor(index_ptr, [index_count])
    index = asctile.copy_in(index_tensor, [0], [read_count])
    tile = asctile.gather(input_tensor, [offset], 0, index, pad_value=pad_value, num_indices=num_indices,
                          check_bounds=check_bounds)
    output_tensor = asctile.global_tensor(output_ptr, [index_count, tile.shape[1]])
    asctile.copy_out(tile, output_tensor, [0, 0])


@pytest.mark.parametrize("index_dtype", (torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("data_dtype",
                         (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "data_count, index_count, index_range, read_count, row_len, offset, check_bounds, num_indices, pad_value",
    [(32, 32, 32, 32, 32, 0, False, None, None),  # simple case
     (32, 32, 32, 32, 32, 0, False, 16, 17),  # runtime num_indices
     (32, 32, 32, 32, 31, 0, False, None, 17),  # padding (1 element)
     (32, 32, 64, 32, 32, 0, True, None, 17),  # out-of-bounds indexes
     (32, 32, 32, 32, 32, 16, True, None, 17),  # read with offset
     ])
def test_gather(data_count, index_count, index_range, read_count, row_len, offset, index_dtype, check_bounds,
                data_dtype, num_indices, pad_value, require_c310):
    require_c310()
    input = torch.arange(data_count * row_len).to(data_dtype).reshape([data_count, row_len])
    index = torch.randint(0, index_range, size=(index_count, ), dtype=index_dtype)
    items_align = 32 // input.element_size()
    result_row_len = asctile.ceildiv(row_len, items_align) * items_align
    result = torch.zeros([read_count, result_row_len], dtype=data_dtype)

    if num_indices is None:
        gather_kernel[1](input, result, index, data_count, index_count, read_count, offset, check_bounds, row_len,
                         pad_value)
    else:
        gather_kernel_partial[1](input, result, index, data_count, index_count, read_count, offset, check_bounds,
                                 row_len, num_indices, pad_value)

    if pad_value is None:
        pad_value = 0.0 if torch.is_floating_point(input) else 0

    golden = torch.zeros((read_count, result_row_len), dtype=data_dtype)
    num_indices = num_indices or read_count
    for row in range(0, num_indices):
        for col in range(0, row_len):
            if index[row] + offset < data_count:
                golden[row][col] = input[index[row] + offset][col]
            else:
                golden[row][col] = pad_value
        if pad_value is not None:
            for col in range(row_len, result_row_len):
                golden[row][col] = pad_value
    torch.testing.assert_close(golden[:num_indices], result[:num_indices])
