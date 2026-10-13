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


@asctile.jit(always_compile=True)
def scatter_kernel(result_ptr: asctile.GlobalAddress, changes_ptr: asctile.GlobalAddress,
                   index_ptr: asctile.GlobalAddress, write_count: int, index_count: int, read_count: asctile.ConstExpr,
                   offset: int, check_bounds: asctile.ConstExpr, row_len: asctile.ConstExpr) -> None:
    result_tensor = asctile.global_tensor(result_ptr, [write_count, row_len])
    index_tensor = asctile.global_tensor(index_ptr, [index_count])
    changes_tensor = asctile.global_tensor(changes_ptr, [index_count, row_len])
    index = asctile.copy_in(index_tensor, [0], [read_count])
    data = asctile.copy_in(changes_tensor, [0, 0], [read_count, row_len])
    asctile.scatter(data, 0, index, result_tensor, [offset], check_bounds=check_bounds)


@asctile.jit(always_compile=True)
def scatter_kernel_partial(result_ptr: asctile.GlobalAddress, changes_ptr: asctile.GlobalAddress,
                           index_ptr: asctile.GlobalAddress, write_count: int, index_count: int,
                           read_count: asctile.ConstExpr, offset: int, check_bounds: asctile.ConstExpr,
                           row_len: asctile.ConstExpr, num_indices: int) -> None:
    result_tensor = asctile.global_tensor(result_ptr, [write_count, row_len])
    index_tensor = asctile.global_tensor(index_ptr, [index_count])
    changes_tensor = asctile.global_tensor(changes_ptr, [index_count, row_len])
    index = asctile.copy_in(index_tensor, [0], [read_count])
    data = asctile.copy_in(changes_tensor, [0, 0], [read_count, row_len])
    asctile.scatter(data, 0, index, result_tensor, [offset], num_indices=num_indices, check_bounds=check_bounds)


@pytest.mark.parametrize("index_dtype", (torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("data_dtype",
                         (torch.int8, torch.int16, torch.int32, torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("data_count, index_count, index_range, read_count, row_len, offset, check_bounds, num_indices",
                         [(32, 32, 32, 32, 32, 0, False, None),  # simple case
                          (32, 32, 32, 32, 32, 0, False, 16),  # runtime num_indices
                          (32, 32, 64, 32, 32, 0, True, None),  # out-of-bounds indexes
                          (32, 32, 32, 32, 32, 16, True, None),  # read with offset
                          ])
def test_scatter(data_count, index_count, index_range, read_count, row_len, offset, index_dtype, check_bounds,
                 data_dtype, num_indices, require_c310):
    require_c310()
    input = torch.arange(data_count * row_len).to(data_dtype).reshape([data_count, row_len])
    index = torch.randint(0, index_range, size=(index_count, ), dtype=index_dtype)
    data = torch.arange(index_count * row_len).to(data_dtype).reshape([index_count, row_len])

    golden = input.clone().detach()
    if num_indices is None:
        scatter_kernel[1](input, data, index, data_count, index_count, read_count, offset, check_bounds, row_len)
    else:
        scatter_kernel_partial[1](input, data, index, data_count, index_count, read_count, offset, check_bounds,
                                  row_len, num_indices)

    num_indices = num_indices or read_count
    for row in range(0, num_indices):
        for col in range(0, row_len):
            if index[row] + offset < data_count:
                golden[index[row] + offset][col] = data[row][col]
    torch.testing.assert_close(golden[:num_indices], input[:num_indices])
