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

# dim, tensor_shape, tile_shape, offsets, is_static
tests = [
    # STATIC
    (1, [64], [16], [0], True),
    (1, [64], [16], [48], True),
    (2, [128, 128], [32, 32], [96, 64], True),
    (1, [53], [16], [22], True),
    (2, [257, 511], [19, 24], [40, 64], True),
    (2, [8, 7], [8, 8], [0, 0], True),
    (2, [8, 15], [8, 8], [0, 0], True),
    (2, [16, 7], [8, 8], [0, 0], True),
    (2, [16, 15], [8, 8], [0, 0], True),
    (2, [7, 8], [8, 8], [0, 0], True),
    (2, [7, 16], [8, 8], [0, 0], True),
    (2, [17, 8], [8, 8], [0, 0], True),
    (2, [17, 16], [8, 8], [0, 0], True),
    (2, [17, 7], [8, 8], [0, 0], True),
    (2, [17, 19], [8, 8], [0, 0], True),

    # DYNAMIC
    (1, [32], [8], [16], False),
    (1, [32], [8], [24], False),
    (2, [16, 2048], [4, 512], [8, 1024], False),
    (2, [16, 2048], [12, 1024], [4, 0], False),
    (2, [512, 512], [64, 64], [128, 256], False),
    (1, [77], [24], [48], False),
    (2, [150, 300], [21, 40], [10, 20], False),

    # Scalar load, store tests
    (1, [32], None, [0], True),
    (2, [32, 32], None, [0, 0], True),
    (1, [1024], None, [0], True),
    (2, [512, 512], None, [0, 0], True),
]


@asctile.jit(always_compile=True)
def kernel_static(x_ptr, y_ptr, z_ptr, tensor_shape: asctile.ConstExpr, tile_shape: asctile.ConstExpr,
                  offsets: asctile.ConstExpr) -> None:
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, tensor_shape), offsets, tile_shape)
    yt = asctile.copy_in(asctile.global_tensor(y_ptr, tensor_shape), offsets, tile_shape)
    zt = xt + yt
    asctile.copy_out(zt, asctile.global_tensor(z_ptr, tensor_shape), offsets)


@asctile.jit(always_compile=True)
def kernel_dynamic_1D(x_ptr, y_ptr, z_ptr, ts0, tile_shape: asctile.ConstExpr, offsets: asctile.ConstExpr) -> None:
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, [ts0]), offsets, tile_shape)
    yt = asctile.copy_in(asctile.global_tensor(y_ptr, [ts0]), offsets, tile_shape)
    zt = xt + yt
    asctile.copy_out(zt, asctile.global_tensor(z_ptr, [ts0]), offsets)


@asctile.jit(always_compile=True)
def kernel_dynamic_2D(x_ptr, y_ptr, z_ptr, ts0, ts1, tile_shape: asctile.ConstExpr, offsets: asctile.ConstExpr) -> None:
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, [ts0, ts1]), offsets, tile_shape)
    yt = asctile.copy_in(asctile.global_tensor(y_ptr, [ts0, ts1]), offsets, tile_shape)
    zt = xt + yt
    asctile.copy_out(zt, asctile.global_tensor(z_ptr, [ts0, ts1]), offsets)


@asctile.jit(always_compile=True)
def kernel_scalar_load_store(x_ptr, y_ptr, z_ptr, tensor_shape: asctile.ConstExpr, offsets: asctile.ConstExpr) -> None:
    xt = asctile.copy_in(asctile.global_tensor(x_ptr, tensor_shape), offsets)
    yt = asctile.copy_in(asctile.global_tensor(y_ptr, tensor_shape), offsets)
    zt = xt + yt
    asctile.copy_out(zt, asctile.global_tensor(z_ptr, tensor_shape), offsets)


@pytest.mark.parametrize("dim, tensor_shape, tile_shape, offsets, is_static", tests)
def test_load_store(require_c310, dim, tensor_shape, tile_shape, offsets, is_static):
    if dim == 2 and not is_static:
        require_c310()
    x, y = [torch.randn(tensor_shape) for _ in range(2)]
    z = torch.zeros(tensor_shape, dtype=torch.float32)
    if is_static:
        if tile_shape is None:
            kernel_scalar_load_store[1](x, y, z, tensor_shape, offsets)
        else:
            kernel_static[1](x, y, z, tensor_shape, tile_shape, offsets)
    else:
        if dim == 1:
            kernel_dynamic_1D[1](x, y, z, tensor_shape[0], tile_shape, offsets)
        else:
            kernel_dynamic_2D[1](x, y, z, tensor_shape[0], tensor_shape[1], tile_shape, offsets)
    if tile_shape is not None:
        slices = tuple(slice(off, off + size) for off, size in zip(offsets, tile_shape))
    else:
        slices = tuple(offsets)
    z_expected = torch.zeros_like(z)
    z_expected[slices] = x[slices] + y[slices]
    torch.testing.assert_close(z, z_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("tensor_shape, offsets", (
    ((16, ), (0, )),
    ((16, ), (7, )),
    ((16, 16), (0, 0)),
    ((16, 16), (7, 7)),
))
def test_store_1elem_tile(tensor_shape, offsets):
    x = torch.randn(tensor_shape, dtype=torch.float32)
    y = torch.zeros_like(x)

    @asctile.jit(always_compile=True)
    def kernel(x_ptr, y_ptr, tensor_shape: asctile.ConstExpr, offsets: asctile.ConstExpr):
        x = asctile.global_tensor(x_ptr, tensor_shape)
        s = asctile.copy_in(x, offsets)
        y = asctile.global_tensor(y_ptr, tensor_shape)
        asctile.copy_out(asctile.full([1], s), y, offsets)

    kernel[1](x, y, tensor_shape, offsets)
    y_ref = y.clone()
    y_ref[offsets] = x[offsets]
    torch.testing.assert_close(y, y_ref)


@asctile.jit(always_compile=True)
def kernel_load_padding(x_ptr, out_ptr, input_shape: asctile.ConstExpr, tile_shape: asctile.ConstExpr,
                        offsets: asctile.ConstExpr, pad_value: asctile.ConstExpr) -> None:
    x_gm = asctile.global_tensor(x_ptr, input_shape)
    out_gm = asctile.global_tensor(out_ptr, tile_shape)
    tile = asctile.copy_in(x_gm, offsets, tile_shape, pad_value=pad_value)
    asctile.copy_out(tile, out_gm, [0, 0])


@pytest.mark.parametrize(
    "input_shape, tile_shape, offsets",
    (
        ([16, 16], [8, 8], [0, 0]),
        ([16, 4], [8, 16], [0, 0]),
        ([12, 12], [24, 16], [0, 0]),
        ([9, 9], [8, 8], [5, 4]),
    ),
)
def test_load_padding(require_c310, input_shape, tile_shape, offsets):
    require_c310()
    pad_value = -1000.0
    x = torch.arange(1, input_shape[0] * input_shape[1] + 1, dtype=torch.float32).reshape(input_shape)
    out = torch.full(tile_shape, pad_value, dtype=torch.float32)
    kernel_load_padding[1](x, out, input_shape, tile_shape, offsets, pad_value)
    row_start, col_start = offsets
    tile_rows, tile_cols = tile_shape
    src_rows, src_cols = input_shape
    valid_rows = max(0, min(src_rows, row_start + tile_rows) - row_start)
    valid_cols = max(0, min(src_cols, col_start + tile_cols) - col_start)
    if valid_rows > 0 and valid_cols > 0:
        torch.testing.assert_close(out[0:valid_rows, 0:valid_cols], x[row_start:row_start + valid_rows,
                                                                      col_start:col_start + valid_cols])


@asctile.jit(always_compile=True)
def reindex_kernel(src_ptr: asctile.GlobalAddress, idx_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
                   src_shape: asctile.ConstExpr, idx_shape: asctile.ConstExpr) -> None:
    src_gm = asctile.global_tensor(src_ptr, src_shape)
    idx_gm = asctile.global_tensor(idx_ptr, idx_shape)
    out_gm = asctile.global_tensor(out_ptr, idx_shape)
    src = asctile.copy_in(src_gm, [0] * len(src_shape), src_shape)
    idx = asctile.copy_in(idx_gm, [0] * len(idx_shape), idx_shape)
    result = asctile.reindex(src, idx)
    asctile.copy_out(result, out_gm, [0] * len(idx_shape))


@pytest.mark.parametrize("src_shape", ([128], [16, 8], [4, 32]))
@pytest.mark.parametrize("idx_shape", ([16], [2, 32]))
@pytest.mark.parametrize("index_dtype", (torch.int8, torch.int16, torch.int32, torch.int64))
def test_reindex(require_c310, src_shape, idx_shape, index_dtype):
    require_c310()
    src_count = 1
    for s in src_shape:
        src_count *= s
    src = torch.randn(src_count, dtype=torch.float32).reshape(src_shape)
    upper = min(src_count, torch.iinfo(index_dtype).max + 1)
    idx = torch.randint(0, upper, idx_shape, dtype=torch.int32).to(index_dtype)
    out = torch.zeros(idx_shape, dtype=torch.float32)
    reindex_kernel[1](src, idx, out, src_shape, idx_shape)
    golden = src.reshape(-1)[idx.to(torch.int64)]
    torch.testing.assert_close(out, golden)
