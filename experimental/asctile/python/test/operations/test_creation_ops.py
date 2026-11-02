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
from .helpers import assert_close

supported_dtypes = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float8_e5m2,
    torch.float8_e4m3fn
]


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_full_1d(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]
    fill_value = 7 if dtype.is_signed or dtype.is_floating_point else 7
    if dtype.is_floating_point:
        fill_value = 3.5

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr, value: asctile.ConstExpr):
        out_gm = asctile.global_tensor(out_ptr, shape)
        tile = asctile.full(shape, value, dtype=out_gm.dtype)
        asctile.copy_out(tile, out_gm, [0])

    out = torch.zeros(shape, dtype=dtype)
    kernel[1](out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32, torch.float8_e5m2, torch.float8_e4m3fn])
def test_full_2d(require_c310, dtype: torch.dtype):
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        require_c310()
    last_dim = 32 // torch.tensor([], dtype=dtype).element_size()
    shape = [4, last_dim]
    fill_value = 2.5 if dtype.is_floating_point else 5

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr, value: asctile.ConstExpr):
        out_gm = asctile.global_tensor(out_ptr, shape)
        tile = asctile.full(shape, value, dtype=out_gm.dtype)
        asctile.copy_out(tile, out_gm, [0, 0])

    out = torch.zeros(shape, dtype=dtype)
    kernel[1](out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_full_like(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]
    fill_value = 3.5 if dtype.is_floating_point else 9

    @asctile.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asctile.ConstExpr, value: asctile.ConstExpr):
        in_gm = asctile.global_tensor(in_ptr, shape)
        out_gm = asctile.global_tensor(out_ptr, shape)
        src = asctile.copy_in(in_gm, [0], shape)
        tile = asctile.full_like(src, value)
        asctile.copy_out(tile, out_gm, [0])

    inp = torch.ones(shape, dtype=dtype)
    out = torch.zeros(shape, dtype=dtype)
    kernel[1](inp, out, shape, fill_value)
    expected = torch.full(shape, fill_value, dtype=dtype)
    assert_close(out, expected)


def test_full_like_2d():
    shape = [4, 8]

    @asctile.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asctile.ConstExpr):
        in_gm = asctile.global_tensor(in_ptr, shape)
        out_gm = asctile.global_tensor(out_ptr, shape)
        src = asctile.copy_in(in_gm, [0, 0], shape)
        tile = asctile.full_like(src, 7)
        asctile.copy_out(tile, out_gm, [0, 0])

    inp = torch.ones(shape, dtype=torch.int32)
    out = torch.zeros(shape, dtype=torch.int32)
    kernel[1](inp, out, shape)
    expected = torch.full(shape, 7, dtype=torch.int32)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_zeros_1d(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr):
        out_gm = asctile.global_tensor(out_ptr, shape)
        tile = asctile.zeros(shape, dtype=out_gm.dtype)
        asctile.copy_out(tile, out_gm, [0])

    out = torch.ones(shape, dtype=dtype)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32, torch.float8_e5m2, torch.float8_e4m3fn])
def test_zeros_2d(require_c310, dtype: torch.dtype):
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        require_c310()
    last_dim = 32 // torch.tensor([], dtype=dtype).element_size()
    shape = [4, last_dim]

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr):
        out_gm = asctile.global_tensor(out_ptr, shape)
        tile = asctile.zeros(shape, dtype=out_gm.dtype)
        asctile.copy_out(tile, out_gm, [0, 0])

    out = torch.ones(shape, dtype=dtype)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    assert_close(out, expected)


def test_zeros_default_dtype():
    shape = [16]

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr):
        out_gm = asctile.global_tensor(out_ptr, shape)
        tile = asctile.zeros(shape)
        asctile.copy_out(tile, out_gm, [0])

    out = torch.ones(shape, dtype=torch.int32)
    kernel[1](out, shape)
    expected = torch.zeros(shape, dtype=torch.int32)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_zeros_like(require_c310, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.float32):
        require_c310()

    shape = [32]

    @asctile.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asctile.ConstExpr):
        in_gm = asctile.global_tensor(in_ptr, shape)
        out_gm = asctile.global_tensor(out_ptr, shape)
        src = asctile.copy_in(in_gm, [0], shape)
        tile = asctile.zeros_like(src)
        asctile.copy_out(tile, out_gm, [0])

    inp = torch.ones(shape, dtype=dtype)
    out = torch.ones(shape, dtype=dtype)
    kernel[1](inp, out, shape)
    expected = torch.zeros(shape, dtype=dtype)
    assert_close(out, expected)


def test_zeros_like_2d():
    shape = [4, 8]

    @asctile.jit(always_compile=True)
    def kernel(in_ptr, out_ptr, shape: asctile.ConstExpr):
        in_gm = asctile.global_tensor(in_ptr, shape)
        out_gm = asctile.global_tensor(out_ptr, shape)
        src = asctile.copy_in(in_gm, [0, 0], shape)
        tile = asctile.zeros_like(src)
        asctile.copy_out(tile, out_gm, [0, 0])

    inp = torch.ones(shape, dtype=torch.float32)
    out = torch.ones(shape, dtype=torch.float32)
    kernel[1](inp, out, shape)
    expected = torch.zeros(shape, dtype=torch.float32)
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_concat_two_tiles_1d(require_c310, dtype: torch.dtype):
    require_c310()
    shape_a = [16]
    shape_b = [16]
    out_shape = [32]

    @asctile.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asctile.ConstExpr, shape_b: asctile.ConstExpr,
               out_shape: asctile.ConstExpr):
        a_gm = asctile.global_tensor(a_ptr, shape_a)
        b_gm = asctile.global_tensor(b_ptr, shape_b)
        out_gm = asctile.global_tensor(out_ptr, out_shape)
        tile_a = asctile.copy_in(a_gm, [0], shape_a)
        tile_b = asctile.copy_in(b_gm, [0], shape_b)
        result = asctile.concat(tile_a, tile_b)
        asctile.copy_out(result, out_gm, [0])

    a = torch.arange(16, dtype=dtype)
    b = torch.arange(16, 32, dtype=dtype)
    out = torch.zeros(out_shape, dtype=dtype)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b])
    assert_close(out, expected)


def test_concat_three_tiles_1d(require_c310):
    require_c310()
    shape_a = [8]
    shape_b = [8]
    shape_c = [8]
    out_shape = [24]

    @asctile.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, shape_a: asctile.ConstExpr, shape_b: asctile.ConstExpr,
               shape_c: asctile.ConstExpr, out_shape: asctile.ConstExpr):
        a_gm = asctile.global_tensor(a_ptr, shape_a)
        b_gm = asctile.global_tensor(b_ptr, shape_b)
        c_gm = asctile.global_tensor(c_ptr, shape_c)
        out_gm = asctile.global_tensor(out_ptr, out_shape)
        tile_a = asctile.copy_in(a_gm, [0], shape_a)
        tile_b = asctile.copy_in(b_gm, [0], shape_b)
        tile_c = asctile.copy_in(c_gm, [0], shape_c)
        result = asctile.concat(tile_a, tile_b, tile_c)
        asctile.copy_out(result, out_gm, [0])

    a = torch.arange(8, dtype=torch.float32)
    b = torch.arange(8, 16, dtype=torch.float32)
    c = torch.arange(16, 24, dtype=torch.float32)
    out = torch.zeros(out_shape, dtype=torch.float32)
    kernel[1](a, b, c, out, shape_a, shape_b, shape_c, out_shape)
    expected = torch.cat([a, b, c])
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_concat_two_tiles_2d(require_c310, dtype: torch.dtype):
    require_c310()
    last_dim = 16 if dtype == torch.float16 else 8
    shape_a = [2, last_dim]
    shape_b = [2, last_dim]
    out_shape = [4, last_dim]

    @asctile.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asctile.ConstExpr, shape_b: asctile.ConstExpr,
               out_shape: asctile.ConstExpr):
        a_gm = asctile.global_tensor(a_ptr, shape_a)
        b_gm = asctile.global_tensor(b_ptr, shape_b)
        out_gm = asctile.global_tensor(out_ptr, out_shape)
        tile_a = asctile.copy_in(a_gm, [0, 0], shape_a)
        tile_b = asctile.copy_in(b_gm, [0, 0], shape_b)
        result = asctile.concat(tile_a, tile_b)
        asctile.copy_out(result, out_gm, [0, 0])

    a = torch.arange(2 * last_dim, dtype=dtype).reshape(2, last_dim)
    b = torch.arange(2 * last_dim, 4 * last_dim, dtype=dtype).reshape(2, last_dim)
    out = torch.zeros(out_shape, dtype=dtype)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b], dim=0)
    assert_close(out, expected)


def test_concat_different_first_dim(require_c310):
    require_c310()
    shape_a = [4]
    shape_b = [12]
    out_shape = [16]

    @asctile.jit(always_compile=True)
    def kernel(a_ptr, b_ptr, out_ptr, shape_a: asctile.ConstExpr, shape_b: asctile.ConstExpr,
               out_shape: asctile.ConstExpr):
        a_gm = asctile.global_tensor(a_ptr, shape_a)
        b_gm = asctile.global_tensor(b_ptr, shape_b)
        out_gm = asctile.global_tensor(out_ptr, out_shape)
        tile_a = asctile.copy_in(a_gm, [0], shape_a)
        tile_b = asctile.copy_in(b_gm, [0], shape_b)
        result = asctile.concat(tile_a, tile_b)
        asctile.copy_out(result, out_gm, [0])

    a = torch.arange(4, dtype=torch.float32)
    b = torch.arange(4, 16, dtype=torch.float32)
    out = torch.zeros(out_shape, dtype=torch.float32)
    kernel[1](a, b, out, shape_a, shape_b, out_shape)
    expected = torch.cat([a, b])
    assert_close(out, expected)
