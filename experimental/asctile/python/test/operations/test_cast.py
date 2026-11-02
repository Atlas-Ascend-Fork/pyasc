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

USE_CORE_NUM = 1
SIZE = 32


@asctile.jit(always_compile=True)
def cast_kernel(x_ptr, z_ptr, size: asctile.ConstExpr, dst_dtype: asctile.ConstExpr) -> None:
    x_gm = asctile.global_tensor(x_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    tile = asctile.copy_in(x_gm, [0], [size])
    casted = tile.to(dst_dtype)
    asctile.copy_out(casted, z_gm, [0])


@asctile.jit(always_compile=True)
def cast_round_mode_kernel(x_ptr, z_ptr, size: asctile.ConstExpr, dst_dtype: asctile.ConstExpr,
                           round_mode: asctile.ConstExpr) -> None:
    x_gm = asctile.global_tensor(x_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    tile = asctile.copy_in(x_gm, [0], [size])
    casted = asctile.cast(tile, dst_dtype, round_mode=round_mode)
    asctile.copy_out(casted, z_gm, [0])


@asctile.jit(always_compile=True)
def cast_to_method_kernel(x_ptr, z_ptr, size: asctile.ConstExpr, dst_dtype: asctile.ConstExpr,
                          round_mode: asctile.ConstExpr) -> None:
    x_gm = asctile.global_tensor(x_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    tile = asctile.copy_in(x_gm, [0], [size])
    casted = tile.to(dst_dtype, round_mode=round_mode)
    asctile.copy_out(casted, z_gm, [0])


def round_half_away_from_zero(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def create_input(dtype: torch.dtype):
    if dtype.is_floating_point:
        return torch.randn(SIZE, dtype=dtype)
    if dtype.is_signed:
        return torch.randint(-100, 100, (SIZE, ), dtype=dtype)


def get_expected(x, src_dtype: torch.dtype, dst_dtype: torch.dtype, round_fn=None):
    if dst_dtype.is_floating_point:
        return x.to(dst_dtype)
    if src_dtype.is_floating_point:
        if round_fn:
            rounded = round_fn(x)
        else:
            rounded = torch.trunc(x)
        if dst_dtype == torch.int8:
            return torch.clamp(rounded, min=-128, max=127).to(torch.int8)
        if dst_dtype == torch.int16:
            return torch.clamp(rounded, min=-32768, max=32767).to(torch.int16)
        if dst_dtype == torch.int32:
            return rounded.to(torch.int32)
        if dst_dtype == torch.int64:
            return rounded.to(torch.int64)
    src_bits = src_dtype.itemsize * 8
    dst_bits = dst_dtype.itemsize * 8
    if src_bits < dst_bits:
        return x.to(dst_dtype)
    if src_bits > dst_bits:
        if dst_dtype == torch.int8:
            return torch.clamp(x.to(torch.int32), min=-128, max=127).to(torch.int8)
        if dst_dtype == torch.int16:
            return torch.clamp(x.to(torch.int32), min=-32768, max=32767).to(torch.int16)
        if dst_dtype == torch.int32:
            return torch.clamp(x.to(torch.int64), min=-2147483648, max=2147483647).to(torch.int32)
    return x.to(dst_dtype)


def get_round_mode_fn(round_mode):
    if round_mode == asctile.RoundMode.Floor:
        return torch.floor
    elif round_mode == asctile.RoundMode.Ceil:
        return torch.ceil
    elif round_mode == asctile.RoundMode.Trunc:
        return torch.trunc
    elif round_mode == asctile.RoundMode.Round:
        return round_half_away_from_zero
    elif round_mode == asctile.RoundMode.Rint:
        return torch.round
    return None


@pytest.mark.parametrize("dst_dtype, torch_src, torch_dst", [
    # float -> float
    (asctile.float16, torch.bfloat16, torch.float16),
    (asctile.float32, torch.bfloat16, torch.float32),
    (asctile.bfloat16, torch.float16, torch.bfloat16),
    (asctile.float32, torch.float16, torch.float32),
    (asctile.bfloat16, torch.float32, torch.bfloat16),
    (asctile.float16, torch.float32, torch.float16),
    # int -> float
    (asctile.float16, torch.int8, torch.float16),
    (asctile.float16, torch.int16, torch.float16),
    (asctile.float32, torch.int16, torch.float32),
    (asctile.float32, torch.int32, torch.float32),
    (asctile.float16, torch.int32, torch.float16),
    (asctile.float32, torch.int64, torch.float32),
    # float -> int
    (asctile.int32, torch.bfloat16, torch.int32),
    (asctile.int8, torch.float16, torch.int8),
    (asctile.int16, torch.float16, torch.int16),
    (asctile.int32, torch.float16, torch.int32),
    (asctile.int16, torch.float32, torch.int16),
    (asctile.int32, torch.float32, torch.int32),
    (asctile.int64, torch.float32, torch.int64),
    # int -> int
    (asctile.int16, torch.int8, torch.int16),
    (asctile.int32, torch.int8, torch.int32),
    (asctile.int32, torch.int16, torch.int32),
    (asctile.int16, torch.int32, torch.int16),
    (asctile.int64, torch.int32, torch.int64),
    (asctile.int32, torch.int64, torch.int32),
])
def test_cast(require_c310, dst_dtype, torch_src, torch_dst):
    if ((torch_src == torch.bfloat16 and torch_dst == torch.float16)
            or (torch_src == torch.float16 and torch_dst == torch.bfloat16)
            or (torch_src == torch.int8 and torch_dst == torch.int16)
            or (torch_src == torch.int8 and torch_dst == torch.int32)
            or (torch_src == torch.int16 and torch_dst == torch.int32)
            or (torch_src == torch.int32 and torch_dst == torch.float16)):
        require_c310()
    x = create_input(torch_src)
    z = torch.zeros(SIZE, dtype=torch_dst)
    cast_kernel[USE_CORE_NUM](x, z, SIZE, dst_dtype)
    expected = get_expected(x, torch_src, torch_dst)
    if torch_dst.is_floating_point:
        atol = 1e-3 if torch_dst == torch.float16 else 1e-2 if torch_dst == torch.bfloat16 else 1e-6
        torch.testing.assert_close(z, expected, atol=atol, rtol=atol)
    else:
        torch.testing.assert_close(z, expected)


@pytest.mark.parametrize("kernel", [
    cast_round_mode_kernel,
    cast_to_method_kernel,
])
@pytest.mark.parametrize("src_torch_dtype, dst_asctile_dtype, dst_torch_dtype, round_mode", [
    # float -> float
    (torch.float32, asctile.float32, torch.float32, asctile.RoundMode.Rint),
    (torch.float32, asctile.float32, torch.float32, asctile.RoundMode.Floor),
    (torch.float32, asctile.float32, torch.float32, asctile.RoundMode.Ceil),
    (torch.float32, asctile.float32, torch.float32, asctile.RoundMode.Round),
    (torch.float32, asctile.float32, torch.float32, asctile.RoundMode.Trunc),
    (torch.float32, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Rint),
    (torch.float32, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Floor),
    (torch.float32, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Ceil),
    (torch.float32, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Round),
    (torch.float32, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Trunc),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Rint),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Floor),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Ceil),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Round),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Trunc),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.Odd),
    (torch.float32, asctile.float16, torch.float16, asctile.RoundMode.NoRound),
    # float -> int
    (torch.float32, asctile.int16, torch.int16, asctile.RoundMode.Rint),
    (torch.float32, asctile.int16, torch.int16, asctile.RoundMode.Floor),
    (torch.float32, asctile.int16, torch.int16, asctile.RoundMode.Ceil),
    (torch.float32, asctile.int16, torch.int16, asctile.RoundMode.Round),
    (torch.float32, asctile.int16, torch.int16, asctile.RoundMode.Trunc),
    (torch.float32, asctile.int32, torch.int32, asctile.RoundMode.Rint),
    (torch.float32, asctile.int32, torch.int32, asctile.RoundMode.Floor),
    (torch.float32, asctile.int32, torch.int32, asctile.RoundMode.Ceil),
    (torch.float32, asctile.int32, torch.int32, asctile.RoundMode.Round),
    (torch.float32, asctile.int32, torch.int32, asctile.RoundMode.Trunc),
    (torch.float32, asctile.int64, torch.int64, asctile.RoundMode.Rint),
    (torch.float32, asctile.int64, torch.int64, asctile.RoundMode.Floor),
    (torch.float32, asctile.int64, torch.int64, asctile.RoundMode.Ceil),
    (torch.float32, asctile.int64, torch.int64, asctile.RoundMode.Round),
    (torch.float32, asctile.int64, torch.int64, asctile.RoundMode.Trunc),
    # half -> int/float
    (torch.float16, asctile.int16, torch.int16, asctile.RoundMode.Rint),
    (torch.float16, asctile.int16, torch.int16, asctile.RoundMode.Floor),
    (torch.float16, asctile.int16, torch.int16, asctile.RoundMode.Ceil),
    (torch.float16, asctile.int16, torch.int16, asctile.RoundMode.Round),
    (torch.float16, asctile.int16, torch.int16, asctile.RoundMode.Trunc),
    (torch.float16, asctile.int32, torch.int32, asctile.RoundMode.Rint),
    (torch.float16, asctile.int32, torch.int32, asctile.RoundMode.Floor),
    (torch.float16, asctile.int32, torch.int32, asctile.RoundMode.Ceil),
    (torch.float16, asctile.int32, torch.int32, asctile.RoundMode.Round),
    (torch.float16, asctile.int32, torch.int32, asctile.RoundMode.Trunc),
    (torch.float16, asctile.int8, torch.int8, asctile.RoundMode.Rint),
    (torch.float16, asctile.int8, torch.int8, asctile.RoundMode.Floor),
    (torch.float16, asctile.int8, torch.int8, asctile.RoundMode.Ceil),
    (torch.float16, asctile.int8, torch.int8, asctile.RoundMode.Round),
    (torch.float16, asctile.int8, torch.int8, asctile.RoundMode.Trunc),
    (torch.float16, asctile.float32, torch.float32, asctile.RoundMode.NoRound),
    (torch.float16, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Rint),
    (torch.float16, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Floor),
    (torch.float16, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Ceil),
    (torch.float16, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Round),
    (torch.float16, asctile.bfloat16, torch.bfloat16, asctile.RoundMode.Trunc),
    # bfloat16 -> int/float
    (torch.bfloat16, asctile.int32, torch.int32, asctile.RoundMode.Rint),
    (torch.bfloat16, asctile.int32, torch.int32, asctile.RoundMode.Floor),
    (torch.bfloat16, asctile.int32, torch.int32, asctile.RoundMode.Ceil),
    (torch.bfloat16, asctile.int32, torch.int32, asctile.RoundMode.Round),
    (torch.bfloat16, asctile.int32, torch.int32, asctile.RoundMode.Trunc),
    (torch.bfloat16, asctile.float16, torch.float16, asctile.RoundMode.Rint),
    (torch.bfloat16, asctile.float16, torch.float16, asctile.RoundMode.Floor),
    (torch.bfloat16, asctile.float16, torch.float16, asctile.RoundMode.Ceil),
    (torch.bfloat16, asctile.float16, torch.float16, asctile.RoundMode.Round),
    (torch.bfloat16, asctile.float16, torch.float16, asctile.RoundMode.Trunc),
    (torch.bfloat16, asctile.float32, torch.float32, asctile.RoundMode.NoRound),
    # int -> float
    (torch.int8, asctile.float16, torch.float16, asctile.RoundMode.NoRound),
    (torch.int16, asctile.float16, torch.float16, asctile.RoundMode.Rint),
    (torch.int16, asctile.float32, torch.float32, asctile.RoundMode.NoRound),
    (torch.int32, asctile.float32, torch.float32, asctile.RoundMode.Rint),
    (torch.int64, asctile.float32, torch.float32, asctile.RoundMode.Rint),
    # int -> int
    (torch.int8, asctile.int16, torch.int16, asctile.RoundMode.NoRound),
    (torch.int8, asctile.int32, torch.int32, asctile.RoundMode.NoRound),
    (torch.int16, asctile.int32, torch.int32, asctile.RoundMode.NoRound),
    (torch.int32, asctile.int16, torch.int16, asctile.RoundMode.NoRound),
    (torch.int32, asctile.int64, torch.int64, asctile.RoundMode.NoRound),
    (torch.int64, asctile.int32, torch.int32, asctile.RoundMode.NoRound),
])
def test_cast_with_round_mode(require_c310, src_torch_dtype, dst_asctile_dtype, dst_torch_dtype, round_mode, kernel):
    require_c310()
    x = create_input(src_torch_dtype)
    x_float = x.float() if src_torch_dtype == torch.bfloat16 else x
    z = torch.zeros(SIZE, dtype=dst_torch_dtype)
    kernel[USE_CORE_NUM](x, z, SIZE, dst_asctile_dtype, round_mode)
    round_fn = get_round_mode_fn(round_mode)
    expected = get_expected(x_float, src_torch_dtype, dst_torch_dtype, round_fn)
    if dst_torch_dtype.is_floating_point:
        atol = 1e-3 if dst_torch_dtype == torch.float16 else 1e-2 if dst_torch_dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(z, expected, atol=atol, rtol=atol)
    else:
        torch.testing.assert_close(z, expected)


@pytest.mark.parametrize("kernel", [
    cast_round_mode_kernel,
    cast_to_method_kernel,
])
@pytest.mark.parametrize("src_torch_dtype, dst_asctile_dtype, dst_torch_dtype, round_mode", [
    (torch.float32, asctile.float8_e5m2, torch.float8_e5m2, asctile.RoundMode.Rint),
    (torch.float32, asctile.float8_e4m3fn, torch.float8_e4m3fn, asctile.RoundMode.Rint),
    (torch.float8_e5m2, asctile.float32, torch.float32, asctile.RoundMode.NoRound),
    (torch.float8_e4m3fn, asctile.float32, torch.float32, asctile.RoundMode.NoRound),
])
def test_fp8_cast(require_c310, src_torch_dtype, dst_asctile_dtype, dst_torch_dtype, round_mode, kernel):
    require_c310()
    x = torch.randn([SIZE]).to(src_torch_dtype)
    z = torch.zeros(SIZE, dtype=dst_torch_dtype)
    kernel[USE_CORE_NUM](x, z, SIZE, dst_asctile_dtype, round_mode)
    round_fn = get_round_mode_fn(round_mode)
    expected = get_expected(x, src_torch_dtype, dst_torch_dtype, round_fn)

    atol = 1e-2
    torch.testing.assert_close(z.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=atol)
