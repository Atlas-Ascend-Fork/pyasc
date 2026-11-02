# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import itertools

from asc.experimental import asctile
import pytest
import torch
from .helpers import assert_close


@pytest.mark.parametrize("asc_op, torch_op, args, shape", (
    (asctile.broadcast_to, torch.broadcast_to, [4, 32], [1, 32]),
    (asctile.broadcast_to, torch.broadcast_to, [50, 32], [1, 32]),
    (asctile.reshape, torch.reshape, [64], [2, 32]),
    (asctile.reshape, torch.reshape, [4, 32], [128]),
    (asctile.ravel, torch.ravel, [], [2, 32]),
    (asctile.expand_dims, torch.unsqueeze, [0], [32]),
    (asctile.squeeze, torch.squeeze, [0], [1, 32]),
))
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
                                   torch.float32, torch.float64, torch.float8_e5m2, torch.float8_e4m3fn))
def test_shape_op(require_c310, asc_op, torch_op, args, shape, dtype: torch.dtype):
    if asc_op is asctile.broadcast_to:
        require_c310()
        if dtype == torch.float64:
            pytest.skip("broadcast_to float64 support is limited")
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        require_c310()

    def create_input(tensor_shape):
        if dtype.is_floating_point:
            return torch.randn(tensor_shape).clamp(1, 100).to(dtype)
        elif dtype.is_signed:
            return torch.randint(1, 100, tensor_shape, dtype=dtype)

    x = create_input(shape)
    if asc_op in (asctile.expand_dims, asctile.squeeze):
        ref_z = torch_op(x, dim=args[0])
    elif not args:
        ref_z = torch_op(x)
    else:
        ref_z = torch_op(x, args)
    z = create_input(ref_z.shape)
    in_offsets = (0, ) * len(x.shape)
    out_offsets = (0, ) * len(ref_z.shape)

    @asctile.jit(always_compile=True)
    def kernel(x_ptr, z_ptr, input_shape: asctile.ConstExpr, output_shape: asctile.ConstExpr,
               in_offsets: asctile.ConstExpr, out_offsets: asctile.ConstExpr, op: asctile.ConstExpr,
               op_param: asctile.ConstExpr) -> None:
        xt = asctile.copy_in(asctile.global_tensor(x_ptr, input_shape), in_offsets, input_shape)
        zt = op(xt, *op_param)
        asctile.copy_out(zt, asctile.global_tensor(z_ptr, output_shape), out_offsets)

    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, args)
    assert_close(z, ref_z)


@pytest.mark.parametrize("iter_factory", (list, tuple, itertools.chain))
@pytest.mark.parametrize("asc_op, torch_op, dst_shape, input_shape", (
    (asctile.reshape, torch.reshape, [64], [2, 32]),
    (asctile.reshape, torch.reshape, [4, 32], [128]),
    (asctile.broadcast_to, torch.broadcast_to, [4, 32], [1, 32]),
))
def test_shape_op_with_list_or_tuple(require_c310, asc_op, torch_op, dst_shape, input_shape, iter_factory):
    if asc_op is asctile.broadcast_to:
        require_c310()

    x = torch.randn(input_shape, dtype=torch.float32).clamp(1, 100)
    ref_z = torch_op(x, dst_shape)
    z = torch.zeros(ref_z.shape, dtype=torch.float32)
    in_offsets = (0, ) * len(x.shape)
    out_offsets = (0, ) * len(ref_z.shape)
    wrapped_args = iter_factory(dst_shape)

    @asctile.jit(always_compile=True)
    def kernel(x_ptr, z_ptr, input_shape: asctile.ConstExpr, output_shape: asctile.ConstExpr,
               in_offsets: asctile.ConstExpr, out_offsets: asctile.ConstExpr, op: asctile.ConstExpr,
               op_param: asctile.ConstExpr) -> None:
        xt = asctile.copy_in(asctile.global_tensor(x_ptr, input_shape), in_offsets, input_shape)
        zt = op(xt, op_param)
        asctile.copy_out(zt, asctile.global_tensor(z_ptr, output_shape), out_offsets)

    kernel[1](x, z, x.shape, ref_z.shape, in_offsets, out_offsets, asc_op, wrapped_args)
    torch.testing.assert_close(z, ref_z)


@pytest.mark.parametrize("shape", ([32], [3, 32]))
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
                                   torch.float32, torch.float8_e5m2, torch.float8_e4m3fn))
def test_broadcast_dup(require_c310, shape, dtype):
    if dtype in (torch.int8, torch.int64, torch.float8_e5m2, torch.float8_e4m3fn):
        require_c310()
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        require_c310()

    @asctile.jit(always_compile=True)
    def kernel(out_ptr, shape: asctile.ConstExpr, offsets: asctile.ConstExpr):
        out_tensor = asctile.global_tensor(out_ptr, shape)
        out = asctile.full([1], 77, out_tensor.dtype).broadcast_to(*out_tensor.shape)
        asctile.copy_out(out, out_tensor, offsets)

    out = torch.zeros(shape, dtype=dtype)
    out_ref = torch.full_like(out, 77)
    size = tuple(out.size())
    kernel[1](out, size, [0] * len(size))
    assert_close(out, out_ref)
