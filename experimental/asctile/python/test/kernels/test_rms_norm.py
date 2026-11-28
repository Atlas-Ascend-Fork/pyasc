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


@asctile.jit(always_compile=True, reuse_alloc=2)
def rms_norm_kernel(input_ptr: asctile.GlobalAddress, gamma_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
                    eps: asctile.ConstExpr, size: int, tile_size: asctile.ConstExpr, total_blocks: int,
                    num_blocks: asctile.ConstExpr, norm_shape: asctile.ConstExpr):
    input_gm = asctile.global_tensor(input_ptr, [size])
    gamma_gm = asctile.global_tensor(gamma_ptr, [norm_shape])
    out_gm = asctile.global_tensor(out_ptr, [size])
    gamma = asctile.copy_in(gamma_gm, [0], [norm_shape])
    loop_num = asctile.ceildiv(asctile.ceildiv(total_blocks, asctile.block_num()), num_blocks)
    block_offset = asctile.block_idx() * tile_size * loop_num
    for i in asctile.range(loop_num, unroll_factor=2):
        offset = block_offset + i * tile_size
        input_tensor = asctile.copy_in(input_gm, [offset], [tile_size])
        tensor = input_tensor.reshape([num_blocks, norm_shape])
        out = asctile.rms_norm(tensor, gamma, eps)
        out = out.reshape(tile_size)
        asctile.copy_out(out, out_gm, [offset])


def rms_norm_launch(x: torch.Tensor, gamma, eps=1e-6):
    output = torch.empty_like(x)
    input_size = torch.numel(x)
    norm_shape = torch.numel(gamma)
    total_blocks = input_size // norm_shape
    num_blocks = 1
    tile_size = num_blocks * norm_shape
    rms_norm_kernel[16](x, gamma, output, eps, input_size, tile_size, total_blocks, num_blocks, norm_shape)
    return output


def rms_norm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * gamma


@pytest.mark.parametrize("input_type", [torch.float16, torch.float32])
def test_rms_norm(input_type):
    x = torch.rand((128, 128), dtype=input_type)
    gamma = torch.rand(128, dtype=input_type)
    out = rms_norm_launch(x, gamma)
    torch.testing.assert_close(out, rms_norm_torch(x, gamma), rtol=1e-3, atol=1e-3)
