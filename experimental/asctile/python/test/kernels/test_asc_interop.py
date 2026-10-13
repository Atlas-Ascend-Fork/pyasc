# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.experimental import asctile
import torch


@asc.jit
def muladd(x: asctile.LocalTensor, y: asctile.LocalTensor, z: asctile.LocalTensor) -> asctile.LocalTensor:
    dst = asctile.to_asc(asctile.empty_like(x, asctile.TensorLocation.UB))
    asc.mul(dst, asctile.to_asc(x), asctile.to_asc(y), x.size)
    asc.add(dst, dst, asctile.to_asc(z), x.size)
    return asctile.from_asc(dst, location=asctile.TensorLocation.UB)


@asctile.jit(always_compile=True, vf_fusion=True)
def vmuladd_kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, z_ptr: asctile.GlobalAddress,
                   out_ptr: asctile.GlobalAddress, size: int, tile_size: asctile.ConstExpr[int],
                   tile_per_block: asctile.ConstExpr[int], buffer_factor: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    z_gm = asctile.global_tensor(z_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    base_offset = asctile.block_idx() * tile_size * tile_per_block
    for i in range(tile_per_block, unroll_factor=buffer_factor):
        tile_offset = base_offset + i * tile_size
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size], location=asctile.TensorLocation.UB)
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size], location=asctile.TensorLocation.UB)
        z = asctile.copy_in(z_gm, [tile_offset], [tile_size], location=asctile.TensorLocation.UB)
        out = muladd(x, y, z)
        asctile.copy_out(out, out_gm, [tile_offset])


def test_vmuladd_asc_interop():
    size = 8192
    x = torch.rand(size, dtype=torch.float32) * 10
    y = torch.rand(size, dtype=torch.float32) * 10
    z = torch.rand(size, dtype=torch.float32) * 10
    out = torch.empty_like(x)
    core_num = 16
    tile_size = 32
    num_tiles = asctile.ceildiv(size, tile_size)
    vmuladd_kernel[core_num](x, y, z, out, size, tile_size, asctile.ceildiv(num_tiles, core_num), buffer_factor=2)
    torch.testing.assert_close(out, x * y + z)
