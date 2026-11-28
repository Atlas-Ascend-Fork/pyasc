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

from .helpers import parametrize_is_static


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def reciprocal(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_length,
               tile_length: asctile.ConstExpr, unroll_factor: asctile.ConstExpr, l2_enable: asctile.ConstExpr):
    x = asctile.global_tensor(input_ptr, [input_length], l2_cache=l2_enable[0])
    z = asctile.global_tensor(output_ptr, [input_length], l2_cache=l2_enable[1])

    block_loop_num = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asctile.block_idx() * block_length

    for i in asctile.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        xt = asctile.copy_in(x, [current_offset], [tile_length])
        zt = asctile.div(1.0, xt)
        asctile.copy_out(zt, z, [current_offset])


@parametrize_is_static()
@pytest.mark.parametrize(
    "block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params",
    [
        # PYASC_TESTS_BEGIN
        pytest.param(1, ([256, 1], ), (torch.float32, ), ([256, 1], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (256, 1, 15872, 512, 1, 1, 1, 512, 256, 15872, 1), id="reciprocal_00001"),
        pytest.param(1, ([1024], ), (torch.float32, ), ([1024], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (1024, 1, 15872, 1024, 1, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00002"),
        pytest.param(2, ([1200], ), (torch.float32, ), ([1200], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (1200, 2, 15872, 1024, 2, 1, 1, 1024, 176, 15872, 1), id="reciprocal_00003"),
        pytest.param(2, ([1500], ), (torch.float32, ), ([1500], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (1500, 2, 15872, 1024, 2, 1, 1, 1024, 476, 15872, 1), id="reciprocal_00004"),
        pytest.param(2, ([2048], ), (torch.float32, ), ([2048], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (2048, 2, 15872, 1024, 2, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00005"),
        pytest.param(2, ([2048, 1], ), (torch.float32, ), ([2048, 1], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (2048, 2, 15872, 1024, 2, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00006"),
        pytest.param(3, ([2400], ), (torch.float32, ), ([2400], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (2400, 3, 15872, 1024, 3, 1, 1, 1024, 352, 15872, 1), id="reciprocal_00007"),
        pytest.param(3, ([2500], ), (torch.float32, ), ([2500], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (2500, 3, 15872, 1024, 3, 1, 1, 1024, 452, 15872, 1), id="reciprocal_00008"),
        pytest.param(4, ([16, 256], ), (torch.float32, ), ([16, 256], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (4096, 4, 15872, 1024, 4, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00009"),
        pytest.param(5, ([16, 5, 1, 64], ), (torch.float32, ), ([16, 5, 1, 64], ), (torch.float32, ), ([False, True], ),
                     (2, ), 7, (5120, 5, 15872, 1024, 5, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00010"),
        pytest.param(5, ([16, 320], ), (torch.float32, ), ([16, 320], ), (torch.float32, ), ([False, True], ), (2, ), 7,
                     (5120, 5, 15872, 1024, 5, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00011"),
        pytest.param(14, ([100, 14, 10], ), (torch.float32, ), ([100, 14, 10], ), (torch.float32, ), ([False, True], ),
                     (2, ), 7, (14000, 14, 15872, 1024, 14, 1, 1, 1024, 688, 15872, 1), id="reciprocal_00012"),
        pytest.param(20, ([1024, 1, 20], ), (torch.float32, ), ([1024, 1, 20], ), (torch.float32, ), ([False, True], ),
                     (2, ), 7, (20480, 20, 15872, 1024, 20, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00013"),
        pytest.param(50, ([1024, 1, 50], ), (torch.float32, ), ([1024, 1, 50], ), (torch.float32, ), ([False, True], ),
                     (2, ), 7, (51200, 50, 15872, 1024, 50, 1, 1, 1024, 1024, 15872, 1), id="reciprocal_00014"),
        pytest.param(72, ([16, 24, 768], ), (torch.float32, ), ([16, 24, 768], ), (torch.float32, ), ([False, True], ),
                     (2, ), 7, (294912, 72, 15872, 4096, 72, 1, 1, 4096, 4096, 15872, 1), id="reciprocal_00015"),
        pytest.param(72, ([128, 1, 2304], ), (torch.float32, ), ([128, 1, 2304], ), (torch.float32, ),
                     ([False, True], ), (2, ), 7,
                     (294912, 72, 15872, 4096, 72, 1, 1, 4096, 4096, 15872, 1), id="reciprocal_00016"),
        pytest.param(72, ([1024, 1, 1000], ), (torch.float32, ), ([1024, 1, 1000], ), (torch.float32, ),
                     ([False, True], ), (2, ), 7,
                     (1024000, 72, 15872, 14336, 72, 1, 1, 14336, 6144, 15872, 1), id="reciprocal_00017"),
        # PYASC_TESTS_END
    ])
def test_reciprocal(profiler, runs, is_static, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes,
                    compile_params, runtime_params, tiling_key, tiling_params):

    input_shape = input_shapes[0]
    input_dtype = input_dtypes[0]
    _, _, ub_former_size, block_former_size, _, ub_former_loop_count, *_ = tiling_params
    input_shape_1d = math.prod(input_shape)
    unroll_factor = runtime_params[0]
    l2_enable = compile_params[-1]

    if ub_former_loop_count > 1:
        tile_size = ub_former_size
    else:
        tile_size = block_former_size

    # For low-precision dtypes bound |x| away from zero: a near-zero x makes 1/x
    # blow past the fp16 range (inf), which diverges from the CPU golden. fp32 has
    # the range to keep randn well-defined.
    if input_dtype == torch.float32:
        in_tensor = torch.randn(input_shape_1d, dtype=input_dtype)
    else:
        in_tensor = torch.randn(input_shape_1d, dtype=input_dtype).abs() + 1.0
    out_tensor = torch.zeros(input_shape_1d, dtype=input_dtype)

    params = [in_tensor, out_tensor]
    if is_static:
        params.append(asctile.ConstExpr(input_shape_1d))
    else:
        params.append(input_shape_1d)
    params.extend([tile_size, unroll_factor, l2_enable])

    with profiler.profile():
        for _ in range(runs):
            reciprocal[block_num](*params)

    # Low-precision dtypes carry more rounding error; reciprocal (1/x) amplifies
    # it further, so widen the tolerance for fp16.
    tol = {torch.float32: 1e-3, torch.float16: 4e-3}[input_dtype]
    expected = torch.reciprocal(in_tensor)
    torch.testing.assert_close(out_tensor, expected, atol=tol, rtol=tol)
