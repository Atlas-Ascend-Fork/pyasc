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


@asctile.jit(reuse_alloc=1, vf_fusion=True)
def sigmoid(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress, input_length: int,
            tile_size: asctile.ConstExpr[int], unroll_factor: asctile.ConstExpr, l2_enable: asctile.ConstExpr):
    x_gm = asctile.global_tensor(x_ptr, [input_length], l2_cache=l2_enable[0])
    out_gm = asctile.global_tensor(out_ptr, [input_length], l2_cache=l2_enable[1])

    tiles_per_core = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_size)
    core_length = tile_size * tiles_per_core
    core_offset = asctile.block_idx() * core_length

    for i in asctile.range(tiles_per_core, unroll_factor=unroll_factor):
        offset = core_offset + i * tile_size
        x = asctile.copy_in(x_gm, [offset], [tile_size])
        out = 1 / (asctile.exp(-x) + 1)
        asctile.copy_out(out, out_gm, [offset])


# yapf: disable
@parametrize_is_static()
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    pytest.param("sigmoid_test_1",  1,  ([16], ), (torch.float32, ), ([16], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (16, 512, 15872, 1, 1, 512, 16, 15872), id="sigmoid_test_1"),
    pytest.param("sigmoid_test_2",  1,  ([100, 1], ), (torch.float32, ), ([100, 1], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (100, 512, 15872, 1, 1, 512, 100, 15872), id="sigmoid_test_2"),
    pytest.param("sigmoid_test_3",  1,  ([1, 256], ), (torch.float32, ), ([1, 256], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (256, 512, 15872, 1, 1, 512, 256, 15872), id="sigmoid_test_3"),
    pytest.param("sigmoid_test_4",  1,  ([1, 10, 32], ), (torch.float32, ), ([1, 10, 32], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (320, 512, 15872, 1, 1, 512, 320, 15872), id="sigmoid_test_4"),
    pytest.param("sigmoid_test_5",  1,  ([128, 3], ), (torch.float16, ), ([128, 3], ), (torch.float16, ), ([False, True], ), (2, ), 100000000000100, (384, 512, 31744, 1, 1, 512, 384, 31744), id="sigmoid_test_5"),
    pytest.param("sigmoid_test_6",  2,  ([1, 1523], ), (torch.float32, ), ([1, 1523], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (1523, 1024, 15872, 1, 1, 1024, 499, 15872), id="sigmoid_test_6"),
    pytest.param("sigmoid_test_7",  2,  ([2048], ), (torch.float32, ), ([2048], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2048, 1024, 15872, 1, 1, 1024, 1024, 15872), id="sigmoid_test_7"),
    pytest.param("sigmoid_test_8",  3,  ([2500, 1], ), (torch.float32, ), ([2500, 1], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2500, 1024, 15872, 1, 1, 1024, 452, 15872), id="sigmoid_test_8"),
    pytest.param("sigmoid_test_9",  3,  ([1, 2953], ), (torch.float32, ), ([1, 2953], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2953, 1024, 15872, 1, 1, 1024, 905, 15872), id="sigmoid_test_9"),
    pytest.param("sigmoid_test_10",  8,  ([128, 64], ), (torch.float32, ), ([128, 64], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (8192, 1024, 15872, 1, 1, 1024, 1024, 15872), id="sigmoid_test_10"),
    pytest.param("sigmoid_test_11", 36, ([1500, 24], ), (torch.float32, ), ([1500, 24], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (36000, 1024, 15872, 1, 1, 1024, 160, 15872), id="sigmoid_test_11"),
    pytest.param("sigmoid_test_12", 63, ([7000, 8, 4], ), (torch.float32, ), ([7000, 8, 4], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (224000, 3584, 15872, 1, 1, 3584, 1792, 15872), id="sigmoid_test_12"),
    pytest.param("sigmoid_test_13", 72, ([128, 1, 2304], ), (torch.float32, ), ([128, 1, 2304], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (294912, 4096, 15872, 1, 1, 4096, 4096, 15872), id="sigmoid_test_13"),
    pytest.param("sigmoid_test_14", 72, ([128, 2, 5, 16, 16], ), (torch.float16, ), ([128, 2, 5, 16, 16], ), (torch.float16, ), ([False, True], ), (2, ), 100000000000100, (327680, 4608, 31744, 1, 1, 4608, 512, 31744), id="sigmoid_test_14"),
    pytest.param("sigmoid_test_15", 69, ([200, 120, 16], ), (torch.float32, ), ([200, 120, 16], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (384000, 5632, 15872, 1, 1, 5632, 1024, 15872), id="sigmoid_test_15"),
    pytest.param("sigmoid_test_16", 67, ([256, 20, 80], ), (torch.float32, ), ([256, 20, 80], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (409600, 6144, 15872, 1, 1, 6144, 4096, 15872), id="sigmoid_test_16"),
    pytest.param("sigmoid_test_17", 72, ([1024, 504], ), (torch.float32, ), ([1024, 504], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (516096, 7168, 15872, 1, 1, 7168, 7168, 15872), id="sigmoid_test_17"),
    pytest.param("sigmoid_test_18", 69, ([128, 64, 64], ), (torch.float16, ), ([128, 64, 64], ), (torch.float16, ), ([False, True], ), (2, ), 100000000000100, (524288, 7680, 31744, 1, 1, 7680, 2048, 31744), id="sigmoid_test_18"),
    pytest.param("sigmoid_test_19", 69, ([4096, 128], ), (torch.float32, ), ([4096, 128], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (524288, 7680, 15872, 1, 1, 7680, 2048, 15872), id="sigmoid_test_19"),
    pytest.param("sigmoid_test_20", 71, ([1500, 1, 384], ), (torch.float32, ), ([1500, 1, 384], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (576000, 8192, 15872, 1, 1, 8192, 2560, 15872), id="sigmoid_test_20"),
    pytest.param("sigmoid_test_21", 72, ([16, 24, 1536], ), (torch.float16, ), ([16, 24, 1536], ), (torch.float16, ), ([False, True], ), (2, ), 100000000000100, (589824, 8192, 31744, 1, 1, 8192, 8192, 31744), id="sigmoid_test_21"),
    pytest.param("sigmoid_test_22", 72, ([3000, 256], ), (torch.float32, ), ([3000, 256], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (768000, 10752, 15872, 1, 1, 10752, 4608, 15872), id="sigmoid_test_22"),
    pytest.param("sigmoid_test_23", 72, ([256, 100, 40], ), (torch.float32, ), ([256, 100, 40], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (1024000, 14336, 15872, 1, 1, 14336, 6144, 15872), id="sigmoid_test_23"),
    pytest.param("sigmoid_test_24", 72, ([1500, 899], ), (torch.float32, ), ([1500, 899], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (1348500, 18944, 15872, 2, 1, 3072, 3476, 15872), id="sigmoid_test_24"),
    pytest.param("sigmoid_test_25", 72, ([2500, 801], ), (torch.float32, ), ([2500, 801], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2002500, 28160, 15872, 2, 1, 12288, 3140, 15872), id="sigmoid_test_25"),
    pytest.param("sigmoid_test_26", 72, ([1024, 2048], ), (torch.float32, ), ([1024, 2048], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2097152, 29184, 15872, 2, 2, 13312, 9216, 15872), id="sigmoid_test_26"),
    pytest.param("sigmoid_test_27", 71, ([1500, 1501], ), (torch.float32, ), ([1500, 1501], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2251500, 31744, 15872, 2, 2, 15872, 13548, 15872), id="sigmoid_test_27"),
    pytest.param("sigmoid_test_28", 72, ([256, 200, 50], ), (torch.float32, ), ([256, 200, 50], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (2560000, 35840, 15872, 3, 1, 4096, 15360, 15872), id="sigmoid_test_28"),
    pytest.param("sigmoid_test_29", 72, ([1500, 2985], ), (torch.float32, ), ([1500, 2985], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (4477500, 62464, 15872, 4, 3, 14848, 10812, 15872), id="sigmoid_test_29"),
    pytest.param("sigmoid_test_30", 72, ([3000, 1501], ), (torch.float32, ), ([3000, 1501], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (4503000, 62976, 15872, 4, 2, 15360, 15832, 15872), id="sigmoid_test_30"),
    pytest.param("sigmoid_test_31", 72, ([1500, 4070], ), (torch.float32, ), ([1500, 4070], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (6105000, 84992, 15872, 6, 5, 5632, 7080, 15872), id="sigmoid_test_31"),
    pytest.param("sigmoid_test_32", 72, ([256, 8, 8, 512], ), (torch.float32, ), ([256, 8, 8, 512], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (8388608, 116736, 15872, 8, 7, 5632, 5120, 15872), id="sigmoid_test_32"),
    pytest.param("sigmoid_test_33", 72, ([1500, 6781], ), (torch.float32, ), ([1500, 6781], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (10171500, 141312, 15872, 9, 9, 14336, 11372, 15872), id="sigmoid_test_33"),
    pytest.param("sigmoid_test_34", 72, ([3000, 4070], ), (torch.float32, ), ([3000, 4070], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (12210000, 169984, 15872, 11, 9, 11264, 14160, 15872), id="sigmoid_test_34"),
    pytest.param("sigmoid_test_35", 72, ([1500, 8829], ), (torch.float32, ), ([1500, 8829], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (13243500, 184320, 15872, 12, 10, 9728, 13932, 15872), id="sigmoid_test_35"),
    pytest.param("sigmoid_test_36", 72, ([512, 200, 200], ), (torch.float32, ), ([512, 200, 200], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (20480000, 284672, 15872, 18, 17, 14848, 14336, 15872), id="sigmoid_test_36"),
    pytest.param("sigmoid_test_37", 72, ([1500, 61, 256], ), (torch.float32, ), ([1500, 61, 256], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (23424000, 325632, 15872, 21, 20, 8192, 2560, 15872), id="sigmoid_test_37"),
    pytest.param("sigmoid_test_38", 72, ([128, 248, 1024], ), (torch.float16, ), ([128, 248, 1024], ), (torch.float16, ), ([False, True], ), (2, ), 100000000000100, (32505856, 451584, 31744, 15, 14, 7168, 30720, 31744), id="sigmoid_test_38"),
    pytest.param("sigmoid_test_39", 72, ([256, 32, 32, 128], ), (torch.float32, ), ([256, 32, 32, 128], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (33554432, 466432, 15872, 30, 28, 6144, 9216, 15872), id="sigmoid_test_39"),
    pytest.param("sigmoid_test_40", 72, ([1500, 61, 496], ), (torch.float32, ), ([1500, 61, 496], ), (torch.float32, ), ([False, True], ), (2, ), 300000000000100, (45384000, 630784, 15872, 40, 38, 11776, 11072, 15872), id="sigmoid_test_40"),
# PYASC_TESTS_END
])
# yapf: enable
def test_sigmoid(profiler, runs, is_static, test_name, block_num, input_shapes, input_dtypes, output_shapes,
                 output_dtypes, compile_params, runtime_params, tiling_key, tiling_params):

    input_shape = input_shapes[0]
    dtype = input_dtypes[0]
    _, block_former_size, ub_former_size, former_loop_count, *_ = tiling_params

    if former_loop_count > 1:
        tile_size = ub_former_size
    else:
        tile_size = block_former_size

    input_shape_1d = [math.prod(input_shape)]
    in_tensor = torch.randn(input_shape_1d, dtype=dtype)
    out_tensor = torch.zeros(input_shape_1d, dtype=dtype)
    unroll_factor = runtime_params[0]
    l2_enable = compile_params[-1]

    params = [in_tensor, out_tensor]
    if is_static:
        params.append(asctile.ConstExpr(input_shape_1d[0]))
    else:
        params.append(input_shape_1d[0])
    params += (tile_size, unroll_factor, l2_enable)

    expected = torch.sigmoid(in_tensor)
    with profiler.profile():
        for _ in range(runs):
            sigmoid[block_num](*params)

    torch.testing.assert_close(out_tensor, expected, atol=1e-3, rtol=1e-3)
