# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_softmax(mock_launcher_run):

    @asc.jit
    def kernel_softmax() -> None:
        dst = asc.LocalTensor(dtype=asc.float16)
        sum_local = asc.LocalTensor(dtype=asc.float16)
        max_local = asc.LocalTensor(dtype=asc.float16)
        src = asc.LocalTensor(dtype=asc.float16)
        tiling = asc.adv.SoftmaxTiling(src_m=1, src_k=1, src_size=64)
        asc.adv.softmax(dst, sum_local, max_local, src, tiling)

    kernel_softmax[1]()
    assert mock_launcher_run.call_count == 1


def test_swiglu(mock_launcher_run):

    @asc.jit
    def kernel_swiglu() -> None:
        dst = asc.LocalTensor(dtype=asc.float32)
        src0 = asc.LocalTensor(dtype=asc.float32)
        src1 = asc.LocalTensor(dtype=asc.float32)
        tmp = asc.LocalTensor(dtype=asc.uint8)

        asc.adv.swiglu(dst, src0, src1)
        asc.adv.swiglu(dst, src0, src1, cal_count=128)
        asc.adv.swiglu(dst, src0, src1, shared_tmp_buffer=tmp)
        asc.adv.swiglu(dst, src0, src1, shared_tmp_buffer=tmp, cal_count=128)

    kernel_swiglu[1]()
    assert mock_launcher_run.call_count == 1


def test_softmax_flash_v2(mock_launcher_run):

    @asc.jit
    def kernel_softmax_flash_v2() -> None:
        dst_half = asc.LocalTensor(dtype=asc.float16)
        state_half = asc.LocalTensor(dtype=asc.float16)
        src_half = asc.LocalTensor(dtype=asc.float16)
        exp_max_half = asc.LocalTensor(dtype=asc.float16)
        state_float = asc.LocalTensor(dtype=asc.float32)
        reduce_max = asc.LocalTensor(dtype=asc.float16)
        shared_tmp = asc.LocalTensor(dtype=asc.uint8)
        tiling = asc.adv.SoftmaxTiling(src_m=8, src_k=512, src_size=4096)
        shape = asc.adv.SoftMaxShapeInfo(8, 512, 8, 512)
        reduce_config = asc.adv.SoftmaxConfig(False, 8, 512, asc.SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC)

        asc.adv.softmax_flash_v2(dst_half, state_half, state_half, src_half, exp_max_half, state_half, state_half,
                                 tiling, shape)
        asc.adv.softmax_flash_v2(dst_half, state_half, state_half, src_half, exp_max_half, state_half, state_half,
                                 tiling, shape, out_reduce_max=reduce_max, is_update=True, config=reduce_config)

        asc.adv.softmax_flash_v2(dst_half, state_float, state_float, src_half, exp_max_half, state_float, state_float,
                                 tiling, shape)

        asc.adv.softmax_flash_v2(dst_half, state_half, state_half, src_half, exp_max_half, state_half, state_half,
                                 tiling, shape, shared_tmp_buffer=shared_tmp)

        asc.adv.softmax_flash_v2(dst_half, state_half, state_half, src_half, exp_max_half, state_half, state_half,
                                 tiling, shape, shared_tmp_buffer=shared_tmp, out_reduce_max=reduce_max, is_update=True,
                                 config=reduce_config)

        full_tile_config = asc.adv.SoftmaxConfig(False, 8, 512)
        asc.adv.softmax_flash_v2(dst_half, state_float, state_float, src_half, exp_max_half, state_float, state_float,
                                 tiling, shape, shared_tmp_buffer=shared_tmp, is_basic_block=True,
                                 config=full_tile_config)

    kernel_softmax_flash_v2[1]()
    assert mock_launcher_run.call_count == 1
