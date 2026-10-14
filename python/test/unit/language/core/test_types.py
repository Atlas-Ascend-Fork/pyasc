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


def test_binary_repeat_params(mock_launcher_run):

    @asc.jit
    def kernel_binary_repeat_params() -> None:
        asc.BinaryRepeatParams()
        asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)

    kernel_binary_repeat_params[1]()
    assert mock_launcher_run.call_count == 1


def test_brcb_repeat_params(mock_launcher_run):

    @asc.jit
    def kernel_brcb_repeat_params() -> None:
        asc.BrcbRepeatParams()
        asc.BrcbRepeatParams(1, 8)

    kernel_brcb_repeat_params[1]()
    assert mock_launcher_run.call_count == 1


def test_data_copy_params(mock_launcher_run):

    @asc.jit
    def kernel_data_copy_params() -> None:
        asc.DataCopyParams()
        asc.DataCopyParams(1, 0, 0, 0)

    kernel_data_copy_params[1]()
    assert mock_launcher_run.call_count == 1


def test_data_copy_enhanced_params(mock_launcher_run):

    @asc.jit
    def kernel_data_copy_enhanced_params() -> None:
        asc.DataCopyEnhancedParams()
        asc.DataCopyEnhancedParams(asc.BlockMode.BLOCK_MODE_NORMAL, asc.DeqScale.DEQ_NONE, 0, 0, False,
                                   asc.pad_t.PAD_NONE, 0)

    kernel_data_copy_enhanced_params[1]()
    assert mock_launcher_run.call_count == 1


def test_shape_info(mock_launcher_run):

    @asc.jit
    def kernel_shape_info() -> None:
        s0 = asc.ShapeInfo()
        s1 = asc.ShapeInfo(asc.array(asc.int32, [8, 8]))
        s2 = asc.ShapeInfo(asc.array(asc.int32, [8, 8]), data_format=asc.DataFormat.ND)
        x = asc.LocalTensor(asc.int32)
        x.set_shape_info(s0)
        x.set_shape_info(s1)
        x.set_shape_info(s2)
        size = asc.get_shape_size(s0)

    kernel_shape_info[1]()
    assert mock_launcher_run.call_count == 1


def test_slice_info(mock_launcher_run):

    @asc.jit
    def kernel_slice_info() -> None:
        asc.SliceInfo(1, 1, 1, 1, 1)

    kernel_slice_info[1]()
    assert mock_launcher_run.call_count == 1


def test_unary_repeat_params(mock_launcher_run):

    @asc.jit
    def kernel_unary_repeat_params() -> None:
        asc.UnaryRepeatParams()
        asc.UnaryRepeatParams(1, 1, 8, 8)

    kernel_unary_repeat_params[1]()
    assert mock_launcher_run.call_count == 1


def test_trans_data_to_5hd_params(mock_launcher_run):

    @asc.jit
    def kernel_trans_data_to_5hd_params() -> None:
        asc.TransDataTo5HDParams()
        asc.TransDataTo5HDParams(False, False, 1, 0, 0)

    kernel_trans_data_to_5hd_params[1]()
    assert mock_launcher_run.call_count == 1


def test_transpose_params_ext(mock_launcher_run):

    @asc.jit
    def kernel_transpose_params_ext() -> None:
        asc.TransposeParamsExt()
        asc.TransposeParamsExt(0, 0, 0, 0, asc.TransposeType.TRANSPOSE_ND2ND_B16)

    kernel_transpose_params_ext[1]()
    assert mock_launcher_run.call_count == 1


def test_gather_mask_params(mock_launcher_run):

    @asc.jit
    def kernel_gather_mask_params() -> None:
        asc.GatherMaskParams()
        asc.GatherMaskParams(1, 1, 0, 0)

    kernel_gather_mask_params[1]()
    assert mock_launcher_run.call_count == 1


def test_nd_2_nz_params(mock_launcher_run):

    @asc.jit
    def kernel_nd_2_nz_params() -> None:
        asc.Nd2NzParams(1, 1, 1, 1, 1, 1, 1, 1)

    kernel_nd_2_nz_params[1]()
    assert mock_launcher_run.call_count == 1


def test_nd_2_nz_params_full(mock_launcher_run):

    @asc.jit
    def kernel_nd_2_nz_params_full() -> None:
        asc.Nz2NdParamsFull(1, 1, 1, 1, 1, 1, 1, 1)

    kernel_nd_2_nz_params_full[1]()
    assert mock_launcher_run.call_count == 1


def test_data_copy_co12_dst_params(mock_launcher_run):

    @asc.jit
    def kernel_data_copy_co12_dst_params() -> None:
        asc.DataCopyCO12DstParams(1, 1, 1, 1, asc.QuantModes.NoQuant, 0, False, False)

    kernel_data_copy_co12_dst_params[1]()
    assert mock_launcher_run.call_count == 1


def test_gather_repeat_params(mock_launcher_run):

    @asc.jit
    def kernel_gather_repeat_params() -> None:
        asc.GatherRepeatParams()
        asc.GatherRepeatParams(1, 8)

    kernel_gather_repeat_params[1]()
    assert mock_launcher_run.call_count == 1
