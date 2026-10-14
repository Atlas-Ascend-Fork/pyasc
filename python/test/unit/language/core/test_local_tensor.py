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


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init() -> None:
        x1_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local_elem = x1_local(128)

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_operator(mock_launcher_run):

    @asc.jit
    def kernel_operator() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local_offset = x_local[128:]

    kernel_operator[1]()
    assert mock_launcher_run.call_count == 1


def test_get_length(mock_launcher_run):

    @asc.jit
    def kernel_get_length() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        length = x_local.get_length()

    kernel_get_length[1]()
    assert mock_launcher_run.call_count == 1


def test_get_phy_addr(mock_launcher_run):

    @asc.jit
    def kernel_get_phy_addr() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        phy_addr = x_local.get_phy_addr()
        phy_addr = x_local.get_phy_addr(256)

    kernel_get_phy_addr[1]()
    assert mock_launcher_run.call_count == 1


def test_get_position(mock_launcher_run):

    @asc.jit
    def kernel_get_position() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        pos = x_local.get_position()

    kernel_get_position[1]()
    assert mock_launcher_run.call_count == 1


def test_get_shape_info(mock_launcher_run):

    @asc.jit
    def kernel_get_shape_info() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        shape = x_local.get_shape_info()

    kernel_get_shape_info[1]()
    assert mock_launcher_run.call_count == 1


def test_get_size(mock_launcher_run):

    @asc.jit
    def kernel_get_size() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        size = x_local.get_size()

    kernel_get_size[1]()
    assert mock_launcher_run.call_count == 1


def test_get_size_for_static_shape(mock_launcher_run):

    @asc.jit
    def kernel_get_size_for_static_shape() -> None:
        x_local = asc.LocalTensorAuto(asc.float16, (2, 3, 4))
        size = x_local.get_size()
        assert size == 24

    kernel_get_size_for_static_shape[1]()
    assert mock_launcher_run.call_count == 1


def test_get_value(mock_launcher_run):

    @asc.jit
    def kernel_get_value() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        value = x_local.get_value(128)

    kernel_get_value[1]()
    assert mock_launcher_run.call_count == 1


def test_get_user_tag(mock_launcher_run):

    @asc.jit
    def kernel_get_user_tag() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        tag = x_local.get_user_tag()

    kernel_get_user_tag[1]()
    assert mock_launcher_run.call_count == 1


def test_reinterpret_cast(mock_launcher_run):

    @asc.jit
    def kernel_reinterpret_cast() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = x_local.reinterpret_cast(asc.float32)

    kernel_reinterpret_cast[1]()
    assert mock_launcher_run.call_count == 1


def test_set_addr_with_offset(mock_launcher_run):

    @asc.jit
    def kernel_set_addr_with_offset() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local.set_addr_with_offset(x_local, 128)

    kernel_set_addr_with_offset[1]()
    assert mock_launcher_run.call_count == 1


def test_set_buffer_len(mock_launcher_run):

    @asc.jit
    def kernel_set_buffer_len() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local.set_buffer_len(256)

    kernel_set_buffer_len[1]()
    assert mock_launcher_run.call_count == 1


def test_set_shape_info(mock_launcher_run):

    @asc.jit
    def kernel_set_shape_info() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        shape_info = asc.ShapeInfo()
        x_local.set_shape_info(shape_info)

    kernel_set_shape_info[1]()
    assert mock_launcher_run.call_count == 1


def test_set_size(mock_launcher_run):

    @asc.jit
    def kernel_set_size() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local.set_size(256)

    kernel_set_size[1]()
    assert mock_launcher_run.call_count == 1


def test_set_value(mock_launcher_run):

    @asc.jit
    def kernel_set_value() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local.set_value(128, 3.14)

    kernel_set_value[1]()
    assert mock_launcher_run.call_count == 1


def test_set_user_tag(mock_launcher_run):

    @asc.jit
    def kernel_set_user_tag() -> None:
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x_local.set_user_tag(10)

    kernel_set_user_tag[1]()
    assert mock_launcher_run.call_count == 1
