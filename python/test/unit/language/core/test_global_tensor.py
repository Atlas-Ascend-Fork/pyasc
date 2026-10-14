# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init() -> None:
        x_gm = asc.GlobalTensor()

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_get_phy_addr(mock_launcher_run):

    @asc.jit
    def kernel_get_phy_addr(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        phy_addr = x_gm.get_phy_addr()
        phy_addr_offset = x_gm.get_phy_addr(128)

    data = MockTensor(asc.float32)
    kernel_get_phy_addr[1](data)
    assert mock_launcher_run.call_count == 1


def test_get_shape_info(mock_launcher_run):

    @asc.jit
    def kernel_get_shape_info(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        shape = x_gm.get_shape_info()

    data = MockTensor(asc.float32)
    kernel_get_shape_info[1](data)
    assert mock_launcher_run.call_count == 1


def test_get_size(mock_launcher_run):

    @asc.jit
    def kernel_get_size(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        x_size = x_gm.get_size()

    data = MockTensor(asc.float32)
    kernel_get_size[1](data)
    assert mock_launcher_run.call_count == 1


def test_get_value(mock_launcher_run):

    @asc.jit
    def kernel_get_value(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        x_value = x_gm.get_value(128)

    data = MockTensor(asc.float32)
    kernel_get_value[1](data)
    assert mock_launcher_run.call_count == 1


def test_operator(mock_launcher_run):

    @asc.jit
    def kernel_operator(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        x_gm_offset = x_gm[128:]
        x_gm_elem = x_gm(128)

    data = MockTensor(asc.float32)
    kernel_operator[1](data)
    assert mock_launcher_run.call_count == 1


def test_set_global_buffer(mock_launcher_run):

    @asc.jit
    def kernel_set_global_buffer(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        x_gm.set_global_buffer(x, 8192)

    data = MockTensor(asc.float32)
    kernel_set_global_buffer[1](data)
    assert mock_launcher_run.call_count == 1


def test_set_l2_cache_hint(mock_launcher_run):

    @asc.jit
    def kernel_set_l2_cache_hint(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        x_gm.set_l2_cache_hint(asc.CacheMode.CACHE_MODE_NORMAL, asc.CacheRwMode.RW)

    data = MockTensor(asc.float32)
    kernel_set_l2_cache_hint[1](data)
    assert mock_launcher_run.call_count == 1


def test_set_shape_info(mock_launcher_run):

    @asc.jit
    def kernel_set_shape_info(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        shape_info = asc.ShapeInfo()
        x_gm.set_shape_info(shape_info)

    data = MockTensor(asc.float32)
    kernel_set_shape_info[1](data)
    assert mock_launcher_run.call_count == 1


def test_set_value(mock_launcher_run):

    @asc.jit
    def kernel_set_value(x: asc.GlobalAddress) -> None:
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)
        y = 3.14
        x_gm.set_value(128, y)

    data = MockTensor(asc.float32)
    kernel_set_value[1](data)
    assert mock_launcher_run.call_count == 1
