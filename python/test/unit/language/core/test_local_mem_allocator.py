# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
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
        allocator = asc.LocalMemAllocator(hard=asc.Hardware.UB)

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_get_cur_addr(mock_launcher_run):

    @asc.jit
    def kernel_get_cur_addr() -> None:
        allocator = asc.LocalMemAllocator()
        addr = allocator.get_cur_addr()

    kernel_get_cur_addr[1]()
    assert mock_launcher_run.call_count == 1


def test_alloc_const(mock_launcher_run):

    @asc.jit
    def kernel_alloc_const() -> None:
        allocator = asc.LocalMemAllocator()
        tensor = allocator.alloc(asc.TPosition.VECIN, asc.float32, 32)

    kernel_alloc_const[1]()
    assert mock_launcher_run.call_count == 1


def test_alloc_dynamic(mock_launcher_run):

    @asc.jit
    def kernel_alloc_dynamic(size: int) -> None:
        allocator = asc.LocalMemAllocator()
        tensor = allocator.alloc(asc.TPosition.VECIN, asc.float32, size)

    kernel_alloc_dynamic[1](1024)
    assert mock_launcher_run.call_count == 1
