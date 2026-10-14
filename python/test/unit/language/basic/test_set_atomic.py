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


def test_set_atomic_add_kernel(mock_launcher_run):

    @asc.jit(always_compile=True)
    def set_atomic_add_kernel() -> None:
        asc.set_atomic_add(asc.half)

    set_atomic_add_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_set_atomic_max_kernel(mock_launcher_run):

    @asc.jit(always_compile=True)
    def set_atomic_max_kernel() -> None:
        asc.set_atomic_max(asc.half)

    set_atomic_max_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_set_atomic_min_kernel(mock_launcher_run):

    @asc.jit(always_compile=True)
    def set_atomic_min_kernel() -> None:
        asc.set_atomic_min(asc.half)

    set_atomic_min_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_set_atomic_none_kernel(mock_launcher_run):

    @asc.jit(always_compile=True)
    def set_atomic_none_kernel() -> None:
        asc.set_atomic_none()

    set_atomic_none_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_set_atomic_type_kernel(mock_launcher_run):

    @asc.jit(always_compile=True)
    def set_atomic_type_kernel() -> None:
        asc.set_atomic_type(asc.half)

    set_atomic_type_kernel[1]()
    assert mock_launcher_run.call_count == 1
