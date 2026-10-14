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
        tmp_buf = asc.TBuf(asc.TPosition.VECCALC)

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_get(mock_launcher_run):

    @asc.jit
    def kernel_get() -> None:
        tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
        tmp_buf.get(dtype=asc.int32)
        tmp_buf.get(dtype=asc.int32, len=256)

    kernel_get[1]()
    assert mock_launcher_run.call_count == 1


def test_get_with_offset(mock_launcher_run):

    @asc.jit
    def kernel_get_with_offset() -> None:
        tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
        tmp_buf.get_with_offset(size=32, buf_offset=64, dtype=asc.int32)

    kernel_get_with_offset[1]()
    assert mock_launcher_run.call_count == 1
