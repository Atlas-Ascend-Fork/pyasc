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
        buf_pool = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_init_buf_pool(mock_launcher_run):

    @asc.jit
    def kernel_init_buf_pool() -> None:
        buf_pool0 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool1 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool2 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool0.init_buf_pool(buf_pool=buf_pool1, len=256)
        buf_pool0.init_buf_pool(buf_pool=buf_pool1, len=256, share_buf=buf_pool2)

    kernel_init_buf_pool[1]()
    assert mock_launcher_run.call_count == 1


def test_init_buffer(mock_launcher_run):

    @asc.jit
    def kernel_init_buffer() -> None:
        que = asc.TQue(asc.TPosition.VECIN, 1)
        tmp_buf = asc.TBuf(asc.TPosition.VECCALC)
        buf_pool0 = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool0.init_buffer(que=que, num=1, len=256)
        buf_pool0.init_buffer(buf=tmp_buf, len=256)

    kernel_init_buffer[1]()
    assert mock_launcher_run.call_count == 1


def test_reset(mock_launcher_run):

    @asc.jit
    def kernel_reset() -> None:
        buf_pool = asc.TBufPool(pos=asc.TPosition.VECIN, buf_id_size=4)
        buf_pool.reset()

    kernel_reset[1]()
    assert mock_launcher_run.call_count == 1
