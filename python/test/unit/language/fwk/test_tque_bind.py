# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.language.fwk.tpipe import TQueBind
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_init(mock_launcher_run):

    @asc.jit
    def kernel_init() -> None:
        que = TQueBind(src=asc.TPosition.VECIN)

    kernel_init[1]()
    assert mock_launcher_run.call_count == 1


def test_alloc_tensor(mock_launcher_run):

    @asc.jit
    def kernel_alloc_tensor() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        x_local = que.alloc_tensor(asc.float16)

        que_in_place = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=0)
        y_local = asc.LocalTensor(asc.float32)
        que_in_place.alloc_tensor(y_local)

    kernel_alloc_tensor[1]()
    assert mock_launcher_run.call_count == 1


def test_deque(mock_launcher_run):

    @asc.jit
    def kernel_deque() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECOUT, depth=1)
        que.deque(asc.float16)
        que.deque(asc.float16, asc.TPosition.GM, asc.TPosition.VECIN)
        que_in_place = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=0)
        y_local = asc.LocalTensor(asc.float32)
        que_in_place.deque(y_local)

    kernel_deque[1]()
    assert mock_launcher_run.call_count == 1


def test_enque(mock_launcher_run):

    @asc.jit
    def kernel_enque() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        x_local = que.alloc_tensor(asc.float16)
        que.enque(x_local)
        que.enque(x_local, asc.TPosition.VECIN, asc.TPosition.GM)

    kernel_enque[1]()
    assert mock_launcher_run.call_count == 1


def test_free_tensor(mock_launcher_run):

    @asc.jit
    def kernel_free_tensor() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        x_local = que.alloc_tensor(asc.float16)
        que.free_tensor(x_local)

    kernel_free_tensor[1]()
    assert mock_launcher_run.call_count == 1


def test_get_tensor_count_in_que(mock_launcher_run):

    @asc.jit
    def kernel_get_tensor_count_in_que() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        que.get_tensor_count_in_que()

    kernel_get_tensor_count_in_que[1]()
    assert mock_launcher_run.call_count == 1


def test_has_idle_buffer(mock_launcher_run):

    @asc.jit
    def kernel_has_idle_buffer() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        que.has_idle_buffer()

    kernel_has_idle_buffer[1]()
    assert mock_launcher_run.call_count == 1


def test_has_tensor_in_que(mock_launcher_run):

    @asc.jit
    def kernel_has_tensor_in_que() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        que.has_tensor_in_que()

    kernel_has_tensor_in_que[1]()
    assert mock_launcher_run.call_count == 1


def test_vacant_in_que(mock_launcher_run):

    @asc.jit
    def kernel_vacant_in_que() -> None:
        que = TQueBind(src=asc.TPosition.VECIN, dst=asc.TPosition.VECIN, depth=1)
        ret = que.vacant_in_que()

    kernel_vacant_in_que[1]()
    assert mock_launcher_run.call_count == 1
