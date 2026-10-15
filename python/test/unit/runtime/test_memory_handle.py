# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
from unittest.mock import patch

import numpy as np
import pytest

from asc.lib import runtime as rt
from asc.runtime.memory_handle import NumpyArrayHandle, TorchCpuTensorHandle


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided-view"])
def test_numpy_array_copy_from_device_preserves_logical_layout(strided):
    sentinel = -1
    backing = np.full(8 if strided else 4, sentinel, dtype=np.int32)
    array = backing[::2] if strided else backing
    expected = np.array([10, 20, 30, 40], dtype=np.int32)

    def copy_result_to_host(dst, _handle, size):
        assert size == expected.nbytes
        ctypes.memmove(dst, expected.ctypes.data, size)

    handle = NumpyArrayHandle(array)
    handle.handle = ctypes.c_void_p(1)
    with patch.object(rt, "copy_data_from_device", side_effect=copy_result_to_host):
        handle.copy_from_device()

    np.testing.assert_array_equal(array, expected)
    if strided:
        np.testing.assert_array_equal(backing[1::2], np.full(4, sentinel, dtype=np.int32))


def test_torch_cpu_tensor_copy_from_device_preserves_logical_layout():
    torch = pytest.importorskip("torch")
    sentinel = -1
    backing = torch.full((8, ), sentinel, dtype=torch.int32)
    tensor = backing[::2]
    expected = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    def copy_result_to_host(dst, _handle, size):
        assert size == expected.numel() * expected.element_size()
        ctypes.memmove(dst, expected.data_ptr(), size)

    handle = TorchCpuTensorHandle(tensor)
    handle.handle = ctypes.c_void_p(1)
    with patch.object(rt, "copy_data_from_device", side_effect=copy_result_to_host):
        handle.copy_from_device()

    assert torch.equal(tensor, expected)
    assert torch.equal(backing[1::2], torch.full((4, ), sentinel, dtype=torch.int32))
