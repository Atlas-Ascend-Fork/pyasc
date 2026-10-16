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

from asc.runtime import launcher
from asc.runtime.launcher import LaunchOptions, Launcher
from asc.runtime.memory_handle import NumpyArrayHandle


@pytest.mark.parametrize("failure_point", [None, "launch", "synchronize"])
def test_launch_releases_h2d_allocations_for_success_and_runtime_errors(failure_point):
    allocated = []
    released = []
    copied_from_device = []
    synchronize_calls = []
    pointers = iter((0x1000, 0x2000))

    def allocate_on_device(_source, _nbytes):
        pointer = ctypes.c_void_p(next(pointers))
        allocated.append(pointer.value)
        return pointer

    def launch(*_args, **_kwargs):
        if failure_point == "launch":
            raise RuntimeError("launch failed")

    def synchronize(*_args, **_kwargs):
        synchronize_calls.append(True)
        if failure_point == "synchronize":
            raise RuntimeError("synchronize failed")

    def copy_from_device(_destination, source, _nbytes):
        copied_from_device.append(source.value)

    arrays = [np.arange(4, dtype=np.float32), np.arange(2, dtype=np.int32)]
    handles = [NumpyArrayHandle(array) for array in arrays]
    runtime = launcher.rt

    with patch.object(runtime, "copy_data_to_device", side_effect=allocate_on_device), \
            patch.object(runtime, "copy_data_from_device", side_effect=copy_from_device), \
            patch.object(runtime, "free", side_effect=lambda handle: released.append(handle.value)), \
            patch.object(runtime, "launch_kernel", side_effect=launch), \
            patch.object(runtime, "synchronize", side_effect=synchronize):
        runner = Launcher(LaunchOptions(core_num=1, stream=object()))
        if failure_point is None:
            runner.launch_kernel(object(), handles, enable_debug=False, func_name="test_kernel")
        else:
            with pytest.raises(RuntimeError, match=failure_point):
                runner.launch_kernel(object(), handles, enable_debug=False, func_name="test_kernel")

    assert allocated == [0x1000, 0x2000]
    assert released == allocated
    assert len(synchronize_calls) == (0 if failure_point == "launch" else 1)
    assert copied_from_device == (allocated if failure_point is None else [])
