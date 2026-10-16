# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The native device ID buffer must match its advertised element count."""
import ctypes

import pytest

from asc.lib.profiling.msprof import AicoreMetrics, MsprofInterface, ProfileType


@pytest.mark.parametrize("ids,generator", [([0], False), ([0, 2], False), ([1, 2, 3], False), ([0, 2], True)])
def test_create_config_allocates_all_device_ids(monkeypatch, ids, generator):
    interface = object.__new__(MsprofInterface)
    calls = []

    def record_call(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return 123

    monkeypatch.setattr(interface, "call", record_call)
    config = interface.create_config(iter(ids) if generator else ids, AicoreMetrics.NONE, [ProfileType.TASK_TIME])
    assert config.value == 123
    name, args, kwargs = calls.pop()
    assert name == "aclprofCreateConfig"
    assert list(args[0]) == ids
    assert len(args[0]) == args[1].value == len(ids)
    assert ctypes.sizeof(args[0]) == len(ids) * ctypes.sizeof(ctypes.c_uint32)
    assert args[4].value == ProfileType.TASK_TIME.value
    assert kwargs["ret_type"] == ctypes.c_void_p
