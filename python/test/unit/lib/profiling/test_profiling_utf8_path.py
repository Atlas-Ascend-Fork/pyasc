# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Pass the UTF-8 byte length required by aclprofInit."""
import pytest

from asc.lib.profiling.msprof import MsprofInterface


@pytest.mark.parametrize("path", ["/tmp/profile", "/tmp/性能报告", "/tmp/café", "/tmp/🚀"])
def test_init_uses_encoded_path_length(monkeypatch, path):
    interface = object.__new__(MsprofInterface)
    calls = []
    monkeypatch.setattr(interface, "call", lambda *args: calls.append(args))
    interface.init(path)
    name, encoded, size = calls.pop()
    assert name == "aclprofInit"
    assert encoded == path.encode("utf-8")
    assert size.value == len(encoded)
