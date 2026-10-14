# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import subprocess
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch


class FileCheck:

    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    def __call__(self, *args, **kwargs):
        from asc.language.core.utils import global_builder
        check_template = inspect.getsource(self.kernel_fn.fn)
        self.kernel_fn[1](*args, **kwargs, always_compile=True)
        mlir = str(global_builder.get_ir_module())

        with tempfile.TemporaryDirectory(prefix="pyasc_cli") as tmpdir:
            tmp_file = Path(tmpdir) / "expected"
            tmp_file.write_text(check_template)
            subprocess.run(["FileCheck", tmp_file], input=mlir, check=True, text=True)


@pytest.fixture
def mock_launcher_run():
    with patch("asc.runtime.launcher.Launcher.run", return_value=None) as mock:
        yield mock


def pytest_addoption(parser):
    parser.addoption("--skip-filecheck", action="store_true", help="Skip filecheck tests")


@pytest.fixture
def filecheck(pytestconfig):
    if pytestconfig.getoption("--skip-filecheck"):
        pytest.skip("Skipping filecheck tests")

    yield FileCheck
