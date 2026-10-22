# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate the public compiler constructor and options with real module imports."""
from pathlib import Path

import pytest

from asc.runtime import compiler as compiler_module
from asc.runtime.compiler import CompileOptions, Compiler
from asc.runtime.config import Platform


@pytest.fixture(autouse=True)
def compiler_environment(monkeypatch, tmp_path):
    # Target selection is the hardware boundary; the real compiler module,
    # options, constructor and command generation remain unchanged.
    monkeypatch.setattr(compiler_module, "get_soc_version", lambda: Platform.Ascend910B1)
    monkeypatch.setenv("PYASC_DUMP_PATH", str(tmp_path / "compiler-dump"))


@pytest.mark.parametrize("level", [False, True, 0.0, 1.0, 2.0, 3.0, -1, 4, "1", None])
def test_invalid_level_is_rejected_by_public_constructor(level):
    with pytest.raises(RuntimeError, match="Please check input compile option"):
        Compiler(CompileOptions(opt_level=level))


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_integer_level_constructs_compiler_and_expected_command(level):
    compiler = Compiler(CompileOptions(opt_level=level))
    argv = compiler._get_compiler_cmd("dav-c220-vec", Path("input.cce"), Path("output.o"), [])
    assert "-O" + str(level) in argv
