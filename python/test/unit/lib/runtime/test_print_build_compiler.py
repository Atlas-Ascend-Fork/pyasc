# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import os
from pathlib import Path
import shlex
import shutil

import pytest

from asc.lib.runtime import print_utils


@pytest.fixture
def native_print_build(tmp_path, monkeypatch):
    sdk = os.environ.get("ASCEND_HOME_PATH")
    if not sdk or not (Path(sdk) / "lib64").is_dir():
        pytest.skip("Native print-library compilation requires a real CANN Toolkit")
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("Native print-library compilation requires a C++ compiler")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("CC", raising=False)
    wrappers = {}
    for label in ("cxx", "cc"):
        trace = tmp_path / (label + "-arguments")
        wrapper = tmp_path / (label + " compiler")
        wrapper.write_text("#!/bin/sh\n" + "printf '%s\\n' \"$@\" >> " + shlex.quote(str(trace)) + "\n" + "exec " +
                           shlex.quote(compiler) + " \"$@\"\n")
        wrapper.chmod(0o700)
        wrappers[label] = (wrapper, trace)
    source = Path(print_utils.__file__).with_suffix(".cpp")
    assert source.is_file()
    return source, wrappers


@pytest.mark.parametrize("cxx,cc,expected", [
    ("cxx", None, "cxx"),
    (None, "cc", "cc"),
    ("cxx", "cc", "cxx"),
    (None, None, None),
])
def test_print_build_respects_cxx_before_cc(native_print_build, tmp_path, monkeypatch, cxx, cc, expected):
    source, wrappers = native_print_build
    if cxx is not None:
        monkeypatch.setenv("CXX", str(wrappers[cxx][0]))
    if cc is not None:
        monkeypatch.setenv("CC", str(wrappers[cc][0]))
    result = print_utils.build_print_utils("print_interface", str(source), str(tmp_path))
    assert Path(result).is_file()
    library = ctypes.CDLL(result, ctypes.RTLD_LOCAL)
    assert getattr(library, "PrintWorkSpace") is not None
    called = [label for label, (_, trace) in wrappers.items() if trace.exists()]
    assert called == ([] if expected is None else [expected])


@pytest.mark.parametrize("variable", ["CXX", "CC"])
def test_print_build_reports_invalid_explicit_compiler(native_print_build, tmp_path, monkeypatch, variable):
    source, _ = native_print_build
    monkeypatch.setenv(variable, str(tmp_path / "missing compiler"))
    with pytest.raises(FileNotFoundError):
        print_utils.build_print_utils("print_interface", str(source), str(tmp_path))
